"""
Celery tasks for image generation.

Runs in background workers, separate from FastAPI process.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from sqlalchemy import select

from app.celery_app import celery_app
from app.database import AsyncSessionLocal

# Import all models to register with SQLAlchemy
from app.modules.auth.model import User              # noqa: F401
from app.modules.generation.model import (
    Generation,
    GenerationStatus,
)
from app.modules.generation.diffusion import (
    generate_image,
    generate_img2img,
)
from app.modules.generation.safety import check_image
from app.utils.image_utils import save_image


# ============ ASYNC WRAPPER ============
def run_async(coro):
    """Run async function from sync Celery context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ============ TASK: TXT2IMG ============
@celery_app.task(name="generation.txt2img", bind=True, max_retries=2)
def generate_txt2img_task(self, generation_id: int) -> dict:
    """Generate a text-to-image in the background."""
    return run_async(_generate_txt2img_async(generation_id))


async def _generate_txt2img_async(generation_id: int) -> dict:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Generation).where(Generation.id == generation_id)
        )
        generation = result.scalar_one_or_none()
        if generation is None:
            return {"status": "error", "message": "Generation not found"}

        generation.status = GenerationStatus.PROCESSING
        await db.commit()

        try:
            image, seed_used = generate_image(
                prompt=generation.prompt,
                negative_prompt=generation.negative_prompt,
                width=generation.width,
                height=generation.height,
                steps=generation.steps,
                cfg_scale=generation.cfg_scale,
                seed=generation.seed,
                show_progress=False,
            )

            is_safe, nsfw_score, label = check_image(image)

            if is_safe:
                relative_path = save_image(
                    image, subdir="generated", prefix="txt2img", seed=seed_used
                )
                generation.output_path = relative_path
                generation.status = GenerationStatus.COMPLETED
                generation.nsfw_score = nsfw_score
            else:
                save_image(
                    image, subdir="rejected", prefix="txt2img", seed=seed_used
                )
                generation.output_path = None
                generation.status = GenerationStatus.BLOCKED_NSFW
                generation.nsfw_score = nsfw_score
                generation.error_message = (
                    f"Output blocked (score: {nsfw_score:.4f}, label: {label})"
                )

            generation.seed = seed_used
            generation.completed_at = datetime.now(timezone.utc)
            await db.commit()

            return {
                "status": "success",
                "generation_id": generation_id,
                "generation_status": generation.status.value,
            }

        except Exception as e:
            generation.status = GenerationStatus.FAILED
            generation.error_message = f"{type(e).__name__}: {str(e)}"
            generation.completed_at = datetime.now(timezone.utc)
            await db.commit()
            return {"status": "error", "message": str(e)}


# ============ TASK: IMG2IMG ============
@celery_app.task(name="generation.img2img", bind=True, max_retries=2)
def generate_img2img_task(self, generation_id: int) -> dict:
    """Generate an image-to-image in the background."""
    return run_async(_generate_img2img_async(generation_id))


async def _generate_img2img_async(generation_id: int) -> dict:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Generation).where(Generation.id == generation_id)
        )
        generation = result.scalar_one_or_none()
        if generation is None:
            return {"status": "error", "message": "Generation not found"}

        # Reference image path check
        if not generation.ref_image_path:
            generation.status = GenerationStatus.FAILED
            generation.error_message = "Missing reference image path"
            generation.completed_at = datetime.now(timezone.utc)
            await db.commit()
            return {"status": "error", "message": "No reference image"}

        generation.status = GenerationStatus.PROCESSING
        await db.commit()

        try:
            # Load reference image from file
            from app.config import settings
            ref_path = settings.STORAGE_DIR / generation.ref_image_path
            reference_image = Image.open(ref_path).convert("RGB")

            image, seed_used = generate_img2img(
                prompt=generation.prompt,
                init_image=reference_image,
                negative_prompt=generation.negative_prompt,
                width=generation.width,
                height=generation.height,
                steps=generation.steps,
                cfg_scale=generation.cfg_scale,
                strength=generation.strength or 0.15,
                seed=generation.seed,
                show_progress=False,
            )

            is_safe, nsfw_score, label = check_image(image)

            if is_safe:
                relative_path = save_image(
                    image, subdir="generated", prefix="img2img", seed=seed_used
                )
                generation.output_path = relative_path
                generation.status = GenerationStatus.COMPLETED
                generation.nsfw_score = nsfw_score
            else:
                save_image(
                    image, subdir="rejected", prefix="img2img", seed=seed_used
                )
                generation.output_path = None
                generation.status = GenerationStatus.BLOCKED_NSFW
                generation.nsfw_score = nsfw_score
                generation.error_message = (
                    f"Output blocked (score: {nsfw_score:.4f}, label: {label})"
                )

            generation.seed = seed_used
            generation.completed_at = datetime.now(timezone.utc)
            await db.commit()

            return {
                "status": "success",
                "generation_id": generation_id,
                "generation_status": generation.status.value,
            }

        except Exception as e:
            generation.status = GenerationStatus.FAILED
            generation.error_message = f"{type(e).__name__}: {str(e)}"
            generation.completed_at = datetime.now(timezone.utc)
            await db.commit()
            return {"status": "error", "message": str(e)}


# ============ TASK: VARIATIONS ============
@celery_app.task(name="generation.variations", bind=True, max_retries=1)
def generate_variations_task(self, generation_ids: list[int]) -> dict:
    """Generate multiple variations in the background."""
    return run_async(_generate_variations_async(generation_ids))


async def _generate_variations_async(generation_ids: list[int]) -> dict:
    """Process a batch of variation generation IDs."""
    results = []

    for gen_id in generation_ids:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Generation).where(Generation.id == gen_id)
            )
            generation = result.scalar_one_or_none()

            if generation is None:
                results.append({"id": gen_id, "status": "not_found"})
                continue

            generation.status = GenerationStatus.PROCESSING
            await db.commit()

            try:
                image, seed_used = generate_image(
                    prompt=generation.prompt,
                    negative_prompt=generation.negative_prompt,
                    width=generation.width,
                    height=generation.height,
                    steps=generation.steps,
                    cfg_scale=generation.cfg_scale,
                    seed=None,  # random per variation
                    show_progress=False,
                )

                is_safe, nsfw_score, label = check_image(image)

                if is_safe:
                    relative_path = save_image(
                        image, subdir="generated", prefix="var", seed=seed_used
                    )
                    generation.output_path = relative_path
                    generation.status = GenerationStatus.COMPLETED
                    generation.nsfw_score = nsfw_score
                else:
                    save_image(
                        image, subdir="rejected", prefix="var", seed=seed_used
                    )
                    generation.output_path = None
                    generation.status = GenerationStatus.BLOCKED_NSFW
                    generation.nsfw_score = nsfw_score
                    generation.error_message = (
                        f"Output blocked (score: {nsfw_score:.4f}, label: {label})"
                    )

                generation.seed = seed_used
                generation.completed_at = datetime.now(timezone.utc)
                await db.commit()

                results.append({
                    "id": gen_id,
                    "status": generation.status.value,
                    "seed": seed_used,
                })

            except Exception as e:
                generation.status = GenerationStatus.FAILED
                generation.error_message = f"{type(e).__name__}: {str(e)}"
                generation.completed_at = datetime.now(timezone.utc)
                await db.commit()
                results.append({"id": gen_id, "status": "failed", "error": str(e)})

    return {"status": "success", "results": results}