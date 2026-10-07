"""
Generation service — business logic.

Qwen-Image 2.1 Turbo use karta hai.
"""

from datetime import datetime, timezone

from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.model import User
from app.modules.generation.model import (
    Generation,
    GenerationStatus,
    GenerationType,
)
from app.modules.generation.prompt_filter import (
    PromptRejectedError,
    check_prompt,
)
from app.modules.generation.history_service import list_generations
from app.modules.generation.safety import check_image
from app.modules.generation.schema import (
    Img2ImgRequest,
    Txt2ImgRequest,
    VariationRequest,
)
from app.modules.generation.qwen_engine import engine
from app.modules.generation.quota_service import check_daily_quota
from app.modules.generation.tasks import (
    generate_img2img_task,
    generate_txt2img_task,
    generate_variations_task,
)
from app.utils.image_utils import auto_adjust_dimensions, save_image


# ============ HELPER: PROCESS OUTPUT ============
async def _process_output(
    db: AsyncSession,
    generation: Generation,
    output_path: str,
    prefix: str,
    seed_used: int = 0,
) -> None:
    """NSFW check + save + DB update (helper)."""
    image = Image.open(output_path)
    is_safe, nsfw_score, label = check_image(image)

    if is_safe:
        relative_path = save_image(
            image, subdir="generated", prefix=prefix, seed=seed_used
        )
        generation.output_path = relative_path
        generation.status = GenerationStatus.COMPLETED
        generation.nsfw_score = nsfw_score
    else:
        save_image(
            image, subdir="rejected", prefix=prefix, seed=seed_used
        )
        generation.output_path = None
        generation.status = GenerationStatus.BLOCKED_NSFW
        generation.nsfw_score = nsfw_score
        generation.error_message = (
            f"Output blocked (score: {nsfw_score:.4f}, label: {label})"
        )

    generation.seed = seed_used
    generation.completed_at = datetime.now(timezone.utc)


# ============ CREATE TXT2IMG GENERATION (SYNC) ============
async def create_txt2img_generation(
    db: AsyncSession,
    user: User,
    data: Txt2ImgRequest,
) -> Generation:
    """Create a txt2img generation (synchronous)."""
    # 0. Quota check
    await check_daily_quota(db, user, cost=1)

    # 1. Auto-adjust dimensions
    adjusted_width, adjusted_height, was_adjusted = auto_adjust_dimensions(
        data.width, data.height
    )

    # 2. Create DB record
    generation = Generation(
        user_id=user.id,
        type=GenerationType.TXT2IMG,
        prompt=data.prompt,
        negative_prompt=data.negative_prompt,
        width=adjusted_width,
        height=adjusted_height,
        requested_width=data.width,
        requested_height=data.height,
        dimensions_adjusted=was_adjusted,
        steps=data.steps,
        cfg_scale=data.cfg_scale,
        seed=data.seed,
        status=GenerationStatus.PENDING,
    )
    db.add(generation)
    await db.commit()
    await db.refresh(generation)

    # ============ LAYER 1: Prompt check ============
    try:
        check_prompt(data.prompt)
    except PromptRejectedError as e:
        generation.status = GenerationStatus.BLOCKED_NSFW
        generation.error_message = (
            f"Prompt blocked ({e.reason}): "
            f"{e.matched_keyword or 'inappropriate content'}"
        )
        generation.completed_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(generation)
        return generation

    # ============ GENERATE + LAYER 3 ============
    try:
        # Qwen se image generate
        output_path = engine.generate(
            prompt=data.prompt,
            negative_prompt=data.negative_prompt or "",
            width=adjusted_width,
            height=adjusted_height,
            steps=data.steps,
            cfg_scale=data.cfg_scale,
            seed=data.seed,
        )

        # NSFW check + save
        await _process_output(
            db=db,
            generation=generation,
            output_path=output_path,
            prefix="txt2img",
            seed_used=data.seed or 0,
        )

    except Exception as e:
        generation.status = GenerationStatus.FAILED
        generation.error_message = f"{type(e).__name__}: {str(e)}"
        generation.completed_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(generation)
    return generation


# ============ CREATE IMG2IMG GENERATION (SYNC) ============
async def create_img2img_generation(
    db: AsyncSession,
    user: User,
    data: Img2ImgRequest,
    reference_image: Image.Image,
    reference_path: str | None = None,
) -> Generation:
    """Create an img2img generation (synchronous)."""
    # 0. Quota check
    await check_daily_quota(db, user, cost=1)

    # 1. Auto-adjust dimensions
    adjusted_width, adjusted_height, was_adjusted = auto_adjust_dimensions(
        data.width, data.height
    )

    # 2. Create DB record
    generation = Generation(
        user_id=user.id,
        type=GenerationType.IMG2IMG,
        prompt=data.prompt,
        negative_prompt=data.negative_prompt,
        ref_image_path=reference_path,
        width=adjusted_width,
        height=adjusted_height,
        requested_width=data.width,
        requested_height=data.height,
        dimensions_adjusted=was_adjusted,
        steps=data.steps,
        cfg_scale=data.cfg_scale,
        seed=data.seed,
        strength=data.strength,
        status=GenerationStatus.PENDING,
    )
    db.add(generation)
    await db.commit()
    await db.refresh(generation)

    # ============ LAYER 1 ============
    try:
        check_prompt(data.prompt)
    except PromptRejectedError as e:
        generation.status = GenerationStatus.BLOCKED_NSFW
        generation.error_message = (
            f"Prompt blocked ({e.reason}): "
            f"{e.matched_keyword or 'inappropriate content'}"
        )
        generation.completed_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(generation)
        return generation

    # ============ GENERATE + LAYER 3 ============
    try:
        # ⚠️ Qwen ka img2img mode abhi sd-cli mein complete nahi
        # Filhal txt2img fallback
        output_path = engine.generate(
            prompt=data.prompt,
            negative_prompt=data.negative_prompt or "",
            width=adjusted_width,
            height=adjusted_height,
            steps=data.steps,
            cfg_scale=data.cfg_scale,
            seed=data.seed,
        )

        await _process_output(
            db=db,
            generation=generation,
            output_path=output_path,
            prefix="img2img",
            seed_used=data.seed or 0,
        )

    except Exception as e:
        generation.status = GenerationStatus.FAILED
        generation.error_message = f"{type(e).__name__}: {str(e)}"
        generation.completed_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(generation)
    return generation


# ============ GET ============
async def get_generation_by_id(
    db: AsyncSession,
    generation_id: int,
) -> Generation | None:
    """Fetch a generation by ID."""
    result = await db.execute(
        select(Generation).where(Generation.id == generation_id)
    )
    return result.scalar_one_or_none()


# ============ LIST ============
async def list_user_generations(
    db: AsyncSession,
    user_id: int,
    skip: int = 0,
    limit: int = 20,
) -> tuple[int, list[Generation]]:
    """List a user's generations."""
    return await list_generations(db, user_id, skip=skip, limit=limit)


# ============ CREATE VARIATIONS (SYNC) ============
async def create_variations(
    db: AsyncSession,
    user: User,
    data: VariationRequest,
) -> list[Generation]:
    """Create multiple variations (synchronous)."""
    # 0. Quota
    await check_daily_quota(db, user, cost=data.num_variations)

    # 1. Auto-adjust dimensions
    adjusted_width, adjusted_height, was_adjusted = auto_adjust_dimensions(
        data.width, data.height
    )

    # ============ LAYER 1 ============
    try:
        check_prompt(data.prompt)
    except PromptRejectedError as e:
        generation = Generation(
            user_id=user.id,
            type=GenerationType.VARIATION,
            prompt=data.prompt,
            negative_prompt=data.negative_prompt,
            width=adjusted_width,
            height=adjusted_height,
            requested_width=data.width,
            requested_height=data.height,
            dimensions_adjusted=was_adjusted,
            steps=data.steps,
            cfg_scale=data.cfg_scale,
            status=GenerationStatus.BLOCKED_NSFW,
            error_message=(
                f"Prompt blocked ({e.reason}): "
                f"{e.matched_keyword or 'inappropriate content'}"
            ),
            completed_at=datetime.now(timezone.utc),
        )
        db.add(generation)
        await db.commit()
        await db.refresh(generation)
        return [generation]

    # ============ GENERATE + LAYER 3 (loop) ============
    generations: list[Generation] = []

    for _ in range(data.num_variations):
        generation = Generation(
            user_id=user.id,
            type=GenerationType.VARIATION,
            prompt=data.prompt,
            negative_prompt=data.negative_prompt,
            width=adjusted_width,
            height=adjusted_height,
            requested_width=data.width,
            requested_height=data.height,
            dimensions_adjusted=was_adjusted,
            steps=data.steps,
            cfg_scale=data.cfg_scale,
            status=GenerationStatus.PENDING,
        )
        db.add(generation)
        await db.commit()
        await db.refresh(generation)

        try:
            output_path = engine.generate(
                prompt=data.prompt,
                negative_prompt=data.negative_prompt or "",
                width=adjusted_width,
                height=adjusted_height,
                steps=data.steps,
                cfg_scale=data.cfg_scale,
                seed=None,  # Random per variation
            )

            await _process_output(
                db=db,
                generation=generation,
                output_path=output_path,
                prefix="var",
                seed_used=0,
            )

        except Exception as e:
            generation.status = GenerationStatus.FAILED
            generation.error_message = f"{type(e).__name__}: {str(e)}"
            generation.completed_at = datetime.now(timezone.utc)

        await db.commit()
        await db.refresh(generation)
        generations.append(generation)

    return generations


# ============ QUEUE TXT2IMG (Celery) ============
async def queue_txt2img_generation(
    db: AsyncSession,
    user: User,
    data: Txt2ImgRequest,
) -> Generation:
    """Queue txt2img as Celery task."""
    await check_daily_quota(db, user, cost=1)

    adjusted_width, adjusted_height, was_adjusted = auto_adjust_dimensions(
        data.width, data.height
    )

    generation = Generation(
        user_id=user.id,
        type=GenerationType.TXT2IMG,
        prompt=data.prompt,
        negative_prompt=data.negative_prompt,
        width=adjusted_width,
        height=adjusted_height,
        requested_width=data.width,
        requested_height=data.height,
        dimensions_adjusted=was_adjusted,
        steps=data.steps,
        cfg_scale=data.cfg_scale,
        seed=data.seed,
        status=GenerationStatus.PENDING,
    )
    db.add(generation)
    await db.commit()
    await db.refresh(generation)

    generate_txt2img_task.delay(generation.id)
    return generation


# ============ QUEUE IMG2IMG ============
async def queue_img2img_generation(
    db: AsyncSession,
    user: User,
    data: Img2ImgRequest,
    reference_image: Image.Image,
) -> Generation:
    """Queue img2img as Celery task."""
    await check_daily_quota(db, user, cost=1)

    reference_filename = save_image(
        reference_image, subdir="uploads", prefix="ref", seed=0
    )

    adjusted_width, adjusted_height, was_adjusted = auto_adjust_dimensions(
        data.width, data.height
    )

    generation = Generation(
        user_id=user.id,
        type=GenerationType.IMG2IMG,
        prompt=data.prompt,
        negative_prompt=data.negative_prompt,
        ref_image_path=reference_filename,
        width=adjusted_width,
        height=adjusted_height,
        requested_width=data.width,
        requested_height=data.height,
        dimensions_adjusted=was_adjusted,
        steps=data.steps,
        cfg_scale=data.cfg_scale,
        seed=data.seed,
        strength=data.strength,
        status=GenerationStatus.PENDING,
    )
    db.add(generation)
    await db.commit()
    await db.refresh(generation)

    # Layer 1
    try:
        check_prompt(data.prompt)
    except PromptRejectedError as e:
        generation.status = GenerationStatus.BLOCKED_NSFW
        generation.error_message = (
            f"Prompt blocked ({e.reason}): "
            f"{e.matched_keyword or 'inappropriate content'}"
        )
        generation.completed_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(generation)
        return generation

    generate_img2img_task.delay(generation.id)
    return generation


# ============ QUEUE VARIATIONS ============
async def queue_variations(
    db: AsyncSession,
    user: User,
    data: VariationRequest,
) -> list[Generation]:
    """Queue variations as single Celery task."""
    await check_daily_quota(db, user, cost=data.num_variations)

    adjusted_width, adjusted_height, was_adjusted = auto_adjust_dimensions(
        data.width, data.height
    )

    # Layer 1
    try:
        check_prompt(data.prompt)
    except PromptRejectedError as e:
        generation = Generation(
            user_id=user.id,
            type=GenerationType.VARIATION,
            prompt=data.prompt,
            negative_prompt=data.negative_prompt,
            width=adjusted_width,
            height=adjusted_height,
            requested_width=data.width,
            requested_height=data.height,
            dimensions_adjusted=was_adjusted,
            steps=data.steps,
            cfg_scale=data.cfg_scale,
            status=GenerationStatus.BLOCKED_NSFW,
            error_message=(
                f"Prompt blocked ({e.reason}): "
                f"{e.matched_keyword or 'inappropriate content'}"
            ),
            completed_at=datetime.now(timezone.utc),
        )
        db.add(generation)
        await db.commit()
        await db.refresh(generation)
        return [generation]

    # Create N records
    generations = []
    for _ in range(data.num_variations):
        generation = Generation(
            user_id=user.id,
            type=GenerationType.VARIATION,
            prompt=data.prompt,
            negative_prompt=data.negative_prompt,
            width=adjusted_width,
            height=adjusted_height,
            requested_width=data.width,
            requested_height=data.height,
            dimensions_adjusted=was_adjusted,
            steps=data.steps,
            cfg_scale=data.cfg_scale,
            status=GenerationStatus.PENDING,
        )
        db.add(generation)
        generations.append(generation)

    await db.commit()
    for g in generations:
        await db.refresh(g)

    generate_variations_task.delay([g.id for g in generations])
    return generations