"""
Generation service — business logic for image generation.

Handles:
- Text-to-image (SD 1.5)
- Image-to-image (SD 1.5)
- Prompt safety (Layer 1)
- Output safety (Layer 3)
- Database persistence
- File storage
"""

from datetime import datetime, timezone

from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.model import User
from app.modules.generation.diffusion import (
    generate_image,
    generate_img2img,
)
from app.modules.generation.model import (
    Generation,
    GenerationStatus,
    GenerationType,
)
from app.modules.generation.prompt_filter import (
    PromptRejectedError,
    check_prompt,
)
from app.modules.generation.safety import check_image
from app.modules.generation.schema import Img2ImgRequest, Txt2ImgRequest
from app.utils.image_utils import auto_adjust_dimensions, save_image


# ============ CREATE TXT2IMG GENERATION ============
async def create_txt2img_generation(
    db: AsyncSession,
    user: User,
    data: Txt2ImgRequest,
) -> Generation:
    """
    Create a text-to-image generation with safety layers.

    Layers:
    - Layer 1: Prompt check
    - Layer 3: Output check
    """
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
        image, seed_used = generate_image(
            prompt=data.prompt,
            negative_prompt=data.negative_prompt,
            width=adjusted_width,
            height=adjusted_height,
            steps=data.steps,
            cfg_scale=data.cfg_scale,
            seed=data.seed,
            show_progress=False,
        )

        is_safe, nsfw_score, label = check_image(image)

        if is_safe:
            relative_path = save_image(image, subdir="generated")
            generation.output_path = relative_path
            generation.status = GenerationStatus.COMPLETED
            generation.nsfw_score = nsfw_score
        else:
            save_image(image, subdir="rejected")
            generation.output_path = None
            generation.status = GenerationStatus.BLOCKED_NSFW
            generation.nsfw_score = nsfw_score
            generation.error_message = (
                f"Output blocked (score: {nsfw_score:.4f}, label: {label})"
            )

        generation.seed = seed_used
        generation.completed_at = datetime.now(timezone.utc)

    except Exception as e:
        generation.status = GenerationStatus.FAILED
        generation.error_message = f"{type(e).__name__}: {str(e)}"
        generation.completed_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(generation)

    return generation


# ============ CREATE IMG2IMG GENERATION ============
async def create_img2img_generation(
    db: AsyncSession,
    user: User,
    data: Img2ImgRequest,
    reference_image: Image.Image,
    reference_path: str | None = None,
) -> Generation:
    """
    Create an image-to-image generation with safety layers.

    Layers:
    - Layer 1: Prompt check
    - Layer 3: Output check

    Args:
        db: Database session.
        user: Current user.
        data: img2img request.
        reference_image: PIL Image — the reference to base generation on.
        reference_path: Optional path where the reference is stored (for records).

    Returns:
        The created Generation record.
    """
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
        image, seed_used = generate_img2img(
            prompt=data.prompt,
            init_image=reference_image,
            negative_prompt=data.negative_prompt,
            width=adjusted_width,
            height=adjusted_height,
            steps=data.steps,
            cfg_scale=data.cfg_scale,
            strength=data.strength,
            seed=data.seed,
            show_progress=False,
        )

        is_safe, nsfw_score, label = check_image(image)

        if is_safe:
            relative_path = save_image(image, subdir="generated")
            generation.output_path = relative_path
            generation.status = GenerationStatus.COMPLETED
            generation.nsfw_score = nsfw_score
        else:
            save_image(image, subdir="rejected")
            generation.output_path = None
            generation.status = GenerationStatus.BLOCKED_NSFW
            generation.nsfw_score = nsfw_score
            generation.error_message = (
                f"Output blocked (score: {nsfw_score:.4f}, label: {label})"
            )

        generation.seed = seed_used
        generation.completed_at = datetime.now(timezone.utc)

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
    """List a user's generations (history), paginated."""
    count_result = await db.execute(
        select(Generation).where(Generation.user_id == user_id)
    )
    total = len(count_result.scalars().all())

    result = await db.execute(
        select(Generation)
        .where(Generation.user_id == user_id)
        .order_by(Generation.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    generations = list(result.scalars().all())

    return total, generations