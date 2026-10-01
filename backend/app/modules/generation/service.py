
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
from app.modules.generation.history_service import (
    get_user_stats,
    list_generations,
)
from app.modules.generation.safety import check_image
from app.modules.generation.schema import Img2ImgRequest, Txt2ImgRequest,VariationRequest
from app.utils.image_utils import auto_adjust_dimensions, save_image
from app.modules.generation.quota_service import check_daily_quota
from app.modules.generation.tasks import (
    generate_img2img_task,
    generate_txt2img_task,
    generate_variations_task,
)
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
    # 0. Check quota (NEW)
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
            relative_path = save_image(
                image,
                subdir="generated",
                prefix="txt2img",
                seed=seed_used,
            )
            generation.output_path = relative_path
            generation.status = GenerationStatus.COMPLETED
            generation.nsfw_score = nsfw_score
        else:
            save_image(
                image,
                subdir="rejected",
                prefix="txt2img",
                seed=seed_used,
            )
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
   # 0. Check quota (NEW)
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
            relative_path = save_image(
                image,
                subdir="generated",
                prefix="img2img",
                seed=seed_used,
            )
            generation.output_path = relative_path
            generation.status = GenerationStatus.COMPLETED
            generation.nsfw_score = nsfw_score
        else:
            save_image(
                image,
                subdir="rejected",
                prefix="img2img",
                seed=seed_used,
            )
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
    """
    List a user's generations (backward-compatible wrapper).

    For advanced filtering, use history_service.list_generations directly.
    """
    from app.modules.generation.history_service import list_generations

    return await list_generations(
        db,
        user_id,
        skip=skip,
        limit=limit,
    )


# ============ CREATE VARIATIONS ============
async def create_variations(
    db: AsyncSession,
    user: User,
    data: VariationRequest,
) -> list[Generation]:
    """
    Create multiple variations from a single prompt.

    """
   # 0. Check quota (NEW)
    await check_daily_quota(db, user, cost=1)

    # 1. Auto-adjust dimensions (once, same for all)
    adjusted_width, adjusted_height, was_adjusted = auto_adjust_dimensions(
        data.width, data.height
    )

    # ============ LAYER 1: Prompt check (ek dafa) ============
    try:
        check_prompt(data.prompt)
    except PromptRejectedError as e:
        # Ek single record with BLOCKED_NSFW, sab variations ke liye
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

    for i in range(data.num_variations):
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
            image, seed_used = generate_image(
                prompt=data.prompt,
                negative_prompt=data.negative_prompt,
                width=adjusted_width,
                height=adjusted_height,
                steps=data.steps,
                cfg_scale=data.cfg_scale,
                seed=None,   # ⭐ Random seed har variation ke liye
                show_progress=False,
            )

            is_safe, nsfw_score, label = check_image(image)

            if is_safe:
                relative_path = save_image(
                    image,
                    subdir="generated",
                    prefix="var",
                    seed=seed_used,
                )
                generation.output_path = relative_path
                generation.status = GenerationStatus.COMPLETED
                generation.nsfw_score = nsfw_score
            else:
                save_image(
                    image,
                    subdir="rejected",
                    prefix="var",
                    seed=seed_used,
                )
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
        generations.append(generation)

    return generations

from app.modules.generation.tasks import generate_txt2img_task


# ============ QUEUE TXT2IMG (async) ============
async def queue_txt2img_generation(
    db: AsyncSession,
    user: User,
    data: Txt2ImgRequest,
) -> Generation:
    """
    Queue a txt2img generation as a Celery task.

    Returns immediately with status=pending.
    Client polls /generate/{id} for status.
    """
    # 0. Quota check
    from app.modules.generation.quota_service import check_daily_quota
    await check_daily_quota(db, user, cost=1)

    # 1. Auto-adjust dimensions
    adjusted_width, adjusted_height, was_adjusted = auto_adjust_dimensions(
        data.width, data.height
    )

    # 2. Create DB record (status=pending)
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

    # 3. Queue Celery task
    generate_txt2img_task.delay(generation.id)

    return generation

# ============ QUEUE IMG2IMG ============
async def queue_img2img_generation(
    db: AsyncSession,
    user: User,
    data: Img2ImgRequest,
    reference_image: Image.Image,
) -> Generation:
    """
    Queue an img2img generation.

    Saves the reference image to uploads/, then queues a Celery task.
    """
    from app.modules.generation.quota_service import check_daily_quota
    from app.modules.generation.prompt_filter import (
        PromptRejectedError,
        check_prompt,
    )

    # 0. Quota check
    await check_daily_quota(db, user, cost=1)

    # 1. Save reference image first
    reference_filename = save_image(
        reference_image,
        subdir="uploads",
        prefix="ref",
        seed=0,   # no seed for reference
    )

    # 2. Auto-adjust dimensions
    adjusted_width, adjusted_height, was_adjusted = auto_adjust_dimensions(
        data.width, data.height
    )

    # 3. Create DB record
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

    # 4. Layer 1: Prompt check
    try:
        check_prompt(data.prompt)
    except PromptRejectedError as e:
        generation.status = GenerationStatus.BLOCKED_NSFW
        generation.error_message = (
            f"Prompt blocked ({e.reason}): "
            f"{e.matched_keyword or 'inappropriate content'}"
        )
        from datetime import datetime, timezone
        generation.completed_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(generation)
        return generation

    # 5. Queue task
    generate_img2img_task.delay(generation.id)

    return generation


# ============ QUEUE VARIATIONS ============
async def queue_variations(
    db: AsyncSession,
    user: User,
    data: VariationRequest,
) -> list[Generation]:
    """
    Queue multiple variations as a single Celery task.

    Creates N DB records, then queues one task with all IDs.
    """
    from app.modules.generation.quota_service import check_daily_quota
    from app.modules.generation.prompt_filter import (
        PromptRejectedError,
        check_prompt,
    )

    # 0. Quota check (cost = num_variations)
    await check_daily_quota(db, user, cost=data.num_variations)

    # 1. Auto-adjust dimensions
    adjusted_width, adjusted_height, was_adjusted = auto_adjust_dimensions(
        data.width, data.height
    )

    # 2. Layer 1: Prompt check (once)
    try:
        check_prompt(data.prompt)
    except PromptRejectedError as e:
        # Single blocked record
        from datetime import datetime, timezone
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

    # 3. Create N records
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

    # 4. Queue single task with all IDs
    generation_ids = [g.id for g in generations]
    generate_variations_task.delay(generation_ids)

    return generations