
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.model import User
from app.modules.generation.model import (
    Generation,
    GenerationStatus,
    GenerationType,
)
from app.modules.generation.schema import Txt2ImgRequest
from app.utils.image_utils import (
    auto_adjust_dimensions,
    generate_dummy_image,
    save_image,
)



async def create_txt2img_generation(
    db: AsyncSession,
    user: User,
    data: Txt2ImgRequest,
) :

    # 1. Auto-adjust dimensions
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

    try:
        image = generate_dummy_image(
            width=adjusted_width,
            height=adjusted_height,
            text=data.prompt[:40],
        )
        relative_path = save_image(image)

        generation.output_path = relative_path
        generation.status = GenerationStatus.COMPLETED
        generation.completed_at = datetime.now(timezone.utc)

    except Exception as e:
        generation.status = GenerationStatus.FAILED
        generation.error_message = str(e)

    await db.commit()
    await db.refresh(generation)

    return generation



async def get_generation_by_id(
    db: AsyncSession,
    generation_id: int,
) -> Generation | None:
    """ID se generation dhoondho."""
    result = await db.execute(
        select(Generation).where(Generation.id == generation_id)
    )
    return result.scalar_one_or_none()



async def list_user_generations(
    db: AsyncSession,
    user_id: int,
    skip: int = 0,
    limit: int = 20,
) -> tuple[int, list[Generation]]:
    """
    User ki generations list karo (history).
    Returns: (total_count, generations)
    """
    # Total count
    count_result = await db.execute(
        select(Generation).where(Generation.user_id == user_id)
    )
    total = len(count_result.scalars().all())

    # Paginated list
    result = await db.execute(
        select(Generation)
        .where(Generation.user_id == user_id)
        .order_by(Generation.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    generations = list(result.scalars().all())

    return total, generations