"""
Batch service — bulk generation from CSV or JSON.

Accepts a list of items, creates Generation records, and queues Celery tasks.
"""

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.model import User
from app.modules.generation.model import (
    BatchJob,
    BatchJobStatus,
    Generation,
    GenerationStatus,
    GenerationType,
)
from app.modules.generation.prompt_filter import (
    PromptRejectedError,
    check_prompt,
)
from app.modules.generation.quota_service import check_daily_quota
from app.modules.generation.tasks import (
    generate_img2img_task,
    generate_txt2img_task,
)
from app.utils.image_utils import auto_adjust_dimensions


# ============ CREATE BATCH JOB ============
async def create_batch_job(
    db: AsyncSession,
    user: User,
    items: list[dict],
    name: str | None = None,
) -> BatchJob:
    """
    Create a batch job from a list of items.

    Each item must have:
    - type: 'txt2img' or 'img2img'
    - prompt: str
    - width, height: int
    - steps: int (optional)
    - strength: float (optional, for img2img)
    - ref_image_path: str (required for img2img)

    Returns:
        BatchJob with all items queued.
    """
    # 1. Quota check — cost = num items
    await check_daily_quota(db, user, cost=len(items))

    # 2. Create BatchJob
    batch = BatchJob(
        user_id=user.id,
        name=name or f"Batch {datetime.now(timezone.utc).isoformat()}",
        total_items=len(items),
        status=BatchJobStatus.PROCESSING,
    )
    db.add(batch)
    await db.commit()
    await db.refresh(batch)

    # 3. Create generation records
    queued_task_ids = []

    for item in items:
        try:
            generation = await _create_generation_for_item(
                db, user, batch.id, item
            )
            if generation is None:
                batch.failed_items += 1
                continue

            # If Layer 1 blocked, count as failed (but no task)
            if generation.status == GenerationStatus.BLOCKED_NSFW:
                batch.failed_items += 1
                continue

            # Queue task
            if generation.type == GenerationType.TXT2IMG:
                generate_txt2img_task.delay(generation.id)
            elif generation.type == GenerationType.IMG2IMG:
                generate_img2img_task.delay(generation.id)

            queued_task_ids.append(generation.id)

        except Exception:
            batch.failed_items += 1

    # 4. Update batch
    batch.status = BatchJobStatus.PROCESSING
    await db.commit()
    await db.refresh(batch)

    return batch


async def _create_generation_for_item(
    db: AsyncSession,
    user: User,
    batch_id: int,
    item: dict,
) -> Generation | None:
    """Create a generation from a batch item."""
    # Validate type
    item_type = item.get("type", "txt2img").lower()
    if item_type not in ("txt2img", "img2img"):
        return None

    gen_type = (
        GenerationType.TXT2IMG if item_type == "txt2img"
        else GenerationType.IMG2IMG
    )

    # Validate prompt
    prompt = item.get("prompt", "").strip()
    if not prompt:
        return None

    # Dimensions
    width = int(item.get("width", 512))
    height = int(item.get("height", 512))
    adjusted_width, adjusted_height, was_adjusted = auto_adjust_dimensions(
        width, height
    )

    # img2img specific
    strength = item.get("strength") if gen_type == GenerationType.IMG2IMG else None
    ref_path = item.get("ref_image_path") if gen_type == GenerationType.IMG2IMG else None

    if gen_type == GenerationType.IMG2IMG and not ref_path:
        return None

    # Create record
    generation = Generation(
        user_id=user.id,
        batch_id=batch_id,
        type=gen_type,
        prompt=prompt,
        negative_prompt=item.get("negative_prompt"),
        ref_image_path=ref_path,
        width=adjusted_width,
        height=adjusted_height,
        requested_width=width,
        requested_height=height,
        dimensions_adjusted=was_adjusted,
        steps=int(item.get("steps", 20)),
        cfg_scale=float(item.get("cfg_scale", 7.0)),
        seed=item.get("seed"),
        strength=float(strength) if strength is not None else None,
        status=GenerationStatus.PENDING,
    )
    db.add(generation)
    await db.commit()
    await db.refresh(generation)

    # Layer 1: Prompt check
    try:
        check_prompt(prompt)
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


# ============ GET BATCH STATUS ============
async def get_batch_status(
    db: AsyncSession,
    batch_id: int,
    user_id: int,
) -> dict | None:
    """
    Get batch status with per-item statuses.

    Returns None if batch not found or doesn't belong to user.
    """
    result = await db.execute(
        select(BatchJob).where(BatchJob.id == batch_id)
    )
    batch = result.scalar_one_or_none()

    if batch is None or batch.user_id != user_id:
        return None

    # Get all items
    items_result = await db.execute(
        select(Generation)
        .where(Generation.batch_id == batch_id)
        .order_by(Generation.id)
    )
    items = list(items_result.scalars().all())

    # Count statuses
    completed = sum(1 for g in items if g.status == GenerationStatus.COMPLETED)
    failed = sum(1 for g in items if g.status in (
        GenerationStatus.FAILED,
        GenerationStatus.BLOCKED_NSFW,
    ))
    pending = sum(1 for g in items if g.status in (
        GenerationStatus.PENDING,
        GenerationStatus.PROCESSING,
    ))

    # Update batch if finished
    if pending == 0 and batch.status == BatchJobStatus.PROCESSING:
        batch.completed_items = completed
        batch.failed_items = failed
        batch.status = (
            BatchJobStatus.COMPLETED if failed == 0
            else BatchJobStatus.PARTIAL
        )
        batch.completed_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(batch)

    return {
        "batch_id": batch.id,
        "name": batch.name,
        "status": batch.status.value,
        "total_items": batch.total_items,
        "completed_items": completed,
        "failed_items": failed,
        "pending_items": pending,
        "created_at": batch.created_at,
        "completed_at": batch.completed_at,
        "items": [
            {
                "id": g.id,
                "type": g.type.value,
                "status": g.status.value,
                "prompt": g.prompt[:100],
                "output_path": g.output_path,
            }
            for g in items
        ],
    }


# ============ LIST USER BATCHES ============
async def list_user_batches(
    db: AsyncSession,
    user_id: int,
    skip: int = 0,
    limit: int = 20,
) -> tuple[int, list[BatchJob]]:
    """List user's batch jobs."""
    count_result = await db.execute(
        select(func.count(BatchJob.id)).where(BatchJob.user_id == user_id)
    )
    total = count_result.scalar() or 0

    result = await db.execute(
        select(BatchJob)
        .where(BatchJob.user_id == user_id)
        .order_by(BatchJob.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    batches = list(result.scalars().all())
    return total, batches