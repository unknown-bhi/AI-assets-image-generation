"""
History service — user generation history and statistics.

Provides filtering, pagination, and aggregated stats.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.generation.model import (
    Generation,
    GenerationStatus,
    GenerationType,
)


# ============ LIST WITH FILTERS ============
async def list_generations(
    db: AsyncSession,
    user_id: int,
    *,
    generation_type: GenerationType | None = None,
    status: GenerationStatus | None = None,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    skip: int = 0,
    limit: int = 20,
) -> tuple[int, list[Generation]]:
    """
    List generations with optional filters.


    """
    # Base filter
    conditions = [Generation.user_id == user_id]

    if generation_type is not None:
        conditions.append(Generation.type == generation_type)

    if status is not None:
        conditions.append(Generation.status == status)

    if from_date is not None:
        conditions.append(Generation.created_at >= from_date)

    if to_date is not None:
        conditions.append(Generation.created_at <= to_date)

    # Total count
    count_result = await db.execute(
        select(func.count(Generation.id)).where(*conditions)
    )
    total = count_result.scalar() or 0

    # Clamp limit
    limit = min(limit, 100)

    # Fetch page
    result = await db.execute(
        select(Generation)
        .where(*conditions)
        .order_by(Generation.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    generations = list(result.scalars().all())

    return total, generations


# ============ STATS ============
async def get_user_stats(
    db: AsyncSession,
    user_id: int,
) -> dict:
    """
    Return aggregated statistics for a user.

    """
    now = datetime.now(timezone.utc)
    today_start = datetime.combine(now.date(), datetime.min.time(), tzinfo=timezone.utc)
    week_start = today_start - timedelta(days=7)

    # Total
    total_result = await db.execute(
        select(func.count(Generation.id))
        .where(Generation.user_id == user_id)
    )
    total = total_result.scalar() or 0

    # Today
    today_result = await db.execute(
        select(func.count(Generation.id))
        .where(Generation.user_id == user_id)
        .where(Generation.created_at >= today_start)
    )
    today = today_result.scalar() or 0

    # This week
    week_result = await db.execute(
        select(func.count(Generation.id))
        .where(Generation.user_id == user_id)
        .where(Generation.created_at >= week_start)
    )
    this_week = week_result.scalar() or 0

    # By status
    status_result = await db.execute(
        select(Generation.status, func.count(Generation.id))
        .where(Generation.user_id == user_id)
        .group_by(Generation.status)
    )
    by_status = {str(status.value): count for status, count in status_result.all()}

    # By type
    type_result = await db.execute(
        select(Generation.type, func.count(Generation.id))
        .where(Generation.user_id == user_id)
        .group_by(Generation.type)
    )
    by_type = {str(t.value): count for t, count in type_result.all()}

    return {
        "total": total,
        "today": today,
        "this_week": this_week,
        "by_status": by_status,
        "by_type": by_type,
    }