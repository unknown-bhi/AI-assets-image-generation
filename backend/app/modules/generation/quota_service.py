
from datetime import datetime, time, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.model import User
from app.modules.generation.model import Generation


# ============ EXCEPTIONS ============
class QuotaExceededError(Exception):
    """Raised when a user exceeds their daily quota."""

    def __init__(self, used: int, limit: int):
        super().__init__(
            f"Daily quota exceeded ({used}/{limit}). "
            f"Resets at midnight UTC."
        )
        self.used = used
        self.limit = limit


# ============ HELPERS ============
def _get_today_start_utc() -> datetime:
    """Return today's start (00:00 UTC)."""
    now = datetime.now(timezone.utc)
    return datetime.combine(now.date(), time.min, tzinfo=timezone.utc)


def _get_tomorrow_start_utc() -> datetime:
    """Return tomorrow's start (00:00 UTC) — when quota resets."""
    today_start = _get_today_start_utc()
    from datetime import timedelta
    return today_start + timedelta(days=1)


# ============ PUBLIC FUNCTIONS ============
async def get_daily_usage(
    db: AsyncSession,
    user: User,
) -> int:
    """
    Count today's generations for a user.

    """
    today_start = _get_today_start_utc()

    result = await db.execute(
        select(func.count(Generation.id))
        .where(Generation.user_id == user.id)
        .where(Generation.created_at >= today_start)
    )
    return result.scalar() or 0


async def get_quota_info(
    db: AsyncSession,
    user: User,
) -> dict:
    """
    Return full quota info for a user.


    """
    used = await get_daily_usage(db, user)
    limit = user.daily_quota
    remaining = max(0, limit - used)

    return {
        "used": used,
        "limit": limit,
        "remaining": remaining,
        "reset_at": _get_tomorrow_start_utc(),
    }


async def check_daily_quota(
    db: AsyncSession,
    user: User,
    cost: int = 1,
) -> None:
    """
    Check if user has enough daily quota.

    """
    used = await get_daily_usage(db, user)
    limit = user.daily_quota

    if used + cost > limit:
        raise QuotaExceededError(used=used, limit=limit)