
from celery import Celery

from app.config import settings


# ============ CELERY APP ============
celery_app = Celery(
    "ai_asset_creator",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.modules.generation.tasks"],
)

from app.modules.auth.model import User            # noqa: F401
from app.modules.generation.model import Generation 
# ============ CONFIGURATION ============
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=1800,          # Hard limit: 30 min
    task_soft_time_limit=1500,     # Soft: 25 min
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # 1 task at a time
    worker_max_tasks_per_child=10, # Restart worker after 10
    result_expires=3600,           # 1 hour
)