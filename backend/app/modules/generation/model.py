

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base



class GenerationType(str, enum.Enum):

    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    VARIATION = "variation"


class GenerationStatus(str, enum.Enum):
  
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED_NSFW = "blocked_nsfw"



class Generation(Base):
  

    __tablename__ = "generations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # Kis user ne banaya
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
        # Batch link (nullable)
    batch_id: Mapped[int | None] = mapped_column(
        ForeignKey("batch_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # Generation type
    type: Mapped[GenerationType] = mapped_column(
        Enum(GenerationType, name="generation_type"),
        nullable=False,
        index=True,
    )

    # Input
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

  
    ref_image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)

    
    requested_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    requested_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dimensions_adjusted: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    # Parameters
    steps: Mapped[int] = mapped_column(Integer, default=20, nullable=False)
    cfg_scale: Mapped[float] = mapped_column(Float, default=7.0, nullable=False)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    strength: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Status
    status: Mapped[GenerationStatus] = mapped_column(
        Enum(GenerationStatus, name="generation_status"),
        default=GenerationStatus.PENDING,
        nullable=False,
        index=True,
    )

    # Output
    output_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Safety
    nsfw_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Errors
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

# ============ BATCH JOB ============
class BatchJobStatus(str, enum.Enum):
    """Batch job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"      # Some failed
    FAILED = "failed"


class BatchJob(Base):
    """A bulk generation job."""

    __tablename__ = "batch_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[BatchJobStatus] = mapped_column(
        Enum(BatchJobStatus, name="batch_job_status"),
        default=BatchJobStatus.PENDING,
        nullable=False,
        index=True,
    )

    total_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<BatchJob id={self.id} status={self.status}>"