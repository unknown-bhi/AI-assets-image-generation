"""
Generation router — HTTP endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.dependencies import get_current_active_user
from app.database import get_db
from app.modules.auth.model import User
from app.modules.generation import service
from app.modules.generation.model import GenerationStatus
from app.modules.generation.schema import (
    GenerationListOut,
    GenerationOut,
    Txt2ImgRequest,
)

router = APIRouter(prefix="/generate", tags=["generation"])


# ============ TXT2IMG ============
@router.post(
    "/txt2img",
    response_model=GenerationOut,
    status_code=status.HTTP_201_CREATED,
)
async def txt2img(
    request: Txt2ImgRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> GenerationOut:
    """
    Generate an image from a text prompt.

    Layer 1 (prompt filter) blocks unsafe prompts immediately with HTTP 400.
    Layer 3 (output filter) blocks unsafe images — status will be 'blocked_nsfw'.
    """
    generation = await service.create_txt2img_generation(
        db, current_user, request
    )

    # Layer 1 rejection → 400 error (immediate feedback)
    if (
        generation.status == GenerationStatus.BLOCKED_NSFW
        and generation.output_path is None
        and generation.nsfw_score is None
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=generation.error_message or "Prompt rejected",
        )

    return generation


# ============ HISTORY ============
@router.get("/history", response_model=GenerationListOut)
async def history(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> GenerationListOut:
    """User's generation history."""
    total, generations = await service.list_user_generations(
        db, current_user.id, skip=skip, limit=limit
    )
    return GenerationListOut(total=total, items=generations)


# ============ GET BY ID ============
@router.get("/{generation_id}", response_model=GenerationOut)
async def get_generation(
    generation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> GenerationOut:
    """Get a generation by ID."""
    generation = await service.get_generation_by_id(db, generation_id)

    if generation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Generation not found",
        )

    if generation.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized",
        )

    return generation