"""
Generation router — HTTP endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status,File,Form,UploadFile,status
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.dependencies import get_current_active_user
from app.database import get_db
from app.modules.auth.model import User
from app.modules.generation import service, batch_service
from app.modules.generation.model import GenerationStatus
from app.modules.generation.schema import (
    BatchCreate,
    BatchListOut,
    BatchOut,
    GenerationListOut,
    GenerationOut,
    Txt2ImgRequest,
    VariationRequest,
    VariationsOut,
)

from PIL import Image
import io
from fastapi.responses import FileResponse
router = APIRouter(prefix="/generate", tags=["generation"])


# ============ TXT2IMG ============
@router.post(
    "/txt2img",
    response_model=GenerationOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def txt2img(
    request: Txt2ImgRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> GenerationOut:
    """Queue a text-to-image generation (async)."""
    from app.modules.generation.service import queue_txt2img_generation
    return await queue_txt2img_generation(db, current_user, request)

# ============ IMG2IMG (File Upload + Queue) ============
@router.post(
    "/img2img",
    response_model=GenerationOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def img2img(
    reference: UploadFile = File(..., description="Reference image"),
    prompt: str = Form(..., min_length=1, max_length=1000),
    negative_prompt: str | None = Form(None),
    width: int = Form(512, ge=256, le=1024),
    height: int = Form(512, ge=256, le=1024),
    steps: int = Form(20, ge=1, le=50),
    cfg_scale: float = Form(7.0, ge=1.0, le=20.0),
    strength: float = Form(0.15, ge=0.05, le=0.95),
    seed: int | None = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> GenerationOut:
    """
    Queue an image-to-image generation.

    Upload a reference image + prompt. The generation runs in background.
    Poll /generate/{id} to check status.
    """
    from app.modules.generation.schema import Img2ImgRequest
    from app.modules.generation.service import queue_img2img_generation

    # Validate file type
    if reference.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image format. Use PNG, JPEG, or WEBP.",
        )

    # Read image
    try:
        contents = await reference.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}",
        )

    # Build request
    data = Img2ImgRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        strength=strength,
        seed=seed,
    )

    # Queue
    generation = await queue_img2img_generation(db, current_user, data, image)

    return generation

# ============ VARIATIONS ============
@router.post(
    "/variations",response_model=VariationsOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def variations(
    request: VariationRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Queue multiple variations from a single prompt.

    Returns a list of Generation records (one per variation).
    """
    from app.modules.generation.service import queue_variations

    generations = await queue_variations(db, current_user, request)

    return VariationsOut(
        total=len(generations),
        generation_ids=[g.id for g in generations],
        items=[GenerationOut.model_validate(g) for g in generations],
    )


# ============ BATCH ============
@router.post(
    "/batch",
    response_model=BatchOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_batch(
    request: BatchCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Create a batch job with multiple items."""
    items = [item.model_dump() for item in request.items]
    batch = await batch_service.create_batch_job(
        db, current_user, items, name=request.name
    )
    return await batch_service.get_batch_status(db, batch.id, current_user.id)


@router.get("/batch/{batch_id}", response_model=BatchOut)
async def get_batch(
    batch_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get batch status."""
    result = await batch_service.get_batch_status(db, batch_id, current_user.id)
    if result is None:
        raise HTTPException(404, "Batch not found")
    return result


@router.get("/batches", response_model=BatchListOut)
async def list_batches(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    total, batches = await batch_service.list_user_batches(
        db, current_user.id, skip=skip, limit=limit
    )

    # Convert to dict format (JSON serializable)
    items = []
    for b in batches:
        # Get full status for each batch
        status = await batch_service.get_batch_status(db, b.id, current_user.id)
        if status:
            items.append(status)

    return {"total": total, "items": items}


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


# ============ STATS ============
@router.get("/stats")
async def stats(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """User's generation stats."""
    from app.modules.generation.history_service import get_user_stats
    from app.modules.generation.quota_service import get_quota_info

    user_stats = await get_user_stats(db, current_user.id)
    quota = await get_quota_info(db, current_user)

    return {
        "stats": user_stats,
        "quota": {
            "used": quota["used"],
            "limit": quota["limit"],
            "remaining": quota["remaining"],
            "reset_at": quota["reset_at"].isoformat(),
        },
    }

# ============ SERVE IMAGE ============


@router.get("/file/{subdir}/{filename}")
async def serve_file(
    subdir: str,
    filename: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Serve a generated image file.

    Only 'generated' subdir is exposed. Path traversal is blocked.
    """
    from pathlib import Path
    from app.config import settings

    # Security: only allow 'generated' subdir
    if subdir not in ("generated", "uploads"):
        raise HTTPException(403, "Access denied")

    # Security: block path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")

    file_path = settings.STORAGE_DIR / subdir / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "File not found")

    return FileResponse(file_path, media_type="image/png")


# ============ GET GENERATION (AAKHIR MEIN) ============
@router.get("/{generation_id}", response_model=GenerationOut)
async def get_generation(
    generation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> GenerationOut:
    generation = await service.get_generation_by_id(db, generation_id)
    if generation is None:
        raise HTTPException(404, "Generation not found")
    if generation.user_id != current_user.id:
        raise HTTPException(403, "Not authorized")
    return generation