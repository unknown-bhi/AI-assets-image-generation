
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.dependencies import get_current_active_user
from app.database import get_db
from app.modules.auth.model import User
from app.modules.generation import service
from app.modules.generation.schema import (
    GenerationListOut,
    GenerationOut,
    Txt2ImgRequest,
)

router = APIRouter(prefix="/generate", tags=["generation"])



@router.post(
    "/txt2img",
    response_model=GenerationOut,
    status_code=status.HTTP_201_CREATED,
)
async def txt2img(
    request: Txt2ImgRequest,                                       # ⭐ "request" naam — koi wrapper nahi
    current_user: User = Depends(get_current_active_user),         # ⭐ Ye dependency hai, body field nahi
    db: AsyncSession = Depends(get_db),                            # ⭐ Ye bhi dependency
) -> GenerationOut:

    return await service.create_txt2img_generation(db, current_user, request)



@router.get("/history", response_model=GenerationListOut)
async def history(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> GenerationListOut:

    total, generations = await service.list_user_generations(
        db, current_user.id, skip=skip, limit=limit
    )
    return GenerationListOut(total=total, items=generations)