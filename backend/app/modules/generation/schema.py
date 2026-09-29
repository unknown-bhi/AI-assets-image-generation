
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.modules.generation.model import GenerationStatus, GenerationType



class Txt2ImgRequest(BaseModel):
    """Text-to-image request."""
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: str | None = Field(None, max_length=1000)

    width: int = Field(512, ge=256, le=1024)
    height: int = Field(512, ge=256, le=1024)

    steps: int = Field(20, ge=1, le=50)
    cfg_scale: float = Field(7.0, ge=1.0, le=20.0)
    seed: int | None = Field(None, ge=0)


class Img2ImgRequest(BaseModel):
    """Image-to-image request."""
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: str | None = Field(None, max_length=1000)

    width: int = Field(512, ge=256, le=1024)
    height: int = Field(512, ge=256, le=1024)

    steps: int = Field(20, ge=1, le=50)
    cfg_scale: float = Field(7.0, ge=1.0, le=20.0)
    seed: int | None = Field(None, ge=0)

    strength: float = Field(
        0.15,
        ge=0.05,
        le=0.95,
        description="How much to change. 0.15 = ~85% same",
    )


class VariationRequest(BaseModel):
   
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: str | None = Field(None, max_length=1000)

    width: int = Field(512, ge=256, le=1024)
    height: int = Field(512, ge=256, le=1024)

    steps: int = Field(20, ge=1, le=50)
    cfg_scale: float = Field(7.0, ge=1.0, le=20.0)

    num_variations: int = Field(4, ge=2, le=8)


# ============ RESPONSE SCHEMAS ============
class GenerationOut(BaseModel):
    """Ek generation ka response."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    type: GenerationType
    prompt: str
    negative_prompt: str | None
    ref_image_path: str | None

    # Dimensions — actual
    width: int
    height: int

    # Dimensions — requested + adjustment info
    requested_width: int | None
    requested_height: int | None
    dimensions_adjusted: bool

    steps: int
    cfg_scale: float
    seed: int | None
    strength: float | None
    status: GenerationStatus
    output_path: str | None
    nsfw_score: float | None
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None


class GenerationListOut(BaseModel):
  
    total: int
    items: list[GenerationOut]


class VariationOut(BaseModel):
  
    parent_id: int
    variations: list[GenerationOut]