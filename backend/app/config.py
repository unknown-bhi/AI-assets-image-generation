from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR =Path(__file__).resolve().parent.parent

class Settings(BaseSettings):


  
    APP_NAME: str = "AI Asset Creator"
    DEBUG: bool = True
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # ============ SECURITY ============
    SECRET_KEY: str = Field(..., min_length=16)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ============ DATABASE ============
    DATABASE_URL: str

    # ============ REDIS ============
    REDIS_URL: str = "redis://localhost:6379/0"

    # ============ ML MODEL ============
    MODEL_ID: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    MODEL_CACHE_DIR: Path = BASE_DIR / "models"
    SAFETY_CHECKER: bool = True

    # ============ STORAGE ============
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOAD_DIR: Path = BASE_DIR / "storage" / "uploads"
    GENERATED_DIR: Path = BASE_DIR / "storage" / "generated"
    REJECTED_DIR: Path = BASE_DIR / "storage" / "rejected"

    # ============ GENERATION LIMITS ============
    DAILY_QUOTA: int = 50
    MAX_WIDTH: int = 1024
    MAX_HEIGHT: int = 1024
    MIN_WIDTH: int = 256
    MIN_HEIGHT: int = 256
    DIMENSION_MULTIPLE: int = 8

    # Pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    def ensure_dirs(self) -> None:
        """Create all storage directories if they don't exist."""
        for path in (
            self.STORAGE_DIR,
            self.UPLOAD_DIR,
            self.GENERATED_DIR,
            self.REJECTED_DIR,
            self.MODEL_CACHE_DIR,
        ):
            path.mkdir(parents=True, exist_ok=True)



settings = Settings()