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

    SD_CLI_PATH: Path = Path(
        "/home/bitech/Desktop/kashmiry/AI_asseet_creator/stable-diffusion.cpp/build/bin/sd-cli"
    )
    QWEN_MODEL_PATH: Path = Path(
        "/home/bitech/Desktop/kashmiry/AI_asseet_creator/qwen_models/qwen_image_2.1_turbo_Q4_K_M.gguf"
    )
    QWEN_VAE_PATH: Path = Path(
        "/home/bitech/Desktop/kashmiry/AI_asseet_creator/qwen_models/qwen_image_2.1_vae_bf16.safetensors"
    )
    QWEN_LLM_PATH: Path = Path(
        "/home/bitech/Desktop/kashmiry/AI_asseet_creator/qwen_models/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"
    )

    # Cores
    QWEN_CORES: str = "0-7"  # 8 cores

    # Output
    OUTPUT_DIR: Path = Path("/tmp/qwen_outputs")

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

    NSFW_THRESHOLD: float = 0.7  
    NSFW_MODEL_ID: str = "Falconsai/nsfw_image_detection"

       # Prompt filter (Layer 1)
    PROMPT_CLASSIFIER_ENABLED: bool = True
    PROMPT_CLASSIFIER_MODEL_ID: str = "unitary/toxic-bert"
    PROMPT_TOXICITY_THRESHOLD: float = 0.5


settings = Settings()