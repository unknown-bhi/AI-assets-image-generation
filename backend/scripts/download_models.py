"""
SD 1.5 model download script.

Ek dafa chalao — model cache ho jayega.

Usage:
    cd backend
    python scripts/download_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings


def main():
    print("=" * 60)
    print("📦 Downloading SD 1.5 model")
    print("=" * 60)
    print(f"Model ID: {settings.MODEL_ID}")
    print(f"Cache dir: {settings.MODEL_CACHE_DIR}")
    print()

    settings.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Main SD 1.5 pipeline
    print("▶️  Downloading Stable Diffusion 1.5 pipeline...")
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        settings.MODEL_ID,
        cache_dir=str(settings.MODEL_CACHE_DIR),
        safety_checker=None,
    )
    print("✅ SD 1.5 pipeline downloaded")

    # 2. NSFW classifier (Falconsai)
    print("\n▶️  Downloading NSFW classifier (Falconsai)...")
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    nsfw_model_id = "Falconsai/nsfw_image_detection"

    processor = AutoImageProcessor.from_pretrained(
        nsfw_model_id,
        cache_dir=str(settings.MODEL_CACHE_DIR),
    )
    model = AutoModelForImageClassification.from_pretrained(
        nsfw_model_id,
        cache_dir=str(settings.MODEL_CACHE_DIR),
    )
    print("✅ NSFW classifier downloaded")

    print("\n" + "=" * 60)
    print("🎉 All models downloaded successfully!")
    print(f"📁 Cached at: {settings.MODEL_CACHE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()