"""
Image utilities — auto-adjust dimensions, generate, save.

Handles filename generation, path building, and image saving.
"""

import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

from app.config import settings


# ============ AUTO-ADJUST DIMENSIONS ============
def auto_adjust_dimensions(width: int, height: int) -> tuple[int, int, bool]:
    """
    Auto-adjust dimensions to valid multiples of DIMENSION_MULTIPLE.

    Returns:
        (adjusted_width, adjusted_height, was_adjusted)
    """
    original = (width, height)
    multiple = settings.DIMENSION_MULTIPLE

    adjusted_width = round(width / multiple) * multiple
    adjusted_height = round(height / multiple) * multiple

    adjusted_width = max(
        settings.MIN_WIDTH,
        min(settings.MAX_WIDTH, adjusted_width),
    )
    adjusted_height = max(
        settings.MIN_HEIGHT,
        min(settings.MAX_HEIGHT, adjusted_height),
    )

    was_adjusted = (adjusted_width, adjusted_height) != original
    return adjusted_width, adjusted_height, was_adjusted


# ============ FILENAME GENERATORS ============
def generate_filename(extension: str = "png") -> str:
    """Generate a UUID-based filename (fallback)."""
    return f"{uuid.uuid4()}.{extension}"


def generate_readable_filename(
    seed: int,
    prefix: str = "img",
    extension: str = "png",
) -> str:
    """
    Generate a readable filename.

    Format: {prefix}_{timestamp}_{seed}.{ext}

    Example:
        prefix="cat", seed=230926787
        → "cat_20261001_110452_230926787.png"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{seed}.{extension}"


# ============ PATHS ============
def get_output_path(filename: str, subdir: str = "generated") -> Path:
    """Build output path (creates folder if needed)."""
    base = settings.STORAGE_DIR / subdir
    base.mkdir(parents=True, exist_ok=True)
    return base / filename


# ============ DUMMY IMAGE (TESTING) ============
def generate_dummy_image(
    width: int,
    height: int,
    color: tuple[int, int, int] = (100, 150, 200),
    text: str | None = None,
) -> Image.Image:
    """Generate a solid-color test image."""
    image = Image.new("RGB", (width, height), color=color)

    if text:
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), text[:50], fill=(255, 255, 255))

    return image


# ============ SAVE IMAGE ============
def save_image(
    image: Image.Image,
    filename: str | None = None,
    subdir: str = "generated",
    prefix: str = "img",
    seed: int | None = None,
) -> str:
    """
    Save image to storage.

    Args:
        image: PIL Image.
        filename: Optional explicit filename.
        subdir: 'generated' | 'rejected' | 'uploads'.
        prefix: Filename prefix (e.g., 'cat', 'txt2img').
        seed: Generation seed (used in filename if given).

    Returns:
        Relative path for DB storage.

    Filename priority:
        1. Explicit `filename` argument
        2. `{prefix}_{timestamp}_{seed}.png` if seed given
        3. UUID fallback
    """
    if filename is None:
        if seed is not None:
            filename = generate_readable_filename(
                seed=seed,
                prefix=prefix,
            )
        else:
            filename = generate_filename()

    full_path = get_output_path(filename, subdir)
    image.save(full_path, format="PNG", optimize=True)

    return f"{subdir}/{filename}"