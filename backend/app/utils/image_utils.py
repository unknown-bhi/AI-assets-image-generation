
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

from app.config import settings


# ============ AUTO-ADJUST DIMENSIONS ============
def auto_adjust_dimensions(width: int, height: int) -> tuple[int, int, bool]:

    original = (width, height)

    multiple = settings.DIMENSION_MULTIPLE

    # Round to nearest multiple (e.g., 8)
    adjusted_width = round(width / multiple) * multiple
    adjusted_height = round(height / multiple) * multiple

    # Clamp to valid range
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



def generate_filename(extension: str = "png") -> str:

    return f"{uuid.uuid4()}.{extension}"


def get_output_path(filename: str, subdir: str = "generated") -> Path:
 
    base = settings.STORAGE_DIR / subdir
    base.mkdir(parents=True, exist_ok=True)
    return base / filename


def generate_dummy_image(
    width: int,
    height: int,
    color: tuple[int, int, int] = (100, 150, 200),
    text: str | None = None,
) -> Image.Image:

    image = Image.new("RGB", (width, height), color=color)

    if text:
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), text[:50], fill=(255, 255, 255))

    return image



def save_image(
    image: Image.Image,
    filename: str | None = None,
    subdir: str = "generated",
) -> str:

    if filename is None:
        filename = generate_filename()

    full_path = get_output_path(filename, subdir)
    image.save(full_path, format="PNG", optimize=True)

    return f"{subdir}/{filename}"