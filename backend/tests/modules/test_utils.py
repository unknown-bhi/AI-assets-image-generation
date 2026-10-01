"""Utility function tests."""

import pytest
from PIL import Image

from app.modules.generation.safety import check_image
from app.utils.image_utils import (
    auto_adjust_dimensions,
    generate_filename,
    generate_readable_filename,
)


# ============ DIMENSION ADJUSTMENT ============
class TestAutoAdjust:
    """Auto-adjust dimension tests."""

    def test_exact_multiple(self):
        """Multiple of 8 stays same."""
        w, h, adjusted = auto_adjust_dimensions(512, 512)
        assert (w, h) == (512, 512)
        assert adjusted is False

    def test_rounds_up(self):
        """510 rounds to 512."""
        w, h, adjusted = auto_adjust_dimensions(510, 510)
        assert w % 8 == 0
        assert h % 8 == 0
        assert adjusted is True

    def test_clamps_min(self):
        """Below min clamped."""
        w, h, _ = auto_adjust_dimensions(100, 100)
        assert w >= 256
        assert h >= 256

    def test_clamps_max(self):
        """Above max clamped."""
        w, h, _ = auto_adjust_dimensions(5000, 5000)
        assert w <= 1024
        assert h <= 1024


# ============ FILENAME ============
class TestFilenames:
    """Filename generator tests."""

    def test_uuid_filename(self):
        """UUID filename has .png extension."""
        fname = generate_filename()
        assert fname.endswith(".png")

    def test_readable_filename(self):
        """Readable filename has prefix and seed."""
        fname = generate_readable_filename(seed=12345, prefix="txt2img")
        assert fname.startswith("txt2img_")
        assert "12345" in fname
        assert fname.endswith(".png")