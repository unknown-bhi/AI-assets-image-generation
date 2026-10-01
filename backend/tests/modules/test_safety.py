"""Safety layer tests (Layer 1 + Layer 3)."""

import pytest
from PIL import Image

from app.config import settings
from app.modules.generation.prompt_filter import (
    PromptRejectedError,
    check_prompt,
)
from app.modules.generation.safety import check_image


# ============ LAYER 1: PROMPT FILTER ============
class TestPromptFilter:
    """Layer 1 prompt filter tests."""

    def test_safe_prompt_passes(self):
        """Safe prompts pass."""
        check_prompt("a cute cat on a windowsill")

    def test_blocked_keyword(self):
        """Keywords are blocked."""
        with pytest.raises(PromptRejectedError):
            check_prompt("nude woman")

    def test_empty_prompt_blocked(self):
        """Empty prompts are rejected."""
        with pytest.raises(PromptRejectedError):
            check_prompt("")

    def test_word_boundary(self):
        """Word boundaries prevent false positives."""
        # "class" contains "ass" but should pass
        check_prompt("a classy grass field")


# ============ LAYER 3: OUTPUT FILTER ============
class TestOutputFilter:
    """Layer 3 output filter tests."""

    def test_safe_solid_image(self):
        """Solid color image is safe."""
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))
        is_safe, score, label = check_image(img)
        assert is_safe is True
        assert score < 0.1
        assert label == "normal"

    def test_blocking_with_low_threshold(self):
        """Force low threshold to test blocking."""
        original = settings.NSFW_THRESHOLD
        settings.NSFW_THRESHOLD = 0.0001
        try:
            img = Image.new("RGB", (512, 512), color=(100, 150, 200))
            is_safe, score, label = check_image(img)
            assert is_safe is False
        finally:
            settings.NSFW_THRESHOLD = original

    def test_returns_score_and_label(self):
        """Check returns proper score and label."""
        img = Image.new("RGB", (512, 512), color=(50, 50, 50))
        is_safe, score, label = check_image(img)
        assert isinstance(is_safe, bool)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(label, str)