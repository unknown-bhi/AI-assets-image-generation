

import threading
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.config import settings



_processor: Any = None
_model: Any = None
_lock = threading.Lock()


# ============ LOAD ============
def load_classifier() -> tuple[Any, Any]:
    """
    Load the NSFW classifier (once).

    Returns:
        Tuple of (processor, model).

    Notes:
        - Uses lazy loading: called on first safety check.
        - Thread-safe: uses a lock to prevent concurrent loads.
    """
    global _processor, _model

    if _processor is not None and _model is not None:
        return _processor, _model

    with _lock:
        if _processor is not None and _model is not None:
            return _processor, _model

        print(f"Loading NSFW classifier from {settings.NSFW_MODEL_ID}...")

        _processor = AutoImageProcessor.from_pretrained(
            settings.NSFW_MODEL_ID,
            cache_dir=str(settings.MODEL_CACHE_DIR),
        )
        _model = AutoModelForImageClassification.from_pretrained(
            settings.NSFW_MODEL_ID,
            cache_dir=str(settings.MODEL_CACHE_DIR),
        )
        _model.eval()

        print("NSFW classifier loaded")

    return _processor, _model


def get_classifier() -> tuple[Any, Any]:
    """
    Return the loaded classifier, loading it if necessary.

    Returns:
        Tuple of (processor, model).
    """
    if _processor is None or _model is None:
        return load_classifier()
    return _processor, _model


# ============ CHECK ============
def check_image(image: Image.Image) -> tuple[bool, float, str]:
    """
    Check if an image is NSFW-safe.

    Args:
        image: PIL Image to check.

    Returns:
        Tuple of (is_safe, nsfw_score, label).

    Notes:
        - is_safe: True if image passes the threshold.
        - nsfw_score: Probability of the image being NSFW (0.0-1.0).
        - label: Detected label ('normal', 'nsfw', 'drawing', 'hentai', 'porn').
    """
    processor, model = get_classifier()

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")

    # Run inference
    with _lock:
        with torch.no_grad():
            outputs = model(**inputs)

    # Get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs = probs.cpu().numpy()[0]

    # Map labels to probabilities
    labels = model.config.id2label
    label_probs = {labels[i]: float(probs[i]) for i in range(len(probs))}

    # Highest scoring label
    top_label = max(label_probs, key=label_probs.get)
    top_score = label_probs[top_label]

    # NSFW score = probability of any non-normal category
    nsfw_score = 1.0 - label_probs.get("normal", 0.0)

    is_safe = nsfw_score < settings.NSFW_THRESHOLD

    return is_safe, nsfw_score, top_label


def is_safe(image: Image.Image) -> bool:
    """
    Convenience wrapper: return True if the image is safe.

    Args:
        image: PIL Image to check.

    Returns:
        True if safe, False otherwise.
    """
    safe, _, _ = check_image(image)
    return safe