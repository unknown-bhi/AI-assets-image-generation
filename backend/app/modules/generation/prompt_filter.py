
import re
import threading
from typing import Any

from app.config import settings


# ============ BLOCKED KEYWORDS (Layer 1a) ============
BLOCKED_KEYWORDS: set[str] = {
    # --- NSFW / Sexual ---
    "nude", "nudes", "nudity", "naked",
    "porn", "porno", "pornographic", "pornography",
    "sex", "sexual", "sexy", "erotic", "erotica",
    "hentai", "nsfw", "xxx", "nsfl", "bikini",
    "genitalia", "genitals", "penis", "vagina", "breasts",
    "orgasm", "masturbation", "intercourse",
    "bdsm", "fetish",
    "lingerie", "striptease",

    # --- Violence / Gore ---
    "gore", "gory", "bloodbath",
    "torture", "mutilation", "mutilated", "dismember",
    "dismemberment", "decapitation", "beheading",
    "massacre", "slaughter", "carnage",
    "murder", "assassination",
    "corpse", "cadaver",

    # --- Illegal ---
    "bestiality", "zoophilia",
    "rape", "rapist",
    "childporn", "child porn",

    # --- Hate ---
    "nigger", "nigga", "kike", "chink", "faggot", "tranny",
}


# ============ EXCEPTIONS ============
class PromptRejectedError(ValueError):
    """Raised when a prompt is rejected by the filter."""

    def __init__(self, message: str, reason: str, matched_keyword: str = ""):
        super().__init__(message)
        self.reason = reason
        self.matched_keyword = matched_keyword


# ============ GLOBAL STATE (classifier) ============
_classifier: Any = None
_tokenizer: Any = None
_classifier_lock = threading.Lock()


# ============ LOAD CLASSIFIER (Layer 1b) ============
def _load_classifier() -> tuple[Any, Any]:
    """
    Load the toxicity classifier (once).

    Returns:
        Tuple of (tokenizer, model).
    """
    global _classifier, _tokenizer

    if _classifier is not None and _tokenizer is not None:
        return _tokenizer, _classifier

    with _classifier_lock:
        if _classifier is not None and _tokenizer is not None:
            return _tokenizer, _classifier

        print(
            f"Loading prompt classifier from "
            f"{settings.PROMPT_CLASSIFIER_MODEL_ID}..."
        )

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(
            settings.PROMPT_CLASSIFIER_MODEL_ID,
            cache_dir=str(settings.MODEL_CACHE_DIR),
        )
        _classifier = AutoModelForSequenceClassification.from_pretrained(
            settings.PROMPT_CLASSIFIER_MODEL_ID,
            cache_dir=str(settings.MODEL_CACHE_DIR),
        )
        _classifier.eval()

        print("Prompt classifier loaded")

    return _tokenizer, _classifier


# ============ LAYER 1a: KEYWORD CHECK ============
def _check_keywords(prompt: str) -> None:
    """
    Check prompt against keyword blocklist.

    Raises:
        PromptRejectedError: If blocked keyword is found.
    """
    prompt_lower = prompt.lower()

    for keyword in BLOCKED_KEYWORDS:
        if " " in keyword:
            if keyword in prompt_lower:
                raise PromptRejectedError(
                    "Prompt contains blocked content. Please rephrase.",
                    reason="keyword",
                    matched_keyword=keyword,
                )
        else:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, prompt_lower):
                raise PromptRejectedError(
                    "Prompt contains blocked content. Please rephrase.",
                    reason="keyword",
                    matched_keyword=keyword,
                )


# ============ LAYER 1b: CLASSIFIER CHECK ============
def _check_classifier(prompt: str) -> None:
    """
    Check prompt using toxicity classifier.

    Raises:
        PromptRejectedError: If toxicity score exceeds threshold.
    """
    import torch

    tokenizer, model = _load_classifier()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with _classifier_lock:
        with torch.no_grad():
            outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs = probs.cpu().numpy()[0]

    # Label mapping from the model config
    labels = model.config.id2label
    label_probs = {labels[i]: float(probs[i]) for i in range(len(probs))}

    # Highest toxic category score (exclude 'toxic' aggregate if present)
    # For unitary/toxic-bert, labels are: toxic, severe_toxic, obscene, threat, insult, identity_hate
    toxic_categories = [
        "toxic", "severe_toxic", "obscene",
        "threat", "insult", "identity_hate",
    ]
    max_toxic_score = max(
        label_probs.get(cat, 0.0) for cat in toxic_categories
    )

    if max_toxic_score >= settings.PROMPT_TOXICITY_THRESHOLD:
        raise PromptRejectedError(
            "Prompt flagged by content classifier. Please rephrase.",
            reason="classifier",
            matched_keyword="",
        )


# ============ PUBLIC: CHECK PROMPT ============
def check_prompt(prompt: str) -> None:
    """
    Check if a prompt is safe (Layer 1a + Layer 1b).

    Args:
        prompt: User's text prompt.

    Raises:
        PromptRejectedError: If the prompt contains blocked content.

    Notes:
        - Empty prompts are rejected.
        - Layer 1a (keywords) runs first for fast rejection.
        - Layer 1b (classifier) runs if enabled and keywords pass.
    """
    if not prompt or not prompt.strip():
        raise PromptRejectedError(
            "Prompt cannot be empty",
            reason="empty",
        )

    # Layer 1a: keywords
    _check_keywords(prompt)

    # Layer 1b: classifier (optional)
    if settings.PROMPT_CLASSIFIER_ENABLED:
        _check_classifier(prompt)


def is_safe_prompt(prompt: str) -> bool:
    """
    Convenience wrapper: return True if prompt is safe.

    Args:
        prompt: User's text prompt.

    Returns:
        True if safe, False otherwise.
    """
    try:
        check_prompt(prompt)
        return True
    except PromptRejectedError:
        return False