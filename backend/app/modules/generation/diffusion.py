"""
Stable Diffusion 1.5 pipeline wrapper.

Loads the model once (singleton pattern) and reuses it for all generations.

CPU optimizations applied:
- Attention slicing (reduces RAM usage)
- VAE slicing (handles larger images)
- float32 precision (CPU compatible)
"""

import threading
from typing import Any

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

from app.config import settings


# ============ GLOBAL STATE ============
_pipeline: StableDiffusionPipeline | None = None
_img2img_pipeline: StableDiffusionImg2ImgPipeline | None = None
_lock = threading.Lock()

# ============ CONSTANTS ============
MAX_SEED = 2**31 - 1   # PostgreSQL INTEGER max (2^31 - 1)

# ============ LOAD TXT2IMG ============
def load_pipeline() -> StableDiffusionPipeline:
    """Load the SD 1.5 text-to-image pipeline (once)."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    with _lock:
        if _pipeline is not None:
            return _pipeline

        print(f"Loading SD 1.5 pipeline from {settings.MODEL_CACHE_DIR}...")

        pipe = StableDiffusionPipeline.from_pretrained(
            settings.MODEL_ID,
            cache_dir=str(settings.MODEL_CACHE_DIR),
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        pipe.enable_attention_slicing()

        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

        pipe = pipe.to("cpu")
        pipe.set_progress_bar_config(disable=True)

        _pipeline = pipe
        print("SD 1.5 pipeline loaded")

    return _pipeline


def get_pipeline() -> StableDiffusionPipeline:
    """Return the loaded text-to-image pipeline."""
    if _pipeline is None:
        return load_pipeline()
    return _pipeline


# ============ LOAD IMG2IMG ============
def load_img2img_pipeline() -> StableDiffusionImg2ImgPipeline:
    """Load the SD 1.5 img2img pipeline (once, reuses components)."""
    global _img2img_pipeline

    if _img2img_pipeline is not None:
        return _img2img_pipeline
    main_pipe = get_pipeline()

    with _lock:
        if _img2img_pipeline is not None:
            return _img2img_pipeline

        print("Loading SD 1.5 img2img pipeline...")

        

        _img2img_pipeline = StableDiffusionImg2ImgPipeline(
            vae=main_pipe.vae,
            text_encoder=main_pipe.text_encoder,
            tokenizer=main_pipe.tokenizer,
            unet=main_pipe.unet,
            scheduler=main_pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        _img2img_pipeline.enable_attention_slicing()
        _img2img_pipeline = _img2img_pipeline.to("cpu")
        _img2img_pipeline.set_progress_bar_config(disable=True)

        print("SD 1.5 img2img pipeline loaded")

    return _img2img_pipeline


def get_img2img_pipeline() -> StableDiffusionImg2ImgPipeline:
    """Return the loaded img2img pipeline."""
    if _img2img_pipeline is None:
        return load_img2img_pipeline()
    return _img2img_pipeline


# ============ UNLOAD ============
def unload_pipeline() -> None:
    """Unload all pipelines and free memory."""
    global _pipeline, _img2img_pipeline
    with _lock:
        _pipeline = None
        _img2img_pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("SD 1.5 pipelines unloaded")


# ============ GENERATE TXT2IMG ============
def generate_image(
    prompt: str,
    negative_prompt: str | None = None,
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg_scale: float = 7.0,
    seed: int | None = None,
    show_progress: bool = True,
) -> tuple[Any, int]:
    """Generate an image from a text prompt."""
    pipe = get_pipeline()

    if seed is None:
        seed = torch.randint(0, MAX_SEED, (1,)).item()

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def progress_callback(step: int, timestep: int, latents: Any) -> None:
        if not show_progress:
            return
        percent = (step + 1) / steps * 100
        bar_length = 30
        filled = int(bar_length * (step + 1) / steps)
        bar = "=" * filled + "-" * (bar_length - filled)
        print(f"\r  [{bar}] {percent:5.1f}% ({step + 1}/{steps})", end="", flush=True)

    with _lock:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
            callback=progress_callback if show_progress else None,
            callback_steps=1,
        )

    if show_progress:
        print()

    return result.images[0], seed


# ============ GENERATE IMG2IMG ============
def generate_img2img(
    prompt: str,
    init_image: Any,
    negative_prompt: str | None = None,
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    cfg_scale: float = 7.0,
    strength: float = 0.15,
    seed: int | None = None,
    show_progress: bool = True,
) -> tuple[Any, int]:
    """
    Generate an image from a reference image and text prompt.

    Args:
        prompt: Text description.
        init_image: Reference PIL Image.
        negative_prompt: Things to avoid.
        width, height: Output dimensions.
        steps: Inference steps.
        cfg_scale: Classifier-free guidance scale.
        strength: How much to change (0.0-1.0). 0.15 = ~85% similar.
        seed: Random seed.
        show_progress: Whether to print progress.

    Returns:
        Tuple of (PIL.Image, seed_used).
    """
    pipe = get_img2img_pipeline()

    init_image = init_image.convert("RGB").resize((width, height))

    if seed is None:
        seed = torch.randint(0, MAX_SEED, (1,)).item()

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def progress_callback(step: int, timestep: int, latents: Any) -> None:
        if not show_progress:
            return
        percent = (step + 1) / steps * 100
        bar_length = 30
        filled = int(bar_length * (step + 1) / steps)
        bar = "=" * filled + "-" * (bar_length - filled)
        print(f"\r  [{bar}] {percent:5.1f}% ({step + 1}/{steps})", end="", flush=True)

    with _lock:
        result = pipe(
            prompt=prompt,
            image=init_image,
            negative_prompt=negative_prompt or "",
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            strength=strength,
            generator=generator,
            callback=progress_callback if show_progress else None,
            callback_steps=1,
        )

    if show_progress:
        print()

    return result.images[0], seed