"""
Qwen-Image 2.1 Turbo engine — sd-cli wrapper.

Ye Python se sd-cli (C++ binary) ko call karta hai.
"""
import subprocess
import uuid
from pathlib import Path

from PIL import Image

from app.config import settings


class QwenEngine:
    """Qwen-Image 2.1 Turbo engine (sd-cli wrapper)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 10,
        cfg_scale: float = 1.0,
        seed: int | None = None,
        output_path: str | None = None,
        timeout: int = 1800,
    ) -> str:
        """Image generate karo (sd-cli call)."""
        # Output path
        if output_path is None:
            output_path = str(settings.OUTPUT_DIR / f"qwen_{uuid.uuid4().hex}.png")

        # Command
        cmd = [
            "taskset", "-c", settings.QWEN_CORES,
            str(settings.SD_CLI_PATH),
            "--diffusion-model", str(settings.QWEN_MODEL_PATH),
            "--vae", str(settings.QWEN_VAE_PATH),
            "--llm", str(settings.QWEN_LLM_PATH),
            "-p", prompt,
            "--steps", str(steps),
            "--cfg-scale", str(cfg_scale),
            "--sampling-method", "euler",
            "-W", str(width),
            "-H", str(height),
            "--mmap",
            "-o", output_path,
        ]

        if negative_prompt:
            cmd.extend(["-n", negative_prompt])

        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        # Run
        print(f"Running: {' '.join(cmd[:8])}...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"sd-cli failed (code {result.returncode}):\n{result.stderr[-2000:]}"
            )

        # Verify output
        if not Path(output_path).exists():
            raise RuntimeError(f"Output file not created: {output_path}")

        return output_path


# Singleton instance
engine = QwenEngine()