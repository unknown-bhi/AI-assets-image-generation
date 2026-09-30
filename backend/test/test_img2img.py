"""
Test img2img generation with a reference image.
"""

import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))


from PIL import Image

from app.modules.generation.diffusion import generate_img2img


def main() -> None:
    """Test img2img with a reference image."""
    print("=" * 60)
    print("Testing img2img generation")
    print("=" * 60)

    # Find reference image
    ref_path = Path("test_output.png")
    if not ref_path.exists():
        print(f"Reference not found: {ref_path}")
        print("Looking for alternatives...")
        candidates = list(Path(".").glob("*.png"))
        if candidates:
            ref_path = candidates[0]
            print(f"Using: {ref_path}")
        else:
            print("No PNG files found. Run test_diffusion.py first.")
            return

    ref_image = Image.open(ref_path).convert("RGB")
    print(f"Reference: {ref_path} ({ref_image.size})")

    # Generate variation
    print("\nGenerating variation (strength=0.15)...")
    start = time.time()

    image, seed = generate_img2img(
        prompt="a beautiful sunset over mountains, oil painting",
        init_image=ref_image,
        width=256,
        height=256,
        steps=15,
        cfg_scale=7.0,
        strength=0.15,
        show_progress=True,
    )

    elapsed = time.time() - start
    print(f"\nGenerated in {elapsed:.1f}s (seed={seed})")

    output_path = Path("test_img2img_output.png")
    image.save(output_path)
    print(f"Saved: {output_path.absolute()}")


if __name__ == "__main__":
    main()