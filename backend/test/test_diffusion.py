"""
Test multiple generations in a single process.

Verifies that the model loads only once.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

import time

from app.modules.generation.diffusion import generate_image


def main() -> None:
    """Generate multiple images and measure times."""
    prompts = [
        "a red apple on a wooden table",
        "a blue ocean with sailboat",
        "a yellow sunflower in a field",
    ]

    print("=" * 60)
    print("Testing multiple generations")
    print("=" * 60)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: {prompt}")
        start = time.time()

        image, seed = generate_image(
            prompt=prompt,
            width=512,
            height=512,
            steps=50,
            
        )

        elapsed = time.time() - start
        output = f"new_{i}.png"
        image.save(output)
        print(f"\n  Done in {elapsed:.1f}s (seed={seed})")
        print(f"  Saved: {output}")

    print("\n" + "=" * 60)
    print("All generations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()