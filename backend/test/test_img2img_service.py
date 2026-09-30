"""
Test the img2img service directly (no API).
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))


from PIL import Image
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.modules.auth.model import User
from app.modules.generation.model import Generation
from app.modules.generation.schema import Img2ImgRequest
from app.modules.generation.service import create_img2img_generation


async def main() -> None:
    """Test img2img service."""
    print("=" * 60)
    print("Testing img2img service")
    print("=" * 60)

    async with AsyncSessionLocal() as db:
        # 1. Get a user
        result = await db.execute(select(User).limit(1))
        user = result.scalar_one_or_none()

        if user is None:
            print("No user found. Register one first.")
            return

        print(f"User: {user.username} (id={user.id})")

        # 2. Load reference image
        ref_path = Path("test_output.png")
        if not ref_path.exists():
            print(f"Reference not found: {ref_path}")
            return

        reference = Image.open(ref_path).convert("RGB")
        print(f"Reference: {ref_path} ({reference.size})")

        # 3. Build request
        data = Img2ImgRequest(
            prompt="a beautiful sunset over mountains, oil painting",
            negative_prompt="blurry, low quality",
            width=256,
            height=256,
            steps=10,
            cfg_scale=7.0,
            strength=0.15,
        )

        print(f"\nRequest:")
        print(f"  Prompt: {data.prompt}")
        print(f"  Size: {data.width}x{data.height}")
        print(f"  Strength: {data.strength}")
        print(f"  Steps: {data.steps}")

        # 4. Generate
        print(f"\nGenerating...")
        generation = await create_img2img_generation(
            db, user, data, reference
        )

        print(f"\nResult:")
        print(f"  ID: {generation.id}")
        print(f"  Type: {generation.type.value}")
        print(f"  Status: {generation.status.value}")
        print(f"  Seed: {generation.seed}")
        print(f"  Strength: {generation.strength}")
        print(f"  NSFW score: {generation.nsfw_score}")
        print(f"  Output: {generation.output_path}")
        print(f"  Error: {generation.error_message}")

    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())