"""(Re)generate golden reference images for visual regression tests.

Run with: .venv/bin/python tests/visual_regression/generate_goldens.py
Only run this deliberately, when an effect's rendering intentionally changes
— it overwrites the images `test_goldens.py` compares against.
"""

from __future__ import annotations

import cv2

from tests.visual_regression.conftest import build_scene
from tests.visual_regression.test_goldens import GOLDEN_DIR, TRANSFORMS


def main() -> None:
    """Render each golden transform at seed 0 over the shared scene and write it, as lossless WebP.

    Run deliberately, when an effect's rendering is meant to change, and
    check the new images before committing them.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    img = build_scene()
    for name, transform in TRANSFORMS.items():
        transform.set_random_seed(0)
        out = transform(image=img)["image"]
        cv2.imwrite(str(GOLDEN_DIR / f"{name}.webp"), out, [cv2.IMWRITE_WEBP_QUALITY, 101])
        print(f"wrote {name}.webp")


if __name__ == "__main__":
    main()
