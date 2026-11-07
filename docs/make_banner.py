"""Draw the repository's social banner from the gallery's own images.

Run with::

    python docs/make_banner.py

It writes ``docs/assets/social_banner.png``, 1280 by 640 pixels, which is the
size GitHub asks for a repository's social preview (Settings, General,
Social preview) and is also the picture at the top of the README.

The banner is one photograph cut into slanted strips, each strip taken from
a different effect's cell of the severity gallery in ``docs/examples``.
Every cell is a render of the same reference photograph, so the road runs
on unbroken from strip to strip while the weather, the lens and the camera
change across it: which is what the library does, shown once.

Nothing here is rendered afresh. Regenerate the gallery first
(``python docs/generate_examples.py``) if the effects have changed, then
run this. It needs Pillow for the lettering, which the gallery does not;
the typeface is DejaVu Sans, found where Linux distributions install it,
with Pillow's built-in face as the fallback.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DOCS = Path(__file__).resolve().parent
EXAMPLES = DOCS / "examples"
OUT = DOCS / "assets" / "social_banner.png"

WIDTH, HEIGHT = 1280, 640
# How far the foot of each strip is to the left of its head, in pixels.
SLANT = 90
# Each strip: the gallery cell it is cut from, and what it is called.
STRIPS = [
    ("00_base_scene", "clean"),
    ("rain_4", "rain"),
    ("ground_fog_2", "ground fog"),
    ("snow_3", "snow"),
    ("frost_on_glass_3", "frost"),
    ("foliage_intrusion_3", "foliage"),
    ("night_ir_1", "night IR"),
    ("spider_web_3", "web"),
    ("night_insects_3", "insects"),
    ("analog_video_4", "analog link"),
]
TITLE = "surmentations"
SUBTITLE = "Physically grounded weather, lens and camera augmentations"
FACTS = "37 transforms for Albumentations  ·  depth-aware  ·  linear light  ·  optional C kernels"

_FONT_DIRS = (Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/dejavu"))


def _font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """DejaVu at a size, or Pillow's own face where DejaVu is not installed."""
    for folder in _FONT_DIRS:
        path = folder / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def _cell(name: str) -> np.ndarray:
    """A gallery cell, enlarged to the banner's width and cropped to its height."""
    image = cv2.imread(str(EXAMPLES / f"{name}.png"))
    if image is None:
        raise FileNotFoundError(f"{name}.png is not in {EXAMPLES}; regenerate the gallery first")
    scale = WIDTH / image.shape[1]
    image = cv2.resize(
        image, (WIDTH, round(image.shape[0] * scale)), interpolation=cv2.INTER_LANCZOS4
    )
    # A little above the middle: the horizon and the road both in the frame.
    top = max(0, min(image.shape[0] - HEIGHT, round(0.42 * (image.shape[0] - HEIGHT))))
    return cv2.cvtColor(image[top : top + HEIGHT], cv2.COLOR_BGR2RGB)


def _strips() -> tuple[np.ndarray, list[float]]:
    """The banner's picture, and where along its top each strip's label goes.

    The strips divide the top edge evenly and lean, their feet ``SLANT`` to
    the left of their heads; the last one's right side is the frame's own,
    so that no corner is left empty.
    """
    banner = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    step = WIDTH / len(STRIPS)
    label_at = []
    for index, (name, _label) in enumerate(STRIPS):
        left, right = index * step, (index + 1) * step
        last = index == len(STRIPS) - 1
        outline = np.array(
            [
                [left, 0],
                [right, 0],
                [right if last else right - SLANT, HEIGHT],
                [left - SLANT, HEIGHT],
            ]
        )
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(outline).astype(np.int32), 255, cv2.LINE_AA)
        share = (mask.astype(np.float32) / 255.0)[..., None]
        banner = (banner * (1.0 - share) + _cell(name) * share).astype(np.uint8)
        label_at.append(0.5 * (left + right) - 0.04 * SLANT)
    # The cuts between strips, as thin pale lines.
    for index in range(1, len(STRIPS)):
        x = index * step
        cv2.line(
            banner, (round(x), 0), (round(x - SLANT), HEIGHT), (245, 245, 245), 2, cv2.LINE_AA
        )
    return banner, label_at


def main() -> None:
    """Cut the strips, darken the foot of the picture, and letter it."""
    picture, label_at = _strips()
    # A shade rising from the bottom, so that the lettering reads over
    # whatever is behind it.
    rise = np.clip((np.arange(HEIGHT) - 0.50 * HEIGHT) / (0.5 * HEIGHT), 0.0, 1.0) ** 1.3
    picture = (picture * (1.0 - 0.9 * rise)[:, None, None]).astype(np.uint8)

    image = Image.fromarray(picture)
    draw = ImageDraw.Draw(image, "RGBA")
    small = _font("DejaVuSansMono-Bold.ttf", 16)
    for (_name, label), x in zip(STRIPS, label_at, strict=True):
        box = draw.textbbox((0, 0), label, font=small)
        width, height = box[2] - box[0], box[3] - box[1]
        left = min(max(8.0, x - width / 2.0), WIDTH - width - 8.0)
        draw.rounded_rectangle(
            (left - 7, 14, left + width + 7, 14 + height + 12), radius=6, fill=(0, 0, 0, 150)
        )
        draw.text((left, 18), label, font=small, fill=(255, 255, 255, 255))

    draw.text((56, 418), TITLE, font=_font("DejaVuSans-Bold.ttf", 92), fill=(255, 255, 255, 255))
    draw.text((60, 532), SUBTITLE, font=_font("DejaVuSans.ttf", 31), fill=(235, 238, 242, 255))
    draw.text((60, 582), FACTS, font=_font("DejaVuSans.ttf", 20), fill=(170, 200, 230, 255))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, optimize=True)
    print(f"wrote {OUT.relative_to(DOCS.parent)}, {image.width} by {image.height}")


if __name__ == "__main__":
    main()
