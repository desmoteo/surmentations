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

One strip is rendered here and not taken from the gallery: packet loss.
A gallery cell shows a whole frame, and where the damage falls in it is
the draw's business; in the cells that exist it is either outside the
strip or a block of solid green that shouts over the rest of the banner.
So that strip is one fixed draw (:data:`PACKET_LOSS`) of a moderate loss,
concealed by the decoder, whose smear lies across the part of the road the
strip shows. It needs what ``TransmissionFaults`` needs, ffmpeg or PyAV.

Regenerate the gallery first (``python docs/generate_examples.py``) if the
effects have changed, then run this. It needs Pillow for the lettering,
which the gallery does not;
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
# The name of the strip that is rendered here, and the draw it is: the
# arguments of ``TransmissionFaults`` and the seed. A fifth of the packets
# lost a few frames after a keyframe, every loss concealed from the frame
# before and none left as the green of an unfinished frame.
RENDERED = "packet loss, rendered"
PACKET_LOSS: dict[str, object] = {
    "packet_loss_range": (0.12, 0.2),
    "scene_motion_range": (0.03, 0.05),
    "missing_reference_probability": 0.0,
    "concealment_probability": 1.0,
    "slices_range": (4, 6),
    "frames_since_keyframe_range": (2, 4),
}
PACKET_LOSS_SEED = 13
# Each strip: the gallery cell it is cut from, and what it is called.
STRIPS = [
    ("00_base_scene", "clean"),
    ("rain_4", "rain"),
    ("ground_fog_2", "ground fog"),
    ("snow_3", "snow"),
    (RENDERED, "packet loss"),
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


def _packet_loss() -> np.ndarray:
    """The reference photograph after the banner's one draw of packet loss, as BGR."""
    # Imported here: the rest of the banner needs neither the library nor a codec.
    from surmentations.camera import TransmissionFaults

    base = cv2.imread(str(EXAMPLES / "00_base_scene.png"))
    if base is None:
        raise FileNotFoundError(f"00_base_scene.png is not in {EXAMPLES}")
    faults = TransmissionFaults(p=1.0, **PACKET_LOSS)
    faults.set_random_seed(PACKET_LOSS_SEED)
    lossy: np.ndarray = faults(image=cv2.cvtColor(base, cv2.COLOR_BGR2RGB))["image"]
    return cv2.cvtColor(lossy, cv2.COLOR_RGB2BGR)


def _cell(name: str) -> np.ndarray:
    """A gallery cell, enlarged to the banner's width and cropped to its height.

    Or, for :data:`RENDERED`, the draw of packet loss made here.
    """
    image = _packet_loss() if name == RENDERED else cv2.imread(str(EXAMPLES / f"{name}.png"))
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
