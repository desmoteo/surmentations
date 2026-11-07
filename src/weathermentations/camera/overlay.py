"""What a camera or a recorder writes over the picture.

Not physics
-----------
Nearly every frame of surveillance footage has something drawn on it that
was never in front of the lens: the date and time, the camera's name, and
on many sites rectangles blanked out for privacy. None of it is weather or
optics and nothing here is derived from anything. It is in this library
because it is in the footage: it sits in the corners, where objects enter
the view; a detector that has never seen it treats lettering as texture;
and a privacy mask removes what is under it altogether.

What is modelled is what such overlays look like and where they are:

* **Text** in a small fixed-pitch bitmap face, white with a dark outline or
  on a dark bar, scaled by whole pixels so its edges stay hard, in a corner
  a little way in from the edge. A date and time in one of the orders
  recorders use, and a channel or site name in another corner.
* **Privacy masks**: rectangles filled with black or grey, or reduced to
  coarse blocks.

**Who draws it decides where in the chain it goes.** A camera burns its
overlay in before it encodes the stream, so the lettering is compressed
with the picture and carries the codec's ringing. A recorder draws its own
on the decoded picture, and it stays clean. ``drawn_by`` says which, and
:func:`~weathermentations.presets.pipeline.physical_order` places the
transform accordingly.

**What is under a privacy mask is gone**, so a box that a mask covers, by
``hidden_box_coverage`` or more of its area, is dropped. Masks and
keypoints are left as they are: the transform does not move anything.

Drawn on code values, as a character generator does, and not in linear
light.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from albumentations.core.type_definitions import Targets
from pydantic import AfterValidator, Field

from weathermentations.base import WeatherTransform
from weathermentations.diagnostics import record_draw

# Where `physical_order` puts an overlay: a camera's just before the codec,
# a recorder's after the link. They extend the table in presets/pipeline.py.
CAMERA_STAGE = 10
RECORDER_STAGE = 13

# A 5 by 7 face: seven rows of five dots a character, top row first.
_FACE = {
    "0": "01110 10001 10011 10101 11001 10001 01110",
    "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111",
    "3": "11111 00010 00100 00010 00001 10001 01110",
    "4": "00010 00110 01010 10010 11111 00010 00010",
    "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110",
    "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110",
    "9": "01110 10001 10001 01111 00001 00010 01100",
    "A": "01110 10001 10001 11111 10001 10001 10001",
    "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01110 10001 10000 10000 10000 10001 01110",
    "D": "11100 10010 10001 10001 10001 10010 11100",
    "E": "11111 10000 10000 11110 10000 10000 11111",
    "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01110 10001 10000 10111 10001 10001 01111",
    "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "01110 00100 00100 00100 00100 00100 01110",
    "J": "00111 00010 00010 00010 00010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001",
    "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001",
    "N": "10001 10001 11001 10101 10011 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110",
    "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101",
    "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110",
    "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110",
    "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 10101 01010",
    "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 10001 01010 00100 00100 00100",
    "Z": "11111 00001 00010 00100 01000 10000 11111",
    "-": "00000 00000 00000 11111 00000 00000 00000",
    "/": "00001 00001 00010 00100 01000 10000 10000",
    ":": "00000 00100 00100 00000 00100 00100 00000",
    ".": "00000 00000 00000 00000 00000 00110 00110",
    " ": "00000 00000 00000 00000 00000 00000 00000",
}
_GLYPHS = {
    character: np.array([[dot == "1" for dot in row] for row in rows.split()], dtype=bool)
    for character, rows in _FACE.items()
}
GLYPH_ROWS, GLYPH_COLUMNS = 7, 5

_DAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_NAMES = (
    "CAM {n:02d}", "CH{n}", "CAMERA {n}", "IPC", "ENTRANCE", "CAR PARK", "GATE {n}",
    "LOADING BAY", "RECEPTION", "YARD", "STAIRS {n}", "NVR CH{n:02d}", "FRONT DOOR",
)  # fmt: skip
_CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")


def render_text(text: str, scale: int) -> np.ndarray:
    """A line of text as dots: ``(7 scale, (6 n - 1) scale)`` bool, each dot ``scale`` square.

    Characters the face does not have are drawn as spaces. One column of
    dots is left between characters.
    """
    cells = []
    for character in text.upper():
        cells.append(_GLYPHS.get(character, _GLYPHS[" "]))
        cells.append(np.zeros((GLYPH_ROWS, 1), dtype=bool))
    line = np.concatenate(cells[:-1], axis=1) if cells else np.zeros((GLYPH_ROWS, 0), dtype=bool)
    dots: np.ndarray = np.repeat(np.repeat(line, scale, axis=0), scale, axis=1)
    return dots


def timestamp(rng: np.random.Generator) -> str:
    """A date and time on some day of the last decade, in an order recorders use."""
    year, month, day = (
        int(rng.integers(2015, 2027)),
        int(rng.integers(1, 13)),
        int(rng.integers(1, 29)),
    )
    hour, minute, second = (
        int(rng.integers(0, 24)),
        int(rng.integers(0, 60)),
        int(rng.integers(0, 60)),
    )
    clock = f"{hour:02d}:{minute:02d}:{second:02d}"
    weekday = _DAYS[int(rng.integers(0, 7))]
    return (
        f"{year}-{month:02d}-{day:02d} {clock}",
        f"{day:02d}/{month:02d}/{year} {clock}",
        f"{month:02d}-{day:02d}-{year} {weekday} {clock}",
        f"{year}/{month:02d}/{day:02d} {clock}",
        f"{day:02d}-{month:02d}-{year} {clock}",
    )[int(rng.integers(0, 5))]


def margin_px(height: int, margin: float) -> int:
    """A margin given as a fraction of the frame's height, in whole pixels."""
    return round(margin * height)


def stamp(
    image: np.ndarray, dots: np.ndarray, corner: str, margin: int, style: str, scale: int
) -> None:
    """Draw a line of text into ``image`` (float, 0 to 1) at a corner, in place.

    White dots, and round them either a dark outline one dot wide
    (``"outline"``), a dark bar behind the whole line (``"bar"``), or
    nothing (``"plain"``). Clipped to the frame if the line is wider.
    """
    height, width = image.shape[:2]
    pad = scale
    block = np.zeros((dots.shape[0] + 2 * pad, dots.shape[1] + 2 * pad), dtype=bool)
    block[pad:-pad, pad:-pad] = dots
    top = margin if corner.startswith("top") else height - margin - block.shape[0]
    left = margin if corner.endswith("left") else width - margin - block.shape[1]
    top, left = max(0, top), max(0, left)
    rows = min(block.shape[0], height - top)
    columns = min(block.shape[1], width - left)
    if rows <= 0 or columns <= 0:
        return
    block = block[:rows, :columns]
    region = image[top : top + rows, left : left + columns]
    if style == "bar":
        region *= 0.25
    elif style == "outline":
        size = 2 * scale + 1
        around = cv2.dilate(block.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)
        region[around] = 0.04
    region[block] = 0.94


def blank(
    image: np.ndarray, box: tuple[int, int, int, int], fill: float | None, block: int
) -> None:
    """Hide a rectangle of ``image`` in place: ``box`` is ``(left, top, right, bottom)``.

    Filled with the grey ``fill``, or with ``fill`` of ``None`` reduced to
    squares ``block`` pixels across, each the mean of what it covers.
    """
    left, top, right, bottom = box
    region = image[top:bottom, left:right]
    if region.size == 0:
        return
    if fill is not None:
        region[:] = fill
        return
    small = cv2.resize(
        region,
        (max(1, region.shape[1] // block), max(1, region.shape[0] // block)),
        interpolation=cv2.INTER_AREA,
    )
    region[:] = cv2.resize(
        small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST
    ).reshape(region.shape)


class OnScreenOverlay(WeatherTransform):
    """A timestamp, a camera name and privacy masks drawn over the frame.

    Needs no depth. Not a physical effect; see the module docstring.

    Args:
        timestamp_probability: chance the frame carries a date and time.
        label_probability: chance it carries a camera or site name.
        text_height_range: height of the lettering, as a fraction of the
            frame's height. Drawn at the nearest whole number of pixels a
            dot, and never under one.
        margin_range: how far in from the frame's edges the text sits, as
            a fraction of its height.
        privacy_mask_count_range: how many rectangles are blanked.
        privacy_mask_size_range: each one's width and height, as fractions
            of the frame's.
        hidden_box_coverage: a box with at least this share of its area
            under privacy masks is dropped. Above 1, none are.
        drawn_by: ``"camera"``, burned in before the stream is encoded, or
            ``"recorder"``, drawn on the decoded picture. Decides where
            ``physical_order`` puts the transform.

    Targets:
        image, bboxes

    Image types:
        uint8, float32
    """

    _targets = (Targets.IMAGE, Targets.BBOXES)

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``OnScreenOverlay`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """

        timestamp_probability: float = Field(ge=0.0, le=1.0)
        label_probability: float = Field(ge=0.0, le=1.0)
        text_height_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.005, 0.2)),
        ]
        margin_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.2)),
        ]
        privacy_mask_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 16)),
        ]
        privacy_mask_size_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.01, 1.0)),
        ]
        hidden_box_coverage: float = Field(gt=0.0, le=2.0)
        drawn_by: Literal["camera", "recorder"]

    def __init__(
        self,
        timestamp_probability: float = 0.9,
        label_probability: float = 0.6,
        text_height_range: tuple[float, float] = (0.022, 0.045),
        margin_range: tuple[float, float] = (0.01, 0.04),
        privacy_mask_count_range: tuple[int, int] = (0, 2),
        privacy_mask_size_range: tuple[float, float] = (0.06, 0.3),
        hidden_box_coverage: float = 0.9,
        drawn_by: Literal["camera", "recorder"] = "camera",
        p: float = 0.5,
    ):
        """Store the settings; each argument is described in the class docstring."""
        super().__init__(p=p)
        self.timestamp_probability = timestamp_probability
        self.label_probability = label_probability
        self.text_height_range = text_height_range
        self.margin_range = margin_range
        self.privacy_mask_count_range = privacy_mask_count_range
        self.privacy_mask_size_range = privacy_mask_size_range
        self.hidden_box_coverage = hidden_box_coverage
        self.drawn_by = drawn_by
        # Read by `presets.pipeline.stage_of`: before the codec for a
        # camera's overlay, after everything for a recorder's.
        self.pipeline_stage = CAMERA_STAGE if drawn_by == "camera" else RECORDER_STAGE

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Decide what is written, where, and what is blanked.

        Returns ``lines``, each a line of text with its corner and style;
        the lettering's ``scale`` and ``margin`` in pixels; and ``masks``,
        each a rectangle with how it is filled.
        """
        del data
        height, width = params["shape"][:2]
        rng = self.random_generator

        scale = max(1, round(self.py_random.uniform(*self.text_height_range) * height / GLYPH_ROWS))
        # A recorder does not write off the edge of its own picture: the
        # longest line it has, a date with the day of the week, must fit.
        longest = (GLYPH_COLUMNS + 1) * 23 + 2
        scale = max(1, min(scale, (width - 2 * margin_px(height, self.margin_range[1])) // longest))
        margin = margin_px(height, self.py_random.uniform(*self.margin_range))
        style = ("outline", "outline", "bar", "plain")[int(rng.integers(0, 4))]
        corners = [_CORNERS[index] for index in rng.permutation(4)]
        lines = []
        if self.py_random.random() < self.timestamp_probability:
            lines.append((timestamp(rng), corners[0], style))
        if self.py_random.random() < self.label_probability:
            name = _NAMES[int(rng.integers(0, len(_NAMES)))].format(n=int(rng.integers(1, 17)))
            lines.append((name, corners[1], style))

        masks = []
        for _ in range(self.py_random.randint(*self.privacy_mask_count_range)):
            mask_w = max(2, round(self.py_random.uniform(*self.privacy_mask_size_range) * width))
            mask_h = max(2, round(self.py_random.uniform(*self.privacy_mask_size_range) * height))
            left = int(rng.integers(0, max(1, width - mask_w + 1)))
            top = int(rng.integers(0, max(1, height - mask_h + 1)))
            kind = int(rng.integers(0, 3))
            fill = (0.0, 0.35, None)[kind]
            block = max(4, round(0.03 * height))
            masks.append(
                ((left, top, min(width, left + mask_w), min(height, top + mask_h)), fill, block)
            )

        record_draw(
            text=[line[0] for line in lines],
            style=style,
            dot_px=scale,
            margin_px=margin,
            privacy_mask_count=len(masks),
        )
        return {"lines": lines, "scale": scale, "margin": margin, "masks": masks}

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        """The image is drawn on; boxes hidden by a privacy mask are dropped."""
        return {**super().targets, "bboxes": self.apply_to_bboxes}

    def apply(
        self,
        img: np.ndarray,
        lines: list[tuple[str, str, str]],
        scale: int,
        margin: int,
        masks: list[tuple[tuple[int, int, int, int], float | None, int]],
        **params: Any,
    ) -> np.ndarray:
        """Blank the privacy masks, then write the text over everything."""
        if not lines and not masks:
            return img
        is_integer = np.issubdtype(img.dtype, np.integer)
        canvas = img.astype(np.float32) / (255.0 if is_integer else 1.0)
        for box, fill, block in masks:
            blank(canvas, box, fill, block)
        for text, corner, style in lines:
            stamp(canvas, render_text(text, scale), corner, margin, style, scale)
        if is_integer:
            return np.rint(canvas * 255.0).astype(img.dtype)
        return canvas.astype(img.dtype)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        masks: list[tuple[tuple[int, int, int, int], float | None, int]],
        **params: Any,
    ) -> np.ndarray:
        """Drop each box that privacy masks cover by ``hidden_box_coverage`` or more.

        Boxes arrive normalised. Coverage is counted on the pixel grid, so
        that two masks over one box are not counted twice where they
        overlap.
        """
        if len(bboxes) == 0 or not masks:
            return bboxes
        height, width = params["shape"][:2]
        hidden = np.zeros((height, width), dtype=np.float32)
        for (left, top, right, bottom), _fill, _block in masks:
            hidden[top:bottom, left:right] = 1.0
        keep = []
        for box in bboxes:
            x0, y0 = int(np.floor(box[0] * width)), int(np.floor(box[1] * height))
            x1, y1 = int(np.ceil(box[2] * width)), int(np.ceil(box[3] * height))
            area = hidden[max(0, y0) : y1, max(0, x0) : x1]
            keep.append(area.size == 0 or float(area.mean()) < self.hidden_box_coverage)
        kept: np.ndarray = bboxes[np.array(keep, dtype=bool)]
        return kept

