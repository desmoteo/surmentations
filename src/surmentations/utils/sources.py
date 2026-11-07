"""Finding the lights that are in the frame.

Glare, and the glow of fog at night, are both things a *light* does, and at
night the lights are in the picture: street lamps, headlights, lit windows.
A light is recorded as a small region at or near clipping. What it does not
record is how bright the light was — a clipped pixel is a lower bound — so
everything downstream takes an over-range factor as a parameter and does not
pretend to have measured it.

Large clipped regions are not lights in this sense. A blown-out sky or a
sunlit wall saturates the sensor without being a compact source, and treating
it as one puts a flare in the middle of every overexposed frame.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from surmentations.utils.channels import channel_max

# Detection runs on a copy this wide, so thresholds on area mean the same
# thing at every resolution.
_WORK_WIDTH = 320

# Linear-light level, in the brightest channel, from which a pixel counts as
# part of a source: sRGB 243 and up.
_SOURCE_LEVEL = 0.9

# A region larger than this fraction of the frame is an overexposed surface,
# not a lamp.
_MAX_AREA_FRACTION = 0.03


# Share of its bounding box a region must fill to count as compact. A disc
# fills 79%; a diagonal line a few percent.
_MIN_BOX_FILL = 0.45


class LightSource(NamedTuple):
    """One compact light in the frame.

    ``x`` and ``y`` are normalised to ``[0, 1]``; ``area`` is the fraction of
    the frame the clipped region covers; ``colour`` is its mean linear RGB,
    normalised so its largest channel is 1.
    """

    x: float
    y: float
    area: float
    colour: tuple[float, float, float]


def source_mask(linear_rgb: np.ndarray) -> np.ndarray:
    """``[0, 1]`` weight of how clipped each pixel is, shape ``(H, W)``."""
    peak = channel_max(linear_rgb)
    return np.clip((peak - _SOURCE_LEVEL) / (1.0 - _SOURCE_LEVEL), 0.0, 1.0).astype(np.float32)


# The smallest clipped region taken for a lamp, as a fraction of the frame:
# nine pixels of an 800 x 533 one. Below it is a glint on a leaf or a kerb.
_MIN_LAMP_AREA_FRACTION = 2e-5
# A lamp's clipped region at the frame's own resolution is a ragged thing,
# the fitting and the flare round it: it fills a quarter of its bounding box
# or more, and the box is at most this many times longer than it is wide. A
# lit kerb or a strip of lit road is longer; a spray of lit leaves is
# emptier.
_MIN_LAMP_BOX_FILL = 0.25
_MAX_LAMP_ASPECT = 2.5


def lamp_mask(linear_rgb: np.ndarray) -> np.ndarray:
    """:func:`source_mask`, kept only where the clipped pixels are a lamp, ``(H, W)``.

    Under a street lamp a good deal more than the lamp is clipped: the pool
    of light on the road, the white line through it, the leaves beside it.
    None of those was many times brighter than the sensor could record, and
    raising them by a lamp's over-range factor before a wet road reflects
    them filled its puddles with slabs of saturated yellow. A lamp is a
    region that is compact, neither a speck nor a surface nor a strip.

    The test is looser than :func:`light_sources`', which looks at a reduced
    copy on which a lamp is a blob. At full size its outline is ragged, and
    asked to fill its box as a disc does, every lamp of the night street
    this was tuned on failed: a wet road then reflected no lamp at all.
    """
    clipped = source_mask(linear_rgb)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (clipped > 0.0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return clipped
    frame = float(clipped.size)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    wide = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64)
    tall = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    lamp = (
        (areas >= _MIN_LAMP_AREA_FRACTION * frame)
        & (areas <= _MAX_AREA_FRACTION * frame)
        & (areas >= _MIN_LAMP_BOX_FILL * wide * tall)
        & (np.maximum(wide, tall) <= _MAX_LAMP_ASPECT * np.minimum(wide, tall))
    )
    lamp[0] = False
    return (clipped * lamp[labels]).astype(np.float32)


def light_sources(linear_rgb: np.ndarray, max_sources: int = 4) -> list[LightSource]:
    """The compact lights in a linear-light frame, largest first."""
    height, width = linear_rgb.shape[:2]
    scale = _WORK_WIDTH / max(width, 1)
    size = (max(8, round(width * scale)), max(8, round(height * scale)))
    small = cv2.resize(linear_rgb, size, interpolation=cv2.INTER_AREA)
    # Thresholded at full size and then reduced, so a lamp a few pixels wide
    # is neither lost between samples nor diluted below the bar, and a bright
    # but unclipped surface does not pass a bar lowered to catch it.
    clipped = (channel_max(linear_rgb) >= _SOURCE_LEVEL).astype(np.float32)
    lit = (cv2.resize(clipped, size, interpolation=cv2.INTER_AREA) > 0.2).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(lit, connectivity=8)

    total = float(size[0] * size[1])
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64) / total
    # Largest first, and only the ones asked for: a noisy bright surface can
    # break into thousands of specks, and reading each one's colour was most
    # of this function's time.
    # A lamp is compact. A clipped road marking or a lit kerb is as small in
    # area and is not a light: it is long and thin, and fills little of the
    # box around it.
    boxes = stats[1:, cv2.CC_STAT_WIDTH] * stats[1:, cv2.CC_STAT_HEIGHT]
    compact = stats[1:, cv2.CC_STAT_AREA] >= _MIN_BOX_FILL * boxes
    order = [
        int(i) + 1
        for i in np.argsort(-areas)
        if areas[i] <= _MAX_AREA_FRACTION and compact[i]
    ]
    found = []
    for label in order[:max_sources]:
        colour = small[labels == label].mean(axis=0)
        colour = colour / max(float(colour.max()), 1e-6)
        found.append(
            LightSource(
                float(centroids[label, 0] + 0.5) / size[0],
                float(centroids[label, 1] + 0.5) / size[1],
                float(areas[label - 1]),
                (float(colour[0]), float(colour[1]), float(colour[2])),
            )
        )
    return found


def light_field(linear_rgb: np.ndarray, sigma_fraction: float) -> np.ndarray:
    """The frame blurred over ``sigma_fraction`` of its diagonal, ``(H, W, C)``.

    Where the light is, with everything that says what it fell on averaged
    away. A blur this wide holds no detail finer than itself, so it is done
    on a copy a few dozen pixels across.
    """
    height, width = linear_rgb.shape[:2]
    factor = max(1, round(max(height, width) / 96))
    size = (max(4, width // factor), max(4, height // factor))
    small = cv2.resize(linear_rgb, size, interpolation=cv2.INTER_AREA)
    sigma = max(0.8, sigma_fraction * float(np.hypot(*size)))
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC).clip(0.0, None)
