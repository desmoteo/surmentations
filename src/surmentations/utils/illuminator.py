"""The camera's own infrared illuminator, and things lit by it close to the lens.

What a web, an insect or a mote of dust in front of the housing have in
common: they are lit by a ring of LEDs around the lens from centimetres
away, they are far inside the near focus limit, so each point of them is
drawn as the aperture's disc, and they are bright enough that the dome and
the lens spread a share of their light around them.

Three pieces, used by ``SpiderWeb``, ``NightInsects`` and ``IrDomeGlare``:
how the lamp's light falls off with distance in the near field
(:func:`lamp_falloff`), a field of discs of light summed into a frame
(:func:`splat_discs`), and the veiling glare of light that bright
(:func:`veiling_glare`).
"""

from __future__ import annotations

import numpy as np

from surmentations import _native
from surmentations.utils.filters import wide_blur

# Where the lamp's falloff is one: about where a web's hub hangs, and where
# ``SpiderWeb`` states its gain.
REFERENCE_DISTANCE_M = 0.1
# Radius of the ring of LEDs around the lens. A thread on the axis at distance
# z is sqrt(z^2 + ring^2) from each of them and that far off their beams,
# which is what stops 1/z^2 from running away in the last centimetres.
RING_RADIUS_M = 0.025
# Veiling glare: the scale of most of it and of its wide tail, as fractions
# of the diagonal, and the tail's share.
GLARE_SIGMA = 0.006
HALO_SIGMA = 0.03
HALO_SHARE = 0.25


def lamp_falloff(distance_m: np.ndarray | float) -> np.ndarray:
    """The illuminator's irradiance at a distance, relative to the reference distance.

    A ring of LEDs ``RING_RADIUS_M`` off the axis, facing forwards and
    radiating as a Lambertian emitter does, by the cosine of the angle off
    its own axis, onto silk that scatters the same from any side. A point on
    the lens's axis at distance ``z`` is ``sqrt(z^2 + ring^2)`` from each LED
    and ``atan(ring / z)`` off its beam, so it receives ``z / (z^2 +
    ring^2)^1.5``: inverse square from afar, greatest about ``0.7 ring`` out
    and falling to nothing at the glass. Scaled to ``1`` at
    ``REFERENCE_DISTANCE_M``.
    """
    distance = np.asarray(distance_m, dtype=np.float64)
    ring = RING_RADIUS_M**2

    def received(z: np.ndarray | float) -> np.ndarray:
        """Irradiance on the axis at ``z``, unscaled."""
        value: np.ndarray = np.asarray(z / (z * z + ring) ** 1.5)
        return value

    falloff: np.ndarray = received(distance) / received(REFERENCE_DISTANCE_M)
    return falloff


def veiling_glare(lit: np.ndarray, glare: float) -> np.ndarray:
    """Spread a share ``glare`` of a field of light around where it fell; in place.

    Veiling glare, taken on the unclipped light: what spills is a share of
    what was returned, not of what the sensor could record of it. Most of
    it stays close; a little goes wide, which is nothing around a thread and
    a halo around a body forty times as bright. ``lit`` is ``(H, W)``
    float32 and is returned.
    """
    if glare <= 0.0:
        return lit
    height, width = lit.shape
    diagonal = float(np.hypot(width, height))
    near = wide_blur(lit, max(1.0, GLARE_SIGMA * diagonal))
    wide = wide_blur(lit, max(1.0, HALO_SIGMA * diagonal))
    near *= np.float32((1.0 - HALO_SHARE) * glare)
    wide *= np.float32(HALO_SHARE * glare)
    lit *= np.float32(1.0 - glare)
    lit += near
    lit += wide
    return lit


# Discs are drawn in batches of at most this many pixels, to bound the memory
# a batch's patches take.
_BATCH_PIXELS = 2_000_000


def splat_discs(
    canvas: np.ndarray, x: np.ndarray, y: np.ndarray, radius: np.ndarray, weight: np.ndarray
) -> None:
    """Add a field of discs of light to ``canvas``, ``(H, W)`` float32.

    Disc ``i`` is centred on ``(x[i], y[i])``, pixel centres being whole
    numbers, with radius ``radius[i]`` and an edge one pixel wide, and holds
    ``weight[i]`` in all: every pixel it covers gains its coverage times
    ``weight / (pi (r^2 + 1/12))``, the area of a disc with that edge. They
    add, as light does, so where threads cross and where beads overlap the
    picture is brighter.

    The compiled kernel draws the field in one pass when it is installed.
    This is the same picture without it, to the last bit: discs taken in
    order of size, a batch of one patch size at a time, a pixel's share
    worked out in float32 and summed in double.
    """
    if len(x) == 0:
        return
    height, width = canvas.shape
    half = np.ceil(radius + np.float32(0.5)).astype(np.int64)
    field = np.zeros((height, width), dtype=np.float64)
    if _native.splat_discs(field, x, y, radius, weight, np.argsort(half, kind="stable")):
        canvas += field.astype(np.float32)
        return
    total = field.reshape(-1)
    column = np.rint(x).astype(np.int64)
    row = np.rint(y).astype(np.int64)
    peak = weight / (np.float32(np.pi) * (radius * radius + np.float32(1.0 / 12.0)))
    for size in np.unique(half):
        chosen = np.flatnonzero(half == size)
        span = np.arange(-size, size + 1)
        batch = max(1, _BATCH_PIXELS // (len(span) ** 2))
        for begin in range(0, len(chosen), batch):
            which = chosen[begin : begin + batch]
            columns = column[which, None] + span[None, :]
            rows = row[which, None] + span[None, :]
            dx = columns.astype(np.float32) - x[which, None]
            dy = rows.astype(np.float32) - y[which, None]
            distance = np.sqrt((dx * dx)[:, None, :] + (dy * dy)[:, :, None])
            reach = radius[which, None, None] + np.float32(0.5)
            cover = np.clip(reach - distance, np.float32(0.0), np.float32(1.0))
            cover *= peak[which, None, None]
            inside = ((rows >= 0) & (rows < height))[:, :, None] & (
                (columns >= 0) & (columns < width)
            )[:, None, :]
            index = (rows[:, :, None] * width + columns[:, None, :])[inside]
            # In place, disc by disc and pixel by pixel in this order: the
            # sum a pixel ends with depends on it in the last bit.
            np.add.at(total, index, cover[inside].astype(np.float64))
    canvas += field.astype(np.float32)
