"""The light that comes with precipitation.

Rain and snow fall from cloud, and cloud changes the light before a single
drop reaches the lens: the sun's direct beam is gone, so shadows fill in and
contrast drops; the sky is cloud, grey and even; colours lose saturation and
the balance moves a little towards blue. A frame with streaks and a wet road
under a blue sky with hard shadows is not a rainy frame, and with the
distance veil at its true strength (``docs/rain_visibility.md``) this light
is the strongest cue of rain a daylit picture has.

What this can and cannot do
---------------------------
It works on a finished photograph, so it can lower what is there and cannot
remove it. Shadow *edges* stay where they were: a hard-shadowed scene comes
out as a softer one, not as an overcast one. That is stated because it is
the limit of the approach, not a detail.

The level of the frame is left alone. Overcast light is several times dimmer
than sunlight, but the camera's auto-exposure takes that back, and the
transforms that call this already normalise the frame's mean.
"""

from __future__ import annotations

import numpy as np

from surmentations import _native, mode
from surmentations.utils.channels import weighted_colour
from surmentations.utils.photometry import TABLE_SIZE, luminance, through_table

# At full strength: how far the logarithm of each pixel's luminance is pulled
# towards the scene's level, which is what losing the direct beam does to
# shadows and highlights alike (0.35 turns a 10:1 sun-to-shade ratio into
# 4.5:1); how far each pixel goes towards its own grey; and how much of the
# sky becomes even cloud.
_CONTRAST_LOSS = 0.35
_SATURATION_LOSS = 0.5
_SKY_COVER = 0.9

# Cloud is darker than the bright end of a clear sky, and slightly blue: it
# is lit from above through its own depth.
_CLOUD_LEVEL = 0.8
_CLOUD_TINT = np.array([0.97, 1.0, 1.04], dtype=np.float32)

# Saturation of the sky above which the frame is clearly a fine day. Below
# it the scene is already part-way overcast and gets less of the change.
_CLEAR_SKY_SATURATION = 0.3


def cloud_cover(rate_mm_h: float, half_rate_mm_h: float) -> float:
    """How overcast the light is, 0 to 1, from the precipitation rate.

    Heavier precipitation falls from deeper cloud. ``half_rate_mm_h`` is the
    rate at which the light is 63% of the way to fully overcast.
    """
    return float(1.0 - np.exp(-max(rate_mm_h, 0.0) / max(half_rate_mm_h, 1e-6)))


def apply_overcast(
    linear_rgb: np.ndarray, amount: float, sky: np.ndarray | None = None
) -> np.ndarray:
    """Relight a linear-light frame towards overcast by ``amount`` in ``[0, 1]``."""
    if amount <= 0.0:
        return linear_rgb
    has_sky = sky is not None and float(sky.mean()) >= 0.01

    if has_sky:
        assert sky is not None
        sky_colour = weighted_colour(linear_rgb, sky) / sky.sum()
        saturation = float((sky_colour.max() - sky_colour.min()) / max(sky_colour.max(), 1e-6))
        # A frame that is already grey overhead has less far to go.
        amount *= 0.3 + 0.7 * min(saturation / _CLEAR_SKY_SATURATION, 1.0)

    # Fill light. Under cloud the whole sky is the source, so what was in
    # shadow gets most of what everything else gets: the *ratio* between lit
    # and shadowed ground shrinks. That is a compression of luminance about
    # the scene's level, in the logarithm, and it acts on a shadow's whole
    # area — pulling pixels towards a local average instead flattened
    # texture and left the middle of every shadow as dark as it was.
    if mode.fast() and linear_rgb.dtype == np.float32 and linear_rgb.ndim == 3:
        return _relight_fast(linear_rgb, amount, sky if has_sky else None)

    luma = np.maximum(luminance(linear_rgb), 1e-4)
    ground = luma[sky < 0.5] if has_sky and sky is not None else luma
    level = float(np.exp(np.log(ground).mean())) if ground.size else float(luma.mean())
    gain = np.clip((level / luma) ** (_CONTRAST_LOSS * amount), 0.5, 3.0)
    out = linear_rgb * gain[..., None]

    grey = luminance(out)[..., None]
    out = out + (_SATURATION_LOSS * amount) * (grey - out)

    if has_sky:
        assert sky is not None
        level = float(np.percentile(luminance(linear_rgb)[sky > 0.5], 85.0)) * _CLOUD_LEVEL
        cover = (sky * (_SKY_COVER * amount))[..., None]
        # Keep a little of the sky's own gradient, so cloud is not a flat fill.
        cloud = level * _CLOUD_TINT * (0.85 + 0.15 * grey / max(float(grey.mean()), 1e-6))
        out = out * (1.0 - cover) + cloud * cover

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# Where in each cell of the gain table its gain is taken: the middle, so that
# a luminance is at most half a cell from the one its gain was worked out for.
_TABLE_CENTRES = (np.arange(TABLE_SIZE, dtype=np.float32) + 0.5) / np.float32(TABLE_SIZE - 1)


def _relight_fast(linear_rgb: np.ndarray, amount: float, sky: np.ndarray | None) -> np.ndarray:
    """:func:`apply_overcast` in fast mode, for a float32 RGB frame.

    The same relighting with three things taken once that the reference
    takes per pixel or more than once:

    * **The contrast gain is a power of luminance**, a million powers a
      frame. Luminance is in ``[0, 1]``, so the power is taken for 65536
      values of it and looked up. A pixel's gain is the gain of a luminance
      within 8 millionths of its own.
    * **Luminance is computed once**, where the reference computes it three
      times. The grey of the relit frame is the gain times the luminance it
      was made from, which is what the luminance of the product comes to.
    * **The frame is gone over once**, by a compiled pass where there is
      one, where the reference makes eight frame-sized results.

    The statistics, the level the frame is compressed towards and the cloud's
    brightness, are the reference's. Against it a frame is within a level.
    """
    raw = luminance(linear_rgb)
    luma = np.maximum(raw, 1e-4)
    ground = luma[sky < 0.5] if sky is not None else luma
    level = float(np.exp(np.log(ground).mean())) if ground.size else float(luma.mean())

    table = np.clip(
        (level / np.maximum(_TABLE_CENTRES, 1e-4)) ** (_CONTRAST_LOSS * amount), 0.5, 3.0
    ).astype(np.float32)
    gain = through_table(raw, table)
    grey = gain * raw
    mean_grey = max(float(grey.mean()), 1e-6)
    desaturate = _SATURATION_LOSS * amount

    cloud = np.zeros(3, dtype=np.float32)
    cover_scale = 0.0
    if sky is not None:
        cloud_level = float(np.percentile(raw[sky > 0.5], 85.0)) * _CLOUD_LEVEL
        cloud = (cloud_level * _CLOUD_TINT).astype(np.float32)
        cover_scale = _SKY_COVER * amount

    done = _native.overcast_blend(
        linear_rgb, gain, grey, sky, cloud, desaturate, cover_scale, mean_grey
    )
    if done is not None:
        return done
    # What the kernel computes, in its order, so that the frame is the same
    # with and without it.
    out = linear_rgb * gain[..., None]
    out = out + desaturate * (grey[..., None] - out)
    if sky is not None:
        cover = (sky * cover_scale)[..., None]
        shade = (0.85 + (0.15 * grey) / mean_grey)[..., None]
        out = out * (1.0 - cover) + (cloud * shade) * cover
    clipped: np.ndarray = np.clip(out, 0.0, 1.0)
    return clipped
