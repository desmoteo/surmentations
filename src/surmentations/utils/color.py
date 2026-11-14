"""Color-estimation helpers shared by atmosphere effects."""

from __future__ import annotations

import cv2
import numpy as np

from surmentations import mode
from surmentations.utils.channels import frame_mean, weighted_colour
from surmentations.utils.photometry import luminance


def sky_radiance_linear(linear_rgb: np.ndarray, percentile: float = 97.0) -> np.ndarray:
    """Estimate the scene's sky/illuminant radiance from a linear-light image.

    Rain streaks and snowflakes are not white objects — they are lenses and
    ice crystals lit by, and imaging, the sky. Compositing them toward this
    colour rather than toward white is what makes them tint with the weather,
    and it automatically makes them vanish against a bright sky and stand out
    against dark foliage, which is how real precipitation behaves.

    Returns an ``(C,)`` linear-light colour: the mean of the brightest
    ``100 - percentile`` percent of pixels by luminance.
    """
    luma = luminance(linear_rgb)
    if mode.fast() and linear_rgb.dtype == np.float32 and linear_rgb.shape[-1] == 3:
        return _brightest_mean(linear_rgb, luma, percentile)
    threshold = float(np.percentile(luma, percentile))
    bright = linear_rgb[luma >= threshold]
    if bright.size == 0:
        return frame_mean(linear_rgb)
    return bright.mean(axis=0).astype(np.float32)


# Bins of the luminance histogram the fast form reads its threshold from.
_LUMA_BINS = 16384


def _brightest_mean(linear_rgb: np.ndarray, luma: np.ndarray, percentile: float) -> np.ndarray:
    """:func:`sky_radiance_linear` in fast mode: the same mean, found by counting.

    A percentile of a million values is a partial sort of them, 5 ms, and
    taking the pixels above it out of the frame is 4 more. A histogram is
    one pass: the threshold is the edge of the bin at which the brightest
    ``100 - percentile`` percent has been counted, which is the percentile
    to within a 16384th of the frame's range, and OpenCV averages what is
    above it in place. The mean is of the same pixels give or take those in
    one bin.
    """
    top = float(luma.max())
    if not top > 0.0:
        return frame_mean(linear_rgb)
    luma = np.ascontiguousarray(luma)
    # The upper end is past the brightest value, which would otherwise fall
    # outside the last bin and not be counted.
    width = top * (1.0 + 1e-6) / _LUMA_BINS
    counts = cv2.calcHist([luma], [0], None, [_LUMA_BINS], [0.0, width * _LUMA_BINS]).ravel()
    wanted = max(1.0, luma.size * (1.0 - percentile / 100.0))
    # Brighter-or-equal counts, by bin; the last bin that still holds enough.
    above = np.cumsum(counts[::-1])[::-1]
    first = int(np.flatnonzero(above >= wanted)[-1])
    bright = (luma >= np.float32(first * width)).view(np.uint8)
    return np.asarray(cv2.mean(linear_rgb, mask=bright)[:3], dtype=np.float32)


def estimate_airlight(image: np.ndarray, top_percent: float = 0.1) -> np.ndarray:
    """Estimate atmospheric light (airlight) color from an image's brightest pixels.

    Not a full dark-channel-prior implementation (He, Sun & Tang 2009,
    "Single Image Haze Removal Using Dark Channel Prior") — a lightweight
    proxy that's good enough for a training-time augmentation: averages the
    color of the brightest ``top_percent`` percent of pixels by per-pixel intensity
    sum, which in most outdoor scenes is sky or existing haze.

    Returns an ``(C,)`` float32 color.
    """
    flat = image.reshape(-1, image.shape[-1]).astype(np.float32)
    intensity = flat.sum(axis=1)
    n = max(1, int(len(intensity) * top_percent / 100.0))
    brightest_idx = np.argpartition(intensity, -n)[-n:]
    return flat[brightest_idx].mean(axis=0)


# Grey-world: an average outdoor scene reflects roughly this fraction of the
# light falling on it.
TYPICAL_SCENE_ALBEDO = 0.20


def illuminant_radiance_linear(linear_rgb: np.ndarray, percentile: float = 97.0) -> np.ndarray:
    """Estimate the radiance a particle *in front of* the scene is lit by.

    Rain streaks and snowflakes image the sky, and the sky is frequently not
    in the frame — a camera angled down at a road sees none of it. Taking the
    particle's radiance from the scene's own brightest pixels then reads the
    brightest patch of *asphalt* as the illuminant, which is several times too
    dark, and the precipitation quietly disappears. That failure is worst
    exactly where these transforms are most used.

    So the estimate is the brighter of two readings:

    * the scene's bright/sky percentile, which is right whenever sky or a
      specular highlight is actually visible, and
    * the grey-world inversion ``mean_luminance / typical_albedo``, which
      recovers the illuminant from a surface-only frame — if an average scene
      reflects about a fifth of what falls on it, the light falling on it is
      about five times as bright as the scene looks.

    Taking the maximum means the sky wins when it is visible and the
    grey-world floor only applies when nothing bright is in frame.
    """
    sky = sky_radiance_linear(linear_rgb, percentile)

    mean_colour = frame_mean(linear_rgb)
    mean_luma = float(luminance(mean_colour.reshape(1, 1, -1)).ravel()[0])
    sky_luma = float(luminance(sky.reshape(1, 1, -1)).ravel()[0])

    implied = mean_luma / TYPICAL_SCENE_ALBEDO
    if sky_luma >= implied or sky_luma < 1e-6:
        return sky
    # Keep the scene's own colour balance; only lift the level.
    return (sky * (implied / sky_luma)).astype(np.float32)


# Sky has to cover this much of the frame before its radiance is read off it.
_MIN_SKY_FRACTION = 0.01

# How far the veil's colour is pulled from the sky's towards neutral, and
# which percentile of the sky's luminance sets its level.
_VEIL_DESATURATION = 0.75
_SKY_LEVEL_PERCENTILE = 85.0

# Blur, as a fraction of the image diagonal, applied before reading the
# brightest region of a frame with no sky in it. Wide enough to dilute a
# street lamp to nothing while leaving a pale wall or a patch of road alone.
_SOURCE_DILUTION_SIGMA = 0.04


def veil_radiance_linear(linear_rgb: np.ndarray, sky: np.ndarray | None = None) -> np.ndarray:
    """Radiance of the light scattered along a long path: the veil's colour.

    Fog, and the distance veils of rain and snow, converge on the radiance
    of the horizon sky. The obvious estimate of that — the brightest pixels
    in the frame — is right when those pixels are sky and badly wrong when
    they are lamps: on a real night street it came out at 17.8 times the
    frame's median and the colour of sodium, and fog turned the whole frame
    yellow.

    So, in order:

    * **If sky is in frame, read it.** ``sky`` is a mask from
      :func:`surmentations.utils.sky.sky_weight`. By day this is what the
      bright percentile was reaching for; at night it is dark, and the veil
      is dark with it. Taken at the sky's bright end and mostly desaturated.
    * **Otherwise infer it.** A surface-only frame under an open sky is lit
      by something about ``1 / albedo`` times brighter than it looks — the
      same grey-world argument as :func:`illuminant_radiance_linear` — and
      neutral in colour. That is floored by the brightest *region* of the
      frame, read after a blur wide enough that point lights do not count.

    Returns an ``(C,)`` linear-light colour.
    """
    channels = linear_rgb.shape[-1]
    if sky is not None and float(sky.mean()) >= _MIN_SKY_FRACTION:
        colour = weighted_colour(linear_rgb, sky) / sky.sum()
        # Two corrections, because the sky in the photograph is the sky
        # before the weather. Water droplets scatter all wavelengths alike,
        # so the veil is nearly white where a clear sky is blue; and it is as
        # bright as the sky's bright end — its haze and cloud — not as its
        # zenith. Without them fog under a blue sky came out as blue smoke.
        luma = luminance(linear_rgb)
        level = float(np.percentile(luma[sky > 0.5], _SKY_LEVEL_PERCENTILE))
        mean_level = max(float(luminance(colour.reshape(1, 1, -1)).ravel()[0]), 1e-6)
        colour = colour * (level / mean_level)
        return (colour + _VEIL_DESATURATION * (level - colour)).astype(np.float32)

    height, width = linear_rgb.shape[:2]
    # The blur is in proportion to the frame, so it is done on a small copy.
    small = linear_rgb[:: max(1, height // 128), :: max(1, width // 128)]
    small = np.ascontiguousarray(small, dtype=np.float32)
    sigma = max(1.0, _SOURCE_DILUTION_SIGMA * float(np.hypot(small.shape[0], small.shape[1])))
    kernel = int(sigma * 6) | 1
    blurred = _gaussian(small, kernel, sigma)
    region = sky_radiance_linear(blurred, 97.0)

    mean_luma = float(luminance(frame_mean(linear_rgb)[None, None, :])[0, 0])
    implied = min(mean_luma / TYPICAL_SCENE_ALBEDO, 1.0)
    region_luma = float(luminance(region.reshape(1, 1, -1)).ravel()[0])
    if region_luma >= implied:
        return region.astype(np.float32)
    return np.full(channels, implied, dtype=np.float32)


def _gaussian(image: np.ndarray, kernel: int, sigma: float) -> np.ndarray:
    """``cv2.GaussianBlur`` with a square kernel, imported where it is used."""
    import cv2

    return np.asarray(cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma))
