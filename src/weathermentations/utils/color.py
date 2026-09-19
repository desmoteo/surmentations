"""Color-estimation helpers shared by atmosphere effects."""

from __future__ import annotations

import numpy as np

from weathermentations.utils.photometry import luminance


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
    threshold = float(np.percentile(luma, percentile))
    bright = linear_rgb[luma >= threshold]
    if bright.size == 0:
        return linear_rgb.reshape(-1, linear_rgb.shape[-1]).mean(axis=0)
    return bright.mean(axis=0).astype(np.float32)


def estimate_airlight(image: np.ndarray, top_percent: float = 0.1) -> np.ndarray:
    """Estimate atmospheric light (airlight) color from an image's brightest pixels.

    Not a full dark-channel-prior implementation (He, Sun & Tang 2009,
    "Single Image Haze Removal Using Dark Channel Prior") — a lightweight
    proxy that's good enough for a training-time augmentation: averages the
    color of the brightest ``top_percent``% of pixels by per-pixel intensity
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

    mean_colour = linear_rgb.reshape(-1, linear_rgb.shape[-1]).mean(axis=0)
    mean_luma = float(luminance(mean_colour.reshape(1, 1, -1)).ravel()[0])
    sky_luma = float(luminance(sky.reshape(1, 1, -1)).ravel()[0])

    implied = mean_luma / TYPICAL_SCENE_ALBEDO
    if sky_luma >= implied or sky_luma < 1e-6:
        return sky
    # Keep the scene's own colour balance; only lift the level.
    return (sky * (implied / sky_luma)).astype(np.float32)
