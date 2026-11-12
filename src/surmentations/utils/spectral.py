"""Thin-film interference for anti-reflection lens coatings.

Every glass-air surface inside a lens reflects a few percent of the light
that hits it, and a ghost is what you see when light bounces off two of them
before reaching the sensor. Modern lenses suppress that with a quarter-wave
anti-reflection coating — but a single layer can only cancel the reflection
at *one* wavelength. Either side of the design wavelength the cancellation
degrades, so the residual reflection is strongly coloured, and it is that
residual, squared by the two bounces, that gives real flare ghosts their
characteristic magenta / cyan / amber / green-gold casts.

This is why picking a random hue per ghost never looks right: the palette of
a real lens flare is narrow and specific, set by the coatings' design
wavelengths, not drawn from the whole colour wheel.

The reflectance of a single film of index ``n1`` and optical thickness
``n1 d`` on a substrate ``n2`` is the standard two-interface Airy sum::

    r1 = (n0 - n1) / (n0 + n1)          (air -> film)
    r2 = (n1 - n2) / (n1 + n2)          (film -> glass)
    delta = 2 pi n1 d cos(theta1) / lambda
    R = (r1^2 + r2^2 + 2 r1 r2 cos 2delta)
        / (1 + r1^2 r2^2 + 2 r1 r2 cos 2delta)

For a quarter-wave coating at design wavelength ``lambda0`` the optical
thickness is ``lambda0 / 4``, so ``delta = (pi/2) (lambda0 / lambda)
cos(theta1)``. The ``cos(theta1)`` factor is the reason a coating's
cancellation notch **shifts toward the blue as the angle of incidence grows**
— the familiar effect of a lens's coatings changing colour as you tilt it,
and the reason the ghosts further along the flare axis (formed by steeper
bounces) are tinted differently from the ones near the source.

Two different jobs live here, and they need different resolutions:

* **Tinting** (a ghost's colour) samples the spectrum at three representative
  wavelengths. The result is a tint for a synthetic flare, not a colorimetric
  prediction, and three samples are visually indistinguishable from a full
  integration for that purpose while being a few lines instead of a
  dependency.
* **Dispersion** (a diffraction corona, where each wavelength lands somewhere
  different) cannot use three samples. Doing so gives three separate red,
  green and blue rings with gaps between them rather than a rainbow, which is
  unmistakably wrong. Those callers integrate over the visible band using
  :func:`wavelength_to_rgb`.

The distinction is whether wavelength changes the colour of a thing or its
*position*. Only the second needs a continuous spectrum.
"""

from __future__ import annotations

import numpy as np

# Approximate dominant wavelengths (nm) of the sRGB primaries, used as the
# three sample points of every spectrum in this module.
RGB_WAVELENGTHS_NM = np.array([610.0, 550.0, 465.0], dtype=np.float32)

N_AIR = 1.0
N_MGF2 = 1.38  # magnesium fluoride, the classic single-layer AR coating
N_CROWN_GLASS = 1.52


def coating_reflectance(
    wavelength_nm: np.ndarray,
    design_wavelength_nm: float,
    incidence_deg: float = 0.0,
    n_film: float = N_MGF2,
    n_substrate: float = N_CROWN_GLASS,
) -> np.ndarray:
    """Reflectance of a quarter-wave AR coating at the given wavelengths.

    Args:
        wavelength_nm: wavelengths to evaluate, in nanometres.
        design_wavelength_nm: the wavelength the coating cancels (its notch).
        incidence_deg: angle of incidence in air. Larger angles blue-shift
            the notch, via the ``cos(theta1)`` term in the phase thickness.
        n_film: refractive index of the coating.
        n_substrate: refractive index of the glass underneath.

    Returns:
        Reflectance in ``[0, 1]``, same shape as ``wavelength_nm``.
    """
    wavelength = np.asarray(wavelength_nm, dtype=np.float32)

    r1 = (N_AIR - n_film) / (N_AIR + n_film)
    r2 = (n_film - n_substrate) / (n_film + n_substrate)

    # Snell into the film, then the phase thickness of a quarter-wave layer.
    sin_theta1 = np.sin(np.deg2rad(incidence_deg)) / n_film
    cos_theta1 = float(np.sqrt(max(0.0, 1.0 - sin_theta1**2)))
    delta = (np.pi / 2.0) * (design_wavelength_nm / wavelength) * cos_theta1

    cos_2delta = np.cos(2.0 * delta)
    numerator = r1**2 + r2**2 + 2.0 * r1 * r2 * cos_2delta
    denominator = 1.0 + (r1**2) * (r2**2) + 2.0 * r1 * r2 * cos_2delta
    return np.clip(numerator / denominator, 0.0, 1.0).astype(np.float32)


def ghost_tint(
    design_wavelength_a_nm: float,
    design_wavelength_b_nm: float,
    incidence_deg: float = 0.0,
    bounces: int = 2,
) -> np.ndarray:
    """Linear-light RGB tint of a ghost formed by bouncing off coated surfaces.

    A ghost needs an **even** number of internal reflections to end up
    travelling back toward the sensor, so its spectrum is a *product* of
    residual reflectances — never a single one. Multiplying shallow notches
    together is what turns a barely-tinted reflection into a saturated one.

    ``bounces`` is that number. Two is the common case and gives the mild
    violet cast a single-layer MgF2 coating is known for. Higher-order
    ghosts genuinely occur in lenses with many elements, and multiplying more
    reflectances together makes them **both fainter and more saturated** —
    which is why the few strongly-coloured ghosts in a real flare are also
    the dim ones. Callers pair this with a matching drop in intensity.

    Returns an RGB triple normalised to a maximum of 1, i.e. the ghost's
    *colour*. Its absolute brightness is handled separately, since that
    depends on aperture and on how far the ghost is defocused, not on the
    coating.
    """
    reflect_a = coating_reflectance(RGB_WAVELENGTHS_NM, design_wavelength_a_nm, incidence_deg)
    reflect_b = coating_reflectance(RGB_WAVELENGTHS_NM, design_wavelength_b_nm, incidence_deg)

    # Normalise the single-pair spectrum *before* raising it to the bounce
    # order. The result is identical after the final normalisation, but the
    # intermediate stays O(1) instead of collapsing toward zero — a six-bounce
    # ghost's raw reflectance product is ~1e-11, small enough that a naive
    # magnitude guard reads it as degenerate and returns white.
    pair_spectrum = reflect_a * reflect_b
    peak = float(pair_spectrum.max())
    if not np.isfinite(peak) or peak <= 0.0:
        return np.ones(3, dtype=np.float32)

    spectrum = (pair_spectrum / peak) ** max(1, bounces // 2)
    return (spectrum / float(spectrum.max())).astype(np.float32)


def wavelength_to_rgb(wavelength_nm: np.ndarray) -> np.ndarray:
    """Approximate linear-light sRGB for a single visible wavelength.

    The piecewise approximation to the CIE colour-matching functions
    popularised by Dan Bruton, which is the standard cheap stand-in for a full
    spectral integration and is more than accurate enough for rendering a
    spectrum into an image.

    This exists because some effects disperse light **continuously** rather
    than tinting it. Sampling such an effect at only the three sRGB primaries
    does not give a rainbow — it gives three separate red, green and blue
    rings with gaps between them, which is what a three-sample dispersion ring
    actually looks like and is unmistakably wrong. A spectrum has to be
    integrated over wavelength, and that needs a colour per wavelength.

    Returns an ``(..., 3)`` array; wavelengths outside ~380-780 nm return 0.
    """
    wavelength = np.asarray(wavelength_nm, dtype=np.float32)
    red = np.zeros_like(wavelength)
    green = np.zeros_like(wavelength)
    blue = np.zeros_like(wavelength)

    bands = [
        # (low, high, r, g, b) as functions of the wavelength in the band.
        (380.0, 440.0, lambda w: -(w - 440.0) / 60.0, lambda w: 0.0 * w, lambda w: 0.0 * w + 1.0),
        (440.0, 490.0, lambda w: 0.0 * w, lambda w: (w - 440.0) / 50.0, lambda w: 0.0 * w + 1.0),
        (490.0, 510.0, lambda w: 0.0 * w, lambda w: 0.0 * w + 1.0, lambda w: -(w - 510.0) / 20.0),
        (510.0, 580.0, lambda w: (w - 510.0) / 70.0, lambda w: 0.0 * w + 1.0, lambda w: 0.0 * w),
        (580.0, 645.0, lambda w: 0.0 * w + 1.0, lambda w: -(w - 645.0) / 65.0, lambda w: 0.0 * w),
        (645.0, 781.0, lambda w: 0.0 * w + 1.0, lambda w: 0.0 * w, lambda w: 0.0 * w),
    ]
    for low, high, r_fn, g_fn, b_fn in bands:
        mask = (wavelength >= low) & (wavelength < high)
        if not np.any(mask):
            continue
        band = wavelength[mask]
        red[mask] = r_fn(band)
        green[mask] = g_fn(band)
        blue[mask] = b_fn(band)

    # The eye's response rolls off at both ends of the visible band, so the
    # extremes are dimmer rather than simply cut off.
    falloff = np.ones_like(wavelength)
    low_end = (wavelength >= 380.0) & (wavelength < 420.0)
    falloff[low_end] = 0.3 + 0.7 * (wavelength[low_end] - 380.0) / 40.0
    high_end = (wavelength >= 700.0) & (wavelength < 781.0)
    falloff[high_end] = 0.3 + 0.7 * (780.0 - wavelength[high_end]) / 80.0
    outside = (wavelength < 380.0) | (wavelength >= 781.0)
    falloff[outside] = 0.0

    return (np.stack([red, green, blue], axis=-1) * falloff[..., None]).astype(np.float32)


# Planck's law constants, SI.
_PLANCK_C1 = 3.741771852e-16  # 2 h c^2  [W m^2]
_PLANCK_C2 = 1.438776877e-2  # h c / k   [m K]


def blackbody_rgb(temperature_k: float, samples: int = 64) -> np.ndarray:
    """Linear-light RGB of a blackbody at ``temperature_k``, normalised to G = 1.

    Planck's law integrated against :func:`wavelength_to_rgb` across the
    visible band. Used to model **auto-white-balance error**: an AWB algorithm
    that guesses the illuminant's colour temperature wrong applies the
    correction for the temperature it guessed, so the residual cast is the
    *ratio* of two blackbody colours, not an arbitrary per-channel gain.

    That distinction is why this is worth integrating rather than faking.
    Real AWB error moves along the blue-orange axis because that is the axis
    illuminants actually vary on; independent random RGB gains wander off it
    and produce casts (strong green, strong magenta) that a white-balance
    algorithm would never leave behind.
    """
    wavelengths_m = np.linspace(380e-9, 780e-9, samples, dtype=np.float64)
    radiance = _PLANCK_C1 / (
        wavelengths_m**5 * np.expm1(_PLANCK_C2 / (wavelengths_m * max(temperature_k, 1.0)))
    )
    colours = wavelength_to_rgb((wavelengths_m * 1e9).astype(np.float32))
    rgb = (colours * radiance[:, None]).sum(axis=0)

    green = float(rgb[1])
    if green <= 1e-30:
        return np.ones(3, dtype=np.float32)
    return (rgb / green).astype(np.float32)
