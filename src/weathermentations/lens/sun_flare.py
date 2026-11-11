"""Physically-motivated compound lens flare.

Real lens flare is not a single glow blob but the sum of several distinct
optical phenomena, all keyed off the line through the light source and the
image (optical) center:

1. **Veiling glare** — a broad, low-intensity haze radiating from the light
   source, caused by diffuse scattering inside the lens.
2. **Halo** — a ring at a roughly fixed angular radius from the light,
   caused by reflection off a single lens element.
3. **Ghosts** — a chain of scaled, recolored copies of the aperture shape,
   caused by light bouncing between multiple internal lens elements before
   reaching the sensor. Because each internal reflection roughly mirrors the
   ray through the lens's optical axis, ghosts line up along the
   light-source-to-image-center line, including *past* the center on the
   opposite side.
4. **Diffraction streaks** — radial spikes from the iris diaphragm's edges
   (an optional "star" pattern).
5. **Dispersion ring** — a narrow rainbow annulus. Unlike the halo this is a
   *diffraction corona* whose radius scales with wavelength, so it separates
   into spectral bands rather than being one tinted ring.

Artefacts specific to surveillance cameras
------------------------------------------
The list above is what any lens does. A fixed CCTV camera adds its own, and
they are often the more recognisable half of the picture — footage is
identifiable as surveillance footage largely by these:

6. **Scratches on the dome or window.** Almost every fixed camera looks
   through an acrylic dome or a weather window, and those get scuffed by
   cleaning, grit and UV ageing. A scratch scatters light perpendicular to
   its own length, throwing a streak across the source. These are *not* the
   iris starburst: arbitrary angles, arbitrary count, unequal lengths, no
   symmetry.
7. **CCD vertical smear.** In an interline or frame-transfer CCD, a source
   bright enough to saturate a well leaks charge into the vertical shift
   register *during* transfer, smearing it uniformly down the entire column.
   The hard-edged full-height stripe through a streetlight is the signature
   artefact of older surveillance video. A CMOS sensor does not do this.
8. **IR-cut leakage.** A day/night camera swings its IR-cut filter out of the
   path at night, and silicon stays sensitive well past 700 nm where all
   three Bayer dyes leak — so hot sources bloom magenta-red, over a wider
   radius than the visible glare.
9. **Flat-window secondary ghost.** A plane protective window reflects a few
   percent at each face; the double bounce returns a faint displaced copy of
   the source. It does **not** lie on the source-to-centre line, because its
   offset comes from the window's thickness and tilt rather than from the
   optical axis — which is exactly how to tell it from a lens ghost.
10. **Onion-ring bokeh.** Concentric turning marks left on a moulded
    aspherical element by the diamond tool that cut its mould. Invisible in a
    focused image, they print straight through any defocused highlight, and
    they mark out the cheap moulded asphericals these lenses use.

Reference: Hullin et al. 2011, "Physically-Based Real-Time Lens Flare
Rendering" (Eurographics/SIGGRAPH). This is a stylized, real-time-friendly
approximation of that model, not a full wave-optics simulation.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from weathermentations import mode
from weathermentations.base import WeatherTransform
from weathermentations.diagnostics import record_draw
from weathermentations.utils.grids import centered_grid, pixel_grid, radial_distance
from weathermentations.utils.photometry import from_linear_unit, srgb_to_linear, to_linear_unit
from weathermentations.utils.spectral import (
    RGB_WAVELENGTHS_NM,
    ghost_tint,
    wavelength_to_rgb,
)


def polygon_distance(
    dy: np.ndarray,
    dx: np.ndarray,
    blades: int,
    rotation: float,
    scalar: Callable[[float], Any] = mode.scalar,
) -> np.ndarray:
    """Signed distance to a regular ``blades``-gon: 0 at centre, 1 on the edge.

    The maximum over the polygon's edge half-planes, which is what gives a
    defocused point its polygonal aperture shape. ``blades < 3`` falls back to
    a circle. ``dy``/``dx`` may be broadcastable axes rather than full grids.

    ``scalar`` decides the precision of the field: by default the mode's
    (float32 in fast mode, the reference code's float64 in exact mode).
    """
    if blades < 3:
        return np.sqrt(dx**2 + dy**2)
    angles = rotation + 2.0 * np.pi * np.arange(blades) / blades
    planes = [dx * scalar(np.cos(a)) + dy * scalar(np.sin(a)) for a in angles]
    return np.max(np.stack(planes, axis=0), axis=0) / scalar(np.cos(np.pi / blades))


def aperture_image(
    size: int, blades: int, rotation: float, edge_gain: float, onion_rings: float = 0.0
) -> np.ndarray:
    """``(size, size)`` image of the aperture: the shape a ghost actually takes.

    A ghost is an out-of-focus image of the iris opening, so it is a polygon
    with one side per aperture blade — not a soft circular blob. Near-field
    Fresnel diffraction at the iris edge also leaves the rim brighter than
    the middle with a couple of damped ripples inside it, which is why real
    ghosts read as crisp-edged discs with structure rather than as airbrushed
    dots.
    """
    dy, dx = centered_grid(size, scale=size / 2.0)
    dist = polygon_distance(dy, dx, blades, rotation)

    inside = np.clip(1.0 - dist, 0.0, 1.0)
    body = np.clip(inside / 0.06, 0.0, 1.0)

    # Diffraction rim plus two damped inward ripples.
    rim = np.exp(-(((dist - 0.93) / 0.07) ** 2))
    ripple = 0.25 * np.exp(-(((dist - 0.78) / 0.06) ** 2)) + 0.12 * np.exp(
        -(((dist - 0.62) / 0.06) ** 2)
    )

    # "Onion rings": concentric turning marks left on a moulded aspherical
    # element by the diamond tool that cut its mould. They are invisible in a
    # focused image but print straight through any defocused highlight, and
    # they are a signature of the cheap moulded asphericals in small
    # surveillance lenses — a premium ground element does not show them.
    onion = 1.0 + onion_rings * np.cos(2.0 * np.pi * dist * _ONION_RING_COUNT)

    return body * (0.35 + edge_gain * rim + ripple) * onion


# Resolution the diffraction pattern is computed at. The FFT is cached, so
# this is paid once per blade count for the life of the process.
_DIFFRACTION_RESOLUTION = 512

# Concentric turning rings across a moulded aspherical element's radius.
_ONION_RING_COUNT = 7.0

# Silicon's residual sensitivity past the visible band leaks through all three
# Bayer dyes, but least through green — so IR bloom reads magenta-red.
_IR_LEAK_COLOUR = np.array([255.0, 90.0, 150.0], dtype=np.float32)

# A scratch throws a streak only a couple of pixels wide, so the Gaussian is
# evaluated only within this many pixels of its axis — beyond about four sigma
# it contributes nothing a uint8 image can represent.
_SCRATCH_WIDTH_PX = 1.6
_SCRATCH_CUTOFF_PX = 4.0 * _SCRATCH_WIDTH_PX

# Exponent the diffraction power spectrum is raised to before use. See
# `diffraction_starburst` — the raw range core-to-spike is ~1e5:1.
_STARBURST_COMPRESSION = 0.2


@lru_cache(maxsize=8)
def diffraction_starburst(blades: int) -> np.ndarray:
    """Fraunhofer diffraction pattern of the aperture: the real starburst.

    A starburst is not a drawn cross — it is the **far-field diffraction
    pattern of the iris opening**, which is the squared magnitude of that
    opening's Fourier transform. Computing it that way rather than stamping
    ``|cos(N theta)|^90`` spikes gets three things right for free that a
    synthetic pattern has to fake, badly:

    * the spike count and orientation follow from the blade geometry (an even
      blade count gives N spikes because opposite edges are parallel and
      their spikes coincide; an odd count gives 2N),
    * each spike carries the **fringe structure** of the edge that threw it,
      fading in bright and dark bands rather than as a smooth taper,
    * the central core is a genuine Airy-like disc of the right relative size.

    Returned normalised to a peak of 1 and strongly compressed. The raw power
    spectrum spans five orders of magnitude between the central core and the
    spikes, which is faithful — in a real photograph you only *see* the
    spikes because the core is blown far past saturation while they sit just
    above the noise floor. Since this pattern is added to an unsaturated
    image rather than run through a sensor, that range is tone-mapped here
    instead. The compression is a display choice, not a claim about the
    optics; the *structure* — spike count, orientation and fringing — is the
    physics and is untouched by it.

    Cached per blade count: the pattern is rotation-independent here and is
    rotated at use, which is exact, since rotating an aperture rotates its
    transform by the same angle.
    """
    size = _DIFFRACTION_RESOLUTION
    dy, dx = centered_grid(size, scale=size / 2.0)
    # The aperture occupies a small part of the grid so the transform is
    # finely sampled — zero padding sets the resolution of the pattern.
    # Always in float64: the pattern is cached, and must not depend on the
    # mode the process was in when it was first asked for.
    dist = polygon_distance(dy * 4.0, dx * 4.0, blades, 0.0, scalar=np.float64)
    aperture = (dist <= 1.0).astype(np.float32)

    spectrum = np.fft.fftshift(np.fft.fft2(aperture))
    power = np.abs(spectrum) ** 2
    power /= max(float(power.max()), 1e-30)
    compressed = power**_STARBURST_COMPRESSION

    # Window the pattern to zero at the edge of its own grid. The true
    # diffraction pattern decays forever; the FFT grid truncates it, and
    # since the compressed pattern still has a broad low-level floor out
    # there, stamping it onto the canvas left a visible **rectangle** around
    # the source where that floor stopped abruptly. The window makes the
    # truncation graceful instead of a step.
    dy_n, dx_n = centered_grid(size, scale=size / 2.0)
    radius = np.sqrt(dx_n**2 + dy_n**2)
    # Flat across the region the spikes actually occupy, rolling off only
    # over the outer third — a window that starts falling at the centre
    # (a plain (1-r)^2, say) shortens the spikes, which is the one feature
    # worth preserving.
    ramp = np.clip((1.0 - radius) / 0.35, 0.0, 1.0)
    window = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep
    return (compressed * window).astype(np.float32)


def _add_starburst(
    canvas: np.ndarray,
    color: np.ndarray,
    source: tuple[float, float],
    size: int,
    blades: int,
    rotation: float,
) -> None:
    """Add the diffraction starburst to ``canvas``, centred on the source.

    Diffraction angle scales with wavelength, so the pattern is sampled at a
    different scale per channel: red diffracts furthest and blue least, which
    is why real starburst spikes fan out into colour along their length
    instead of staying white.
    """
    pattern = diffraction_starburst(max(blades, 3))

    half = size / 2.0
    x0 = max(0, int(np.floor(source[0] - half)))
    y0 = max(0, int(np.floor(source[1] - half)))
    x1 = min(canvas.shape[1], int(np.ceil(source[0] + half)))
    y1 = min(canvas.shape[0], int(np.ceil(source[1] + half)))
    if x1 <= x0 or y1 <= y0:
        return

    # Sample the cached pattern over just the part of the canvas it lands on.
    dy = (np.arange(y0, y1, dtype=np.float32) - source[1]).reshape(-1, 1)
    dx = (np.arange(x0, x1, dtype=np.float32) - source[0]).reshape(1, -1)
    cos_r, sin_r = mode.scalar(np.cos(rotation)), mode.scalar(np.sin(rotation))
    rot_x = dx * cos_r - dy * sin_r
    rot_y = dx * sin_r + dy * cos_r

    # rot_x/rot_y are still broadcastable axes; cv2.remap needs dense maps.
    shape = np.broadcast_shapes(rot_x.shape, rot_y.shape)
    rot_x = np.broadcast_to(rot_x, shape)
    rot_y = np.broadcast_to(rot_y, shape)

    centre = pattern.shape[0] / 2.0
    for channel in range(3):
        # Diffraction scales the pattern linearly with wavelength.
        scale = float(RGB_WAVELENGTHS_NM[channel] / RGB_WAVELENGTHS_NM[1])
        pixels_per_unit = centre / (half * scale)
        map_x = np.ascontiguousarray(rot_x * pixels_per_unit + centre, dtype=np.float32)
        map_y = np.ascontiguousarray(rot_y * pixels_per_unit + centre, dtype=np.float32)
        sampled = cv2.remap(
            pattern,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        canvas[y0:y1, x0:x1, channel] += sampled * color[channel]


# The ring's colour depends only on radius, so it tables into one dimension
# instead of costing a full-frame pass per wavelength sample.
_RING_LUT_SIZE = 2048
_RING_LUT_MAX_RADIUS = 1.6
_RING_SPECTRUM_SAMPLES = 48


def dispersion_ring_profile(ring_radius: float, ring_width: float) -> np.ndarray:
    """Radial colour profile of a diffraction corona, integrated over wavelength.

    The corona's angular radius scales linearly with wavelength, so each
    wavelength lands on its own circle and the ring spreads into a spectrum.
    Evaluating that at only the three sRGB primaries produces three separate
    red, green and blue rings with visible gaps — unmistakably not a rainbow —
    so the spectrum is genuinely integrated across the visible band, with
    :func:`~weathermentations.utils.spectral.wavelength_to_rgb` giving each
    sample its colour.

    Returned as a ``(_RING_LUT_SIZE, 3)`` lookup indexed by radius normalised
    to the image half-diagonal. Tabling it this way is what makes the
    integration affordable: the colour depends only on radius, so 48
    wavelengths cost 48 passes over a 2048-entry array instead of 48 passes
    over the whole frame — about 1 ms, against ~26 ms for the single
    full-frame gather that applies it.

    Deliberately **not** cached. Both parameters are drawn from continuous
    ranges per call, so a key built from them never repeats: measured over 40
    varied calls an ``lru_cache`` here returned a 0% hit rate, costing memory
    and implying a saving it could not deliver. Quantising the key hard enough
    to collide would have to be coarse enough to be visible. At 1 ms the build
    is not worth either.
    """
    radii = np.linspace(0.0, _RING_LUT_MAX_RADIUS, _RING_LUT_SIZE, dtype=np.float32)
    wavelengths = np.linspace(400.0, 700.0, _RING_SPECTRUM_SAMPLES, dtype=np.float32)
    colours = wavelength_to_rgb(wavelengths)

    profile = np.zeros((_RING_LUT_SIZE, 3), dtype=np.float32)
    reference = float(RGB_WAVELENGTHS_NM[1])
    width = max(ring_width, 1e-4)
    for wavelength, colour in zip(wavelengths, colours, strict=True):
        band_radius = ring_radius * float(wavelength) / reference
        band = np.exp(-((radii - band_radius) ** 2) / (2.0 * width**2))
        profile += band[:, None] * colour

    peak = float(profile.max())
    if peak > 1e-8:
        profile /= peak
    return profile


def _stretch_along(patch: np.ndarray, angle: float, factor: float) -> np.ndarray:
    """Stretch a square patch by ``factor`` along ``angle``, keeping its size.

    Astigmatism images an off-axis point as a short line rather than a point,
    so a ghost well away from the optical axis is visibly non-circular. Done
    as an affine warp about the patch centre, which keeps the paste geometry
    unchanged.
    """
    if factor <= 1.001:
        return patch
    size = patch.shape[0]
    centre = (size / 2.0 - 0.5, size / 2.0 - 0.5)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # Rotate to the axis, scale x, rotate back — composed into one matrix.
    rotate = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    scale = np.array([[factor, 0.0], [0.0, 1.0]], dtype=np.float32)
    linear = rotate @ scale @ rotate.T
    offset = np.array(centre, dtype=np.float32) - linear @ np.array(centre, dtype=np.float32)
    matrix = np.concatenate([linear, offset.reshape(2, 1)], axis=1)
    warped: np.ndarray = cv2.warpAffine(
        patch, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=0.0
    )
    return warped


def add_sensor_smear(
    canvas: np.ndarray,
    colour: np.ndarray,
    source: tuple[float, float],
    strength: float,
    width_px: float,
) -> None:
    """Vertical bloom column through a bright source — the classic CCD artefact.

    An interline or frame-transfer CCD shifts charge down the sensor through
    the same columns that collect it. Under a source bright enough to saturate
    a well, charge leaks into the vertical shift register *while the transfer
    is happening*, so it is smeared uniformly down the entire column rather
    than blooming into neighbouring pixels. The result is a hard-edged stripe
    running the full height of the frame through the light, of roughly
    constant brightness — the single most recognisable signature of older
    surveillance footage.

    It is full-height by construction (every row of that column is shifted
    past the source) and near-achromatic, since the leakage is a charge effect
    rather than an optical one — it takes only the mild tint of whatever
    saturated the well.
    """
    if strength <= 1e-4 or width_px <= 0.0:
        return

    width = canvas.shape[1]
    columns = np.arange(width, dtype=np.float32)
    profile = np.exp(-0.5 * ((columns - source[0]) / max(width_px, 0.4)) ** 2)

    # Constant down the column: the smear is not a glow around the source, it
    # is charge dragged through the shift register, so it does not fall off
    # with distance from the light.
    canvas += profile[None, :, None] * colour * strength


def add_scratch_streaks(
    canvas: np.ndarray,
    colour: np.ndarray,
    source: tuple[float, float],
    scratches: list[tuple[float, float, float]],
) -> None:
    """Light scattered along fine scratches on a dome or protective window.

    Almost every fixed CCTV camera looks through an acrylic dome or a flat
    weather window, and those get scuffed by cleaning, grit and UV ageing. A
    scratch is effectively a cylindrical scatterer: it spreads light in the
    plane **perpendicular to its own length**, throwing a streak across the
    source.

    These are visually distinct from the iris starburst and must not be
    confused with it. A starburst is symmetric and its spike count is fixed by
    the blade geometry; scratch streaks sit at arbitrary angles, come in
    arbitrary numbers, and have unequal lengths and brightnesses because no
    two scratches are alike. A frame showing a handful of unequal streaks at
    odd angles is showing scratches, not diffraction.

    ``scratches`` is a list of ``(angle_rad, length_px, strength)``.
    """
    if not scratches:
        return
    height, width = canvas.shape[:2]
    yy, xx = pixel_grid(height, width)
    dx = xx - source[0]
    dy = yy - source[1]

    for angle, length, strength in scratches:
        if strength <= 1e-4 or length <= 1.0:
            continue
        sin_a, cos_a = mode.scalar(np.sin(angle)), mode.scalar(np.cos(angle))

        # Distance across the streak's own axis. A scratch streak is only a
        # couple of pixels wide, so it is evaluated only in the thin band it
        # actually occupies: the band test is two multiplies per pixel,
        # against a transcendental over the whole frame for every scratch.
        across = -dx * sin_a + dy * cos_a
        band = np.abs(across) <= _SCRATCH_CUTOFF_PX
        if not band.any():
            continue

        along = (dx * cos_a + dy * sin_a) * np.ones_like(across)
        # Narrow across, long along, and fading with distance from the source.
        core = np.exp(-0.5 * (across[band] / _SCRATCH_WIDTH_PX) ** 2)
        falloff = np.exp(-np.abs(along[band]) / max(length, 1.0))
        canvas[band] += (core * falloff)[:, None] * colour * strength


def _paste_additive(
    canvas: np.ndarray, patch: np.ndarray, color: np.ndarray, cx: float, cy: float
) -> None:
    size = patch.shape[0]
    half = size / 2.0
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    h, w = canvas.shape[:2]
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(w, x0 + size), min(h, y0 + size)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)
    region = canvas[dst_y0:dst_y1, dst_x0:dst_x1]
    region += patch[src_y0:src_y1, src_x0:src_x1, None] * color


class SunFlare(WeatherTransform):
    """Compound, physically-motivated sun/light-source lens flare.

    Args:
        flare_position: ``(x, y)`` in normalized ``[0, 1]`` image coordinates,
            or ``None`` to pick a random position (biased toward the upper
            image, where light sources typically appear) each call.
        intensity_range: overall brightness multiplier for the whole flare.
        ghost_count_range: number of internal-reflection ghost copies.
        halo_radius_range: halo ring radius as a fraction of the image's
            half-diagonal.
        streak_probability: chance of rendering the iris-diffraction
            starburst.
        aperture_blades_range: number of aperture blades, drawn per call.
            Sets both the ghosts' polygonal shape and the starburst's spike
            count (N spikes for an even count, 2N for an odd one).
        dispersion_ring_probability: chance of rendering the rainbow
            diffraction ring.
        dispersion_ring_radius_range: its radius as a fraction of the image's
            half-diagonal.
        dispersion_ring_intensity_range: how strong the ring is. This is the
            vividly rainbow-edged annulus seen in strongly flared frames: a
            diffraction corona thrown by the aperture edge and by dust on the
            elements, whose radius scales *linearly with wavelength*, so it
            separates into spectral bands rather than being one tinted ring.
        onion_ring_range: depth of the concentric "onion ring" texture inside
            defocused highlights, left by the diamond-turned mould of a cheap
            aspherical element. A signature of small surveillance lenses; a
            ground element shows none.
        sensor_smear_probability: chance of rendering CCD vertical smear.
        sensor_smear_strength_range: how bright that column is.
        sensor_smear_width_range: its width in pixels. The classic full-height
            stripe through a bright light in older surveillance footage —
            charge leaking into the vertical shift register during transfer,
            so it is uniform down the whole column and hard-edged. Set the
            probability to 0 to model a CMOS camera, which smears differently
            (rolling-shutter skew) and not like this.
        scratch_count_range: number of scratch streaks from a scuffed dome or
            weather window. Distinct from the iris starburst: arbitrary
            angles, arbitrary count, unequal lengths — a scratch scatters
            light perpendicular to its own length, and no two scratches match.
        scratch_strength_range: how brightly those scratches scatter.
        ir_leak_probability: chance of rendering IR-cut leakage.
        ir_leak_strength_range: its strength. A day/night camera swings its
            IR-cut filter out at night and silicon stays sensitive past
            700 nm, so a hot source blooms magenta-red over a wider radius
            than the visible glare.
        window_ghost_probability: chance of a flat-window secondary ghost.
        window_ghost_offset_range: its displacement as a fraction of the
            half-diagonal. A plane protective window reflects a few percent at
            each face and returns a faint displaced copy of the source —
            **not** on the source-to-centre line, since the offset comes from
            the window's thickness and tilt rather than from the optical axis.
        lateral_chromatic_range: lateral chromatic aberration on the ghosts,
            as a fraction of a ghost's own size. The lens magnifies each
            wavelength slightly differently, so a ghost's channels land at
            different *positions* as well as different sizes — which is what
            puts warm colour on one edge of a ghost and cool on the other.
        chroma_shift_range: how far the red and blue channels of a ghost are
            scaled apart geometrically, which fringes its *edges* with colour.
        coating_wavelength_range: the wavelengths (nm) the lens's
            anti-reflection coatings are designed to cancel. A single-layer
            coating can only null one wavelength, so the residual reflection
            either side of it is strongly coloured — and since a ghost is a
            product of two such residuals, this is where its tint actually
            comes from. Real coatings cluster around the middle of the
            visible band, which is why flare ghosts run violet/amber/cyan
            rather than any hue at all.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        flare_position: tuple[float, float] | None
        intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 3.0)),
        ]
        ghost_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 12)),
        ]
        halo_radius_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        streak_probability: float = Field(ge=0.0, le=1.0)
        aperture_blades_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(3, 12)),
        ]
        dispersion_ring_probability: float = Field(ge=0.0, le=1.0)
        dispersion_ring_radius_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 1.2)),
        ]
        dispersion_ring_intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        lateral_chromatic_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        onion_ring_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        sensor_smear_probability: float = Field(ge=0.0, le=1.0)
        sensor_smear_strength_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        sensor_smear_width_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.3, 40.0)),
        ]
        scratch_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 24)),
        ]
        scratch_strength_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        ir_leak_probability: float = Field(ge=0.0, le=1.0)
        ir_leak_strength_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        window_ghost_probability: float = Field(ge=0.0, le=1.0)
        window_ghost_offset_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.5)),
        ]
        chroma_shift_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.15)),
        ]
        coating_wavelength_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(380.0, 750.0)),
        ]

    def __init__(
        self,
        flare_position: tuple[float, float] | None = None,
        intensity_range: tuple[float, float] = (0.8, 1.9),
        ghost_count_range: tuple[int, int] = (4, 10),
        halo_radius_range: tuple[float, float] = (0.15, 0.35),
        streak_probability: float = 0.75,
        aperture_blades_range: tuple[int, int] = (5, 9),
        dispersion_ring_probability: float = 0.45,
        dispersion_ring_radius_range: tuple[float, float] = (0.22, 0.6),
        dispersion_ring_intensity_range: tuple[float, float] = (0.10, 0.35),
        lateral_chromatic_range: tuple[float, float] = (0.05, 0.3),
        chroma_shift_range: tuple[float, float] = (0.01, 0.05),
        coating_wavelength_range: tuple[float, float] = (460.0, 620.0),
        onion_ring_range: tuple[float, float] = (0.0, 0.22),
        sensor_smear_probability: float = 0.25,
        sensor_smear_strength_range: tuple[float, float] = (0.05, 0.35),
        sensor_smear_width_range: tuple[float, float] = (1.5, 7.0),
        scratch_count_range: tuple[int, int] = (0, 7),
        scratch_strength_range: tuple[float, float] = (0.04, 0.3),
        ir_leak_probability: float = 0.3,
        ir_leak_strength_range: tuple[float, float] = (0.05, 0.3),
        window_ghost_probability: float = 0.35,
        window_ghost_offset_range: tuple[float, float] = (0.03, 0.14),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.flare_position = flare_position
        self.intensity_range = intensity_range
        self.ghost_count_range = ghost_count_range
        self.halo_radius_range = halo_radius_range
        self.streak_probability = streak_probability
        self.aperture_blades_range = aperture_blades_range
        self.dispersion_ring_probability = dispersion_ring_probability
        self.dispersion_ring_radius_range = dispersion_ring_radius_range
        self.dispersion_ring_intensity_range = dispersion_ring_intensity_range
        self.lateral_chromatic_range = lateral_chromatic_range
        self.onion_ring_range = onion_ring_range
        self.sensor_smear_probability = sensor_smear_probability
        self.sensor_smear_strength_range = sensor_smear_strength_range
        self.sensor_smear_width_range = sensor_smear_width_range
        self.scratch_count_range = scratch_count_range
        self.scratch_strength_range = scratch_strength_range
        self.ir_leak_probability = ir_leak_probability
        self.ir_leak_strength_range = ir_leak_strength_range
        self.window_ghost_probability = window_ghost_probability
        self.window_ghost_offset_range = window_ghost_offset_range
        self.chroma_shift_range = chroma_shift_range
        self.coating_wavelength_range = coating_wavelength_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        height, width = params["shape"][:2]

        if self.flare_position is not None:
            fx, fy = self.flare_position
        else:
            fx = self.py_random.uniform(0.0, 1.0)
            fy = self.py_random.uniform(0.0, 0.5)
        source = (fx * width, fy * height)
        center = (width / 2.0, height / 2.0)
        half_diag = float(np.hypot(width, height)) / 2.0

        intensity = self.py_random.uniform(*self.intensity_range)
        ghost_count = self.py_random.randint(*self.ghost_count_range)
        halo_radius = self.py_random.uniform(*self.halo_radius_range)
        draw_streaks = self.py_random.random() < self.streak_probability
        streak_angle = self.py_random.uniform(0, np.pi)

        # Angle of incidence at the internal surfaces grows with how far
        # off-axis the source sits, which blue-shifts the coatings' notch —
        # the reason ghosts from a source near the frame edge are tinted
        # differently from ones near the middle.
        offset = float(np.hypot(source[0] - center[0], source[1] - center[1]))
        incidence = 45.0 * min(1.0, offset / max(float(np.hypot(*center)), 1e-6))

        record_draw(
            intensity=intensity,
            ghost_count=ghost_count,
            halo_radius=halo_radius,
            incidence_deg=incidence,
            source_offset_frac=offset / max(float(np.hypot(*center)), 1e-6),
            draw_streaks=draw_streaks,
        )
        ghosts = []
        for _i in range(ghost_count):
            t = self.py_random.uniform(-0.4, 1.3)
            gx = center[0] + (source[0] - center[0]) * t
            gy = center[1] + (source[1] - center[1]) * t
            gsize = self.py_random.uniform(0.015, 0.085)
            # Each ghost is a bounce between two coated surfaces whose design
            # wavelengths differ, and higher-order ghosts bounce more times.
            # Both are drawn here; the resulting tint comes from thin-film
            # interference rather than from a random hue rotation, which is
            # why the palette stays in the narrow violet/amber/cyan band a
            # real coated lens produces instead of wandering the colour wheel.
            design_a = self.py_random.uniform(*self.coating_wavelength_range)
            design_b = self.py_random.uniform(*self.coating_wavelength_range)
            bounces = 2 * self.py_random.randint(1, 3)

            # No two ghosts in a real flare look alike, because each is formed
            # by a different pair of surfaces and so sits at a different
            # distance from focus with a different amount of aberration.
            # Three consequences, all visible in any real flare photograph:
            #
            #  - Some ghosts are filled discs and others are bright *rings*.
            #    A ghost close to focus images the iris as a solid patch; one
            #    far from focus concentrates its light at the rim, because
            #    spherical aberration throws the marginal rays outward.
            #  - The rings are the strongly coloured ones. Each bounce adds
            #    another pass of chromatic aberration, so the higher-order
            #    ghosts fringe hardest — this is where the rainbow-edged
            #    annulus in a real flare comes from.
            #  - Off-axis ghosts are *elongated*. Astigmatism stretches a
            #    point into a line whose orientation depends on the ray
            #    bundle's tilt, so the further along the axis a ghost sits,
            #    the less circular it is.
            edge_gain = self.py_random.uniform(0.3, 2.6)
            chroma = self.py_random.uniform(*self.chroma_shift_range) * (1.0 + 0.9 * bounces)
            elongation = 1.0 + self.py_random.uniform(0.0, 0.55) * abs(t)
            ghosts.append(
                (gx, gy, gsize, design_a, design_b, bounces, incidence,
                 edge_gain, chroma, elongation)
            )

        return {
            "source": source,
            "center": center,
            "intensity": intensity,
            "halo_radius": halo_radius,
            "ghosts": ghosts,
            "draw_streaks": draw_streaks,
            "streak_angle": streak_angle,
            "blade_rotation": self.py_random.uniform(0, 2 * np.pi),
            "chroma_shift": self.py_random.uniform(*self.chroma_shift_range),
            "glare_coating_nm": self.py_random.uniform(*self.coating_wavelength_range),
            "blades": self.py_random.randint(*self.aperture_blades_range),
            "ring_radius": self.py_random.uniform(*self.dispersion_ring_radius_range),
            # Width as a *fraction of the radius*, because that is what sets
            # how much the spectral bands overlap: the dispersion spreads the
            # ring over ~0.55x its radius across the visible band, so a ring
            # much narrower than that separates into hard primary arcs and one
            # much wider washes out to white. Real coronae sit in between.
            "ring_width": (
                self.py_random.uniform(*self.dispersion_ring_radius_range)
                * self.py_random.uniform(0.07, 0.18)
            ),
            "ring_intensity": (
                self.py_random.uniform(*self.dispersion_ring_intensity_range)
                if self.py_random.random() < self.dispersion_ring_probability
                else 0.0
            ),
            "lateral_ca": self.py_random.uniform(*self.lateral_chromatic_range),
            "onion_rings": self.py_random.uniform(*self.onion_ring_range),
            "smear_strength": (
                self.py_random.uniform(*self.sensor_smear_strength_range)
                if self.py_random.random() < self.sensor_smear_probability
                else 0.0
            ),
            "smear_width": self.py_random.uniform(*self.sensor_smear_width_range),
            "scratches": [
                (
                    self.py_random.uniform(0.0, np.pi),
                    self.py_random.uniform(0.05, 0.6) * half_diag,
                    self.py_random.uniform(*self.scratch_strength_range),
                )
                for _ in range(self.py_random.randint(*self.scratch_count_range))
            ],
            "ir_leak": (
                self.py_random.uniform(*self.ir_leak_strength_range)
                if self.py_random.random() < self.ir_leak_probability
                else 0.0
            ),
            "window_ghost_offset": (
                (
                    self.py_random.uniform(-1.0, 1.0)
                    * self.py_random.uniform(*self.window_ghost_offset_range)
                    * half_diag,
                    self.py_random.uniform(-1.0, 1.0)
                    * self.py_random.uniform(*self.window_ghost_offset_range)
                    * half_diag,
                )
                if self.py_random.random() < self.window_ghost_probability
                else None
            ),
        }

    def apply(
        self,
        img: np.ndarray,
        source: tuple[float, float],
        center: tuple[float, float],
        intensity: float,
        halo_radius: float,
        ghosts: list[
            tuple[float, float, float, float, float, int, float, float, float, float]
        ],
        draw_streaks: bool,
        streak_angle: float,
        blade_rotation: float,
        chroma_shift: float,
        glare_coating_nm: float,
        blades: int,
        ring_radius: float,
        ring_width: float,
        ring_intensity: float,
        lateral_ca: float,
        onion_rings: float,
        smear_strength: float,
        smear_width: float,
        scratches: list[tuple[float, float, float]],
        ir_leak: float,
        window_ghost_offset: tuple[float, float] | None,
        **params: Any,
    ) -> np.ndarray:
        height, width = img.shape[:2]
        warm_white = np.array([255.0, 244.0, 214.0])
        half_diag = float(np.hypot(width, height)) / 2.0

        # 1 & 2. Veiling glare (broad haze around the source) and the halo ring
        #    left by a reflection off a single element. Both are radial about
        #    the source, so the distance field is built once and both live in
        #    one greyscale field — a flare has one colour, so there is no
        #    reason to carry three channels through the accumulation.
        dist = radial_distance(height, width, *source)
        # Veiling glare is scattering off imperfections, dust and coating
        # roughness inside the barrel, so its point-spread function is the
        # same heavy-tailed Lorentzian family used for lens grime in
        # ``lens/dirt.py`` — a bright core with a wide 1/r^2 skirt. The
        # previous ``(1 - r/r_max)^3`` ramp had no tail at all and, worse,
        # hit exactly zero at the far corner, putting a visible edge on the
        # glare where physically there is only a slow falloff.
        core = 0.12 * half_diag
        glare = 1.0 / (1.0 + (dist / core) ** 2)
        # The halo is light reflected from a *curved* element, so it is
        # imaged well out of focus and arrives as a broad, soft annulus. At
        # the previous width (2% of the half-diagonal) it rendered as a
        # crisp drawn ring — the single most artificial-looking part of the
        # flare — and at 0.6 it outshone the source itself, which no
        # single-surface reflection ever does.
        halo_width = max(0.05, halo_radius * 0.45)
        halo = np.exp(-((dist / half_diag - halo_radius) ** 2) / (2 * halo_width**2))

        # Veiling glare is light scattered off every coated surface in the
        # barrel, so it carries the coatings' residual reflectance just as the
        # ghosts do — which is why the broad wash across a badly flared frame
        # is so often distinctly violet or amber rather than neutral white.
        # It is a single-pass scatter, not a two-bounce reflection, so the
        # tint is milder than a ghost's and is blended rather than applied
        # outright.
        glare_tint = ghost_tint(glare_coating_nm, glare_coating_nm, 0.0, bounces=2)
        veiling_colour = warm_white * (0.45 + 0.55 * glare_tint)

        canvas = (glare * 0.55)[..., None] * veiling_colour + (halo * 0.12)[..., None] * warm_white
        canvas = canvas.astype(np.float32)

        # 2b. Dispersion ring. The broad halo above is a reflection off a
        #     curved element and is essentially achromatic. The narrow,
        #     vividly rainbow-edged annulus in a real flare is a different
        #     thing: a **diffraction corona**, thrown by the aperture edge and
        #     by dust and coating texture on the elements. Its angular radius
        #     scales *linearly with wavelength*, so red lands on a visibly
        #     larger circle than blue and the ring separates into spectral
        #     bands.
        #
        #     That is why it has to be drawn per channel with a different
        #     radius rather than tinted: a single-radius ring multiplied by a
        #     colour is one coloured ring, which is not what dispersion looks
        #     like. It is also why the ring is narrow — the bands only
        #     resolve when the ring is thinner than the spread between them.
        if ring_intensity > 1e-3:
            profile = dispersion_ring_profile(ring_radius, ring_width)
            normalised = dist / half_diag
            index = np.clip(
                normalised * (_RING_LUT_SIZE / _RING_LUT_MAX_RADIUS), 0, _RING_LUT_SIZE - 1
            ).astype(np.int32)
            # The profile is normalised to a peak of 1, but the rest of the
            # canvas is accumulated on the same 0-255 scale as `warm_white`.
            canvas += profile[index] * (ring_intensity * float(warm_white.max()))

        # 3. Ghost chain along the source-center line. Each ghost is an image
        #    of the aperture, tinted by the residual reflectance of the lens
        #    coatings it bounced off, and split chromatically because that
        #    reflectance varies with wavelength: the three channels are drawn
        #    at slightly different scales, which is what gives real ghosts
        #    their coloured fringes.
        # Ghost brightness falls steeply with f-number (Hullin: I ~ f^-4) and
        # each bounce loses all but a couple of percent of the light, so
        # ghosts sit well below the source itself.
        ghost_intensity = 0.28
        # A ghost carries a fixed amount of light however far it is
        # defocused, so its surface brightness goes as 1/area. Giving every
        # ghost the same per-pixel intensity — the previous behaviour — makes
        # the large ones far too bright and is what turns them into flat
        # pasted discs; in a real flare the big ghosts are the faint ones.
        reference_size = 0.04
        axis_angle = float(np.arctan2(source[1] - center[1], source[0] - center[0]))
        for (
            gx, gy, gsize, design_a, design_b, bounces, incidence,
            edge_gain, ghost_chroma, elongation,
        ) in ghosts:
            size_px = max(8, int(round(gsize * half_diag * 2)))
            colour = warm_white * ghost_tint(design_a, design_b, incidence, bounces)
            # Each extra pair of bounces costs another factor of the coating's
            # residual reflectance, so the saturated high-order ghosts are
            # also the dim ones.
            order_falloff = 0.45 ** (bounces // 2 - 1)
            area_falloff = (reference_size / max(gsize, 1e-6)) ** 2
            brightness = ghost_intensity * order_falloff * min(1.0, area_falloff)
            spread = chroma_shift + ghost_chroma
            for channel, chroma in enumerate((1.0 + spread, 1.0, 1.0 - spread)):
                scaled = max(8, int(round(size_px * chroma)))
                patch = aperture_image(
                    scaled, blades, blade_rotation, edge_gain=edge_gain, onion_rings=onion_rings
                )
                patch = _stretch_along(patch, axis_angle, elongation)

                # Lateral chromatic aberration: the lens focuses each
                # wavelength at a slightly different magnification, so an
                # off-axis ghost's channels land at slightly different
                # *positions* along the flare axis as well as at different
                # sizes. Scaling alone only fringes a ghost's rim
                # symmetrically; the positional shift is what puts warm
                # colour on one edge and cool on the other, which is how real
                # ghosts read.
                shift = (chroma - 1.0) * lateral_ca * size_px
                offset_x = gx + shift * np.cos(axis_angle)
                offset_y = gy + shift * np.sin(axis_angle)

                channel_colour = np.zeros(3, dtype=np.float32)
                # Each channel is pasted separately, so scale by 1/3 to keep the
                # ghost's total brightness independent of the chromatic split.
                channel_colour[channel] = colour[channel] * brightness / 3.0
                _paste_additive(canvas, patch, channel_colour, offset_x, offset_y)

        # 4. Iris-diffraction starburst through the source.
        if draw_streaks:
            burst_size = max(16, int(round(half_diag * 1.6)) | 1)
            _add_starburst(
                canvas,
                warm_white * 0.30,
                source,
                burst_size,
                blades,
                streak_angle,
            )

        # 5. Artefacts specific to how surveillance cameras are *built*, as
        #    opposed to the lens optics above. A CCTV camera looks through a
        #    weatherproof dome or window it does not control the cleanliness
        #    of, often runs an older CCD sensor, and switches its IR-cut
        #    filter out at night — and each of those leaves its own signature.

        # 5a. Scratches on the dome or window. A scratch is a cylindrical
        #     scatterer, throwing light perpendicular to its own length.
        add_scratch_streaks(canvas, warm_white * 0.35, source, scratches)

        # 5b. IR-cut leakage. A day/night camera swings its IR-cut filter out
        #     of the path at night, and silicon stays sensitive well past
        #     700 nm — so a hot source blooms in a band the Bayer filters all
        #     leak in, which reads as the magenta-red halo familiar from night
        #     surveillance stills. Rendered as its own broad glow rather than
        #     as a tint, because it has a different *extent* from the visible
        #     glare: IR scatters further.
        if ir_leak > 1e-3:
            ir_glow = 1.0 / (1.0 + (dist / (core * 2.2)) ** 2)
            canvas += ir_glow[..., None] * _IR_LEAK_COLOUR * ir_leak

        # 5c. Flat-window secondary ghost. A plane protective window reflects
        #     a few percent at each face, and the double bounce returns a faint
        #     displaced copy of the source. Unlike a lens ghost it does **not**
        #     lie on the source-to-centre line: its offset comes from the
        #     window's thickness and tilt, so it sits at an arbitrary angle
        #     close to the source.
        if window_ghost_offset is not None:
            ghost_x = source[0] + window_ghost_offset[0]
            ghost_y = source[1] + window_ghost_offset[1]
            secondary = radial_distance(height, width, ghost_x, ghost_y)
            canvas += (
                np.exp(-((secondary / (core * 0.45)) ** 2))[..., None] * warm_white * 0.22
            )

        # 5d. CCD vertical smear, applied last so it is not softened by
        #     anything else — the real artefact has hard edges because it is a
        #     charge-transfer effect, not an optical one.
        add_sensor_smear(canvas, warm_white * 0.85, source, smear_strength, smear_width)

        canvas *= intensity

        # Flare is *added radiance*, and radiance only adds linearly. Summing
        # it onto gamma-encoded sRGB values lifts the mid-tones far too much
        # and compresses the highlights far too little, which is what makes
        # naive flare look like a translucent white decal laid over the photo
        # instead of light flooding the lens.
        linear, scale, is_integer = to_linear_unit(img)
        flare_linear = srgb_to_linear(np.clip(canvas / 255.0, 0.0, 1.0))
        return from_linear_unit(linear + flare_linear, scale, is_integer, img.dtype)

