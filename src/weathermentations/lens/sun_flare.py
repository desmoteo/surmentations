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

Reference: Hullin et al. 2011, "Physically-Based Real-Time Lens Flare
Rendering" (Eurographics/SIGGRAPH). This is a stylized, real-time-friendly
approximation of that model, not a full wave-optics simulation.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from weathermentations.base import WeatherTransform
from weathermentations.utils.grids import centered_grid, radial_distance
from weathermentations.utils.photometry import from_linear_unit, srgb_to_linear, to_linear_unit


def polygon_distance(
    dy: np.ndarray, dx: np.ndarray, blades: int, rotation: float
) -> np.ndarray:
    """Signed distance to a regular ``blades``-gon: 0 at centre, 1 on the edge.

    The maximum over the polygon's edge half-planes, which is what gives a
    defocused point its polygonal aperture shape. ``blades < 3`` falls back to
    a circle. ``dy``/``dx`` may be broadcastable axes rather than full grids.
    """
    if blades < 3:
        return np.sqrt(dx**2 + dy**2)
    angles = rotation + 2.0 * np.pi * np.arange(blades) / blades
    return np.max(
        np.stack([dx * np.cos(a) + dy * np.sin(a) for a in angles], axis=0), axis=0
    ) / np.cos(np.pi / blades)


def aperture_image(size: int, blades: int, rotation: float, edge_gain: float) -> np.ndarray:
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
    return body * (0.35 + edge_gain * rim + ripple)


def starburst(size: int, blades: int, rotation: float) -> np.ndarray:
    """Iris diffraction spikes. Even blade counts give N spikes, odd give 2N.

    Light diffracting off each straight blade edge throws a spike
    perpendicular to it. With an even number of blades opposite edges are
    parallel and their spikes coincide, so you see N; with an odd number they
    do not, and you see 2N.

    """
    dy, dx = centered_grid(size, scale=size / 2.0)
    return _starburst_field(dy, dx, blades, rotation)


def _starburst_field(dy: np.ndarray, dx: np.ndarray, blades: int, rotation: float) -> np.ndarray:
    """Starburst evaluated on arbitrary (broadcastable) normalized coordinates.

    Separated out so the pattern can be evaluated only over the part of the
    canvas it actually lands on. The burst patch is 1.6x the half-diagonal
    across and is centred on the light source, which usually sits near a
    frame edge, so most of that patch falls outside the image — and
    ``arctan2`` plus a 90th power over a million-odd throwaway pixels was the
    single most expensive step in the whole flare.
    """
    spikes = blades if blades % 2 == 0 else 2 * blades
    radius = np.sqrt(dx**2 + dy**2)
    # Outside the unit disc the radial term is exactly zero, so only the disc
    # needs the expensive angular term.
    radial = np.exp(-radius * 3.5) * np.clip(1.0 - radius, 0.0, 1.0)
    theta = np.arctan2(*np.broadcast_arrays(dy, dx))
    angular = np.abs(np.cos(spikes * (theta - rotation) / 2.0)) ** 90
    return angular * radial


def _add_starburst(
    canvas: np.ndarray,
    color: np.ndarray,
    source: tuple[float, float],
    size: int,
    blades: int,
    rotation: float,
) -> None:
    """Add the starburst to ``canvas``, evaluating only where it lands on it."""
    half = size / 2.0
    x0 = max(0, int(np.floor(source[0] - half)))
    y0 = max(0, int(np.floor(source[1] - half)))
    x1 = min(canvas.shape[1], int(np.ceil(source[0] + half)))
    y1 = min(canvas.shape[0], int(np.ceil(source[1] + half)))
    if x1 <= x0 or y1 <= y0:
        return

    dy = (np.arange(y0, y1, dtype=np.float32) - source[1]).reshape(-1, 1) / half
    dx = (np.arange(x0, x1, dtype=np.float32) - source[0]).reshape(1, -1) / half
    burst = _starburst_field(dy, dx, blades, rotation)
    canvas[y0:y1, x0:x1] += burst[..., None] * color


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
        aperture_blades: number of aperture blades. Sets both the ghosts'
            polygonal shape and the starburst's spike count (N spikes for an
            even count, 2N for an odd one). 0 gives circular ghosts.
        chroma_shift_range: how far the red and blue channels of a ghost are
            scaled apart, from the wavelength dependence of the coating
            reflectance — this is what fringes ghosts with colour.

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
        aperture_blades: int = Field(ge=0, le=12)
        chroma_shift_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.15)),
        ]

    def __init__(
        self,
        flare_position: tuple[float, float] | None = None,
        intensity_range: tuple[float, float] = (0.5, 1.2),
        ghost_count_range: tuple[int, int] = (3, 7),
        halo_radius_range: tuple[float, float] = (0.15, 0.35),
        streak_probability: float = 0.5,
        aperture_blades: int = 7,
        chroma_shift_range: tuple[float, float] = (0.01, 0.05),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.flare_position = flare_position
        self.intensity_range = intensity_range
        self.ghost_count_range = ghost_count_range
        self.halo_radius_range = halo_radius_range
        self.streak_probability = streak_probability
        self.aperture_blades = aperture_blades
        self.chroma_shift_range = chroma_shift_range

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

        intensity = self.py_random.uniform(*self.intensity_range)
        ghost_count = self.py_random.randint(*self.ghost_count_range)
        halo_radius = self.py_random.uniform(*self.halo_radius_range)
        draw_streaks = self.py_random.random() < self.streak_probability
        streak_angle = self.py_random.uniform(0, np.pi)

        ghosts = []
        for _i in range(ghost_count):
            t = self.py_random.uniform(-0.4, 1.3)
            gx = center[0] + (source[0] - center[0]) * t
            gy = center[1] + (source[1] - center[1]) * t
            gsize = self.py_random.uniform(0.015, 0.085)
            hue_shift = self.py_random.uniform(-40, 40)
            ghosts.append((gx, gy, gsize, hue_shift))

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
        }

    def apply(
        self,
        img: np.ndarray,
        source: tuple[float, float],
        center: tuple[float, float],
        intensity: float,
        halo_radius: float,
        ghosts: list[tuple[float, float, float, float]],
        draw_streaks: bool,
        streak_angle: float,
        blade_rotation: float,
        chroma_shift: float,
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
        corners = ((0, 0), (width, 0), (0, height), (width, height))
        max_dist = max(
            float(np.hypot(source[0] - cx, source[1] - cy)) for cx, cy in corners
        )
        glare = np.clip(1.0 - dist / max(max_dist, 1e-6), 0.0, 1.0) ** 3
        halo = np.exp(-((dist / half_diag - halo_radius) ** 2) / (2 * 0.02**2))
        canvas = (glare * 0.5 + halo * 0.6)[..., None] * warm_white
        canvas = canvas.astype(np.float32)

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
        for gx, gy, gsize, hue_shift in ghosts:
            size_px = max(8, int(round(gsize * half_diag * 2)))
            colour = _hue_shifted(warm_white, hue_shift)
            for channel, chroma in enumerate((1.0 + chroma_shift, 1.0, 1.0 - chroma_shift)):
                scaled = max(8, int(round(size_px * chroma)))
                patch = aperture_image(scaled, self.aperture_blades, blade_rotation, edge_gain=1.1)
                channel_colour = np.zeros(3, dtype=np.float32)
                # Each channel is pasted separately, so scale by 1/3 to keep the
                # ghost's total brightness independent of the chromatic split.
                channel_colour[channel] = colour[channel] * ghost_intensity / 3.0
                _paste_additive(canvas, patch, channel_colour, gx, gy)

        # 4. Iris-diffraction starburst through the source.
        if draw_streaks:
            burst_size = max(16, int(round(half_diag * 1.6)) | 1)
            _add_starburst(
                canvas,
                warm_white * 0.55,
                source,
                burst_size,
                self.aperture_blades,
                streak_angle,
            )

        canvas *= intensity

        # Flare is *added radiance*, and radiance only adds linearly. Summing
        # it onto gamma-encoded sRGB values lifts the mid-tones far too much
        # and compresses the highlights far too little, which is what makes
        # naive flare look like a translucent white decal laid over the photo
        # instead of light flooding the lens.
        linear, scale, is_integer = to_linear_unit(img)
        flare_linear = srgb_to_linear(np.clip(canvas / 255.0, 0.0, 1.0))
        return from_linear_unit(linear + flare_linear, scale, is_integer, img.dtype)


def _hue_shifted(rgb: np.ndarray, hue_shift_degrees: float) -> np.ndarray:
    hsv = cv2.cvtColor(rgb.reshape(1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + hue_shift_degrees) % 360
    rgb_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255.0
    return rgb_shifted.reshape(3)
