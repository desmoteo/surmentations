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
from weathermentations.utils.compositing import normalized_radial_falloff


def _soft_circle(size: int, softness: float = 0.5) -> np.ndarray:
    """``(size, size)`` float32 radial gradient, 1 at center, feathered edge."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    center = size / 2.0
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) / center
    return np.clip(1.0 - dist, 0.0, 1.0) ** (1.0 / max(softness, 1e-3))


def _ring(size: int, radius_frac: float, thickness_frac: float) -> np.ndarray:
    """``(size, size)`` float32 ring mask at ``radius_frac`` of the half-size."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    center = size / 2.0
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) / center
    return np.exp(-((dist - radius_frac) ** 2) / (2 * thickness_frac**2))


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
        streak_probability: chance of rendering iris-diffraction streaks.

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

    def __init__(
        self,
        flare_position: tuple[float, float] | None = None,
        intensity_range: tuple[float, float] = (0.5, 1.2),
        ghost_count_range: tuple[int, int] = (3, 7),
        halo_radius_range: tuple[float, float] = (0.15, 0.35),
        streak_probability: float = 0.5,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.flare_position = flare_position
        self.intensity_range = intensity_range
        self.ghost_count_range = ghost_count_range
        self.halo_radius_range = halo_radius_range
        self.streak_probability = streak_probability

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
            gsize = self.py_random.uniform(0.03, 0.18)
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
        **params: Any,
    ) -> np.ndarray:
        height, width = img.shape[:2]
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        warm_white = np.array([255.0, 244.0, 214.0])

        # 1. Veiling glare: broad haze around the source.
        glare = normalized_radial_falloff(height, width, *source) ** 3
        canvas += glare[..., None] * warm_white * 0.5

        # 2. Halo ring, sized relative to the half-diagonal.
        half_diag = float(np.hypot(width, height)) / 2.0
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        dist_from_source = np.sqrt((xx - source[0]) ** 2 + (yy - source[1]) ** 2) / half_diag
        halo = np.exp(-((dist_from_source - halo_radius) ** 2) / (2 * 0.02**2))
        canvas += halo[..., None] * warm_white * 0.6

        # 3. Ghost chain along the source-center line.
        for gx, gy, gsize, hue_shift in ghosts:
            size_px = max(8, int(round(gsize * half_diag * 2)))
            patch = _soft_circle(size_px, softness=0.6)
            color = _hue_shifted(warm_white, hue_shift)
            _paste_additive(canvas, patch, color, gx, gy)

        # 4. Optional iris-diffraction streaks through the source.
        if draw_streaks:
            streak_len = half_diag * 0.9
            for angle_offset in (streak_angle, streak_angle + np.pi / 2):
                dx, dy = np.cos(angle_offset), np.sin(angle_offset)
                pt1 = (int(source[0] - dx * streak_len), int(source[1] - dy * streak_len))
                pt2 = (int(source[0] + dx * streak_len), int(source[1] + dy * streak_len))
                streak_layer = np.zeros((height, width), dtype=np.float32)
                cv2.line(streak_layer, pt1, pt2, 1.0, thickness=2, lineType=cv2.LINE_AA)
                streak_blurred: np.ndarray = cv2.GaussianBlur(streak_layer, (0, 0), sigmaX=3.0)
                canvas += streak_blurred[..., None] * warm_white * 0.35

        canvas *= intensity
        out = img.astype(np.float32) + canvas
        if np.issubdtype(img.dtype, np.integer):
            return np.clip(out, 0, 255).astype(img.dtype)
        return np.clip(out, 0.0, 1.0).astype(img.dtype)


def _hue_shifted(rgb: np.ndarray, hue_shift_degrees: float) -> np.ndarray:
    hsv = cv2.cvtColor(rgb.reshape(1, 1, 3).astype(np.float32) / 255.0, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + hue_shift_degrees) % 360
    rgb_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255.0
    return rgb_shifted.reshape(3)
