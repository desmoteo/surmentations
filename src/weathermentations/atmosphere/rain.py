"""Depth-conditioned rain: streaks, a distant rain-veil haze, and optional ground wetness.

Physical model, following Garg & Nayar ("Vision and Rain", IJCV 2007;
"Photorealistic Rendering of Rain Streaks", SIGGRAPH 2006):

- A falling raindrop, imaged during a finite camera exposure, streaks into a
  thin, elongated, motion-blurred highlight rather than a sharp drop shape.
  We render each streak as an anti-aliased line at a near-vertical slant
  (adjustable for wind), then apply a directional blur along the streak's
  own axis to approximate exposure-time motion blur.
- Streak apparent size depends on distance: nearer drops project to longer,
  wider, more blurred streaks; farther drops shrink and, in aggregate,
  scatter into a soft haze rather than resolving as individual streaks —
  the same Koschmieder atmospheric-scattering law used for fog
  (``atmosphere/fog.py``), applied here with rain-specific coefficients and
  a rain-veil color instead of general airlight.
- Wet ground is optically different from dry ground: standing water adds a
  specular, mirror-like reflection of what's above it (blurred by surface
  ripples) on top of a slightly darkened diffuse base. We approximate this
  by vertically reflecting a band of the image just above the "ground" band,
  blurring it, and blending it in.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from weathermentations.base import WeatherTransform, require_depth_provider
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.depth.scale import to_pseudo_metric
from weathermentations.utils.compositing import directional_blur_kernel, feathered_band_blend

_RAIN_VEIL_COLOR = np.array([185.0, 190.0, 200.0])


class RandomRain(WeatherTransform):
    """Physically-grounded rain: depth-scaled streaks, rain-veil haze, optional wet ground.

    Args:
        depth_provider: shared :class:`~weathermentations.depth.provider.SharedDepthProvider`.
        streak_density_range: number of streaks to render.
        streak_length_range: streak length in pixels *at the nearest depth*;
            farther streaks are proportionally shorter (see module docstring).
        angle_range: slant angle in degrees from vertical (wind); negative
            leans left, positive leans right.
        brightness_range: streak highlight alpha/brightness.
        veil_beta_range: Koschmieder extinction coefficient (1/meters) for
            the distant rain-veil haze layer.
        wetness_probability: chance of rendering the ground-wetness pass.
        wetness_band_fraction: fraction of the image height (from the
            bottom) treated as "ground" for the wetness pass.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        depth_provider: SharedDepthProvider
        streak_density_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 2000)),
        ]
        streak_length_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(2.0, 200.0)),
        ]
        angle_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-45.0, 45.0)),
        ]
        brightness_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        veil_beta_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        wetness_probability: float
        wetness_band_fraction: float = Field(ge=0.0, le=1.0)

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        streak_density_range: tuple[int, int] = (150, 400),
        streak_length_range: tuple[float, float] = (10.0, 35.0),
        angle_range: tuple[float, float] = (-8.0, 8.0),
        brightness_range: tuple[float, float] = (0.3, 0.6),
        veil_beta_range: tuple[float, float] = (0.01, 0.05),
        wetness_probability: float = 0.4,
        wetness_band_fraction: float = 0.2,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.streak_density_range = streak_density_range
        self.streak_length_range = streak_length_range
        self.angle_range = angle_range
        self.brightness_range = brightness_range
        self.veil_beta_range = veil_beta_range
        self.wetness_probability = wetness_probability
        self.wetness_band_fraction = wetness_band_fraction

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        image = data["image"]
        height, width = params["shape"][:2]

        relative_depth = self.depth_provider.get_depth(image)
        depth_pm = to_pseudo_metric(relative_depth, None)

        count = self.py_random.randint(*self.streak_density_range)
        angle = self.py_random.uniform(*self.angle_range)
        brightness = self.py_random.uniform(*self.brightness_range)
        veil_beta = self.py_random.uniform(*self.veil_beta_range)
        base_length = self.py_random.uniform(*self.streak_length_range)

        streaks = []
        for _ in range(count):
            x = self.py_random.uniform(0, width)
            y = self.py_random.uniform(0, height)
            px = int(min(width - 1, max(0, x)))
            py = int(min(height - 1, max(0, y)))
            near_depth = relative_depth[py, px]
            depth_factor = max(0.15, 1.0 - near_depth)  # nearer (small relative_depth) -> bigger
            streaks.append((x, y, base_length * depth_factor, max(1, int(round(2 * depth_factor)))))

        draw_wetness = self.py_random.random() < self.wetness_probability

        return {
            "streaks": streaks,
            "angle": angle,
            "brightness": brightness,
            "veil_transmission": np.exp(-veil_beta * depth_pm).astype(np.float32),
            "draw_wetness": draw_wetness,
        }

    def apply(
        self,
        img: np.ndarray,
        streaks: list[tuple[float, float, float, int]],
        angle: float,
        brightness: float,
        veil_transmission: np.ndarray,
        draw_wetness: bool,
        **params: Any,
    ) -> np.ndarray:
        height, width = img.shape[:2]
        angle_rad = np.deg2rad(angle)

        streak_layer = np.zeros((height, width), dtype=np.float32)
        for x, y, length, thickness in streaks:
            x2 = x + length * np.sin(angle_rad)
            y2 = y + length * np.cos(angle_rad)
            cv2.line(
                streak_layer,
                (int(round(x)), int(round(y))),
                (int(round(x2)), int(round(y2))),
                1.0,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

        if streaks:
            avg_length = float(np.mean([s[2] for s in streaks]))
            kernel = directional_blur_kernel(max(3, int(avg_length * 0.4)), angle)
            blurred_streaks: np.ndarray = cv2.filter2D(streak_layer, -1, kernel)
            streak_layer = blurred_streaks

        img_f = img.astype(np.float32)

        # Rain-veil haze: same Koschmieder compositing as fog, with a cool gray-blue tint.
        t = veil_transmission[..., None]
        img_f = img_f * t + _RAIN_VEIL_COLOR * (1.0 - t)

        # Streak highlights, alpha-composited on top of the veiled image.
        alpha = np.clip(streak_layer * brightness, 0.0, 1.0)[..., None]
        streak_color = np.array([210.0, 215.0, 220.0])
        img_f = img_f * (1.0 - alpha) + streak_color * alpha

        if draw_wetness:
            img_f = _apply_ground_wetness(img_f, self.wetness_band_fraction)

        if np.issubdtype(img.dtype, np.integer):
            return np.clip(img_f, 0, 255).astype(img.dtype)
        return np.clip(img_f, 0.0, 1.0).astype(img.dtype)


def _apply_ground_wetness(img_f: np.ndarray, band_fraction: float) -> np.ndarray:
    height, width = img_f.shape[:2]
    band_start = int(height * (1.0 - band_fraction))
    if band_start >= height:
        return img_f

    band_height = height - band_start
    reflect_start = max(0, band_start - band_height)
    reflection_source = img_f[reflect_start:band_start]
    if reflection_source.shape[0] == 0:
        return img_f
    reflection = reflection_source[::-1]
    if reflection.shape[0] != band_height:
        reflection = cv2.resize(reflection, (width, band_height), interpolation=cv2.INTER_LINEAR)
    reflection = cv2.GaussianBlur(reflection, (0, 0), sigmaX=3.0, sigmaY=1.0)

    def blend(ground: np.ndarray) -> np.ndarray:
        darkened = ground * 0.85
        return darkened * 0.6 + reflection * 0.4

    return feathered_band_blend(img_f, band_fraction, blend)
