"""Depth-conditioned snow: falling flakes plus a heuristic ground-accumulation overlay.

Physical model for the falling flakes mirrors ``atmosphere/rain.py``: each
flake is a small soft sprite whose apparent size and motion-blur elongation
scale with distance (nearer flakes are bigger and more visibly blurred by
their fall; farther flakes shrink toward a diffuse haze), and flakes drift at
a slant angle (wind) with a much smaller motion-blur elongation than rain
streaks, since snow falls roughly 3-5x slower than rain.

Ground accumulation is **not** modeled from first principles the way fog,
rain-veil, and the flakes themselves are: how much snow settles on a surface
actually depends on that surface's orientation and material (needs real
surface normals, which this library doesn't estimate), not just depth. We
render it as an explicitly-heuristic brightness/desaturation overlay on the
image's lower band — documented here as a heuristic, not a physical claim,
per the plan's validation strategy.
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
from weathermentations.utils.compositing import directional_blur_kernel, feathered_band_blend

_SNOW_COLOR = np.array([245.0, 248.0, 252.0])
_ACCUMULATION_COLOR = np.array([235.0, 240.0, 245.0])


class RandomSnow(WeatherTransform):
    """Depth-scaled falling snowflakes with an optional, explicitly-heuristic
    ground-accumulation overlay.

    Args:
        depth_provider: shared :class:`~weathermentations.depth.provider.SharedDepthProvider`.
        flake_density_range: number of flakes to render.
        flake_size_range: flake radius in pixels *at the nearest depth*;
            farther flakes are proportionally smaller (see module docstring).
        fall_angle_range: slant angle in degrees from vertical (wind).
        motion_blur_range: motion-blur kernel length as a fraction of a
            flake's own size (0 = crisp round flake, higher = more streaked).
        accumulation_probability: chance of rendering the ground-accumulation
            overlay.
        accumulation_strength_range: how strongly the ground band is pushed
            toward the accumulation color.
        accumulation_band_fraction: fraction of the image height (from the
            bottom) treated as "ground" for the accumulation pass.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        depth_provider: SharedDepthProvider
        flake_density_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 3000)),
        ]
        flake_size_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 60.0)),
        ]
        fall_angle_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-45.0, 45.0)),
        ]
        motion_blur_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        accumulation_probability: float = Field(ge=0.0, le=1.0)
        accumulation_strength_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        accumulation_band_fraction: float = Field(ge=0.0, le=1.0)

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        flake_density_range: tuple[int, int] = (200, 600),
        flake_size_range: tuple[float, float] = (2.0, 9.0),
        fall_angle_range: tuple[float, float] = (-5.0, 5.0),
        motion_blur_range: tuple[float, float] = (0.1, 0.4),
        accumulation_probability: float = 0.4,
        accumulation_strength_range: tuple[float, float] = (0.2, 0.5),
        accumulation_band_fraction: float = 0.15,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.flake_density_range = flake_density_range
        self.flake_size_range = flake_size_range
        self.fall_angle_range = fall_angle_range
        self.motion_blur_range = motion_blur_range
        self.accumulation_probability = accumulation_probability
        self.accumulation_strength_range = accumulation_strength_range
        self.accumulation_band_fraction = accumulation_band_fraction

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        image = data["image"]
        height, width = params["shape"][:2]

        relative_depth = self.depth_provider.get_depth(image)

        count = self.py_random.randint(*self.flake_density_range)
        angle = self.py_random.uniform(*self.fall_angle_range)
        motion_blur_amount = self.py_random.uniform(*self.motion_blur_range)
        base_size = self.py_random.uniform(*self.flake_size_range)

        flakes = []
        for _ in range(count):
            x = self.py_random.uniform(0, width)
            y = self.py_random.uniform(0, height)
            px = int(min(width - 1, max(0, x)))
            py = int(min(height - 1, max(0, y)))
            near_depth = relative_depth[py, px]
            depth_factor = max(0.2, 1.0 - near_depth)  # nearer (small relative_depth) -> bigger
            flakes.append((x, y, base_size * depth_factor))

        draw_accumulation = self.py_random.random() < self.accumulation_probability
        accumulation_strength = (
            self.py_random.uniform(*self.accumulation_strength_range) if draw_accumulation else 0.0
        )

        return {
            "flakes": flakes,
            "angle": angle,
            "motion_blur_amount": motion_blur_amount,
            "draw_accumulation": draw_accumulation,
            "accumulation_strength": accumulation_strength,
        }

    def apply(
        self,
        img: np.ndarray,
        flakes: list[tuple[float, float, float]],
        angle: float,
        motion_blur_amount: float,
        draw_accumulation: bool,
        accumulation_strength: float,
        **params: Any,
    ) -> np.ndarray:
        height, width = img.shape[:2]

        flake_layer = np.zeros((height, width), dtype=np.float32)
        for x, y, size in flakes:
            radius = max(1, int(round(size / 2)))
            cv2.circle(flake_layer, (int(round(x)), int(round(y))), radius, 1.0, thickness=-1)

        if flakes:
            avg_size = float(np.mean([f[2] for f in flakes]))
            blur_length = max(3, int(avg_size * (1.0 + motion_blur_amount * 3)))
            kernel = directional_blur_kernel(blur_length, angle)
            blurred: np.ndarray = cv2.filter2D(flake_layer, -1, kernel)
            flake_layer = np.clip(blurred * (1.0 + motion_blur_amount), 0.0, 1.0)

        img_f = img.astype(np.float32)

        alpha = flake_layer[..., None]
        img_f = img_f * (1.0 - alpha) + _SNOW_COLOR * alpha

        if draw_accumulation:
            def blend(band: np.ndarray) -> np.ndarray:
                keep = 1.0 - accumulation_strength
                return band * keep + _ACCUMULATION_COLOR * accumulation_strength

            img_f = feathered_band_blend(img_f, self.accumulation_band_fraction, blend)

        if np.issubdtype(img.dtype, np.integer):
            return np.clip(img_f, 0, 255).astype(img.dtype)
        return np.clip(img_f, 0.0, 1.0).astype(img.dtype)
