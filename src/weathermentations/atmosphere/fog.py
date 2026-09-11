"""Depth-conditioned fog / haze via Koschmieder's atmospheric scattering law.

    I(x) = I0(x) * t(x) + A * (1 - t(x))
    t(x) = exp(-beta * d(x))

where ``d(x)`` is scene depth (pseudo-meters, from the shared depth
provider), ``beta`` is the atmospheric extinction/scattering coefficient
(1/meters — higher means denser fog, lower visibility), and ``A`` is the
airlight/atmospheric light color (what a maximally-hazy pixel converges to).

Reference: Koschmieder 1924; the same law underlies the dark-channel-prior
dehazing literature (He, Sun & Tang 2009) that this transform's forward
model inverts.
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from weathermentations.base import WeatherTransform, require_depth_provider
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.depth.scale import to_pseudo_metric
from weathermentations.utils.color import estimate_airlight


class RandomFog(WeatherTransform):
    """Physically-grounded fog/haze via Koschmieder's law, conditioned on scene depth.

    Args:
        depth_provider: a :class:`~weathermentations.depth.provider.SharedDepthProvider`,
            shared across every depth-conditioned transform in the pipeline
            (see ``plan.md`` §2.3).
        beta_range: atmospheric extinction coefficient range, in 1/meters
            (against ``depth_scale``'s pseudo-meters). Higher = denser fog.
            Roughly: 0.02 is light haze, 0.1+ is dense fog at a 50m scale.
        atmospheric_light: fixed airlight RGB color, or ``None`` to estimate
            it per-image from the brightest pixels (see
            ``utils/color.py::estimate_airlight``).
        depth_scale: pseudo-meters at relative depth 1.0. ``None`` uses the
            heuristic default (see ``depth/scale.py``).

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        # `BaseTransformInitSchema.model_config` already sets
        # `arbitrary_types_allowed=True`, so `SharedDepthProvider` (not a
        # pydantic-aware type) validates here without extra config.
        depth_provider: SharedDepthProvider
        beta_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        atmospheric_light: tuple[int, int, int] | None
        depth_scale: float | None

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        beta_range: tuple[float, float] = (0.02, 0.1),
        atmospheric_light: tuple[int, int, int] | None = None,
        depth_scale: float | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.beta_range = beta_range
        self.atmospheric_light = atmospheric_light
        self.depth_scale = depth_scale

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        image = data["image"]
        relative_depth = self.depth_provider.get_depth(image)
        depth_pm = to_pseudo_metric(relative_depth, self.depth_scale)

        beta = self.py_random.uniform(*self.beta_range)
        transmission = np.exp(-beta * depth_pm).astype(np.float32)

        if self.atmospheric_light is not None:
            airlight = np.array(self.atmospheric_light, dtype=np.float32)
        else:
            airlight = estimate_airlight(image)

        return {"transmission": transmission, "airlight": airlight}

    def apply(
        self,
        img: np.ndarray,
        transmission: np.ndarray,
        airlight: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        img_f = img.astype(np.float32)
        t = transmission[..., None]
        out = img_f * t + airlight * (1.0 - t)

        if np.issubdtype(img.dtype, np.integer):
            return np.clip(out, 0, 255).astype(img.dtype)
        return np.clip(out, 0.0, 1.0).astype(img.dtype)
