"""Fog at night: light that stays near the lamps.

``RandomFog`` is a daylight model. By day the fog is lit from everywhere by
the sky, so the light it scatters towards the camera is the same along every
line of sight and the result is a veil that thickens with distance. At night
there is no such light. The fog is lit by the lamps that are in it, and what
the camera sees is not a veil at all:

* **Everything is dimmed with distance**, as by day: ``exp(-beta z)``.
* **The light removed from each lamp reappears around it.** Fog droplets
  scatter strongly forward, so most of what leaves the direct beam is turned
  by a few degrees and still reaches the lens, from a little to one side.
  That is the glow: a halo round every light, wider and brighter relative to
  its core the more fog lies in between. Dark parts of the frame far from
  any light stay dark, which a daylight veil cannot do.
* **The camera's own illuminator lights the fog in front of it.** With the
  infrared ring on, the droplets nearest the lens scatter its light straight
  back. The ``1/z^2`` fall of the lamp and the ``z^2`` growth of the lit
  volume cancel, so this is an even wall of light across the beam that
  builds up over the first few metres and then saturates — the reason an
  infrared camera sees almost nothing in fog. The scene behind it is also
  attenuated twice, out and back.

The glow is computed from the frame itself in linear light: the fraction
``1 - exp(-beta z)`` of each pixel's radiance is taken out and returned
through a wide, heavy-tailed blur, with clipped pixels counted as several
times full scale because a lamp's recorded value is only a lower bound.

What it does not know about is a lamp outside the frame, which lights fog
inside it all the same.
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from weathermentations.base import WeatherTransform, first_frame, require_depth_provider, scene_of
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.diagnostics import record_draw
from weathermentations.lens.dirt import scatter_psf
from weathermentations.utils.color import veil_radiance_linear
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit
from weathermentations.utils.sky import lift_sky, sky_weight
from weathermentations.utils.sources import source_mask

# Koschmieder: visibility is the distance at which contrast falls to 2%.
_KOSCHMIEDER = 3.912

# Share of the light scattered out of a line of sight that still reaches the
# lens from nearby, as glow. The rest leaves at wide angles and is lost to
# this camera: it is what lights the fog for somebody standing elsewhere.
_FORWARD_FRACTION = 0.6

# How many times full scale a clipped pixel is taken to be.
_OVER_RANGE = 6.0


class NightFog(WeatherTransform):
    """Fog lit by the lamps in the frame, and optionally by the camera.

    Args:
        depth_provider: shared :class:`~weathermentations.depth.provider.SharedDepthProvider`.
        visibility_range_m: meteorological visibility, drawn per call on a
            log scale. 1000 m is mist, 200 m fog, 50 m dense fog. It has to
            be read against the provider's ``depth_range_m``: 200 m of
            visibility does little to a scene 50 m deep.
        glow_radius_range: how far the glow round a light spreads, as a
            fraction of the image diagonal, for fog one optical depth thick.
            It widens with the fog in front of each light.
        illuminator_backscatter_range: brightness of the wall of light the
            camera's own illuminator raises in the fog, relative to full
            scale. Zero — the default — means the illuminator is off. Set it
            only for frames in infrared mode, and apply this transform
            *after* ``NightIrMode``; the shipped pipelines do both.
        beam_falloff_range: how sharply the illuminator's beam dims off-axis.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        depth_provider: SharedDepthProvider
        visibility_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(5.0, 50000.0)),
        ]
        glow_radius_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.005, 0.3)),
        ]
        illuminator_backscatter_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        beam_falloff_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 10.0)),
        ]

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        visibility_range_m: tuple[float, float] = (40.0, 600.0),
        glow_radius_range: tuple[float, float] = (0.02, 0.06),
        illuminator_backscatter_range: tuple[float, float] = (0.0, 0.0),
        beam_falloff_range: tuple[float, float] = (0.6, 2.2),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.visibility_range_m = visibility_range_m
        self.glow_radius_range = glow_radius_range
        self.illuminator_backscatter_range = illuminator_backscatter_range
        self.beam_falloff_range = beam_falloff_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        scene = scene_of(data)
        height, width = params["shape"][:2]
        relative_depth = self.depth_provider.get_depth(scene)
        linear, _scale, _is_integer = to_linear_unit(first_frame(scene))
        sky = sky_weight(linear, relative_depth)
        depth_m = lift_sky(self.depth_provider.to_metric(relative_depth), sky)

        low, high = self.visibility_range_m
        visibility = float(np.exp(self.py_random.uniform(np.log(low), np.log(high))))
        beta = _KOSCHMIEDER / visibility
        backscatter = self.py_random.uniform(*self.illuminator_backscatter_range)
        falloff = self.py_random.uniform(*self.beam_falloff_range)

        yy = (np.arange(height, dtype=np.float32).reshape(-1, 1) - height / 2.0) / (height / 2.0)
        xx = (np.arange(width, dtype=np.float32).reshape(1, -1) - width / 2.0) / (width / 2.0)
        beam = np.exp(-falloff * (xx * xx + yy * yy)).astype(np.float32)

        record_draw(visibility_m=visibility, beta=beta, backscatter=backscatter)
        return {
            "optical_depth": (beta * depth_m).astype(np.float32),
            "sky": sky,
            "glow_radius": self.py_random.uniform(*self.glow_radius_range)
            * float(np.hypot(width, height)),
            "backscatter": backscatter,
            "beam": beam,
        }

    def apply(
        self,
        img: np.ndarray,
        optical_depth: np.ndarray,
        sky: np.ndarray,
        glow_radius: float,
        backscatter: float,
        beam: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        linear, scale, is_integer = to_linear_unit(img)
        illuminated = backscatter > 0.0

        # Lit by the camera, the scene's light crosses the fog twice.
        one_way = np.exp(-optical_depth)[..., None]
        transmission = one_way * one_way if illuminated else one_way

        # What the fog took out of each line of sight, with lamps counted at
        # what they plausibly were and not at where the sensor clipped.
        clipped = source_mask(linear)[..., None]
        radiance = linear * (1.0 + (_OVER_RANGE - 1.0) * clipped)
        removed = radiance * (1.0 - one_way)

        # More fog in front of the lights, wider glow. Weighted by what was
        # removed, so it is the fog in front of the *lights* that counts.
        weight = removed.sum(axis=2)
        thickness = float((optical_depth * weight).sum() / max(float(weight.sum()), 1e-9))
        glow = scatter_psf(removed, glow_radius * (0.5 + min(thickness, 3.0)))

        # The little the night sky itself puts into the fog.
        ambient = veil_radiance_linear(linear, sky).reshape(1, 1, -1) * (1.0 - one_way)

        out = linear * transmission + _FORWARD_FRACTION * glow + (1.0 - _FORWARD_FRACTION) * ambient
        if illuminated:
            wall = backscatter * beam * (1.0 - np.exp(-2.0 * optical_depth))
            out = out + wall[..., None]
        return from_linear_unit(out, scale, is_integer, img.dtype)
