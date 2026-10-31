"""Fog at night: light that stays near the lamps.

``RandomFog`` is a daylight model. By day the fog is lit from everywhere by
the sky, so the light it scatters towards the camera is the same along every
line of sight and the result is a veil that thickens with distance. At night
there is no such light. The fog is lit by the lamps that are in it, and what
the camera sees is not a veil at all:

* **Everything is dimmed with distance**, as by day: ``exp(-beta z)``.
* **The fog is lit where the lamps light it, and that is the veil.** A lamp
  lights the air under it as it lights the road, and fog is white: a thick
  bank of it under a lamp is brighter than the asphalt the same lamp falls
  on. So the frame does not fade to black with distance. It fades to the
  fog's own brightness, which is high near the lamps, takes their colour,
  and falls away into the dark between and above them. This is what makes
  a picture read as fog and not as a dim, soft copy of itself: the blacks
  lift, most where the light is.
* **Each lamp has a halo.** Fog droplets scatter strongly forward, so much
  of what leaves a lamp's direct beam is turned by a few degrees and still
  reaches the lens from a little to one side. The halo is wider and
  brighter relative to its core the more fog lies in between.
* **The camera's own illuminator lights the fog in front of it.** With the
  infrared ring on, the droplets nearest the lens scatter its light straight
  back. The ``1/z^2`` fall of the lamp and the ``z^2`` growth of the lit
  volume cancel, so this is an even wall of light across the beam that
  builds up over the first few metres and then saturates — the reason an
  infrared camera sees almost nothing in fog. The scene behind it is also
  attenuated twice, out and back.

Nobody tells the transform where the lamps are or how strong, so it reads
both off the frame. How much light reaches the fog in some part of the
picture is taken from the surfaces there: what they send back, divided by a
typical albedo, is what fell on them, blurred wide because the air a lamp
lights is a volume and not the patch of road under it. A clipped pixel is
counted as several times full scale, a lamp's recorded value being only a
lower bound.

An earlier version had only the halo, built by blurring whatever the fog
removed from the whole picture. With no veil that is a dimmed frame plus a
blurred copy of itself, and it looked like exactly that.

What it does not know about is a lamp outside the frame, which lights fog
inside it all the same.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from weathermentations.base import WeatherTransform, first_frame, require_depth_provider, scene_of
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.diagnostics import record_draw
from weathermentations.lens.dirt import scatter_psf
from weathermentations.utils.color import TYPICAL_SCENE_ALBEDO
from weathermentations.utils.photometry import from_linear_unit, to_linear_unit
from weathermentations.utils.sky import lift_sky, sky_weight
from weathermentations.utils.sources import light_field, source_mask

# Koschmieder: visibility is the distance at which contrast falls to 2%.
_KOSCHMIEDER = 3.912

# Share of the light a lamp loses to the fog that still reaches the lens
# from close beside it, as the halo.
_FORWARD_FRACTION = 0.6

# How many times full scale a clipped pixel is taken to be.
_OVER_RANGE = 6.0

# Radiance of thick fog relative to a white surface under the same light.
# Below one: a street lamp throws its light down, and the camera looks along
# the fog from the side, where droplets scatter least.
_FOG_RADIANCE_SHARE = 0.4

# Width of the region whose surfaces say how brightly the fog over them is
# lit, as a fraction of the diagonal.
_LIT_VOLUME_SIGMA = 0.1

# Share of the frame's mean light that reaches fog everywhere: lamps out of
# frame, and light scattered more than once.
_DIFFUSE_SHARE = 0.25


# How far the sky mask is allowed to grow to meet the depth map's own edge,
# as a fraction of the diagonal, and how close in relative depth a pixel has
# to be to the sky's to be let in.
_SKY_EDGE_REACH = 0.04
_SKY_DEPTH_TOLERANCE = 0.04


def _sky_to_depth_edge(sky: np.ndarray, relative_depth: np.ndarray) -> np.ndarray:
    """Redraw the sky mask's outline where the depth map puts it.

    The mask is made at low resolution and its edge wanders a few percent of
    the frame either side of the real skyline. By day that is hidden, the
    veil being about as bright as the sky. At night the fog is brighter than
    the sky, so every patch of sky the mask misses next to a tree stays a
    black hole in a glowing bank. The depth map has the skyline to the
    pixel; what it lacks is which side is sky, which the mask knows.
    """
    core = sky > 0.9
    if not core.any():
        return sky
    level = float(np.median(relative_depth[core]))
    near_level = np.clip(
        2.0 - np.abs(relative_depth - level) / _SKY_DEPTH_TOLERANCE, 0.0, 1.0
    )
    reach = max(3, round(_SKY_EDGE_REACH * float(np.hypot(*sky.shape))) | 1)
    grown = cv2.dilate(sky, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach)))
    return (grown * near_level).astype(np.float32)


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
        sky = _sky_to_depth_edge(sky_weight(linear, relative_depth), relative_depth)
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

        # Lamps counted at what they plausibly were, not at where the
        # sensor clipped: a lamp seen through thin fog is still clipped.
        clipped = source_mask(linear)[..., None]
        radiance = linear * (1.0 + (_OVER_RANGE - 1.0) * clipped)

        # The halo: what the fog took from each lamp, returned around it.
        # More fog in front of the lamps, wider halo.
        removed = radiance * clipped * (1.0 - one_way)
        weight = removed.sum(axis=2)
        thickness = float((optical_depth * weight).sum() / max(float(weight.sum()), 1e-9))
        halo = scatter_psf(removed, glow_radius * (0.5 + min(thickness, 3.0)))

        out = radiance * transmission + _FORWARD_FRACTION * halo
        if not illuminated:
            # The veil: fog lit by the lamps, as bright as the light the
            # surfaces around it show. Under the illuminator those surfaces
            # are lit by the camera, and the fog the camera lights is the
            # wall below, so the term would count it twice.
            lit = light_field(linear, _LIT_VOLUME_SIGMA)
            diffuse = linear.reshape(-1, linear.shape[-1]).mean(axis=0)
            fog_radiance = (_FOG_RADIANCE_SHARE / TYPICAL_SCENE_ALBEDO) * (
                (1.0 - _DIFFUSE_SHARE) * lit + _DIFFUSE_SHARE * diffuse.reshape(1, 1, -1)
            )
            out = out + fog_radiance * (1.0 - one_way)
        if illuminated:
            wall = backscatter * beam * (1.0 - np.exp(-2.0 * optical_depth))
            out = out + wall[..., None]
        return from_linear_unit(out, scale, is_integer, img.dtype)
