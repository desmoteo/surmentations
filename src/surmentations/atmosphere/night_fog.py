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

import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from surmentations import _native
from surmentations.base import (
    CameraTransform,
    first_frame,
    lamp_cones,
    require_depth_provider,
    scene_of,
)
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import record_draw
from surmentations.lens.dirt import scatter_psf
from surmentations.utils.channels import channel_sum, frame_mean
from surmentations.utils.color import TYPICAL_SCENE_ALBEDO
from surmentations.utils.photometry import from_linear_unit, to_linear_unit
from surmentations.utils.sky import lift_sky, scene_extent_m, sky_to_depth_edge, sky_weight
from surmentations.utils.sources import lamp_mask, light_field

# A lamp against a lit surface that reaches full scale: the share of the
# frame's diagonal its surroundings are averaged over, and the levels of
# that average, in linear light, from which a bright patch is no lamp at
# all and under which it wholly is.
_LAMP_STANDS_ALONE = (0.03, 0.6, 0.35)
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


class NightFog(CameraTransform):
    """Fog lit by the lamps in the frame, and optionally by the camera.

    Args:
        depth_provider: shared :class:`~surmentations.depth.provider.SharedDepthProvider`.
        visibility_range_scene: visibility as a multiple of the scene's
            depth, in place of ``visibility_range_m``: the same fog on a
            courtyard and on a long street. See ``RandomFog``. ``None``, the
            default, uses ``visibility_range_m``.
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
        lamp_cone_range: how strongly the fog shows in the light of the
            scene's lamps, drawn per call; 0 is not at all. It does
            something on a frame that
            :class:`~surmentations.camera.night_ir.NightIrMode` or
            :class:`~surmentations.camera.night_color.NightColorMode` made
            a night of with a provider that knows where its lamps are: the
            fog is lit in a cone under each street lamp and in a wedge
            ahead of each headlamp, and dark elsewhere, which is how
            fog at night is mostly seen. See
            :func:`~surmentations.base.lamp_cones`.
            The fog's own glow is then that of its lamps, and no longer
            the scene's blurred brightness, which over a lit road was a
            white sheet.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``NightFog`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """
        depth_provider: SharedDepthProvider
        visibility_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(5.0, 50000.0)),
        ]
        visibility_range_scene: (
            Annotated[
                tuple[float, float],
                AfterValidator(nondecreasing),
                AfterValidator(check_range_bounds(0.01, 100.0)),
            ]
            | None
        )
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
        lamp_cone_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 4.0)),
        ]

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        visibility_range_m: tuple[float, float] = (40.0, 600.0),
        glow_radius_range: tuple[float, float] = (0.02, 0.06),
        illuminator_backscatter_range: tuple[float, float] = (0.0, 0.0),
        beam_falloff_range: tuple[float, float] = (0.6, 2.2),
        visibility_range_scene: tuple[float, float] | None = None,
        lamp_cone_range: tuple[float, float] = (0.6, 1.4),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring.

        Raises ``ValueError`` without a ``depth_provider``: the one
        :class:`~surmentations.depth.provider.SharedDepthProvider` of the
        pipeline must be passed in, so that depth is estimated once per frame.
        """
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.visibility_range_m = visibility_range_m
        self.visibility_range_scene = visibility_range_scene
        self.glow_radius_range = glow_radius_range
        self.illuminator_backscatter_range = illuminator_backscatter_range
        self.beam_falloff_range = beam_falloff_range
        self.lamp_cone_range = lamp_cone_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the night fog for one frame from its depth.

        Returns the ``optical_depth`` ``beta * z`` with ``beta`` from a
        visibility drawn log-uniformly, the ``sky`` mask redrawn to the depth
        map's skyline, the lamps' ``glow_radius``, the illuminator's
        ``backscatter`` and its ``beam`` across the frame.
        """
        scene = scene_of(data)
        height, width = params["shape"][:2]
        relative_depth = self.depth_provider.get_depth(scene)
        linear, _scale, _is_integer = to_linear_unit(first_frame(scene))
        sky = sky_to_depth_edge(
            sky_weight(linear, relative_depth, self.depth_provider.get_labels(scene)),
            relative_depth,
            linear,
        )
        depth_m = lift_sky(self.depth_provider.to_metric(relative_depth), sky)

        low, high = self.visibility_range_m
        if self.visibility_range_scene is not None:
            # Against the scene and not in metres; see `RandomFog`.
            extent = scene_extent_m(depth_m, sky)
            low, high = (factor * extent for factor in self.visibility_range_scene)
        visibility = float(np.exp(self.py_random.uniform(np.log(low), np.log(high))))
        beta = _KOSCHMIEDER / visibility
        backscatter = self.py_random.uniform(*self.illuminator_backscatter_range)
        falloff = self.py_random.uniform(*self.beam_falloff_range)

        yy = (np.arange(height, dtype=np.float32).reshape(-1, 1) - height / 2.0) / (height / 2.0)
        xx = (np.arange(width, dtype=np.float32).reshape(1, -1) - width / 2.0) / (width / 2.0)
        beam = np.exp(-falloff * (xx * xx + yy * yy)).astype(np.float32)

        # The fog in the lamps' own light, where the frame's lamps are
        # known: the last of the draws.
        cone = self.py_random.uniform(*self.lamp_cone_range)
        cones = lamp_cones(data, beta) if cone > 0.0 else None
        if cones is not None:
            cones = cones * np.float32(cone)
        record_draw(
            visibility_m=visibility,
            beta=beta,
            backscatter=backscatter,
            lamp_cones=cone,
            lamp_cones_drawn=cones is not None,
        )
        return {
            "cones": cones,
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
        cones: np.ndarray | None = None,
        **params: Any,
    ) -> np.ndarray:
        """Attenuate the scene, then add what the fog returns of each light.

        Each lamp's lost light comes back as a halo around it, wider in
        thicker fog. Lit by the lamps, the fog glows as brightly as the
        surfaces around it; lit by the camera's illuminator (``backscatter``
        above 0), the scene's light crosses the fog twice and the fog is a
        wall of light in the beam instead.
        """
        linear, scale, is_integer = to_linear_unit(img)
        illuminated = backscatter > 0.0

        # Lit by the camera, the scene's light crosses the fog twice.
        through = np.exp(-optical_depth)

        # Lamps counted at what they plausibly were, not at where the
        # sensor clipped: a lamp seen through thin fog is still clipped.
        # Lamps, and not whatever is near full scale: the road under a
        # lamp is as bright as the sensor records and no brighter, and
        # counted as a lamp it came through thin fog as a white sheet with
        # a ragged edge where its grain crossed the level.
        # And a lamp stands in the dark: where everything round a bright
        # patch is bright too, it is a lit surface, however compact the
        # part of it that reaches full scale. (Channel against channel: a
        # maximum along an axis of three is several times slower.)
        nearby = light_field(linear, _LAMP_STANDS_ALONE[0])
        around = np.maximum(np.maximum(nearby[..., 0], nearby[..., 1]), nearby[..., 2])
        alone = np.clip(
            (_LAMP_STANDS_ALONE[1] - around)
            / (_LAMP_STANDS_ALONE[1] - _LAMP_STANDS_ALONE[2]),
            0.0,
            1.0,
        )
        lamps = lamp_mask(linear) * alone

        # The halo: what the fog took from each lamp, returned around it.
        # More fog in front of the lamps, wider halo.
        lost = _native.fog_lost(linear, lamps, through, _OVER_RANGE - 1.0)
        if lost is not None:
            radiance, removed, weight = lost
        else:
            clipped = lamps[..., None]
            radiance = linear * (1.0 + (_OVER_RANGE - 1.0) * clipped)
            removed = radiance * clipped * (1.0 - through[..., None])
            weight = channel_sum(removed)
        thickness = float((optical_depth * weight).sum() / max(float(weight.sum()), 1e-9))
        halo = scatter_psf(removed, glow_radius * (0.5 + min(thickness, 3.0)))

        # What the fog itself sends. Lit by lamps whose places are known,
        # it glows where their light is, and by the scene's general level
        # elsewhere. Without them it is the veil: fog lit by the lamps, as
        # bright as the light the surfaces around it show. Under the
        # illuminator those surfaces are lit by the camera, and the fog
        # the camera lights is the wall, so the veil would count it twice.
        level: np.ndarray = np.zeros(3, dtype=np.float32)
        diffuse: np.ndarray = np.zeros(3, dtype=np.float32)
        veil_share = lit_part = 0.0
        lit = wall = None
        if cones is not None:
            share = _FOG_RADIANCE_SHARE * _DIFFUSE_SHARE / TYPICAL_SCENE_ALBEDO
            level = (share * frame_mean(linear)).astype(np.float32)
        elif not illuminated:
            lit = light_field(linear, _LIT_VOLUME_SIGMA)
            veil_share = _FOG_RADIANCE_SHARE / TYPICAL_SCENE_ALBEDO
            lit_part = 1.0 - _DIFFUSE_SHARE
            diffuse = (_DIFFUSE_SHARE * frame_mean(linear)).astype(np.float32)
        if illuminated:
            wall = (backscatter * beam * (1.0 - np.exp(-2.0 * optical_depth))).astype(np.float32)
        out = _native.fog_veil(
            radiance,
            halo,
            through,
            [_FORWARD_FRACTION, float(illuminated), *level, veil_share, lit_part, *diffuse],
            cones,
            lit,
            wall,
        )
        if out is None:
            one_way = through[..., None]
            transmission = one_way * one_way if illuminated else one_way
            out = radiance * transmission + _FORWARD_FRACTION * halo
            if cones is not None:
                out = out + cones + level.reshape(1, 1, -1) * (1.0 - one_way)
            elif lit is not None:
                fog_radiance = np.float32(veil_share) * (
                    np.float32(lit_part) * lit + diffuse.reshape(1, 1, -1)
                )
                out = out + fog_radiance * (1.0 - one_way)
            if wall is not None:
                out = out + wall[..., None]
        return from_linear_unit(out, scale, is_integer, img.dtype)
