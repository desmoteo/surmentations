"""Night mode: IR-cut filter removed, scene lit by the camera's own illuminator.

Physical model
--------------
A day/night camera swings its IR-cut filter out of the optical path after
dark and switches on a ring of 850 nm LEDs around the lens. Three things
change at once, and none of them is "the image gets darker":

**The image goes monochrome, and not by luminance.** Silicon is sensitive
well past 700 nm, and all three Bayer dyes transmit there, so every channel
sees mostly the same infrared signal and the colour collapses. But what it
collapses *to* is near-infrared **reflectance**, which is not the visible
brightness of a material. The largest discrepancy is vegetation: chlorophyll
absorbs red strongly and reflects NIR strongly — the "red edge" at ~700 nm —
so foliage that reads dark green by day is among the *brightest* things in
frame at night. Converting to luminance gets this exactly backwards, which is
the single most visible error in a naive night-mode augmentation.

We have no NIR channel to read, so reflectance is estimated from RGB: a broad
red-weighted response (silicon's sensitivity rises toward the red end) plus
an explicit boost where the pixel is green relative to red, which is where
chlorophyll is. It is an approximation, and it is calibrated to reproduce the
one large, well-known material difference rather than to be spectrally exact.

**The scene is lit by the camera, not by the sun.** The illuminator sits at
the lens, so irradiance falls as ``1/z^2`` and is shaped by the LED beam
profile. This is why night CCTV looks the way it does: a subject at two
metres is blown to featureless white while the background five metres behind
is nearly black, and no exposure setting fixes it, because the ratio is 6:1
in the illumination itself. Getting the inverse-square right is what produces
that automatically instead of having to paint it on.

**Gain goes up, so noise does.** Shot noise scales with the square root of
the signal and read noise does not scale at all, so a gained-up dark frame is
noisiest in the shadows — not uniformly noisy.

**Traffic has its lamps on.** A photograph taken by day shows cars with their
lamps off, and by night they are the brightest things in the frame: a pair of
points far over full scale, each with the halo a saturated source has on
these sensors. Which lamps a vehicle shows, and where, is not something
colour says in a picture that is about to lose its colour, so they are lit
only where the provider has a vehicle reader
(:mod:`surmentations.perception.vehicles`): headlamps on a vehicle seen from
the front, tail lamps on one seen from behind, where the reader found them.
With the filter out the sensor sees a red tail lamp as it sees a white
headlamp, by how bright it is, and both come out white.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from surmentations import mode
from surmentations.base import CameraTransform, first_frame, require_depth_provider, scene_of
from surmentations.camera.night_lighting import (
    SceneLighting,
    light_scene,
    lit_air_of,
    note_lit_air,
)
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import record_draw
from surmentations.perception.labels import VEGETATION, SceneClass, class_weight
from surmentations.perception.vehicles import VehicleView
from surmentations.utils.channels import percentile
from surmentations.utils.grids import on_coarse_grid, pixel_grid
from surmentations.utils.night_light import even_road
from surmentations.utils.photometry import from_linear_unit, to_linear_unit
from surmentations.utils.road import paint_on
from surmentations.utils.sky import sky_to_depth_edge, sky_weight

# Broad, red-weighted stand-in for silicon's response with the IR-cut filter
# removed. Not a luminance vector: the eye's green-heavy weighting is exactly
# what does not apply once the sensor is seeing past 700 nm.
_NIR_RESPONSE = np.array([0.50, 0.30, 0.20], dtype=np.float32)


# What is left of the sky at night, as a share of what the ambient light
# would make of a surface as pale. The ambient term is light *reflected*:
# moonlight and spill off the ground and the walls. The sky reflects
# nothing, and in a frame the illuminator cannot reach, which is every frame
# whose distances are real and long, it came out as bright as by day.
_NIGHT_SKY_LEVEL = 0.12

# How green a pixel labelled as vegetation is taken to be at the least, on
# the scale the chlorophyll term works in: about what a leaf in full light
# measures. A leaf in shade, in autumn or under a sodium lamp measures
# nothing, and reflects the near infrared all the same.
_LABELLED_GREENNESS = 0.3

# How much of the glow round a lamp the sensor shows: a source far over
# full scale blooms on it whatever the air is like.
_LAMP_BLOOM = 0.3
# What a road in a beam sends back at the least, as a share of the middle
# value of the frame's road.
_BEAM_ROAD = 0.7
# The auto-exposure, once lamps light the scene, holds the brightest
# hundredth of its surfaces to this level: a camera that exposed a lit road
# for its dark verges would show the road as a white sheet.
_HIGHLIGHT_LEVEL = 0.95
_HIGHLIGHT_PERCENTILE = 99.0
# How many times what a white matt surface sends back to a lamp at the lens
# each of these does: the sheeting of a road sign and of a number plate,
# and the glass beads of road paint. And the least a sheeting is taken to
# send back of that, whatever colour it photographed as: a green sign's
# ground returns a seventh of a white one's, and is still sheeting.
_SHEETING_GAIN = 500.0
_PAINT_GAIN = 3.0
_SHEETING_FLOOR = 0.15
_RETRO_SOFT_PX = 0.8


def retroreflectors(
    linear_rgb: np.ndarray,
    labels: np.ndarray,
    views: list[VehicleView] | None,
    signs: np.ndarray | None,
) -> np.ndarray | None:
    """How many times a matt surface's return each pixel sends back to a lamp at the lens.

    ``(H, W)`` float32, 0 where the surface is an ordinary one; ``None``
    where the frame has nothing of the kind.

    Some things on a road are made to send light back where it came from:
    the sheeting of signs and number plates, the glass beads in road paint.
    To a driver that is what makes them show in headlamps. To a camera
    whose illuminator sits round its lens it is the same thing, and
    stronger: the lens is on the lamp's own axis. Under infrared a sign is
    a white rectangle with nothing on it and a number plate is a white bar,
    which is why cameras that read plates meter for the plate and not the
    scene. Under anyone else's lamp, a street lamp's or a car's, none of
    this happens: the light goes back to that lamp.

    * **Road paint** is the scene's labels' road and the thin bright
      things on it (:func:`~surmentations.utils.road.paint_on`): a few
      times a matt white's return.
    * **Signs** (``signs``) and **number plates** (``views``) are the
      provider's reader's, and return some five hundred times it. A matt
      white returns about a third of a candela per lux and square metre;
      sheeting is quoted at some seventy for the plainest grade to several
      hundred for the prismatic ones, so the figure is the middle of a
      range of ten to one, and not one that was read off a standard.

    It multiplies what the surface's own brightness makes of the
    illuminator's light, so a plate's dark characters stay darker than its
    ground until both are at full scale. On sheeting the result is
    negative, as a mark for the caller: that brightness is not to be taken
    below a floor there.
    """
    height, width = labels.shape
    gain = np.zeros((height, width), dtype=np.float32)
    road = labels == int(SceneClass.ROAD)
    if int(road.sum()) > 64:
        gain[paint_on(linear_rgb, road)] = _PAINT_GAIN
    boxes = [] if signs is None else [tuple(row[:4]) for row in signs.tolist()]
    boxes += [view.plate for view in views or () if view.plate is not None]
    for x0, y0, x1, y1 in boxes:
        left, top = max(0, round(x0)), max(0, round(y0))
        right, bottom = min(width, max(round(x1), left + 1)), min(height, max(round(y1), top + 1))
        gain[top:bottom, left:right] = -_SHEETING_GAIN
    if float(np.abs(gain).max()) == 0.0:
        return None
    # A lens does not draw an edge a pixel sharp, least of all on the
    # brightest thing in its frame.
    soft: np.ndarray = cv2.GaussianBlur(gain, (0, 0), sigmaX=_RETRO_SOFT_PX)
    return soft


def near_infrared_reflectance(
    linear_rgb: np.ndarray, vegetation_gain: float, vegetation: np.ndarray | None = None
) -> np.ndarray:
    """Estimate NIR reflectance from a visible-light image.

    Broad red-weighted response plus a chlorophyll term. ``vegetation_gain``
    controls how strongly foliage lifts; 0 disables the correction and leaves
    a plain broad-band conversion.

    ``vegetation`` is the weight of what the scene's labels call vegetation,
    ``(H, W)`` in ``[0, 1]``, where there are labels
    (:mod:`surmentations.perception.labels`). The chlorophyll term is
    read off colour, and colour says "leaf" only for a green leaf in good
    light; a labelled one is lifted whatever colour it photographed as.
    """
    broadband = linear_rgb @ _NIR_RESPONSE

    # Greenness relative to red is where chlorophyll is, and chlorophyll is
    # what reflects hard in the near infrared.
    red, green, blue = linear_rgb[..., 0], linear_rgb[..., 1], linear_rgb[..., 2]
    greenness = np.clip(
        (green - np.maximum(red, blue)) / np.maximum(green + red + blue, 1e-4), 0.0, 1.0
    )
    if vegetation is not None:
        # By the pixel's own brightness against the frame's, so that the
        # shade inside a crown stays shade and a tree keeps its shape.
        lit = np.clip(broadband / max(float(broadband.mean()), 1e-4), 0.0, 1.0)
        greenness = np.maximum(greenness, np.float32(_LABELLED_GREENNESS) * vegetation * lit)
    return np.clip(broadband + vegetation_gain * greenness, 0.0, 4.0)


class NightIrMode(CameraTransform):
    """Monochrome IR night mode with the camera's own illuminator.

    With a provider that carries scene labels
    (:mod:`surmentations.perception.labels`): vegetation is lifted by
    name as well as by colour, so a tree is bright in the near infrared in
    shade, in autumn and under a sodium lamp too; and the sky, which
    reflects neither the illuminator nor the ambient light, is dark. Where
    the camera's height is known the illuminator's inverse-square law runs
    over real metres, and a scene beyond its reach is lit by the ambient
    term alone.

    With a provider that also carries a vehicle reader
    (:mod:`surmentations.perception.vehicles`) the traffic's lamps are
    lit: see ``vehicle_lamp_range``.

    Args:
        depth_provider: shared :class:`~surmentations.depth.provider.SharedDepthProvider`.
        illuminator_range_m: distance at which the illuminator delivers unit
            exposure. Objects nearer than this blow out, objects further fade
            into the noise — and the ``1/z^2`` between them is why night CCTV
            has such a narrow band of usable distance.
        beam_falloff_range: how tightly the LED beam is focused. Higher
            concentrates light on the optical axis and darkens the corners
            further; the exponent of a ``cos^k`` beam profile.
        ambient_range: residual scene lighting (moon, streetlights, spill) as
            a fraction of unit exposure, so the far field is not pure black.
        vegetation_gain_range: how strongly foliage lifts relative to a plain
            broad-band conversion — the chlorophyll red-edge term.
        target_level_range: level the camera's auto-exposure drives the scene
            to, as a fraction of full scale measured at the 80th percentile.
            This is not a cosmetic normalisation — a night camera really does
            gain up until the scene is usable, and without modelling it the
            result depends entirely on how the depth map happened to be
            scaled: pick an illuminator range slightly too short and the whole
            lower frame saturates. Auto-exposure fixes the *level* while
            leaving the ``1/z^2`` **ratio** between near and far intact, and
            that ratio is the part that makes night footage recognisable.
        gain_range: sensor gain, which sets how far the noise floor is
            lifted. It does not scale the signal: the auto-exposure has
            already placed that, and gain is one of the levers it used to get
            there. Higher gain therefore means a noisier frame at the same
            brightness, which is what raising ISO actually buys you.
        read_noise_range: signal-independent noise, as a fraction of unit
            exposure. Unlike shot noise this does not shrink in the shadows,
            which is why a gained-up night frame is noisiest where it is
            darkest.
        street_lamp_range: how much light a street lamp throws on the road
            straight under it, in unit exposures, drawn per call; 0 is
            none, and the scene is lit by the ambient term alone, as
            before. Above it the scene is lit as a road is at night, from
            a few points: each street lamp throws a pool under itself that
            falls off with the square of the distance, the lamps
            themselves burn, and what lies between is as dark as
            ``ambient_range`` makes it. Where the lamps are comes from the
            provider: from its vehicle reader, which finds them all along
            a road, or from its scene labels, which find the nearest. The
            scene's ground has to be labelled, since the light is worked
            out in space. With neither lamps nor labels this does nothing.
            The labelled road also has the day's light and shade taken off
            it: the sun's shadows have no business under street lamps.
        retroreflector_range: how strongly the things made to send light
            back where it came from return the illuminator's, drawn per
            call, as a multiple of what they usually do; 0 is none. Road
            paint, from the scene's labels, comes back a few times as
            bright as a white surface would. Road signs and number plates,
            which the provider's vehicle reader finds, come back some
            five hundred times as bright: within the illuminator's
            reach a sign is a white rectangle and a plate a white bar, as
            they are on a real camera that meters for the scene and not
            for the plate. It is the illuminator's light alone that comes
            back this way, since the lens is beside the lamp; street lamps
            and headlamps get theirs back themselves. See
            :func:`retroreflectors`.
        vehicle_lamp_range: how brightly the lamps of the vehicles in the
            frame are lit, drawn per call; 1 is dipped beams, 0 none. It
            does something only on a provider built with ``vehicles=``:
            headlamps on each vehicle its reader saw from the front and
            tail lamps on each it saw from behind, where it found them,
            far over full scale and white, as an infrared camera shows
            both, with the bloom of a saturated source. Most vehicles are
            lit and some are not, as parked ones are not. Lit here and not
            before, the lamps are light and not paint: the illuminator's
            falloff does not dim them, and whatever follows in a pipeline,
            fog, rain or snow at night, finds them as it finds any lamp in
            a frame. A lit vehicle also throws its beam on the road ahead
            of it. Without a vehicle reader no lamp is lit, whatever this
            is.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``NightIrMode`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """
        depth_provider: SharedDepthProvider
        illuminator_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.5, 200.0)),
        ]
        beam_falloff_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 12.0)),
        ]
        ambient_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        vegetation_gain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        target_level_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 0.95)),
        ]
        gain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 20.0)),
        ]
        read_noise_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.2)),
        ]
        residual_tint: float = Field(ge=0.0, le=0.5)
        vehicle_lamp_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        street_lamp_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 5.0)),
        ]
        retroreflector_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 3.0)),
        ]

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        illuminator_range_m: tuple[float, float] = (4.0, 14.0),
        beam_falloff_range: tuple[float, float] = (1.0, 4.0),
        ambient_range: tuple[float, float] = (0.01, 0.12),
        vegetation_gain_range: tuple[float, float] = (0.25, 0.7),
        target_level_range: tuple[float, float] = (0.35, 0.6),
        gain_range: tuple[float, float] = (0.8, 1.6),
        read_noise_range: tuple[float, float] = (0.004, 0.03),
        residual_tint: float = 0.04,
        vehicle_lamp_range: tuple[float, float] = (0.7, 1.3),
        street_lamp_range: tuple[float, float] = (0.6, 1.4),
        retroreflector_range: tuple[float, float] = (0.6, 1.4),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring.

        Raises ``ValueError`` without a ``depth_provider``: the one
        :class:`~surmentations.depth.provider.SharedDepthProvider` of the
        pipeline must be passed in, so that depth is estimated once per frame.
        """
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.illuminator_range_m = illuminator_range_m
        self.beam_falloff_range = beam_falloff_range
        self.ambient_range = ambient_range
        self.vegetation_gain_range = vegetation_gain_range
        self.target_level_range = target_level_range
        self.gain_range = gain_range
        self.read_noise_range = read_noise_range
        self.residual_tint = residual_tint
        self.vehicle_lamp_range = vehicle_lamp_range
        self.street_lamp_range = street_lamp_range
        self.retroreflector_range = retroreflector_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the illuminator and the camera, and how much light reaches each pixel.

        Returns the ``exposure``: the inverse square of the distance from the
        illuminator at the lens, times its beam profile. With it, the ambient
        light, the auto-exposure target, the vegetation gain, the sensor gain
        and read noise, and a seed for the noise.
        """
        scene = scene_of(data)
        height, width = params["shape"][:2]

        depth_m = np.maximum(self.depth_provider.get_metric_depth(scene), 0.3)

        reference = self.py_random.uniform(*self.illuminator_range_m)
        falloff = self.py_random.uniform(*self.beam_falloff_range)

        # Inverse-square from an illuminator sitting at the lens.
        irradiance = (reference / depth_m) ** 2

        # LED beam profile: brightest on the optical axis, falling as cos^k of
        # the field angle. Approximated from normalised image radius, which is
        # monotonic in field angle.
        cx, cy = width / 2.0, height / 2.0

        def beam_at(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
            """The illuminator's beam at the given coordinates: ``cos(field angle) ** falloff``.

            The field angle is taken from the normalised distance from the
            frame's centre, at most 60 degrees at its edge.
            """
            radius = np.sqrt(((xx - cx) / max(cx, 1)) ** 2 + ((yy - cy) / max(cy, 1)) ** 2)
            profile: np.ndarray = (
                np.clip(np.cos(np.clip(radius, 0.0, 1.0) * (np.pi / 3.0)), 0.0, 1.0) ** falloff
            )
            return profile

        # The beam changes over a third of the frame. In fast mode its cosine
        # and power are taken on a grid four times coarser and enlarged.
        beam = (
            on_coarse_grid(height, width, beam_at)
            if mode.fast()
            else beam_at(*pixel_grid(height, width))
        )

        # What the scene's labels call vegetation, where there are labels:
        # the near infrared's brightest material, whatever colour it is.
        labels = self.depth_provider.get_labels(scene)
        vegetation = night_sky = None
        if labels is not None:
            soften = max(1.0, 0.003 * float(np.hypot(height, width)))
            vegetation = class_weight(labels, VEGETATION, soften_px=soften)
            # And the sky, which no lamp lights and nothing is reflected by.
            # Drawn up to the skyline's own edge: a network's outline is a
            # few pixels out, and a band of sky left as bright as by day
            # is a white line round every roof.
            relative_depth = self.depth_provider.get_depth(scene)
            linear, _scale, _is_integer = to_linear_unit(first_frame(scene))
            night_sky = sky_to_depth_edge(
                sky_weight(linear, relative_depth, labels), relative_depth, linear
            )

        record_draw(
            illuminator_range_m=reference,
            beam_falloff=falloff,
            depth_min_m=lambda: float(depth_m.min()),
            depth_p50_m=lambda: float(np.median(depth_m)),
            depth_max_m=lambda: float(depth_m.max()),
        )
        drawn = {
            "exposure": (irradiance * beam).astype(np.float32),
            "ambient": self.py_random.uniform(*self.ambient_range),
            "target_level": self.py_random.uniform(*self.target_level_range),
            "vegetation_gain": self.py_random.uniform(*self.vegetation_gain_range),
            "gain": self.py_random.uniform(*self.gain_range),
            "read_noise": self.py_random.uniform(*self.read_noise_range),
            "noise_seed": self.py_random.randint(0, 2**31 - 1),
            "vegetation": vegetation,
            "night_sky": night_sky,
        }

        # The lamps, drawn after everything else is: a provider that knows
        # of none draws what it always drew.
        lamps = self.py_random.uniform(*self.vehicle_lamp_range)
        lamp_seed = self.py_random.randint(0, 2**31 - 1)
        street = self.py_random.uniform(*self.street_lamp_range)
        lighting = light_scene(
            self.depth_provider, scene, labels, depth_m, lamp_seed, lamps, street
        )
        drawn["lighting"] = lighting
        if lighting.in_beam is not None and labels is not None:
            # What a road in a beam sends back at the least: the road's own
            # middle value, where there is a road to take it from.
            paved = labels == int(SceneClass.ROAD)
            if int(paved.sum()) > 50:
                linear = to_linear_unit(first_frame(scene))[0]
                drawn["road_level"] = float(np.median(linear[paved] @ _NIR_RESPONSE))
        # The light in the air, for whatever puts something in that air
        # afterwards. Colourless: this camera sees no colour.
        white = np.ones(3, dtype=np.float32)
        drawn["air"] = lit_air_of(lighting, white, white)
        drawn["exposed"] = [1.0]

        # What is made to send light back where it came from, under a lamp
        # that is at the lens: the last of the draws.
        retro = self.py_random.uniform(*self.retroreflector_range)
        returning = None
        if retro > 0.0 and labels is not None:
            linear = to_linear_unit(first_frame(scene))[0]
            returning = retroreflectors(
                linear,
                labels,
                self.depth_provider.get_vehicles(scene),
                self.depth_provider.get_signs(scene),
            )
            if returning is not None:
                drawn["retro"] = np.float32(retro) * returning
        record_draw(
            vehicle_lamps=lamps,
            vehicles_read=lighting.vehicles_read,
            vehicles_lit=lighting.vehicles_lit,
            street_lamps=lighting.street_lamps if lighting.lit else 0,
            street_lamp_level=street,
            retroreflectors=retro,
            retroreflectors_drawn=returning is not None,
        )
        return drawn

    def apply_with_params(
        self, params: dict[str, Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Apply, then note for the frame returned what light hangs in its air.

        For the fog, rain or snow a pipeline puts in that air afterwards
        (:func:`~surmentations.camera.night_lighting.note_lit_air`).
        """
        result = super().apply_with_params(params, *args, **kwargs)
        note_lit_air(self.depth_provider, params, result)
        return result

    def apply(
        self,
        img: np.ndarray,
        exposure: np.ndarray,
        ambient: float,
        target_level: float,
        vegetation_gain: float,
        gain: float,
        read_noise: float,
        noise_seed: int,
        vegetation: np.ndarray | None = None,
        night_sky: np.ndarray | None = None,
        lighting: SceneLighting | None = None,
        road_level: float | None = None,
        retro: np.ndarray | None = None,
        exposed: list[float] | None = None,
        **params: Any,
    ) -> np.ndarray:
        """Render the frame as the near-infrared camera sees it at night.

        Near-infrared reflectance times the light that arrived, scaled by the
        auto-exposure to put the 80th percentile at the target, plus the
        traffic's lamps where there are any to light, plus shot and read
        noise, in a grey with the residual tint of the colour filters.
        """
        linear, scale, is_integer = to_linear_unit(img)

        reflectance = near_infrared_reflectance(linear, vegetation_gain, vegetation)
        arriving = exposure + ambient
        lighting = SceneLighting() if lighting is None else lighting
        if lighting.lit:
            # Lit by lamps: the road loses the day's light and shade, which
            # no lamp explains, and takes the pools and the beams.
            if lighting.road is not None:
                reflectance = even_road(reflectance, lighting.road)
            if lighting.in_beam is not None and road_level is not None:
                # A beam falls on a road, and a road in the day's deep
                # shade photographed as black: lit, it would stay black.
                # The shade was the light and not the road, so where a
                # beam falls the road sends back most of what a road does.
                floor = np.float32(_BEAM_ROAD * road_level) * lighting.in_beam
                reflectance = np.maximum(reflectance, floor)
            for thrown in (lighting.street, lighting.beams):
                if thrown is not None:
                    arriving = arriving + thrown
        signal = reflectance * arriving
        if night_sky is not None:
            signal = signal * (1.0 - np.float32(1.0 - _NIGHT_SKY_LEVEL) * night_sky)

        # Auto-exposure. The camera gains up until the scene is usable, so the
        # absolute level is set by the AE loop rather than by the illuminator's
        # nominal range — while the near/far *ratio* the inverse-square law
        # produced is untouched, which is the part that reads as night footage.
        reference_level = percentile(signal, 80.0)
        if reference_level > 1e-6:
            level = target_level / reference_level
            if lighting.lit:
                # A scene lit from a few points is mostly dark, and its
                # 80th percentile is in the dark. Exposed for that, the
                # lit road is a white sheet; a camera holds its highlights.
                brightest = percentile(signal, _HIGHLIGHT_PERCENTILE)
                level = min(level, _HIGHLIGHT_LEVEL / max(brightest, 1e-6))
            signal = signal * level
            if exposed is not None:
                exposed[0] = float(level)
            if retro is not None:
                # Sheeting and glass beads send the illuminator's light
                # back down its own beam, to a lens that is beside it: many
                # times what a white surface sends. Added once the camera
                # has exposed for the scene, which is not for these.
                sheeting = retro < 0.0
                returned = np.where(
                    sheeting, np.maximum(reflectance, _SHEETING_FLOOR), reflectance
                )
                signal = signal + np.float32(level) * np.abs(retro) * returned * exposure

        # Gain deliberately does *not* rescale the signal here. On a real
        # camera gain is one of the things the AE loop uses to *reach* its
        # target, not something applied on top of a level it already hit.
        # Multiplying here as well drove the 80th percentile to
        # target x gain -- up to 0.6 x 1.6 = 0.96 -- so a fifth of the frame
        # sat against the clipping point and the near road washed out to a
        # featureless white plate covering 24% of the image. What gain does
        # control is the noise, below.

        # The lamps. They are sources and not surfaces: what they send the
        # lens is theirs, whatever the illuminator delivers where they are
        # and wherever the auto-exposure has put the scene, which is
        # exposed for the road and not for them. So they are added here, at
        # their own level, after both. White whatever their colour: with
        # the filter out a red lamp is a bright one and no more. (Channel
        # against channel: a maximum along an axis of three is far slower.)
        for light, glow in (
            (lighting.vehicle_light, lighting.vehicle_glow),
            (lighting.head_light, lighting.head_glow),
        ):
            if light is None or glow is None:
                continue
            if light.ndim == 3:
                light = np.maximum(np.maximum(light[..., 0], light[..., 1]), light[..., 2])
                glow = np.maximum(np.maximum(glow[..., 0], glow[..., 1]), glow[..., 2])
            signal = signal + light + np.float32(_LAMP_BLOOM) * glow

        # Shot noise scales as sqrt(signal); read noise does not scale at all,
        # which is why the shadows of a gained-up frame are the noisy part.
        rng = np.random.default_rng(noise_seed)
        noise = rng.standard_normal(signal.shape).astype(np.float32)
        signal = signal + noise * (np.sqrt(np.maximum(signal, 0.0)) * 0.02 + read_noise) * gain

        signal = np.clip(signal, 0.0, 1.0)

        # Near-monochrome, with the slight residual cast these sensors keep
        # rather than a mathematically perfect grey.
        tint = np.array([1.0 + self.residual_tint, 1.0, 1.0 - self.residual_tint * 0.5])
        out = signal[..., None] * tint.astype(np.float32)[None, None, :]
        np.clip(out, 0.0, 1.0, out=out)
        return from_linear_unit(out, scale, is_integer, img.dtype)
