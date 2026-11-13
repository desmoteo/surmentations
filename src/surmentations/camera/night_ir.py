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
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import record_draw
from surmentations.perception.geometry import focal_px
from surmentations.perception.labels import GROUND, VEGETATION, SceneClass, class_weight
from surmentations.utils.channels import percentile
from surmentations.utils.grids import on_coarse_grid, pixel_grid
from surmentations.utils.night_light import (
    even_road,
    fit_ground,
    headlamp_light,
    lamp_heads,
    lamps_in_space,
    scene_points,
    street_light,
)
from surmentations.utils.photometry import from_linear_unit, to_linear_unit
from surmentations.utils.road import RoadVehicle, vehicle_lamps, vehicles_as_seen
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

# The share of vehicles seen from the front, and from behind, whose lamps
# are on at night: the rest are parked. And how much of the glow round a
# lamp the sensor shows: a source far over full scale blooms on it whatever
# the air is like.
_LIT_AT_NIGHT = (0.85, 0.85)
_LAMP_BLOOM = 0.3
# How much further over full scale a lamp is at night than in daylight
# rain, which is what the lamps' own strengths are set for: the lamp is the
# same lamp, and the camera has opened up for the dark by about this much.
_NIGHT_LAMP_OVER = 8.0
# A dipped beam on the road just ahead of its vehicle, in unit exposures,
# at a lamp strength of 1.
_BEAM_LEVEL = 1.2
# The field of view a frame is taken to have where nothing says: the
# lighting needs the scene in space, and is not sensitive to it.
_ASSUMED_FOV_DEG = 60.0
# The auto-exposure, once lamps light the scene, holds the brightest
# hundredth of its surfaces to this level: a camera that exposed a lit road
# for its dark verges would show the road as a white sheet.
_HIGHLIGHT_LEVEL = 0.95
_HIGHLIGHT_PERCENTILE = 99.0
# A piece of the scene's lamp label is a street lamp when it is this many
# times taller than wide, at least, and this share of the frame tall.
_LABELLED_POST = (1.6, 0.08)


def _labelled_lamps(labels: np.ndarray) -> np.ndarray:
    """The street lamps a scene's labels name, as a vehicle reader would give them.

    For a provider with no reader. A tall piece of the lamp label is a
    street lamp, post and arm and head in one; its head is the end of its
    top that is furthest from its post. The labels find the nearest lamps
    of a road and miss the thin far ones, so a road lit by these alone is
    lit near the camera and dark beyond.
    """
    height = labels.shape[0]
    mask = (labels == int(SceneClass.LAMP)).astype(np.uint8)
    count, parts, stats, _centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lamps = []
    for index in range(1, count):
        left, top, wide, tall, _area = (int(value) for value in stats[index])
        if tall < _LABELLED_POST[0] * wide or tall < _LABELLED_POST[1] * height:
            continue
        rows, columns = np.nonzero(parts[top : top + tall, left : left + wide] == index)
        post = float(np.median(columns[rows > 0.6 * tall]))
        head = columns[rows <= max(1, tall // 10)]
        head_x = float(head[np.argmax(np.abs(head - post))])
        lamps.append((left, top, left + wide, top + tall, left + head_x, top + 0.02 * tall, 1.0))
    return np.array(lamps, dtype=np.float32).reshape(-1, 7)


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
        views = self.depth_provider.get_vehicles(scene)
        lit: list[tuple[RoadVehicle, list[tuple[float, float]]]] = []
        seen: list[RoadVehicle] = []
        emitted = np.zeros((height, width), dtype=np.float32)
        glowing = np.zeros((height, width), dtype=np.float32)
        if lamps > 0.0 and views and labels is not None:
            seen = vehicles_as_seen(labels == int(SceneClass.VEHICLE), views)
            light, glow, _cores = vehicle_lamps(
                None,
                seen,
                np.random.default_rng(lamp_seed),
                lamps * _NIGHT_LAMP_OVER,
                shape=(height, width),
                lit_share=_LIT_AT_NIGHT,
                lit=lit,
            )
            # White whatever their colour: with the filter out a red lamp
            # is a bright one and no more.
            # (Channel against channel: a maximum taken along an axis of
            # three is thirty times as slow.)
            emitted += np.maximum(np.maximum(light[..., 0], light[..., 1]), light[..., 2])
            glowing += np.maximum(np.maximum(glow[..., 0], glow[..., 1]), glow[..., 2])

        # What the lamps light. In space, so the scene's ground has to be
        # known, and a plane fitted through it.
        posts = np.zeros((0, 7), dtype=np.float32)
        if street > 0.0 and labels is not None:
            found = self.depth_provider.get_street_lamps(scene)
            posts = _labelled_lamps(labels) if found is None or not len(found) else found
        thrown = None
        if labels is not None and (len(posts) or lit):
            camera = self.depth_provider.settle_camera(scene)
            focal = focal_px(width, _ASSUMED_FOV_DEG) if camera is None else camera.focal_px
            points, _step = scene_points(depth_m, focal)
            small = cv2.resize(labels, points.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            ground = np.isin(small, [int(one) for one in GROUND]).astype(np.float32)
            plane = fit_ground(points, ground)
            if plane is not None:
                middle, up = plane
                reaching = np.zeros(points.shape[:2], dtype=np.float32)
                if len(posts):
                    placed = lamps_in_space(posts, focal, (height, width), middle, up)
                    reaching += np.float32(street) * street_light(points, up, placed)
                    burning, halo = lamp_heads((height, width), posts, street)
                    emitted += burning
                    glowing += halo
                beams = []
                for vehicle, places in lit:
                    # Its front points from its middle to its lamps where
                    # it is coming on, and the other way where it is seen
                    # from behind.
                    at_x = float(np.mean([x for x, _y in places]))
                    at_y = float(np.mean([y for _x, y in places]))
                    ahead = np.array(
                        [at_x - float(vehicle.columns.mean()), at_y - float(vehicle.rows.mean())]
                    )
                    if vehicle.away > 0.0:
                        ahead = -ahead
                    long = float(np.hypot(*ahead))
                    ahead = ahead / long if long > 1e-3 else np.array([0.0, -vehicle.away])
                    foot = (float(vehicle.foot[1]), float(vehicle.foot[0]))
                    beams.append((foot, (float(ahead[0]), float(ahead[1])), vehicle.away > 0.0))
                if beams:
                    reaching += np.float32(_BEAM_LEVEL * lamps) * headlamp_light(
                        points, middle, up, focal, (height, width), beams
                    )
                thrown = cv2.resize(reaching, (width, height), interpolation=cv2.INTER_LINEAR)
        if thrown is not None:
            drawn["lamp_exposure"] = thrown
            if len(posts):
                # Lit by street lamps, the road is a road at night: the
                # day's light and shade go, and the camera exposes for the
                # pools. Under headlamps alone it stays the illuminator's.
                drawn["road"] = cv2.GaussianBlur(
                    (labels == int(SceneClass.ROAD)).astype(np.float32), (0, 0), sigmaX=1.5
                )
        if float(emitted.max()) > 0.0:
            drawn["lamp_light"], drawn["lamp_glow"] = emitted, glowing
        record_draw(
            vehicle_lamps=lamps,
            vehicles_read=len(seen),
            vehicles_lit=len(lit),
            street_lamps=int(len(posts)) if thrown is not None else 0,
            street_lamp_level=street,
        )
        return drawn

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
        lamp_light: np.ndarray | None = None,
        lamp_glow: np.ndarray | None = None,
        lamp_exposure: np.ndarray | None = None,
        road: np.ndarray | None = None,
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
        if lamp_exposure is not None:
            # Lit by lamps: the road loses the day's light and shade, which
            # no lamp explains, and takes the pools and the beams.
            if road is not None:
                reflectance = even_road(reflectance, road)
            arriving = arriving + lamp_exposure
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
            if lamp_exposure is not None and road is not None:
                # A scene lit from a few points is mostly dark, and its
                # 80th percentile is in the dark. Exposed for that, the
                # lit road is a white sheet; a camera holds its highlights.
                brightest = percentile(signal, _HIGHLIGHT_PERCENTILE)
                level = min(level, _HIGHLIGHT_LEVEL / max(brightest, 1e-6))
            signal = signal * level

        # Gain deliberately does *not* rescale the signal here. On a real
        # camera gain is one of the things the AE loop uses to *reach* its
        # target, not something applied on top of a level it already hit.
        # Multiplying here as well drove the 80th percentile to
        # target x gain -- up to 0.6 x 1.6 = 0.96 -- so a fifth of the frame
        # sat against the clipping point and the near road washed out to a
        # featureless white plate covering 24% of the image. What gain does
        # control is the noise, below.

        if lamp_light is not None and lamp_glow is not None:
            # The traffic's lamps. They are sources and not surfaces: what
            # they send the lens is theirs, whatever the illuminator
            # delivers where they are and wherever the auto-exposure has
            # put the scene, which is exposed for the road and not for
            # them. So they are added here, at their own level, after both.
            signal = signal + lamp_light + np.float32(_LAMP_BLOOM) * lamp_glow

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
