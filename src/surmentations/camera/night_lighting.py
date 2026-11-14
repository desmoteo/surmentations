"""A night scene's lamps, from what a provider knows of it, as light.

A daylight photograph made into a night frame has to be lit by something.
:mod:`surmentations.utils.night_light` has the lamps' geometry; this module
asks a :class:`~surmentations.depth.provider.SharedDepthProvider` where the
lamps of one scene are and returns what they do, for the two transforms
that make a night of a frame
(:class:`~surmentations.camera.night_ir.NightIrMode`, in the near infrared,
and :class:`~surmentations.camera.night_color.NightColorMode`, in colour):

* the light the street lamps and the traffic's headlamps throw on each
  pixel's surface;
* the lamps themselves, burning, and the glow round them;
* the light they throw on the air along each line of sight, which a
  transform that puts fog, rain or snow in that air afterwards turns into
  the cone under a lamp and the wedge ahead of a headlamp
  (:func:`note_lit_air`).

With a provider that knows of no lamps, everything here is empty and a
night is lit as it was before there was any of this.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import cv2
import numpy as np

from surmentations.depth.provider import SharedDepthProvider
from surmentations.perception.geometry import focal_px
from surmentations.perception.labels import GROUND, SceneClass
from surmentations.utils.night_light import (
    BeamSource,
    fit_ground,
    headlamp_light,
    lamp_heads,
    lamps_in_space,
    scene_points,
    street_air,
    street_light,
)
from surmentations.utils.road import RoadVehicle, vehicle_lamps, vehicles_as_seen

# The share of vehicles seen from the front, and from behind, whose lamps
# are on at night: the rest are parked.
_LIT_AT_NIGHT = (0.85, 0.85)
# How much further over full scale a lamp is at night than in daylight
# rain, which is what the lamps' own strengths are set for: the lamp is the
# same lamp, and the camera has opened up for the dark by about this much.
_NIGHT_LAMP_OVER = 8.0
# A dipped beam on the road just ahead of its vehicle, in unit exposures,
# at a lamp strength of 1: several times what a street lamp puts under
# itself, as the pool ahead of a car is on a lit road.
_BEAM_LEVEL = 4.0
# An outline this many times longer than wide has a long way to go by.
_LONG_OUTLINE = 1.3
# The field of view a frame is taken to have where nothing says: the
# lighting needs the scene in space, and is not sensitive to it.
_ASSUMED_FOV_DEG = 60.0
# A piece of the scene's lamp label is a street lamp when it is this many
# times taller than wide, at least, and this share of the frame tall.
_LABELLED_POST = (1.6, 0.08)
# What stands on the ground and is never the ground itself, whatever its
# depth is against the ground's.
_STANDING = (
    SceneClass.VEHICLE,
    SceneClass.PERSON,
    SceneClass.POLE,
    SceneClass.LAMP,
    SceneClass.FOLIAGE,
    SceneClass.SKY,
)
# The share of a post's box, from its top, in which its head is looked for.
_HEAD_REGION = 0.15
# An arm reaches at least this share of its post's box out from the post.
_ARM_LEAST = 0.25
# How deep the slab of air a dipped beam lights is, in metres, and the
# least sine of the angle a line of sight is taken to cross it at.
_BEAM_SLAB_M = 0.8
_SHALLOWEST = 0.15


class SceneLighting(NamedTuple):
    """What the lamps of one scene do, as :func:`light_scene` works it out.

    Maps are ``(H, W)`` float32 at the frame's size unless said otherwise,
    and ``None`` where there is nothing of the kind.

    The fields:

    * ``street``: the street lamps' light on each pixel's surface, in unit
      exposures: the strength asked for on the ground under a lamp.
    * ``beams``: the traffic's headlamps' light on each surface, the same.
    * ``in_beam``: how far each pixel is in a headlamp's beam, 0 to 1.
    * ``vehicle_light``, ``vehicle_glow``: the traffic's lamps themselves
      and the glow round them, ``(H, W, 3)`` in their own colours, over
      full scale.
    * ``head_light``, ``head_glow``: the street lamps themselves, burning,
      and their glow, colourless.
    * ``road``: the weight of the labelled road, where street lamps light
      it and it should lose the day's light and shade.
    * ``air_street``, ``air_beams``: the light each kind throws on the air
      along each line of sight, on a coarse grid, in unit exposures times
      metres (:func:`~surmentations.utils.night_light.street_air`).
    * ``air_street_far``, ``air_beams_far``: how far off along the line of
      sight each is, in metres, on the same grid.
    * ``vehicles_read``, ``vehicles_lit``, ``street_lamps``: how many of
      each were found and used.
    """

    street: np.ndarray | None = None
    beams: np.ndarray | None = None
    in_beam: np.ndarray | None = None
    vehicle_light: np.ndarray | None = None
    vehicle_glow: np.ndarray | None = None
    head_light: np.ndarray | None = None
    head_glow: np.ndarray | None = None
    road: np.ndarray | None = None
    air_street: np.ndarray | None = None
    air_beams: np.ndarray | None = None
    air_street_far: np.ndarray | None = None
    air_beams_far: np.ndarray | None = None
    vehicles_read: int = 0
    vehicles_lit: int = 0
    street_lamps: int = 0

    @property
    def lit(self) -> bool:
        """Whether any lamp throws light on the scene."""
        return self.street is not None or self.beams is not None


def labelled_lamps(labels: np.ndarray) -> np.ndarray:
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


def heads_by_label(posts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Street lamps with their heads moved to where the scene's labels show a luminaire.

    A reader gives each lamp a post and a head, and where it boxed no head
    it puts one at the middle of the top of the post's box
    (:func:`~surmentations.perception.vehicles.lamp_posts`). For a lamp
    whose arm reaches out to one side that is the empty air between the
    post and the luminaire. The labels often know better: a head with no
    box of its own is moved to the end, furthest from the post, of
    whatever is labelled lamp or post in the top of its post's box, and a
    lamp with an arm either side is given a head at each. A head the
    reader boxed is left where it is.
    """
    height, width = labels.shape
    moved: list[list[float]] = []
    for row in posts.reshape(-1, 7).tolist():
        x0, y0, x1, y1, head_x, head_y, score = row
        moved.append(row)
        if abs(head_x - 0.5 * (x0 + x1)) > 0.5 or abs(head_y - y0) > 0.5:
            continue
        slack = 0.2 * (x1 - x0) + 2.0
        left, right = max(0, int(x0 - slack)), min(width, int(x1 + slack) + 1)
        top = max(0, int(y0 - slack))
        bottom = min(height, int(y0 + _HEAD_REGION * (y1 - y0)) + 1)
        if right - left < 2 or bottom - top < 2:
            continue
        region = labels[top:bottom, left:right]
        metal = (region == int(SceneClass.LAMP)) | (region == int(SceneClass.POLE))
        if int(metal.sum()) < 6:
            continue
        rows, columns = np.nonzero(metal)
        # The post is the column the metal in the lower part of the box
        # stands in, and the box's own middle where the labels show none:
        # a lamp with one arm is not in the middle of its box.
        lower = labels[bottom : min(height, int(y1) + 1), left:right]
        tall = ((lower == int(SceneClass.LAMP)) | (lower == int(SceneClass.POLE))).sum(axis=0)
        # The column with the most of it: the post runs the height of the
        # box, and whatever else is labelled metal there does not.
        post = float(np.argmax(tall)) if int(tall.max(initial=0)) >= 6 else 0.5 * (x0 + x1) - left
        # A luminaire is at the end of an arm: as far from the post as the
        # metal in the top of the box goes, on either side that has an arm.
        # Not the middle of what is labelled lamp, which is often arm and
        # post and all.
        heads = []
        for side in (-1.0, 1.0):
            reach = side * (columns - post)
            if float(reach.max()) < _ARM_LEAST * (x1 - x0):
                continue
            end = reach >= np.percentile(reach[reach > 0.0], 85.0)
            heads.append((left + float(columns[end].mean()), top + float(rows[end].mean())))
        for index, (at_x, at_y) in enumerate(heads):
            if index == 0:
                moved[-1] = [x0, y0, x1, y1, at_x, at_y, score]
            else:
                moved.append([x0, y0, x1, y1, at_x, at_y, score])
    return np.array(moved, dtype=np.float32).reshape(-1, 7)


def _beam_source(vehicle: RoadVehicle, places: list[tuple[float, float]]) -> BeamSource:
    """A lit vehicle as the beams' geometry needs it: its foot, its lean, its lamps."""
    offsets = np.stack(
        [vehicle.columns - vehicle.columns.mean(), vehicle.rows - vehicle.rows.mean()]
    )
    values, vectors = np.linalg.eigh(np.cov(offsets) + 1e-6 * np.eye(2))
    long_way = None
    if values[1] > _LONG_OUTLINE**2 * values[0]:
        # The long way of its outline: seen from above, the way it points.
        long_way = (float(vectors[0, 1]), float(vectors[1, 1]))
    foot = (float(vehicle.foot[1]), float(vehicle.foot[0]))
    lean = None
    if len(places) == 2:
        apart = abs(places[1][0] - places[0][0])
        between = 0.5 * (places[0][0] + places[1][0])
        if apart >= 2.0:
            lean = (float(vehicle.columns.mean()) - between) / apart
    pair = tuple(sorted(places)) if lean is not None else ()
    return BeamSource(foot, vehicle.away > 0.0, lean, long_way, pair)


def light_scene(
    provider: SharedDepthProvider,
    scene: np.ndarray,
    labels: np.ndarray | None,
    depth_m: np.ndarray,
    lamp_seed: int,
    vehicle_lamp: float,
    street_lamp: float,
) -> SceneLighting:
    """Light one scene from the lamps its provider knows of.

    ``vehicle_lamp`` and ``street_lamp`` are the two strengths, 0 for
    none; ``lamp_seed`` decides which vehicles are lit and how brightly.

    * **The traffic's lamps** are lit on each vehicle the provider's reader
      read (:func:`~surmentations.utils.road.vehicle_lamps`), most and not
      all, and each lit vehicle throws its two beams
      (:func:`~surmentations.utils.night_light.headlamp_light`).
    * **The street lamps** are the reader's, or without one the scene's
      lamp label's (:func:`labelled_lamps`), and each throws its pool
      (:func:`~surmentations.utils.night_light.street_light`).

    The light is worked out in space, on a plane fitted through what is
    labelled ground, so nothing is thrown without labels or without ground;
    the traffic's lamps are still lit where a reader found them.
    """
    height, width = depth_m.shape
    views = provider.get_vehicles(scene)
    lit: list[tuple[RoadVehicle, list[tuple[float, float]]]] = []
    seen: list[RoadVehicle] = []
    vehicle_light = vehicle_glow = None
    if vehicle_lamp > 0.0 and views and labels is not None:
        seen = vehicles_as_seen(labels == int(SceneClass.VEHICLE), views)
        vehicle_light, vehicle_glow, _cores = vehicle_lamps(
            None,
            seen,
            np.random.default_rng(lamp_seed),
            vehicle_lamp * _NIGHT_LAMP_OVER,
            shape=(height, width),
            lit_share=_LIT_AT_NIGHT,
            lit=lit,
        )
        if not lit:
            vehicle_light = vehicle_glow = None
    posts = np.zeros((0, 7), dtype=np.float32)
    if street_lamp > 0.0 and labels is not None:
        found = provider.get_street_lamps(scene)
        posts = labelled_lamps(labels)
        if found is not None and len(found):
            posts = heads_by_label(found, labels)
    unthrown = SceneLighting(
        vehicle_light=vehicle_light,
        vehicle_glow=vehicle_glow,
        vehicles_read=len(seen),
        vehicles_lit=len(lit),
    )
    if labels is None or not (len(posts) or lit):
        return unthrown
    camera = provider.settle_camera(scene)
    focal = focal_px(width, _ASSUMED_FOV_DEG) if camera is None else camera.focal_px
    points, _step = scene_points(depth_m, focal)
    small = cv2.resize(labels, points.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    plane = fit_ground(points, np.isin(small, [int(one) for one in GROUND]).astype(np.float32))
    if plane is None:
        return unthrown
    middle, up = plane

    def enlarged(coarse: np.ndarray) -> np.ndarray:
        """A map on the points' grid at the frame's size."""
        full: np.ndarray = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR)
        return full

    street = head_light = head_glow = road = air_street = air_street_far = None
    if len(posts):
        placed = lamps_in_space(posts, focal, (height, width), middle, up)
        street = enlarged(np.float32(street_lamp) * street_light(points, up, placed))
        air_street, air_street_far = street_air(points, up, placed)
        air_street = np.float32(street_lamp) * air_street
        head_light, head_glow = lamp_heads((height, width), posts, street_lamp)
        # Lit by street lamps, the road is a road at night: the day's
        # light and shade go. Under headlamps alone it stays as it was.
        road = cv2.GaussianBlur(
            (labels == int(SceneClass.ROAD)).astype(np.float32), (0, 0), sigmaX=1.5
        )
    beams = in_beam = air_beams = air_beams_far = None
    if lit:
        sources = [_beam_source(vehicle, places) for vehicle, places in lit]
        stood = np.isin(small, [int(one) for one in _STANDING])
        beamed = headlamp_light(points, middle, up, focal, (height, width), sources, stood)
        level = np.float32(_BEAM_LEVEL * vehicle_lamp)
        beams = enlarged(level * beamed)
        in_beam = enlarged(np.minimum(beamed, 1.0))
        # A dipped beam lights a slab of air just over the road, and a line
        # of sight crosses it aslant: the lit air is the beam's own shape
        # on the road, by the length of the crossing.
        far = np.maximum(np.sqrt((points**2).sum(axis=2)), 1e-3)
        steep = np.clip(-(points @ up) / far, _SHALLOWEST, 1.0)
        air_beams = level * beamed * np.float32(_BEAM_SLAB_M) / steep
        air_beams_far = far.astype(np.float32)
    return SceneLighting(
        street,
        beams,
        in_beam,
        vehicle_light,
        vehicle_glow,
        head_light,
        head_glow,
        road,
        air_street,
        air_beams,
        air_street_far,
        air_beams_far,
        vehicles_read=len(seen),
        vehicles_lit=len(lit),
        street_lamps=int(len(posts)),
    )


def lit_air_of(
    lighting: SceneLighting, street_colour: np.ndarray, head_colour: np.ndarray
) -> np.ndarray | None:
    """The light a scene's lamps throw on its air, as a provider keeps it; ``None`` with none.

    ``(h, w, 4)`` float32 on the lighting's coarse grid: the street lamps'
    light in ``street_colour`` and the headlamps' in ``head_colour``,
    linear RGB, and in the fourth plane how far off along each line of
    sight the lit air is, in metres, weighted by how much each kind lights.
    """
    lit = None
    weight = reach = None
    for coarse, far, colour in (
        (lighting.air_street, lighting.air_street_far, street_colour),
        (lighting.air_beams, lighting.air_beams_far, head_colour),
    ):
        if coarse is None or far is None:
            continue
        tinted = coarse[..., None] * np.asarray(colour, dtype=np.float32).reshape(1, 1, 3)
        lit = tinted if lit is None else lit + tinted
        weight = coarse if weight is None else weight + coarse
        reach = coarse * far if reach is None else reach + coarse * far
    if lit is None or weight is None or reach is None:
        return None
    distance = reach / np.maximum(weight, 1e-9)
    return np.concatenate([lit, distance[..., None]], axis=2).astype(np.float32)


def note_lit_air(
    provider: SharedDepthProvider, params: dict[str, Any], result: dict[str, Any]
) -> None:
    """Tell a provider what light hangs in the air of the frame a night transform returned.

    ``params`` are the transform's own for the call, with the light on the
    air under ``"air"`` (:func:`lit_air_of`), and the exposure the transform
    settled on in the one-element list under ``"exposed"``: the two are
    known at different moments of a call, and their product is what the
    frame's own units make of the air's light. ``result`` is what the
    transform returned; its frame is the one the note is for
    (:meth:`~surmentations.depth.provider.SharedDepthProvider.set_lit_air`).
    """
    air, exposed = params.get("air"), params.get("exposed")
    frame = result.get("image", result.get("images"))
    if not isinstance(frame, np.ndarray):
        return
    lit = None
    if air is not None and exposed:
        lit = air.astype(np.float32, copy=True)
        lit[..., :3] *= np.float32(exposed[0])
    provider.set_lit_air(frame, lit)
