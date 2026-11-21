"""What lights a scene at night: its street lamps and its traffic's headlamps.

A photograph taken by day is lit evenly, by the sky. Turned into a night
frame with one level of ambient light, it is the same photograph in grey:
the road is as bright at three hundred metres as at thirty, and the sun's
own light and shade are still on it. A road at night is lit from a few
points. Each street lamp throws a pool under itself that falls off with the
square of the distance, each vehicle throws a beam on the road ahead of it,
and what lies between is nearly black.

Both need to know where things are in space and not only in the picture, so
this works on the scene's points: each pixel's position in the camera's own
frame, from its depth, with the ground a plane fitted through the pixels
that are labelled ground.

* :func:`scene_points`, :func:`fit_ground`: the scene in space.
* :func:`lamps_in_space`, :func:`street_light`: the lamps' pools.
* :func:`street_air`: the lamps' light on the air itself, which fog, rain
  and snow make visible.
* :func:`lamp_heads`: the lamps themselves, burning.
* :func:`vehicle_heading`, :func:`beam_origins`, :func:`headlamp_light`:
  the traffic's beams.
* :func:`even_road`: a road with the day's light and shade taken off it.

The numbers are the round ones of road lighting, set by hand and not taken
from a table: columns some 8 to 12 m tall, standing a few times their
height apart, which is what makes the scalloped pools of a lit road; and a
dipped headlamp that lights the road usefully for some tens of metres and
is cut off above the horizontal. The two documents below are where such
figures are laid down; neither was read for these.

References:

* CIE (2010). Lighting of Roads for Motor and Pedestrian Traffic.
  CIE 115:2010. Commission Internationale de l'Eclairage.
* Sun, B., Ramamoorthi, R., Narasimhan, S. G., Nayar, S. K. (2005). A
  Practical Analytic Single Scattering Model for Real Time Rendering. ACM
  SIGGRAPH 2005 Papers, 1040-1049.
  https://doi.org/10.1145/1186822.1073309 (the closed form for a point
  light's glow in a scattering medium)
* UNECE (2013). Regulation No. 112: Uniform provisions concerning the
  approval of motor vehicle headlamps emitting an asymmetrical passing beam.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import cv2
import numpy as np

from surmentations import _native

# The longer side of the grid the light is worked out on. Light on a
# surface changes slowly, and at a quarter of a megapixel it is the same
# picture as at eight.
_GRID = 256
# How a lamp's light falls off to the side of straight down: the power of
# the cosine of the angle from it, which takes in the luminaire's own
# spread and the slant the light meets the ground at.
_LAMP_SPREAD = 1.5
# What a lamp sends upwards and sideways, as a share of what it sends
# straight down: enough to show the crown of a tree beside it.
_LAMP_SPILL = 0.03
# The nearest a line of sight is taken to pass a lamp, in metres: nearer,
# it is looking into the luminaire, which is drawn as a lamp and not as air.
_NEAREST_PASS_M = 0.6
# A lamp is taken to stand between these heights, in metres, whatever its
# box and the depth map make of it.
_LAMP_HEIGHTS = (4.0, 14.0)
# A dipped beam: the distance at which it has fallen to half, one lamp's
# half width at the vehicle and how fast that grows with distance, how far
# it reaches at all, and the height above the road it lights things up to.
_BEAM_HALF_M = 14.0
_BEAM_WIDTH_M = (0.35, 0.12)
# A vehicle's two headlamps: how far apart, and what each puts on the road
# just ahead of itself, the two together being a little over 1 further on.
_HEADLAMPS_APART_M = 1.3
_EACH_LAMP = 0.6
# A pair of lamps on one end of a vehicle: how far apart and how high. And
# the scales a vehicle is allowed against the scene's: a ground plane is
# not out by more than this, and a pair of lamps misread may be.
_LAMPS_APART_M = 1.4
_LAMP_HIGH_M = 0.8
_VEHICLE_SCALES = (0.5, 3.0)
_BEAM_REACH_M = 75.0
_BEAM_HIGH_M = 1.2
# How long a vehicle is, for where the front of one seen from behind is.
_VEHICLE_LENGTH_M = 4.5
# An outline's long way is its vehicle's heading when it lies within this
# cosine of the line from the camera: further off, the outline is long
# for some other reason, two vehicles in one or a lorry seen side on.
_HEADING_AGREES = 0.5
# The spacing of a vehicle's lamps over half its length: what turns how
# far its outline leans from its lamps into the tangent of its angle.
_LAMPS_OVER_HALF_LENGTH = 1.4 / 2.25
# A thing stands in front of the ground, and is lit as itself, when it is
# nearer than this share of the distance to the ground on its line of sight.
_STANDS_IN_FRONT = 0.8
# How much of the day's light and shade is taken off a road, and over what
# share of the frame's diagonal the light is taken to have varied.
_EVEN = 0.8
_EVEN_REACH = 0.04
# A burning lamp's head: its size as a share of its post's height in the
# picture, how far over full scale it is, and its glow's size and share.
# Which way the light is going. A road at a grazing angle throws a beam's
# light on ahead of the vehicle: a camera the vehicle drives towards sees
# its pool brighter than one behind it does, which sees this share of it.
# And what is in the air scatters light on more than back: the share of
# its scattering that is forward, as the asymmetry of a Henyey-Greenstein
# phase function. Fog's own is about 0.85; less is taken, for the light
# scattered more than once, which comes from every side.
_BEAM_FROM_BEHIND = 0.4
_SCATTERS_FORWARD = 0.5
# The lit air along a line of sight is summed at nine points, evenly spaced
# as angles seen from the lamp, by Simpson's rule: each point's weight, to
# be divided by three. Nine are within 0.4% of a fine sum on average and
# within 1% at ninety-nine lines in a hundred.
_SIMPSON = (1.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 1.0)
# An angle from its sine, as far as the sine of a sixteenth of a turn: the
# arcsine's series, x (1 + x^2 (1/6 + x^2 (3/40 + x^2 15/336))).
_ARCSINE = (1.0 / 6.0, 3.0 / 40.0, 15.0 / 336.0)
_HEAD_SIZE = 0.01
_HEAD_PEAK = 30.0
_HEAD_GLOW = (3.5, 0.012)


def scene_points(depth_m: np.ndarray, focal_px: float) -> tuple[np.ndarray, float]:
    """Each pixel's place in space, on a coarse grid, and the grid's scale.

    ``(h, w, 3)`` float32 in the camera's frame, metres: x to the right, y
    down the picture, z along the lens's axis. The second value is how
    many of the frame's pixels one of the grid's is.
    """
    height, width = depth_m.shape
    step = max(1.0, max(height, width) / _GRID)
    rows, columns = max(2, round(height / step)), max(2, round(width / step))
    depth = cv2.resize(depth_m.astype(np.float32), (columns, rows), interpolation=cv2.INTER_AREA)
    x = (np.arange(columns, dtype=np.float32) + 0.5) * (width / columns) - 0.5 * width
    y = (np.arange(rows, dtype=np.float32) + 0.5) * (height / rows) - 0.5 * height
    points = np.empty((rows, columns, 3), dtype=np.float32)
    points[..., 0] = depth * (x[None, :] / focal_px)
    points[..., 1] = depth * (y[:, None] / focal_px)
    points[..., 2] = depth
    return points, float(width / columns)


def fit_ground(points: np.ndarray, ground: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The plane of the ground: a point on it and the direction up from it.

    Fitted through the points that are ground (``ground``, a weight on the
    grid of ``points``), twice, the second time without what the first
    left far off it: kerbs, a bank, a depth map's mistakes. ``None`` with
    too little ground to fit a plane to. Up is the side the camera is on.
    """
    chosen = points[ground > 0.5]
    if len(chosen) < 30:
        return None
    for _ in range(2):
        middle = chosen.mean(axis=0)
        _u, _s, vectors = np.linalg.svd(chosen - middle, full_matrices=False)
        up = vectors[2]
        off = (chosen - middle) @ up
        near = np.abs(off) <= 2.0 * max(float(off.std()), 1e-3)
        if int(near.sum()) < 30:
            break
        chosen = chosen[near]
    if float(-middle @ up) < 0.0:
        up = -up
    return middle.astype(np.float32), up.astype(np.float32)


def _on_ground(ray: np.ndarray, middle: np.ndarray, up: np.ndarray) -> np.ndarray | None:
    """Where a line of sight from the lens meets the ground's plane; ``None`` if it never does."""
    falling = float(ray @ up)
    if falling > -1e-4:
        return None
    return ray * (float(middle @ up) / falling)


def lamps_in_space(
    lamps: np.ndarray, focal_px: float, shape: tuple[int, int], middle: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """Where each street lamp's head is, in the camera's frame, and how high it stands.

    ``lamps`` are as :func:`~surmentations.perception.vehicles.lamp_posts`
    gives them. A post is thin and a depth map is poor at thin things, so
    its depth is not read: its foot is where the line of sight through the
    bottom of its box meets the ground, and its head is on the line of
    sight through the head, where that passes nearest the upright through
    the foot. So a lamp is in space where it is seen in the picture, at
    the end of its arm; it used to be put over its foot, and its light
    fell from the top of the post while the lamp burned beside it. A row
    of the result is the head's three coordinates and its height. A lamp
    whose foot is not on the ground in front of the camera is left out.
    """
    height, width = shape
    placed = []
    for x0, _y0, x1, y1, head_x, head_y, _score in lamps.reshape(-1, 7).tolist():
        foot_ray = np.array(
            [(0.5 * (x0 + x1) - 0.5 * width) / focal_px, (y1 - 0.5 * height) / focal_px, 1.0]
        )
        foot = _on_ground(foot_ray, middle, up)
        if foot is None or foot[2] <= 0.0:
            continue
        head_ray = np.array(
            [(head_x - 0.5 * width) / focal_px, (head_y - 0.5 * height) / focal_px, 1.0]
        )
        # The head is on its own line of sight, where that line passes
        # nearest the upright through the foot: at the end of its arm, and
        # not over the foot, which is where the post is.
        along_up = float(head_ray @ up)
        apart = float(head_ray @ head_ray) - along_up**2
        head = None
        if apart > 1e-6:
            reach = (float(head_ray @ foot) - along_up * float(up @ foot)) / apart
            if reach > 0.0:
                head = head_ray * reach
        tall = _LAMP_HEIGHTS[0] if head is None else float((head - middle) @ up)
        if head is None or not _LAMP_HEIGHTS[0] <= tall <= _LAMP_HEIGHTS[1]:
            # No such point, or one no lamp is at: at a lamp's height on the
            # same line of sight, so that it is still seen where it is.
            tall = float(np.clip(tall, *_LAMP_HEIGHTS))
            head = foot + tall * up
            if abs(along_up) > 0.05:
                reach = (tall + float(middle @ up)) / along_up
                if reach > 0.0:
                    head = head_ray * reach
        placed.append((*head.tolist(), tall))
    return np.array(placed, dtype=np.float32).reshape(-1, 4)


def street_light(points: np.ndarray, up: np.ndarray, lamps: np.ndarray) -> np.ndarray:
    """How much of each lamp's light reaches each point, ``(h, w)`` float32.

    1 on the ground straight under a lamp, falling off with the square of
    the distance from its head and with the angle from straight down, and
    summed over the lamps: the pools of a lit road, overlapping where the
    columns stand close. A point above a lamp gets only its spill.
    """
    light = np.zeros(points.shape[:2], dtype=np.float32)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    # How high each point stands along the plane's own upright, once.
    raised = x * up[0] + y * up[1] + z * up[2]
    heads = np.ascontiguousarray(lamps, dtype=np.float32).reshape(-1, 4)
    upright = np.asarray(up, dtype=np.float32).reshape(3)
    if raised.dtype == np.float32 and _native.street_light(
        light, np.ascontiguousarray(points), raised, heads, upright, _LAMP_SPILL
    ):
        return light
    for lamp_x, lamp_y, lamp_z, tall in lamps.reshape(-1, 4).tolist():
        dx, dy, dz = lamp_x - x, lamp_y - y, lamp_z - z
        far = np.sqrt(dx * dx + dy * dy + dz * dz)
        np.maximum(far, 0.5, out=far)
        over = float(lamp_x * up[0] + lamp_y * up[1] + lamp_z * up[2]) - raised
        below = np.clip(over / far, 0.0, 1.0)
        reach = (np.float32(tall) / far) ** 2
        light += reach * (_LAMP_SPILL + (1.0 - _LAMP_SPILL) * below**_LAMP_SPREAD)
    return light


def _halfway(
    one: tuple[np.ndarray, np.ndarray], other: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The direction halfway between two, each a cosine and a sine: their sum, made a unit."""
    cosine, sine = one[0] + other[0], one[1] + other[1]
    unit = np.float32(1.0) / np.sqrt(np.maximum(cosine * cosine + sine * sine, np.float32(1e-12)))
    return cosine * unit, sine * unit


def _air_of_lamp(
    air: np.ndarray,
    reach: np.ndarray,
    ray: tuple[np.ndarray, np.ndarray, np.ndarray],
    far: np.ndarray,
    rising: np.ndarray,
    lamp: np.ndarray,
    up: np.ndarray,
) -> None:
    """Add one lamp's light on the air into ``air``, and that times how far off into ``reach``.

    The definition of ``sm_street_air`` in ``native/sm_kernels.c``, which
    does the same float32 arithmetic in the same order.
    """
    one = np.float32(1.0)
    x, y, z, tall = (np.float32(value) for value in lamp)
    apart = x * x + y * y + z * z
    lamp_up = x * up[0] + y * up[1] + z * up[2]
    along = ray[0] * x + ray[1] * y + ray[2] * z
    passes = np.sqrt(np.maximum(apart - along * along, np.float32(_NEAREST_PASS_M**2)))
    nearness = one / passes
    # The nearest approach, wherever on the line it falls: how the way to
    # the lamp from it lies against up.
    over_lamp = (lamp_up - along * rising) * nearness
    # The line's two ends as tangents of angles seen from the lamp, and
    # those as directions; the seven between them by halving, three times.
    first, last = -along * nearness, (far - along) * nearness
    unit_first = one / np.sqrt(one + first * first)
    unit_last = one / np.sqrt(one + last * last)
    at: list[tuple[np.ndarray, np.ndarray]] = [(unit_first, first * unit_first)] * 9
    at[8] = (unit_last, last * unit_last)
    at[4] = _halfway(at[0], at[8])
    at[2], at[6] = _halfway(at[0], at[4]), _halfway(at[4], at[8])
    for odd in (1, 3, 5, 7):
        at[odd] = _halfway(at[odd - 1], at[odd + 1])
    # The angle between two neighbours, from its sine.
    sine = at[0][0] * at[1][1] - at[0][1] * at[1][0]
    square = sine * sine
    sixth, fortieth, last_term = (np.float32(term) for term in _ARCSINE)
    step = sine * (one + square * (sixth + square * (fortieth + square * last_term)))

    gain = np.float32(_SCATTERS_FORWARD)
    spill = np.float32(_LAMP_SPILL)
    least, twice, most = one + gain * gain, np.float32(2.0) * gain, one - gain * gain
    summed = np.zeros(far.shape, dtype=np.float32)
    for weight, (cosine, sine_here) in zip(_SIMPSON, at, strict=True):
        below = np.clip(cosine * over_lamp - sine_here * rising, 0.0, 1.0)
        # Light on its way to the lens is scattered through no angle at
        # all, and its cosine is 1: before the nearest approach.
        turned = least + twice * sine_here
        forward = most / (turned * np.sqrt(turned))
        summed += np.float32(weight) * (forward * (spill + (one - spill) * (below * below)))
    lit = (tall * tall) * (step * np.float32(1.0 / 3.0) * summed * nearness)
    air += lit
    reach += lit * np.clip(along, 0.0, far)


def street_air(
    points: np.ndarray, up: np.ndarray, lamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The light the street lamps throw on the air along each line of sight, and how far off.

    Two ``(h, w)`` float32 maps.

    For each pixel, the lamps' light summed along its line of sight from
    the lens to what it shows, in the units of :func:`street_light` times
    metres. Times an extinction coefficient and the phase function of
    scattering that is the same in all directions, it is the light that
    fog, rain or snow in that air sends to the lens: the cone of lit air
    under each lamp.

    A lamp's light falls as the square of the distance from it. Along a
    straight line that passes the lamp at a distance ``h``, seen from the
    lamp as an angle ``phi`` from the nearest approach, that is ``d phi /
    h`` (Sun et al., 2005), and two things weigh each ``d phi`` of it:

    * **How far below the lamp** that point of the line is, as the lamp's
      light on a surface is weighed: a luminaire throws downwards, and the
      lit air is a cone under it and not a ball round it. It used to be
      taken once, at the nearest approach, for the whole line.
    * **Which way the light is going there against the way to the lens.**
      What is in the air scatters light on more than back or aside
      (:data:`_SCATTERS_FORWARD`): the air between a lamp and the camera,
      where the light is coming on, sends more of it to the lens than the
      air beyond the lamp does, and a cone seen from the side less than
      either was drawn as. The weight is 1 for scattering that is the
      same all round.

    The sum over ``phi`` is Simpson's rule on nine points evenly spaced in
    angle. No angle is ever taken: a point at ``s`` along a line that
    passes the lamp at ``h`` is seen from the lamp in the direction
    ``(h, s)``, the direction halfway between two is their sum, and the
    angle between two neighbours comes from its sine. So the whole sum is
    multiplications and square roots, which is what lets a compiler work
    it eight pixels at a time (``sm_street_air``); it took an arctangent,
    a sine and a cosine a point, at eight points, and five times as long.

    The second map is how far along the line of sight that lit air is, in
    metres, taking each lamp's at its nearest approach: what is in the air
    dims the light on its way to the lens, and a lamp's cone two hundred
    metres off in falling snow is not there to be seen.
    """
    x, y, z = (np.ascontiguousarray(points[..., axis], dtype=np.float32) for axis in range(3))
    far = np.sqrt(x * x + y * y + z * z)
    np.maximum(far, np.float32(1e-3), out=far)
    ray = (x / far, y / far, z / far)
    upright = np.asarray(up, dtype=np.float32).reshape(3)
    rising = ray[0] * upright[0] + ray[1] * upright[1] + ray[2] * upright[2]
    air = np.zeros(far.shape, dtype=np.float32)
    reach = np.zeros(far.shape, dtype=np.float32)
    heads = np.ascontiguousarray(lamps, dtype=np.float32).reshape(-1, 4)
    if not _native.street_air(
        air, reach, ray, far, rising, heads, upright,
        _NEAREST_PASS_M**2, _LAMP_SPILL, _SCATTERS_FORWARD,
    ):
        for lamp in heads:
            _air_of_lamp(air, reach, ray, far, rising, lamp, upright)
    return air, reach / np.maximum(air, np.float32(1e-9))


class BeamSource(NamedTuple):
    """A vehicle whose headlamps are lit, as far as the picture places it.

    The fields:

    * ``foot``: the column and row where it stands on the road, at the
      middle of the end of it that is seen.
    * ``from_behind``: whether that end is its rear.
    * ``lean``: how far the middle of its outline lies to the right of the
      middle of its pair of lamps, in units of the distance between the
      lamps; ``None`` where no pair was found.
    * ``long_way``: the long way of its outline in the picture, as a unit
      vector, where the outline is long; ``None`` otherwise.
    * ``lamps``: the column and row of each of the pair of lamps found on
      the end that is seen; empty where no pair was found.
    """

    foot: tuple[float, float]
    from_behind: bool
    lean: float | None = None
    long_way: tuple[float, float] | None = None
    lamps: tuple[tuple[float, float], ...] = ()


def vehicle_heading(
    source: BeamSource,
    middle: np.ndarray,
    up: np.ndarray,
    focal_px: float,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Where a vehicle stands on the ground and which way it points along it.

    Two vectors in the camera's frame: the place of its foot on the
    ground's plane, and a unit vector along the plane. ``None`` for a
    vehicle whose foot is not on the ground in front of the camera.

    **Which end is seen** gives the heading to within a quarter turn: a
    vehicle seen from behind is driving away from the camera, and one seen
    from the front is coming at it, along the ground on the line from the
    camera.

    **How its outline lies about its lamps** gives the rest. The lamps are
    on the end that is seen, one at either side, and the rest of the
    vehicle is behind that end. Seen square on, the outline is centred on
    the lamps. Seen from one side as well, its body shows on that side,
    and the middle of the outline lies to that side of the lamps by half
    the vehicle's length times the sine of the angle, where the lamps are
    apart by their spacing times its cosine: the angle's tangent is the
    ``lean`` times the lamps' spacing over half a length. It is a ratio of
    two lengths in the picture, so it needs no distances; it is taken to
    45 degrees at the most.

    **Its outline's long way**, where no pair of lamps was found: a
    vehicle seen from above is long the way it points. Only where that
    lies near the line from the camera; an outline is long for other
    reasons, a rear and a side seen together among them.

    Two things are not used. Where a vehicle's lamps sit above or below
    the middle of its outline: on a car seen from behind at a low angle
    they are above it, and a heading taken from that threw the beam back
    at the camera. And the slope of the line between its lamps, which is
    the same angle in principle and far too fine a measurement: a few
    pixels' error in where a lamp was found, on ground seen nearly edge
    on, turned a van through a right angle.
    """
    height, width = shape
    ray = np.array(
        [
            (source.foot[0] - 0.5 * width) / focal_px,
            (source.foot[1] - 0.5 * height) / focal_px,
            1.0,
        ]
    )
    here = _on_ground(ray, middle, up)
    if here is None or here[2] <= 0.0:
        return None
    away = here - float(here @ up) * up
    length = float(np.linalg.norm(away))
    if length < 1e-4:
        return None
    away = away / length
    sense = 1.0 if source.from_behind else -1.0
    if source.lean is not None:
        # The body lies behind the end that is seen, to the side it leans.
        right = np.cross(away, up)
        right = right / max(float(np.linalg.norm(right)), 1e-6)
        if right[0] < 0.0:
            right = -right
        turned = float(np.clip(source.lean * _LAMPS_OVER_HALF_LENGTH, -1.0, 1.0))
        body = away + turned * right
        return here, sense * body / float(np.linalg.norm(body))
    if source.long_way is not None:
        step = 0.02 * float(np.hypot(height, width))
        there = _on_ground(
            np.array(
                [
                    (source.foot[0] + step * source.long_way[0] - 0.5 * width) / focal_px,
                    (source.foot[1] + step * source.long_way[1] - 0.5 * height) / focal_px,
                    1.0,
                ]
            ),
            middle,
            up,
        )
        if there is not None and there[2] > 0.0:
            own = there - here
            long = float(np.linalg.norm(own))
            agrees = float(own @ away) / max(long, 1e-6)
            if long > 1e-4 and abs(agrees) > _HEADING_AGREES:
                return here, sense * own / long * (1.0 if agrees > 0.0 else -1.0)
    return here, sense * away


def beam_origins(
    source: BeamSource,
    middle: np.ndarray,
    up: np.ndarray,
    focal_px: float,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Where a vehicle's two headlamps throw from: the middle of its front, its heading, its scale.

    The middle of its front end on the ground, in the camera's frame; the
    unit vector it points along (:func:`vehicle_heading`); and how many of
    the scene's units a metre of this vehicle is. The two headlamps are
    either side of that middle, across the heading. ``None`` for a
    vehicle that does not stand on the ground in front of the camera.

    **By its own lamps, where a pair was found.** They are the surest
    thing known of a vehicle: two points a known distance apart, at a
    known height, on a known end.

    * The **scale** is their spacing on the ground, across the line from
      the camera, against a real pair's 1.4 m. A camera's height is known
      roughly and the ground's plane with it; a car measured in that
      plane's units may be two cars long. Everything of the vehicle and of
      its beam is in its own metres: its length, how high its lamps are,
      how far apart its headlamps, how far its beam reaches.
    * The **end that is seen** is under the middle of the pair, at the
      height lamps are. Its lowest pixel, which was used before, is the
      corner of it nearest the camera.
    * The **front** is that end where the front is seen, and a vehicle's
      length along its heading from it where the rear is.

    The far end of its outline is not used: a semantic outline shared out
    among touching vehicles has pieces of its neighbours in it, and the
    front of one car was found on the bumper of the next.

    With no pair the vehicle stands at its foot and is taken at the
    scene's own scale.
    """
    stands = vehicle_heading(source, middle, up, focal_px, shape)
    if stands is None:
        return None
    here, heading = stands
    height, width = shape
    scale, end = 1.0, here
    if len(source.lamps) == 2:

        def on_plane(column: float, row: float, raised: float = 0.0) -> np.ndarray | None:
            """Where the line of sight through a pixel meets the ground, or a plane above it."""
            ray = np.array(
                [(column - 0.5 * width) / focal_px, (row - 0.5 * height) / focal_px, 1.0]
            )
            return _on_ground(ray, middle + raised * up, up)

        away = here - float(here @ up) * up
        away = away / max(float(np.linalg.norm(away)), 1e-6)
        right = np.cross(away, up)
        right = right / max(float(np.linalg.norm(right)), 1e-6)
        square_on = max(abs(float(heading @ away)), 0.5)
        # Twice: the lamps are above the ground by their own height, which
        # is in the vehicle's metres, and the second look cuts their lines
        # of sight at the height the first one found.
        for _look in range(2):
            raised = _LAMP_HIGH_M * scale
            first, second = (on_plane(column, row, raised) for column, row in source.lamps)
            if first is None or second is None:
                break
            # Across the line from the camera, which a picture measures
            # well, and not along it, which on ground seen nearly edge on
            # it does not.
            apart = abs(float((second - first) @ right)) / square_on
            scale = float(np.clip(apart / _LAMPS_APART_M, *_VEHICLE_SCALES))
        if first is not None and second is not None:
            between = (
                0.5 * (source.lamps[0][0] + source.lamps[1][0]),
                0.5 * (source.lamps[0][1] + source.lamps[1][1]),
            )
            raised = _LAMP_HIGH_M * scale
            under = on_plane(*between, raised)
            if under is not None and under[2] > 0.0:
                end = under - raised * up
    front = end + (_VEHICLE_LENGTH_M * scale * heading if source.from_behind else 0.0)
    return front, heading, scale


def headlamp_light(
    points: np.ndarray,
    middle: np.ndarray,
    up: np.ndarray,
    focal_px: float,
    shape: tuple[int, int],
    vehicles: Sequence[BeamSource],
    standing: np.ndarray | None = None,
) -> np.ndarray:
    """How much of the traffic's headlamps reaches each point, ``(h, w)`` float32.

    ``vehicles`` are those whose lamps are lit; where each stands and
    which way it points is :func:`vehicle_heading`.

    A vehicle has two headlamps, and throws two beams: from either side
    of the middle of its front (:func:`beam_origins`), each a narrow pool
    just ahead of it that widens and runs into the other within a few
    metres. Together they are about 1 on the road just ahead and fall to
    half in fourteen metres; and they light the road and what stands low
    on it, not what is above it: a dipped beam is cut off at the
    horizontal. The beams of a vehicle seen from behind show as the road
    lit beyond its front, and fainter (:data:`_BEAM_FROM_BEHIND`): a road
    met at a grazing angle throws light on ahead, at a camera the vehicle
    is driving towards and away from one behind it. Lengths are in the
    vehicle's own metres.

    **Where the road is** is the ground's plane, for every pixel that
    shows something at least as far off as the plane on its line of sight
    and is not known to stand on the ground (``standing``, a mask on the
    grid of ``points``: vehicles, people, posts, trees). The scene's own
    depth is used only for those and for what is nearer than the plane, a
    wall or a bank. A depth map is poor at a road in deep shade and puts
    it well beyond where the road is, and a beam that went by the depth
    lit nothing ahead of a car driving into the shade under a bridge.
    """
    light = np.zeros(points.shape[:2], dtype=np.float32)
    # The scene laid flat: each point by two distances along the ground,
    # and its height above it. Worked out once; a beam is then a matter of
    # two numbers a point.
    first = np.cross(up, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    if float(first @ first) < 1e-6:
        first = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    first = (first / np.linalg.norm(first)).astype(np.float32)
    second = np.cross(up, first).astype(np.float32)
    falling = points[..., 0] * up[0] + points[..., 1] * up[1] + points[..., 2] * up[2]
    reach = np.where(falling < -1e-4, float(middle @ up) / np.minimum(falling, -1e-4), 0.0)
    on_ground = (reach > 0.0) & (reach <= 1.0 / _STANDS_IN_FRONT)
    if standing is not None:
        on_ground &= ~standing
    placed = np.where(on_ground[..., None], points * reach[..., None], points).astype(np.float32)
    x, y, z = placed[..., 0] - middle[0], placed[..., 1] - middle[1], placed[..., 2] - middle[2]
    along_first = x * first[0] + y * first[1] + z * first[2]
    along_second = x * second[0] + y * second[1] + z * second[2]
    above = x * up[0] + y * up[1] + z * up[2]
    # Nothing well above any vehicle's beam, whatever the vehicle's scale.
    low = (above < 1.5 * _VEHICLE_SCALES[1] * _BEAM_HIGH_M).astype(np.float32)
    # Each lit vehicle's pair of beams as six numbers: where its front is
    # along the ground's two axes, which way it points along them, its
    # scale, and what the lens gets of its pool.
    beams = []
    for source in vehicles:
        thrown = beam_origins(source, middle, up, focal_px, shape)
        if thrown is None:
            continue
        front, heading, scale = thrown
        front = front - middle
        # Seen from ahead of the vehicle the road throws the beam's light
        # at the lens; seen from behind it, away from it.
        to_it = front - float(front @ up) * up + middle - float(middle @ up) * up
        coming = 0.5 * (1.0 - float(heading @ to_it) / max(float(np.linalg.norm(to_it)), 1e-6))
        beams.append(
            (
                float(front @ first),
                float(front @ second),
                float(heading @ first),
                float(heading @ second),
                float(scale),
                _BEAM_FROM_BEHIND + (1.0 - _BEAM_FROM_BEHIND) * coming,
            )
        )
    table = np.array(beams, dtype=np.float32).reshape(-1, 6)
    planes = [
        np.ascontiguousarray(plane, dtype=np.float32)
        for plane in (along_first, along_second, above)
    ]
    if _native.headlamp_beams(
        light,
        planes[0],
        planes[1],
        planes[2],
        low,
        table,
        [*_BEAM_WIDTH_M, _HEADLAMPS_APART_M / 2, _EACH_LAMP, _BEAM_HALF_M, _BEAM_REACH_M]
        + [_BEAM_HIGH_M],
    ):
        return light
    for front_first, front_second, ahead_first, ahead_second, scale, facing in table:
        du = along_first - front_first
        dv = along_second - front_second
        # In the vehicle's own metres: a beam is as long against its car
        # in the picture as it is on the road.
        along = (du * ahead_first + dv * ahead_second) / scale
        aside = (du * ahead_second - dv * ahead_first) / scale
        forward = np.maximum(along, 0.0)
        wide = _BEAM_WIDTH_M[0] + _BEAM_WIDTH_M[1] * forward
        # Two lamps, one either side of the middle: two pools on the road
        # just ahead of the vehicle, running into one a few metres on.
        beam = np.exp(-(((aside - _HEADLAMPS_APART_M / 2) / wide) ** 2))
        beam += np.exp(-(((aside + _HEADLAMPS_APART_M / 2) / wide) ** 2))
        beam *= _EACH_LAMP / (1.0 + (forward / _BEAM_HALF_M) ** 2)
        beam *= np.clip(along, 0.0, 1.0) * np.clip((_BEAM_REACH_M - along) / 15.0, 0.0, 1.0)
        beam *= facing
        light += (
            beam * low * np.clip(1.5 - above / (scale * np.float32(_BEAM_HIGH_M)), 0.0, 1.0)
        ).astype(np.float32)
    return light


def even_road(reflectance: np.ndarray, road: np.ndarray) -> np.ndarray:
    """A road's reflectance with the day's light and shade taken off it.

    A photograph taken in the sun has the sun's light on its road: a
    bright carriageway, the shadow of every car and tree. None of it is
    there at night, and relit by lamps it shows through as light and shade
    with no lamp to explain them. A road is very nearly one material, so
    what varies slowly across it is taken to have been the light, and is
    divided out towards the road's own middle value; its paint, its
    patches and its texture, which vary quickly, stay.

    ``road`` is the weight of what is labelled road, ``(H, W)``. Elsewhere
    the reflectance is returned as it was.
    """
    weight = road.astype(np.float32)
    if float(weight.sum()) < 50.0:
        return reflectance
    height, width = reflectance.shape
    reach = max(2.0, _EVEN_REACH * float(np.hypot(height, width)))
    small = max(1.0, reach / 4.0)
    size = (max(2, round(width / small)), max(2, round(height / small)))
    carried = cv2.resize(reflectance * weight, size, interpolation=cv2.INTER_AREA)
    counted = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)
    carried = cv2.GaussianBlur(carried, (0, 0), sigmaX=4.0)
    counted = cv2.GaussianBlur(counted, (0, 0), sigmaX=4.0)
    slow = cv2.resize(
        carried / np.maximum(counted, 1e-4), (width, height), interpolation=cv2.INTER_LINEAR
    )
    usual = float(np.median(reflectance[weight > 0.5])) if (weight > 0.5).any() else 0.0
    if usual <= 0.0:
        return reflectance
    lit = np.clip(slow / usual, 0.25, 4.0)
    evened = reflectance / (1.0 + np.float32(_EVEN) * (lit - 1.0))
    out: np.ndarray = reflectance + weight * (evened - reflectance)
    return out.astype(np.float32)


def lamp_heads(
    shape: tuple[int, int], lamps: np.ndarray, strength: float
) -> tuple[np.ndarray, np.ndarray]:
    """The street lamps themselves, burning: their light and the glow round them.

    Two ``(H, W)`` float32 maps over full scale, as a vehicle's lamps are
    drawn: a head a few pixels across, larger for a nearer lamp, far over
    full scale at ``strength`` 1, and a glow several times its size.
    ``lamps`` are as :func:`~surmentations.perception.vehicles.lamp_posts`
    gives them.
    """
    height, width = shape
    light = np.zeros((height, width), dtype=np.float32)
    glow = np.zeros((height, width), dtype=np.float32)
    for _x0, y0, _x1, y1, head_x, head_y, _score in lamps.reshape(-1, 7).tolist():
        size = float(np.clip(_HEAD_SIZE * (y1 - y0), 0.8, 0.012 * float(np.hypot(height, width))))
        for target, radius, peak in (
            (light, size, strength * _HEAD_PEAK),
            (glow, _HEAD_GLOW[0] * size, strength * _HEAD_PEAK * _HEAD_GLOW[1]),
        ):
            reach = int(3.0 * radius) + 1
            x0, x1 = max(0, int(head_x) - reach), min(width, int(head_x) + reach + 1)
            y_first, y_last = max(0, int(head_y) - reach), min(height, int(head_y) + reach + 1)
            if x0 >= x1 or y_first >= y_last:
                continue
            dx = np.arange(x0, x1, dtype=np.float32)[None, :] - np.float32(head_x)
            dy = np.arange(y_first, y_last, dtype=np.float32)[:, None] - np.float32(head_y)
            target[y_first:y_last, x0:x1] += np.float32(peak) * np.exp(
                -0.5 * (dx**2 + dy**2) / radius**2
            )
    return light, glow
