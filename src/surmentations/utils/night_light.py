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
* :func:`lamp_heads`: the lamps themselves, burning.
* :func:`headlamp_light`: the traffic's beams.
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
* UNECE (2013). Regulation No. 112: Uniform provisions concerning the
  approval of motor vehicle headlamps emitting an asymmetrical passing beam.
"""

from __future__ import annotations

import cv2
import numpy as np

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
# A lamp is taken to stand between these heights, in metres, whatever its
# box and the depth map make of it.
_LAMP_HEIGHTS = (4.0, 14.0)
# A dipped beam: the distance at which it has fallen to half, its half
# width at the vehicle and how fast that grows with distance, how far it
# reaches at all, and the height above the road it lights things up to.
_BEAM_HALF_M = 14.0
_BEAM_WIDTH_M = (1.2, 0.12)
_BEAM_REACH_M = 75.0
_BEAM_HIGH_M = 1.2
# How long a vehicle is, for where the front of one seen from behind is.
_VEHICLE_LENGTH_M = 4.5
# An outline's long way is its vehicle's heading when it lies within this
# cosine of the line from the camera: further off, the outline is long
# for some other reason, two vehicles in one or a lorry seen side on.
_HEADING_AGREES = 0.5
# How much of the day's light and shade is taken off a road, and over what
# share of the frame's diagonal the light is taken to have varied.
_EVEN = 0.8
_EVEN_REACH = 0.04
# A burning lamp's head: its size as a share of its post's height in the
# picture, how far over full scale it is, and its glow's size and share.
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
    bottom of its box meets the ground, and its head is as far above that
    as the box is tall at that distance. A row of the result is the head's
    three coordinates and its height. A lamp whose foot is not on the
    ground in front of the camera is left out.
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
        # The head is on its line of sight, at the distance at which it
        # stands over the foot: the two lines' closest approach, level.
        level_foot = foot - float(foot @ up) * up
        level_ray = head_ray - float(head_ray @ up) * up
        along = float(level_ray @ level_foot) / max(float(level_ray @ level_ray), 1e-6)
        tall = float((head_ray * along - foot) @ up)
        tall = float(np.clip(tall, *_LAMP_HEIGHTS))
        # Over its own foot. Its arm reaches a metre or two over the road
        # from there, which is less than this is good to.
        head = foot + tall * up
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
    for lamp_x, lamp_y, lamp_z, tall in lamps.reshape(-1, 4).tolist():
        dx, dy, dz = lamp_x - x, lamp_y - y, lamp_z - z
        far = np.sqrt(dx * dx + dy * dy + dz * dz)
        np.maximum(far, 0.5, out=far)
        over = float(lamp_x * up[0] + lamp_y * up[1] + lamp_z * up[2]) - raised
        below = np.clip(over / far, 0.0, 1.0)
        reach = (np.float32(tall) / far) ** 2
        light += reach * (_LAMP_SPILL + (1.0 - _LAMP_SPILL) * below**_LAMP_SPREAD)
    return light


def headlamp_light(
    points: np.ndarray,
    middle: np.ndarray,
    up: np.ndarray,
    focal_px: float,
    shape: tuple[int, int],
    vehicles: list[tuple[tuple[float, float], tuple[float, float] | None, bool]],
) -> np.ndarray:
    """How much of the traffic's headlamps reaches each point, ``(h, w)`` float32.

    ``vehicles`` are those whose lamps are lit, each as the place in the
    picture where it stands on the road (column and row of the middle of
    its foot), the long way of its outline in the picture where it has
    one, and whether it is seen from behind.

    **Which way a vehicle points** is which end of it is seen. One seen
    from behind is driving away from the camera and one seen from the
    front is coming at it: along the ground, on the line from the camera
    to the vehicle. That is exact for a vehicle seen end on, and where its
    outline is long, as it is seen from above, the outline's own long way
    is its heading, taken in the sense that agrees. Where its lamps are on
    its outline says nothing: on a car seen from behind at a low angle the
    tail lamps are above the middle of it, and a heading taken from them
    threw its beam back at the camera.

    A beam starts at the vehicle's front, which for one seen from behind
    is a vehicle's length further on; it runs along the road, widens as it
    goes, is 1 at the vehicle and falls to half in fourteen metres; and it
    lights the road and what stands low on it, not what is above it: a
    dipped beam is cut off at the horizontal. Most of the beam of a
    vehicle seen from behind is hidden by the vehicle; what shows is the
    road lit beyond it and to either side.
    """
    height, width = shape
    light = np.zeros(points.shape[:2], dtype=np.float32)
    # The scene laid flat: each point by two distances along the ground,
    # and its height above it. Worked out once; a beam is then a matter of
    # two numbers a point.
    first = np.cross(up, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    if float(first @ first) < 1e-6:
        first = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    first = (first / np.linalg.norm(first)).astype(np.float32)
    second = np.cross(up, first).astype(np.float32)
    x, y, z = points[..., 0] - middle[0], points[..., 1] - middle[1], points[..., 2] - middle[2]
    along_first = x * first[0] + y * first[1] + z * first[2]
    along_second = x * second[0] + y * second[1] + z * second[2]
    above = x * up[0] + y * up[1] + z * up[2]
    low = np.clip(1.5 - above / _BEAM_HIGH_M, 0.0, 1.0)
    step = 0.02 * float(np.hypot(height, width))
    for (column, row), long_way, from_behind in vehicles:
        here = _on_ground(
            np.array([(column - 0.5 * width) / focal_px, (row - 0.5 * height) / focal_px, 1.0]),
            middle,
            up,
        )
        if here is None or here[2] <= 0.0:
            continue
        # Away from the camera along the ground, or towards it.
        away = here - float(here @ up) * up
        length = float(np.linalg.norm(away))
        if length < 1e-4:
            continue
        heading = (away / length) * (1.0 if from_behind else -1.0)
        if long_way is not None:
            there = _on_ground(
                np.array(
                    [
                        (column + step * long_way[0] - 0.5 * width) / focal_px,
                        (row + step * long_way[1] - 0.5 * height) / focal_px,
                        1.0,
                    ]
                ),
                middle,
                up,
            )
            if there is not None and there[2] > 0.0:
                own = there - here
                long = float(np.linalg.norm(own))
                agrees = float(own @ heading) / max(long, 1e-6)
                if long > 1e-4 and abs(agrees) > _HEADING_AGREES:
                    heading = own / long * (1.0 if agrees > 0.0 else -1.0)
        front = here + (_VEHICLE_LENGTH_M * heading if from_behind else 0.0) - middle
        ahead = (float(heading @ first), float(heading @ second))
        du = along_first - np.float32(front @ first)
        dv = along_second - np.float32(front @ second)
        along = du * np.float32(ahead[0]) + dv * np.float32(ahead[1])
        aside = du * np.float32(ahead[1]) - dv * np.float32(ahead[0])
        forward = np.maximum(along, 0.0)
        wide = _BEAM_WIDTH_M[0] + _BEAM_WIDTH_M[1] * forward
        beam = np.exp(-((aside / wide) ** 2)) / (1.0 + (forward / _BEAM_HALF_M) ** 2)
        beam *= np.clip(along, 0.0, 1.0) * np.clip((_BEAM_REACH_M - along) / 15.0, 0.0, 1.0)
        light += (beam * low).astype(np.float32)
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
