"""What lights a scene at night, and the far field of falling rain and snow.

Two things a daylight photograph does not have. ``utils.night_light`` lights
a scene from its street lamps and its traffic's headlamps, in space, and
``NightIrMode`` uses it where a provider knows where the lamps are.
``utils.precipitation.far_field_texture`` gives the veil of distant rain
and snow the grain of a count of particles, in front of each thing by as
much air as there is in front of it. The scenes here are drawn: a level
ground seen by a camera of known height and tilt, so that where a lamp
stands and what it lights can be said in metres.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest

from surmentations import mode
from surmentations.atmosphere import RandomRain, RandomSnow
from surmentations.atmosphere.night_fog import NightFog
from surmentations.atmosphere.night_rain import NightRain
from surmentations.atmosphere.night_snow import NightSnow
from surmentations.camera.night_color import NightColorMode
from surmentations.camera.night_ir import NightIrMode, retroreflectors
from surmentations.camera.night_lighting import heads_by_label
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import recording
from surmentations.perception import (
    InstallationGeometry,
    SceneClass,
    SceneLabelEstimator,
    VehicleView,
    VehicleViewEstimator,
)
from surmentations.perception.vehicles import lamp_posts
from surmentations.utils.night_light import (
    BeamSource,
    beam_origins,
    even_road,
    fit_ground,
    headlamp_light,
    lamp_heads,
    lamps_in_space,
    scene_points,
    street_air,
    street_light,
    vehicle_heading,
)
from surmentations.utils.precipitation import (
    far_field_texture,
    rain_number_density,
    snow_number_density,
)

HEIGHT, WIDTH = 240, 320
FOCAL = 280.0
CAMERA_M, PITCH_DEG = 8.0, 20.0


def _ground_depth(far_m: float = 400.0) -> np.ndarray:
    """Depth along the lens's axis of a level ground under a camera 8 m up, tilted down 20 degrees.

    Above the horizon, where a line of sight never meets the ground, the
    depth is ``far_m``.
    """
    pitch = np.deg2rad(PITCH_DEG)
    up = np.array([0.0, -np.cos(pitch), -np.sin(pitch)])
    x = (np.arange(WIDTH) + 0.5 - 0.5 * WIDTH) / FOCAL
    y = (np.arange(HEIGHT) + 0.5 - 0.5 * HEIGHT) / FOCAL
    falling = up[1] * y[:, None] + up[2] + 0.0 * x[None, :]
    depth = np.where(falling < -1e-3, -CAMERA_M / np.minimum(falling, -1e-3), far_m)
    return np.minimum(depth, far_m).astype(np.float32)


def _pixel_of(point: np.ndarray) -> tuple[float, float]:
    """The column and row at which a point of the camera's frame is seen."""
    return (
        float(point[0] / point[2] * FOCAL + 0.5 * WIDTH),
        float(point[1] / point[2] * FOCAL + 0.5 * HEIGHT),
    )


def _ground_at(ahead_m: float, right_m: float = 0.0, above_m: float = 0.0) -> np.ndarray:
    """A point that far ahead of the camera along the ground, to its right and above the ground."""
    pitch = np.deg2rad(PITCH_DEG)
    up = np.array([0.0, -np.cos(pitch), -np.sin(pitch)])
    forward = np.array([0.0, -np.sin(pitch), np.cos(pitch)])
    return -CAMERA_M * up + ahead_m * forward + right_m * np.array([1.0, 0.0, 0.0]) + above_m * up


def test_the_ground_s_plane_comes_back_from_its_depth() -> None:
    """A level ground under a camera 8 m up: the plane fitted through its
    points is 8 m below the lens and up is the way the camera was tilted
    from. A kerb's worth of things standing on it does not move it."""
    depth = _ground_depth()
    depth[150:170, 100:130] *= 0.9  # something standing on the ground
    points, step = scene_points(depth, FOCAL)
    assert points.shape[2] == 3 and step >= 1.0
    ground = (depth < 399.0).astype(np.float32)
    ground = ground[:: int(HEIGHT / points.shape[0]), :: int(WIDTH / points.shape[1])]
    plane = fit_ground(points, ground[: points.shape[0], : points.shape[1]])
    assert plane is not None
    middle, up = plane
    pitch = np.deg2rad(PITCH_DEG)
    assert float(up @ np.array([0.0, -np.cos(pitch), -np.sin(pitch)])) > 0.999
    assert float(-middle @ up) == pytest.approx(CAMERA_M, rel=0.05)
    assert fit_ground(points, np.zeros(points.shape[:2], np.float32)) is None


def test_a_lamp_stands_over_its_foot_as_tall_as_its_box() -> None:
    """A post 6 m tall, 30 m off: its head is placed 6 m over the foot
    of its box, by the ground and not by a depth map, which is poor at
    posts. One taller than any column is taken for a tall column."""
    points, _step = scene_points(_ground_depth(), FOCAL)
    middle, up = _ground_at(20.0), -_ground_at(0.0) / CAMERA_M
    foot, head = _pixel_of(_ground_at(30.0, 4.0)), _pixel_of(_ground_at(30.0, 4.0, 6.0))
    lamps = np.array(
        [[foot[0] - 3, head[1], foot[0] + 3, foot[1], head[0], head[1], 0.5]], np.float32
    )
    placed = lamps_in_space(
        lamps, FOCAL, (HEIGHT, WIDTH), middle.astype(np.float32), up.astype(np.float32)
    )
    assert placed.shape == (1, 4)
    assert float(placed[0, 3]) == pytest.approx(6.0, rel=0.1)
    assert np.allclose(placed[0, :3], _ground_at(30.0, 4.0, float(placed[0, 3])), atol=0.6)
    tall = lamps.copy()
    tall[0, 1] = tall[0, 5] = -400.0
    assert float(lamps_in_space(tall, FOCAL, (HEIGHT, WIDTH), middle, up)[0, 3]) == 14.0
    # A box that stands on nothing, above the horizon, is no lamp on this ground.
    sky = np.array([[100, 0, 106, 6, 103, 1, 0.5]], np.float32)
    assert lamps_in_space(sky, FOCAL, (HEIGHT, WIDTH), middle, up).shape == (0, 4)


def test_a_street_lamp_throws_a_pool_that_falls_off_with_distance() -> None:
    """1 straight under it, a quarter at its own height to the side and
    less, since the light meets the ground aslant; next to nothing fifty
    metres off; and two lamps light the ground between them together."""
    up = (-_ground_at(0.0) / CAMERA_M).astype(np.float32)
    spots = np.stack(
        [_ground_at(30.0), _ground_at(30.0, 10.0), _ground_at(80.0), _ground_at(30.0, 5.0, 12.0)]
    ).astype(np.float32)[None]
    lamp = np.array([[*_ground_at(30.0, 0.0, 10.0), 10.0]], np.float32)
    under, beside, far, above = street_light(spots, up, lamp)[0].tolist()
    assert under == pytest.approx(1.0, rel=0.02)
    assert 0.1 < beside < 0.5 * under
    assert far < 0.02
    assert above < 0.2 * under, "a lamp lighting what is above it as it lights the road"
    pair = np.array([[*_ground_at(30.0, 0.0, 10.0), 10.0], [*_ground_at(30.0, 20.0, 10.0), 10.0]])
    assert float(street_light(spots, up, pair.astype(np.float32))[0, 1]) > 1.9 * beside
    assert float(street_light(spots, up, np.zeros((0, 4), np.float32)).max()) == 0.0


def test_a_headlamp_lights_the_road_ahead_of_its_vehicle_and_not_what_is_above_it() -> None:
    """A beam along the road from the vehicle's front, strongest just ahead
    and gone by eighty metres; nothing behind it or far to the side; and a
    dipped beam does not light a wall above a metre and a half. A vehicle
    seen from behind has its front a length further on."""
    depth = _ground_depth()
    points, _step = scene_points(depth, FOCAL)
    middle, up = (
        _ground_at(20.0).astype(np.float32),
        (-_ground_at(0.0) / CAMERA_M).astype(np.float32),
    )
    foot = _pixel_of(_ground_at(25.0))
    frame = (HEIGHT, WIDTH)
    scale = HEIGHT / points.shape[0]

    def lit(beams: np.ndarray, ahead_m: float, right_m: float = 0.0) -> float:
        """A beam's light at a place on the ground."""
        column, row = _pixel_of(_ground_at(ahead_m, right_m))
        return float(beams[int(row / scale), int(column / scale)])

    # Seen from behind: driving away from the camera, up the picture.
    rear = headlamp_light(points, middle, up, FOCAL, frame, [BeamSource(foot, True)])
    assert lit(rear, 36.0) > 0.4 and lit(rear, 36.0) > lit(rear, 50.0) > lit(rear, 70.0) > 0.0
    assert lit(rear, 28.0) < 0.05, "a beam from the middle of a vehicle seen from behind"
    assert lit(rear, 20.0) == 0.0 and lit(rear, 15.0) == 0.0 and lit(rear, 40.0, 12.0) < 0.02
    # The same vehicle seen from the front: its beam is at the camera.
    front = headlamp_light(points, middle, up, FOCAL, frame, [BeamSource(foot, False)])
    assert lit(front, 18.0) > 0.3 and lit(front, 40.0) == 0.0
    # A wall across the road at 40 m, nearer than the ground behind it:
    # lit at its foot, dark at 3 m up.
    wall = points.copy()
    for above in (0.3, 3.0):
        wall[0, 0] = _ground_at(40.0, 0.0, above)
        got = headlamp_light(wall, middle, up, FOCAL, frame, [BeamSource(foot, True)])
        assert (float(got[0, 0]) > 0.2) is (above < 1.0)


def test_a_vehicle_points_the_way_its_outline_leans_from_its_lamps() -> None:
    """Which end is seen says away from the camera or at it. Seen from one
    side as well, the body shows on that side of the lamps, and the
    vehicle is turned that way by the angle the lean makes: none for an
    outline centred on its lamps, to the right for one that leans right,
    and never more than half a right angle. An outline that is long
    across the line from the camera says nothing."""
    middle, up = _ground_at(20.0), -_ground_at(0.0) / CAMERA_M
    foot = _pixel_of(_ground_at(25.0))
    frame = (HEIGHT, WIDTH)
    away = _ground_at(1.0) - _ground_at(0.0)
    right = np.array([1.0, 0.0, 0.0])

    def heading(**reading: object) -> np.ndarray:
        """Which way a vehicle standing 25 m off points, as read."""
        stands = vehicle_heading(BeamSource(foot, **reading), middle, up, FOCAL, frame)  # type: ignore[arg-type]
        assert stands is not None
        assert np.allclose(stands[0], _ground_at(25.0), atol=0.3)
        assert float(np.linalg.norm(stands[1])) == pytest.approx(1.0)
        return stands[1]

    assert float(heading(from_behind=True) @ away) > 0.999
    assert float(heading(from_behind=False) @ away) < -0.999
    assert float(heading(from_behind=True, lean=0.0) @ away) > 0.999
    turned = heading(from_behind=True, lean=0.6)
    assert 0.2 < float(turned @ right) < 0.5 and float(turned @ away) > 0.85
    assert float(heading(from_behind=True, lean=-0.6) @ right) < -0.2
    assert float(heading(from_behind=True, lean=9.0) @ away) == pytest.approx(
        np.sqrt(0.5), abs=0.01
    )
    # Seen from the front, the body is behind its lamps and it points the other way.
    coming = heading(from_behind=False, lean=0.6)
    assert float(coming @ away) < -0.85 and float(coming @ right) < -0.2
    # The long way of an outline, with no pair of lamps to go by: used
    # where it runs with the line from the camera, not across it.
    assert float(heading(from_behind=True, long_way=(1.0, 0.0)) @ away) > 0.999
    slanted = heading(from_behind=True, long_way=(0.5, -0.87))
    assert float(slanted @ right) > 0.05 and float(slanted @ away) > 0.5
    above_the_horizon = BeamSource((160.0, 2.0), True)
    assert vehicle_heading(above_the_horizon, middle, up, FOCAL, frame) is None


def _tail_lamps(ahead_m: float, apart_m: float = 1.4, high_m: float = 0.8) -> tuple:
    """Where the two lamps of a vehicle's near end are seen, that far ahead on the road."""
    return tuple(_pixel_of(_ground_at(ahead_m, side * apart_m / 2, high_m)) for side in (-1, 1))


def test_a_vehicle_s_beams_are_placed_and_sized_by_its_own_lamps() -> None:
    """A pair of lamps is two points a known distance apart at a known
    height. The end that is seen is under the middle of them; the front of
    a vehicle seen from behind is its own length further on; and a vehicle
    whose lamps are twice as far apart as the ground's plane allows is
    twice the size in that plane's units, beam and all."""
    middle, up = _ground_at(20.0), -_ground_at(0.0) / CAMERA_M
    frame = (HEIGHT, WIDTH)
    foot = _pixel_of(_ground_at(26.5, 0.9))  # the corner of it nearest the camera, not its middle
    seen = beam_origins(
        BeamSource(foot, True, 0.0, None, _tail_lamps(25.0)), middle, up, FOCAL, frame
    )
    assert seen is not None
    front, heading, scale = seen
    assert scale == pytest.approx(1.0, rel=0.1)
    assert np.allclose(front, _ground_at(29.5), atol=0.5), "not a length on from under its lamps"
    assert float(heading @ (_ground_at(1.0) - _ground_at(0.0))) > 0.999
    # Seen from the front, its front is where its lamps are.
    coming = beam_origins(
        BeamSource(foot, False, 0.0, None, _tail_lamps(25.0)), middle, up, FOCAL, frame
    )
    assert coming is not None and np.allclose(coming[0], _ground_at(25.0), atol=0.5)
    # Lamps 2.8 m apart on this plane: a plane out by two, and a car twice as long on it.
    big = beam_origins(
        BeamSource(foot, True, 0.0, None, _tail_lamps(25.0, 2.8, 1.6)), middle, up, FOCAL, frame
    )
    assert big is not None and big[2] == pytest.approx(2.0, rel=0.1)
    assert np.allclose(big[0], _ground_at(34.0), atol=1.0)
    # With no pair it stands at its foot, at the scene's own scale.
    alone = beam_origins(BeamSource(foot, True), middle, up, FOCAL, frame)
    assert alone is not None and alone[2] == 1.0
    assert np.allclose(alone[0], _ground_at(31.0, 0.9), atol=0.6)
    assert beam_origins(BeamSource((160.0, 2.0), True), middle, up, FOCAL, frame) is None


def test_a_vehicle_throws_two_beams_that_run_into_one() -> None:
    """One from each headlamp: two pools on the road just ahead of the
    vehicle, with less light between them, and one beam a few lengths on."""
    points, _step = scene_points(_ground_depth(), FOCAL)
    middle, up = (
        _ground_at(20.0).astype(np.float32),
        (-_ground_at(0.0) / CAMERA_M).astype(np.float32),
    )
    scale = HEIGHT / points.shape[0]
    source = BeamSource(_pixel_of(_ground_at(12.0)), True, 0.0, None, _tail_lamps(12.0))
    beams = headlamp_light(points, middle, up, FOCAL, (HEIGHT, WIDTH), [source])

    def lit(ahead_m: float, right_m: float = 0.0) -> float:
        """The light at a place on the ground."""
        column, row = _pixel_of(_ground_at(ahead_m, right_m))
        return float(beams[int(row / scale), int(column / scale)])

    # Its front is at 16.5 m. A metre and a half on, each lamp has its pool.
    assert lit(18.0, 0.65) > 1.4 * lit(18.0) and lit(18.0, -0.65) > 1.4 * lit(18.0)
    assert lit(18.0, 0.65) > 0.4
    # Fifteen metres on they are one beam, brightest in the middle.
    assert lit(31.0) > lit(31.0, 0.9) > 0.0
    assert lit(15.0) == 0.0, "light under the vehicle"


def test_a_beam_falls_on_the_road_whatever_the_depth_made_of_it() -> None:
    """A road in the deep shade under a bridge comes out of a depth map
    far beyond where it is. It is the ground all the same, and a car that
    drives into the shade lights it; but a car standing there, which the
    labels know, is lit as a car and not as the road beyond it."""
    points, _step = scene_points(_ground_depth(), FOCAL)
    middle, up = (
        _ground_at(20.0).astype(np.float32),
        (-_ground_at(0.0) / CAMERA_M).astype(np.float32),
    )
    foot = _pixel_of(_ground_at(25.0))
    frame = (HEIGHT, WIDTH)
    scale = HEIGHT / points.shape[0]
    column, row = (int(value / scale) for value in _pixel_of(_ground_at(38.0)))
    shaded = points.copy()
    shaded[row - 3 : row + 4, column - 3 : column + 4] *= 2.2
    beam = [BeamSource(foot, True)]
    plain = headlamp_light(points, middle, up, FOCAL, frame, beam)
    in_shade = headlamp_light(shaded, middle, up, FOCAL, frame, beam)
    assert float(in_shade[row, column]) == pytest.approx(float(plain[row, column]), rel=0.05)
    assert float(plain[row, column]) > 0.3
    # A bonnet at that pixel, a metre and a half up on a thing the labels
    # say stands there: lit as what it is, at the top of the beam, and not
    # as the road its line of sight comes down on a few metres beyond.
    stood = points.copy()
    stood[row, column] = _ground_at(36.0, 0.0, 1.5)
    standing = np.zeros(points.shape[:2], dtype=bool)
    standing[row, column] = True
    as_itself = float(headlamp_light(stood, middle, up, FOCAL, frame, beam, standing)[row, column])
    as_road = float(headlamp_light(stood, middle, up, FOCAL, frame, beam)[row, column])
    assert as_road > 0.25 and as_itself < 0.5 * as_road


def test_a_road_loses_the_day_s_light_and_shade_and_keeps_its_paint() -> None:
    """A shadow across a road in the sun is still there by night unless it
    is taken off. What varies slowly on a road was the light; what varies
    quickly is the road. Off the road nothing changes."""
    reflectance = np.full((HEIGHT, WIDTH), 0.3, dtype=np.float32)
    reflectance[100:160] = 0.12  # a broad shadow
    reflectance[:, 158:162] = 0.8  # a painted line
    reflectance[:, :60] = 0.5  # the verge
    road = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    road[:, 60:] = 1.0
    evened = even_road(reflectance, road)
    shade, sun = float(evened[125:135, 80:140].mean()), float(evened[30:60, 80:140].mean())
    assert shade / sun > 0.65, "the shadow is still on the road"
    assert float(evened[130, 160]) > 2.0 * shade, "the paint went with the shadow"
    assert np.array_equal(evened[:, :55], reflectance[:, :55])
    assert np.array_equal(even_road(reflectance, np.zeros_like(road)), reflectance)


def test_a_lamp_s_head_burns_where_its_post_ends() -> None:
    lamps = np.array([[100, 40, 110, 200, 104, 42, 0.5]], np.float32)
    light, glow = lamp_heads((HEIGHT, WIDTH), lamps, 1.0)
    row, column = np.unravel_index(int(light.argmax()), light.shape)
    assert (int(row), int(column)) == (42, 104) and float(light.max()) > 5.0
    assert float(light[120:].max()) == 0.0
    assert float(glow.max()) < 0.2 * float(light.max())
    assert int((glow > 0.3 * glow.max()).sum()) > 4 * int((light > 0.3 * light.max()).sum())


def test_street_lamps_are_the_tall_boxes_and_their_heads_the_flat_ones_on_top() -> None:
    """A detector asked for a street light answers with posts and with
    heads. A post is tall; its heads are the flat boxes at its very top,
    both of them for a column with two arms; a post down the road inside
    the box of a nearer one is still a post; a hoarding is neither."""
    found = np.array(
        [
            [100, 20, 130, 200, 0, 0.5],  # a post
            [122, 18, 134, 26, 0, 0.4],  # its head, at the end of its arm
            [200, 60, 260, 200, 0, 0.45],  # a column with two arms
            [198, 58, 212, 66, 0, 0.3],
            [248, 58, 262, 66, 0, 0.3],
            [110, 90, 118, 150, 0, 0.3],  # a further post, inside the first one's box
            [108, 100, 120, 108, 0, 0.3],  # a head half way down the first post: the further one's
            [20, 100, 90, 130, 0, 0.6],  # a hoarding
            [280, 20, 290, 200, 0, 0.1],  # not sure enough
        ],
        dtype=np.float32,
    )
    lamps = lamp_posts(found, HEIGHT)
    heads = sorted((round(float(row[4])), round(float(row[5]))) for row in lamps)
    assert heads == [(114, 90), (128, 22), (205, 62), (255, 62)]
    assert lamp_posts(np.zeros((0, 6), np.float32), HEIGHT).shape == (0, 7)


class _Ground(DepthEstimator):
    """Relative depth of the level ground the lighting tests use."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """The ground's depth as a share of the farthest of it."""
        depth = _ground_depth()
        return (depth / depth.max()).astype(np.float32)


class _Labels(SceneLabelEstimator):
    """A label map that is always the same one."""

    def __init__(self, labels: np.ndarray) -> None:
        """Keep the map."""
        self.labels = labels

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """The map, whatever the frame."""
        return self.labels


class _Lamps(VehicleViewEstimator):
    """A reader that finds no vehicles and the street lamps it is told of."""

    def __init__(self, lamps: np.ndarray) -> None:
        """Keep the lamps."""
        self.lamps = lamps

    def estimate(self, image: np.ndarray, vehicles: np.ndarray) -> list[VehicleView]:
        """No vehicles."""
        return []

    def street_lamps(self, image: np.ndarray) -> np.ndarray:
        """The lamps, whatever the frame."""
        return self.lamps


def _night_road() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A road in daylight with the sun's shadow across it, its labels, and one lamp beside it."""
    depth = _ground_depth()
    labels = np.where(depth < 399.0, int(SceneClass.ROAD), int(SceneClass.SKY)).astype(np.uint8)
    image = np.full((HEIGHT, WIDTH, 3), 120, dtype=np.uint8)
    image[labels == int(SceneClass.SKY)] = (150, 180, 230)
    foot, head = _pixel_of(_ground_at(30.0, 3.0)), _pixel_of(_ground_at(30.0, 3.0, 6.0))
    lamps = np.array(
        [[foot[0] - 3, head[1], foot[0] + 3, foot[1], head[0], head[1], 0.5]], np.float32
    )
    return image, labels, lamps


def _provider(labels: np.ndarray, lamps: np.ndarray | None) -> SharedDepthProvider:
    """A provider over the level ground: labels, a known lens, and a reader if there are lamps."""
    return SharedDepthProvider(
        _Ground(),
        depth_range_m=(float(_ground_depth().min()), 400.0),
        labels=_Labels(labels),
        geometry=InstallationGeometry(
            horizontal_fov_deg=float(np.rad2deg(2 * np.arctan(0.5 * WIDTH / FOCAL)))
        ),
        vehicles=None if lamps is None else _Lamps(lamps),
    )


def test_a_night_road_is_lit_under_its_lamps_and_dark_between() -> None:
    """A daylight road is as bright at a hundred metres as at thirty. By
    night with a street lamp it is bright under the lamp and dark far from
    it, the lamp itself burns, and the camera holds the lit road below
    white. Without lamps, or with them turned off, the frame is what it
    was."""
    image, labels, lamps = _night_road()
    settings = {"illuminator_range_m": (4.0, 4.0), "ambient_range": (0.02, 0.02)}

    def night(provider: SharedDepthProvider, **more: object) -> tuple[np.ndarray, dict]:
        """The road as the night camera sees it, and what was drawn."""
        transform = NightIrMode(depth_provider=provider, p=1.0, **settings, **more)  # type: ignore[arg-type]
        transform.set_random_seed(2)
        with recording() as log:
            out = transform(image=image)["image"].astype(np.float32).mean(axis=2)
        return out, log[0]["drew"]

    def at(picture: np.ndarray, ahead_m: float, right_m: float = 0.0) -> float:
        """The picture's level about a place on the ground."""
        column, row = (round(value) for value in _pixel_of(_ground_at(ahead_m, right_m)))
        return float(picture[row - 2 : row + 3, column - 2 : column + 3].mean())

    lit, drew = night(_provider(labels, lamps))
    assert drew["street_lamps"] == 1 and drew["street_lamp_level"] > 0.0
    assert at(lit, 30.0, 1.0) > 3.0 * at(lit, 90.0), (
        "the road is as bright far from the lamp as under it"
    )
    assert at(lit, 30.0, 1.0) < 250.0, "the lit road is a white sheet"
    head = tuple(round(float(value)) for value in lamps[0, [5, 4]])
    assert float(lit[head]) > 240.0, "the lamp itself is not burning"
    plain, drew = night(_provider(labels, None))
    assert drew["street_lamps"] == 0
    assert at(plain, 30.0, 1.0) < 1.5 * at(plain, 90.0) + 5.0
    off, _ = night(_provider(labels, lamps), street_lamp_range=(0.0, 0.0))
    assert float(np.abs(off - plain).max()) == 0.0


def test_the_labels_lamps_light_a_road_where_there_is_no_reader() -> None:
    """A tall piece of the lamp label is a street lamp: the nearest of a
    road's are found that way with no reader at all."""
    image, labels, lamps = _night_road()
    left, top, right, bottom = (round(float(value)) for value in lamps[0, :4])
    labels = labels.copy()
    labels[top:bottom, left:right] = int(SceneClass.LAMP)
    provider = _provider(labels, None)
    assert provider.get_street_lamps(image) is None
    transform = NightIrMode(depth_provider=provider, p=1.0)
    transform.set_random_seed(2)
    with recording() as log:
        transform(image=image)
    assert log[0]["drew"]["street_lamps"] == 1
    assert _provider(labels, lamps).get_street_lamps(image).shape == (1, 7)  # type: ignore[union-attr]


def _stepped_depth() -> np.ndarray:
    """A wall three metres off on the left of the frame and open ground to 200 m on the right."""
    depth = np.tile(np.linspace(200.0, 12.0, HEIGHT, dtype=np.float32)[:, None], (1, WIDTH))
    depth[:, :100] = 3.0
    return depth


@pytest.mark.parametrize("fast", [True, False])
def test_the_far_veil_has_a_count_s_grain_and_the_scene_s_depth(fast: bool) -> None:
    """About 1 on average; nothing at all in front of a wall nearer than the
    last drop drawn; plainer in front of the middle distance than in front
    of the far end, where thousands of drops are an even grey; and none of
    it when it is turned off."""
    depth = _stepped_depth()
    density = rain_number_density(20.0, 0.0)
    was = mode.fast()
    try:
        mode.set_fast(fast)
        texture = far_field_texture(
            depth, np.random.default_rng(0), density, FOCAL, 6.0, FOCAL * 6.0 * 0.01, 5.0
        )
        smooth = far_field_texture(
            depth, np.random.default_rng(0), density, FOCAL, 6.0, FOCAL * 6.0 * 0.01, 5.0, 0.0
        )
    finally:
        mode.set_fast(was)
    assert texture.shape == depth.shape and texture.dtype == np.float32
    assert float(smooth.min()) == float(smooth.max()) == 1.0
    assert float(texture.min()) >= 0.0
    assert np.all(texture[:, :100] == 1.0), "rain in front of a wall nearer than the nearest of it"
    open_ground = texture[:, 110:]
    assert float(open_ground.mean()) == pytest.approx(1.0, abs=0.12)

    def grain(rows: slice) -> float:
        """How much the veil varies from one pixel to the next over some rows.

        From neighbour to neighbour, which leaves the sheets out: they
        are a third of the frame across.
        """
        return float(np.diff(open_ground[rows], axis=1).std())

    assert grain(slice(200, 240)) > 2.0 * grain(slice(0, 40)) > 0.0


def test_far_rain_is_streaked_the_way_it_falls_and_far_snow_is_dots() -> None:
    """The grain is as long as the particles fall during the exposure: rain
    at a hundredth of a second is streaks down the frame, and slow snow is
    speckle with no direction."""
    depth = np.full((HEIGHT, WIDTH), 14.0, dtype=np.float32)

    def stretch(texture: np.ndarray) -> float:
        """How much further the grain is alike down the frame than across it."""
        centred = texture - texture.mean()
        down = float((centred[3:] * centred[:-3]).mean())
        across = float((centred[:, 3:] * centred[:, :-3]).mean())
        return down / max(float(centred.var()), 1e-9) - across / max(float(centred.var()), 1e-9)

    rain = far_field_texture(
        depth,
        np.random.default_rng(1),
        rain_number_density(20.0, 0.0),
        FOCAL,
        6.0,
        FOCAL * 6.5 * 0.02,
    )
    snow = far_field_texture(
        depth,
        np.random.default_rng(1),
        snow_number_density(2.0, 0.0),
        FOCAL,
        6.0,
        FOCAL * 1.0 * 0.002,
    )
    assert stretch(rain) > 0.2
    assert abs(stretch(snow)) < 0.08


def test_lit_from_the_lens_the_grain_does_not_wait_on_the_scene() -> None:
    """At night what comes back to the lens comes from the air just beyond
    the last flake drawn, whatever stands behind it."""
    density = snow_number_density(2.0, 0.0)
    flat = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    texture = far_field_texture(
        flat, np.random.default_rng(0), density, FOCAL, 8.0, FOCAL * 0.03, lit_from_lens=True
    )
    assert float(texture.mean()) == pytest.approx(1.0, abs=0.1)
    assert float(texture.std()) > 0.05 and float(texture.min()) >= 0.0


class _Stepped(DepthEstimator):
    """Relative depth of a near wall beside open ground."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """The stepped scene's depth as a share of the farthest of it."""
        return (_stepped_depth() / 200.0).astype(np.float32)


@pytest.mark.parametrize(
    ("transform", "knob", "settings"),
    [
        (RandomRain, "distant_rain_range", {"max_streaks": 0, "wetness_probability": 0.0}),
        (RandomSnow, "distant_snow_range", {"max_flakes": 0, "accumulation_probability": 0.0}),
    ],
)
def test_rain_and_snow_keep_their_grain_in_the_distance(
    transform: type, knob: str, settings: dict
) -> None:
    """With nothing drawn near the lens, what is left is the veil: even
    when the grain is turned off, textured when it is on, and the same in
    front of a wall three metres off either way."""
    image = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    provider = SharedDepthProvider(_Stepped(), depth_range_m=(1.0, 200.0))
    assert float(provider.get_metric_depth(image)[:, :95].max()) < 4.5, "the wall is not near"
    outs = {}
    for amount in (0.0, 1.0):
        built = transform(
            depth_provider=provider,
            p=1.0,
            veil_beta_range=(0.02, 0.02),
            overcast_range=(0.0, 0.0),
            **{knob: (amount, amount)},
            **settings,
        )
        built.set_random_seed(4)
        with recording() as log:
            outs[amount] = built(image=image)["image"].astype(np.float32).mean(axis=2)
        assert log[0]["drew"][knob.removesuffix("_range")] == amount

    def grain(picture: np.ndarray) -> float:
        """How much the open ground varies from one pixel to the next along its rows."""
        return float(np.diff(picture[120:, 110:], axis=1).std())

    assert grain(outs[1.0]) > 3.0 * grain(outs[0.0]) + 0.5
    assert float(np.abs(outs[1.0][:, :95] - outs[0.0][:, :95]).max()) <= 1.0
    assert abs(float(outs[1.0][:, 110:].mean() - outs[0.0][:, 110:].mean())) < 4.0


@pytest.mark.parametrize("transform", [NightRain, NightSnow])
def test_the_night_glow_is_grained_and_can_be_made_even(transform: type) -> None:
    """The glow of what the beam lights and the lens cannot resolve is a
    count's, and has its grain; at 0 it is the even wall it was."""
    # Wide enough for a pixel to look through a few drops and not hundreds:
    # in a small frame far rain is as even as it looks.
    image = np.zeros((480, 960, 3), dtype=np.uint8)
    limit = "max_streaks" if transform is NightRain else "max_flakes"
    outs = {}
    for amount in (0.0, 1.0):
        built = transform(
            p=1.0,
            backscatter_range=(0.2, 0.2),
            backscatter_grain_range=(amount, amount),
            beam_falloff_range=(0.0, 0.0),
            **{limit: 1},
        )
        built.set_random_seed(3)
        outs[amount] = built(image=image)["image"].astype(np.float32).mean(axis=2)
    even, grained = outs[0.0][80:400, 80:880], outs[1.0][80:400, 80:880]
    assert float(np.percentile(even, 95) - np.percentile(even, 5)) <= 2.0
    assert float(grained.std()) > 2.0
    assert float(grained.mean()) == pytest.approx(float(even.mean()), rel=0.2)


def test_the_light_on_the_air_is_a_cone_under_each_lamp() -> None:
    """Along a line of sight, a lamp's light summed over the air it passes
    through: most where the line passes just under the lamp, little far to
    one side, next to nothing above it, since a luminaire throws downwards;
    and the lit air is as far off as the lamp is."""
    up = (-_ground_at(0.0) / CAMERA_M).astype(np.float32)
    lamp = np.array([[*_ground_at(30.0, 0.0, 10.0), 10.0]], np.float32)
    looked_at = np.stack(
        [
            _ground_at(60.0, 0.0, 8.0),  # through the air a little under the lamp
            _ground_at(60.0, 14.0, 8.0),  # the same height, well to one side
            _ground_at(60.0, 0.0, 26.0),  # passing well over the lamp
            _ground_at(12.0, 0.0, 8.0),  # stopping short of it
        ]
    ).astype(np.float32)[None]
    air, reach = street_air(looked_at, up, lamp)
    under, beside, over, short = air[0].tolist()
    assert under > 4.0 * beside > 0.0
    assert over < 0.3 * under, "a ball of lit air round the lamp, and not a cone under it"
    assert short < 0.25 * under
    assert 20.0 < float(reach[0, 0]) < 45.0
    none, _far = street_air(looked_at, up, np.zeros((0, 4), np.float32))
    assert float(none.max()) == 0.0


def test_a_lamp_s_head_is_in_space_where_it_is_seen() -> None:
    """A head at the end of an arm two metres to one side of its post: the
    lamp is put on the line of sight through the head, and not over the
    foot of the post, so its light falls from where it burns."""
    middle, up = _ground_at(20.0), -_ground_at(0.0) / CAMERA_M
    foot, head = _pixel_of(_ground_at(30.0, 4.0)), _pixel_of(_ground_at(30.0, 6.0, 7.0))
    lamps = np.array([[foot[0] - 3, head[1], foot[0] + 3, foot[1], head[0], head[1], 0.5]])
    placed = lamps_in_space(lamps.astype(np.float32), FOCAL, (HEIGHT, WIDTH), middle, up)[0]
    seen_at = _pixel_of(placed[:3])
    assert abs(seen_at[0] - head[0]) < 0.5 and abs(seen_at[1] - head[1]) < 0.5
    assert np.allclose(placed[:3], _ground_at(30.0, 6.0, 7.0), atol=0.7)


def test_a_headless_lamp_takes_its_head_from_the_end_of_its_labelled_arm() -> None:
    """A reader that boxed no head puts one at the middle of the top of the
    post's box, which for a lamp with an arm is empty air. The labels show
    the arm: the head goes to its end, one for each arm, and a head the
    reader did box stays where it is."""
    labels = np.full((HEIGHT, WIDTH), int(SceneClass.ROAD), dtype=np.uint8)
    labels[60:220, 98:102] = int(SceneClass.POLE)  # a post
    labels[60:64, 98:150] = int(SceneClass.POLE)  # its arm, to the right
    labels[58:66, 144:152] = int(SceneClass.LAMP)  # the luminaire at its end
    one_arm = np.array([[96.0, 58.0, 152.0, 220.0, 124.0, 58.0, 0.5]], np.float32)
    moved = heads_by_label(one_arm, labels)
    assert moved.shape == (1, 7)
    assert 143.0 <= float(moved[0, 4]) <= 152.0 and 57.0 <= float(moved[0, 5]) <= 66.0
    boxed = one_arm.copy()
    boxed[0, 4:6] = (110.0, 61.0)
    assert np.array_equal(heads_by_label(boxed, labels), boxed)
    labels[60:64, 50:98] = int(SceneClass.POLE)  # a second arm, to the left
    two_arms = np.array([[48.0, 58.0, 152.0, 220.0, 100.0, 58.0, 0.5]], np.float32)
    heads = sorted(float(row[4]) for row in heads_by_label(two_arms, labels))
    assert len(heads) == 2 and heads[0] < 60.0 and heads[1] > 140.0
    nothing = np.full_like(labels, int(SceneClass.ROAD))
    assert np.array_equal(heads_by_label(one_arm, nothing), one_arm)


def test_a_night_in_colour_is_lit_in_its_lamps_colour_and_dark_between() -> None:
    """Under sodium the lit road is orange, with next to no blue; under a
    cool LED it is not. Either way it is bright under the lamp and dark far
    from it, and the lamp burns. With no lamps known the frame is lit by
    the sky's glow alone: dim, even and noisy."""
    image, labels, lamps = _night_road()

    def night(provider: SharedDepthProvider, kelvin: float) -> tuple[np.ndarray, dict]:
        """The road as a colour camera sees it at night, and what was drawn."""
        transform = NightColorMode(
            depth_provider=provider, p=1.0, lamp_temperature_range_k=(kelvin, kelvin)
        )
        transform.set_random_seed(2)
        with recording() as log:
            out = transform(image=image)["image"].astype(np.float32)
        return out, log[0]["drew"]

    def at(picture: np.ndarray, ahead_m: float, right_m: float = 0.0) -> np.ndarray:
        """The picture's colour about a place on the ground."""
        column, row = (round(value) for value in _pixel_of(_ground_at(ahead_m, right_m)))
        return picture[row - 2 : row + 3, column - 2 : column + 3].reshape(-1, 3).mean(axis=0)

    sodium, drew = night(_provider(labels, lamps), 2100.0)
    assert drew["street_lamps"] == 1 and drew["lamp_temperature_k"] == 2100.0
    under, far = at(sodium, 30.0, 1.0), at(sodium, 90.0)
    assert float(under.mean()) > 3.0 * float(far.mean())
    assert under[0] > 1.25 * under[1] > 1.5 * under[2], "sodium light that is not orange"
    cool, _ = night(_provider(labels, lamps), 5600.0)
    under = at(cool, 30.0, 1.0)
    assert under[2] > 0.9 * under[0], "a cool lamp as orange as sodium"
    unlit, drew = night(_provider(labels, None), 2100.0)
    assert drew["street_lamps"] == 0
    assert float(at(unlit, 30.0, 1.0).mean()) < 1.6 * float(at(unlit, 90.0).mean()) + 6.0
    assert float(unlit[labels == int(SceneClass.ROAD)].mean()) < 110.0


def test_signs_and_plates_send_the_illuminator_s_light_back_and_paint_a_little() -> None:
    """Sheeting returns hundreds of times what a matt surface does to a
    lamp at the lens; road paint a few times. So a sign within reach of the
    illuminator is at full scale where the wall beside it is dark, and
    turning the return off leaves the frame as it was."""
    scene = np.full((HEIGHT, WIDTH, 3), 0.05, dtype=np.float32)
    labels = np.full((HEIGHT, WIDTH), int(SceneClass.ROAD), dtype=np.uint8)
    scene[:, 158:162] = 0.6  # a painted line down the road
    none = np.zeros((0, 5), dtype=np.float32)
    assert retroreflectors(np.full_like(scene, 0.05), labels, None, none) is None
    signs = np.array([[40.0, 60.0, 70.0, 80.0, 0.5]], np.float32)
    plate = VehicleView((200, 150, 60, 40), 0.0, 0.4, np.zeros((0, 5), np.float32))._replace(
        plate=(220.0, 170.0, 240.0, 176.0)
    )
    back = retroreflectors(scene, labels, [plate], signs)
    assert back is not None and back.shape == (HEIGHT, WIDTH)
    assert float(back[70, 55]) < -300.0 and float(back[173, 230]) < -100.0
    assert 1.0 < float(back[120, 160]) < 10.0
    assert float(back[30, 250]) == 0.0

    image, road_labels, _lamps = _night_road()

    class _Signs(_Lamps):
        """A reader that knows of one sign and nothing else."""

        def street_signs(self, image: np.ndarray) -> np.ndarray:
            """One sign, up the road on the left."""
            return np.array([[60.0, 90.0, 90.0, 110.0, 0.5]], np.float32)

    def infrared(**more: object) -> np.ndarray:
        """The road under the illuminator, with a reader that found a sign."""
        provider = SharedDepthProvider(
            _Ground(),
            depth_range_m=(float(_ground_depth().min()), 400.0),
            labels=_Labels(road_labels),
            vehicles=_Signs(np.zeros((0, 7), np.float32)),
        )
        transform = NightIrMode(
            depth_provider=provider, p=1.0, illuminator_range_m=(14.0, 14.0), **more
        )  # type: ignore[arg-type]
        transform.set_random_seed(4)
        return transform(image=image)["image"].astype(np.float32).mean(axis=2)

    with_return, without = infrared(), infrared(retroreflector_range=(0.0, 0.0))
    assert float(with_return[95:105, 65:85].mean()) > 240.0
    assert float(without[95:105, 65:85].mean()) < 0.6 * float(with_return[95:105, 65:85].mean())
    assert float(np.abs(with_return[150:, 200:] - without[150:, 200:]).max()) == 0.0


@pytest.mark.parametrize(
    ("weather", "settings"),
    [
        (NightRain, {"rain_rate_range_mm_h": (30.0, 30.0), "max_streaks": 1}),
        (NightSnow, {"snowfall_rate_range_mm_h": (2.0, 2.0), "max_flakes": 1}),
    ],
)
def test_what_falls_at_night_shows_in_the_lamps_light(weather: type, settings: dict) -> None:
    """Rain or snow drawn over a frame a provider lit from its lamps is
    brightest in the air under the lamp; turned off, or over a frame nobody
    lit, there is no such cone."""
    image, labels, lamps = _night_road()
    provider = _provider(labels, lamps)
    head = (round(float(lamps[0, 5])), round(float(lamps[0, 4])))
    under = (slice(head[0] + 8, head[0] + 30), slice(head[1] - 6, head[1] + 7))

    def falling(cone: float, lit: bool = True) -> np.ndarray:
        """The night road with something falling through it."""
        night = NightIrMode(
            depth_provider=provider if lit else _provider(labels, None),
            p=1.0,
            vehicle_lamp_range=(0.0, 0.0),
        )
        chain = A.Compose(
            [
                night,
                weather(
                    p=1.0,
                    backscatter_range=(0.0, 0.0),
                    lamp_cone_range=(cone, cone),
                    **settings,
                ),
            ],
            seed=3,
        )
        return chain(image=image)["image"].astype(np.float32).mean(axis=2)

    with_cone, without = falling(1.0), falling(0.0)
    lift = with_cone - without
    assert float(lift[under].mean()) > 4.0, "nothing lit in the air under the lamp"
    assert float(lift[under].mean()) > 5.0 * abs(float(lift[-40:, :60].mean())) + 1.0
    unlit = falling(1.0, lit=False) - falling(0.0, lit=False)
    assert float(np.abs(unlit).max()) == 0.0


def test_fog_at_night_glows_under_the_lamps_it_knows_of() -> None:
    """Fog over a lit night frame is lit by the lamps' own light, in a cone
    under each, and the note of that light follows the frame through a
    transform that comes between."""
    image, labels, lamps = _night_road()
    provider = _provider(labels, lamps)
    head = (round(float(lamps[0, 5])), round(float(lamps[0, 4])))
    under = (slice(head[0] + 8, head[0] + 30), slice(head[1] - 6, head[1] + 7))

    def foggy(cone: float) -> np.ndarray:
        """The night road in fog, with rain drawn first to come between."""
        chain = A.Compose(
            [
                NightColorMode(depth_provider=provider, p=1.0, vehicle_lamp_range=(0.0, 0.0)),
                NightRain(
                    p=1.0, max_streaks=1, backscatter_range=(0.0, 0.0), lamp_cone_range=(0, 0)
                ),
                NightFog(
                    depth_provider=provider,
                    p=1.0,
                    visibility_range_m=(300.0, 300.0),
                    lamp_cone_range=(cone, cone),
                ),
            ],
            seed=3,
        )
        return chain(image=image)["image"].astype(np.float32).mean(axis=2)

    lit, plain = foggy(1.0), foggy(0.0)
    assert float(lit[under].mean()) > float(plain[under].mean()) + 5.0
    assert float(lit[under].mean()) > float(lit[-40:, :60].mean()) + 5.0


def test_a_scene_is_read_from_the_frame_it_started_as() -> None:
    """A provider asked about a frame that transforms made from another reads
    the other: what stands in a scene is looked for in the photograph, and
    not in the photograph with something drawn over it."""

    class _Seen(DepthEstimator):
        """A depth estimator that remembers the frames it was shown."""

        def __init__(self) -> None:
            """Start with none."""
            self.frames: list[np.ndarray] = []

        def estimate(self, image: np.ndarray) -> np.ndarray:
            """Note the frame and call it level."""
            self.frames.append(image)
            return np.full(image.shape[:2], 0.5, dtype=np.float32)

    estimator = _Seen()
    provider = SharedDepthProvider(estimator)
    image = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    # A transform with no provider first, then one that asks.
    chain = A.Compose([NightRain(p=1.0), NightFog(depth_provider=provider, p=1.0)], seed=1)
    chain(image=image)
    assert len(estimator.frames) == 1
    assert estimator.frames[0] is image, "depth was read from the frame with rain over it"


def test_a_reader_looks_at_the_original_and_answers_in_the_frame_s_pixels() -> None:
    """Given the frame at the size it was taken at, the labels and the
    reader are run on that, and what they find comes back in the pixels of
    the frame the pipeline works on. An original of another shape is not
    the same picture."""

    class _Sized(_Labels):
        """Labels that note the size of each frame they were asked about."""

        def __init__(self) -> None:
            """Start with none seen."""
            self.sizes: list[tuple[int, int]] = []

        def estimate(self, image: np.ndarray) -> np.ndarray:
            """Road everywhere, with a vehicle in the lower right quarter."""
            self.sizes.append(image.shape[:2])
            labels = np.full(image.shape[:2], int(SceneClass.ROAD), dtype=np.uint8)
            labels[image.shape[0] // 2 :, image.shape[1] // 2 :] = int(SceneClass.VEHICLE)
            return labels

    class _Reader(VehicleViewEstimator):
        """A reader that notes sizes too, and finds a lamp post and a plate."""

        def __init__(self) -> None:
            """Start with none seen."""
            self.sizes: list[tuple[int, int]] = []

        def estimate(self, image: np.ndarray, vehicles: np.ndarray) -> list[VehicleView]:
            """One vehicle: the mask's own box, with a plate at its middle."""
            self.sizes.append(vehicles.shape)
            rows, columns = np.nonzero(vehicles)
            box = (
                int(columns.min()),
                int(rows.min()),
                int(np.ptp(columns)) + 1,
                int(np.ptp(rows)) + 1,
            )
            lamps = np.array([[box[0] + 40.0, box[1] + 60.0, 8.0, 6.0, 0.5]], np.float32)
            plate = (box[0] + 100.0, box[1] + 80.0, box[0] + 140.0, box[1] + 92.0)
            return [
                VehicleView(box, 0.0, 0.5, lamps, np.ones((box[3], box[2]), bool), "car", plate)
            ]

        def street_lamps(self, image: np.ndarray) -> np.ndarray:
            """One post, a tenth of the way in."""
            wide, tall = image.shape[1], image.shape[0]
            return np.array(
                [[0.1 * wide, 0.1 * tall, 0.12 * wide, 0.9 * tall, 0.11 * wide, 0.1 * tall, 0.5]]
            )

    labeller, reader = _Sized(), _Reader()
    provider = SharedDepthProvider(_Ground(), labels=labeller, vehicles=reader)
    frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    original = np.full((4 * HEIGHT, 4 * WIDTH, 3), 90, dtype=np.uint8)
    provider.remember(frame, original=original)
    assert labeller.sizes == [(4 * HEIGHT, 4 * WIDTH)] and reader.sizes == [(4 * HEIGHT, 4 * WIDTH)]
    assert provider.get_labels(frame).shape == (HEIGHT, WIDTH)  # type: ignore[union-attr]
    (view,) = provider.get_vehicles(frame) or []
    assert view.box == (WIDTH // 2, HEIGHT // 2, WIDTH // 2, HEIGHT // 2)
    assert view.mask is not None and view.mask.shape == (HEIGHT // 2, WIDTH // 2)
    assert view.lamps[0, :4].tolist() == [WIDTH // 2 + 10.0, HEIGHT // 2 + 15.0, 2.0, 1.5]
    assert view.plate == (
        WIDTH // 2 + 25.0,
        HEIGHT // 2 + 20.0,
        WIDTH // 2 + 35.0,
        HEIGHT // 2 + 23.0,
    )
    post = provider.get_street_lamps(frame)
    assert post is not None and np.allclose(post[0, :4], [32.0, 24.0, 38.4, 216.0])
    with pytest.raises(ValueError, match="not the same shape"):
        provider.remember(frame, original=np.zeros((HEIGHT, 3 * WIDTH, 3), np.uint8))
