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

import numpy as np
import pytest

from surmentations import mode
from surmentations.atmosphere import RandomRain, RandomSnow
from surmentations.atmosphere.night_rain import NightRain
from surmentations.atmosphere.night_snow import NightSnow
from surmentations.camera.night_ir import NightIrMode
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
    even_road,
    fit_ground,
    headlamp_light,
    lamp_heads,
    lamps_in_space,
    scene_points,
    street_light,
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
    # Driving away from the camera, up the picture, seen from behind.
    beams = headlamp_light(points, middle, up, FOCAL, (HEIGHT, WIDTH), [(foot, (0.0, -1.0), True)])

    def lit(ahead_m: float, right_m: float = 0.0) -> float:
        """The beam at a place on the ground."""
        column, row = _pixel_of(_ground_at(ahead_m, right_m))
        scale = HEIGHT / beams.shape[0]
        return float(beams[int(row / scale), int(column / scale)])

    assert lit(36.0) > 0.4 and lit(36.0) > lit(50.0) > lit(70.0) > 0.0
    assert lit(28.0) < 0.05, "a beam from the middle of a vehicle seen from behind"
    assert lit(20.0) == 0.0 and lit(40.0, 12.0) < 0.02
    # The same vehicle coming on: its beam is towards the camera.
    oncoming = headlamp_light(
        points, middle, up, FOCAL, (HEIGHT, WIDTH), [(foot, (0.0, 1.0), False)]
    )
    scale = HEIGHT / oncoming.shape[0]
    near = _pixel_of(_ground_at(18.0))
    assert float(oncoming[int(near[1] / scale), int(near[0] / scale)]) > 0.3
    # Which way it points is which end is seen, with nothing else to go
    # by: away from the camera seen from behind, at it seen from the
    # front. An outline that is long across that line says nothing.
    for long_way in (None, (1.0, 0.0)):
        frame = (HEIGHT, WIDTH)
        rear = headlamp_light(points, middle, up, FOCAL, frame, [(foot, long_way, True)])
        front = headlamp_light(points, middle, up, FOCAL, frame, [(foot, long_way, False)])
        beyond, nearer = _pixel_of(_ground_at(40.0)), _pixel_of(_ground_at(15.0))
        assert float(rear[int(beyond[1] / scale), int(beyond[0] / scale)]) > 0.3
        assert float(rear[int(nearer[1] / scale), int(nearer[0] / scale)]) == 0.0
        assert float(front[int(nearer[1] / scale), int(nearer[0] / scale)]) > 0.3
        assert float(front[int(beyond[1] / scale), int(beyond[0] / scale)]) == 0.0
    # A wall across the road at 40 m: lit at its foot, dark at 3 m up.
    wall = points.copy()
    for above in (0.3, 3.0):
        wall[0, 0] = _ground_at(40.0, 0.0, above)
        got = headlamp_light(wall, middle, up, FOCAL, (HEIGHT, WIDTH), [(foot, (0.0, -1.0), True)])
        assert (float(got[0, 0]) > 0.2) is (above < 1.0)


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
