"""The ground has its own ladder.

How wet a road is and how much snow lies on it used to climb with the
precipitation's severity: light rain meant a dry road, a blizzard meant deep
snow. That made the two most ordinary mismatches impossible to ask for — the
first minute of a storm, and the day after one.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest

from surmentations.atmosphere.rain import RandomRain
from surmentations.atmosphere.snow import RandomSnow
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import recording
from surmentations.presets import GROUND_SEVERITY, SEVERITY, surveillance, weather_state

GROUND_KEYS = {
    "RandomRain": {"wetness_probability", "wetness_darkening_range", "puddle_coverage_range"},
    "RandomSnow": {"accumulation_probability", "accumulation_strength_range"},
}
SIZE = 160


class _GroundPlane(DepthEstimator):
    """A level road under the lower half of the frame, a far wall above it."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Depth of a level road under the lower half of the frame, and a far wall above it."""
        height, width = image.shape[:2]
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        depth = np.clip(1.5 * 300.0 / np.maximum(rows - height / 2.0, 1e-3), 0.0, 50.0)
        return np.broadcast_to(depth / 50.0, (height, width)).astype(np.float32)


def _provider() -> SharedDepthProvider:
    """A depth provider over the ground plane."""
    return SharedDepthProvider(_GroundPlane())


def _scene() -> np.ndarray:
    """A noisy mid-grey frame."""
    rng = np.random.default_rng(0)
    return rng.integers(70, 130, size=(SIZE, SIZE, 3), dtype=np.uint8)


def _road(image: np.ndarray) -> float:
    """The mean level of the bottom quarter of the frame: the road nearest the camera."""
    return float(image[SIZE * 3 // 4 :].astype(np.float32).mean())


def test_the_precipitation_ladder_no_longer_sets_the_ground() -> None:
    """If a ground argument creeps back into `SEVERITY`, combining the two
    ladders raises on the duplicate keyword, and before that the coupling is
    back."""
    for level, presets in SEVERITY.items():
        for name, keys in GROUND_KEYS.items():
            assert not keys & set(presets[name]), f"{name} level {level} sets the ground"
    assert sorted(GROUND_SEVERITY) == [0, 1, 2, 3, 4, 5]
    for presets in GROUND_SEVERITY.values():
        assert set(presets) == set(GROUND_KEYS)
        for name, keys in GROUND_KEYS.items():
            assert set(presets[name]) <= keys


@pytest.mark.parametrize("weather", [1, 5])
@pytest.mark.parametrize("ground", [0, 5])
def test_any_weather_goes_with_any_ground(weather: int, ground: int) -> None:
    """A cloudburst on a dry road and drizzle on a soaked one; a blizzard on
    bare tarmac and a flurry over deep snow."""
    scene = _scene()
    rain = RandomRain(
        depth_provider=_provider(), p=1.0,
        **SEVERITY[weather]["RandomRain"], **GROUND_SEVERITY[ground]["RandomRain"],
    )
    with recording() as log:
        rain(image=scene)
    assert log[0]["drew"]["draw_wetness"] is (ground > 0)

    snow = RandomSnow(
        depth_provider=_provider(), p=1.0, max_flakes=0, veil_beta_range=(0.0, 0.0),
        overcast_range=(0.0, 0.0),
        **SEVERITY[weather]["RandomSnow"], **GROUND_SEVERITY[ground]["RandomSnow"],
    )
    snow.set_random_seed(0)
    whitened = _road(snow(image=scene)["image"]) - _road(scene)
    if ground == 0:
        assert abs(whitened) < 1.5, "bare ground was asked for"
    else:
        assert whitened > 40.0, "deep snow was asked for"


def test_the_ground_ladder_is_monotonic_under_fixed_weather() -> None:
    scene = _scene()
    snow_cover, road_level = [], []
    for level in sorted(GROUND_SEVERITY):
        snow = RandomSnow(
            depth_provider=_provider(), p=1.0, max_flakes=0, veil_beta_range=(0.0, 0.0),
            overcast_range=(0.0, 0.0), accumulation_patchiness_range=(0.0, 0.0),
            **GROUND_SEVERITY[level]["RandomSnow"],
        )
        rain = RandomRain(
            depth_provider=_provider(), p=1.0, max_streaks=0, veil_beta_range=(0.0, 0.0),
            overcast_range=(0.0, 0.0), **GROUND_SEVERITY[level]["RandomRain"],
        )
        covers, levels = [], []
        for seed in range(6):
            snow.set_random_seed(seed)
            rain.set_random_seed(seed)
            covers.append(_road(snow(image=scene)["image"]))
            levels.append(_road(rain(image=scene)["image"]))
        snow_cover.append(float(np.mean(covers)))
        road_level.append(float(np.mean(levels)))

    assert all(a < b for a, b in zip(snow_cover, snow_cover[1:], strict=False)), snow_cover
    # A wetter road is a darker one.
    assert all(a > b for a, b in zip(road_level, road_level[1:], strict=False)), road_level


def _wet_share(transform: A.BasicTransform, samples: int = 60) -> tuple[float, list[float]]:
    """How often a transform drew the wet road over ``samples`` calls, and how dark each was."""
    transform.set_random_seed(3)
    image = np.full((96, 96, 3), 120, dtype=np.uint8)
    wet, darkening = [], []
    for _ in range(samples):
        with recording() as log:
            transform(image=image)
        for entry in log:
            if entry["t"] == "RandomRain":
                wet.append(bool(entry["drew"]["draw_wetness"]))
                darkening.append(float(entry["drew"]["wetness_darkening"]))
    assert len(wet) > samples // 2, "rain hardly fired; the test is not measuring anything"
    return float(np.mean(wet)), darkening


def test_pipelines_take_the_ground_separately_from_the_weather() -> None:
    def state(**kwargs: object) -> A.BasicTransform:
        return A.Compose(
            [weather_state(_provider(), rain=0.97, snow=0.01, fog=0.01, **kwargs)]  # type: ignore[arg-type]
        )

    assert _wet_share(state(severity=5, ground_severity=0))[0] == 0.0
    share, darkening = _wet_share(state(severity=1, ground_severity=5))
    assert share == 1.0 and max(darkening) <= 0.55

    # Left unset, every state of the ground turns up under one weather level,
    # and the road is wet in about half the rain frames: it was 0.85 until
    # a wet road turned out to be two to three times the cost of the frame.
    share, darkening = _wet_share(state(severity=3), samples=150)
    assert 0.35 < share < 0.65
    assert min(darkening) < 0.5 and max(darkening) > 0.8

    with pytest.raises(ValueError, match="ground_severity"):
        state(ground_severity=6)
    surveillance(_provider(), codec=0.0, ground_severity=4)


# ------------------------------------------------- what a wet road looks like


def test_wetting_darkens_dark_surfaces_spares_white_and_deepens_colour() -> None:
    """Asphalt goes nearly black, the paint on it stays white, brick turns a
    deeper red. A flat multiplier does the first and gets the other two
    wrong: grey markings, and colour no stronger than before."""
    from surmentations.atmosphere.rain import wet_albedo

    assert float(wet_albedo(np.float32(0.1), 0.5)) == pytest.approx(0.05, rel=0.02)
    assert float(wet_albedo(np.float32(0.97), 0.5)) > 0.85
    # A sunlit road is recorded bright, and still has to darken.
    assert float(wet_albedo(np.float32(0.47), 0.45)) < 0.5 * 0.47

    brick = np.array([0.5, 0.25, 0.2], dtype=np.float32)
    wet = wet_albedo(brick, 0.5)
    assert wet[0] / wet[2] > brick[0] / brick[2]


def test_a_soaked_road_is_darker_in_the_finished_frame() -> None:
    """Two things used to hand the darkening back. Level ground carried
    40-70% of the effect, because estimated normals wander; and the
    exposure correction, run last, brightened the frame again whenever the
    road was most of it."""
    scene = np.full((SIZE, SIZE, 3), 150, dtype=np.uint8)
    rain = RandomRain(
        depth_provider=_provider(), p=1.0, max_streaks=0, veil_beta_range=(0.0, 0.0),
        overcast_range=(0.0, 0.0), wetness_probability=1.0,
        wetness_darkening_range=(0.45, 0.45), puddle_coverage_range=(0.0, 0.0),
    )
    rain.set_random_seed(0)
    out = rain(image=scene)["image"].astype(np.float32)
    # Nearest the camera, where the road is seen steeply and reflects least.
    assert out[-20:].mean() < 0.8 * 150.0


def test_a_car_s_reflection_lies_under_the_car() -> None:
    """A mirror shows a thing upside down below where it stands. Mirrored
    about the horizon instead, a car 80 rows below it is reflected 160 rows
    below it, under nothing."""
    from surmentations.atmosphere.rain import _apply_ground_wetness

    height, width, horizon = 240, 80, 40.0
    image = np.full((height, width, 3), 0.3, dtype=np.float32)
    image[100:120, 30:50] = 0.0  # something dark standing on the road
    wetness = np.ones((height, width), dtype=np.float32)
    wetness[:40] = 0.0
    wetness[100:120, 30:50] = 0.0

    out = _apply_ground_wetness(image, wetness, 0.5, horizon_px=horizon, focal_px=120.0)
    road = out[:, 5:20].mean(axis=(1, 2))
    under = out[:, 34:46].mean(axis=(1, 2))
    darker = road - under
    assert darker[122:136].mean() > 0.004, "no reflection at the object's foot"
    assert darker[122:136].mean() > 4.0 * abs(float(darker[185:215].mean()))


def test_wet_ground_reaches_the_foot_of_what_stands_on_it() -> None:
    """Depth is blurred at every object's edge, and the band of road in that
    blur was left dry: a pale outline round each car."""
    from surmentations.utils.ground import fill_to_edges

    road = np.full((240, 240, 3), 0.2, dtype=np.float32)
    road[100:140, 100:140] = (0.6, 0.1, 0.1)  # a red car
    wetness = np.ones((240, 240), dtype=np.float32)
    wetness[98:142, 98:142] = 0.0  # the car and two pixels of road round it

    filled = fill_to_edges(wetness, road)
    ring = np.ones((240, 240), dtype=bool)
    ring[100:140, 100:140] = False
    ring &= wetness == 0.0
    assert filled[ring].mean() > 0.6, "the road beside the car is still dry"
    assert filled[102:138, 102:138].max() < 0.05, "the car got wet"


def test_puddles_need_standing_water_and_cover_what_was_asked() -> None:
    from surmentations.atmosphere.rain import _apply_ground_wetness, _puddle_map

    depth = _provider().to_metric(_provider().get_depth(_scene()))
    rng = np.random.default_rng(0)
    for coverage in (0.2, 0.5):
        puddles = _puddle_map(depth, 150.0, 10.0, coverage, rng)
        assert puddles[SIZE // 2 + 10 :].mean() == pytest.approx(coverage, abs=0.15)

    # Sky above, road below: a puddle mirrors the sky and is brighter than
    # the film beside it. On a road that is only damp there is no film to
    # stand in, and the same map changes nothing.
    image = np.full((SIZE, SIZE, 3), 0.2, dtype=np.float32)
    image[: SIZE // 2] = 0.9
    wetness = np.zeros((SIZE, SIZE), dtype=np.float32)
    wetness[SIZE // 2 :] = 1.0
    everywhere = np.ones((SIZE, SIZE), dtype=np.float32)

    def road(darkening: float, puddles: np.ndarray | None) -> float:
        out = _apply_ground_wetness(
            image, wetness, darkening, horizon_px=SIZE / 2.0, focal_px=150.0, puddles=puddles
        )
        return float(out[SIZE // 2 + 5 : SIZE // 2 + 25].mean())

    assert road(0.5, everywhere) > 1.2 * road(0.5, None)
    assert road(0.9, everywhere) == pytest.approx(road(0.9, None), rel=1e-4)


def test_reflections_do_not_rule_the_road_with_stripes() -> None:
    """Every column used to keep the mirror line of the nearest thing above
    it down to the bottom of the frame, so each car and each lamp post drew
    a stripe the height of the road. A reflection is as long as the thing
    is tall, and a post across the foreground has none."""
    from surmentations.atmosphere.rain import _apply_ground_wetness

    height, width = 240, 200
    image = np.full((height, width, 3), 0.3, dtype=np.float32)
    image[:60] = 0.8  # sky
    image[:, 150:154] = 0.02  # a lamp post from top to bottom of the frame
    image[100:120, 40:70] = 0.02  # a car
    wetness = np.ones((height, width), dtype=np.float32)
    wetness[:60] = 0.0
    wetness[:, 150:154] = 0.0
    wetness[100:120, 40:70] = 0.0

    out = _apply_ground_wetness(image, wetness, 0.5, horizon_px=60.0, focal_px=150.0)
    road = out.mean(axis=2)
    open_road = road[:, 100:120].mean(axis=1)
    below_car = np.abs(road[:, 45:65].mean(axis=1) - open_road)
    assert below_car[122:135].mean() > 5.0 * below_car[160:230].mean(), (
        "the car's reflection runs on past the car's own height"
    )
    beside_post = np.abs(road[70:230, 158:166].mean(axis=1) - road[70:230, 170:178].mean(axis=1))
    assert beside_post.max() < 0.01, "the post drew a stripe down the road"


# ------------------------------------------------ what settled snow looks like


def _settled(scene: np.ndarray, **kwargs: object) -> np.ndarray:
    """``scene`` with only settled snow on it: no flakes, no veil, no overcast, full cover."""
    snow = RandomSnow(
        depth_provider=_provider(), p=1.0, max_flakes=0, veil_beta_range=(0.0, 0.0),
        overcast_range=(0.0, 0.0), accumulation_probability=1.0,
        accumulation_strength_range=(1.0, 1.0), accumulation_patchiness_range=(0.0, 0.0),
        **kwargs,  # type: ignore[arg-type]
    )
    snow.set_random_seed(0)
    return np.asarray(snow(image=scene)["image"]).astype(np.float32)


def test_deep_snow_covers_level_ground_and_is_white_on_black_asphalt() -> None:
    """Two ways a buried road came out grey. Estimated normals left level
    ground at half cover; and snow was shaded by the light the surface under
    it seemed to be in, which for black asphalt is almost none."""
    scene = np.full((SIZE, SIZE, 3), 190, dtype=np.uint8)  # an overcast wall of sky
    scene[SIZE // 2 :] = 25  # new asphalt
    out = _settled(scene)
    assert out[SIZE * 3 // 4 :].mean() > 160.0


def test_snow_under_sodium_lamps_is_orange_and_dark_where_no_lamp_reaches() -> None:
    scene = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    scene[:] = (10, 6, 2)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    pool = np.exp(-(((xx - 40) / 25.0) ** 2 + ((yy - 130) / 12.0) ** 2))[..., None]
    scene = np.clip(scene + pool * np.array([150.0, 90.0, 25.0]), 0, 255).astype(np.uint8)

    out = _settled(scene)
    lit = out[122:138, 25:55].mean(axis=(0, 1))
    assert lit[0] > 1.5 * lit[2], "snow under a sodium lamp came out white"
    assert lit.mean() > scene[122:138, 25:55].mean(), "snow is brighter than the road it covers"
    assert out[100:, -40:].mean() < 60.0, "snow glows where no lamp reaches"


def test_snow_buries_a_lawn_and_only_dusts_a_canopy() -> None:
    """Seen from above, a tree reads as level ground. Covered like ground it
    was a grey smear with the tree gone; snow lies on foliage in clumps, on
    the parts that catch the light, and the tree is still there under it."""
    rng = np.random.default_rng(0)
    scene = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    scene[: SIZE // 2] = 190
    scene[SIZE // 2 :] = (70, 130, 50)  # a lawn
    clumps = rng.uniform(0.25, 1.75, size=(SIZE // 8, SIZE // 8))
    clumps = np.kron(clumps, np.ones((4, 4)))[: SIZE // 2, : SIZE // 2, None]
    scene[SIZE // 2 :, SIZE // 2 :] = np.clip(clumps * np.array([50.0, 100.0, 40.0]), 0, 255)

    out = _settled(scene)
    lawn = out[SIZE * 3 // 4 :, 10 : SIZE // 2 - 10]
    canopy = out[SIZE * 3 // 4 :, SIZE // 2 + 10 : -10]
    assert lawn.mean() > 180.0 and lawn.std() < 15.0
    assert canopy.std() > 2.0 * lawn.std(), "the canopy was smoothed over"
    green = canopy[..., 1] - canopy[..., [0, 2]].mean(axis=2)
    assert (green > 15.0).mean() > 0.15, "no foliage left showing"
    assert (canopy.min(axis=2) > 170.0).mean() > 0.1, "no snow on the foliage"


def test_a_car_s_roof_is_told_from_the_road_by_being_an_island() -> None:
    """Level and a car's width across is a car. What an opening removes is
    not: that also takes the road between two cars, and left the open road
    as a set of discs."""
    from surmentations.utils.ground import small_level_areas

    level = np.ones((200, 200), dtype=np.float32)
    level[:40] = 0.0
    for left in (40, 80):  # two cars, a narrow strip of road between them
        level[98:132, left - 2 : left + 32] = 0.0  # sides
        level[104:126, left + 4 : left + 26] = 1.0  # roof
    vehicles = small_level_areas(level, 0.12)

    assert vehicles[108:122, 48:62].min() > 0.9 and vehicles[108:122, 88:102].min() > 0.9
    assert vehicles[100:130, 73:77].max() < 0.05, "the road between the cars is not a car"
    assert vehicles[150:, :].max() == 0.0


def test_how_much_snow_vehicles_carry_is_drawn_per_frame() -> None:
    snow = RandomSnow(
        depth_provider=_provider(), p=1.0, max_flakes=0, vehicle_cover_range=(0.25, 0.25)
    )
    with recording() as log:
        snow(image=_scene())
    assert log[0]["drew"]["vehicle_cover"] == pytest.approx(0.25)


def test_the_sky_edge_is_not_drawn_round_a_distant_tower() -> None:
    """A tower a mile off is at the sky's depth as far as the network can
    tell. Redrawing the sky's edge by depth alone took the tower's edge in,
    and it came out with a pale outline."""
    from surmentations.utils.sky import sky_to_depth_edge

    image = np.full((120, 120, 3), 0.6, dtype=np.float32)  # sky
    image[40:, 50:70] = 0.15  # a dark tower, at the same depth
    depth = np.full((120, 120), 0.5, dtype=np.float32)
    sky = np.ones((120, 120), dtype=np.float32)
    sky[39:, 49:71] = 0.0  # the mask stops a pixel short of the tower

    by_depth = sky_to_depth_edge(sky, depth)
    by_colour_too = sky_to_depth_edge(sky, depth, image)
    assert by_depth[60:, 50:52].mean() > 0.5, "the fixture should show the failure"
    assert by_colour_too[60:, 50:70].max() < 0.05
    assert by_colour_too[60:, 49].mean() > 0.8, "the sky beside the tower is sky"


# ------------------------------------------------- where the road ends, and what stands on it


def _plane(height: int, width: int, horizon: float, far_m: float = 400.0) -> np.ndarray:
    """Depth in metres of level ground under a camera whose horizon is at row ``horizon``."""
    rows = np.arange(height, dtype=np.float32).reshape(height, 1)
    depth = np.clip(4.0 * 300.0 / np.maximum(rows - horizon, 1e-3), 0.0, far_m)
    return np.ascontiguousarray(np.broadcast_to(depth, (height, width))).astype(np.float32)


def test_the_road_is_mirrored_about_its_far_end_and_seen_from_the_horizon() -> None:
    """Two rows, and they are not the same one. The reflectance of a wet road
    goes by the angle it is seen at, which is measured from the horizon; what
    it reflects stands at the road's far end and is mirrored about that."""
    from surmentations.atmosphere.rain import ground_horizon

    height, width, true_horizon = 240, 320, 60.0
    depth = _plane(height, width, true_horizon)
    ground = np.zeros((height, width), dtype=np.float32)
    # The road runs from the bottom of the frame up to a wall at row 110.
    ground[110:] = 1.0
    horizon, far_end = ground_horizon(depth, ground)
    assert far_end == pytest.approx(110.0, abs=1.0)
    assert horizon == pytest.approx(true_horizon, abs=4.0), "exact depth gives the true horizon"

    # A monocular depth map puts far ground too near. The horizon moves with
    # it; the row the road is mirrored about does not.
    compressed = 4.0 + 20.0 * (depth - 4.0) / (depth + 20.0)
    squeezed, same_end = ground_horizon(compressed.astype(np.float32), ground)
    assert same_end == pytest.approx(far_end, abs=1.0)
    assert squeezed <= same_end


def test_ground_that_fills_the_frame_has_its_horizon_above_the_picture() -> None:
    from surmentations.atmosphere.rain import ground_horizon

    depth = _plane(240, 320, -150.0)
    horizon, far_end = ground_horizon(depth, np.ones((240, 320), dtype=np.float32))
    assert horizon == pytest.approx(-150.0, abs=2.0)
    assert far_end == horizon, "nothing stands at the end of a road that leaves the frame"


def test_a_hedge_s_top_is_not_the_far_end_of_the_road() -> None:
    """Whatever faces upwards reads as level ground: a hedge's top, a roof.
    They stand on the ground, nearer than the ground behind them, so following
    the road up the frame the distance stops growing where they begin. Taking
    them for road put a night street's horizon in the crown of a tree."""
    from surmentations.atmosphere.rain import ground_horizon

    height, width = 240, 320
    depth = _plane(height, width, 60.0, far_m=150.0)
    ground = np.zeros((height, width), dtype=np.float32)
    ground[120:] = 1.0
    # A band of upward-facing surface above the road's end, twelve metres off.
    depth[40:120] = 12.0
    ground[40:120, :200] = 1.0
    _horizon, far_end = ground_horizon(depth, ground)
    assert far_end == pytest.approx(120.0, abs=2.0)


def _road_with_a_car(height: int = 240, width: int = 320) -> tuple[np.ndarray, ...]:
    """A grey road seen from above, a darker car on it that the depth map
    shows only as a bump, and a pool of lamp light that is in no depth map."""
    rng = np.random.default_rng(2)
    depth = _plane(height, width, -40.0)
    scene = np.full((height, width, 3), 0.2, dtype=np.float32)
    scene += rng.normal(0.0, 0.005, scene.shape).astype(np.float32)
    car = (slice(100, 130), slice(140, 190))
    depth[car] *= 0.85
    scene[car] = (0.02, 0.02, 0.03)
    pool = (slice(170, 220), slice(40, 110))
    scene[pool] *= 3.0
    ground = np.ones((height, width), dtype=np.float32)
    return depth, np.clip(scene, 0.0, 1.0), ground, car, pool


def test_what_stands_on_the_road_is_told_by_being_nearer_and_looking_different() -> None:
    """A car far enough off is a smooth bump in the depth map, and a wet road
    darkened it and laid reflections over it. It is nearer than the road
    round it *and* does not look like it. A pool of lamp light looks nothing
    like the road beside it either, and is not nearer: it stays road."""
    from surmentations.utils.ground import standing_objects

    depth, scene, ground, car, pool = _road_with_a_car()
    objects = standing_objects(depth, scene, ground)
    assert objects.shape == ground.shape and objects.dtype == np.float32
    assert float(objects[car].mean()) > 0.9, "the car is still part of the road"
    assert float(objects[pool].max()) < 0.05, "a pool of light was taken for an object"
    open_road = objects[200:, 200:]
    assert float(open_road.max()) < 0.05
    # The road at the car's foot is road: no dry rim round it.
    rim = objects[136:140, 140:190]
    assert float(rim.mean()) < 0.2


def test_a_bump_the_colour_of_the_road_is_still_an_object_if_it_stands_high() -> None:
    """A grey car on a grey road. Nothing in its look says so, and a bump of
    a few per cent could be the depth map's own unevenness; one of fifteen
    could not."""
    from surmentations.utils.ground import standing_objects

    depth, scene, ground, _car, _pool = _road_with_a_car()
    # Far enough off to be a dozen pixels of the depth map.
    car = (slice(60, 70), slice(240, 256))
    scene[car] = 0.2
    slight, high = depth.copy(), depth.copy()
    slight[car] *= 0.97
    high[car] *= 0.85
    assert float(standing_objects(slight, scene, ground)[car].mean()) < 0.1
    assert float(standing_objects(high, scene, ground)[car].mean()) > 0.5


def test_a_wet_road_leaves_the_vehicles_on_it_as_they_were() -> None:
    from surmentations.atmosphere.rain import _read_wet_ground

    depth, scene, _ground, car, _pool = _road_with_a_car()
    wetness, _horizon, _far_end = _read_wet_ground(depth, scene, 60.0, 30.0, 6.0)
    assert float(wetness[car].mean()) < 0.1
    assert float(wetness[200:, 200:].mean()) > 0.9


def test_small_far_things_on_the_road_are_not_painted_over() -> None:
    """A car four hundred metres off is a dozen pixels and no bump in any
    depth map, and the far end of a wet road is where it is most nearly a
    mirror: the traffic there was being painted out with sky. A thing that
    small is still unlike the road immediately round it. A broad change, the
    dark half of a road lit from one side, is not a thing."""
    from surmentations.utils.ground import small_things

    rng = np.random.default_rng(3)
    scene = np.full((240, 320, 3), 0.2, dtype=np.float32)
    scene[:, 160:] = 0.07
    scene += rng.normal(0.0, 0.004, scene.shape).astype(np.float32)
    scene[60:66, 40:52] = 0.03  # a far car on the lit half
    scene[150:156, 230:242] = 0.4  # a pale one on the dark half
    scene[100:240, 100:103] = 0.8  # a lane marking
    ground = np.ones((240, 320), dtype=np.float32)
    things = small_things(np.clip(scene, 0.0, 1.0), ground)
    assert things.shape == ground.shape and things.dtype == np.float32
    assert float(things[61:65, 42:50].mean()) > 0.8
    assert float(things[151:155, 232:240].mean()) > 0.8
    assert float(things[120:220, 101].mean()) > 0.8, "a marking is kept as it is"
    # The open road on either side of the change of light is road.
    assert float(things[:, 10:30].max()) < 0.05 and float(things[:, 290:310].max()) < 0.05
    assert float(things[20:40, 200:280].max()) < 0.05


def test_a_soaked_road_is_plainly_darker_and_its_far_end_is_not() -> None:
    """What a wet road looks like from above: dark underfoot, where water
    reflects a few per cent of the sky, and pale towards the horizon, where
    it reflects most of it."""
    from surmentations.atmosphere.rain import _apply_ground_wetness

    height, width = 240, 320
    image = np.full((height, width, 3), 0.3, dtype=np.float32)
    image[:60] = 0.8  # sky
    wetness = np.zeros((height, width), dtype=np.float32)
    wetness[64:] = 1.0
    out = _apply_ground_wetness(
        image, wetness, 0.37, horizon_px=60.0, focal_px=280.0,
        sky_radiance=np.full(3, 0.8, dtype=np.float32),
    )  # fmt: skip
    near, far = float(out[225:238].mean()), float(out[68:74].mean())
    assert near < 0.6 * 0.3, "underfoot it is asphalt at well under its dry brightness"
    assert far > 0.3, "and at its far end it is paler than it was dry"
    assert far > 2.0 * near
