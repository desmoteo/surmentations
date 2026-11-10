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

from weathermentations.atmosphere.rain import RandomRain
from weathermentations.atmosphere.snow import RandomSnow
from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.diagnostics import recording
from weathermentations.presets import GROUND_SEVERITY, SEVERITY, surveillance, weather_state

GROUND_KEYS = {
    "RandomRain": {"wetness_probability", "wetness_darkening_range", "puddle_coverage_range"},
    "RandomSnow": {"accumulation_probability", "accumulation_strength_range"},
}
SIZE = 160


class _GroundPlane(DepthEstimator):
    """A level road under the lower half of the frame, a far wall above it."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        depth = np.clip(1.5 * 300.0 / np.maximum(rows - height / 2.0, 1e-3), 0.0, 50.0)
        return np.broadcast_to(depth / 50.0, (height, width)).astype(np.float32)


def _provider() -> SharedDepthProvider:
    return SharedDepthProvider(_GroundPlane())


def _scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(70, 130, size=(SIZE, SIZE, 3), dtype=np.uint8)


def _road(image: np.ndarray) -> float:
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

    # Left unset, every state of the ground turns up under one weather level.
    share, darkening = _wet_share(state(severity=3), samples=150)
    assert 0.7 < share < 0.97
    assert min(darkening) < 0.5 and max(darkening) > 0.8

    with pytest.raises(ValueError, match="ground_severity"):
        state(ground_severity=6)
    surveillance(_provider(), codec=0.0, ground_severity=4)


# ------------------------------------------------- what a wet road looks like


def test_wetting_darkens_dark_surfaces_spares_white_and_deepens_colour() -> None:
    """Asphalt goes nearly black, the paint on it stays white, brick turns a
    deeper red. A flat multiplier does the first and gets the other two
    wrong: grey markings, and colour no stronger than before."""
    from weathermentations.atmosphere.rain import wet_albedo

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
    from weathermentations.atmosphere.rain import _apply_ground_wetness

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
    from weathermentations.atmosphere.rain import _fill_to_edges

    road = np.full((240, 240, 3), 0.2, dtype=np.float32)
    road[100:140, 100:140] = (0.6, 0.1, 0.1)  # a red car
    wetness = np.ones((240, 240), dtype=np.float32)
    wetness[98:142, 98:142] = 0.0  # the car and two pixels of road round it

    filled = _fill_to_edges(wetness, road)
    ring = np.ones((240, 240), dtype=bool)
    ring[100:140, 100:140] = False
    ring &= wetness == 0.0
    assert filled[ring].mean() > 0.6, "the road beside the car is still dry"
    assert filled[102:138, 102:138].max() < 0.05, "the car got wet"


def test_puddles_need_standing_water_and_cover_what_was_asked() -> None:
    from weathermentations.atmosphere.rain import _apply_ground_wetness, _puddle_map

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
    from weathermentations.atmosphere.rain import _apply_ground_wetness

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
