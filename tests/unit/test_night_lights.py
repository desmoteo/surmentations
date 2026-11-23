"""Glare and fog at night are things the lights in the frame do.

Both transforms here used to be missing, and the daylight models that stood
in for them were wrong in the same way: nothing was tied to where the lamps
are. Fog was an even veil, and flare needed a position chosen by the caller.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from surmentations.atmosphere.night_fog import NightFog
from surmentations.atmosphere.rain import RandomRain, _apply_ground_wetness, _lamp_light
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import recording
from surmentations.lens.source_glare import SourceGlare
from surmentations.presets.severity import SEVERITY
from surmentations.utils.photometry import srgb_to_linear
from surmentations.utils.sources import light_sources

SIZE = 240
LAMP = (60, 80)


def _night(
    lamps: tuple[tuple[int, int], ...] = (LAMP,), radius: int = 4, peak: int = 255
) -> np.ndarray:
    """A dark frame with saturated lamps drawn as discs at ``lamps``."""
    image = np.full((SIZE, SIZE, 3), 12, dtype=np.uint8)
    for x, y in lamps:
        cv2.circle(image, (x, y), radius, (peak, peak - 20, peak - 65), -1)
    return image


class _Wall(DepthEstimator):
    """Everything at the far plane, so fog acts evenly on the whole frame."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """A wall at the far plane, with one near value so the provider's range does not collapse.
        """
        depth = np.ones(image.shape[:2], dtype=np.float32)
        depth[0, 0] = 0.0  # keeps the provider's range from collapsing
        return depth


def _provider() -> SharedDepthProvider:
    """A depth provider over the wall, 3 to 50 m."""
    return SharedDepthProvider(_Wall(), depth_range_m=(3.0, 50.0))


def _ring(image: np.ndarray, centre: tuple[int, int], inner: float, outer: float) -> np.ndarray:
    """The pixels of ``image`` between ``inner`` and ``outer`` pixels from ``centre``."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    distance = np.hypot(xx - centre[0], yy - centre[1])
    return image[(distance >= inner) & (distance < outer)].astype(np.float32)


# ------------------------------------------------------------------ sources


def test_lamps_are_found_and_overexposed_surfaces_are_not() -> None:
    image = _night(lamps=((60, 80), (180, 150)), radius=5)
    image[200:, :] = 255  # a blown-out strip: saturated, but not a lamp
    found = light_sources(srgb_to_linear(image))

    assert len(found) == 2
    positions = sorted((round(s.x * SIZE), round(s.y * SIZE)) for s in found)
    for (x, y), (lamp_x, lamp_y) in zip(positions, [(60, 80), (180, 150)], strict=True):
        assert abs(x - lamp_x) <= 2 and abs(y - lamp_y) <= 2
    assert all(s.colour[0] >= s.colour[2] for s in found), "the lamps are warm"


def test_a_frame_without_lights_has_no_sources() -> None:
    rng = np.random.default_rng(0)
    daylight = rng.integers(40, 200, size=(SIZE, SIZE, 3), dtype=np.uint8)
    assert light_sources(srgb_to_linear(daylight)) == []


# -------------------------------------------------------------------- glare


def test_glare_comes_from_the_lamp_and_only_from_a_lamp() -> None:
    transform = SourceGlare(p=1.0, intensity_range=(0.4, 0.4))
    transform.set_random_seed(0)
    image = _night()
    out = transform(image=image)["image"]

    near = _ring(out, LAMP, 8, 30).mean() - _ring(image, LAMP, 8, 30).mean()
    far = _ring(out, LAMP, 120, 160).mean() - _ring(image, LAMP, 120, 160).mean()
    assert near > 5.0, f"no glare round the lamp: +{near:.2f} levels"
    assert near > 4.0 * max(far, 0.01)

    dark = np.full((SIZE, SIZE, 3), 12, dtype=np.uint8)
    np.testing.assert_array_equal(transform(image=dark)["image"], dark)


def test_each_lamp_flares_up_to_the_limit_asked_for() -> None:
    image = _night(lamps=((50, 60), (190, 60), (120, 190)))

    def flares(limit: int) -> int:
        transform = SourceGlare(p=1.0, max_sources_range=(limit, limit))
        transform.set_random_seed(1)
        with recording() as log:
            transform(image=image)
        return int(log[0]["drew"]["sources"])

    assert flares(1) == 1
    assert flares(3) == 3
    assert flares(6) == 3, "there are only three lamps"


def test_glare_is_not_logged_as_sunlight() -> None:
    """The optics are `SunFlare`'s, the cause is not: a pipeline that rules
    out sun at night must not find one in its log."""
    transform = SourceGlare(p=1.0)
    with recording() as log:
        transform(image=_night())
    assert [record["t"] for record in log] == ["SourceGlare"]


def test_glare_ladder_orders_and_float_matches_uint8() -> None:
    image = _night(lamps=((60, 80), (170, 140)))
    clean = image.astype(np.float32)
    strengths = []
    for level in sorted(SEVERITY):
        transform = SourceGlare(p=1.0, **SEVERITY[level]["SourceGlare"])
        samples = []
        for seed in range(10):
            transform.set_random_seed(seed)
            out = transform(image=image)["image"].astype(np.float32)
            samples.append(np.abs(out - clean).mean())
        strengths.append(float(np.mean(samples)))
    assert strengths[0] > 1.0
    assert all(a < b for a, b in zip(strengths, strengths[1:], strict=False)), strengths

    transform = SourceGlare(p=1.0)
    transform.set_random_seed(3)
    as_uint8 = transform(image=image)["image"].astype(np.float32)
    transform.set_random_seed(3)
    as_float = transform(image=(image / 255.0).astype(np.float32))["image"] * 255.0
    assert np.abs(as_float - as_uint8).mean() < 1.5


# ---------------------------------------------------------------------- fog


def _fog(image: np.ndarray, visibility: float, **kwargs: object) -> np.ndarray:
    """``image`` through ``NightFog`` at one visibility, seeded."""
    transform = NightFog(
        depth_provider=_provider(), p=1.0, visibility_range_m=(visibility, visibility), **kwargs
    )  # type: ignore[arg-type]
    transform.set_random_seed(0)
    return np.asarray(transform(image=image)["image"])


def test_night_fog_glows_round_the_lamp_and_dims_its_core() -> None:
    """What a daylight veil cannot do: the light the fog takes from a lamp
    comes back around it, and the fog is brightest where the lamp is."""
    image = _night()
    out = _fog(image, 40.0)

    assert out[LAMP[1], LAMP[0]].mean() <= image[LAMP[1], LAMP[0]].mean()
    halo = _ring(out, LAMP, 8, 24).mean() - _ring(image, LAMP, 8, 24).mean()
    assert halo > 6.0, f"no glow round the lamp: +{halo:.2f} levels"
    corner = out[-30:, -30:].astype(np.float32).mean()
    beside = _ring(out, LAMP, 30, 50).mean()
    assert beside > corner + 8.0, "fog is lit by the lamp, so it must fade away from it"
    assert corner < 30.0, "fog far from any light must stay dark"


def test_a_brightly_lit_road_is_not_a_lamp_to_the_fog() -> None:
    """A lamp was recorded at the sensor's limit and was many times
    brighter; the road under it was recorded as bright as it was. Counted
    as lamps, the pixels of a lit road that touched full scale came through
    thin fog several times over it: a white sheet with a ragged edge."""
    rng = np.random.default_rng(0)
    image = _night()
    # A pool of light on the road, grainy, some of it at full scale.
    pool = np.clip(rng.normal(238.0, 12.0, size=(70, 120)), 0, 255).astype(np.uint8)
    image[150:220, 60:180] = pool[..., None]
    assert float((image[150:220, 60:180] >= 250).mean()) > 0.1
    fogged = _fog(image, 400.0).astype(np.float32)
    road = fogged[160:210, 70:170].mean(axis=2)
    assert float((road >= 250).mean()) < 0.02, "the lit road, blown out by the fog"
    assert float(road.std()) < 14.0, "a ragged sheet where the road's grain crossed full scale"
    # The lamp is still a lamp: its light is in the fog round it.
    clear = image.astype(np.float32)
    assert float(_ring(fogged, LAMP, 8, 20).mean()) > float(_ring(clear, LAMP, 8, 20).mean()) + 10.0


def test_night_fog_is_a_veil_and_not_a_blur() -> None:
    """The failure this replaced: with a halo and no veil, fog was a dimmed
    frame plus a blurred copy of it. Real fog lifts the blacks and leaves
    edges where they were, so a sharp dark edge in a lit scene must come out
    paler on its dark side and no softer."""
    image = np.full((SIZE, SIZE, 3), 90, dtype=np.uint8)
    image[:, SIZE // 2 :] = 8
    out = _fog(image, 300.0).astype(np.float32).mean(axis=2)

    row = out[SIZE // 2]
    edge = SIZE // 2 - 1
    assert row[edge + 20] > 8.0 + 15.0, "the dark side was not veiled"
    step = np.abs(np.diff(row))
    beside = np.concatenate([step[edge - 4 : edge], step[edge + 1 : edge + 5]])
    assert step[edge] > 8.0 * beside.max(), "the edge was smeared over its neighbours"
    assert row[edge + 20] > row[-10], "the veil is brightest next to the light"


def test_thicker_fog_moves_a_lamp_s_light_from_its_core_to_its_halo() -> None:
    image = _night()

    def halo_to_core(visibility: float) -> float:
        out = _fog(image, visibility).astype(np.float32)
        veil = out[-30:, -30:].mean()
        return float((_ring(out, LAMP, 8, 24).mean() - veil) / out[LAMP[1], LAMP[0]].mean())

    assert halo_to_core(30.0) > 3.0 * halo_to_core(400.0)


def test_night_fog_is_lit_by_the_scene_s_own_light() -> None:
    """The veil's brightness and colour come from the light in the frame. A
    black frame has none and stays black; a warm lamp makes warm fog; and a
    scene twice as bright has fog twice as bright, not fog of a fixed grey."""
    black = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    assert int(_fog(black, 60.0).max()) == 0

    beside = _ring(_fog(_night(), 60.0), LAMP, 30, 50).mean(axis=0)
    assert beside[0] > beside[2] + 3.0, "fog round a warm lamp came out neutral"

    def veil(level: int) -> float:
        image = np.full((SIZE, SIZE, 3), level, dtype=np.uint8)
        gain = srgb_to_linear(_fog(image, 60.0)).mean() - srgb_to_linear(image).mean()
        return float(gain)

    dim, bright = veil(40), veil(57)  # sRGB 57 is twice the light of sRGB 40
    assert bright == pytest.approx(2.0 * dim, rel=0.15)


def test_the_illuminator_raises_a_wall_of_light_in_the_fog() -> None:
    """Under infrared the fog in front of the lens returns the camera's own
    light, evenly across the beam, and that is what blinds it."""
    dark = np.full((SIZE, SIZE, 3), 12, dtype=np.uint8)
    off = _fog(dark, 50.0).astype(np.float32)
    on = _fog(dark, 50.0, illuminator_backscatter_range=(0.2, 0.2), beam_falloff_range=(1.5, 1.5))
    on = on.astype(np.float32)

    assert off.max() <= 22.0, "lamp-lit fog over a dim frame is dim"
    centre, edge = on[100:140, 100:140].mean(), on[:20, :20].mean()
    assert centre > 60.0, "the wall should cost the frame its blacks"
    assert centre > edge > off.mean(), "brightest on the beam's axis, present across it"

    clear = _fog(dark, 20000.0, illuminator_backscatter_range=(0.2, 0.2)).astype(np.float32)
    assert clear.mean() < 0.25 * on.mean(), "no fog, no wall"


def test_night_fog_visibility_is_recorded() -> None:
    transform = NightFog(depth_provider=_provider(), p=1.0, visibility_range_m=(80.0, 80.0))
    with recording() as log:
        transform(image=_night())
    assert log[0]["drew"]["visibility_m"] == pytest.approx(80.0)
    assert log[0]["drew"]["beta"] == pytest.approx(3.912 / 80.0, rel=1e-3)


# --------------------------------------------------------------------- rain


def _added_by_rain(lamp_glint: float) -> tuple[float, float]:
    """Mean light rain adds beside the lamp and in the far corner, in levels."""
    size = 480
    lamp = (120, 160)
    image = np.full((size, size, 3), 12, dtype=np.uint8)
    cv2.circle(image, lamp, 6, (255, 235, 190), -1)
    transform = RandomRain(
        depth_provider=_provider(),
        p=1.0,
        rain_rate_range_mm_h=(100.0, 100.0),
        scene_illuminance_range_lux=(50.0, 50.0),
        veil_beta_range=(0.0, 0.0),
        wetness_probability=0.0,
        overcast_range=(0.0, 0.0),
        lamp_glint_range=(lamp_glint, lamp_glint),
    )
    added = np.zeros((size, size), dtype=np.float32)
    for seed in range(6):
        transform.set_random_seed(seed)
        added += (transform(image=image)["image"].astype(np.float32) - image).mean(axis=2)
    added /= 6.0
    yy, xx = np.mgrid[0:size, 0:size]
    distance = np.hypot(xx - lamp[0], yy - lamp[1])
    return float(added[(distance > 10) & (distance < 40)].mean()), float(added[-120:, -120:].mean())


def test_night_rain_shows_round_the_lamps_and_nowhere_else() -> None:
    """Rain at night is seen where a lamp lights it. Lit by the frame's
    average, as by day, it is either invisible everywhere or — when the
    average is read off the lamps — a grey hatching over a black sky."""
    beside, far = _added_by_rain(lamp_glint=8.0)
    assert beside > 8.0, f"no streaks lit beside the lamp: +{beside:.2f} levels"
    assert far < 1.0, f"streaks lit in the dark, far from any lamp: +{far:.2f} levels"

    beside_by_day_model, _ = _added_by_rain(lamp_glint=0.0)
    assert beside > 10.0 * max(beside_by_day_model, 0.1)


def test_lamp_light_on_rain_is_local_and_takes_the_lamp_s_colour() -> None:
    image = _night(lamps=((60, 80),), radius=5)
    light = _lamp_light(srgb_to_linear(image))
    at_lamp, far_away = light[80, 60], light[-1, -1]
    assert at_lamp.max() > 0.5 and far_away.max() < 1e-3
    assert at_lamp[0] > at_lamp[2], "a warm lamp lit the rain blue-white"

    unlit = np.full((SIZE, SIZE, 3), 40, dtype=np.uint8)
    assert float(_lamp_light(srgb_to_linear(unlit)).max()) == 0.0


def test_a_wet_road_reflects_a_lamp_at_more_than_its_clipped_value() -> None:
    """Two percent of a clipped pixel is nothing, and a real wet street is
    striped with the lamps' reflections: the lamp was far brighter than the
    sensor could record, and the reflection is a share of that."""
    image = _night(lamps=((120, 60),), radius=5)
    linear = srgb_to_linear(image)
    wet = np.zeros((SIZE, SIZE), dtype=np.float32)
    wet[SIZE // 2 :] = 1.0

    def reflection(lamp_gain: float) -> float:
        out = _apply_ground_wetness(
            linear, wet, darkening=0.5, horizon_px=SIZE / 2.0, focal_px=200.0, lamp_gain=lamp_gain
        )
        mirrored_row = SIZE - 60
        return float(out[mirrored_row - 6 : mirrored_row + 6, 114:126].mean())

    assert reflection(8.0) > 3.0 * reflection(1.0)


def test_a_lamp_lays_a_long_streak_down_a_soaked_road() -> None:
    """A rough film's facets mostly lie flat and a few are steep, so a lamp
    has a bright reflection close under it and a faint one running on down
    the road. The streak is narrow, it needs a film (a damp road has none),
    and it is found below the row the road is mirrored about, which is the
    road's far end and not the horizon."""
    size = 240
    image = np.full((size, size, 3), 12, dtype=np.uint8)
    cv2.circle(image, (120, 80), 4, (255, 200, 120), -1)
    linear = srgb_to_linear(image)
    wet = np.zeros((size, size), dtype=np.float32)
    wet[100:] = 1.0

    def road(darkening: float) -> np.ndarray:
        return _apply_ground_wetness(
            linear, wet, darkening, horizon_px=70.0, focal_px=300.0, lamp_gain=8.0, mirror_px=100.0
        )

    soaked = road(0.5)
    # The lamp is 20 rows above the road's end: its image is 20 rows below it,
    # and the streak is still there 60 rows further down, under the lamp alone.
    far_down = slice(175, 185)
    under_lamp = float(soaked[far_down, 116:125].mean())
    to_one_side = float(soaked[far_down, 30:60].mean())
    assert under_lamp > 2.0 * to_one_side
    # Narrow: a tenth of its length to either side it has all but gone.
    assert float(soaked[far_down, 150:160].mean()) < 0.5 * under_lamp
    # Warm, as the lamp is.
    assert soaked[180, 120, 0] > soaked[180, 120, 2]

    damp = road(0.9)
    assert float(damp[far_down, 116:125].mean()) < 1.2 * float(damp[far_down, 30:60].mean())


# ------------------------------------------------------------------ night snow


def test_a_near_flake_is_a_long_bright_streak_uneven_along_it_and_soft_at_its_edge() -> None:
    """Close to the lamp and behind a slow shutter a flake is a long bright
    streak, and it should be. What it was besides, and a real one is not:
    one level from end to end, and ruled at its edge."""
    from surmentations.atmosphere.night_snow import NightSnow

    def flake(x: float, smear: float) -> dict[str, list[float]]:
        return {
            "x": [x], "y": [120.0], "distance": [0.4], "radius": [4.0], "blur_radius": [1.0],
            "smear": [smear], "dir_x": [0.0], "dir_y": [1.0], "brightness": [0.6],
        }

    def drawn(x: float, smear: float, seed: int = 1) -> np.ndarray:
        flakes = {
            name: np.array(values, dtype=np.float32) for name, values in flake(x, smear).items()
        }
        snow = NightSnow(p=1.0)
        out = snow.apply(
            np.zeros((SIZE, SIZE, 3), dtype=np.float32),
            flakes=flakes,
            glare=0.0,
            beam=np.ones((SIZE, SIZE), dtype=np.float32),
            backscatter=0.0,
            tumble_seed=seed,
        )
        return np.asarray(out[..., 0])

    near = drawn(120.0, 120.0)
    along = near[75:165, 120]
    assert float(along.min()) > 0.15, "a streak with a gap in it"
    assert float(along.max()) > 1.25 * float(along.min()), "one level from end to end"
    across = near[120, 108:133]
    part_lit = (across > 0.1 * across.max()) & (across < 0.8 * across.max())
    assert int(part_lit.sum()) >= 2, "a ruled edge"
    # As long as it was: the shutter's doing, and nothing here shortens it.
    lit_rows = np.flatnonzero(near[:, 120] > 0.2 * float(near.max()))
    assert int(np.ptp(lit_rows)) > 100
    # Another fall, another turning; and a far flake is left as it was drawn.
    assert not np.array_equal(near, drawn(120.0, 120.0, seed=2))
    assert np.array_equal(drawn(120.0, 2.0, seed=1), drawn(120.0, 2.0, seed=2))


# ------------------------------------------------------- lamps on the rain


def _rain_under(
    lamps: object | None, image: np.ndarray, provider: SharedDepthProvider
) -> np.ndarray:
    """What NightRain adds, in levels, with no lamp at the lens and ``lamps`` in the scene."""
    from surmentations.atmosphere.night_rain import NightRain

    assert provider.set_lit_air(image, None, lamps)
    transform = NightRain(
        p=1.0,
        rain_rate_range_mm_h=(10.0, 10.0),
        illuminator_gain_range=(0.0, 0.0),
        backscatter_range=(0.0, 0.0),
        lamp_cone_range=(0.0, 0.0),
        horizontal_fov_range_deg=(60.0, 60.0),
    )
    transform.set_random_seed(3)
    return (transform(image=image)["image"].astype(np.float32) - image).mean(axis=2)


def test_rain_at_a_colour_night_is_lit_by_the_street_lamp_and_not_in_the_dark() -> None:
    """A night in colour has no lamp at the lens; the drops are lit by the
    lamps the night put in the scene, each by where it is. They used to be
    drawn at a gain of zero, and a third of the pipeline's night rain
    changed no pixel."""
    from surmentations.camera.night_lighting import SceneLamps
    from surmentations.utils.precipitation import focal_length_px

    image = np.full((SIZE, SIZE, 3), 10, dtype=np.uint8)
    provider = _provider()
    provider.get_depth(image)
    # The camera on the lamp's own post, 6.5 m over the ground and looking
    # level; the head 8 m up, half a metre ahead and a metre and a half to
    # the right, so the drops on the right of the frame are the nearer to
    # it. (Streaks fall down the frame from where their drop is, so above
    # and below are not compared.)
    lamps = SceneLamps(
        heads=np.array([[1.5, -1.5, 0.5, 8.0]], dtype=np.float32),
        street_level=1.0,
        street_colour=np.array([1.0, 0.8, 0.5], dtype=np.float32),
        sources=(),
        beam_level=0.0,
        head_colour=np.ones(3, dtype=np.float32),
        middle=np.array([0.0, 6.5, 10.0], dtype=np.float32),
        up=np.array([0.0, -1.0, 0.0], dtype=np.float32),
        focal_px=focal_length_px(SIZE, SIZE, 60.0),
        shape=(SIZE, SIZE),
        exposed=0.3,
    )
    nothing = _rain_under(None, image, provider)
    assert nothing.max() < 1.0, f"streaks lit with no lamp anywhere: +{nothing.max():.1f} levels"

    added = _rain_under(lamps, image, provider)
    # A drop half a metre from the lens, a pixel wide and a hundred long, is
    # faint even in a lamp's full light: alpha is w^2/L. Ten levels or so.
    assert added.max() > 8.0, f"no streak lit under the lamp: +{added.max():.1f} levels at most"
    assert float((added > 2.0).mean()) > 0.002
    # The same drops with the lamp moved to the left: each side of the frame
    # is lit more with the lamp on its own side. Paired, because where the
    # drops happen to fall decides more of a side's mean than the lamp does.
    mirrored = _rain_under(lamps._replace(heads=lamps.heads * [[-1, 1, 1, 1]]), image, provider)
    third = SIZE // 3
    right = float(added[:, -third:].sum()) / max(float(mirrored[:, -third:].sum()), 1e-6)
    left = float(mirrored[:, :third].sum()) / max(float(added[:, :third].sum()), 1e-6)
    assert right > 1.2 and left > 1.2, f"lamp's side lit {right:.2f}x and {left:.2f}x the other"

    with recording() as log:
        _rain_under(lamps, image, provider)
    drew = log[-1]["drew"]
    assert drew["lamp_lit_streaks"] > 0 and drew["lamp_lit_peak"] > 0.0


def test_rain_in_a_made_night_shows_in_front_of_the_lamps_in_the_picture() -> None:
    """With no lamp at the lens, the drops near the line of sight to a lamp
    the night lit throw its light at the lens, as RandomRain's do on a night
    photograph, and nothing else does: not a white thing the night left
    bright, which is not a lamp. Without it, rain in a night in colour
    changed no pixel on most frames."""
    from surmentations.atmosphere.night_rain import NightRain
    from surmentations.camera.night_lighting import SceneLamps
    from surmentations.utils.precipitation import focal_length_px

    size, lamp, shirt = 480, (120, 160), (360, 360)
    image = np.full((size, size, 3), 12, dtype=np.uint8)
    cv2.circle(image, lamp, 6, (255, 235, 190), -1)
    # A white thing as bright as the lamp, which the night did not light.
    cv2.circle(image, shirt, 6, (255, 255, 255), -1)
    focal = focal_length_px(size, size, 60.0)
    lamps = SceneLamps(
        heads=np.array(
            [[(lamp[0] - size / 2) * 10.0 / focal, (lamp[1] - size / 2) * 10.0 / focal, 10.0, 6.0]],
            dtype=np.float32,
        ),
        street_level=1.0,
        street_colour=np.array([1.0, 0.9, 0.7], dtype=np.float32),
        sources=(),
        beam_level=0.0,
        head_colour=np.ones(3, dtype=np.float32),
        middle=np.array([0.0, 6.0, 10.0], dtype=np.float32),
        up=np.array([0.0, -1.0, 0.0], dtype=np.float32),
        focal_px=focal,
        shape=(size, size),
    )
    provider = _provider()
    provider.get_depth(image)

    def added(glint: float, known: object | None) -> np.ndarray:
        assert provider.set_lit_air(image, None, known)
        transform = NightRain(
            p=1.0,
            rain_rate_range_mm_h=(10.0, 10.0),
            illuminator_gain_range=(0.0, 0.0),
            backscatter_range=(0.0, 0.0),
            lamp_cone_range=(0.0, 0.0),
            lamp_glint_range=(glint, glint),
            horizontal_fov_range_deg=(60.0, 60.0),
        )
        total = np.zeros((size, size), dtype=np.float32)
        for seed in range(4):
            transform.set_random_seed(seed)
            total += (transform(image=image)["image"].astype(np.float32) - image).mean(axis=2)
        return total / 4.0

    def ring(field: np.ndarray, centre: tuple[int, int]) -> float:
        yy, xx = np.mgrid[0:size, 0:size]
        distance = np.hypot(xx - centre[0], yy - centre[1])
        return float(field[(distance > 10) & (distance < 40)].mean())

    lit = added(8.0, lamps)
    assert ring(lit, lamp) > 3.0, f"no streaks lit beside the lamp: +{ring(lit, lamp):.2f} levels"
    assert ring(lit, shirt) < 0.5, "streaks lit round a white thing the night did not light"
    assert float(lit[:100, -100:].mean()) < 0.5, "streaks lit in the dark, far from any lamp"
    assert float(added(8.0, None).max()) < 1.0, "a frame with no lamps known has no glint"
    assert float(added(0.0, lamps).max()) < 1.0, "with no glint the rain adds nothing here"
