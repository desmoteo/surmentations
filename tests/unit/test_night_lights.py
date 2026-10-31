"""Glare and fog at night are things the lights in the frame do.

Both transforms here used to be missing, and the daylight models that stood
in for them were wrong in the same way: nothing was tied to where the lamps
are. Fog was an even veil, and flare needed a position chosen by the caller.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from weathermentations.atmosphere.night_fog import NightFog
from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.diagnostics import recording
from weathermentations.lens.source_glare import SourceGlare
from weathermentations.presets.severity import SEVERITY
from weathermentations.utils.photometry import srgb_to_linear
from weathermentations.utils.sources import light_sources

SIZE = 240
LAMP = (60, 80)


def _night(
    lamps: tuple[tuple[int, int], ...] = (LAMP,), radius: int = 4, peak: int = 255
) -> np.ndarray:
    image = np.full((SIZE, SIZE, 3), 12, dtype=np.uint8)
    for x, y in lamps:
        cv2.circle(image, (x, y), radius, (peak, peak - 20, peak - 65), -1)
    return image


class _Wall(DepthEstimator):
    """Everything at the far plane, so fog acts evenly on the whole frame."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        depth = np.ones(image.shape[:2], dtype=np.float32)
        depth[0, 0] = 0.0  # keeps the provider's range from collapsing
        return depth


def _provider() -> SharedDepthProvider:
    return SharedDepthProvider(_Wall(), depth_range_m=(3.0, 50.0))


def _ring(image: np.ndarray, centre: tuple[int, int], inner: float, outer: float) -> np.ndarray:
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
    transform = NightFog(
        depth_provider=_provider(), p=1.0, visibility_range_m=(visibility, visibility), **kwargs
    )  # type: ignore[arg-type]
    transform.set_random_seed(0)
    return np.asarray(transform(image=image)["image"])


def test_night_fog_glows_round_the_lamp_and_dims_its_core() -> None:
    """What a daylight veil cannot do: the light the fog takes from a lamp
    comes back around it, and nowhere else."""
    image = _night()
    out = _fog(image, 40.0)

    assert out[LAMP[1], LAMP[0]].mean() <= image[LAMP[1], LAMP[0]].mean()
    halo = _ring(out, LAMP, 8, 24).mean() - _ring(image, LAMP, 8, 24).mean()
    assert halo > 6.0, f"no glow round the lamp: +{halo:.2f} levels"
    corner = out[-30:, -30:].astype(np.float32).mean()
    assert corner < 12.0 + 3.0, "fog far from any light must stay dark"


def test_thicker_fog_spreads_the_glow_further() -> None:
    image = _night()
    thin, thick = _fog(image, 400.0), _fog(image, 30.0)

    def spread(out: np.ndarray) -> float:
        gain = np.clip(out.astype(np.float32).mean(axis=2) - 12.0, 0.0, None)
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        distance = np.hypot(xx - LAMP[0], yy - LAMP[1])
        return float((gain * distance).sum() / gain.sum())

    assert spread(thick) > 1.5 * spread(thin)


def test_night_fog_adds_no_light_of_its_own() -> None:
    """With the illuminator off, all the light in the frame was already
    there, and the total can only fall. (For lamps the sensor did not clip:
    a clipped one is counted as brighter than its recorded value, which is
    the point of the over-range and not a creation of light.)"""
    image = _night(lamps=((60, 80), (170, 140)), peak=225)
    out = _fog(image, 60.0)
    assert float(srgb_to_linear(out).sum()) <= float(srgb_to_linear(image).sum()) * 1.02
    black = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    assert int(_fog(black, 60.0).max()) == 0


def test_the_illuminator_raises_a_wall_of_light_in_the_fog() -> None:
    """Under infrared the fog in front of the lens returns the camera's own
    light, evenly across the beam, and that is what blinds it."""
    dark = np.full((SIZE, SIZE, 3), 12, dtype=np.uint8)
    off = _fog(dark, 50.0).astype(np.float32)
    on = _fog(dark, 50.0, illuminator_backscatter_range=(0.2, 0.2), beam_falloff_range=(1.5, 1.5))
    on = on.astype(np.float32)

    assert off.max() <= 13.0
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
