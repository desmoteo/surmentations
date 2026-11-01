"""Fast and exact mode: that the switch works, and what it is allowed to change.

Exact mode computes what the reference numpy and OpenCV code computes. Fast
mode may differ from it, and the amounts asserted here are the contract:
they are what ``docs/native_kernels.md`` quotes. A new fast path goes in with
a test of its own in this file.
"""

from __future__ import annotations

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

import weathermentations
from weathermentations import _native, mode
from weathermentations.atmosphere import NightSnow, rain
from weathermentations.atmosphere.snow import render_flake_field
from weathermentations.camera import SensorNoise, TemporalDenoiseArtifacts
from weathermentations.depth.normals import upward_facing
from weathermentations.lens import DirtSoil, HousingOccluders, SunFlare
from weathermentations.lens.sun_flare import diffraction_starburst, polygon_distance
from weathermentations.utils.channels import weighted_colour
from weathermentations.utils.filters import wide_blur, wide_dilate
from weathermentations.utils.grids import centered_grid
from weathermentations.utils.ground import ground_noise


@pytest.fixture(autouse=True)
def _restore_mode():
    before = mode.fast()
    yield
    mode.set_fast(before)


# ------------------------------------------------------------- the switch


def test_the_mode_is_set_and_restored() -> None:
    weathermentations.set_fast(True)
    assert weathermentations.fast()
    with weathermentations.exact():
        assert not weathermentations.fast()
        with weathermentations.exact():
            assert not weathermentations.fast()
        assert not weathermentations.fast()
    assert weathermentations.fast()

    weathermentations.set_fast(False)
    with weathermentations.exact():
        pass
    assert not weathermentations.fast(), "leaving the block must not switch fast mode on"


def test_an_error_inside_an_exact_block_still_restores_the_mode() -> None:
    mode.set_fast(True)
    with pytest.raises(RuntimeError), mode.exact():
        raise RuntimeError
    assert mode.fast()


@pytest.mark.parametrize(
    "value, expected", [("0", False), ("off", False), ("1", True), (None, True)]
)
def test_the_environment_sets_the_mode_at_import(value: str | None, expected: bool) -> None:
    """Fast unless told otherwise, and told through the environment for a
    spawned worker, which never sees the parent's `set_fast`."""
    env = {key: item for key, item in os.environ.items() if key != "WEATHERMENTATIONS_FAST"}
    if value is not None:
        env["WEATHERMENTATIONS_FAST"] = value
    code = "import weathermentations; print(weathermentations.fast())"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.stdout.strip() == str(expected), out.stderr


def test_a_scalar_keeps_a_field_float32_only_in_fast_mode() -> None:
    field = np.ones((4, 4), dtype=np.float32)
    mode.set_fast(True)
    assert (field * mode.scalar(np.cos(0.3))).dtype == np.float32
    mode.set_fast(False)
    assert (field * mode.scalar(np.cos(0.3))).dtype == np.float64


# ------------------------------------------------- exact mode is the reference


def _frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.uniform(size=(240, 320, 3)).astype(np.float32)
    image[:80] *= 0.3
    return image


def test_exact_mode_filters_at_full_size() -> None:
    image = _frame()
    mask = np.zeros((240, 320), dtype=np.float32)
    mask[:70] = 1.0
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    with mode.exact():
        blurred, grown = wide_blur(image, 40.0), wide_dilate(mask, 41)
    assert np.array_equal(blurred, cv2.GaussianBlur(image, (0, 0), sigmaX=40.0))
    assert np.array_equal(grown, cv2.dilate(mask, disc))

    mode.set_fast(True)
    assert not np.array_equal(wide_blur(image, 40.0), blurred)
    assert not np.array_equal(wide_dilate(mask, 41), grown)


def test_exact_mode_sums_a_weighted_colour_as_the_reduction_does() -> None:
    image = _frame(1)
    weight = np.random.default_rng(2).uniform(size=(240, 320)).astype(np.float32)
    reference = (image * weight[..., None]).sum(axis=(0, 1))
    with mode.exact():
        assert np.array_equal(weighted_colour(image, weight), reference)
    mode.set_fast(True)
    assert weighted_colour(image, weight) == pytest.approx(reference, rel=1e-5)


def test_the_cached_starburst_does_not_depend_on_the_mode() -> None:
    """It is computed once per blade count and kept. Were its aperture drawn
    in the mode of the moment, the first caller would decide the pattern for
    every later one."""
    build = diffraction_starburst.__wrapped__
    mode.set_fast(True)
    in_fast = build(6)
    mode.set_fast(False)
    assert np.array_equal(in_fast, build(6))


# --------------------------------------------------- what fast mode may change


def test_a_coarser_working_sigma_reduces_sooner_and_differs_by_more() -> None:
    image = _frame()
    exact = cv2.GaussianBlur(image, (0, 0), sigmaX=10.0)
    mode.set_fast(True)
    # At the default, a sigma of 10 pixels is not worth reducing.
    assert np.array_equal(wide_blur(image, 10.0), exact)
    coarse = wide_blur(image, 10.0, work_sigma=4.0)
    assert not np.array_equal(coarse, exact)
    assert np.abs(coarse - exact).max() < 0.02 * exact.max()
    mode.set_fast(False)
    assert np.array_equal(wide_blur(image, 10.0, work_sigma=4.0), exact)


def test_a_blur_longer_one_way_is_reduced_that_way_only() -> None:
    image = _frame()
    exact = cv2.GaussianBlur(image, (0, 0), sigmaX=3.0, sigmaY=30.0)
    mode.set_fast(True)
    fast = wide_blur(image, 3.0, sigma_y=30.0)
    assert fast.shape == image.shape and not np.array_equal(fast, exact)
    assert np.abs(fast - exact).max() < 0.02 * exact.max()
    mode.set_fast(False)
    assert np.array_equal(wide_blur(image, 3.0, sigma_y=30.0), exact)


def test_normals_read_at_the_depth_map_s_size_find_the_same_ground() -> None:
    """A road receding under a camera, a kerb across it and a box on it, at
    1200 pixels. Fast mode reads the surface normals at 518 and enlarges
    the result; what counts as level ground moves at the edges only."""
    height, width = 900, 1200
    rows = np.arange(height, dtype=np.float32).reshape(height, 1)
    depth = np.broadcast_to(1.5 * 800.0 / np.maximum(rows - 250.0, 1.0), (height, width)).copy()
    depth = np.minimum(depth, 60.0)
    depth[500:700, 400:650] = depth[700, 0]  # a box standing on the road
    depth[780:800] *= 0.97  # a kerb
    facing = {}
    for flag in (True, False):
        mode.set_fast(flag)
        facing[flag] = upward_facing(depth, camera_pitch_deg=10.0)
    assert facing[True].shape == facing[False].shape == (height, width)
    ground = {flag: facing[flag] > 0.5 for flag in facing}
    assert 0.2 < ground[False].mean() < 0.9
    assert (ground[True] == ground[False]).mean() > 0.97
    # A frame of the depth map's own size is read as it is, in either mode.
    small = np.ascontiguousarray(depth[::2, ::2])
    mode.set_fast(True)
    in_fast = upward_facing(small)
    mode.set_fast(False)
    assert np.array_equal(in_fast, upward_facing(small))


def _street(height: int = 900, width: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    """A grey road receding under a camera with a box on it, and its depth."""
    rows = np.arange(height, dtype=np.float32).reshape(height, 1)
    depth = np.broadcast_to(1.5 * 800.0 / np.maximum(rows - 250.0, 1.0), (height, width)).copy()
    depth = np.minimum(depth, 60.0)
    depth[500:700, 400:650] = depth[700, 0]
    rng = np.random.default_rng(4)
    scene = np.full((height, width, 3), 0.18, dtype=np.float32)
    scene[:250] = (0.5, 0.6, 0.75)
    scene[500:700, 400:650] = (0.4, 0.05, 0.05)
    scene += rng.normal(0.0, 0.004, scene.shape).astype(np.float32)
    return depth, np.clip(scene, 0.0, 1.0)


def test_the_wet_ground_read_at_the_depth_map_s_size_is_the_same_ground() -> None:
    """Fast mode reads where the road is wet on a 518-pixel copy and
    enlarges the map. Same road, same horizon, edges a pixel or two off."""
    depth, scene = _street()
    read = {}
    for flag in (True, False):
        mode.set_fast(flag)
        read[flag] = rain._wet_ground(depth, scene, 60.0, 10.0, 6.0)
    (fast_wet, fast_horizon), (exact_wet, exact_horizon) = read[True], read[False]
    assert fast_wet.shape == exact_wet.shape == depth.shape and fast_wet.dtype == np.float32
    assert 0.2 < (exact_wet > 0.5).mean() < 0.9
    assert ((fast_wet > 0.5) == (exact_wet > 0.5)).mean() > 0.97
    assert abs(fast_horizon - exact_horizon) < 0.01 * depth.shape[0]
    # The box is dry in both, and so is the sky.
    assert fast_wet[550:650, 450:600].max() < 0.2 and fast_wet[:200].max() < 0.05


def test_puddles_laid_out_at_the_depth_map_s_size_cover_the_same_share() -> None:
    depth, _ = _street()
    cover = {}
    for flag in (True, False):
        mode.set_fast(flag)
        cover[flag] = rain._puddles(depth, 1040.0, 10.0, 0.3, np.random.default_rng(8))
    assert cover[True].shape == cover[False].shape == depth.shape
    assert cover[False].mean() == pytest.approx(0.3, abs=0.05)
    assert cover[True].mean() == pytest.approx(cover[False].mean(), abs=0.03)
    assert np.abs(cover[True] - cover[False]).mean() < 0.05


def test_the_ground_pattern_is_folded_to_the_same_place_either_way() -> None:
    """Fast mode folds the noise tile's coordinate with a floor where exact
    mode takes a remainder. Millionths of a pixel apart."""
    depth = np.repeat(np.linspace(40.0, 2.0, 200, dtype=np.float32)[:, None], 300, axis=1)
    fields = {}
    for flag in (True, False):
        mode.set_fast(flag)
        fields[flag] = ground_noise(depth, 400.0, 20.0, 1.2, np.random.default_rng(5))
    assert fields[False].std() > 0.05
    assert np.abs(fields[True] - fields[False]).max() < 1e-3


def test_fast_mode_keeps_aperture_fields_float32_and_within_rounding() -> None:
    dy, dx = centered_grid(129, scale=40.0)
    mode.set_fast(False)
    reference = polygon_distance(dy, dx, 7, 0.3)
    mode.set_fast(True)
    fast = polygon_distance(dy, dx, 7, 0.3)
    assert reference.dtype == np.float64 and fast.dtype == np.float32
    assert np.abs(fast - reference).max() < 1e-6


def _flakes(count: int, size: tuple[int, int], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    columns = {
        "x": rng.uniform(-30.0, size[1] + 30.0, count),
        "y": rng.uniform(-30.0, size[0] + 30.0, count),
        "radius": rng.uniform(0.3, 4.0, count),
        "blur_radius": rng.uniform(0.0, 1.0, count) ** 4 * 60.0,
        "smear": rng.uniform(0.0, 12.0, count),
        "dir_x": np.sin(angle),
        "dir_y": np.cos(angle),
    }
    return {key: value.astype(np.float32) for key, value in columns.items()}


@pytest.mark.parametrize("use_kernels", [True, False], ids=["kernels", "numpy"])
@pytest.mark.parametrize("blades", [0, 6, 9])
def test_fast_sprites_are_the_exact_ones_to_a_few_parts_in_a_million(
    use_kernels: bool, blades: int
) -> None:
    """Fast mode multiplies by a reciprocal where exact mode divides, in the
    kernel, and keeps the numpy field float32. A sprite's peak is about 1."""
    if use_kernels and not _native.enabled():
        pytest.skip("the native kernels are not built or are switched off")
    was_enabled = _native.enabled()
    _native.set_enabled(use_kernels)
    try:
        size = (300, 420)
        flakes = _flakes(300, size, seed=blades)
        layers = {}
        for flag in (True, False):
            mode.set_fast(flag)
            layers[flag] = np.zeros(size, dtype=np.float32)
            render_flake_field(layers[flag], flakes, blades, 0.3, 1.3, 0.4, (210.0, 150.0))
    finally:
        _native.set_enabled(was_enabled)
    assert layers[False].max() > 0.1
    assert np.abs(layers[True] - layers[False]).max() < 5e-6


def _scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    image = rng.integers(20, 90, (240, 320, 3), dtype=np.uint8)
    image[:90] += 120
    image[40:46, 200:206] = 255
    return image


def _large_scene() -> np.ndarray:
    """Large enough for a blur a hundredth of the diagonal to be reduced."""
    image = cv2.resize(_scene(), (1280, 960), interpolation=cv2.INTER_CUBIC)
    noise = np.random.default_rng(1).integers(-6, 7, image.shape)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    "transform",
    [HousingOccluders(p=1.0), DirtSoil(p=1.0), TemporalDenoiseArtifacts(p=1.0)],
    ids=["occluders", "dirt", "denoise"],
)
def test_a_reduced_blur_moves_a_frame_by_a_few_levels_at_most(transform) -> None:
    """Defocused dirt and insects and smeared chroma: fields that are soft by
    nature, blurred on a smaller copy in fast mode."""
    image = _large_scene()
    drew = 0
    for seed in range(4):
        frames = {}
        for flag in (True, False):
            mode.set_fast(flag)
            transform.set_random_seed(seed)
            frames[flag] = transform(image=image)["image"].astype(np.int16)
        difference = np.abs(frames[True] - frames[False])
        # An occluder field may legitimately come up empty for a seed.
        drew += not np.array_equal(frames[False], image)
        assert difference.max() <= 4
        assert difference.mean() < 0.1
    assert drew >= 2, "the fixture hardly ever drew anything"


@pytest.mark.parametrize(
    "transform",
    [SunFlare(p=1.0), NightSnow(p=1.0), SensorNoise(p=1.0)],
    ids=["sun-flare", "night-snow", "sensor-noise"],
)
def test_a_frame_differs_between_the_modes_by_one_level_at_most(transform) -> None:
    """Of the transforms that need no depth. Those that redraw the sky's edge
    differ by more, along the skyline only; that is measured on real scenes
    and quoted in docs/native_kernels.md."""
    image = _scene()
    for seed in range(3):
        frames = {}
        for flag in (True, False):
            mode.set_fast(flag)
            transform.set_random_seed(seed)
            frames[flag] = transform(image=image)["image"].astype(np.int16)
        difference = np.abs(frames[True] - frames[False])
        assert difference.max() <= 1
        assert (difference > 0).mean() < 0.01
