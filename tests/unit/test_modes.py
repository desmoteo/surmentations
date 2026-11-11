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
from weathermentations.atmosphere import NightSnow
from weathermentations.atmosphere.snow import render_flake_field
from weathermentations.camera import SensorNoise
from weathermentations.lens import SunFlare
from weathermentations.lens.sun_flare import diffraction_starburst, polygon_distance
from weathermentations.utils.channels import weighted_colour
from weathermentations.utils.grids import centered_grid
from weathermentations.utils.sky import wide_blur, wide_dilate


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
