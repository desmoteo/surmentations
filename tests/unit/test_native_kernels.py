"""The compiled kernels against the numpy code they stand in for.

The library is optional, so the numpy path is the definition and the kernels
are held to it: bit for bit where the arithmetic allows (the colour tables,
unblurred sprites and streaks, the wet road, the denoise filter against
OpenCV's), to float rounding where a blur's sums are taken in another order.
Every instruction-set tier of the build must give the same bytes as every
other, so that a picture does not depend on the CPU.

Without the library, or with it switched off, these tests skip, and the last
section checks the numpy path alone, which is what such an install runs.
"""

from __future__ import annotations

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from weathermentations import _native, mode
from weathermentations.atmosphere import NightRain, NightSnow
from weathermentations.atmosphere import rain as rain_module
from weathermentations.atmosphere.snow import render_flake_field
from weathermentations.camera import TemporalDenoiseArtifacts
from weathermentations.camera import denoise as denoise_module
from weathermentations.lens import DirtSoil, SpiderWeb, WaterDroplets
from weathermentations.lens import water_droplets as droplets_module
from weathermentations.utils import ground as ground_module
from weathermentations.utils import overcast, photometry
from weathermentations.utils.illuminator import splat_discs

# Built and switched on: with WEATHERMENTATIONS_NATIVE=0 the library is there
# and every comparison below would be numpy against itself.
requires_native = pytest.mark.skipif(
    not _native.enabled(),
    reason="the native kernels are not built (python hatch_build.py) or are switched off",
)

TIERS = _native.tiers() or ("none",)


@pytest.fixture(autouse=True)
def _exact_mode_and_restored_state():
    """Exact mode is where the kernels are held to the numpy code; what fast
    mode may differ by is in test_modes.py. And a test that switches tier or
    disables the kernels must not leak it."""
    was_enabled, tier, was_fast = _native.enabled(), _native.tier(), mode.fast()
    mode.set_fast(False)
    yield
    mode.set_fast(was_fast)
    if tier is not None:
        _native.set_tier(tier)
    _native.set_enabled(was_enabled)


# ----------------------------------------------------------------- colour


def _linear(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Linear light with values below 0 and above 1, as renderers produce."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.2, 1.4, shape).astype(np.float32)


def _textbook_encode_u8(linear: np.ndarray) -> np.ndarray:
    """`from_linear_unit` as it was before it had a table of code values."""
    index = (np.clip(linear, 0.0, 1.0) * 65535).astype(np.int32)
    encoded = photometry._LINEAR_TO_SRGB_LUT[index] * 255.0
    return np.clip(np.rint(encoded), 0, 255).astype(np.uint8)


@requires_native
@pytest.mark.parametrize("tier", TIERS)
# 96 x 96 x 3 is not a multiple of the kernel's block of 512, nor of a vector.
@pytest.mark.parametrize("shape", [(96, 96, 3), (61, 47, 3), (300, 50, 3)])
def test_encode_and_decode_are_exact(tier: str, shape: tuple[int, ...]) -> None:
    _native.set_tier(tier)
    linear = _linear(shape)
    assert linear.size >= photometry._EXACT_BELOW
    native = photometry.from_linear_unit(linear, 255.0, True, np.uint8)
    as_float = photometry.from_linear_unit(linear, 1.0, False, np.float32)
    decoded = photometry.to_linear_unit(native)[0]
    with _native.disabled():
        assert np.array_equal(native, photometry.from_linear_unit(linear, 255.0, True, np.uint8))
        assert np.array_equal(as_float, photometry.from_linear_unit(linear, 1.0, False, np.float32))
        assert np.array_equal(decoded, photometry.to_linear_unit(native)[0])
    assert np.array_equal(native, _textbook_encode_u8(linear))
    assert native.dtype == np.uint8 and as_float.dtype == np.float32
    assert decoded.dtype == np.float32


@requires_native
def test_the_table_ends_are_reached_and_not_passed() -> None:
    """Exactly 0 and exactly 1 index the first and last entries; a value a
    hair inside 1 must not round up past the table."""
    edge = np.array([0.0, 1.0, np.nextafter(np.float32(1.0), np.float32(0.0)), -0.0, 5.0, -5.0])
    linear = np.tile(edge.astype(np.float32), 4096)
    out = _native.encode_u8(linear, photometry._LINEAR_TO_SRGB_U8_LUT)
    assert out is not None
    assert out[:6].tolist() == [0, 255, 255, 0, 255, 0]


@requires_native
def test_a_view_that_is_not_contiguous_is_converted_whole() -> None:
    linear = _linear((128, 200, 3))
    view = linear[::2, ::-1]
    native = photometry.from_linear_unit(view, 255.0, True, np.uint8)
    assert native.shape == view.shape
    assert np.array_equal(native, _textbook_encode_u8(view))


def test_nan_is_refused_the_same_way_with_and_without_the_kernels() -> None:
    """A NaN in a frame is a renderer's bug, and must not come out as a black
    pixel. It was refused by accident on x86-64, where converting it to a
    table index gave one out of range, and went through as black on arm64,
    where it gave 0. It is now looked for, on every machine, either way."""
    linear = _linear((96, 96, 3))
    linear[40, 40, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        photometry.from_linear_unit(linear, 255.0, True, np.uint8)
    with _native.disabled(), pytest.raises(ValueError, match="NaN"):
        photometry.from_linear_unit(linear, 255.0, True, np.uint8)
    float_table = photometry._LINEAR_TO_SRGB_LUT
    with pytest.raises(ValueError, match="NaN"):
        photometry.through_table(linear, float_table)
    with _native.disabled(), pytest.raises(ValueError, match="NaN"):
        photometry.through_table(linear, float_table)


# ----------------------------------------------------------------- flakes


def _flakes(count: int, size: tuple[int, int], largest: float, seed: int) -> dict[str, np.ndarray]:
    """Flakes over and around a ``size`` frame, most small and a few huge."""
    rng = np.random.default_rng(seed)
    height, width = size
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    columns = {
        "x": rng.uniform(-30.0, width + 30.0, count),
        "y": rng.uniform(-30.0, height + 30.0, count),
        "radius": rng.uniform(0.3, 4.0, count),
        "blur_radius": rng.uniform(0.0, 1.0, count) ** 4 * largest,
        "smear": rng.uniform(0.0, 12.0, count),
        "dir_x": np.sin(angle),
        "dir_y": np.cos(angle),
        "brightness": rng.uniform(0.2, 2.0, count),
    }
    return {key: value.astype(np.float32) for key, value in columns.items()}


def _both_ways(size, flakes, **optics) -> tuple[np.ndarray, np.ndarray]:
    """The same flake field rendered with the kernel and with numpy."""
    native = np.zeros(size, dtype=np.float32)
    render_flake_field(native, flakes, **optics)
    reference = np.zeros(size, dtype=np.float32)
    with _native.disabled():
        render_flake_field(reference, flakes, **optics)
    return native, reference


ROUND = {"blades": 0, "blade_rotation": 0.0, "rim_gain": 1.0, "cats_eye": 0.0}
IRIS = {"blades": 6, "blade_rotation": 0.3, "rim_gain": 1.4, "cats_eye": 0.5}


@requires_native
@pytest.mark.parametrize("tier", TIERS)
def test_small_round_flakes_are_exact(tier: str) -> None:
    """Below a render radius of 17 pixels a sprite is not blurred, and every
    step is float32 arithmetic in numpy's order: the same bits."""
    _native.set_tier(tier)
    size = (240, 320)
    flakes = _flakes(800, size, largest=10.0, seed=1)
    native, reference = _both_ways(size, flakes, frame_centre=(160.0, 120.0), **ROUND)
    assert reference.max() > 0.1, "the fixture drew nothing"
    assert np.array_equal(native, reference)


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize(
    "optics, gain",
    [
        pytest.param(ROUND, True, id="round-lit-by-a-lamp"),
        pytest.param(IRIS, False, id="iris-with-cats-eye"),
        pytest.param({**IRIS, "blades": 7, "cats_eye": 0.0, "rim_gain": 0.7}, False, id="seven"),
        pytest.param({**ROUND, "cats_eye": 0.6, "rim_gain": 1.2}, False, id="round-cats-eye"),
    ],
)
def test_large_flakes_agree_to_float_rounding(tier: str, optics: dict, gain: bool) -> None:
    """Large sprites are blurred, and a polygonal aperture is evaluated in
    float64 by the numpy code (an accident of its scalar types) and float32
    here. Either moves a sprite by rounding and no more."""
    _native.set_tier(tier)
    size = (300, 420)
    flakes = _flakes(400, size, largest=70.0, seed=2)
    native, reference = _both_ways(
        size,
        flakes,
        frame_centre=(210.0, 150.0),
        gain=flakes["brightness"] if gain else None,
        **optics,
    )
    assert reference.max() > 0.1
    assert np.abs(native - reference).max() < 5e-6


@requires_native
def test_flakes_off_the_frame_and_empty_fields_are_handled() -> None:
    size = (40, 60)
    far = {key: value[:5].copy() for key, value in _flakes(5, size, 5.0, 3).items()}
    far["x"] += 5000.0
    native, reference = _both_ways(size, far, frame_centre=(30.0, 20.0), **ROUND)
    assert not native.any() and not reference.any()

    none = {key: value[:0] for key, value in far.items()}
    native, reference = _both_ways(size, none, frame_centre=(30.0, 20.0), **ROUND)
    assert not native.any()

    # A sprite far larger than the canvas it lands on.
    huge = {key: value[:1].copy() for key, value in _flakes(1, size, 5.0, 4).items()}
    huge["x"][:], huge["y"][:], huge["blur_radius"][:] = 30.0, 20.0, 400.0
    native, reference = _both_ways(size, huge, frame_centre=(30.0, 20.0), **IRIS)
    assert reference.max() > 0.0
    assert np.abs(native - reference).max() < 5e-6


def test_a_flake_that_is_not_a_number_is_an_error_either_way() -> None:
    """The kernel declines a field it cannot size a sprite for, and the loop
    raises as it did before there was a kernel."""
    size = (40, 60)
    flakes = _flakes(8, size, 5.0, 3)
    flakes["x"][5] = np.nan
    for use_kernels in (True, False):
        _native.set_enabled(use_kernels)
        with pytest.raises(ValueError):
            render_flake_field(
                np.zeros(size, dtype=np.float32), flakes, frame_centre=(30.0, 20.0), **ROUND
            )


@requires_native
def test_a_canvas_the_kernel_cannot_write_is_left_to_numpy() -> None:
    """Not contiguous, so the kernel declines and the loop draws it."""
    size = (120, 160)
    flakes = _flakes(50, size, 8.0, 5)
    wide = np.zeros((120, 320), dtype=np.float32)
    view = wide[:, ::2]
    render_flake_field(view, flakes, frame_centre=(80.0, 60.0), **ROUND)
    reference = np.zeros(size, dtype=np.float32)
    with _native.disabled():
        render_flake_field(reference, flakes, frame_centre=(80.0, 60.0), **ROUND)
    assert np.array_equal(view, reference)
    assert not wide[:, 1::2].any()


# ---------------------------------------------------------------- streaks


def _streaks(count: int, size: int, seed: int, widths: tuple[int, int], blur: float) -> dict:
    """Streaks at every heading, falling and rising, short and long."""
    rng = np.random.default_rng(seed)
    rising = np.where(rng.uniform(size=count) < 0.3, 180.0, 0.0)
    columns = {
        "x": rng.uniform(-40.0, size + 40.0, count),
        "y": rng.uniform(-60.0, size + 40.0, count),
        "length": rng.uniform(0.0, 1.0, count) ** 3 * 70.0 + 0.3,
        "angle": rng.uniform(-80.0, 80.0, count) + rising,
        "bend": rng.uniform(-0.15, 0.15, count),
        "alpha": rng.uniform(0.01, 0.2, count),
        "phase": rng.uniform(0.0, 2.0 * np.pi, count),
        "blur_sigma": rng.uniform(0.0, 1.0, count) ** 2 * blur,
    }
    streaks = {key: value.astype(np.float32) for key, value in columns.items()}
    streaks["thickness"] = rng.integers(widths[0], widths[1] + 1, count).astype(np.int32)
    return streaks


def _layers(size: int, streaks: dict) -> tuple[np.ndarray, np.ndarray]:
    """The same streak layer rendered with the kernel and with numpy."""
    native = rain_module.render_streak_layer(size, size, streaks)
    with _native.disabled():
        reference = rain_module.render_streak_layer(size, size, streaks)
    return native, reference


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("widths", [(1, 1), (2, 8), (1, 8)], ids=["thin", "wide", "mixed"])
def test_unblurred_streaks_set_the_pixels_opencv_sets(tier: str, widths: tuple[int, int]) -> None:
    """The kernel walks lines, fills quadrilaterals and caps them as
    OpenCV's drawing code does. One pixel more or less on one streak in
    thousands would show here as a difference of that streak's alpha."""
    _native.set_tier(tier)
    for seed in range(3):
        streaks = _streaks(4000, 256, seed, widths, blur=0.0)
        native, reference = _layers(256, streaks)
        assert np.count_nonzero(reference) > 20_000
        assert np.array_equal(native, reference), f"seed {seed}"


@requires_native
@pytest.mark.parametrize("tier", TIERS)
def test_blurred_streaks_agree_to_float_rounding(tier: str) -> None:
    _native.set_tier(tier)
    streaks = _streaks(3000, 256, 7, (1, 8), blur=2.5)
    assert (streaks["blur_sigma"] > 0.3).mean() > 0.3
    native, reference = _layers(256, streaks)
    assert np.abs(native - reference).max() < 1e-6
    assert native.sum() == pytest.approx(reference.sum(), rel=1e-6)


@requires_native
def test_every_streak_of_a_real_draw_is_drawn_natively() -> None:
    """The kernel takes a streak only when its patch has room for the line,
    since OpenCV clips one that does not and the kernel does not reproduce
    the clipping. `streak_geometry` leaves that room, so nothing should be
    left for the loop; if its margins change, this says so."""
    transform = NightRain(p=1.0)
    image = np.zeros((360, 480, 3), dtype=np.uint8)
    for seed in range(5):
        transform.set_random_seed(seed)
        params = transform.get_params_dependent_on_data({"shape": image.shape}, {"image": image})
        streaks = params["streaks"]
        geometry = rain_module.streak_geometry(streaks)
        layer = np.zeros((360, 480), dtype=np.float32)
        left = rain_module._render_streaks_natively(layer, geometry, streaks)
        assert len(streaks["alpha"]) > 100
        assert len(left) == 0


@requires_native
def test_a_streak_without_room_in_its_patch_is_left_to_the_loop(monkeypatch) -> None:
    streaks = _streaks(200, 128, 9, (3, 6), blur=0.0)
    real = rain_module.streak_geometry

    def cramped(field: dict) -> dict:
        geometry = real(field)
        geometry["vertices"] = geometry["vertices"].copy()
        geometry["vertices"][::4, 0, 0] = 0  # on the patch's edge
        return geometry

    monkeypatch.setattr(rain_module, "streak_geometry", cramped)
    geometry = cramped(streaks)
    layer = np.zeros((128, 128), dtype=np.float32)
    left = rain_module._render_streaks_natively(layer, geometry, streaks)
    assert set(left) == set(range(0, 200, 4))
    # And the picture is still the loop's, to the order the two halves add in.
    native, reference = _layers(128, streaks)
    assert np.abs(native - reference).max() < 1e-6


@requires_native
def test_an_empty_field_draws_nothing() -> None:
    streaks = {key: value[:0] for key, value in _streaks(4, 64, 0, (1, 3), 0.0).items()}
    native, reference = _layers(64, streaks)
    assert native.shape == (64, 64) and not native.any() and not reference.any()


# --------------------------------------------------------------- wet road


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("with_puddles", [True, False])
def test_the_wet_road_kernel_is_the_numpy_arithmetic(tier: str, with_puddles: bool) -> None:
    """Fast mode's, where the cube is two multiplications: every operation
    is then float32 arithmetic in numpy's order, and the kernel is exact."""
    _native.set_tier(tier)
    mode.set_fast(True)
    rng = np.random.default_rng(3)
    linear = rng.uniform(0.0, 1.1, (150, 211, 3)).astype(np.float32)  # some above 1
    wetness = np.zeros((150, 211), dtype=np.float32)
    wetness[60:] = rng.uniform(0.0, 1.0, (90, 211)).astype(np.float32)
    puddles = np.clip(rng.uniform(-1.0, 1.0, (150, 211)), 0.0, 1.0).astype(np.float32)
    sky = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    for darkening in rng.uniform(0.4, 0.9, 6):
        arguments = (linear, wetness, float(darkening), 40.0, 250.0)
        extra = {"sky_radiance": sky, "puddles": puddles if with_puddles else None}
        native = rain_module._apply_ground_wetness(*arguments, **extra)
        with _native.disabled():
            reference = rain_module._apply_ground_wetness(*arguments, **extra)
        assert np.abs(reference - linear).max() > 0.05
        assert np.array_equal(native, reference), f"darkening {darkening}"


def _road_with_things_on_it(seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """A frame, its wet ground with dry blocks standing on it (one touching
    the top of the frame, which nothing may be mirrored about), a horizon."""
    rng = np.random.default_rng(seed)
    height, width = int(rng.integers(120, 300)), int(rng.integers(150, 400))
    linear = rng.uniform(0.0, 1.2, (height, width, 3)).astype(np.float32)
    wetness = np.zeros((height, width), dtype=np.float32)
    top = int(rng.integers(0, height - 20))
    wetness[top:] = rng.uniform(0.0, 1.0, (height - top, width)).astype(np.float32)
    for _ in range(int(rng.integers(1, 9))):
        y0, x0 = int(rng.integers(0, height - 5)), int(rng.integers(0, width - 5))
        wetness[y0 : y0 + int(rng.integers(3, height // 2)), x0 : x0 + int(rng.integers(3, 80))] = 0
    wetness[0:40, 10:30] = 0.0
    return linear, wetness, float(rng.uniform(-50.0, 0.7 * height))


@requires_native
@pytest.mark.parametrize("tier", TIERS)
def test_what_each_pixel_mirrors_is_found_the_same_in_one_sweep(tier: str) -> None:
    """In either mode: it is integer bookkeeping down each column."""
    _native.set_tier(tier)
    for seed in range(8):
        _, wetness, horizon = _road_with_things_on_it(seed)
        native = rain_module._mirrored_rows(wetness, horizon)
        with _native.disabled():
            reference = rain_module._mirrored_rows(wetness, horizon)
        assert reference[2].max() > 0.0, "nothing in the fixture is mirrored about its foot"
        for ours, theirs in zip(native, reference, strict=True):
            assert np.array_equal(ours, theirs), f"seed {seed}"


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("with_puddles", [True, False])
def test_the_whole_wet_road_does_not_depend_on_the_kernels(tier: str, with_puddles: bool) -> None:
    """Fast mode, where the mirror image is the library's own arithmetic and
    not OpenCV's warp: a row of the frame at a fractional height is two rows
    weighted, in numpy and in C alike."""
    _native.set_tier(tier)
    mode.set_fast(True)
    for seed in range(6):
        linear, wetness, horizon = _road_with_things_on_it(seed)
        rng = np.random.default_rng(seed + 100)
        puddles = np.clip(rng.uniform(-1.0, 1.0, wetness.shape), 0.0, 1.0).astype(np.float32)
        extra = {
            "sky_radiance": rng.uniform(0.2, 0.9, 3).astype(np.float32),
            "lamp_gain": 3.0,
            "puddles": puddles if with_puddles else None,
        }
        arguments = (linear, wetness, float(rng.uniform(0.4, 0.95)), horizon, 300.0)
        native = rain_module._apply_ground_wetness(*arguments, **extra)
        with _native.disabled():
            reference = rain_module._apply_ground_wetness(*arguments, **extra)
        assert np.abs(reference - linear).max() > 0.05
        assert np.array_equal(native, reference), f"seed {seed}"


# ------------------------------------------------------- denoise filter


def _noisy_blocks(height: int, width: int, seed: int) -> np.ndarray:
    """Flat blocks with edges between them, a gradient and noise: what an
    edge-preserving filter has something to do with."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    blocks = ((xx // 23) * 53 + (yy // 17) * 97) % 200
    image = blocks[..., None] + 0.2 * xx[..., None] + rng.integers(-12, 13, (height, width, 3))
    return np.clip(image, 0, 255).astype(np.uint8)


def _opencv_filter(image: np.ndarray, sigma_s: float, sigma_r: float) -> np.ndarray:
    """OpenCV's own edge-preserving filter, the reference the kernel is held to."""
    return cv2.edgePreservingFilter(
        image, flags=cv2.RECURS_FILTER, sigma_s=sigma_s, sigma_r=sigma_r
    )


@requires_native
@pytest.mark.parametrize("tier", TIERS)
def test_the_denoise_filter_is_opencv_s_to_the_byte(tier: str) -> None:
    """The kernel is OpenCV's arithmetic in OpenCV's order. Whether that
    comes to the same bytes rests on the C library's ``powf``, which the
    transform tries before relying on it: where it has found that it does
    not, there is nothing to hold the kernel to here."""
    if not denoise_module._kernel_matches_opencv():
        pytest.skip("this C library's powf is not the one OpenCV was built against")
    _native.set_tier(tier)
    rng = np.random.default_rng(3)
    shapes = [(1, 1), (1, 40), (40, 1), (2, 3), (37, 53), (96, 128), (5, 201)]
    for height, width in shapes:
        image = _noisy_blocks(height, width, seed=height * width)
        sigma_s, sigma_r = float(rng.uniform(1.0, 200.0)), float(rng.uniform(0.01, 1.0))
        ours = _native.edge_preserving(image, sigma_s, sigma_r, tabled=False)
        assert ours is not None
        assert np.array_equal(ours, _opencv_filter(image, sigma_s, sigma_r)), (height, width)


@requires_native
@pytest.mark.parametrize("tier", TIERS)
def test_the_tabled_denoise_filter_is_within_a_level_of_the_exact_one(tier: str) -> None:
    """Fast mode's form: the power from a table of its 766 possible values."""
    _native.set_tier(tier)
    image = _noisy_blocks(240, 320, seed=1)
    for sigma_s, sigma_r in ((4.0, 0.05), (25.0, 0.2), (120.0, 0.6)):
        exact = _native.edge_preserving(image, sigma_s, sigma_r, tabled=False)
        tabled = _native.edge_preserving(image, sigma_s, sigma_r, tabled=True)
        assert exact is not None and tabled is not None
        assert not np.array_equal(exact, image), "the filter did nothing"
        difference = np.abs(exact.astype(np.int16) - tabled.astype(np.int16))
        assert difference.max() <= 1
        assert (difference > 0).mean() < 1e-3


@requires_native
def test_the_denoise_filter_declines_what_it_was_not_written_for() -> None:
    grey = np.zeros((16, 16), dtype=np.uint8)
    four = np.zeros((16, 16, 4), dtype=np.uint8)
    floats = np.zeros((16, 16, 3), dtype=np.float32)
    for image in (grey, four, floats):
        assert _native.edge_preserving(image, 10.0, 0.2, tabled=False) is None


@requires_native
def test_the_denoise_filter_takes_a_view_that_is_not_contiguous() -> None:
    image = _noisy_blocks(64, 96, seed=2)
    mirrored = image[:, ::-1]
    whole = _native.edge_preserving(np.ascontiguousarray(mirrored), 12.0, 0.2, tabled=True)
    assert np.array_equal(_native.edge_preserving(mirrored, 12.0, 0.2, tabled=True), whole)


def test_a_filter_seen_to_differ_from_opencv_is_not_used_in_exact_mode(monkeypatch) -> None:
    """On a C library whose powf rounds another way, exact mode is OpenCV."""
    image = _noisy_blocks(48, 64, seed=5)
    monkeypatch.setattr(denoise_module, "_kernel_is_opencv", False)
    monkeypatch.setattr(
        _native, "edge_preserving", lambda *a, **k: pytest.fail("the kernel was called")
    )
    assert np.array_equal(
        denoise_module._edge_preserving(image, 9.0, 0.15), _opencv_filter(image, 9.0, 0.15)
    )


@requires_native
def test_fast_mode_denoising_depends_on_the_kernels_by_a_level_or_two() -> None:
    """The one transform whose fast-mode frame is not the same with and
    without the kernels: with them the filter's power comes from a table,
    without them it is OpenCV's. Up to three passes of a level each."""
    transform = TemporalDenoiseArtifacts(p=1.0)
    image = _noisy_blocks(240, 320, seed=7)
    mode.set_fast(True)
    for seed in range(3):
        transform.set_random_seed(seed)
        native = transform(image=image)["image"]
        transform.set_random_seed(seed)
        with _native.disabled():
            reference = transform(image=image)["image"]
        difference = np.abs(native.astype(np.int16) - reference.astype(np.int16))
        assert difference.max() <= 3
        assert (difference > 0).mean() < 5e-3


# ------------------------------------------------------------ lens dirt


def _dirt_inputs(seed: int, pixels: tuple[int, int] = (97, 131)) -> dict[str, np.ndarray]:
    """Random inputs for the dirt, veil and overcast kernels, some past the ranges the clips allow.
    """
    rng = np.random.default_rng(seed)
    f32 = np.float32
    return {
        "linear": rng.uniform(-0.05, 1.3, (*pixels, 3)).astype(f32),
        "spread": rng.uniform(0.0, 2.0, (*pixels, 3)).astype(f32),
        "albedo": rng.uniform(0.2, 0.95, (*pixels, 3)).astype(f32),
        # Past both ends of what the clips let through.
        "mask": rng.uniform(-0.2, 1.6, pixels).astype(f32),
        "luma": rng.uniform(-0.1, 1.2, pixels).astype(f32),
        "around": rng.uniform(0.0, 0.5, 3).astype(f32),
    }


@requires_native
@pytest.mark.parametrize("tier", TIERS)
def test_the_dirt_kernels_are_the_numpy_expressions_to_the_bit(tier: str) -> None:
    """`DirtSoil.apply` either side of its scatter blur, as numpy writes it."""
    _native.set_tier(tier)
    for seed in range(3):
        v = _dirt_inputs(seed)
        boost, strength, absorption = 1.5 + seed, 0.35 + 0.2 * seed, 0.03 * (seed + 1)
        highlight = np.clip((v["luma"] - 0.35) / 0.65, 0.0, 1.0)[..., None]
        emphasised = v["linear"] * (1.0 + boost * highlight)
        assert np.array_equal(_native.dirt_emphasise(v["linear"], v["luma"], boost), emphasised)

        mask = v["mask"][..., None]
        arriving = v["spread"] * (1.0 - 0.4)
        arriving += v["around"]
        scattered = arriving * v["albedo"]
        veil = np.clip(strength * mask, 0.0, 0.95)
        out = v["linear"] * (1.0 - veil) + scattered * veil
        body = v["albedo"] * 0.35
        out = out * (1.0 - absorption * mask) + body * (absorption * mask)
        ours = _native.dirt_composite(
            v["linear"], v["spread"], v["mask"], v["albedo"], v["around"],
            1.0 - 0.4, strength, absorption, 0.35,
        )
        assert out.dtype == np.float32
        assert np.array_equal(ours, out)


@requires_native
def test_the_dirt_kernels_decline_what_they_were_not_written_for() -> None:
    v = _dirt_inputs(0, (16, 16))
    grey = v["linear"][..., 0]
    wide = v["linear"].astype(np.float64)
    assert _native.dirt_emphasise(grey, v["luma"], 2.0) is None
    assert _native.dirt_emphasise(wide, v["luma"], 2.0) is None
    assert _native.dirt_emphasise(v["linear"], v["luma"][:8], 2.0) is None
    arguments = (v["mask"], v["albedo"], v["around"], 0.6, 0.5, 0.05, 0.35)
    assert _native.dirt_composite(wide, v["spread"], *arguments) is None
    assert _native.dirt_composite(v["linear"], v["spread"][:8], *arguments) is None
    assert (
        _native.dirt_composite(
            v["linear"], v["spread"], v["mask"][:8], v["albedo"], v["around"], 0.6, 0.5, 0.05, 0.35
        )
        is None
    )


@requires_native
@pytest.mark.parametrize("fast", [True, False], ids=["fast", "exact"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
def test_dirt_does_not_depend_on_the_kernels_in_either_mode(fast: bool, dtype) -> None:
    transform = DirtSoil(p=1.0)
    image = _noisy_blocks(240, 320, seed=11)
    if dtype is np.float32:
        image = (image / np.float32(255.0)).astype(np.float32)
    mode.set_fast(fast)
    for seed in range(3):
        transform.set_random_seed(seed)
        native = transform(image=image)["image"]
        transform.set_random_seed(seed)
        with _native.disabled():
            reference = transform(image=image)["image"]
        assert not np.array_equal(native, image)
        assert np.array_equal(native, reference)


# ------------------------------------------------- veils and relighting


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("light", ["none", "three", "map"])
def test_the_veil_kernel_is_the_two_numpy_expressions_to_the_bit(tier: str, light: str) -> None:
    """`RandomRain.apply` and `RandomSnow.apply`: a frame through a veil,
    then through a layer of particles lit by one colour or by a field."""
    _native.set_tier(tier)
    for seed in range(3):
        v = _dirt_inputs(seed)
        linear, through, air = v["linear"], v["luma"].clip(0.0, 1.0), v["around"]
        cover = v["mask"].clip(0.0, 1.0)
        t = through[..., None]
        expected = linear * t + air[None, None, :] * (1.0 - t)
        if light == "none":
            ours = _native.veil_blend(linear, through, air)
        else:
            lit = v["albedo"][0, 0].copy() if light == "three" else v["spread"]
            a = cover[..., None]
            expected = expected * (1.0 - a) + lit * a
            ours = _native.veil_blend(linear, through, air, cover, lit)
        assert expected.dtype == np.float32
        assert np.array_equal(ours, expected)


@requires_native
def test_the_veil_kernel_declines_what_it_was_not_written_for() -> None:
    v = _dirt_inputs(0, (16, 16))
    linear, through, air, cover = v["linear"], v["luma"], v["around"], v["mask"]
    assert _native.veil_blend(linear.astype(np.float64), through, air) is None
    assert _native.veil_blend(linear, np.stack([through] * 3, axis=-1), air) is None
    assert _native.veil_blend(linear, through, air.astype(np.float64)) is None
    assert _native.veil_blend(linear, through, air, cover, None) is None
    assert _native.veil_blend(linear, through, air, cover, v["spread"][:8]) is None
    assert _native.veil_blend(linear, through, air, cover, air.astype(np.float64)) is None


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("with_sky", [True, False])
def test_the_overcast_kernel_is_the_numpy_it_stands_in_for(tier: str, with_sky: bool) -> None:
    """Fast mode's relighting with the kernel and without it: the same floats,
    so that the frame does not depend on the kernels being installed."""
    _native.set_tier(tier)
    mode.set_fast(True)
    for seed in range(3):
        v = _dirt_inputs(seed)
        linear = v["linear"].clip(0.0, 1.0)
        sky = v["mask"].clip(0.0, 1.0) if with_sky else None
        native = overcast.apply_overcast(linear, 0.3 + 0.3 * seed, sky)
        with _native.disabled():
            reference = overcast.apply_overcast(linear, 0.3 + 0.3 * seed, sky)
        assert not np.array_equal(native, linear)
        assert native.dtype == np.float32
        assert np.array_equal(native, reference)


# ------------------------------------------------ masks grown by colour


def _grow_inputs(seed: int, shape: tuple[int, int] = (83, 117)) -> dict[str, np.ndarray]:
    """Random inputs for the grown-mask kernel, with a fifth of the pixels nowhere near the mask."""
    rng = np.random.default_rng(seed)
    f32 = np.float32
    weight = rng.uniform(0.0, 1.0, shape).astype(f32)
    weight[rng.random(shape) < 0.2] = 0.0  # no mask nearby at all
    return {
        "linear": rng.uniform(0.0, 1.0, (*shape, 3)).astype(f32),
        "nearby_sum": (rng.uniform(0.0, 1.0, (*shape, 3)) * weight[..., None]).astype(f32),
        "weight": weight,
        "mask": rng.uniform(-0.1, 1.1, shape).astype(f32).clip(0.0, 1.0),
        "grown": rng.uniform(0.0, 1.0, shape).astype(f32),
        "depth": rng.uniform(0.0, 1.0, shape).astype(f32),
    }


def _grow_in_numpy(v: dict[str, np.ndarray], with_depth: bool) -> np.ndarray:
    """What `sky_to_depth_edge` and `fill_to_edges` compute, as they write it."""
    nearby = v["nearby_sum"] / np.maximum(v["weight"], 1e-4)[..., None]
    distance = np.abs(v["linear"] - nearby)
    unlike = np.maximum(np.maximum(distance[..., 0], distance[..., 1]), distance[..., 2])
    alike = np.clip(2.0 - unlike / 0.06, 0.0, 1.0) * (v["weight"] > 0.02)
    if not with_depth:
        return np.maximum(v["mask"], v["grown"] * alike).astype(np.float32)
    plainly_not = np.clip(2.0 - np.abs(v["depth"] - 0.45) / 0.16, 0.0, 1.0)
    return np.maximum(v["mask"] * plainly_not, v["grown"] * alike * plainly_not)


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("with_depth", [True, False], ids=["sky", "ground"])
def test_a_mask_grown_by_colour_is_the_numpy_one_to_the_bit(tier: str, with_depth: bool) -> None:
    _native.set_tier(tier)
    for seed in range(3):
        v = _grow_inputs(seed)
        ours = _native.grow_alike(
            v["linear"], v["nearby_sum"], v["weight"], v["mask"], v["grown"],
            v["depth"] if with_depth else None, 0.45, 0.16 if with_depth else 1.0,
            0.06, 1e-4, 0.02,
        )
        assert np.array_equal(ours, _grow_in_numpy(v, with_depth))


@requires_native
@pytest.mark.parametrize("fast", [True, False], ids=["fast", "exact"])
def test_the_ground_filled_to_its_edges_does_not_depend_on_the_kernels(fast: bool) -> None:
    rng = np.random.default_rng(4)
    linear = rng.uniform(0.0, 1.0, (180, 240, 3)).astype(np.float32)
    ground = np.zeros((180, 240), dtype=np.float32)
    ground[100:, 30:200] = rng.uniform(0.5, 1.0, (80, 170)).astype(np.float32)
    linear[100:, 30:200] = 0.3
    mode.set_fast(fast)
    native = ground_module.fill_to_edges(ground, linear)
    with _native.disabled():
        reference = ground_module.fill_to_edges(ground, linear)
    assert native.max() > 0.5 and np.array_equal(native, reference)


@requires_native
def test_a_mask_grown_by_colour_declines_what_it_was_not_written_for() -> None:
    v = _grow_inputs(0, (16, 16))
    args = (0.45, 0.16, 0.06, 1e-4, 0.02)
    planes = (v["weight"], v["mask"], v["grown"], v["depth"])
    wide = v["linear"].astype(np.float64)
    assert _native.grow_alike(wide, v["nearby_sum"], *planes, *args) is None
    assert _native.grow_alike(v["linear"], v["nearby_sum"][:8], *planes, *args) is None
    cut = v["weight"][:8]
    assert _native.grow_alike(v["linear"], v["nearby_sum"], cut, *planes[1:], *args) is None


# -------------------------------------------------------- water droplets


def _placement(seed: int) -> list[tuple[float, float, float, float]]:
    """Drop positions drawn by the placement routine from one seed."""
    rng = np.random.default_rng(seed)
    return droplets_module._sample_positions(
        rng, 400, 320, 240, (0.8, 14.0), 2.2, 9.0, (1.5, 4.0), 0.15
    )


@requires_native
def test_drops_are_placed_where_the_numpy_loop_places_them() -> None:
    for seed in range(4):
        native = _placement(seed)
        with _native.disabled():
            reference = _placement(seed)
        assert len(native) > 100
        assert native == reference


def _lobes(seed: int, count: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """``count`` drop lobes with outline harmonics, some off the frame, many overlapping."""
    rng = np.random.default_rng(seed)
    lobes = np.column_stack(
        [
            rng.uniform(-20, 340, count), rng.uniform(-20, 260, count),
            rng.uniform(1.0, 25.0, count), rng.uniform(1.0, 1.6, count),
            rng.uniform(0.2, 0.7, count), rng.uniform(0.0, 0.15, count),
            rng.uniform(0.0, 0.3, count),
        ]
    )
    harmonics = np.stack(
        [rng.uniform(0.0, 0.06, (count, 4)), rng.uniform(0.0, 2 * np.pi, (count, 4))], axis=-1
    )
    return lobes, harmonics


@requires_native
@pytest.mark.parametrize("tier", TIERS)
def test_the_water_surface_in_c_is_numpy_s_to_float_rounding(tier: str) -> None:
    """Fast mode's water surface takes the outline's harmonics from a
    recurrence on the angle's sine and cosine, not from `arctan2` and `sin`:
    the same surface to a few units in the last place of a float."""
    _native.set_tier(tier)
    for seed in range(3):
        lobes, harmonics = _lobes(seed)
        mode.set_fast(True)
        level, height_sq = droplets_module._water_surface(lobes, 240, 320, harmonics)
        mode.set_fast(False)
        exact_level, exact_height = droplets_module._water_surface(lobes, 240, 320, harmonics)
        assert exact_level.max() > 1.0, "the fixture's drops do not overlap"
        assert np.abs(level - exact_level).max() < 5e-5
        assert np.abs(height_sq - exact_height).max() < 5e-5 * max(1.0, float(exact_height.max()))


@requires_native
@pytest.mark.parametrize("fast", [True, False], ids=["fast", "exact"])
def test_water_droplets_depend_on_the_kernels_only_in_fast_mode_and_by_a_level(fast: bool) -> None:
    transform = WaterDroplets(p=1.0)
    image = _noisy_blocks(240, 320, seed=13)
    mode.set_fast(fast)
    for seed in range(3):
        transform.set_random_seed(seed)
        native = transform(image=image)["image"]
        transform.set_random_seed(seed)
        with _native.disabled():
            reference = transform(image=image)["image"]
        difference = np.abs(native.astype(np.int16) - reference.astype(np.int16))
        if fast:
            assert difference.max() <= 1 and (difference > 0).mean() < 1e-3
        else:
            assert not difference.any()


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("film", [True, False])
@pytest.mark.parametrize("drops", [True, False])
def test_the_droplet_composite_is_the_numpy_arithmetic(tier: str, film: bool, drops: bool) -> None:
    _native.set_tier(tier)
    v = _dirt_inputs(2)
    base, light = v["linear"], v["spread"]
    fraction = v["mask"].clip(0.0, 1.0)[..., None]
    layers = np.concatenate([v["albedo"] * 0.5, v["luma"].clip(0.0, 1.0)[..., None]], axis=-1)
    expected = base
    if film:
        expected = expected + (light - expected) * fraction
    if drops:
        expected = expected * (1.0 - layers[..., 3:]) + layers[..., :3]
    ours = _native.droplet_composite(
        base, light if film else None, fraction if film else None, layers if drops else None
    )
    assert np.array_equal(ours, expected)


# ------------------------------------------------------------------ discs


def _discs(seed: int, count: int = 4000, size: tuple[int, int] = (200, 300)) -> list[np.ndarray]:
    """Discs of light from under a pixel to forty across, some over the frame's edge."""
    rng = np.random.default_rng(seed)
    height, width = size
    return [
        rng.uniform(-25.0, width + 25.0, count).astype(np.float32),
        rng.uniform(-25.0, height + 25.0, count).astype(np.float32),
        (0.5 + 20.0 * rng.uniform(size=count) ** 2).astype(np.float32),
        rng.uniform(0.1, 40.0, count).astype(np.float32),
    ]


def _splat(discs: list[np.ndarray], size: tuple[int, int] = (200, 300)) -> np.ndarray:
    """The field of light the discs add up to."""
    canvas = np.zeros(size, dtype=np.float32)
    splat_discs(canvas, *discs)
    return canvas


@requires_native
@pytest.mark.parametrize("tier", TIERS)
def test_discs_of_light_are_the_numpy_sum_to_the_bit(tier: str) -> None:
    _native.set_tier(tier)
    discs = _discs(3)
    ours = _splat(discs)
    with _native.disabled():
        theirs = _splat(discs)
    assert ours.max() > 1.0
    assert np.array_equal(ours, theirs)


@requires_native
@pytest.mark.parametrize("fast", [True, False])
def test_a_spider_s_web_does_not_depend_on_the_kernels(fast: bool) -> None:
    mode.set_fast(fast)
    image = _night_street()
    for seed in range(3):
        frames = []
        for kernels in (True, False):
            _native.set_enabled(kernels)
            transform = SpiderWeb(p=1.0, spider_probability=1.0)
            transform.set_random_seed(seed)
            frames.append(transform(image=image)["image"])
        _native.set_enabled(True)
        assert np.array_equal(frames[0], frames[1])


# ------------------------------------------------------------------ tiers


@requires_native
def test_every_tier_gives_the_same_bytes() -> None:
    """One wheel holds every tier, and which runs depends on the CPU. The
    picture must not."""
    linear = _linear((200, 300, 3), seed=4)
    flakes = _flakes(300, (200, 300), 60.0, seed=6)
    streaks = _streaks(2000, 200, 8, (1, 8), blur=2.0)
    frame = _noisy_blocks(200, 300, seed=9)
    dirt = _dirt_inputs(5)
    results = []
    for tier in _native.tiers():
        _native.set_tier(tier)
        assert _native.tier() == tier
        canvas = np.zeros((200, 300), dtype=np.float32)
        render_flake_field(canvas, flakes, frame_centre=(150.0, 100.0), **IRIS)
        results.append(
            (
                photometry.from_linear_unit(linear, 255.0, True, np.uint8),
                canvas,
                rain_module.render_streak_layer(200, 200, streaks),
                _native.edge_preserving(frame, 20.0, 0.2, tabled=False),
                _native.edge_preserving(frame, 20.0, 0.2, tabled=True),
                _native.dirt_emphasise(dirt["linear"], dirt["luma"], 3.0),
                _native.veil_blend(
                    dirt["linear"], dirt["luma"], dirt["around"], dirt["mask"], dirt["spread"]
                ),
                _native.overcast_blend(
                    dirt["linear"], dirt["luma"], dirt["mask"], dirt["luma"], dirt["around"],
                    0.3, 0.5, 0.4,
                ),
                _native.dirt_composite(
                    dirt["linear"], dirt["spread"], dirt["mask"], dirt["albedo"], dirt["around"],
                    0.6, 0.6, 0.08, 0.35,
                ),
                _native.grow_alike(*_grow_inputs(6).values(), 0.45, 0.16, 0.06, 1e-4, 0.02),
                np.asarray(_placement(7)),
                _water_surface_fast(8),
                _native.droplet_composite(
                    dirt["linear"], dirt["spread"], dirt["mask"],
                    np.concatenate([dirt["albedo"], dirt["luma"][..., None]], axis=-1),
                ),
                _splat(_discs(9)),
            )
        )
    for other in results[1:]:
        for first, second in zip(results[0], other, strict=True):
            assert np.array_equal(first, second)


def _water_surface_fast(seed: int) -> np.ndarray:
    """The water surface of :func:`_lobes` drawn by the kernel, level and thickness stacked."""
    lobes, harmonics = _lobes(seed)
    level = np.zeros((240, 320), dtype=np.float32)
    weighted = np.zeros((240, 320), dtype=np.float32)
    assert _native.water_lobes(
        level, weighted, lobes, harmonics, (2, 3, 4, 5), 4.0, droplets_module._OUTLINE_LEVEL
    )
    return np.stack([level, weighted])


@requires_native
def test_an_unknown_tier_is_refused() -> None:
    with pytest.raises(ValueError, match="avx512"):
        _native.set_tier("avx512")


# ------------------------------------------------------------- transforms


def _night_street() -> np.ndarray:
    """A dark noisy street with a brighter road and one saturated lamp."""
    rng = np.random.default_rng(0)
    image = rng.integers(5, 40, (240, 320, 3), dtype=np.uint8)
    image[150:] += 25
    image[60:66, 150:156] = 255
    return image


@requires_native
@pytest.mark.parametrize(
    "transform",
    [NightSnow(p=1.0), NightRain(p=1.0), TemporalDenoiseArtifacts(p=1.0)],
    ids=["snow", "rain", "denoise"],
)
def test_a_transform_s_output_does_not_depend_on_the_kernels(transform) -> None:
    """What a user sees: the same uint8 frame either way."""
    image = _night_street()
    for seed in range(3):
        transform.set_random_seed(seed)
        native = transform(image=image)["image"]
        transform.set_random_seed(seed)
        with _native.disabled():
            reference = transform(image=image)["image"]
        assert not np.array_equal(native, image)
        assert np.array_equal(native, reference)


def test_the_environment_switches_the_kernels_off() -> None:
    code = "from weathermentations import _native; print(_native.enabled())"
    off = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "WEATHERMENTATIONS_NATIVE": "0"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert off.stdout.strip() == "False"


def test_without_the_kernels_every_entry_point_declines() -> None:
    """What an install without the library runs: each kernel says "not done"
    and the caller's numpy code does the work."""
    with _native.disabled():
        assert not _native.enabled()
        linear = _linear((96, 96, 3))
        assert _native.encode_u8(linear, photometry._LINEAR_TO_SRGB_U8_LUT) is None
        assert _native.encode_f32(linear, photometry._LINEAR_TO_SRGB_LUT) is None
        codes = np.zeros((96, 96, 3), dtype=np.uint8)
        assert _native.decode_u8(codes, photometry._U8_TO_LINEAR_LUT) is None
        assert _native.edge_preserving(codes, 10.0, 0.2, tabled=False) is None
        assert _native.dirt_emphasise(linear, linear[..., 0], 2.0) is None
        assert _native.veil_blend(linear, linear[..., 0], linear[0, 0]) is None
        assert np.array_equal(
            photometry.from_linear_unit(linear, 255.0, True, np.uint8),
            _textbook_encode_u8(linear),
        )
