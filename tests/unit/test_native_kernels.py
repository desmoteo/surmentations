"""The compiled kernels against the numpy code they stand in for.

The library is optional, so the numpy path is the definition and the kernels
are held to it: bit for bit where the arithmetic allows (the colour tables,
unblurred sprites and streaks), to float rounding where a blur's sums are
taken in another order. Every instruction-set tier of the build must give the
same bytes as every other, so that a picture does not depend on the CPU.

Without the library, or with it switched off, these tests skip, and the last
section checks the numpy path alone, which is what such an install runs.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from weathermentations import _native, mode
from weathermentations.atmosphere import NightRain, NightSnow
from weathermentations.atmosphere import rain as rain_module
from weathermentations.atmosphere.snow import render_flake_field
from weathermentations.utils import photometry

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


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
def test_nan_is_refused_the_same_way_with_and_without_the_kernels() -> None:
    """A NaN in a frame is a renderer's bug. The table lookup has always
    raised on one, and the kernels must not turn it into a black pixel."""
    linear = _linear((96, 96, 3))
    linear[40, 40, 1] = np.nan
    with pytest.raises(IndexError):
        photometry.from_linear_unit(linear, 255.0, True, np.uint8)
    with _native.disabled(), pytest.raises(IndexError):
        photometry.from_linear_unit(linear, 255.0, True, np.uint8)


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


# ------------------------------------------------------------------ tiers


@requires_native
def test_every_tier_gives_the_same_bytes() -> None:
    """One wheel holds every tier, and which runs depends on the CPU. The
    picture must not."""
    linear = _linear((200, 300, 3), seed=4)
    flakes = _flakes(300, (200, 300), 60.0, seed=6)
    streaks = _streaks(2000, 200, 8, (1, 8), blur=2.0)
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
            )
        )
    for other in results[1:]:
        for first, second in zip(results[0], other, strict=True):
            assert np.array_equal(first, second)


@requires_native
def test_an_unknown_tier_is_refused() -> None:
    with pytest.raises(ValueError, match="avx512"):
        _native.set_tier("avx512")


# ------------------------------------------------------------- transforms


def _night_street() -> np.ndarray:
    rng = np.random.default_rng(0)
    image = rng.integers(5, 40, (240, 320, 3), dtype=np.uint8)
    image[150:] += 25
    image[60:66, 150:156] = 255
    return image


@requires_native
@pytest.mark.parametrize("transform", [NightSnow(p=1.0), NightRain(p=1.0)], ids=["snow", "rain"])
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
        assert np.array_equal(
            photometry.from_linear_unit(linear, 255.0, True, np.uint8),
            _textbook_encode_u8(linear),
        )
