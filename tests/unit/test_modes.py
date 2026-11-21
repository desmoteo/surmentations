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

import surmentations
from surmentations import _native, mode
from surmentations.atmosphere import NightSnow, rain
from surmentations.atmosphere.night_rain import _beam_profile
from surmentations.atmosphere.snow import render_flake_field
from surmentations.camera import (
    NightIrMode,
    SensorNoise,
    TemporalDenoiseArtifacts,
    WdrToneMapping,
)
from surmentations.depth.normals import upward_facing
from surmentations.lens import DirtSoil, HousingOccluders, SunFlare
from surmentations.lens import dirt as dirt_module
from surmentations.lens import sun_flare as flare_module
from surmentations.lens.sun_flare import diffraction_starburst, polygon_distance
from surmentations.utils import color, overcast
from surmentations.utils import sky as sky_module
from surmentations.utils.channels import channel_max, frame_mean, percentile, weighted_colour
from surmentations.utils.filters import wide_blur, wide_dilate
from surmentations.utils.grids import centered_grid, on_coarse_grid, pixel_grid
from surmentations.utils.ground import ground_noise


@pytest.fixture(autouse=True)
def _restore_mode():
    """Restore the fast or exact mode a test changed."""
    before = mode.fast()
    yield
    mode.set_fast(before)


# ------------------------------------------------------------- the switch


def test_the_mode_is_set_and_restored() -> None:
    surmentations.set_fast(True)
    assert surmentations.fast()
    with surmentations.exact():
        assert not surmentations.fast()
        with surmentations.exact():
            assert not surmentations.fast()
        assert not surmentations.fast()
    assert surmentations.fast()

    surmentations.set_fast(False)
    with surmentations.exact():
        pass
    assert not surmentations.fast(), "leaving the block must not switch fast mode on"


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
    env = {key: item for key, item in os.environ.items() if key != "SURMENTATIONS_FAST"}
    if value is not None:
        env["SURMENTATIONS_FAST"] = value
    code = "import surmentations; print(surmentations.fast())"
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
    """A linear-light frame of noise with a darker top third."""
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
    1200 pixels. Fast mode reads the surface normals at 384 and enlarges
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
    # A frame no larger than the detail a depth map holds is read as it
    # is, in either mode.
    small = np.ascontiguousarray(depth[::3, ::3])
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
    """Fast mode reads where the road is wet on a 384-pixel copy and
    enlarges the map. Same road, same horizon, edges a pixel or two off."""
    depth, scene = _street()
    read = {}
    for flag in (True, False):
        mode.set_fast(flag)
        read[flag] = rain._wet_ground(depth, scene, 60.0, 10.0, 6.0)
    (fast_wet, fast_horizon, fast_end), (exact_wet, exact_horizon, exact_end) = (
        read[True],
        read[False],
    )
    assert fast_wet.shape == exact_wet.shape == depth.shape and fast_wet.dtype == np.float32
    assert 0.2 < (exact_wet > 0.5).mean() < 0.9
    assert ((fast_wet > 0.5) == (exact_wet > 0.5)).mean() > 0.97
    assert abs(fast_horizon - exact_horizon) < 0.01 * depth.shape[0]
    # The row the road is mirrored about is found on the reduced copy too.
    assert abs(fast_end - exact_end) < 0.01 * depth.shape[0]
    assert fast_horizon <= fast_end and exact_horizon <= exact_end
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
    mode takes a remainder, and keeps it float32 where the reference widened
    it to float64 by adding a random offset that numpy hands back as one.
    Thousandths of a tile pixel apart at most, out to hundreds of metres."""
    for far, scale in ((40.0, 1.2), (400.0, 0.3)):
        depth = np.repeat(np.linspace(far, 2.0, 200, dtype=np.float32)[:, None], 300, axis=1)
        fields = {}
        for flag in (True, False):
            mode.set_fast(flag)
            fields[flag] = ground_noise(depth, 400.0, 20.0, scale, np.random.default_rng(5))
        assert fields[True].dtype == fields[False].dtype == np.float32
        assert fields[False].std() > 0.05
        assert np.abs(fields[True] - fields[False]).max() < 1e-3
        assert np.abs(fields[True] - fields[False]).mean() < 1e-4


def _beam_as_first_written(height: int, width: int, falloff: float) -> np.ndarray:
    """The night illuminator's beam as it was written before it took two lines: the reference."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = np.hypot(
        (xx - width / 2.0) / max(width / 2.0, 1.0),
        (yy - height / 2.0) / max(height / 2.0, 1.0),
    )
    return np.exp(-falloff * radius**2).astype(np.float32)


@pytest.mark.parametrize("size", [(240, 320), (97, 131), (1, 1), (64, 1)])
def test_the_lamp_s_beam_is_the_same_field_from_two_lines(size: tuple[int, int]) -> None:
    """A Gaussian of the distance from the centre is a profile across the
    frame times one down it. Exact mode still takes it from the distance,
    on coordinates that broadcast, and gives the first version's floats;
    fast mode takes the product and is a float's last bit away."""
    for falloff in (0.4, 1.7, 3.0):
        reference = _beam_as_first_written(*size, falloff)
        mode.set_fast(False)
        assert np.array_equal(_beam_profile(*size, falloff), reference)
        mode.set_fast(True)
        product = _beam_profile(*size, falloff)
        assert product.dtype == np.float32 and product.shape == size
        assert np.abs(product - reference).max() < 1e-6


def test_fast_mode_keeps_aperture_fields_float32_and_within_rounding() -> None:
    dy, dx = centered_grid(129, scale=40.0)
    mode.set_fast(False)
    reference = polygon_distance(dy, dx, 7, 0.3)
    mode.set_fast(True)
    fast = polygon_distance(dy, dx, 7, 0.3)
    assert reference.dtype == np.float64 and fast.dtype == np.float32
    assert np.abs(fast - reference).max() < 1e-6


def _flakes(count: int, size: tuple[int, int], seed: int) -> dict[str, np.ndarray]:
    """A field of ``count`` flakes, some off the frame, as the flake renderer takes them."""
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
    """A dark noisy frame with a bright top band and one saturated square."""
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
    [HousingOccluders(p=1.0), TemporalDenoiseArtifacts(p=1.0)],
    ids=["occluders", "denoise"],
)
def test_a_reduced_blur_moves_a_frame_by_a_few_levels_at_most(transform) -> None:
    """Defocused insects and smeared chroma: fields that are soft by nature,
    blurred on a smaller copy in fast mode."""
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


# ------------------------------------------------------------- lens dirt


def test_the_deposit_field_is_built_a_quarter_of_the_size_where_its_blur_allows() -> None:
    mode.set_fast(True)
    assert dirt_module._deposit_reduction(1024, 1024) == 4
    assert dirt_module._deposit_reduction(480, 480) == 4
    assert dirt_module._deposit_reduction(256, 256) == 2
    assert dirt_module._deposit_reduction(96, 96) == 1
    mode.set_fast(False)
    assert dirt_module._deposit_reduction(1024, 1024) == 1


def test_dirt_built_small_is_the_same_deposit_in_the_same_place() -> None:
    """Fast mode builds the deposit field a quarter of the size and enlarges
    it. The draws are the same, so it is the same grime: as much of it, in
    the same places to within the two pixels a patch's corner is rounded by,
    and a frame a tenth of a level away on average."""
    image = _large_scene()
    transform = DirtSoil(p=1.0)
    for seed in range(4):
        fields, frames = {}, {}
        for flag in (True, False):
            mode.set_fast(flag)
            transform.set_random_seed(seed)
            params = transform.get_params_dependent_on_data({"shape": image.shape}, {})
            fields[flag] = params["dirt_mask"], params["deposit_albedo"]
            frames[flag] = transform.apply(image, **params).astype(np.int16)
        for small, full in zip(fields[True], fields[False], strict=True):
            assert small.shape == full.shape and small.dtype == np.float32
        mask_small, mask_full = fields[True][0], fields[False][0]
        assert mask_full.mean() > 0.05, "the fixture drew next to nothing"
        assert abs(mask_small.mean() - mask_full.mean()) < 0.01 * mask_full.mean()
        assert np.corrcoef(mask_small.ravel(), mask_full.ravel())[0, 1] > 0.995
        assert np.abs(mask_small - mask_full).mean() < 0.01
        difference = np.abs(frames[True] - frames[False])
        assert difference.mean() < 0.3
        assert difference.max() <= 12


def test_a_frame_too_small_to_reduce_gets_the_full_size_field_in_fast_mode() -> None:
    image = _scene()[:90, :90]
    transform = DirtSoil(p=1.0)
    frames = {}
    for flag in (True, False):
        mode.set_fast(flag)
        transform.set_random_seed(2)
        frames[flag] = transform(image=image)["image"].astype(np.int16)
    assert np.abs(frames[True] - frames[False]).max() <= 2


# ------------------------------------------------------------- sun flare


@pytest.mark.parametrize("angle", [0.0, 0.3, np.pi / 4, 1.2, np.pi / 2, 2.0, 3.0, -0.7])
@pytest.mark.parametrize("through", [(150.3, 99.6), (-40.0, 20.0), (500.0, 400.0)])
def test_every_pixel_near_a_line_is_among_those_tried(angle: float, through) -> None:
    """Scratch streaks are evaluated on the pixels around their line and not
    over the frame, so the list of those must miss none that the test over
    the whole frame finds."""
    height, width, reach = 180, 260, 6.4
    sin_a, cos_a = float(np.sin(angle)), float(np.cos(angle))
    rows, cols = flare_module._near_line(height, width, through, sin_a, cos_a, reach)
    assert rows.min(initial=0) >= 0 and rows.max(initial=0) < height
    assert cols.min(initial=0) >= 0 and cols.max(initial=0) < width
    tried = np.zeros((height, width), dtype=bool)
    tried[rows, cols] = True
    assert tried.sum() == len(rows), "a pixel is listed twice, and would be added to twice"
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    across = -(xx - through[0]) * sin_a + (yy - through[1]) * cos_a
    assert not (np.abs(across) <= reach)[~tried].any()
    assert tried.sum() < 0.25 * height * width


def test_scratch_streaks_are_the_same_light_either_way() -> None:
    colour = np.array([90.0, 85.0, 75.0])
    scratches = [(0.2, 120.0, 0.8), (1.9, 60.0, 0.5), (np.pi / 2, 200.0, 1.0), (0.0, 40.0, 0.3)]
    canvases = {}
    for flag in (True, False):
        mode.set_fast(flag)
        canvases[flag] = np.zeros((200, 300, 3), dtype=np.float32)
        tint = colour.astype(np.float32) if flag else colour
        flare_module.add_scratch_streaks(canvases[flag], tint, (140.5, 90.25), scratches)
    assert canvases[False].max() > 50.0
    assert np.abs(canvases[True] - canvases[False]).max() < 1e-3


def test_the_broad_light_drawn_small_is_within_a_quarter_of_a_level() -> None:
    """Veiling glare, halo and infrared glow, evaluated on a grid four times
    coarser and enlarged, against the same three at every pixel. The most
    it is out by is at the glare's peak, where the frame is white anyway."""
    veil = np.array([140.0, 130.0, 110.0], dtype=np.float32)
    halo = np.array([30.0, 29.0, 26.0], dtype=np.float32)
    infrared = np.array([60.0, 20.0, 35.0], dtype=np.float32)
    for (height, width), source in (((768, 1024), (700.5, 200.25)), ((600, 600), (-30.0, 650.0))):
        half_diag = float(np.hypot(width, height)) / 2.0
        core = 0.12 * half_diag
        assert core / flare_module._BROAD_CORE_PX >= flare_module._BROAD_REDUCTION
        small = flare_module._broad_light(
            height, width, source, half_diag, 0.3, veil, halo, infrared
        )
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        dist = np.hypot(xx - source[0], yy - source[1])
        full = (1.0 / (1.0 + (dist / core) ** 2))[..., None] * veil
        full += np.exp(-((dist / half_diag - 0.3) ** 2) / (2 * 0.135**2))[..., None] * halo
        full += (1.0 / (1.0 + (dist / (core * 2.2)) ** 2))[..., None] * infrared
        assert small.shape == full.shape and small.dtype == np.float32
        assert np.abs(small - full).max() < 0.25
        assert np.abs(small - full).mean() < 0.01


def test_a_small_frame_s_broad_light_is_drawn_at_full_size() -> None:
    veil = np.array([140.0, 130.0, 110.0], dtype=np.float32)
    small = flare_module._broad_light(60, 80, (20.0, 30.0), 50.0, 0.3, veil, veil * 0.2, None)
    yy, xx = np.mgrid[0:60, 0:80].astype(np.float64)
    dist = np.hypot(xx - 20.0, yy - 30.0)
    full = (1.0 / (1.0 + (dist / 6.0) ** 2))[..., None] * veil
    full += np.exp(-((dist / 50.0 - 0.3) ** 2) / (2 * 0.135**2))[..., None] * (veil * 0.2)
    assert np.abs(small - full).max() < 1e-3


def test_a_glow_drawn_in_its_own_square_is_the_glow_over_the_frame() -> None:
    colour = np.array([56.0, 54.0, 47.0], dtype=np.float32)
    for centre in ((120.3, 80.7), (-10.0, 5.0), (299.0, 199.0), (900.0, 900.0)):
        canvas = np.zeros((200, 300, 3), dtype=np.float32)
        flare_module._add_compact_glow(canvas, colour, centre, 14.0)
        yy, xx = np.mgrid[0:200, 0:300].astype(np.float64)
        square = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2
        full = np.exp(-square / 14.0**2)[..., None] * colour
        assert np.abs(canvas - full).max() < 1e-3


# ---------------------------------------------------- rain's relighting


def _daylight(height: int = 300, width: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A linear-light frame with a bright sky over darker textured ground,
    and the sky's mask with a soft lower edge."""
    rng = np.random.default_rng(seed)
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    sky = np.clip((0.42 - rows) / 0.04, 0.0, 1.0) * np.ones((1, width), dtype=np.float32)
    ground = rng.uniform(0.02, 0.35, (height, width, 3)).astype(np.float32)
    above = np.array([0.45, 0.6, 0.85], dtype=np.float32) + rng.uniform(
        -0.03, 0.03, (height, width, 3)
    ).astype(np.float32)
    linear = ground * (1.0 - sky[..., None]) + above * sky[..., None]
    return linear.astype(np.float32), sky.astype(np.float32)


def test_the_mean_colour_of_a_frame_is_the_same_mean() -> None:
    linear, _sky = _daylight()
    mode.set_fast(False)
    exact = frame_mean(linear)
    assert np.array_equal(exact, linear.reshape(-1, 3).mean(axis=0))
    mode.set_fast(True)
    assert frame_mean(linear).dtype == np.float32
    # Five digits: the reference adds float32 values in sequence, and it is
    # the one that is out. Against a sum in double the fast form is exact.
    assert np.abs(frame_mean(linear) - exact).max() < 2e-5
    in_double = linear.reshape(-1, 3).astype(np.float64).mean(axis=0)
    assert np.abs(frame_mean(linear) - in_double).max() < 1e-7


@pytest.mark.parametrize("with_sky", [True, False])
@pytest.mark.parametrize("amount", [0.2, 0.6, 1.0])
def test_overcast_from_a_table_is_within_a_level(with_sky: bool, amount: float) -> None:
    """Fast mode takes the contrast gain, a power of luminance, from a table
    of 65536 of its values and goes over the frame once."""
    linear, sky = _daylight()
    frames = {}
    for flag in (True, False):
        mode.set_fast(flag)
        frames[flag] = overcast.apply_overcast(linear, amount, sky if with_sky else None)
    assert np.abs(frames[False] - linear).max() > 0.02, "the fixture was not relit"
    assert frames[True].dtype == frames[False].dtype == np.float32
    difference = np.abs(frames[True] - frames[False])
    assert difference.max() < 1.0 / 255.0
    assert difference.mean() < 1e-4


@pytest.mark.parametrize("percentile", [90.0, 97.0, 99.5])
def test_the_brightest_pixels_mean_by_counting_is_the_same_colour(percentile: float) -> None:
    """Fast mode finds the percentile from a histogram and averages above
    it in place, where the reference sorts for it and copies the pixels out."""
    for seed in range(3):
        linear, _sky = _daylight(seed=seed)
        colours = {}
        for flag in (True, False):
            mode.set_fast(flag)
            colours[flag] = color.sky_radiance_linear(linear, percentile)
        assert colours[True].dtype == np.float32 and colours[True].shape == (3,)
        assert np.abs(colours[True] - colours[False]).max() < 2e-3
    flat = np.full((40, 50, 3), 0.25, dtype=np.float32)
    mode.set_fast(True)
    assert np.allclose(color.sky_radiance_linear(flat, 97.0), 0.25)
    assert np.allclose(color.sky_radiance_linear(np.zeros_like(flat), 97.0), 0.0)


def _sky_edge_over_the_whole_frame(sky, relative_depth, linear_rgb=None):
    """`sky_to_depth_edge` as it was before it kept to the rows the mask reaches."""
    core = sky > 0.9
    if not core.any():
        return sky
    level = float(np.median(relative_depth[core]))
    near_level = np.clip(
        2.0 - np.abs(relative_depth - level) / sky_module._SKY_DEPTH_TOLERANCE, 0.0, 1.0
    )
    tolerance = sky_module._SKY_KEPT_TOLERANCE * sky_module._SKY_DEPTH_TOLERANCE
    plainly_not = np.clip(2.0 - np.abs(relative_depth - level) / tolerance, 0.0, 1.0)
    kept = sky * plainly_not
    reach = max(3, round(sky_module._SKY_EDGE_REACH * float(np.hypot(*sky.shape))) | 1)
    grown = wide_dilate(sky, reach)
    if linear_rgb is not None:
        sigma = float(reach)
        weight = wide_blur(sky, sigma)
        nearby = wide_blur(linear_rgb * sky[..., None], sigma)
        nearby = nearby / np.maximum(weight, 1e-4)[..., None]
        unlike = channel_max(np.abs(linear_rgb - nearby))
        alike = np.clip(2.0 - unlike / sky_module._SKY_COLOUR_TOLERANCE, 0.0, 1.0)
        alike = alike * (weight > 0.02)
        return np.maximum(kept, grown * alike * plainly_not).astype(np.float32)
    return np.maximum(kept, grown * near_level).astype(np.float32)


@pytest.mark.parametrize("fast", [True, False], ids=["fast", "exact"])
@pytest.mark.parametrize("with_colour", [True, False])
def test_the_sky_s_edge_redrawn_on_its_own_rows_is_the_same_mask(
    fast: bool, with_colour: bool
) -> None:
    """Below the lowest row the grown mask reaches there is nothing to
    redraw, and the redraw is done on the rows above it. The same bytes as
    going over the frame, in either mode."""
    mode.set_fast(fast)
    for seed, (height, width) in enumerate(((300, 400), (480, 640), (97, 131))):
        linear, sky = _daylight(height, width, seed)
        rng = np.random.default_rng(seed)
        depth = (0.9 * sky + 0.3 * (1.0 - sky) * np.linspace(1.0, 0.2, height)[:, None]).astype(
            np.float32
        )
        depth += rng.uniform(-0.01, 0.01, depth.shape).astype(np.float32)
        colour = linear if with_colour else None
        ours = sky_module.sky_to_depth_edge(sky, depth, colour)
        reference = _sky_edge_over_the_whole_frame(sky, depth, colour)
        assert ours.dtype == np.float32 and ours.shape == sky.shape
        assert ours[: height // 3].max() > 0.9 and not ours[-height // 4 :].any()
        assert np.array_equal(ours, reference)
    everywhere = np.ones((60, 80), dtype=np.float32)
    even = np.full((60, 80), 0.9, dtype=np.float32)
    assert np.array_equal(
        sky_module.sky_to_depth_edge(everywhere, even),
        _sky_edge_over_the_whole_frame(everywhere, even),
    )


# ------------------------------------------- the seventh round's fast forms


def _photo_like(height: int = 360, width: int = 480, seed: int = 0) -> np.ndarray:
    """Strong edges, a gradient and texture: what a tone mapper's halos and a
    shadow's penumbra show up on."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    blocks = ((xx // 37) * 71 + (yy // 29) * 113) % 230
    image = 12 + blocks[..., None] * 0.9 + rng.integers(0, 20, (height, width, 3))
    image[: height // 3] = np.clip(image[: height // 3] + 60, 0, 255)
    return np.clip(image, 0, 255).astype(np.uint8)


def _both_modes(transform, image: np.ndarray, seeds: int = 3) -> list[np.ndarray]:
    """The difference between fast and exact mode's frames, for each of ``seeds`` seeds.

    Fails if the transform leaves the frame unchanged in exact mode.
    """
    differences = []
    for seed in range(seeds):
        frames = {}
        for flag in (True, False):
            mode.set_fast(flag)
            transform.set_random_seed(seed)
            frames[flag] = transform(image=image)["image"].astype(np.int16)
        assert not np.array_equal(frames[False], image), "the transform did nothing"
        differences.append(np.abs(frames[True] - frames[False]))
    return differences


def test_tone_mapping_s_coarser_base_moves_edges_only() -> None:
    """Fast mode filters the base layer at a radius of 8 for 12 on a smaller
    copy. Where the base meets a strong edge the halo moves by a pixel or
    two; elsewhere the frame is the same to a level or so."""
    for difference in _both_modes(WdrToneMapping(p=1.0), _photo_like()):
        assert difference.mean() < 2.5
        assert np.percentile(difference, 99.0) <= 25
        # The large differences are lines along edges, not areas.
        assert (difference > 10).mean() < 0.02


def test_a_shadow_s_penumbra_blurred_small_is_within_a_level_or_two() -> None:
    """CastShadows blurs its shadows on a reduced copy in fast mode once the
    penumbra is wide. With no depth to give a surface, the shadows here are
    applied through a ground plane, as the depth-sharing tests do. On a
    frame small enough that the surface itself is read at full size in
    both modes: it is the blur that is measured here, and the surface read
    at the depth map's own detail has a test of its own."""
    from surmentations.depth import SharedDepthProvider
    from surmentations.depth.estimator import DepthEstimator
    from surmentations.scene import CastShadows

    class GroundPlane(DepthEstimator):
        """A level road under the lower half of the frame."""
        def estimate(self, image: np.ndarray) -> np.ndarray:
            height, width = image.shape[:2]
            rows = np.arange(height, dtype=np.float32).reshape(height, 1)
            depth = np.clip(450.0 / np.maximum(rows - height / 2.0, 1e-3), 0.0, 50.0)
            return np.broadcast_to(depth / 50.0, (height, width)).astype(np.float32)

    transform = CastShadows(
        depth_provider=SharedDepthProvider(GroundPlane()), p=1.0, penumbra_range=(0.012, 0.02)
    )
    for difference in _both_modes(transform, _photo_like(360, 480)):
        assert difference.max() <= 4
        assert difference.mean() < 0.1


def test_the_night_illuminator_by_the_short_way_is_within_a_few_levels() -> None:
    """Fast mode takes the beam on a coarse grid and the exposure level from
    a histogram."""
    from surmentations.depth import SharedDepthProvider
    from surmentations.depth.estimator import DepthEstimator

    class Receding(DepthEstimator):
        """Depth rising from 0.05 at the top row to 1 at the bottom."""
        def estimate(self, image: np.ndarray) -> np.ndarray:
            height, width = image.shape[:2]
            return np.broadcast_to(
                np.linspace(0.05, 1.0, height, dtype=np.float32)[:, None], (height, width)
            ).copy()

    transform = NightIrMode(depth_provider=SharedDepthProvider(Receding()), p=1.0)
    for difference in _both_modes(transform, _photo_like()):
        assert difference.max() <= 4
        assert difference.mean() < 0.1


@pytest.mark.parametrize("q", [5.0, 50.0, 80.0, 99.5])
def test_a_percentile_by_counting_is_the_sorted_one_to_a_bin(q: float) -> None:
    rng = np.random.default_rng(int(q))
    for values in (
        rng.gamma(2.0, 0.1, (300, 400)).astype(np.float32),
        rng.uniform(-3.0, 5.0, (256, 256)).astype(np.float32),
        np.where(rng.random((200, 200)) < 0.7, 0.0, rng.random((200, 200))).astype(np.float32),
    ):
        mode.set_fast(False)
        exact = percentile(values, q)
        assert exact == float(np.percentile(values, q))
        mode.set_fast(True)
        bin_width = float(values.max() - values.min()) / 16384
        assert abs(percentile(values, q) - exact) <= 2.0 * bin_width
    mode.set_fast(True)
    assert percentile(np.full((10, 10), 0.25, np.float32), 80.0) == 0.25


def test_a_smooth_field_from_a_coarse_grid_is_the_field_to_a_fraction_of_a_level() -> None:
    def beam(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        radius = np.sqrt(((xx - 320.0) / 320.0) ** 2 + ((yy - 240.0) / 240.0) ** 2)
        return np.clip(np.cos(np.clip(radius, 0.0, 1.0) * (np.pi / 3.0)), 0.0, 1.0) ** 2.5

    for height, width in ((480, 640), (481, 641), (37, 53)):
        full = beam(*pixel_grid(height, width)).astype(np.float32)
        coarse = on_coarse_grid(height, width, beam)
        assert coarse.shape == (height, width) and coarse.dtype == np.float32
        # A third of a percent of the peak, at the kink where the profile is
        # clipped near the corners; a twentieth of that on average.
        assert np.abs(coarse - full).max() < 4e-3
        assert np.abs(coarse - full).mean() < 2e-4
    assert np.array_equal(on_coarse_grid(20, 30, beam, reduction=1), beam(*pixel_grid(20, 30)))
