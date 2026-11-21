"""The kernels of what reads the scene, against the numpy code they stand in for.

The lamps' light on the air, the spray's puffs, the surface normals, the
two nights' compositing, the night fog's, the lying snow and the ground's
noise. As in ``test_native_kernels.py`` the numpy path is the definition:
each kernel is held to it bit for bit where its arithmetic is numpy's, and
to a stated bound in the two places it is not (the spray's Gaussian, which
is a polynomial, and the two weighted sums of three channels that numpy
leaves to its matrix product). Every tier must give the same bytes.

Without the library, or with it switched off, these skip.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from surmentations import _native, mode
from surmentations.atmosphere.night_fog import NightFog
from surmentations.camera import NightColorMode
from surmentations.camera import night_ir as night_ir_module
from surmentations.camera.night_ir import NightIrMode
from surmentations.camera.night_lighting import SceneLighting
from surmentations.depth import normals as normals_module
from surmentations.utils import ground as ground_module
from surmentations.utils import road as road_module
from surmentations.utils.night_light import (
    BeamSource,
    fit_ground,
    headlamp_light,
    scene_points,
    street_air,
    street_light,
)

requires_native = pytest.mark.skipif(
    not _native.enabled(),
    reason="the native kernels are not built (python hatch_build.py) or are switched off",
)

TIERS = _native.tiers() or ("none",)
# Not a multiple of the kernels' block of 512, nor of a vector.
HEIGHT, WIDTH = 61, 147


@pytest.fixture(autouse=True)
def _restored_state():
    """A test that switches tier or mode, or disables the kernels, must not leak it."""
    was_enabled, tier, was_fast = _native.enabled(), _native.tier(), mode.fast()
    yield
    mode.set_fast(was_fast)
    if tier is not None:
        _native.set_tier(tier)
    _native.set_enabled(was_enabled)


def _plane(seed: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """A float32 plane of the tests' size, uniform between two levels."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, (HEIGHT, WIDTH)).astype(np.float32)


def _field(seed: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """A float32 three-channel field of the tests' size."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, (HEIGHT, WIDTH, 3)).astype(np.float32)


def _depth(seed: int = 0) -> np.ndarray:
    """A ground that recedes up the frame with things standing on it: steps in depth."""
    rng = np.random.default_rng(seed)
    rows = np.linspace(90.0, 5.0, HEIGHT, dtype=np.float32)[:, None]
    depth = np.broadcast_to(rows, (HEIGHT, WIDTH)).copy()
    for _ in range(6):
        top, left = int(rng.integers(0, HEIGHT - 12)), int(rng.integers(0, WIDTH - 20))
        depth[top : top + 12, left : left + 20] = float(rng.uniform(4.0, 60.0))
    return (depth + rng.uniform(0.0, 0.05, depth.shape)).astype(np.float32)


# ---------------------------------------------------------------- lit air


@requires_native
@pytest.mark.parametrize("tier", TIERS)
def test_the_lit_air_kernel_is_the_numpy_sum_to_the_bit(tier: str) -> None:
    """Every lamp's light along every line of sight: square roots and no
    angle, the same float32 arithmetic in the same order."""
    _native.set_tier(tier)
    points, _step = scene_points(_depth(), 140.0)
    up = np.array([0.0, -0.94, -0.34], dtype=np.float32)
    up /= np.linalg.norm(up)
    rng = np.random.default_rng(1)
    lamps = np.stack(
        [
            rng.uniform(-30, 30, 7),
            rng.uniform(-9, -5, 7),
            rng.uniform(8, 110, 7),
            rng.uniform(6, 12, 7),
        ],
        axis=1,
    ).astype(np.float32)
    # One on a line of sight, nearer than the nearest a line is taken to pass.
    lamps[0, :3] = 0.5 * points[30, 70]
    air, reach = street_air(points, up, lamps)
    with _native.disabled():
        air_numpy, reach_numpy = street_air(points, up, lamps)
    assert float(air.max()) > 0.0 and np.isfinite(air).all()
    assert np.array_equal(air, air_numpy)
    assert np.array_equal(reach, reach_numpy)


def test_the_lit_air_is_the_sum_over_angle_it_is_written_as() -> None:
    """Nine points by Simpson's rule on halved directions, against the same
    sum taken at four hundred points with the angles themselves."""
    points, _step = scene_points(_depth(), 140.0)
    up = np.array([0.0, -0.94, -0.34])
    up /= np.linalg.norm(up)
    lamp = np.array([[3.0, -7.0, 40.0, 9.0]], dtype=np.float32)
    with _native.disabled():
        air, _reach = street_air(points, up.astype(np.float32), lamp)
    far = np.sqrt((points.astype(np.float64) ** 2).sum(axis=2))
    ray = points / far[..., None]
    head = lamp[0, :3].astype(np.float64)
    along, rising = ray @ head, ray @ up
    passes = np.sqrt(np.maximum(head @ head - along**2, 0.6**2))
    first, last = np.arctan(-along / passes), np.arctan((far - along) / passes)
    over = (head @ up - along * rising) / passes
    nodes, weights = np.polynomial.legendre.leggauss(400)
    summed = np.zeros(far.shape)
    for node, weight in zip(nodes, weights, strict=True):
        phi = 0.5 * (last + first) + 0.5 * (last - first) * node
        below = np.clip(np.cos(phi) * over - np.sin(phi) * rising, 0.0, 1.0)
        forward = 0.75 / (1.25 + np.sin(phi)) ** 1.5
        summed += weight * forward * (0.03 + 0.97 * below**2)
    fine = 81.0 * 0.5 * (last - first) * summed / passes
    assert float(np.abs(air - fine).mean()) < 0.01 * float(fine.mean())
    assert float(np.abs(air - fine).max()) < 0.05 * float(fine.max())


@requires_native
def test_the_lit_air_kernel_declines_what_it_was_not_written_for() -> None:
    plane = _plane(0)
    ray = (plane, plane, plane)
    lamps, up = np.zeros((2, 4), np.float32), np.zeros(3, np.float32)
    air, reach = np.zeros_like(plane), np.zeros_like(plane)
    assert _native.street_air(air, reach, ray, plane, plane, lamps, up, 0.36, 0.03, 0.5)
    wide = plane.astype(np.float64)
    assert not _native.street_air(air, reach, ray, wide, plane, lamps, up, 0.36, 0.03, 0.5)
    assert not _native.street_air(air, reach, ray, plane[:8], plane, lamps, up, 0.36, 0.03, 0.5)
    assert not _native.street_air(air, reach, ray, plane, plane, lamps[:, :3], up, 0.36, 0.03, 0.5)
    assert not _native.street_air(
        air, reach, ray, plane, plane, lamps, up.astype(np.float64), 0.36, 0.03, 0.5
    )


# ------------------------------------------------------------------ spray


def _puffs(count: int, seed: int, gated: bool) -> np.ndarray:
    """A table of puffs: round and long, thin and dense, some over the frame's edge."""
    rng = np.random.default_rng(seed)
    table = np.zeros((count, 15))
    table[:, 0] = rng.uniform(-10, WIDTH + 10, count)
    table[:, 1] = rng.uniform(-10, HEIGHT + 10, count)
    turn = rng.uniform(0.0, 2.0 * np.pi, count)
    table[:, 2], table[:, 3] = np.cos(turn), np.sin(turn)
    table[:, 4] = rng.uniform(0.4, 25.0, count)
    table[:, 5] = rng.uniform(0.4, 9.0, count)
    table[:, 6] = rng.uniform(0.01, 1.0, count)
    table[:, 7] = rng.uniform(0.0, HEIGHT, count)
    table[:, 8] = np.where(rng.uniform(size=count) < 0.7, rng.uniform(5.0, 80.0, count), 0.0)
    table[:, 9] = rng.uniform(1.0, 3.0, count)
    if gated:
        table[:, 10], table[:, 11] = table[:, 0] - 4.0, table[:, 1] + 3.0
        table[:, 12], table[:, 13] = table[:, 2], table[:, 3]
        table[:, 14] = np.where(rng.uniform(size=count) < 0.5, rng.uniform(1.0, 9.0, count), 0.0)
    return table


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("with_depth", [True, False])
@pytest.mark.parametrize("gated", [True, False])
def test_the_spray_kernel_draws_the_numpy_puffs(tier: str, with_depth: bool, gated: bool) -> None:
    """A table of puffs in one call against one numpy puff at a time. The
    kernel's Gaussian is a polynomial raised to a power and its divisions
    are by reciprocals: within a hundred-thousandth of a puff's density,
    where spray is seen at a hundredth."""
    _native.set_tier(tier)
    puffs = _puffs(150, 3, gated)
    stands, hidden = _plane(1, 0.0, HEIGHT), _plane(2)
    depth = _depth(4) if with_depth else None
    ours = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    assert _native.spray_puffs(ours, puffs, stands, hidden, depth, 2.5, 0.5, 0.7)
    theirs = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for puff in puffs.tolist():
        road_module._add_puff(theirs, puff, stands, hidden, depth, 2.5)
    assert float(theirs.max()) > 1.0
    assert float(np.abs(ours - theirs).max()) < 1e-5 * float(theirs.max())


@requires_native
def test_the_spray_kernel_declines_what_it_was_not_written_for_and_draws_nothing_odd() -> None:
    puffs = _puffs(20, 0, True)
    stands, hidden = _plane(1, 0.0, HEIGHT), _plane(2)
    spray = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    assert not _native.spray_puffs(spray, puffs[:, :14], stands, hidden, None, 2.5, 0.5, 0.7)
    assert not _native.spray_puffs(
        spray, puffs.astype(np.float32), stands, hidden, None, 2.5, 0.5, 0.7
    )
    assert not _native.spray_puffs(spray, puffs, stands[:8], hidden, None, 2.5, 0.5, 0.7)
    assert not _native.spray_puffs(
        spray, puffs, stands, hidden, hidden.astype(np.float64), 2.5, 0.5, 0.7
    )
    assert float(spray.max()) == 0.0
    # A puff that is nowhere, or of no size a frame holds, is left out and
    # nothing is written outside the frame for it.
    odd = _puffs(4, 1, False)
    odd[0, 0], odd[1, 4], odd[2, 1], odd[3, 9] = np.nan, 1e12, -1e9, np.inf
    assert _native.spray_puffs(spray, odd, stands, hidden, None, 2.5, 0.5, 0.7)
    assert np.isfinite(spray).all()


# ---------------------------------------------------------------- normals


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("size", [(HEIGHT, WIDTH), (5, 5), (6, 31), (40, 5)])
@pytest.mark.parametrize("sigma", [1.0, 0.0])
def test_the_normals_kernel_is_numpy_to_the_bit(
    tier: str, size: tuple[int, int], sigma: float
) -> None:
    """One-sided differences, the cross product and the turn to face the
    lens: numpy's arithmetic in its order, on the frame's borders too."""
    _native.set_tier(tier)
    depth = _depth(2)[: size[0], : size[1]].copy()
    ours, jump = normals_module._normals_and_jump(depth, 60.0, sigma)
    with _native.disabled():
        theirs, jump_numpy = normals_module._normals_and_jump(depth, 60.0, sigma)
    for plane, plane_numpy in zip(ours, theirs, strict=True):
        assert np.array_equal(plane, plane_numpy)
    assert np.array_equal(jump, jump_numpy)


@requires_native
def test_the_normals_kernel_declines_a_frame_too_small_to_difference() -> None:
    across = np.zeros(4, dtype=np.float32)
    assert _native.normals(np.ones((9, 4), np.float32), across, np.zeros(9, np.float32), 2) is None
    depth = np.ones((9, 9), np.float32)
    row = np.zeros(9, dtype=np.float32)
    assert _native.normals(depth, row, row, 2) is not None
    assert _native.normals(depth.astype(np.float64), row, row, 2) is None
    assert _native.normals(depth, row[:8], row, 2) is None
    assert _native.normals(depth, row, row, 0) is None


# ------------------------------------------------------------- the nights


def _lighting(lamps: bool) -> SceneLighting:
    """A scene's lighting with every field there is, or with none."""
    if not lamps:
        return SceneLighting()
    return SceneLighting(
        street=_plane(11, 0.0, 2.0),
        beams=_plane(12, 0.0, 1.5),
        in_beam=_plane(13),
        vehicle_light=_field(14, 0.0, 0.6),
        vehicle_glow=_field(15, 0.0, 0.3),
        head_light=_plane(16, 0.0, 0.5),
        head_glow=_plane(17, 0.0, 0.2),
        road=(_plane(18) > 0.4).astype(np.float32),
    )


def _image(seed: int = 5) -> np.ndarray:
    """An 8-bit frame of the tests' size."""
    return np.random.default_rng(seed).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def _infrared(lamps: bool) -> dict[str, Any]:
    """What ``NightIrMode.apply`` is called with: every optional part, or none."""
    params: dict[str, Any] = {
        "exposure": _plane(21, 0.0, 3.0),
        "ambient": 0.02,
        "target_level": 0.4,
        "vegetation_gain": 0.6,
        "gain": 1.4,
        "read_noise": 0.01,
        "noise_seed": 7,
        "lighting": _lighting(lamps),
    }
    if lamps:
        params.update(
            vegetation=_plane(22),
            night_sky=_plane(23),
            road_level=0.3,
            retro=_plane(24, 0.0, 2.0),
            sheeting=_plane(25, 0.0, 30.0) * (_plane(26) > 0.97),
            exposed=[1.0],
        )
    return params


def _colour(lamps: bool) -> dict[str, Any]:
    """What ``NightColorMode.apply`` is called with: every optional part, or none."""
    params: dict[str, Any] = {
        "lighting": _lighting(lamps),
        "street_colour": np.array([1.0, 0.75, 0.45], dtype=np.float32),
        "head_colour": np.array([1.0, 0.95, 0.85], dtype=np.float32),
        "sky_colour": np.array([0.4, 0.5, 0.8], dtype=np.float32),
        "sky_glow": 0.05,
        "target_level": 0.35,
        "gain": 1.4,
        "read_noise": 0.01,
        "noise_seed": 7,
    }
    if lamps:
        params.update(night_sky=_plane(23), road_level=0.3, exposed=[1.0])
    return params


def _applied(cls: type, params: dict[str, Any]) -> np.ndarray:
    """A night transform's ``apply`` on the tests' frame, without its constructor."""
    transform = cls.__new__(cls)
    transform.residual_tint = 0.04
    out: np.ndarray = cls.apply(transform, _image(), **params)
    return out


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("lamps", [True, False])
@pytest.mark.parametrize("night", ["infrared", "colour"])
def test_the_nights_in_exact_mode_are_numpy_s_frames_to_the_byte(
    tier: str, lamps: bool, night: str
) -> None:
    """Exact mode takes only the kernels whose arithmetic is numpy's: the
    exposure, the lamps and the grain. The frame is the same frame."""
    _native.set_tier(tier)
    mode.set_fast(False)
    cls, params = (
        (NightIrMode, _infrared(lamps))
        if night == "infrared"
        else (
            NightColorMode,
            _colour(lamps),
        )
    )
    ours = _applied(cls, params)
    with _native.disabled():
        theirs = _applied(cls, params)
    assert ours.std() > 5.0
    assert np.array_equal(ours, theirs)


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("lamps", [True, False])
@pytest.mark.parametrize("night", ["infrared", "colour"])
def test_the_nights_in_fast_mode_are_numpy_s_frames_within_a_level(
    tier: str, lamps: bool, night: str
) -> None:
    """Fast mode also takes the light on the surfaces in one pass. A sum of
    three weighted channels is added in the kernel's order and in whatever
    order numpy's matrix product has: a level in a few values at most."""
    _native.set_tier(tier)
    mode.set_fast(True)
    cls, params = (
        (NightIrMode, _infrared(lamps))
        if night == "infrared"
        else (
            NightColorMode,
            _colour(lamps),
        )
    )
    ours = _applied(cls, params)
    with _native.disabled():
        theirs = _applied(cls, params)
    differ = np.abs(ours.astype(np.int16) - theirs.astype(np.int16))
    assert int(differ.max()) <= 1
    assert float((differ > 0).mean()) < 0.002


@requires_native
@pytest.mark.parametrize("tier", TIERS)
def test_the_reflectance_kernel_is_the_formula_with_its_sum_written_out(tier: str) -> None:
    """Bit for bit against the numpy expressions with the three channels
    added in the kernel's order, and within rounding of the function, which
    leaves that sum to numpy's matrix product."""
    _native.set_tier(tier)
    linear, vegetation = _field(31, 0.0, 1.2), _plane(32)
    response = night_ir_module._NIR_RESPONSE
    red, green, blue = linear[..., 0], linear[..., 1], linear[..., 2]
    broadband = red * response[0] + green * response[1] + blue * response[2]
    greenness = np.clip(
        (green - np.maximum(red, blue)) / np.maximum(green + red + blue, 1e-4), 0.0, 1.0
    )
    usual = 0.37
    for named in (vegetation, None):
        expected = greenness
        if named is not None:
            lit = np.clip(broadband / usual, 0.0, 1.0)
            expected = np.maximum(greenness, np.float32(0.5) * named * lit)
        expected = np.clip(broadband + 0.6 * expected, 0.0, 4.0)
        ours = _native.nir_reflectance(linear, named, response, 0.6, 0.5, usual)
        assert expected.dtype == np.float32
        assert np.array_equal(ours, expected)
    mode.set_fast(True)
    fast = night_ir_module.near_infrared_reflectance(linear, 0.6, vegetation)
    mode.set_fast(False)
    reference = night_ir_module.near_infrared_reflectance(linear, 0.6, vegetation)
    assert float(np.abs(fast - reference).max()) < 2e-5


@requires_native
def test_the_night_kernels_decline_what_they_were_not_written_for() -> None:
    plane, field = _plane(1), _field(2)
    wide = plane.astype(np.float64)
    response = np.ones(3, dtype=np.float32)
    assert _native.nir_reflectance(field.astype(np.float64), None, response, 0.5, 0.5, 0.3) is None
    assert _native.nir_reflectance(field, wide, response, 0.5, 0.5, 0.3) is None
    assert _native.night_ir_light(plane, wide, None, None, None, None, 0.0, 0.0, 0.5) is None
    assert _native.night_ir_light(plane, plane, plane[:8], None, None, None, 0.0, 0.0, 0.5) is None
    assert _native.night_ir_expose(plane, plane, [1.0] * 7) is None
    assert _native.night_ir_expose(plane, wide, [1.0] * 8) is None
    assert _native.night_ir_expose(plane, plane, [1.0] * 8, vehicle_lamps=(plane, plane)) is None
    assert _native.night_colour_light(field, None, None, None, None, None, [1.0] * 13) is None
    assert _native.night_colour_light(field, wide, None, None, None, None, [1.0] * 14) is None
    assert _native.night_colour_expose(field, field, response, [1.0] * 4) is None
    assert _native.night_colour_expose(field, plane, response, [1.0] * 5) is None
    assert _native.night_colour_expose(plane, plane, response, [1.0] * 5) is None


# ---------------------------------------------------------- fog at night


def _fog(cones: bool, illuminated: bool) -> dict[str, Any]:
    """What ``NightFog.apply`` is called with, in each of its three lightings."""
    return {
        "optical_depth": _plane(41, 0.0, 2.5),
        "sky": _plane(42),
        "glow_radius": 6.0,
        "backscatter": 0.5 if illuminated else 0.0,
        "beam": _plane(43),
        "cones": _field(44, 0.0, 0.2) if cones else None,
    }


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("cones", [True, False])
@pytest.mark.parametrize("illuminated", [True, False])
def test_the_night_fog_with_its_kernels_is_numpy_s_frame_to_the_byte(
    tier: str, cones: bool, illuminated: bool
) -> None:
    """What the fog takes of the lamps and what it sends back: numpy's
    arithmetic in its order, in exact mode, so the same frame."""
    _native.set_tier(tier)
    mode.set_fast(False)
    image = _image(6)
    image[20:26, 60:66] = 255
    transform = NightFog.__new__(NightFog)
    ours = NightFog.apply(transform, image, **_fog(cones, illuminated))
    with _native.disabled():
        theirs = NightFog.apply(transform, image, **_fog(cones, illuminated))
    assert not np.array_equal(ours, image)
    assert np.array_equal(ours, theirs)


@requires_native
@pytest.mark.parametrize("tier", TIERS)
def test_the_fog_kernels_are_the_numpy_expressions_to_the_bit(tier: str) -> None:
    _native.set_tier(tier)
    linear, halo, cones, lit = _field(51, 0.0, 1.3), _field(52), _field(53, 0.0, 0.3), _field(54)
    lamps, through, wall = _plane(55), _plane(56), _plane(57, 0.0, 0.4)
    clipped, one_way = lamps[..., None], through[..., None]
    radiance = linear * (1.0 + 5.0 * clipped)
    removed = radiance * clipped * (1.0 - one_way)
    weight = removed[..., 0] + removed[..., 1] + removed[..., 2]
    ours = _native.fog_lost(linear, lamps, through, 5.0)
    assert ours is not None
    for got, expected in zip(ours, (radiance, removed, weight), strict=True):
        assert expected.dtype == np.float32
        assert np.array_equal(got, expected)
    level = np.array([0.05, 0.04, 0.03], dtype=np.float32)
    diffuse = np.array([0.01, 0.012, 0.014], dtype=np.float32)
    # The lamps' cones under an illuminator: the light crosses the fog twice.
    expected = radiance * (one_way * one_way) + 0.6 * halo
    expected = expected + cones + level.reshape(1, 1, 3) * (1.0 - one_way)
    expected = expected + wall[..., None]
    got = _native.fog_veil(
        radiance, halo, through, [0.6, 1.0, *level, 0.0, 0.0, 0.0, 0.0, 0.0], cones, None, wall
    )
    assert np.array_equal(got, expected)
    # The veil: the lit volume and the scene's own level.
    expected = radiance * one_way + 0.6 * halo
    veil = np.float32(2.2) * (np.float32(0.75) * lit + diffuse.reshape(1, 1, 3))
    expected = expected + veil * (1.0 - one_way)
    got = _native.fog_veil(
        radiance, halo, through, [0.6, 0.0, 0.0, 0.0, 0.0, 2.2, 0.75, *diffuse], None, lit, None
    )
    assert np.array_equal(got, expected)
    assert _native.fog_lost(linear, lamps.astype(np.float64), through, 5.0) is None
    assert _native.fog_veil(radiance, halo, through, [0.0] * 9) is None
    assert _native.fog_veil(radiance, halo[..., 0], through, [0.0] * 10) is None


# ------------------------------------------------- lying snow, ground noise


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("labelled", [True, False])
@pytest.mark.parametrize("patchy", [True, False])
def test_the_lying_snow_kernel_is_the_numpy_expressions_to_the_bit(
    tier: str, labelled: bool, patchy: bool
) -> None:
    """Where the ground is a canopy, how the snow lies in patches and in
    the surface's grain, what a canopy's proud clumps hold, nothing on the
    sky: twenty numpy expressions, in fast mode's float32, as one pass."""
    _native.set_tier(tier)
    level, relief, steady, green = _plane(61), _plane(62, 0.0, 0.5), _plane(63), _plane(64)
    grain, luma, sky = _plane(65), _plane(66), _plane(67)
    around = np.maximum(_plane(68), 1e-4)
    lawn, named = (_plane(69), _plane(70)) if labelled else (None, None)
    noise = _plane(71) if patchy else None
    strength, patchiness, thin = 0.7, 0.5 if patchy else 0.0, 0.3
    rough = np.clip((relief - 0.18) / 0.18, 0.0, 1.0)
    unsteady = np.clip((0.85 - steady) / 0.25, 0.0, 1.0) * np.clip((steady - 0.15) / 0.2, 0.0, 1.0)
    canopy = rough * np.maximum(green, unsteady)
    if lawn is not None and named is not None:
        canopy = np.maximum(canopy * (1.0 - lawn), named)
    ground = level * (1.0 - canopy)
    if noise is not None:
        patches = np.clip((noise - 0.5) * 2.5 + 0.5, 0.0, 1.0)
        ground = ground * (1.0 - patchiness * (1.0 - patches))
    uneven = 1.0 + 0.2 * thin * (2.0 * grain - 1.0)
    settled = ground * strength * uneven
    proud = np.clip((luma / around - 0.95) / 0.35, 0.0, 1.0)
    settled = settled + strength * 0.7 * canopy * proud
    settled = settled * (1.0 - sky)
    ours = _native.snow_settled(
        level,
        relief,
        steady,
        green,
        grain,
        luma,
        around,
        sky,
        [0.18, 0.18, patchiness, 2.5, strength, 0.2 * thin, strength * 0.7],
        lawn,
        named,
        noise,
    )
    assert settled.dtype == np.float32 and float(settled.max()) > 0.1
    assert np.array_equal(ours, settled)


@requires_native
def test_the_lying_snow_kernel_declines_what_it_was_not_written_for() -> None:
    plane = _plane(1)
    planes = [plane] * 8
    assert _native.snow_settled(*planes, [0.2] * 7) is not None
    assert _native.snow_settled(*planes, [0.2] * 6) is None
    assert _native.snow_settled(*planes, [0.2] * 7, lawn=plane.astype(np.float64)) is None
    assert _native.snow_settled(*planes[:7], plane[:8], [0.2] * 7) is None
    assert _native.snow_settled(plane.astype(np.float64), *planes[1:], [0.2] * 7) is None


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("uniform", [True, False])
def test_the_ground_s_noise_with_its_kernel_is_numpy_s_to_the_bit(tier: str, uniform: bool) -> None:
    """Each pixel's place in the noise tile, folded at the tile's edges: in
    fast mode, numpy's float32 arithmetic and its floor."""
    _native.set_tier(tier)
    mode.set_fast(True)
    depth = _depth(3)

    def noise() -> np.ndarray:
        rng = np.random.default_rng(8)
        return ground_module.ground_noise(depth, 140.0, 18.0, 1.5, rng, uniform=uniform)

    ours = noise()
    with _native.disabled():
        theirs = noise()
    assert float(ours.std()) > 0.05
    assert np.array_equal(ours, theirs)
    assert _native.ground_tile(depth.astype(np.float64), [1.0] * 8) is None
    assert _native.ground_tile(depth, [1.0] * 7) is None


# ------------------------------------------------- lamps on surfaces


def _lit_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A level road under a camera on a mast: its points, a point on it, and up."""
    rows = np.arange(240, dtype=np.float32)[:, None]
    depth = np.clip(8.0 * 280.0 / np.maximum(rows - 60.0, 1.0), 4.0, 300.0)
    points, _step = scene_points(np.broadcast_to(depth, (240, 320)).astype(np.float32), 280.0)
    ground = np.ones(points.shape[:2], dtype=np.float32)
    ground[: points.shape[0] // 3] = 0.0
    plane = fit_ground(points, ground)
    assert plane is not None
    return points, plane[0], plane[1]


@requires_native
@pytest.mark.parametrize("tier", TIERS)
def test_the_street_light_kernel_is_numpy_s_to_a_bit_of_a_float(tier: str) -> None:
    """Every lamp's pool on every point. numpy raises a cosine to the power
    of one and a half; the kernel multiplies it by its root, which is the
    same number to the last bit of a float or one off it."""
    _native.set_tier(tier)
    points, _middle, up = _lit_scene()
    rng = np.random.default_rng(2)
    lamps = np.stack(
        [rng.uniform(-20, 20, 9), rng.uniform(-9, -5, 9), rng.uniform(8, 90, 9), np.full(9, 9.0)],
        axis=1,
    ).astype(np.float32)
    ours = street_light(points, up, lamps)
    with _native.disabled():
        theirs = street_light(points, up, lamps)
    assert float(theirs.max()) > 0.5
    assert float(np.abs(ours - theirs).max()) <= 1e-6 * float(theirs.max())
    assert not _native.street_light(ours, points.astype(np.float64), ours, lamps, up, 0.03)
    assert not _native.street_light(ours, points, ours, lamps[:, :3], up, 0.03)
    assert not _native.street_light(ours[:4], points, ours, lamps, up, 0.03)


@requires_native
@pytest.mark.parametrize("tier", TIERS)
def test_the_headlamp_kernel_lays_numpy_s_beams(tier: str) -> None:
    """Two pools ahead of each lit vehicle, coming on and driving away. The
    kernel's exponential is a polynomial: within a hundred-thousandth of
    the brightest pool."""
    _native.set_tier(tier)
    points, middle, up = _lit_scene()
    rng = np.random.default_rng(4)
    sources = [
        BeamSource((float(rng.uniform(40, 280)), float(rng.uniform(120, 235))), bool(index % 2))
        for index in range(8)
    ]
    standing = np.zeros(points.shape[:2], dtype=bool)
    standing[60:80, 100:140] = True
    ours = headlamp_light(points, middle, up, 280.0, (240, 320), sources, standing)
    with _native.disabled():
        theirs = headlamp_light(points, middle, up, 280.0, (240, 320), sources, standing)
    assert float(theirs.max()) > 0.3
    assert float(np.abs(ours - theirs).max()) < 1e-5 * float(theirs.max())
    plane = np.zeros(points.shape[:2], dtype=np.float32)
    beams = np.zeros((2, 6), dtype=np.float32)
    assert not _native.headlamp_beams(plane, plane, plane, plane, plane, beams, [1.0] * 6)
    assert not _native.headlamp_beams(plane, plane, plane, plane, plane, beams[:, :5], [1.0] * 7)
    assert not _native.headlamp_beams(
        plane, plane.astype(np.float64), plane, plane, plane, beams, [1.0] * 7
    )


# ---------------------------------------------------------- a road's frame


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("bend", [0.0, 900.0, -2500.0])
def test_the_road_frame_s_kernels_are_numpy_s_to_the_bit(tier: str, bend: float) -> None:
    """Which carriageway a pixel is on, how far across it, how wide it is
    there and which way it runs: numpy's double arithmetic, narrowed where
    numpy narrows it, on a straight road and on two bends."""
    _native.set_tier(tier)
    height, width = 150, 211
    rows, columns = np.mgrid[0:height, 0:width]
    model = road_module.RoadModel(centre=100.0, horizon=31.4, bend=bend)
    place = np.nan_to_num(model.offset(columns, rows), nan=9.0)
    # Two carriageways and the reserve between them, and nothing above the horizon.
    road = ((np.abs(place + 0.9) < 0.6) | (np.abs(place - 0.8) < 0.5)) & (rows > 40)
    on_road = np.zeros_like(road)
    on_road[100:110, 60:75] = True
    ours = road_module.model_frame(model, road, on_road)
    with _native.disabled():
        theirs = road_module.model_frame(model, road, on_road)
    assert ours is not None and theirs is not None
    assert int(theirs.piece.max()) == 2
    for field in theirs._fields:
        assert np.array_equal(getattr(ours, field), getattr(theirs, field)), field


@requires_native
def test_the_road_frame_s_kernels_decline_what_they_were_not_written_for() -> None:
    assert _native.road_along(0, 8, 4.0, -3.0, 0.0) is None
    offset = np.zeros((6, 8))
    at = np.ones((6, 8), dtype=np.int64)
    whole = np.ones((6, 8), dtype=bool)
    below = np.arange(1.0, 7.0)
    of_position = np.array([0, 1, 0], dtype=np.int32)
    tables = (of_position, np.zeros(2), np.ones(2))
    given = ((10, 8), 4, offset, at, whole, below)
    assert _native.road_places(*given, *tables) is not None
    assert _native.road_places(*given[:2], offset.astype(np.float32), *given[3:], *tables) is None
    assert _native.road_places(*given[:3], at.astype(np.int32), *given[4:], *tables) is None
    # A position no table has, or a carriageway none has, is not looked up.
    assert _native.road_places(*given[:3], at + 2, *given[4:], *tables) is None
    assert _native.road_places(*given, of_position + 1, *tables[1:]) is None
    assert _native.road_places((10, 8), 5, *given[2:], *tables) is None


# ----------------------------------------------------------- wheel tracks


def _two_roads() -> tuple[road_module.RoadFrame, road_module.Lanes]:
    """Two carriageways side by side, of four lanes and two, and a verge between."""
    rng = np.random.default_rng(6)
    piece = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    piece[10:, 5:70] = 1
    piece[10:, 80:140] = 2
    across = rng.uniform(-0.05, 1.05, (HEIGHT, WIDTH)).astype(np.float32)
    width_px = rng.uniform(2.0, 90.0, (HEIGHT, WIDTH)).astype(np.float32)
    frame = road_module.RoadFrame(
        weight=(piece > 0).astype(np.float32),
        across=across,
        width_px=width_px,
        along=np.zeros((HEIGHT, WIDTH, 2), dtype=np.float32),
        piece=piece,
        heights_wide=np.zeros(3),
    )
    edges = [np.zeros(0), np.array([0.02, 0.27, 0.5, 0.74, 0.97]), np.array([0.1, 0.52, 0.9])]
    return frame, road_module.Lanes(edges, [0, 3, 2])


@requires_native
@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("rough", [True, False])
@pytest.mark.parametrize("fast", [True, False])
def test_the_wheel_track_kernel_is_numpy_s_to_the_bit(tier: str, rough: bool, fast: bool) -> None:
    """Each pixel's lane found among its road's lines, and the two tracks
    down it: numpy's search as a scan without a branch, and its arithmetic
    in double where numpy's is and float where it is."""
    _native.set_tier(tier)
    mode.set_fast(fast)
    frame, lanes = _two_roads()
    extra = (_plane(7), _plane(8), 0.3) if rough else ()

    def tracks() -> np.ndarray:
        return road_module.wheel_tracks(frame, lanes, 0.09, np.random.default_rng(1), *extra)

    ours = tracks()
    with _native.disabled():
        theirs = tracks()
    assert float(theirs.max()) > 0.5 and not theirs[:10].any()
    assert np.array_equal(ours, theirs)


@requires_native
def test_the_wheel_track_kernel_declines_what_it_was_not_written_for() -> None:
    frame, lanes = _two_roads()
    use = np.ones(64, dtype=np.float32)
    how = [0.09, 0.1, 0.22, 0.25, 1.5]
    given = (frame.across, frame.width_px, frame.weight, None, None, lanes.edges)
    assert _native.wheel_tracks(frame.piece, *given, use, how) is not None
    assert _native.wheel_tracks(frame.piece.astype(np.int64), *given, use, how) is None
    assert _native.wheel_tracks(frame.piece, *given, use[:32], how) is None
    assert _native.wheel_tracks(frame.piece, *given, use, how[:4]) is None
    wide = frame.across.astype(np.float64)
    assert _native.wheel_tracks(frame.piece, wide, *given[1:], use, how) is None
    # A road with one line, or none, has no lane to put a track in.
    lone = [np.zeros(0), np.array([0.5]), np.zeros(0)]
    nothing = _native.wheel_tracks(frame.piece, *given[:5], lone, use, how)
    assert nothing is not None and not nothing.any()


# ------------------------------------------------ rain, lamps and spray


@requires_native
@pytest.mark.parametrize("tier", TIERS)
def test_the_rain_s_lamp_and_spray_kernels_are_the_numpy_expressions_to_the_bit(tier: str) -> None:
    _native.set_tier(tier)
    scene, lamp_light, lamp_glow = _field(81), _field(82, 0.0, 6.0), _field(83, 0.0, 0.4)
    lamp_light[lamp_light < 4.0] = 0.0
    spray, veil = _plane(84, 0.0, 0.8), _plane(85, 0.3, 1.0)
    through = veil[..., None]
    lit = (np.array([0.5, 0.55, 0.6], dtype=np.float32) * np.float32(0.9)).astype(np.float32)
    ours = scene.copy()
    assert _native.lamps_lit(ours, lamp_light, veil)
    assert np.array_equal(ours, scene + np.minimum(lamp_light, 1.0) * through)
    for with_spray in (True, False):
        for with_lamps in (True, False):
            if not (with_spray or with_lamps):
                continue
            expected, clear, wet_air = scene, through, 0.35
            if with_spray:
                share = spray[..., None]
                expected = expected * (1.0 - share) + lit[None, None, :] * share
                clear = through * (1.0 - share)
                wet_air = 0.35 + (1.0 - 0.35) * share
            if with_lamps:
                expected = expected + (lamp_light - np.minimum(lamp_light, 1.0)) * clear
                expected = expected + lamp_glow * wet_air
            ours = scene.copy()
            assert _native.spray_and_lamps(
                ours,
                spray if with_spray else None,
                (lamp_light, lamp_glow) if with_lamps else None,
                veil,
                lit,
                0.35,
            )
            assert expected.dtype == np.float32
            assert np.array_equal(ours, expected)
    assert not _native.spray_and_lamps(scene.copy(), spray[:8], None, veil, lit, 0.35)
    assert not _native.spray_and_lamps(scene.astype(np.float64), spray, None, veil, lit, 0.35)
    assert not _native.spray_and_lamps(scene.copy(), spray, None, veil, lit[:2], 0.35)
    assert not _native.lamps_lit(scene.copy(), lamp_light, veil[:8])
    assert not _native.lamps_lit(scene[:, ::2], lamp_light[:, ::2], veil[:, ::2])


# ---------------------------------------------------------------- the tiers


@requires_native
def test_every_tier_gives_the_same_scene() -> None:
    """A picture must not depend on the CPU: what each of these kernels
    computes is the same bytes on every tier of the build."""
    points, _step = scene_points(_depth(), 140.0)
    up = np.array([0.0, -0.94, -0.34], dtype=np.float32)
    lamps = np.array([[3.0, -7.0, 40.0, 9.0], [-8.0, -6.0, 15.0, 8.0]], dtype=np.float32)
    puffs = _puffs(120, 5, True)
    stands, hidden, depth = _plane(1, 0.0, HEIGHT), _plane(2), _depth(4)
    results = []
    for tier in _native.tiers():
        _native.set_tier(tier)
        spray = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        _native.spray_puffs(spray, puffs, stands, hidden, depth, 2.5, 0.5, 0.7)
        mode.set_fast(True)
        frames = [
            _applied(NightIrMode, _infrared(True)),
            _applied(NightColorMode, _colour(True)),
            ground_module.ground_noise(depth, 140.0, 18.0, 1.5, np.random.default_rng(8)),
        ]
        mode.set_fast(False)
        results.append((*street_air(points, up, lamps), spray, *frames))
    for other in results[1:]:
        for first, second in zip(results[0], other, strict=True):
            assert np.array_equal(first, second)


def test_without_the_kernels_every_scene_entry_point_declines() -> None:
    """What an install without the library runs: each says "not done"."""
    plane, field = _plane(1), _field(2)
    row = np.zeros(WIDTH, dtype=np.float32)
    with _native.disabled():
        assert not _native.street_air(
            plane, plane, (plane, plane, plane), plane, plane, np.zeros((1, 4), np.float32),
            np.zeros(3, np.float32), 0.36, 0.03, 0.5,
        )  # fmt: skip
        assert not _native.spray_puffs(plane, _puffs(3, 0, False), plane, plane, None, 2, 0.5, 0.7)
        assert _native.normals(plane, row, np.zeros(HEIGHT, np.float32), 2) is None
        assert _native.nir_reflectance(field, None, np.ones(3, np.float32), 0.5, 0.5, 0.3) is None
        assert _native.night_ir_light(plane, plane, None, None, None, None, 0.0, 0.0, 0.5) is None
        assert _native.night_ir_expose(plane, plane, [1.0] * 8) is None
        assert _native.night_colour_light(field, None, None, None, None, None, [1.0] * 14) is None
        assert _native.night_colour_expose(field, field, np.ones(3), [1.0] * 5) is None
        assert _native.fog_lost(field, plane, plane, 5.0) is None
        assert _native.fog_veil(field, field, plane, [0.0] * 10) is None
        assert _native.snow_settled(*([plane] * 8), [0.2] * 7) is None
        assert _native.ground_tile(plane, [1.0] * 8) is None
        lamps = np.zeros((1, 4), np.float32)
        assert not _native.street_light(plane, field, plane, lamps, np.zeros(3, np.float32), 0.03)
        beams = np.zeros((1, 6), np.float32)
        assert not _native.headlamp_beams(plane, plane, plane, plane, plane, beams, [1.0] * 7)
        frame, lanes = _two_roads()
        assert (
            _native.wheel_tracks(
                frame.piece, plane, plane, plane, None, None, lanes.edges,
                np.ones(64, np.float32), [0.1] * 5,
            )
            is None
        )  # fmt: skip
        assert _native.road_along(8, 8, 4.0, -3.0, 0.0) is None
        assert not _native.lamps_lit(field.copy(), field, plane)
        assert not _native.spray_and_lamps(field.copy(), plane, None, plane, field[0, 0], 0.35)
