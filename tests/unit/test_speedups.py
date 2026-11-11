"""Rewrites that are faster and give the same result, checked against what
they replaced.

Each of these stands in for code that is no longer in the library, so the
old code is written out here, in its plainest form, as the thing to agree
with. Fast-mode forms, which are allowed to differ, are in test_modes.py.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from weathermentations import mode
from weathermentations.atmosphere import rain
from weathermentations.camera import NightIrMode
from weathermentations.depth import DepthAnythingV2Estimator
from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.diagnostics import record_draw, recording
from weathermentations.utils import precipitation
from weathermentations.utils.color import veil_radiance_linear
from weathermentations.utils.sources import source_mask

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the depth model requires CUDA"
)


# ---------------------------------------------------------------- sampling


def _sample_every_candidate(rng, count, slope, d_min, d_max, near_m, far_m, focal_px,
                            min_width_px, size_scale=1.0, shape_mu=0.0):  # fmt: skip
    """`sample_resolvable_particles` before it learned to skip candidates:
    every one gets a diameter and a distance, and then most are thrown away."""
    acceptance = precipitation.resolvable_fraction(
        slope, d_min, d_max, near_m, far_m, focal_px, min_width_px, size_scale, shape_mu=shape_mu
    )
    acceptance = max(acceptance, 1e-4)
    diameters, distances, found = [], [], 0
    for _ in range(8):
        batch = int(min(2_000_000, max(1024, (count - found) / acceptance * 1.4)))
        candidate_d = precipitation.sample_gamma_diameters(
            rng, batch, slope, d_min, d_max, shape_mu
        )
        candidate_z = precipitation.volume_uniform_distances(rng, batch, near_m, far_m)
        width = focal_px * (candidate_d * size_scale * 1e-3) / candidate_z
        keep = width > min_width_px
        if keep.any():
            diameters.append(candidate_d[keep])
            distances.append(candidate_z[keep])
            found += int(keep.sum())
        if found >= count:
            break
    if not diameters:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    return (
        np.concatenate(diameters)[:count].astype(np.float32),
        np.concatenate(distances)[:count].astype(np.float32),
    )


NIGHT_RAIN = {
    "count": 4000, "slope": 1.65, "d_min": 0.8, "d_max": 6.0, "near_m": 0.25, "far_m": 12.0,
    "focal_px": 721.0, "min_width_px": 0.46, "size_scale": 1.0, "shape_mu": 0.0,
}  # fmt: skip


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({}, id="marshall-palmer"),
        pytest.param({"shape_mu": 2.5}, id="gamma"),
        pytest.param({"shape_mu": 6.0, "slope": 3.1}, id="narrow-gamma"),
        pytest.param({"min_width_px": 1.2, "count": 800}, id="few-resolve"),
        pytest.param(
            {"size_scale": 9.0, "slope": 2.2, "d_min": 0.3, "far_m": 40.0}, id="snow-sized"
        ),
        pytest.param({"min_width_px": 0.05}, id="nearly-all-resolve"),
        pytest.param({"far_m": 3.0, "focal_px": 3000.0}, id="all-resolve"),
        pytest.param({"min_width_px": 40.0, "count": 50}, id="none-resolve"),
        pytest.param({"near_m": 0.0, "count": 200}, id="from-the-lens"),
        pytest.param({"d_min": 0.2, "d_max": 2.0, "shape_mu": 1.0}, id="small-drops"),
    ],
)
def test_skipping_hopeless_candidates_returns_the_same_particles(changes: dict) -> None:
    """Same uniforms in the same order, the same pairs out, and the generator
    left where it was: a frame's rain is the rain it had."""
    arguments = {**NIGHT_RAIN, **changes}
    for seed in range(2):
        ours, reference = np.random.default_rng(seed), np.random.default_rng(seed)
        diameters, distances = precipitation.sample_resolvable_particles(ours, **arguments)
        expected_d, expected_z = _sample_every_candidate(reference, **arguments)
        assert np.array_equal(diameters, expected_d)
        assert np.array_equal(distances, expected_z)
        assert ours.random() == reference.random(), "the generator was advanced differently"


# ------------------------------------------------------------- diagnostics


def test_a_costly_diagnostic_is_computed_only_while_recording() -> None:
    calls = []

    def median() -> float:
        calls.append(1)
        return 3.5

    record_draw(depth_p50_m=median)
    assert not calls, "nobody was recording"

    class _Ground(DepthEstimator):
        """Depth rising from 0.2 at the top row to 1 at the bottom."""
        def estimate(self, image: np.ndarray) -> np.ndarray:
            rows = np.linspace(0.2, 1.0, image.shape[0], dtype=np.float32)
            return np.repeat(rows[:, None], image.shape[1], axis=1)

    image = np.full((64, 96, 3), 90, dtype=np.uint8)
    transform = NightIrMode(depth_provider=SharedDepthProvider(_Ground()), p=1.0)
    with recording() as log:
        transform(image=image)
    drew = log[0]["drew"]
    assert drew["depth_min_m"] <= drew["depth_p50_m"] <= drew["depth_max_m"]
    assert isinstance(drew["depth_p50_m"], float)


# -------------------------------------------------------------------- depth


@requires_cuda
def test_preprocessing_without_the_processor_gives_the_same_depth() -> None:
    """The estimator skips the Hugging Face processor only after checking, on
    this installation, that it gets the processor's tensor bit for bit."""
    estimator = DepthAnythingV2Estimator(half_precision=False)
    rng = np.random.default_rng(0)
    for shape in ((240, 320, 3), (518, 518, 3), (300, 900, 3)):
        frame = rng.integers(0, 256, shape, dtype=np.uint8)
        direct = estimator._preprocess_directly(frame)
        if not estimator._direct:
            pytest.skip("this transformers version preprocesses by other code")
        reference = estimator._processor(images=frame, return_tensors="pt")["pixel_values"]
        assert torch.equal(direct.cpu(), reference)
        estimator._direct = True
        with_direct = estimator.estimate(frame)
        estimator._direct = False
        assert np.array_equal(with_direct, estimator.estimate(frame))
        estimator._direct = True


@requires_cuda
def test_half_precision_follows_the_mode_unless_told() -> None:
    """Fast mode runs the network in float16, which moves the depth a little;
    exact mode and `half_precision=False` do not."""
    frame = np.random.default_rng(1).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    follows, never = DepthAnythingV2Estimator(), DepthAnythingV2Estimator(half_precision=False)
    was_fast = mode.fast()
    try:
        mode.set_fast(False)
        exact = follows.estimate(frame)
        assert np.array_equal(exact, never.estimate(frame))
        mode.set_fast(True)
        assert np.array_equal(never.estimate(frame), exact)
        fast = follows.estimate(frame)
        assert not np.array_equal(fast, exact)
        assert np.abs(fast - exact).mean() < 0.01
    finally:
        mode.set_fast(was_fast)


# ----------------------------------------------------------------- wet road


def _wet_road_over_the_whole_frame(linear, wetness, darkening, horizon_px, focal_px,
                                   sky_radiance=None, lamp_gain=1.0, puddles=None):  # fmt: skip
    """`_apply_ground_wetness` before it worked on the wet rows alone: every
    array the height of the frame."""
    height, width = linear.shape[:2]
    rows = np.arange(height, dtype=np.float32)
    about_horizon, about_foot, under_object = rain._mirrored_rows(wetness, horizon_px)
    source = linear
    if lamp_gain > 1.0:
        source = linear * (1.0 + (lamp_gain - 1.0) * source_mask(linear)[..., None])
    map_x = np.ascontiguousarray(
        np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    )

    def mirror(source_rows):
        map_y = np.clip(np.broadcast_to(source_rows, (height, width)), 0.0, height - 1.0)
        map_y = np.ascontiguousarray(map_y, dtype=np.float32)
        linear_, edge = cv2.INTER_LINEAR, cv2.BORDER_REPLICATE
        return cv2.remap(source, map_x, map_y, interpolation=linear_, borderMode=edge)

    if sky_radiance is None:
        sky_radiance = veil_radiance_linear(linear)
    margin = rain._OFF_FRAME_BLEND * height
    off_frame = np.clip((margin - about_horizon) / margin, 0.0, 1.0)[..., None]
    distant = mirror(about_horizon) * (1.0 - off_frame) + sky_radiance.reshape(1, 1, -1) * off_frame
    reflection = distant
    if float(under_object.max()) > 0.0:
        weight = under_object[..., None]
        reflection = distant * (1.0 - weight) + mirror(about_foot) * weight
    sharp = cv2.GaussianBlur(reflection, (0, 0), sigmaX=1.0, sigmaY=3.0)
    smear = max(4.0, rain._FILM_SMEAR * height)
    rough = cv2.GaussianBlur(reflection, (0, 0), sigmaX=max(1.5, 0.3 * smear), sigmaY=smear)
    drop = np.maximum(rows - horizon_px, 0.0)
    cos_theta = drop / np.sqrt(drop * drop + focal_px * focal_px)
    grazing = (1.0 - cos_theta).reshape(height, 1, 1)
    fresnel = rain._WATER_R0 + (1.0 - rain._WATER_R0) * grazing**5
    film_fresnel = rain._WATER_R0 + (1.0 - rain._WATER_R0) * grazing**rain._FILM_GRAZING_EXPONENT
    film = np.float32(
        np.clip(
            (rain._DAMP_ALBEDO_RATIO - darkening)
            / (rain._DAMP_ALBEDO_RATIO - rain._SOAKED_ALBEDO_RATIO),
            0.0,
            1.0,
        )
    )
    water = np.zeros((height, width, 1), dtype=np.float32)
    if puddles is not None:
        water = (puddles * film)[..., None]
    mirror_weight = fresnel * water + film_fresnel * (1.0 - water) * film * rain._FILM_GLOSS
    mirrored = sharp * water + rough * (1.0 - water)
    wet = rain.wet_albedo(linear, darkening) * (1.0 - mirror_weight) + mirrored * mirror_weight
    w = wetness[..., None]
    return (linear * (1.0 - w) + wet * w).astype(linear.dtype)


def _road(seed: int, top: int, height: int = 240, width: int = 320):
    """A frame with wet ground from row ``top`` down, a dry island in it and
    a clipped lamp above it."""
    rng = np.random.default_rng(seed)
    linear = rng.uniform(0.02, 0.9, (height, width, 3)).astype(np.float32)
    linear[30:36, 100:106] = 1.0
    wetness = np.zeros((height, width), dtype=np.float32)
    wetness[top:] = rng.uniform(0.0, 1.0, (height - top, width)).astype(np.float32)
    wetness[top + 20 : top + 50, 60:130] = 0.0
    puddles = np.clip(rng.uniform(-1.0, 1.0, (height, width)), 0.0, 1.0).astype(np.float32)
    return linear, wetness, puddles


@pytest.mark.parametrize("top", [0, 90, 200, 239], ids=["all-wet", "half", "a-strip", "one-row"])
@pytest.mark.parametrize("with_puddles", [True, False])
def test_working_on_the_wet_rows_alone_gives_the_whole_frame_s_road(
    top: int, with_puddles: bool
) -> None:
    """In exact mode. The reflection is blurred down the frame, so the rows
    worked on include a margin as deep as the blur's kernel; with it the
    band blurs to the values the whole frame would."""
    linear, wetness, puddles = _road(top, top)
    arguments = {
        "sky_radiance": np.array([0.5, 0.6, 0.7], dtype=np.float32),
        "lamp_gain": 4.0,
        "puddles": puddles if with_puddles else None,
    }
    with mode.exact():
        ours = rain._apply_ground_wetness(linear, wetness, 0.5, 80.0, 300.0, **arguments)
        whole = _wet_road_over_the_whole_frame(linear, wetness, 0.5, 80.0, 300.0, **arguments)
    assert np.abs(whole - linear).max() > 0.05, "the fixture's road is not wet"
    assert np.array_equal(ours, whole)


def test_a_frame_with_no_wet_ground_comes_back_as_it_was() -> None:
    linear, wetness, _ = _road(1, 100)
    untouched = rain._apply_ground_wetness(linear, np.zeros_like(wetness), 0.5, 80.0, 300.0)
    assert np.array_equal(untouched, linear) and untouched is not linear
