"""Sensor, ISP and transmission effects (plan.md 6.2 / Phase 7).

Each test pins the property that distinguishes the effect from a superficially
similar one — banding from a gradient, flattening from a blur, a codec from
JPEG — since that distinction is the whole reason the transform exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from weathermentations.camera import (
    AcFlicker,
    ExposureTransient,
    NightIrMode,
    RollingShutter,
    TemporalDenoiseArtifacts,
    VideoCodecArtifacts,
    WdrToneMapping,
)
from weathermentations.camera.codec import MACROBLOCK_PX, ffmpeg_path
from weathermentations.camera.flicker import flicker_modulation_depth
from weathermentations.camera.night_ir import near_infrared_reflectance
from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.utils.photometry import srgb_to_linear

requires_ffmpeg = pytest.mark.skipif(
    ffmpeg_path() is None, reason="needs the ffmpeg binary"
)


class _RampDepth(DepthEstimator):
    """Depth increasing with row: near at the top, far at the bottom."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        rows = np.linspace(0.05, 1.0, height, dtype=np.float32).reshape(height, 1)
        return np.broadcast_to(rows, (height, width)).astype(np.float32)


@pytest.fixture
def scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:, :] = (110, 120, 105)
    img[80:160, 60:260] = (170, 60, 55)
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# AcFlicker
# --------------------------------------------------------------------------


def test_flicker_cancels_at_a_whole_number_of_periods() -> None:
    """The physical basis of a camera's anti-flicker shutter setting: a row
    integrating over exactly one flicker period averages the whole waveform
    and records no modulation at all. If this did not go to zero, the sinc
    term would just be a fudge factor."""
    flicker_hz = 100.0
    assert flicker_modulation_depth(0.5, 1.0 / flicker_hz, flicker_hz) == pytest.approx(0, abs=1e-6)
    assert flicker_modulation_depth(0.5, 2.0 / flicker_hz, flicker_hz) == pytest.approx(0, abs=1e-6)
    # A short exposure records nearly the full modulation.
    assert flicker_modulation_depth(0.5, 0.0001, flicker_hz) > 0.49


def test_flicker_bands_are_horizontal_not_vertical(scene: np.ndarray) -> None:
    """Rows are read at different times; columns are not. Banding therefore
    has to be row-structured, which is what separates it from a gradient or
    a vignette."""
    t = AcFlicker(
        p=1.0, modulation_depth_range=(0.3, 0.3), exposure_time_range_s=(0.0005, 0.0005)
    )
    t.set_random_seed(0)
    delta = t(image=scene)["image"].astype(np.float32) - scene.astype(np.float32)

    row_profile = delta.mean(axis=(1, 2))
    column_profile = delta.mean(axis=(0, 2))
    assert row_profile.std() > 6.0 * column_profile.std()


def test_flicker_zero_depth_is_identity(scene: np.ndarray) -> None:
    t = AcFlicker(p=1.0, modulation_depth_range=(0.0, 0.0))
    t.set_random_seed(0)
    np.testing.assert_allclose(
        t(image=scene)["image"].astype(np.float32), scene.astype(np.float32), atol=1.0
    )


# --------------------------------------------------------------------------
# RollingShutter
# --------------------------------------------------------------------------


def test_rolling_shutter_shears_rather_than_blurs(scene: np.ndarray) -> None:
    """Each row is displaced as a unit, so a row's *content* survives — it is
    shifted, not smeared. Sorting a row's pixels makes it invariant to that
    shift, so the sorted rows must be nearly unchanged; a blur would fail."""
    t = RollingShutter(
        p=1.0,
        pan_shift_range_px=(25.0, 25.0),
        wobble_amplitude_range_px=(0.0, 0.0),
        vertical_fraction_range=(0.0, 0.0),
    )
    t.set_random_seed(0)
    out = t(image=scene)["image"]

    middle = scene.shape[0] // 2
    before = np.sort(scene[middle, :, 0])
    after = np.sort(out[middle, :, 0])
    assert np.abs(before.astype(np.float32) - after.astype(np.float32)).mean() < 6.0
    # And the frame really did move.
    assert np.abs(out.astype(np.float32) - scene.astype(np.float32)).mean() > 2.0


def test_rolling_shutter_zero_motion_is_identity(scene: np.ndarray) -> None:
    t = RollingShutter(
        p=1.0, pan_shift_range_px=(0.0, 0.0), wobble_amplitude_range_px=(0.0, 0.0)
    )
    t.set_random_seed(0)
    np.testing.assert_array_equal(t(image=scene)["image"], scene)


# --------------------------------------------------------------------------
# ExposureTransient
# --------------------------------------------------------------------------


def test_exposure_error_is_a_gain_in_linear_light() -> None:
    """One stop is a factor of two in *radiance*, not in pixel value. A mid
    grey pushed up one stop must land where doubling the linear value lands,
    which is nowhere near double the 8-bit code."""
    patch = np.full((32, 32, 3), 110, dtype=np.uint8)
    t = ExposureTransient(
        p=1.0,
        exposure_error_stops_range=(1.0, 1.0),
        assumed_temperature_range_k=(6500.0, 6500.0),
        actual_temperature_range_k=(6500.0, 6500.0),
        tint_error_range=(0.0, 0.0),
    )
    t.set_random_seed(0)
    out = t(image=patch)["image"]

    expected_linear = srgb_to_linear(np.array([110 / 255.0], dtype=np.float32)) * 2.0
    actual_linear = srgb_to_linear(np.array([out[16, 16, 0] / 255.0], dtype=np.float32))
    assert actual_linear[0] == pytest.approx(float(expected_linear[0]), rel=0.02)


def test_white_balance_error_moves_along_the_blue_orange_axis() -> None:
    """AWB estimates a colour *temperature*, so its residual error is the
    ratio of two blackbody colours and must lie on that axis. A warm-biased
    error raises red and lowers blue together; green stays put. Independent
    channel gains would not do that."""
    patch = np.full((32, 32, 3), 128, dtype=np.uint8)
    t = ExposureTransient(
        p=1.0,
        exposure_error_stops_range=(0.0, 0.0),
        assumed_temperature_range_k=(6500.0, 6500.0),
        actual_temperature_range_k=(3000.0, 3000.0),
        tint_error_range=(0.0, 0.0),
    )
    t.set_random_seed(0)
    out = t(image=patch)["image"].astype(np.float32)

    red, green, blue = out[16, 16]
    assert red > 128.0 and blue < 128.0
    assert green == pytest.approx(128.0, abs=2.0)


# --------------------------------------------------------------------------
# WdrToneMapping
# --------------------------------------------------------------------------


def test_wdr_lifts_shadows_and_pulls_highlights_together() -> None:
    """Range compression is the whole point: the dark end must come up and the
    bright end come down, narrowing the spread between them."""
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:64] = 235
    image[64:] = 18

    t = WdrToneMapping(
        p=1.0,
        compression_range=(0.35, 0.35),
        detail_gain_range=(1.0, 1.0),
        halo_strength_range=(0.0, 0.0),
    )
    t.set_random_seed(0)
    out = t(image=image)["image"].astype(np.float32)

    before = float(image[:64].mean() - image[64:].mean())
    after = float(out[:64].mean() - out[64:].mean())
    assert after < before
    assert out[64:].mean() > image[64:].mean()


def test_wdr_halo_appears_only_with_a_leaky_filter() -> None:
    """The halo is a *consequence* of the base layer leaking across an edge, so
    raising `halo_strength` — how much the filter leaks — is what produces it.

    Note the sign. Next to a *bright* region the leaked base is pulled up, and
    since the base is then compressed toward the mean, the dark side comes out
    **darker** than dark areas further away. That is the well-known "black
    halo" of local tone mapping, not a bright fringe; a test asserting a
    brightening here would be asserting the wrong artefact.
    """
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :64] = 30
    image[:, 64:] = 225

    def overshoot(halo: float) -> float:
        t = WdrToneMapping(
            p=1.0,
            compression_range=(0.4, 0.4),
            detail_gain_range=(1.0, 1.0),
            halo_strength_range=(halo, halo),
            filter_radius_range=(0.06, 0.06),
        )
        t.set_random_seed(0)
        out = t(image=image)["image"].astype(np.float32)
        # Dark side just left of the edge, against the dark region far away.
        return float(out[:, 50:62].mean() - out[:, 5:17].mean())

    # No leak, no halo.
    assert overshoot(0.0) == pytest.approx(0.0, abs=0.5)
    # A leaky filter darkens the band beside the bright region.
    assert overshoot(1.0) < -4.0
    assert overshoot(1.0) < overshoot(0.5) < overshoot(0.0)


# --------------------------------------------------------------------------
# TemporalDenoiseArtifacts
# --------------------------------------------------------------------------


def test_denoise_removes_texture_but_keeps_edges() -> None:
    """The defining property, and what separates it from a blur: low-contrast
    texture collapses while a strong edge survives. A Gaussian blur would soften
    both, so a test that only checked 'high frequencies went down' would pass
    for the wrong operation."""
    rng = np.random.default_rng(0)
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    image[:, :80] = 70
    image[:, 80:] = 200
    texture = rng.integers(-12, 13, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + texture, 0, 255).astype(np.uint8)

    t = TemporalDenoiseArtifacts(
        p=1.0,
        strength_range=(1.0, 1.0),
        iterations_range=(3, 3),
        edge_threshold_range=(40.0, 40.0),
        spatial_radius_range=(0.02, 0.02),
    )
    t.set_random_seed(0)
    out = t(image=image)["image"].astype(np.float32)

    # Texture energy well away from the edge.
    def texture_energy(arr: np.ndarray) -> float:
        patch = arr[20:140, 10:60, 0]
        return float(np.abs(np.diff(patch, axis=1)).mean())

    assert texture_energy(out) < 0.5 * texture_energy(image.astype(np.float32))

    # The edge itself is still an edge.
    step = float(out[:, 90:110, 0].mean() - out[:, 50:70, 0].mean())
    original_step = float(image[:, 90:110, 0].mean() - image[:, 50:70, 0].mean())
    assert step > 0.85 * original_step


# --------------------------------------------------------------------------
# NightIrMode
# --------------------------------------------------------------------------


def test_night_mode_is_near_monochrome(scene: np.ndarray) -> None:
    provider = SharedDepthProvider(_RampDepth())
    t = NightIrMode(depth_provider=provider, p=1.0, read_noise_range=(0.0, 0.0))
    t.set_random_seed(0)
    out = t(image=scene)["image"].astype(np.float32)

    spread = (out.max(axis=2) - out.min(axis=2)).mean()
    original_spread = (
        scene.astype(np.float32).max(axis=2) - scene.astype(np.float32).min(axis=2)
    ).mean()
    assert spread < 0.35 * original_spread


def test_foliage_brightens_relative_to_a_luminance_conversion() -> None:
    """Chlorophyll reflects hard past 700 nm, so foliage is among the
    *brightest* things in an IR frame despite being dark green by day. A
    luminance conversion gets this backwards, which is the single most visible
    error in a naive night-mode augmentation."""
    foliage = srgb_to_linear(np.array([[[0.10, 0.28, 0.09]]], dtype=np.float32))
    asphalt = srgb_to_linear(np.array([[[0.26, 0.26, 0.27]]], dtype=np.float32))

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luma_foliage = float((foliage @ weights).ravel()[0])
    luma_asphalt = float((asphalt @ weights).ravel()[0])
    # By luminance the asphalt is the brighter of the two.
    assert luma_asphalt > luma_foliage

    # In the infrared that ordering reverses outright.
    nir_foliage = float(near_infrared_reflectance(foliage, vegetation_gain=0.6).ravel()[0])
    nir_asphalt = float(near_infrared_reflectance(asphalt, vegetation_gain=0.6).ravel()[0])
    assert nir_foliage > nir_asphalt
    assert nir_foliage / nir_asphalt > 3.0 * (luma_foliage / luma_asphalt)


def test_near_surfaces_blow_out_before_far_ones() -> None:
    """The illuminator is at the lens, so irradiance falls as 1/z^2 — which is
    why night footage has such a narrow band of usable distance. On a uniform
    wall the near end must be brighter than the far end."""
    flat = np.full((240, 320, 3), 120, dtype=np.uint8)
    provider = SharedDepthProvider(_RampDepth())
    t = NightIrMode(
        depth_provider=provider,
        p=1.0,
        read_noise_range=(0.0, 0.0),
        beam_falloff_range=(0.0, 0.0),
    )
    t.set_random_seed(0)
    out = t(image=flat)["image"].astype(np.float32)

    near = out[:40].mean()  # top rows are the near end of the ramp
    far = out[-40:].mean()
    assert near > far + 20.0


# --------------------------------------------------------------------------
# VideoCodecArtifacts
# --------------------------------------------------------------------------


@requires_ffmpeg
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_codec_damage_increases_with_crf(scene: np.ndarray, codec: str) -> None:
    """CRF is the quality dial, so damage has to rise monotonically with it —
    otherwise the severity ladder built on it means nothing.

    Note what is deliberately *not* asserted here. 4:2:0 subsampling does make
    chroma degrade faster than luma, and it does on real photographs (measured
    on the reference image: luma retains 0.73 of its detail against chroma's
    0.24). But it does not hold on synthetic test content, where the channels
    are uncorrelated and chroma carries as much genuine high-frequency
    structure as luma does — so pinning it here would be pinning a property of
    the fixture rather than of the codec. It is documented in the module
    instead.
    """
    damage = []
    for crf in (24, 34, 44):
        t = VideoCodecArtifacts(
            p=1.0, codec=codec, crf_range=(crf, crf), stale_block_probability=0.0
        )
        t.set_random_seed(0)
        out = t(image=scene)["image"].astype(np.float32)
        damage.append(float(np.abs(out - scene.astype(np.float32)).mean()))

    assert damage[0] < damage[1] < damage[2], damage


@requires_ffmpeg
def test_codec_stale_blocks_are_macroblock_aligned(scene: np.ndarray) -> None:
    """A decoder holds whole macroblocks stale, so the boundary between fresh
    and stale content lands on the 16-pixel grid. That hard block-aligned
    edge is what a detector trips over, and it is what distinguishes this from
    a soft blend."""
    clean = VideoCodecArtifacts(p=1.0, crf_range=(30, 30), stale_block_probability=0.0)
    clean.set_random_seed(0)
    baseline = clean(image=scene)["image"].astype(np.float32)

    stale = VideoCodecArtifacts(
        p=1.0,
        crf_range=(30, 30),
        stale_block_probability=1.0,
        stale_block_fraction_range=(0.4, 0.4),
        stale_crf_delta_range=(18, 18),
    )
    stale.set_random_seed(0)
    damaged = stale(image=scene)["image"].astype(np.float32)

    difference = np.abs(damaged - baseline).mean(axis=2)
    assert difference.max() > 2.0, "stale blocks made no difference"

    # Column-to-column change should concentrate on the macroblock grid.
    column_step = np.abs(np.diff(difference, axis=1)).mean(axis=0)
    on_grid = column_step[MACROBLOCK_PX - 1 :: MACROBLOCK_PX].mean()
    off_grid = np.delete(
        column_step, np.arange(MACROBLOCK_PX - 1, len(column_step), MACROBLOCK_PX)
    ).mean()
    assert on_grid > 1.5 * off_grid


@requires_ffmpeg
def test_codec_is_not_jpeg(scene: np.ndarray) -> None:
    """Sanity check on the premise of the whole transform: at comparable
    damage, H.264 and JPEG do not produce the same image. If they did, the
    ffmpeg dependency would not be worth carrying."""
    import cv2

    t = VideoCodecArtifacts(p=1.0, crf_range=(42, 42), stale_block_probability=0.0)
    t.set_random_seed(0)
    codec_out = t(image=scene)["image"].astype(np.float32)

    ok, buffer = cv2.imencode(".jpg", scene, [int(cv2.IMWRITE_JPEG_QUALITY), 12])
    assert ok
    jpeg_out = cv2.imdecode(buffer, cv2.IMREAD_COLOR).astype(np.float32)

    assert np.abs(codec_out - jpeg_out).mean() > 3.0


def test_flicker_harmonics_all_cancel_at_an_anti_flicker_exposure() -> None:
    """Modelling the lamp's harmonics must not break the cancellation.

    Each harmonic is box-filtered by the exposure independently, so each
    carries its own ``sinc(n * T * f)``. When ``T * f`` is a whole number so
    is every multiple of it, so the entire profile has to vanish together —
    not just the fundamental. A model that left harmonics ringing at an
    anti-flicker exposure would contradict the reason that setting exists.
    """
    scene = np.full((96, 128, 3), 120, dtype=np.uint8)
    flicker_hz = 100.0
    for periods in (1, 2, 3):
        exposure = periods / flicker_hz
        t = AcFlicker(
            p=1.0,
            mains_frequency_hz=flicker_hz / 2.0,
            exposure_time_range_s=(exposure, exposure),
            modulation_depth_range=(0.6, 0.6),
            pulse_exponent_range=(5.0, 5.0),  # strongly harmonic lamp
        )
        t.set_random_seed(0)
        np.testing.assert_array_equal(t(image=scene)["image"], scene)


def test_flicker_lamp_waveform_harmonics_match_the_named_lamp_types() -> None:
    """The exponent is the only knob for waveform shape, so it has to land on
    the right physics at the values the docstring names."""
    from weathermentations.camera.flicker import lamp_waveform_harmonics

    # sin^2 *is* a sinusoid at the flicker frequency: no harmonics at all.
    sinusoid = lamp_waveform_harmonics(2.0)
    assert sinusoid[0] == pytest.approx(1.0)
    assert np.allclose(sinusoid[1:], 0.0, atol=1e-6)

    # A discharge lamp and an LED pulse both carry harmonics; the pulse more.
    discharge = lamp_waveform_harmonics(1.0)
    led = lamp_waveform_harmonics(6.0)
    assert discharge[1] > 0.1
    assert led[1] > discharge[1]

    # Normalised so peak modulation equals the caller's nominal depth.
    for exponent in (1.0, 2.0, 4.0, 6.0):
        assert lamp_waveform_harmonics(exponent).sum() == pytest.approx(1.0, abs=1e-6)


def test_flicker_band_count_follows_readout_time() -> None:
    """Band count, not depth, is what makes the artefact legible — two broad
    bands read as an exposure difference however deep they are. It must
    follow `flicker_hz * readout` rather than being a free parameter."""
    scene = np.full((256, 64, 3), 120, dtype=np.uint8)
    counts = []
    for readout in (0.02, 0.06):
        t = AcFlicker(
            p=1.0,
            readout_time_range_s=(readout, readout),
            exposure_time_range_s=(1e-4, 1e-4),
            modulation_depth_range=(0.3, 0.3),
            pulse_exponent_range=(2.0, 2.0),
        )
        t.set_random_seed(0)
        profile = t(image=scene)["image"].astype(np.float32).mean(axis=(1, 2))
        centred = profile - profile.mean()
        counts.append(int(np.sum(np.diff(np.sign(centred)) != 0) / 2))
    assert counts[0] == pytest.approx(2, abs=1), counts
    assert counts[1] == pytest.approx(6, abs=1), counts


# --- RollingShutter moves annotations with the pixels --------------------


def _annotated_scene() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    mask = np.zeros((240, 320), dtype=np.uint8)
    boxes = [(60, 40, 120, 200), (200, 100, 260, 230), (10, 5, 40, 30)]
    for index, (x0, y0, x1, y1) in enumerate(boxes):
        image[y0:y1, x0:x1] = 60 + 60 * index
        mask[y0:y1, x0:x1] = index + 1
    return image, mask, boxes


@pytest.mark.parametrize(
    "motion",
    [
        pytest.param(
            {"pan_shift_range_px": (30.0, 30.0), "wobble_amplitude_range_px": (0.0, 0.0),
             "vertical_fraction_range": (0.0, 0.0)},
            id="shear",
        ),
        pytest.param(
            {"pan_shift_range_px": (-25.0, -25.0), "wobble_amplitude_range_px": (10.0, 10.0),
             "wobble_cycles_range": (3.0, 3.0), "vertical_fraction_range": (0.35, 0.35)},
            id="wobble-with-vertical-motion",
        ),
    ],
)
def test_rolling_shutter_moves_masks_boxes_and_keypoints_with_the_image(
    motion: dict[str, tuple[float, float]],
) -> None:
    """The one transform here that moves pixels has to move their labels.

    It was image-only like every appearance effect, so a 30 px shear moved the
    picture 29 px and left the mask and the boxes exactly where they were —
    silently wrong labels for anyone training a detector with it.
    """
    import albumentations as A

    image, mask, boxes = _annotated_scene()
    corners = [(60.0, 40.0), (119.0, 199.0), (230.0, 165.0)]
    pipeline = A.Compose(
        [RollingShutter(p=1.0, **motion)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        seed=1,
    )
    out = pipeline(image=image, mask=mask, bboxes=boxes, labels=[1, 2, 3], keypoints=corners)

    assert not np.array_equal(out["image"], image)
    # The mask still covers the objects, wherever they went.
    agreement = ((out["image"][..., 0] > 30) == (out["mask"] > 0)).mean()
    assert agreement > 0.995, f"image and mask agree on only {agreement:.1%} of pixels"
    assert set(np.unique(out["mask"])) <= {0, 1, 2, 3}, "mask labels were interpolated"

    # Each box encloses its object's new extent, to the pixel.
    for box, label in zip(out["bboxes"], out["labels"], strict=True):
        ys, xs = np.nonzero(out["mask"] == label)
        truth = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)
        np.testing.assert_allclose(np.asarray(box[:4], dtype=np.float64), truth, atol=1.5)

    # A keypoint on an object's corner lands on that object's moved corner.
    x, y = out["keypoints"][0][:2]
    ys, xs = np.nonzero(out["mask"] == 1)
    assert abs(y - ys.min()) <= 1.5
    assert abs(x - xs[ys == ys.min()].min()) <= 1.5


def test_rolling_shutter_with_no_motion_leaves_annotations_alone() -> None:
    import albumentations as A

    image, mask, boxes = _annotated_scene()
    still = {
        "pan_shift_range_px": (0.0, 0.0),
        "wobble_amplitude_range_px": (0.0, 0.0),
    }
    pipeline = A.Compose(
        [RollingShutter(p=1.0, **still)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        seed=0,
    )
    out = pipeline(image=image, mask=mask, bboxes=boxes, labels=[1, 2, 3])
    np.testing.assert_array_equal(out["mask"], mask)
    np.testing.assert_allclose(np.asarray(out["bboxes"])[:, :4], np.asarray(boxes), atol=1e-3)


def test_wdr_base_filter_radius_never_exceeds_its_cap() -> None:
    """The bilateral filter costs O(radius^2) per pixel, and the cap on its
    radius is the only thing keeping the transform inside a frame budget.

    The downsampling factor used to be rounded down, which left every radius
    between one cap and two filtered at full resolution: a 24 px radius at
    512 px took two seconds against 0.12 s. Half of the draws at that size
    fell in the gap, so the transform was ten times slower than documented
    for no change in what it rendered.
    """
    from weathermentations.camera.tone_mapping import (
        _BASE_FILTER_MAX_RADIUS_PX,
        _base_filter_scale,
    )

    for radius in np.linspace(2.0, 400.0, 800):
        factor, at_scale = _base_filter_scale(float(radius))
        assert at_scale <= _BASE_FILTER_MAX_RADIUS_PX + 1e-6, (radius, factor, at_scale)
        # ...without downsampling further than the cap requires.
        assert factor == 1 or radius / (factor - 1) > _BASE_FILTER_MAX_RADIUS_PX
