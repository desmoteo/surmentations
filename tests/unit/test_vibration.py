"""CameraVibration: a smear as long as the camera moved meanwhile, and a frame moved with it."""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest

from weathermentations.camera import CameraVibration
from weathermentations.camera.vibration import swing_path


def _extent(kernel: np.ndarray, axis: int = 1) -> float:
    """Second moment of a kernel along an axis, as a full width: how long the smear is."""
    profile = kernel.sum(axis=1 - axis)
    at = np.arange(len(profile)) - (len(profile) - 1) / 2.0
    mean = float((profile * at).sum())
    return float(np.sqrt((profile * (at - mean) ** 2).sum()) * np.sqrt(12.0))


def _shake(**settings: object) -> CameraVibration:
    """A horizontal 5 Hz swing with every range pinned unless changed."""
    arguments: dict[str, object] = {
        "amplitude_range": (0.02, 0.02),
        "frequency_range_hz": (5.0, 5.0),
        "exposure_time_range": (0.01, 0.01),
        "direction_range_deg": (0.0, 0.0),
        **settings,
    }
    return CameraVibration(p=1.0, **arguments)


def test_the_kernel_sums_to_one_and_is_centred_on_the_mean() -> None:
    for cycles in (0.01, 0.2, 0.5, 1.0, 3.3):
        kernel, _ = swing_path(8.0, cycles, 0.4, 0.3)
        assert kernel.sum() == pytest.approx(1.0, abs=1e-5)
        rows, columns = np.indices(kernel.shape)
        centre = (kernel.shape[0] - 1) / 2.0
        assert float((kernel * (columns - centre)).sum()) == pytest.approx(0.0, abs=0.05)
        assert float((kernel * (rows - centre)).sum()) == pytest.approx(0.0, abs=0.05)


def test_the_smear_grows_with_the_shutter_time_until_it_holds_the_whole_swing() -> None:
    # Through the middle of the swing, where the camera moves fastest and
    # evenly: 2 pi amplitude cycles pixels of travel.
    short = _extent(swing_path(20.0, 0.05, 0.0, 0.0)[0])
    longer = _extent(swing_path(20.0, 0.1, 0.0, 0.0)[0])
    assert short == pytest.approx(2.0 * np.pi * 20.0 * 0.05, rel=0.25)
    assert longer == pytest.approx(2.0 * short, rel=0.15)
    # A whole cycle and ten of them are the same picture: the full swing.
    whole = _extent(swing_path(20.0, 1.0, 0.0, 0.0)[0])
    many = _extent(swing_path(20.0, 10.0, 0.0, 0.0)[0])
    assert many == pytest.approx(whole, rel=0.05)
    assert whole > 30.0


def test_a_swing_spends_longest_at_its_ends() -> None:
    kernel, _ = swing_path(20.0, 1.0, 0.0, 0.0)
    profile = kernel.sum(axis=0)
    middle = len(profile) // 2
    assert profile[middle - 18 : middle - 14].mean() > 1.5 * profile[middle - 2 : middle + 2].mean()


def test_a_short_shutter_displaces_the_frame_and_does_not_smear_it() -> None:
    # A quarter of the way round, the camera is at the end of its swing.
    kernel, shift = swing_path(12.0, 1e-4, 0.5 * np.pi, 0.0)
    assert kernel.shape == (3, 3)
    assert shift[0] == pytest.approx(12.0, abs=0.01)
    assert shift[1] == pytest.approx(0.0, abs=1e-6)


def test_the_smear_lies_along_the_swing() -> None:
    kernel, _ = swing_path(15.0, 0.5, 0.0, 0.5 * np.pi)
    assert _extent(kernel, axis=0) > 10.0 * max(_extent(kernel, axis=1), 0.1)


def test_boxes_keypoints_and_masks_move_with_the_frame() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[40:60, 50:70] = 255
    mask = (image[..., 0] > 0).astype(np.uint8)
    pipeline = A.Compose(
        [_shake(exposure_time_range=(1e-4, 1e-4))],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        keypoint_params=A.KeypointParams(format="xy"),
        seed=3,
    )
    out = pipeline(
        image=image, mask=mask, bboxes=[(50, 40, 70, 60)], labels=[1], keypoints=[(60.0, 50.0)]
    )
    ys, xs = np.nonzero(out["mask"])
    moved_x = xs.mean() - 59.5
    assert abs(moved_x) > 0.5
    box = out["bboxes"][0]
    assert box[0] - 50 == pytest.approx(moved_x, abs=0.6)
    assert box[2] - 70 == pytest.approx(moved_x, abs=0.6)
    assert out["keypoints"][0][0] - 60.0 == pytest.approx(moved_x, abs=0.6)
    bright = np.nonzero(out["image"][..., 0] > 127)[1]
    assert bright.mean() - 59.5 == pytest.approx(moved_x, abs=0.6)
    assert box[1] == pytest.approx(40.0, abs=0.01) and ys.min() == 40


def test_depth_is_not_carried_past_a_camera_that_moved() -> None:
    assert CameraVibration.preserves_scene_geometry is False


def test_a_still_camera_changes_nothing() -> None:
    image = np.random.default_rng(0).integers(0, 256, (60, 80, 3), dtype=np.uint8)
    still = _shake(amplitude_range=(0.0, 0.0))
    still.set_random_seed(0)
    assert np.array_equal(still(image=image)["image"], image)


def test_the_frame_keeps_its_light() -> None:
    import cv2

    # Smooth, as a photograph is: averaging noise in linear light raises
    # its level in code values by itself.
    rng = np.random.default_rng(1)
    small = rng.random((8, 10, 3)).astype(np.float32) * 0.6 + 0.2
    image = cv2.resize(small, (100, 80), interpolation=cv2.INTER_CUBIC)
    shake = _shake(exposure_time_range=(0.04, 0.04))
    shake.set_random_seed(2)
    out = shake(image=image)["image"]
    inner = (slice(20, -20), slice(30, -30))
    assert float(out[inner].mean()) == pytest.approx(float(image[inner].mean()), rel=0.03)
