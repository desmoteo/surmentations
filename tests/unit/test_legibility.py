"""A label is kept while its object can be made out, and dealt with when it cannot.

The filter errs in one direction: it makes a label background only where
the object's pixels were replaced, and otherwise can only suppress, remove
or blank it.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
from albumentations import BboxParams, KeypointParams

from surmentations import legibility
from surmentations.atmosphere.fog import RandomFog
from surmentations.base import CameraTransform, Compose
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import recording
from surmentations.legibility import LabelLegibility, _Record, task_probability
from surmentations.lens.tamper import LensTamper

SIZE = 256
BOX = (60, 60, 108, 108)
SIDE = BOX[2] - BOX[0]


class _Wall(DepthEstimator):
    """Everything at the far plane, with one near value so the range does not collapse."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """A wall at the far plane."""
        depth = np.ones(image.shape[:2], dtype=np.float32)
        depth[0, 0] = 0.0
        return depth


class _Night(CameraTransform):
    """A frame made darker and grainier by fixed amounts, as a night made of a day is."""

    def __init__(self, gain: float, noise: float, p: float = 1.0) -> None:
        """Keep the gain on the light and the noise's deviation, in levels."""
        super().__init__(p=p)
        self.gain = gain
        self.noise = noise

    def apply(self, img: np.ndarray, **params: object) -> np.ndarray:
        """Darken by ``gain`` and add the noise, from a fixed seed."""
        grain = np.random.default_rng(5).normal(0.0, self.noise, img.shape)
        return np.clip(img.astype(np.float32) * self.gain + grain, 0, 255).astype(np.uint8)


def _scene() -> dict:
    """A grey frame with a checkerboard object in BOX, and every kind of label on it."""
    image = np.full((SIZE, SIZE, 3), 110, dtype=np.uint8)
    x0, y0, x1, y1 = BOX
    yy, xx = np.mgrid[y0:y1, x0:x1]
    board = ((yy - y0) // 6 + (xx - x0) // 6) % 2 == 0
    image[y0:y1, x0:x1] = np.where(board[..., None], 200, 40).astype(np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 3
    return {
        "image": image,
        "bboxes": [list(BOX)],
        "labels": [7],
        "mask": mask,
        "keypoints": [((x0 + x1) / 2, (y0 + y1) / 2)],
    }


def _pipeline(*transforms: CameraTransform, **filter_args: object) -> Compose:
    """``transforms`` and then the filter, with boxes and keypoints carried."""
    return Compose(
        [*transforms, LabelLegibility(**filter_args)],
        bbox_params=BboxParams(format="pascal_voc", label_fields=["labels"]),
        keypoint_params=KeypointParams(format="xy"),
        seed=1,
    )


def _fog(beta: float) -> RandomFog:
    """Fog of extinction ``beta`` per metre over a wall 50 m away."""
    provider = SharedDepthProvider(_Wall(), depth_range_m=(3.0, 50.0))
    return RandomFog(depth_provider=provider, p=1.0, beta_range=(beta, beta))


def _judged(log: list[dict]) -> dict:
    """What the filter recorded of its last call."""
    return [record for record in log if record["t"] == "LabelLegibility"][-1]["drew"]


def test_johnson_s_curve_is_half_at_the_task_s_cycles_and_rises() -> None:
    assert task_probability(4.0, 4.0) == pytest.approx(0.5)
    assert task_probability(8.0, 4.0) > 0.9 > 0.5 > task_probability(2.0, 4.0) > 0.0
    assert task_probability(0.0, 4.0) == 0.0


def test_planes_fold_in_as_a_union_and_a_quadrature_sum_from_any_shape() -> None:
    """A cover combines as independent covers do, a blur in quadrature; a
    plane may come per channel or as a scalar, at any size."""
    record = _Record(np.zeros((64, 64, 3), dtype=np.uint8))
    record.add({"hidden": np.full((64, 64), 0.5, np.float32)}, (64, 64))
    record.add({"hidden": np.full((32, 32, 3), 0.5, np.float32)}, (64, 64))
    record.add({"blur": 3.0}, (64, 64))
    record.add({"blur": np.full((16, 16), 4.0, np.float32)}, (64, 64))
    assert record.hidden.shape == (16, 16) and record.hidden[0, 0] == pytest.approx(0.75)
    assert record.blur[0, 0] == pytest.approx(5.0)
    with pytest.raises(KeyError):
        record.add({"contrast": 0.5}, (64, 64))


def test_a_clean_frame_keeps_every_label() -> None:
    data = _scene()
    out = _pipeline()(**data)
    assert len(out["bboxes"]) == 1 and len(out["keypoints"]) == 1
    np.testing.assert_array_equal(out["mask"], data["mask"])
    np.testing.assert_array_equal(out["image"], data["image"])
    assert out["ignore_regions"].shape == (0, 4)


def test_a_lightly_veiled_object_is_kept() -> None:
    out = _pipeline(_fog(0.002))(**_scene())
    assert len(out["bboxes"]) == 1 and len(out["keypoints"]) == 1
    assert (out["mask"] == 3).sum() == SIDE * SIDE


def test_an_object_made_dark_and_grainy_but_still_plain_is_kept() -> None:
    """A night made of a day darkens everything and adds grain; an object
    whose edges still stand well clear of the grain is as legible as it was.
    Judged by how its edges compare with the clean frame's, without a floor
    on the noise, a bus in an infrared night was suppressed."""
    out = _pipeline(_Night(gain=0.2, noise=3.0))(**_scene())
    assert len(out["bboxes"]) == 1 and len(out["keypoints"]) == 1


def test_an_object_drowned_in_grain_is_suppressed() -> None:
    out = _pipeline(_Night(gain=0.05, noise=40.0))(**_scene())
    assert len(out["bboxes"]) == 0


def test_a_covered_object_is_background_for_every_kind_of_label() -> None:
    """Tape over the whole lens replaces every pixel: the box and the point
    go, the mask is background, nothing is suppressed, and the filter says
    so in its record."""
    cover = LensTamper(p=1.0, kinds=("cover",), coverage_range=(1.0, 1.0))
    with recording() as log:
        out = _pipeline(cover)(**_scene())
    assert len(out["bboxes"]) == 0 and len(out["keypoints"]) == 0
    assert not (out["mask"] == 3).any() and not (out["mask"] == 255).any()
    assert out["ignore_regions"].shape == (0, 4)
    judged = _judged(log)
    assert judged["original_known"] is True
    assert judged["background"] == 3 and judged["kept"] == 0


def test_an_object_lost_in_fog_is_suppressed_by_default() -> None:
    """Suppressed: the label goes, its region comes back for the loss to
    leave out, and a semantic mask's pixels are marked to be ignored."""
    with recording() as log:
        out = _pipeline(_fog(0.3))(**_scene())
    assert len(out["bboxes"]) == 0 and len(out["keypoints"]) == 0
    assert (out["mask"] == 255).sum() == SIDE * SIDE and not (out["mask"] == 3).any()
    regions = out["ignore_regions"]
    assert regions.shape == (2, 4), "the box's region and the keypoint's"
    np.testing.assert_allclose(regions[0], BOX, atol=1.0)
    judged = _judged(log)
    assert judged["background"] == 0, "washed out is never taken to be gone"
    assert judged["suppressed"] == 3


def test_an_object_lost_in_fog_is_removed_or_kept_as_asked() -> None:
    dropped = _pipeline(_fog(0.3), on_doubt="drop")(**_scene())
    assert len(dropped["bboxes"]) == 0 and dropped["ignore_regions"].shape == (0, 4)
    assert not (dropped["mask"] == 3).any() and not (dropped["mask"] == 255).any()

    kept = _pipeline(_fog(0.3), on_doubt="keep")(**_scene())
    assert len(kept["bboxes"]) == 1 and len(kept["keypoints"]) == 1
    assert (kept["mask"] == 3).sum() == SIDE * SIDE


def test_blanking_paints_the_object_out_with_its_surroundings() -> None:
    """Asked to blank, the filter takes the object out of the picture too, so
    that the background it leaves is true: the fill is the surroundings'
    grey, not a blur of the object's own pattern."""
    data = _scene()
    out = _pipeline(_fog(0.3), on_doubt="blank")(**data)
    x0, y0, x1, y1 = BOX
    inside = out["image"][y0 + 4 : y1 - 4, x0 + 4 : x1 - 4].astype(np.float32)
    around = out["image"][y0 - 20 : y0 - 8, x0:x1].astype(np.float32)
    assert len(out["bboxes"]) == 0 and out["ignore_regions"].shape == (0, 4)
    assert inside.std() < 2.0
    assert abs(float(inside.mean()) - float(around.mean())) < 3.0


def test_blanking_leaves_a_kept_neighbour_alone() -> None:
    """Painting out one object does not damage another whose label is kept."""
    from surmentations.legibility import _blank

    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    image[10:30, 10:30] = 200
    protected = np.zeros((64, 64), dtype=bool)
    protected[20:30, 20:30] = True
    out = _blank(image, [(None, (10, 10, 30, 30))], protected)
    assert (out[20:30, 20:30] == 200).all()
    assert out[12:18, 12:18].mean() < 150


def test_the_filter_alone_changes_nothing_it_was_not_told_about() -> None:
    """With no transform before it there is no record: every label stays."""
    data = _scene()
    out = LabelLegibility()(image=data["image"], mask=data["mask"])
    np.testing.assert_array_equal(out["mask"], data["mask"])
    np.testing.assert_array_equal(out["image"], data["image"])


def test_calls_are_noted_only_while_a_filter_exists() -> None:
    """Every transform's call is noted while a filter is alive, and none once
    it is gone: a pipeline without one pays nothing."""
    gc.collect()
    assert not legibility.tracking()
    the_filter = LabelLegibility()
    assert legibility.tracking()
    del the_filter
    gc.collect()
    assert not legibility.tracking()


def test_the_task_is_a_name_or_a_number() -> None:
    assert LabelLegibility(task="identify").needed_cycles == pytest.approx(6.4)
    assert LabelLegibility(task=2.5).needed_cycles == 2.5
    with pytest.raises(ValueError):
        LabelLegibility(task="read")
