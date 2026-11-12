"""FocusShift: a blur disc whose size is the thin-lens formula's, per pixel where depth is known."""

from __future__ import annotations

import numpy as np
import pytest

from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.lens import FocusShift
from weathermentations.lens.focus import blur_by_depth, blur_diameter

SHAPE = (120, 160)


class _NearLeftFarRight(DepthEstimator):
    """The left half of the frame at the provider's near distance, the right at its far one."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Relative depth 0 over the left half and 1 over the right."""
        height, width = image.shape[:2]
        depth = np.zeros((height, width), dtype=np.float32)
        depth[:, width // 2 :] = 1.0
        return depth


def _texture() -> np.ndarray:
    """Fine detail everywhere: a checkerboard two pixels to a square."""
    rows, columns = np.indices(SHAPE)
    board = (((rows // 2) + (columns // 2)) % 2 * 200 + 20).astype(np.uint8)
    return np.repeat(board[..., None], 3, axis=2)


def _detail(image: np.ndarray) -> np.ndarray:
    """Mean difference between neighbouring pixels in each column: what is left of the board."""
    return np.abs(np.diff(image[..., 0].astype(np.float32), axis=0)).mean(axis=0)


def _lens(**settings: object) -> FocusShift:
    """A 2 mm stop behind a long lens, 607 px at this size, focused at 0.5 m unless changed."""
    arguments: dict[str, object] = {
        "focus_distance_range_m": (0.5, 0.5),
        "aperture_diameter_range_mm": (2.0, 2.0),
        "horizontal_fov_range_deg": (15.0, 15.0),
        **settings,
    }
    return FocusShift(p=1.0, **arguments)


def test_the_disc_is_the_thin_lens_formula_s() -> None:
    # 2 mm stop, 1000 px focal length, focused at 0.5 m.
    assert blur_diameter(np.inf, 0.5, 2e-3, 1000.0) == pytest.approx(4.0)
    assert blur_diameter(0.5, 0.5, 2e-3, 1000.0) == pytest.approx(0.0, abs=1e-5)
    assert blur_diameter(1.0, 0.5, 2e-3, 1000.0) == pytest.approx(2.0)
    # Nearer than the focus it grows without limit, and beyond a few times
    # the focus distance it has all but reached its far value.
    assert blur_diameter(0.25, 0.5, 2e-3, 1000.0) == pytest.approx(4.0)
    assert blur_diameter(10.0, 0.5, 2e-3, 1000.0) == pytest.approx(3.8)


def test_focusing_nearer_and_opening_the_lens_both_blur_more() -> None:
    image = _texture()

    def left(**settings: object) -> float:
        """Detail left after the blur."""
        lens = _lens(**settings)
        lens.set_random_seed(0)
        return float(_detail(lens(image=image)["image"]).mean())

    sharp = float(_detail(image).mean())
    assert left(focus_distance_range_m=(0.2, 0.2)) < left(focus_distance_range_m=(0.5, 0.5)) < sharp
    assert left(aperture_diameter_range_mm=(4.0, 4.0)) < left(aperture_diameter_range_mm=(2.0, 2.0))


def test_without_depth_the_frame_is_blurred_evenly() -> None:
    lens = _lens()
    lens.set_random_seed(0)
    detail = _detail(lens(image=_texture())["image"])[8:-8]
    assert float(detail.std() / detail.mean()) < 0.1


def test_with_depth_what_stands_at_the_focus_distance_stays_sharp() -> None:
    provider = SharedDepthProvider(_NearLeftFarRight(), depth_range_m=(0.5, 20.0))
    lens = _lens(depth_provider=provider, focus_distance_range_m=(0.5, 0.5))
    lens.set_random_seed(0)
    image = _texture()
    detail = _detail(lens(image=image)["image"])
    sharp = float(_detail(image).mean())
    # The left half of the scene is at the focus distance; the right half
    # is far beyond it.
    assert detail[4:70].mean() > 0.9 * sharp
    assert detail[-70:-4].mean() < 0.5 * sharp


def test_blur_by_depth_matches_one_blur_where_depth_does_not_vary() -> None:
    rng = np.random.default_rng(0)
    linear = rng.random((60, 80, 3)).astype(np.float32)
    even = blur_by_depth(linear, np.full((60, 80), 3.0, dtype=np.float32))
    ramp = np.tile(np.linspace(1.0, 5.0, 80, dtype=np.float32), (60, 1))
    varied = blur_by_depth(linear, ramp)
    # Where the ramp passes through 3 px the two agree.
    assert np.abs(varied[:, 39:41] - even[:, 39:41]).mean() < 0.02
    assert np.abs(varied[:, :5] - even[:, :5]).mean() > 0.02


def test_a_lens_focused_far_away_changes_nothing() -> None:
    image = _texture()
    lens = _lens(focus_distance_range_m=(500.0, 500.0))
    lens.set_random_seed(0)
    assert np.abs(lens(image=image)["image"].astype(int) - image.astype(int)).max() <= 1
