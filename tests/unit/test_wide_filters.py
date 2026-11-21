"""Blurs and dilations computed on a reduced copy of the frame.

They stand in for filters whose cost grows with their reach, and are allowed
to differ from them: by how much, and where, is what is checked here.
"""

from __future__ import annotations

import cv2
import numpy as np

from surmentations import mode
from surmentations.utils.filters import blur_of_lights, wide_blur, wide_dilate


def test_a_wide_blur_is_the_blur_it_stands_in_for() -> None:
    rng = np.random.default_rng(13)
    image = rng.uniform(size=(240, 320, 3)).astype(np.float32)
    image[:80] *= 0.2
    exact = cv2.GaussianBlur(image, (0, 0), sigmaX=40.0)
    assert np.abs(wide_blur(image, 40.0) - exact).max() < 0.01 * exact.max()
    # Too narrow to be worth reducing: the plain blur, bit for bit.
    assert np.array_equal(wide_blur(image, 6.0), cv2.GaussianBlur(image, (0, 0), sigmaX=6.0))


def test_a_wide_dilation_differs_only_where_the_reach_ends() -> None:
    mask = np.zeros((240, 320), dtype=np.float32)
    mask[:70] = 1.0
    mask[150:170, 200:230] = 0.6
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    exact = cv2.dilate(mask, disc)
    grown = wide_dilate(mask, 41)
    assert grown.shape == mask.shape
    # Everything the mask held is still held, and well inside or well
    # outside the reach the two agree; they differ in a band at its edge.
    assert (grown >= mask - 1e-6).all()
    block = 41 // 9
    band = cv2.dilate(exact, np.ones((3, 3), np.uint8), iterations=2 * block) != cv2.erode(
        exact, np.ones((3, 3), np.uint8), iterations=2 * block
    )
    assert np.abs(grown - exact)[~band].max() < 1e-3
    assert band.mean() < 0.25
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    assert np.array_equal(wide_dilate(mask, 11), cv2.dilate(mask, small))


def _lamps(count: int, channels: int, seed: int = 3) -> np.ndarray:
    """A dark field with a few lamps in its lower half, some at the frame's edge."""
    rng = np.random.default_rng(seed)
    shape = (240, 320) if channels == 1 else (240, 320, channels)
    field = np.zeros(shape, dtype=np.float32)
    for _ in range(count):
        row, column = int(rng.integers(120, 236)), int(rng.integers(0, 314))
        field[row : row + 4, column : column + 6] = rng.uniform(0.5, 6.0, size=channels)
    return field


def test_a_blur_of_lights_is_the_blur_of_the_whole_frame_to_the_bit() -> None:
    """Taken only on the box the lamps are in, with the filter's reach
    round it: in exact mode the very values of the blur of the frame."""
    with mode.exact():
        for channels in (1, 3):
            for count in (1, 2, 12):
                field = _lamps(count, channels)
                for sigma in (0.7, 3.3, 10.9):
                    whole = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma)
                    assert np.array_equal(blur_of_lights(field, sigma), whole)
        nothing = np.zeros((40, 60, 3), dtype=np.float32)
        assert not blur_of_lights(nothing, 5.0).any()
        everywhere = np.random.default_rng(1).uniform(size=(40, 60)).astype(np.float32)
        assert np.array_equal(
            blur_of_lights(everywhere, 2.0), cv2.GaussianBlur(everywhere, (0, 0), sigmaX=2.0)
        )


def test_a_blur_of_lights_in_fast_mode_is_within_a_hundredth_or_two_of_the_halo() -> None:
    """A wide halo is blurred on a smaller copy in fast mode; a narrow one
    is not, and is still the blur to the bit."""
    was = mode.fast()
    mode.set_fast(True)
    try:
        field = _lamps(12, 3)
        wide = cv2.GaussianBlur(field, (0, 0), sigmaX=10.9)
        assert np.abs(blur_of_lights(field, 10.9) - wide).max() < 0.015 * wide.max()
        near = cv2.GaussianBlur(field, (0, 0), sigmaX=3.3)
        assert np.array_equal(blur_of_lights(field, 3.3), near)
    finally:
        mode.set_fast(was)
