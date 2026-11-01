"""Blurs and dilations computed on a reduced copy of the frame.

They stand in for filters whose cost grows with their reach, and are allowed
to differ from them: by how much, and where, is what is checked here.
"""

from __future__ import annotations

import cv2
import numpy as np

from weathermentations.utils.sky import wide_blur, wide_dilate


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
