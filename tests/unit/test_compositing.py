import numpy as np

from weathermentations.utils.compositing import (
    alpha_composite,
    directional_blur_kernel,
    feathered_band_blend,
)


def test_alpha_composite_full_alpha_is_overlay() -> None:
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    out = alpha_composite(base, (200, 100, 50), np.ones((4, 4), dtype=np.float32))
    np.testing.assert_array_equal(out, np.full((4, 4, 3), (200, 100, 50), dtype=np.uint8))


def test_alpha_composite_zero_alpha_is_identity() -> None:
    base = (np.random.default_rng(0).random((4, 4, 3)) * 255).astype(np.uint8)
    out = alpha_composite(base, (200, 100, 50), np.zeros((4, 4), dtype=np.float32))
    np.testing.assert_array_equal(out, base)


def test_directional_blur_kernel_is_normalized() -> None:
    kernel = directional_blur_kernel(9, 15.0)
    assert kernel.shape == (9, 9)
    np.testing.assert_allclose(kernel.sum(), 1.0, atol=1e-5)


def test_feathered_band_blend_only_touches_band() -> None:
    img = np.random.default_rng(0).uniform(0, 255, size=(20, 10, 3)).astype(np.float32)
    out = feathered_band_blend(img.copy(), band_fraction=0.3, blend=lambda band: band * 0.0)
    band_start = int(20 * 0.7)
    np.testing.assert_array_equal(out[:band_start], img[:band_start])
    assert not np.array_equal(out[band_start:], img[band_start:])


def test_feathered_band_blend_starts_at_original_and_ends_at_target() -> None:
    img = np.full((20, 10, 3), 100.0, dtype=np.float32)
    out = feathered_band_blend(img.copy(), band_fraction=0.5, blend=lambda band: band * 0.0)
    band_start = int(20 * 0.5)
    # First row of the band should be ~unchanged (feather starts at 0)...
    np.testing.assert_allclose(out[band_start], img[band_start], atol=1.0)
    # ...and the last row should be ~fully blended toward the target (0).
    np.testing.assert_allclose(out[-1], np.zeros((10, 3)), atol=1.0)
