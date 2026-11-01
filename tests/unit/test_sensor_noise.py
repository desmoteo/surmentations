"""Sensor noise has to behave like counted electrons, not like added grain.

Each test is a property that per-pixel Gaussian noise of a fixed strength —
the usual stand-in — does not have.
"""

from __future__ import annotations

import numpy as np
import pytest

from weathermentations.camera.sensor_noise import (
    SensorNoise,
    _one_colour,
    bayer_sites,
    demosaic,
    photon_transfer_noise,
)


def _noise(level: float, gain_db: float, read_noise: float = 2.0, size: int = 256) -> np.ndarray:
    """Noise drawn for a flat frame at ``level``, with the given gain in dB and read noise."""
    image = np.full((size, size, 3), level, dtype=np.float32)
    rng = np.random.default_rng(0)
    return photon_transfer_noise(image, rng, 10.0 ** (gain_db / 20.0), 8000.0, read_noise)


def _sampled_std(noise: np.ndarray, channel: int) -> float:
    """Noise at the mosaic sites that actually measured ``channel``."""
    sites = bayer_sites(*noise.shape[:2])
    return float(noise[..., channel][sites == channel].std())


def test_shot_noise_grows_as_the_square_root_of_the_signal() -> None:
    """Four times the light is twice the noise: the signature of counting.
    A fixed-strength grain is the same at every level."""
    dim, bright = _sampled_std(_noise(0.1, 0.0), 1), _sampled_std(_noise(0.4, 0.0), 1)
    assert bright / dim == pytest.approx(2.0, rel=0.05)
    # In electrons: 0.4 of an 8000 e- well is 3200 e-, noise sqrt(3200 + 4).
    assert bright * 8000.0 == pytest.approx(np.sqrt(3204.0), rel=0.03)


def test_read_noise_is_what_is_left_in_the_dark() -> None:
    black = _sampled_std(_noise(0.0, 0.0, read_noise=4.0), 1)
    assert black * 8000.0 == pytest.approx(4.0, rel=0.03)
    assert _sampled_std(_noise(0.0, 0.0, read_noise=0.0), 1) == 0.0


def test_gain_makes_the_same_picture_noisier() -> None:
    """Gain creates no noise. It lets fewer electrons count as full scale:
    at 20 dB a tenth as many, so shot noise is sqrt(10) times larger beside
    the picture and read noise ten times."""
    base, gained = _sampled_std(_noise(0.3, 0.0, 0.0), 1), _sampled_std(_noise(0.3, 20.0, 0.0), 1)
    assert gained / base == pytest.approx(np.sqrt(10.0), rel=0.05)
    dark_base = _sampled_std(_noise(0.0, 0.0, 3.0), 1)
    dark_gained = _sampled_std(_noise(0.0, 20.0, 3.0), 1)
    assert dark_gained / dark_base == pytest.approx(10.0, rel=0.05)


def test_colour_noise_is_coarser_than_green() -> None:
    """One site in four is red, so demosaicing spreads its noise over more
    pixels than green's. That is the blotchy colour of a real low-light
    frame; independent per-channel noise has no neighbour correlation at
    all."""
    noise = _noise(0.2, 12.0)

    def neighbour_correlation(plane: np.ndarray) -> float:
        return float(np.corrcoef(plane[:, :-1].ravel(), plane[:, 1:].ravel())[0, 1])

    red, green = neighbour_correlation(noise[..., 0]), neighbour_correlation(noise[..., 1])
    assert green > 0.2, "demosaiced noise must be correlated between neighbours"
    assert red > green + 0.1
    # And the channels are not copies of each other: the frame has colour noise.
    assert abs(np.corrcoef(noise[..., 0].ravel(), noise[..., 1].ravel())[0, 1]) < 0.2


def test_demosaicing_returns_what_was_sampled() -> None:
    """A flat field per channel must come back flat: the interpolation
    neither gains nor loses."""
    sites = bayer_sites(32, 32)
    mosaic = np.choose(sites, [0.2, 0.5, 0.8]).astype(np.float32)
    out = demosaic(mosaic, sites)
    np.testing.assert_allclose(out.reshape(-1, 3).mean(axis=0), [0.2, 0.5, 0.8], atol=1e-5)
    assert float(out.std(axis=(0, 1)).max()) < 1e-5


def test_noise_does_not_shift_the_frame() -> None:
    """It is zero-mean in linear light. Away from black and white, where
    clipping would bias it, the level stays where it was."""
    image = np.full((256, 256, 3), 128, dtype=np.uint8)
    transform = SensorNoise(p=1.0, gain_range_db=(18.0, 18.0))
    transform.set_random_seed(0)
    out = transform(image=image)["image"].astype(np.float32)
    assert out.std() > 2.0
    assert abs(out.mean() - 128.0) < 1.5


def test_shadows_are_the_noisy_part_of_a_picture() -> None:
    """In the encoded picture, where it is looked at, noise is strongest in
    the dark half: relative shot noise is higher there and the tone curve
    stretches it."""
    image = np.zeros((128, 256, 3), dtype=np.uint8)
    image[:, :128] = 40
    image[:, 128:] = 200
    transform = SensorNoise(p=1.0, gain_range_db=(18.0, 18.0))
    transform.set_random_seed(1)
    out = transform(image=image)["image"].astype(np.float32)
    assert out[:, 8:120].std() > 1.5 * out[:, 136:248].std()


def test_the_mosaic_is_what_indexing_by_site_gives() -> None:
    """The Bayer mosaic and its colour planes are made by strided copies and
    not by looking each pixel's site up. Same arrays."""
    rng = np.random.default_rng(3)
    mosaic = rng.standard_normal((63, 70)).astype(np.float32)  # odd height too
    sites = bayer_sites(*mosaic.shape)
    assert sites.dtype == np.uint8
    for channel in range(3):
        by_site = np.where(sites == channel, mosaic, 0.0).astype(np.float32)
        assert np.array_equal(_one_colour(mosaic, channel), by_site)
    assert np.array_equal(demosaic(mosaic, sites), demosaic(mosaic))
