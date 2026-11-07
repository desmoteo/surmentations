"""The channel helpers against the numpy reductions they replace."""

from __future__ import annotations

import numpy as np
import pytest

from surmentations.utils.channels import (
    channel_max,
    channel_mean,
    channel_min,
    channel_sum,
    weighted_colour,
)


def test_channel_reductions_are_the_reductions_they_replace() -> None:
    rgb = np.random.default_rng(11).uniform(-0.2, 1.4, (90, 70, 3)).astype(np.float32)
    assert np.array_equal(channel_max(rgb), rgb.max(axis=2))
    assert np.array_equal(channel_min(rgb), rgb.min(axis=2))
    assert np.array_equal(channel_sum(rgb), rgb.sum(axis=2))
    assert np.array_equal(channel_mean(rgb), rgb.mean(axis=2))
    weight = np.random.default_rng(12).uniform(size=(90, 70)).astype(np.float32)
    expected = (rgb * weight[..., None]).sum(axis=(0, 1))
    assert weighted_colour(rgb, weight) == pytest.approx(expected, rel=1e-5)
