"""LocalPlumes: a cloud that is somewhere, sized by its distance, hiding only what is behind it."""

from __future__ import annotations

import numpy as np
import pytest

from surmentations.atmosphere import LocalPlumes
from surmentations.atmosphere.plumes import plume_depth
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider

SHAPE = (240, 320)
PLUME = {"source": (160.0, 200.0), "width_px": 30.0, "rise_px": 150.0, "lean": 0.0, "peak": 2.0}


class _Wall(DepthEstimator):
    """Everything at one distance, except a near post down the middle of the frame."""

    def __init__(self, post: bool = False) -> None:
        """Whether the post is there."""
        self.post = post

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Relative depth 0.5 everywhere, and 0 on the post."""
        depth = np.full(image.shape[:2], 0.5, dtype=np.float32)
        if self.post:
            depth[:, 150:170] = 0.0
        return depth


def _depth(seed: int = 0, **changes: object) -> np.ndarray:
    """One plume's optical depth."""
    return plume_depth(np.random.default_rng(seed), *SHAPE, **{**PLUME, **changes})


def _plumes(post: bool = False, **settings: object) -> LocalPlumes:
    """One dense plume in front of a wall 20 m away, or of the same wall with a post at 2 m."""
    provider = SharedDepthProvider(_Wall(post), depth_range_m=(2.0, 38.0))
    arguments: dict[str, object] = {
        "plume_count_range": (1, 1),
        "opacity_range": (4.0, 4.0),
        "source_width_range_m": (1.0, 1.0),
        "rise_range_m": (5.0, 5.0),
        **settings,
    }
    return LocalPlumes(depth_provider=provider, p=1.0, **arguments)


def test_a_plume_rises_from_its_source_and_ends() -> None:
    depth = np.mean([_depth(seed) for seed in range(8)], axis=0)
    assert float(depth[210:].max()) == 0.0
    assert float(depth[:45].max()) < 0.02
    assert float(depth[150:195].sum()) > float(depth[60:105].sum()) > 0.0


def test_it_spreads_and_thins_as_it_rises() -> None:
    depth = np.mean([_depth(seed) for seed in range(24)], axis=0)

    def width(row: int) -> float:
        """Second-moment width of the plume along a row."""
        line = depth[row].astype(np.float64)
        at = np.arange(len(line))
        mean = (line * at).sum() / line.sum()
        return float(np.sqrt((line * (at - mean) ** 2).sum() / line.sum()))

    assert width(80) > 1.5 * width(185)
    assert float(depth[185].max()) > 1.5 * float(depth[80].max())


def test_the_wind_leans_it() -> None:
    def top(lean: float) -> float:
        """Where the plume's light is centred, across, near its top."""
        depth = np.mean([_depth(seed, lean=lean) for seed in range(12)], axis=0)
        band = depth[70:100].sum(axis=0)
        return float((band * np.arange(len(band))).sum() / band.sum())

    assert top(0.5) - top(0.0) == pytest.approx(0.5 * 115.0, rel=0.3)
    assert top(-0.5) < top(0.0)


def test_a_plume_is_the_size_its_distance_makes_it() -> None:
    def covered(far: float) -> float:
        """Share of the frame a plume hides, on a wall at half the far distance."""
        provider = SharedDepthProvider(_Wall(), depth_range_m=(2.0, far))
        shares = []
        for seed in range(10):
            plumes = LocalPlumes(
                depth_provider=provider, p=1.0, plume_count_range=(1, 1), opacity_range=(4.0, 4.0)
            )
            plumes.set_random_seed(seed)
            image = np.zeros((*SHAPE, 3), dtype=np.uint8)
            veiled = plumes.get_params_dependent_on_data({"shape": image.shape}, {"image": image})[
                "veiled"
            ]
            shares.append(float((veiled > 0.3).mean()))
        return float(np.mean(shares))

    # A wall three times as far: a plume a ninth of the area.
    assert covered(18.0) > 4.0 * covered(58.0) > 0.0


def test_what_stands_in_front_of_a_plume_is_not_veiled() -> None:
    image = np.full((*SHAPE, 3), 40, dtype=np.uint8)
    for seed in range(6):
        plumes = _plumes(post=True)
        plumes.set_random_seed(seed)
        params = plumes.get_params_dependent_on_data({"shape": image.shape}, {"image": image})
        veiled = params["veiled"]
        # The plume stands where its source is: on the wall, or on the post
        # itself, in which case the post is not in front of it.
        if float(veiled[:, 150:170].max()) > 0.0:
            continue
        assert float(veiled.max()) > 0.5
        assert float(veiled[:, 150:170].max()) == 0.0
        return
    pytest.fail("no plume stood behind the post in six draws")


def test_a_plume_takes_the_level_of_the_light_on_it() -> None:
    image = np.full((*SHAPE, 3), 60, dtype=np.uint8)

    def level(brightness: float) -> float:
        """Level of the frame where the plume is thickest."""
        plumes = _plumes(brightness_range=(brightness, brightness))
        plumes.set_random_seed(1)
        params = plumes.get_params_dependent_on_data({"shape": image.shape}, {"image": image})
        out = plumes.apply(image, **params)
        return float(out[params["veiled"] > 0.9].mean())

    assert level(0.9) > level(0.3) > 60.0


def test_a_plume_is_brightest_on_top_and_greyer_underneath() -> None:
    image = np.full((*SHAPE, 3), 60, dtype=np.uint8)
    plumes = _plumes()
    for seed in range(6):
        plumes.set_random_seed(seed)
        params = plumes.get_params_dependent_on_data({"shape": image.shape}, {"image": image})
        shade, thick = params["shade"], params["veiled"] > 0.5
        rows = np.nonzero(thick.any(axis=1))[0]
        if len(rows) < 30:
            continue
        top, bottom = rows[: len(rows) // 4], rows[-len(rows) // 4 :]
        assert float(shade[top][thick[top]].mean()) > float(shade[bottom][thick[bottom]].mean())
        assert float(shade.min()) >= 0.55 - 1e-6
        return
    pytest.fail("no plume large enough to measure in six draws")


def test_no_plumes_leave_the_frame_alone() -> None:
    image = np.random.default_rng(0).integers(0, 256, (*SHAPE, 3), dtype=np.uint8)
    plumes = _plumes(plume_count_range=(0, 0))
    plumes.set_random_seed(0)
    assert np.array_equal(plumes(image=image)["image"], image)
