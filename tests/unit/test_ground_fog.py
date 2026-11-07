"""GroundFog: a layer whose optical depth is an integral along each ray, checked against it."""

from __future__ import annotations

import math

import numpy as np
import pytest

from surmentations.atmosphere import GroundFog
from surmentations.atmosphere.ground_fog import (
    layer_optical_depth,
    open_sky_optical_depth,
    scene_points,
)
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider

LAYER = {"beta_ground": 0.05, "scale_height_m": 3.0, "camera_height_m": 6.0}


def _numerically(distance: float, relative_height: float, steps: int = 20000) -> float:
    """The same integral by adding up the extinction along the ray."""
    along = (np.arange(steps) + 0.5) / steps
    height = LAYER["camera_height_m"] + along * relative_height
    beta = LAYER["beta_ground"] * np.exp(-height / LAYER["scale_height_m"])
    return float(beta.mean() * distance)


class _Road(DepthEstimator):
    """A level road under a camera six metres up, pitched down twenty degrees, to fifty metres."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Relative depth of the ground plane, capped at the far plane."""
        height, width = image.shape[:2]
        focal = width / (2.0 * math.tan(math.radians(30.0)))
        y = (np.arange(height, dtype=np.float32) - (height - 1) / 2.0)[:, None] / focal
        pitch = math.radians(20.0)
        down = y * math.cos(pitch) + math.sin(pitch)
        depth = np.where(down > 1e-3, 6.0 / np.maximum(down, 1e-3), 50.0)
        metres = np.clip(depth, 3.0, 50.0)
        return np.broadcast_to((metres - 3.0) / 47.0, (height, width)).astype(np.float32)


def _fog(**settings: object) -> GroundFog:
    """An even layer over the road of :class:`_Road`, every range pinned unless changed."""
    arguments: dict[str, object] = {
        "visibility_range_m": (40.0, 40.0),
        "layer_height_range_m": (3.0, 3.0),
        "camera_height_range_m": (6.0, 6.0),
        "camera_pitch_range_deg": (20.0, 20.0),
        "horizontal_fov_range_deg": (60.0, 60.0),
        "patchiness_range": (0.0, 0.0),
        **settings,
    }
    provider = SharedDepthProvider(_Road(), depth_range_m=(3.0, 50.0))
    return GroundFog(depth_provider=provider, p=1.0, **arguments)


def _transmission(fog: GroundFog, seed: int = 0) -> np.ndarray:
    """The transmission a fog gives over a grey frame."""
    image = np.full((120, 160, 3), 90, dtype=np.uint8)
    fog.set_random_seed(seed)
    result: np.ndarray = fog.get_params_dependent_on_data({"shape": image.shape}, {"image": image})[
        "transmission"
    ]
    return result


@pytest.mark.parametrize(
    ("distance", "relative_height"),
    [(30.0, 0.0), (30.0, -6.0), (80.0, -3.0), (40.0, 10.0), (200.0, 60.0), (10.0, -5.999)],
)
def test_the_closed_form_is_the_integral_along_the_ray(
    distance: float, relative_height: float
) -> None:
    ours = float(layer_optical_depth(np.array(distance), np.array(relative_height), **LAYER))
    assert ours == pytest.approx(_numerically(distance, relative_height), rel=2e-3)


def test_a_ray_that_dips_into_the_layer_meets_more_fog_than_a_level_one() -> None:
    level = float(layer_optical_depth(np.array(50.0), np.array(0.0), **LAYER))
    down = float(layer_optical_depth(np.array(50.0), np.array(-6.0), **LAYER))
    up = float(layer_optical_depth(np.array(50.0), np.array(20.0), **LAYER))
    assert down > 2.0 * level > 4.0 * up


def test_the_sky_is_clear_overhead_and_closed_at_the_horizon() -> None:
    focal = 160 / (2.0 * math.tan(math.radians(30.0)))
    # A camera looking up at thirty degrees: the horizon is below the
    # frame's middle, and the top of the frame is steeply up.
    tau = open_sky_optical_depth(120, 160, focal, math.radians(-30.0), **LAYER)
    centre = tau[:, 80]
    assert (np.diff(centre) >= -1e-6).all()
    # Straight along the axis, thirty degrees up: the column over the sine.
    column = LAYER["beta_ground"] * math.exp(-2.0) * LAYER["scale_height_m"]
    assert float(tau[60, 80]) == pytest.approx(column / math.sin(math.radians(30.0)), rel=0.05)
    # And a camera looking down sees only ground along its axis, which a
    # ray into the sky is not: it is given the ceiling.
    down = open_sky_optical_depth(120, 160, focal, math.radians(40.0), **LAYER)
    assert float(down[100, 80]) == pytest.approx(12.0)


def test_a_point_is_not_taken_below_the_ground() -> None:
    at_ground = layer_optical_depth(np.array(30.0), np.array(-6.0), **LAYER)
    below = layer_optical_depth(np.array(30.0), np.array(-40.0), **LAYER)
    assert float(below) == pytest.approx(float(at_ground))


def test_pixels_of_a_level_road_are_all_at_the_camera_s_height_below_it() -> None:
    fog = _fog()
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    depth = fog.depth_provider.get_metric_depth(image)
    focal = 160 / (2.0 * math.tan(math.radians(30.0)))
    distance, relative_height, across, forward = scene_points(depth, focal, math.radians(20.0))
    on_road = depth < 49.0
    assert np.allclose(relative_height[on_road], -6.0, atol=0.05)
    assert (distance >= depth - 1e-4).all()
    assert forward[on_road].min() > 0.0
    assert abs(float(across[:, 80].mean())) < abs(float(across[:, 150].mean()))


def test_a_deeper_layer_and_a_shorter_visibility_both_hide_more() -> None:
    def clear(**settings: object) -> float:
        """Mean transmission over the frame."""
        return float(_transmission(_fog(**settings)).mean())

    assert clear(layer_height_range_m=(8.0, 8.0)) < clear(layer_height_range_m=(1.5, 1.5))
    assert clear(visibility_range_m=(15.0, 15.0)) < clear(visibility_range_m=(150.0, 150.0))


def test_a_camera_above_the_layer_sees_the_road_through_less_of_it() -> None:
    low = _transmission(_fog(camera_height_range_m=(6.0, 6.0)))
    # The same road and the same depths, said to be seen from further up.
    high = _transmission(_fog(camera_height_range_m=(20.0, 20.0)))
    assert float(high.mean()) > float(low.mean())


def test_banks_move_the_fog_about_and_add_none() -> None:
    even = _transmission(_fog())
    banked = [_transmission(_fog(patchiness_range=(0.8, 0.8)), seed) for seed in range(6)]
    assert np.mean([-np.log(b).mean() for b in banked]) == pytest.approx(
        float(-np.log(even).mean()), rel=0.2
    )
    assert float(np.abs(banked[0] - even).max()) > 0.05


def test_fog_veils_towards_the_airlight_and_never_past_it() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(20, 120, (120, 160, 3), dtype=np.uint8)
    fog = _fog(visibility_range_m=(10.0, 10.0), layer_height_range_m=(10.0, 10.0))
    fog.set_random_seed(0)
    out = fog(image=image)["image"].astype(np.float32)
    contrast_before = float(image[20:60].std())
    assert float(out[20:60].std()) < 0.5 * contrast_before
    assert out.max() <= 255.0 and out.min() >= 0.0
