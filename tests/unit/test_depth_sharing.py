"""Depth must belong to the image it is used on, and be estimated once per scene.

Two failures this file exists to prevent, both invisible in a single render
and both found by measuring rather than by looking:

* **The wrong image's depth.** The cache used to key on ``id(image)``. CPython
  reuses a freed object's address for the next allocation of the same size, so
  a ``Dataset.__getitem__`` that loads, transforms and returns allocates every
  frame where the previous one was — and 51 of 60 consecutive frames were
  rendered with the *previous* frame's depth. Each looked fine: fog still got
  thicker "with distance", just not with that scene's distance.
* **No sharing at all.** A transform returns a new array, so the second
  depth-conditioned transform in a chain never saw the array depth had been
  estimated for. The provider only ever shared depth with a transform whose
  predecessor had not run, so every transform that fired paid for its own
  estimate.
"""

from __future__ import annotations

import pickle

import albumentations as A
import numpy as np
import pytest

from weathermentations.atmosphere import RandomFog, RandomRain, RandomSnow
from weathermentations.camera import NightIrMode, RollingShutter
from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.lens import DirtSoil
from weathermentations.scene import CastShadows


class _SignedDepth(DepthEstimator):
    """Depth that says which image it was computed from.

    The top-left pixel's red value is written across the whole map, so a depth
    map handed out for the wrong image is detectable by value.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[np.ndarray] = []

    def estimate(self, image: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.seen.append(image.copy())
        return np.full(image.shape[:2], float(image[0, 0, 0]) / 255.0, dtype=np.float32)


class _GroundPlane(DepthEstimator):
    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[np.ndarray] = []

    def estimate(self, image: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.seen.append(image.copy())
        height, width = image.shape[:2]
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        depth = np.clip(1.5 * 300.0 / np.maximum(rows - height / 2.0, 1e-3), 0.0, 50.0)
        return np.broadcast_to(depth / 50.0, (height, width)).astype(np.float32)


def _scene(seed: int = 0, size: tuple[int, int] = (96, 128)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((*size, 3), dtype=np.uint8)
    image[: size[0] // 3] = (200, 210, 225)
    image[size[0] // 3 :] = (70, 72, 66)
    noise = rng.integers(-10, 11, image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# --- depth belongs to the image -------------------------------------------


def test_a_freed_images_depth_is_never_served_to_the_next_image() -> None:
    """The ``Dataset.__getitem__`` pattern: each frame is allocated, used and
    freed inside one function, so the next lands at the same address."""
    estimator = _SignedDepth()
    provider = SharedDepthProvider(estimator)

    def getitem(index: int) -> tuple[int, float]:
        image = np.full((64, 64, 3), index, dtype=np.uint8)
        return id(image), float(provider.get_depth(image)[0, 0])

    addresses = set()
    for index in range(1, 200):
        address, depth = getitem(index)
        addresses.add(address)
        assert depth == pytest.approx(index / 255.0), (
            f"frame {index} was given the depth of another image"
        )

    # The premise, checked rather than assumed: addresses really are reused,
    # which is what makes an id() key unsafe. Without it this test would pass
    # against the broken cache too.
    assert len(addresses) < 100, "allocator did not reuse addresses; test proves nothing"
    assert estimator.calls == 199


def test_a_buffer_overwritten_in_place_is_a_new_scene() -> None:
    """``capture.read(frame)`` reuses one array for every frame of a video.
    The object is genuinely the same, so only its contents can tell."""
    estimator = _SignedDepth()
    provider = SharedDepthProvider(estimator)

    frame = np.full((64, 64, 3), 10, dtype=np.uint8)
    assert provider.get_depth(frame)[0, 0] == pytest.approx(10 / 255.0)
    frame[:] = 200
    assert provider.get_depth(frame)[0, 0] == pytest.approx(200 / 255.0)
    assert estimator.calls == 2


def test_the_same_unchanged_array_is_a_hit() -> None:
    estimator = _SignedDepth()
    provider = SharedDepthProvider(estimator)
    image = _scene()
    first = provider.get_depth(image)
    second = provider.get_depth(image)
    assert estimator.calls == 1
    assert first is second
    assert provider.cache_info()[:2] == (1, 1)


# --- depth is shared along a pipeline -------------------------------------


def test_chained_depth_transforms_estimate_depth_once() -> None:
    """Shadows, fog and rain all firing on one frame is one estimate, not three."""
    estimator = _GroundPlane()
    provider = SharedDepthProvider(estimator)
    pipeline = A.Compose(
        [
            CastShadows(depth_provider=provider, p=1.0),
            RandomFog(depth_provider=provider, p=1.0),
            RandomRain(depth_provider=provider, p=1.0, max_streaks=200),
        ],
        seed=0,
    )
    image = _scene()
    pipeline(image=image)

    assert estimator.calls == 1
    info = provider.cache_info()
    assert (info.hits, info.misses) == (2, 1)


def test_depth_is_estimated_from_the_clean_frame_not_a_degraded_one() -> None:
    """Fog does not move the scene, so the depth already held is the right
    depth: the estimator must be shown the clean frame, once, and nothing
    downstream of it."""
    estimator = _GroundPlane()
    provider = SharedDepthProvider(estimator)
    pipeline = A.Compose(
        [
            RandomFog(depth_provider=provider, p=1.0, beta_range=(0.1, 0.1)),
            NightIrMode(depth_provider=provider, p=1.0),
        ],
        seed=0,
    )
    image = _scene()
    pipeline(image=image)

    assert len(estimator.seen) == 1
    np.testing.assert_array_equal(estimator.seen[0], image)


def test_the_scene_survives_a_transform_with_no_provider_of_its_own() -> None:
    """Physical order puts the lens between the atmosphere and the sensor, so
    fog -> dirt -> night mode is the ordinary case, not a contrived one."""
    estimator = _GroundPlane()
    provider = SharedDepthProvider(estimator)
    pipeline = A.Compose(
        [
            RandomFog(depth_provider=provider, p=1.0),
            DirtSoil(p=1.0),
            NightIrMode(depth_provider=provider, p=1.0),
        ],
        seed=0,
    )
    pipeline(image=_scene())
    assert estimator.calls == 1


def test_a_transform_that_moves_pixels_does_not_pass_depth_on() -> None:
    """Rolling shutter displaces rows, so the old depth no longer lines up."""
    assert RollingShutter.preserves_scene_geometry is False

    estimator = _GroundPlane()
    provider = SharedDepthProvider(estimator)
    pipeline = A.Compose(
        [
            RandomFog(depth_provider=provider, p=1.0),
            RollingShutter(p=1.0, pan_shift_range_px=(20.0, 20.0)),
            NightIrMode(depth_provider=provider, p=1.0),
        ],
        seed=0,
    )
    pipeline(image=_scene())
    assert estimator.calls == 2


def test_a_foreign_transform_breaks_the_chain_safely_and_carry_restores_it() -> None:
    """Something this library did not write cannot be assumed to keep geometry
    (a crop does not), so depth is re-estimated: slower, never wrong."""
    estimator = _GroundPlane()
    provider = SharedDepthProvider(estimator)
    fog = RandomFog(depth_provider=provider, p=1.0)
    night = NightIrMode(depth_provider=provider, p=1.0)
    fog.set_random_seed(0)
    night.set_random_seed(0)
    image = _scene()

    fogged = fog(image=image)["image"]
    brightened = np.clip(fogged.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    night(image=brightened)
    assert estimator.calls == 2

    # The caller knows a brightness shift keeps geometry, and can say so.
    estimator.calls = 0
    fogged = fog(image=image)["image"]
    brightened = np.clip(fogged.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    assert provider.carry(fogged, brightened) is True
    night(image=brightened)
    assert estimator.calls == 0, "image was already cached, and carry() kept the scene"


def test_two_providers_do_not_see_each_others_scenes() -> None:
    first, second = _GroundPlane(), _GroundPlane()
    pipeline = A.Compose(
        [
            RandomFog(depth_provider=SharedDepthProvider(first), p=1.0),
            NightIrMode(depth_provider=SharedDepthProvider(second), p=1.0),
        ],
        seed=0,
    )
    pipeline(image=_scene())
    assert (first.calls, second.calls) == (1, 1)


def test_successive_frames_through_one_pipeline_each_get_their_own_depth() -> None:
    estimator = _SignedDepth()
    provider = SharedDepthProvider(estimator)
    pipeline = A.Compose(
        [
            RandomFog(depth_provider=provider, p=1.0),
            NightIrMode(depth_provider=provider, p=1.0),
        ],
        seed=0,
    )

    def getitem(index: int) -> None:
        pipeline(image=np.full((48, 48, 3), index, dtype=np.uint8))

    for index in range(1, 80):
        getitem(index)
    assert estimator.calls == 79
    reds = [int(seen[0, 0, 0]) for seen in estimator.seen]
    assert reds == list(range(1, 80)), "depth was estimated from something other than the input"


# --- the provider itself ---------------------------------------------------


def test_cached_depth_is_read_only() -> None:
    """It is shared between every transform in the chain, so one of them
    writing into it would silently change the scene for the rest."""
    provider = SharedDepthProvider(_GroundPlane())
    depth = provider.get_depth(_scene())
    with pytest.raises(ValueError, match="read-only"):
        depth[0, 0] = 1.0


def test_an_estimator_returning_the_wrong_shape_is_reported() -> None:
    class Half(DepthEstimator):
        def estimate(self, image: np.ndarray) -> np.ndarray:
            return np.zeros((image.shape[0] // 2, image.shape[1] // 2), dtype=np.float32)

    with pytest.raises(ValueError, match="one value per pixel"):
        SharedDepthProvider(Half()).get_depth(_scene())


def test_scenes_are_dropped_once_their_images_are_freed() -> None:
    """Nothing is cached across samples: depth is megabytes per frame."""
    provider = SharedDepthProvider(_GroundPlane(), max_scenes=4)
    for seed in range(10):
        provider.get_depth(_scene(seed))
    assert provider.cache_info().scenes == 1

    held = [_scene(seed) for seed in range(10)]
    for image in held:
        provider.get_depth(image)
    assert provider.cache_info().scenes == 4


def test_provider_pickles_as_its_estimator_and_starts_empty() -> None:
    """A ``DataLoader`` worker receives the pipeline by pickle."""
    provider = SharedDepthProvider(_GroundPlane(), max_scenes=2)
    image = _scene()
    provider.get_depth(image)

    clone = pickle.loads(pickle.dumps(provider))
    assert clone.cache_info() == (0, 0, 0)
    np.testing.assert_array_equal(clone.get_depth(image), provider.get_depth(image))

    fog = RandomFog(depth_provider=provider, p=1.0)
    restored = pickle.loads(pickle.dumps(fog))
    restored.set_random_seed(0)
    assert restored(image=image)["image"].shape == image.shape


def test_max_scenes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_scenes"):
        SharedDepthProvider(_GroundPlane(), max_scenes=0)


# --- clips ------------------------------------------------------------------


def _clip(frames: int = 3) -> np.ndarray:
    return np.stack([_scene(seed) for seed in range(frames)])


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda p: RandomFog(depth_provider=p, p=1.0), id="RandomFog"),
        pytest.param(
            lambda p: RandomRain(depth_provider=p, p=1.0, max_streaks=100), id="RandomRain"
        ),
        pytest.param(lambda p: RandomSnow(depth_provider=p, p=1.0), id="RandomSnow"),
        pytest.param(lambda p: NightIrMode(depth_provider=p, p=1.0), id="NightIrMode"),
        pytest.param(lambda p: CastShadows(depth_provider=p, p=1.0), id="CastShadows"),
    ],
)
def test_depth_conditioned_transforms_accept_a_clip(build) -> None:  # type: ignore[no-untyped-def]
    """``images=`` is how Albumentations takes a clip. The transforms that
    read the picture to sample their parameters indexed ``data["image"]`` and
    died with a ``KeyError``; the ones that did not, worked."""
    estimator = _GroundPlane()
    transform = build(SharedDepthProvider(estimator))
    transform.set_random_seed(0)
    clip = _clip()

    out = transform(images=clip)["images"]

    assert out.shape == clip.shape and out.dtype == clip.dtype
    assert not np.array_equal(out, clip)
    assert estimator.calls == 1, "a clip is one scene: one depth estimate, from its first frame"
    np.testing.assert_array_equal(estimator.seen[0], clip[0])


def test_a_clip_keeps_its_depth_along_a_pipeline() -> None:
    estimator = _GroundPlane()
    provider = SharedDepthProvider(estimator)
    pipeline = A.Compose(
        [
            RandomFog(depth_provider=provider, p=1.0),
            DirtSoil(p=1.0),
            NightIrMode(depth_provider=provider, p=1.0),
        ],
        seed=0,
    )
    pipeline(images=_clip())
    assert estimator.calls == 1


# --- calibration --------------------------------------------------------------


def test_the_provider_carries_the_distance_calibration_for_every_transform() -> None:
    """Only ``RandomFog`` could be told how far away a scene is; rain, snow,
    shadows and night mode read module constants. The calibration belongs to
    the scene, so it lives on the object that knows the scene."""
    relative = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    default = SharedDepthProvider(_GroundPlane())
    np.testing.assert_allclose(default.to_metric(relative), [[3.0, 26.5, 50.0]])

    indoor = SharedDepthProvider(_GroundPlane(), depth_range_m=(1.0, 11.0))
    np.testing.assert_allclose(indoor.to_metric(relative), [[1.0, 6.0, 11.0]])
    # One transform may still override the far plane for itself.
    np.testing.assert_allclose(indoor.to_metric(relative, far_m=21.0), [[1.0, 11.0, 21.0]])

    for bad in ((0.0, 10.0), (5.0, 5.0), (9.0, 2.0)):
        with pytest.raises(ValueError, match="depth_range_m"):
            SharedDepthProvider(_GroundPlane(), depth_range_m=bad)


def test_calibration_changes_what_a_transform_renders() -> None:
    """Fog at a fixed extinction is thicker over 400 m than over 12."""
    image = _scene()

    def fogged(depth_range_m: tuple[float, float]) -> np.ndarray:
        provider = SharedDepthProvider(_GroundPlane(), depth_range_m=depth_range_m)
        fog = RandomFog(depth_provider=provider, p=1.0, beta_range=(0.02, 0.02))
        fog.set_random_seed(0)
        result: np.ndarray = fog(image=image)["image"]
        return result

    near_scene, far_scene = fogged((1.0, 12.0)), fogged((15.0, 400.0))
    contrast = lambda frame: float(frame.astype(np.float32).std())  # noqa: E731
    assert contrast(far_scene) < 0.7 * contrast(near_scene)

    calibrated = SharedDepthProvider(_GroundPlane(), depth_range_m=(2.0, 9.0))
    assert pickle.loads(pickle.dumps(calibrated)).depth_range_m == (2.0, 9.0)
