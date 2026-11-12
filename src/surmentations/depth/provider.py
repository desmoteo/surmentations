"""Shared depth for depth-conditioned transforms.

One :class:`SharedDepthProvider` is constructed by the user and passed to
every depth-conditioned transform, so that depth is estimated **once per
scene** and reused by every transform that needs it.

What "the same scene" means
---------------------------
Two things, and the second is the one that makes sharing work at all:

* **The same array, unchanged.** An entry is recognised by the *identity* of
  the image array, held through a weak reference, plus a sparse fingerprint of
  its contents.

  Identity alone is not enough, and ``id(image)`` — what this used to key on —
  is worse than not enough. CPython hands a freed object's address to the next
  allocation of the same size, so in the canonical ``Dataset.__getitem__``
  body (load an image, transform it, return) every frame is allocated where
  the previous one was, gets the same ``id``, and was served the *previous
  frame's* depth: measured, 51 of 60 consecutive frames. A weak reference
  cannot be fooled that way — it dies with the array it points at. The
  fingerprint covers the remaining hole, a buffer that is reused in place
  (``capture.read(frame)`` in a video loop), where the object really is the
  same and only its pixels have changed.

* **An image derived from it.** Fog returns a new array, so a rain transform
  chained after it never sees the array depth was estimated for. Keying on the
  array alone therefore shares depth only between transforms of which the
  first *did not run* — that is, never. Each one that fired paid for its
  own estimate, made from whatever the frame had become by then.

  The loss is cost rather than accuracy, and it is worth being exact about
  that. Measured on the reference photo, depth re-estimated from a fogged
  frame differs from the clean frame's by 1-3% (median, relative), and from a
  frame with droplets on the glass by up to 5.5% — the network copes. What
  re-estimating costs is the tens of milliseconds each time.

  So every transform in this library reports its output back through
  :func:`carry_depth`, and the provider records that the new array shows the
  scene it already has depth for. Depth is a property of the scene's geometry,
  which no appearance-only effect changes, so the clean frame's depth is the
  right one for everything downstream of it. A transform that *does* move
  pixels (``RollingShutter``) opts out, and anything from outside this library
  in the middle of a pipeline simply breaks the chain: the next request
  re-estimates, which is the old behaviour and never a wrong answer.

Nothing is cached across samples. The cache holds a handful of scenes, and a
scene is dropped the moment every array belonging to it has been freed.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any, NamedTuple

import numpy as np

from surmentations import mode
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.scale import (
    DEFAULT_MAX_DISTANCE_METERS,
    DEFAULT_MIN_DISTANCE_METERS,
    to_pseudo_metric,
)

# Samples per axis of the content fingerprint. 32 x 32 pixels is a few
# kilobytes and a few microseconds, against tens of milliseconds for the
# estimate it guards.
_FINGERPRINT_GRID = 32

# Arrays remembered per scene: the image depth was estimated from plus every
# output derived from it since. A pipeline is a dozen transforms deep at most.
_MAX_ALIASES = 16

# Every live provider, so a transform with no provider of its own (lens dirt
# between fog and night mode, say) can still pass the scene along.
_LIVE_PROVIDERS: weakref.WeakSet[SharedDepthProvider] = weakref.WeakSet()


class DepthCacheInfo(NamedTuple):
    """Counters from :meth:`SharedDepthProvider.cache_info`."""

    hits: int
    misses: int
    scenes: int


def _reference_frame(image: np.ndarray) -> np.ndarray:
    """The single frame depth is estimated from: the image, or a clip's first.

    A clip passed as ``images=`` is one camera looking at one scene, and
    Albumentations already applies one set of parameters to all of its frames.
    Its geometry is likewise taken once, from the first frame.
    """
    return image[0] if image.ndim == 4 else image


def _fingerprint(image: np.ndarray) -> tuple[Any, ...]:
    """Shape, dtype and a sparse grid of pixels.

    Not a hash of the whole image: its job is to notice that a buffer was
    overwritten with a different frame, which changes essentially every pixel,
    and a full checksum would cost milliseconds per megapixel on every lookup
    to catch nothing more than that.
    """
    frame = _reference_frame(image)
    height, width = frame.shape[:2]
    step_y = max(1, height // _FINGERPRINT_GRID)
    step_x = max(1, width // _FINGERPRINT_GRID)
    return (image.shape, image.dtype.str, frame[::step_y, ::step_x].tobytes())


class _Alias:
    """One array known to show a scene."""

    __slots__ = ("fingerprint", "ref")

    def __init__(self, image: np.ndarray) -> None:
        """Hold ``image`` weakly, with a fingerprint of its contents."""
        self.ref = weakref.ref(image)
        self.fingerprint = _fingerprint(image)

    def is_alive(self) -> bool:
        """Whether the array this alias stands for still exists."""
        return self.ref() is not None

    def matches(self, image: np.ndarray) -> bool:
        """Whether ``image`` is this alias's array, unchanged since it was seen.

        The same object with the same fingerprint: an array modified in
        place, or a new one at a freed one's address, is not the same scene.
        """
        return self.ref() is image and self.fingerprint == _fingerprint(image)


class _Scene:
    """A depth map and the arrays it is valid for.

    And the mode it was estimated in (:mod:`surmentations.mode`): the
    bundled estimator runs in half precision in fast mode, so a map made
    there is not the one exact mode would make, and is not handed to it.
    """

    __slots__ = ("aliases", "depth", "fast")

    def __init__(self, depth: np.ndarray, image: np.ndarray) -> None:
        """A scene known by one array, with its depth and the mode it was estimated in."""
        self.depth = depth
        self.aliases = [_Alias(image)]
        self.fast = mode.fast()

    def knows(self, image: np.ndarray) -> bool:
        """Whether ``image`` shows this scene, in the mode its depth was made in."""
        if self.fast != mode.fast():
            return False
        return any(alias.matches(image) for alias in self.aliases)

    def adopt(self, image: np.ndarray) -> None:
        """Add ``image`` as another array showing this scene.

        Aliases whose arrays have been freed are dropped, and only the last
        few are kept.
        """
        live = [alias for alias in self.aliases if alias.is_alive() and alias.ref() is not image]
        self.aliases = [*live[-(_MAX_ALIASES - 1) :], _Alias(image)]


class SharedDepthProvider:
    """Estimates depth once per scene and shares it between transforms.

    Construct **one** and pass that instance to every depth-conditioned
    transform in a pipeline. See the module docstring for what is recognised
    as the same scene, and why.

    Args:
        estimator: the depth backend.
        depth_range_m: the distances, in metres, that the nearest and the
            farthest point of a scene are taken to be at. Monocular depth has
            no absolute scale, so this is the calibration every
            depth-conditioned transform reads its metres through — set it once
            here for cameras whose scenes do not span the default 3-50 m. An
            indoor camera might be ``(1.0, 12.0)``; a mast over a motorway,
            ``(15.0, 400.0)``. It matters linearly to fog, rain and snow, and
            quadratically to ``NightIrMode``'s inverse-square illuminator.
        max_scenes: how many scenes to keep depth for. One is all a single
            pipeline call needs; the default leaves room for a few calls in
            flight at once (a thread pool sharing one provider), which would
            otherwise evict each other's depth between two transforms and fall
            back to re-estimating. A scene whose arrays have all been freed is
            dropped regardless, so in an ordinary loop only one is ever held.

    The provider pickles as its estimator alone — cached depth is per-process
    state — so it survives being sent to a ``DataLoader`` worker if the
    estimator does.
    """

    def __init__(
        self,
        estimator: DepthEstimator,
        max_scenes: int = 4,
        depth_range_m: tuple[float, float] = (
            DEFAULT_MIN_DISTANCE_METERS,
            DEFAULT_MAX_DISTANCE_METERS,
        ),
    ) -> None:
        """Wrap ``estimator``; see the class docstring for the arguments.

        Raises ``ValueError`` if ``max_scenes`` is below 1 or the depth range
        is not ``0 < near < far``.
        """
        if max_scenes < 1:
            raise ValueError(f"max_scenes must be at least 1, got {max_scenes}")
        near, far = depth_range_m
        if not 0.0 < near < far:
            raise ValueError(
                f"depth_range_m must be (near, far) with 0 < near < far, got {depth_range_m}"
            )
        self._estimator = estimator
        self._max_scenes = max_scenes
        self.depth_range_m = (float(near), float(far))
        self._init_state()

    def _init_state(self) -> None:
        """Start with no scenes held, and register with the live providers.

        Called on construction and on unpickling: what is cached is per
        process, and a worker starts empty.
        """
        self._scenes: list[_Scene] = []  # least recently used first
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()
        _LIVE_PROVIDERS.add(self)

    def __getstate__(self) -> dict[str, Any]:
        """Pickle as the estimator and the settings; the scenes held stay behind."""
        return {
            "_estimator": self._estimator,
            "_max_scenes": self._max_scenes,
            "depth_range_m": self.depth_range_m,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the estimator and settings, with nothing cached."""
        self._estimator = state["_estimator"]
        self._max_scenes = state["_max_scenes"]
        self.depth_range_m = state["depth_range_m"]
        self._init_state()

    def _find(self, image: np.ndarray) -> _Scene | None:
        """The scene held for ``image``, most recently used first; ``None`` if there is none."""
        for scene in reversed(self._scenes):
            if scene.knows(image):
                return scene
        return None

    def _prune(self) -> None:
        """Forget scenes nothing can ask for again, then enforce the cap."""
        self._scenes = [
            scene for scene in self._scenes if any(alias.is_alive() for alias in scene.aliases)
        ]
        del self._scenes[: max(0, len(self._scenes) - self._max_scenes)]

    def get_depth(self, image: np.ndarray) -> np.ndarray:
        """Relative depth for ``image``, estimated at most once per scene.

        ``image`` is one ``(H, W, 3)`` frame or an ``(N, H, W, 3)`` clip; a
        clip gets one ``(H, W)`` depth map, from its first frame. The returned
        array is shared between every caller and is therefore read-only; copy
        it before modifying it.
        """
        with self._lock:
            scene = self._find(image)
            if scene is not None:
                self._hits += 1
                self._scenes.remove(scene)
                self._scenes.append(scene)
                return scene.depth

            self._misses += 1
            frame = _reference_frame(image)
            depth = np.asarray(self._estimator.estimate(frame))
            if depth.shape != frame.shape[:2]:
                raise ValueError(
                    f"{type(self._estimator).__name__}.estimate returned depth of shape "
                    f"{depth.shape} for an image of shape {frame.shape}; a DepthEstimator "
                    f"must return one value per pixel, shape {frame.shape[:2]}."
                )
            # A view, so the flag does not leak onto an array the estimator
            # may still own.
            depth = depth.view()
            depth.flags.writeable = False

            self._scenes.append(_Scene(depth, image))
            self._prune()
            return depth

    def to_metric(self, relative_depth: np.ndarray, far_m: float | None = None) -> np.ndarray:
        """Relative depth in pseudo-metres, through this provider's calibration.

        ``far_m`` overrides the far plane for one call, which is what
        ``RandomFog``'s ``depth_scale`` does.
        """
        near, far = self.depth_range_m
        return to_pseudo_metric(relative_depth, far if far_m is None else far_m, near)

    def get_metric_depth(self, image: np.ndarray) -> np.ndarray:
        """:meth:`get_depth` followed by :meth:`to_metric`."""
        return self.to_metric(self.get_depth(image))

    def carry(self, source: np.ndarray, derived: np.ndarray) -> bool:
        """Record that ``derived`` shows the same scene geometry as ``source``.

        After this, :meth:`get_depth` answers for ``derived`` with the depth
        already held for ``source``. Transforms in this library call it for
        you; call it yourself after a geometry-preserving step of your own
        (a colour jitter, a blur) that sits between two depth-conditioned
        transforms.

        Returns whether anything was recorded — ``False`` when ``source`` is
        not a scene this provider holds depth for, or the two differ in size.
        """
        if derived is source or derived.shape[:-1] != source.shape[:-1]:
            return False
        with self._lock:
            scene = self._find(source)
            if scene is None:
                return False
            scene.adopt(derived)
            return True

    def cache_info(self) -> DepthCacheInfo:
        """Hits, misses and scenes held — the way to check depth is being shared.

        A pipeline in which two depth-conditioned transforms fired on one
        image should show one miss and one hit for it, not two misses.
        """
        with self._lock:
            return DepthCacheInfo(self._hits, self._misses, len(self._scenes))


def carry_depth(source: np.ndarray, derived: np.ndarray) -> None:
    """Tell every live provider that ``derived`` shows ``source``'s scene.

    Called by :class:`~surmentations.base.CameraTransform` after each
    transform, so the scene survives a transform that has no provider of its
    own. A no-op when no provider holds depth for ``source``, which is the
    common case and costs one identity comparison per provider.
    """
    for provider in list(_LIVE_PROVIDERS):
        provider.carry(source, derived)
