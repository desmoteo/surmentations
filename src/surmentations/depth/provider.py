"""Shared depth for depth-conditioned transforms.

One :class:`SharedDepthProvider` is constructed by the user and passed to
every depth-conditioned transform, so that depth is estimated **once per
scene** and reused by every transform that needs it.

The same goes for the two things that may be worked out alongside depth
(:mod:`surmentations.perception`): what each pixel of the scene is, and
how the camera is mounted over it. Both are optional, both are made when
the scene's depth is, from the same frame, and both are held with it.

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

Nothing is held in memory across samples. The cache holds a handful of
scenes, and a scene is dropped the moment every array belonging to it has
been freed.

Across epochs
-------------
A frame of a training set comes back every epoch, as a new array each time,
and is read again each time: the depth, and with a provider that names the
scene and reads its vehicles some 170 ms of networks. ``disk_cache`` keeps
each reading in a file (:mod:`surmentations.depth.store`), found again by
a hash of the frame's pixels and of what read it, so that the reading is
made once for as long as the files are kept. It is off unless asked for.
"""

from __future__ import annotations

import os
import threading
import weakref
from typing import Any, NamedTuple

import cv2
import numpy as np

from surmentations import mode
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.scale import (
    DEFAULT_MAX_DISTANCE_METERS,
    DEFAULT_MIN_DISTANCE_METERS,
    to_pseudo_metric,
)
from surmentations.depth.store import ReadingStore, SceneReading, StoreInfo, default_directory
from surmentations.perception.geometry import (
    Camera,
    GeometryEstimator,
    InstallationGeometry,
    metric_depth,
    metric_scale,
)
from surmentations.perception.labels import (
    GROUND,
    SceneClass,
    SceneLabelEstimator,
    check_labels,
    class_weight,
)
from surmentations.perception.vehicles import (
    VehicleView,
    VehicleViewEstimator,
    check_views,
    scaled_view,
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
# How many frames made from frames it has not been asked about a provider
# remembers the start of: a pipeline is a dozen transforms at most.
_MAX_LINEAGE = 16


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

    __slots__ = ("fingerprint", "lit_air", "ref")

    def __init__(self, image: np.ndarray, lit_air: np.ndarray | None = None) -> None:
        """Hold ``image`` weakly, with a fingerprint of its contents.

        ``lit_air`` is what is known of the light in the air of this one
        picture of the scene (:meth:`SharedDepthProvider.set_lit_air`): it
        is the picture's and not the scene's, since the same scene by day
        has none.
        """
        self.ref = weakref.ref(image)
        self.fingerprint = _fingerprint(image)
        self.lit_air = lit_air

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

    __slots__ = (
        "aliases",
        "asked",
        "depth",
        "disparity",
        "fast",
        "fov",
        "geometry",
        "labels",
        "metric",
        "pitch",
        "signs",
        "street_lamps",
        "vehicles",
    )

    def __init__(self, depth: np.ndarray, image: np.ndarray) -> None:
        """A scene known by one array, with its depth and the mode it was estimated in.

        What else is known of the scene starts empty: its labels and its
        geometry are filled in by the provider that made it, and the field
        of view and pitch its camera is taken to have are settled the first
        time a transform asks.
        """
        self.depth = depth
        self.aliases = [_Alias(image)]
        self.fast = mode.fast()
        # Whether its depth has been asked for yet, which is what the
        # provider's count of cache hits goes by.
        self.asked = False
        self.labels: np.ndarray | None = None
        self.geometry: InstallationGeometry | None = None
        self.vehicles: list[VehicleView] | None = None
        self.street_lamps: np.ndarray | None = None
        self.signs: np.ndarray | None = None
        self.disparity: np.ndarray | None = None
        self.fov: float | None = None
        self.pitch: float | None = None
        # The two numbers that turn disparity into metres; False once they
        # have been looked for and there are none.
        self.metric: tuple[float, float] | bool | None = None

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

    def alias_of(self, image: np.ndarray) -> _Alias | None:
        """The alias that stands for ``image``, if this scene has one."""
        for alias in self.aliases:
            if alias.matches(image):
                return alias
        return None


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
        labels: what names the pixels of each scene
            (:mod:`surmentations.perception.labels`), or ``None``. With
            one, every effect that reads the ground, the sky or the
            vegetation is told what the network saw as well as what the
            depth map suggests.
        geometry: how the camera is mounted
            (:mod:`surmentations.perception.geometry`), or ``None``.
            Give an :class:`~surmentations.perception.geometry.InstallationGeometry`
            for a fixed camera whose mounting is known, or a
            :class:`~surmentations.perception.geometry.GeometryEstimator`
            to have it worked out of each frame. With one, every transform
            that asks sees the scene through the same camera, and where the
            camera's height is known the depth is in metres measured from
            it and ``depth_range_m`` is not used.
        vehicles: what says which end of each vehicle is seen and where its
            lamps are (:mod:`surmentations.perception.vehicles`), or
            ``None``. It reads the vehicles the labels name, so it needs
            ``labels`` too. With one, spray is laid behind each vehicle and
            its lamps are lit where they are, by a network's reading and
            not by colour. It is the dearest thing a provider runs.
        max_scenes: how many scenes to keep depth for. One is all a single
            pipeline call needs; the default leaves room for a few calls in
            flight at once (a thread pool sharing one provider), which would
            otherwise evict each other's depth between two transforms and fall
            back to re-estimating. A scene whose arrays have all been freed is
            dropped regardless, so in an ordinary loop only one is ever held.
        disk_cache: where to keep what is read of each frame from one run
            to the next, and from one epoch to the next: a directory, or
            ``True`` for ``surmentations`` under the user's cache
            directory (``$SURMENTATIONS_CACHE`` names another). ``None``,
            the default, keeps nothing on disk. A reading is found again
            by the frame's own pixels, the estimators that made it, the
            mode it was made in and the library's version, so nothing
            stale is ever handed back; a frame given its labels by hand
            (:meth:`remember`) is not kept, nor is anything made by an
            estimator that does not describe itself
            (:meth:`~surmentations.depth.estimator.DepthEstimator.signature`).
            About half a megabyte a frame at 512 pixels.
        disk_cache_max_gb: the most the directory may hold, or ``None``
            for no limit. Past it nothing more is written and nothing is
            thrown out.

    The provider pickles as its estimators and its settings — what it
    holds in memory is per-process state — so it survives being sent to a
    ``DataLoader`` worker if they do, and every worker then reads and
    writes the same directory.
    """

    def __init__(
        self,
        estimator: DepthEstimator,
        max_scenes: int = 4,
        depth_range_m: tuple[float, float] = (
            DEFAULT_MIN_DISTANCE_METERS,
            DEFAULT_MAX_DISTANCE_METERS,
        ),
        labels: SceneLabelEstimator | None = None,
        geometry: InstallationGeometry | GeometryEstimator | None = None,
        vehicles: VehicleViewEstimator | None = None,
        disk_cache: str | os.PathLike[str] | bool | None = None,
        disk_cache_max_gb: float | None = None,
    ) -> None:
        """Wrap ``estimator``; see the class docstring for the arguments.

        Raises ``ValueError`` if ``max_scenes`` is below 1, the depth range
        is not ``0 < near < far`` or the disk cache's limit is below 0.
        """
        if max_scenes < 1:
            raise ValueError(f"max_scenes must be at least 1, got {max_scenes}")
        near, far = depth_range_m
        if not 0.0 < near < far:
            raise ValueError(
                f"depth_range_m must be (near, far) with 0 < near < far, got {depth_range_m}"
            )
        if labels is not None and not isinstance(labels, SceneLabelEstimator):
            raise TypeError(
                f"labels must be a SceneLabelEstimator or None, got {type(labels).__name__}"
            )
        if geometry is not None and not isinstance(
            geometry, (InstallationGeometry, GeometryEstimator)
        ):
            raise TypeError(
                "geometry must be an InstallationGeometry, a GeometryEstimator or None, "
                f"got {type(geometry).__name__}"
            )
        if vehicles is not None and not isinstance(vehicles, VehicleViewEstimator):
            raise TypeError(
                f"vehicles must be a VehicleViewEstimator or None, got {type(vehicles).__name__}"
            )
        self._estimator = estimator
        self._labels = labels
        self._geometry = geometry
        self._vehicles = vehicles
        self._max_scenes = max_scenes
        self.depth_range_m = (float(near), float(far))
        if disk_cache_max_gb is not None and disk_cache_max_gb < 0.0:
            raise ValueError(f"disk_cache_max_gb must be 0 or more, got {disk_cache_max_gb}")
        # Kept as a directory's name and a limit, which is what a worker
        # is sent; the store itself is each process's own.
        self._disk_cache: str | None = None
        if disk_cache is True:
            self._disk_cache = str(default_directory())
        elif disk_cache:
            self._disk_cache = os.fspath(disk_cache)
        self._disk_cache_max_gb = disk_cache_max_gb
        self._init_state()

    def _init_state(self) -> None:
        """Start with no scenes held, and register with the live providers.

        Called on construction and on unpickling: what is cached is per
        process, and a worker starts empty.
        """
        self._scenes: list[_Scene] = []  # least recently used first
        # Frames made from frames this provider has not been asked about:
        # each with the frame it started as, both held weakly.
        self._lineage: list[tuple[_Alias, weakref.ref[np.ndarray]]] = []
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()
        self._store: ReadingStore | None = None
        if self._disk_cache is not None:
            self._store = ReadingStore(self._disk_cache, self._disk_cache_max_gb)
        _LIVE_PROVIDERS.add(self)

    def __getstate__(self) -> dict[str, Any]:
        """Pickle as the estimator and the settings; the scenes held stay behind."""
        return {
            "_estimator": self._estimator,
            "_labels": self._labels,
            "_geometry": self._geometry,
            "_vehicles": self._vehicles,
            "_max_scenes": self._max_scenes,
            "depth_range_m": self.depth_range_m,
            "_disk_cache": self._disk_cache,
            "_disk_cache_max_gb": self._disk_cache_max_gb,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the estimator and settings, with nothing cached."""
        self._estimator = state["_estimator"]
        self._labels = state.get("_labels")
        self._geometry = state.get("_geometry")
        self._vehicles = state.get("_vehicles")
        self._max_scenes = state["_max_scenes"]
        self.depth_range_m = state["depth_range_m"]
        self._disk_cache = state.get("_disk_cache")
        self._disk_cache_max_gb = state.get("_disk_cache_max_gb")
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

    def _scene_for(
        self,
        image: np.ndarray,
        labels: np.ndarray | None = None,
        geometry: InstallationGeometry | None = None,
        original: np.ndarray | None = None,
    ) -> _Scene:
        """The scene ``image`` shows, made now if it is not held. Call with the lock held.

        Making a scene estimates its depth, and with them its labels and its
        geometry where the provider has an estimator for them, all from this
        one frame: the frame a pipeline starts from, before fog or rain has
        been drawn over it. ``labels`` and ``geometry`` given here are used
        in place of the estimators'.

        **From the frame as it was.** A frame this provider has not been
        asked about, made by transforms from one it can still find (they
        tell it: :meth:`carry`), is read as that first frame: what stands
        in a scene is looked for in the photograph, and not in the
        photograph with a flare or a night drawn over it. And where
        ``original`` gives the frame at the size it was taken at, what
        names and finds things is run on that
        (:meth:`remember`).
        """
        scene = self._find(image)
        if scene is not None:
            self._scenes.remove(scene)
            self._scenes.append(scene)
            return scene
        started_as = self._started_as(image)
        if started_as is not None and labels is None and geometry is None:
            scene = self._scene_for(started_as, original=original)
            scene.adopt(image)
            return scene

        self._misses += 1
        frame = _reference_frame(image)
        source = frame if original is None else _reference_frame(original)
        if source.shape[0] * frame.shape[1] != source.shape[1] * frame.shape[0]:
            across, down = source.shape[1] / frame.shape[1], source.shape[0] / frame.shape[0]
            if abs(across / down - 1.0) > 0.02:
                raise ValueError(
                    f"original is {source.shape[1]} x {source.shape[0]} and the frame "
                    f"{frame.shape[1]} x {frame.shape[0]}: the original of a frame is the "
                    "same picture at another size, and these are not the same shape."
                )
        # What was read of this frame another time, where readings are
        # kept: not for a frame whose labels are the caller's own.
        key = self._store_key(frame, source) if labels is None else None
        kept = None
        if key is not None and self._store is not None:
            kept = self._store.load(key, frame.shape[:2])
        if kept is not None:
            depth = kept.depth
        else:
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
        scene = _Scene(depth, image)

        named = None
        if kept is not None:
            labels = kept.labels
        elif labels is None and self._labels is not None:
            # Named in the frame as it was taken, and brought to the size
            # of the frame the pipeline works at.
            named = np.asarray(self._labels.estimate(source))
            labels = named
            if source is not frame and named.shape == source.shape[:2]:
                labels = cv2.resize(
                    named, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
                )
        if labels is not None:
            given_by = "remember()" if self._labels is None else type(self._labels).__name__
            scene.labels = check_labels(labels, frame.shape[:2], source=given_by).view()
            scene.labels.flags.writeable = False
        scene.disparity = self._estimator.disparity(depth)
        if isinstance(self._geometry, GeometryEstimator):
            estimated = self._geometry.estimate(frame, scene.disparity, scene.labels)
            geometry = estimated if geometry is None else geometry.filled_from(estimated)
        elif geometry is None:
            geometry = self._geometry
        scene.geometry = geometry
        if kept is not None:
            scene.vehicles, scene.street_lamps, scene.signs = (
                kept.vehicles,
                kept.street_lamps,
                kept.signs,
            )
        else:
            self._read_vehicles(scene, frame, source, named)
            if key is not None and self._store is not None:
                self._store.save(
                    key,
                    SceneReading(
                        depth, scene.labels, scene.vehicles, scene.street_lamps, scene.signs
                    ),
                )

        self._scenes.append(scene)
        self._prune()
        return scene

    def _store_key(self, frame: np.ndarray, source: np.ndarray) -> str | None:
        """What a reading of ``frame`` is filed under on disk; ``None`` if it is not to be kept.

        Not kept without a store, nor when one of the estimators cannot
        say what it is: its readings could not be told from another's.
        """
        if self._store is None:
            return None
        signatures = []
        for name, estimator in (
            ("depth", self._estimator),
            ("labels", self._labels),
            ("vehicles", self._vehicles),
        ):
            said = "" if estimator is None else getattr(estimator, "signature", lambda: None)()
            if said is None:
                return None
            signatures.append(f"{name}={said}")
        return self._store.key(frame, source, signatures)

    def get_depth(self, image: np.ndarray) -> np.ndarray:
        """Relative depth for ``image``, estimated at most once per scene.

        ``image`` is one ``(H, W, 3)`` frame or an ``(N, H, W, 3)`` clip; a
        clip gets one ``(H, W)`` depth map, from its first frame. The returned
        array is shared between every caller and is therefore read-only; copy
        it before modifying it.
        """
        with self._lock:
            scene = self._scene_for(image)
            # A hit is a request for depth that did not need an estimate of
            # its own: every one for a scene after the first. Asking after a
            # scene's labels or its camera is not a request for depth, and
            # the first one that follows it is the one the estimate was for.
            self._hits += scene.asked
            scene.asked = True
            return scene.depth

    def _started_as(self, image: np.ndarray) -> np.ndarray | None:
        """The frame ``image`` was made from by transforms, if that is known and still there."""
        alive = [pair for pair in self._lineage if pair[0].is_alive() and pair[1]() is not None]
        self._lineage = alive
        for made, first in self._lineage:
            root = first()
            if root is not None and made.matches(image) and root.shape == image.shape:
                return root
        return None

    def _read_vehicles(
        self,
        scene: _Scene,
        frame: np.ndarray,
        source: np.ndarray | None = None,
        named: np.ndarray | None = None,
    ) -> None:
        """Read which end of each of a scene's vehicles is seen, where there is a reader.

        From the scene's labels, which say where its vehicles are: without
        them there is nothing to read, and a scene with no vehicle labelled
        is not shown to the network at all.

        ``source`` is the frame at the size it was taken at, where that is
        not ``frame``, and ``named`` its labels at that size. The reader
        looks at ``source``: a lamp that is three pixels in the frame a
        pipeline works at is a dozen in the photograph. What it finds is
        brought to the frame's own pixels.
        """
        scene.vehicles = scene.street_lamps = scene.signs = None
        if self._vehicles is None:
            return
        source = frame if source is None else source
        across = frame.shape[1] / source.shape[1]
        down = frame.shape[0] / source.shape[0]
        signs = np.asarray(self._vehicles.street_signs(source), dtype=np.float32).reshape(-1, 5)
        scene.signs = signs * np.array([across, down, across, down, 1.0], dtype=np.float32)
        # The reader's street lamps need no labels: they are found in the
        # frame, and a night frame is lit by them whatever else is in it.
        lamps = np.asarray(self._vehicles.street_lamps(source), dtype=np.float32).reshape(-1, 7)
        scene.street_lamps = lamps * np.array(
            [across, down, across, down, across, down, 1.0], dtype=np.float32
        )
        if scene.labels is None:
            return
        if named is None or named.shape != source.shape[:2]:
            named = scene.labels
            if named.shape != source.shape[:2]:
                named = cv2.resize(
                    named, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST
                )
        vehicles = named == int(SceneClass.VEHICLE)
        if not vehicles.any():
            scene.vehicles = []
            return
        reader = type(self._vehicles).__name__
        views = check_views(self._vehicles.estimate(source, vehicles), source=reader)
        if source.shape[:2] != frame.shape[:2]:
            views = [scaled_view(view, across, down, frame.shape[:2]) for view in views]
            views = [view for view in views if view.box[2] > 0 and view.box[3] > 0]
        scene.vehicles = views

    def get_vehicles(self, image: np.ndarray) -> list[VehicleView] | None:
        """Which end of each vehicle of ``image``'s scene is seen, or ``None`` with nothing to say.

        One :class:`~surmentations.perception.vehicles.VehicleView` for each
        outline of the scene's vehicle label that was read, known by its
        box. Made with the scene's depth and labels, from the same frame.
        ``None`` when the provider has no vehicle estimator or the scene no
        labels; an empty list for a scene with no vehicle in it.
        """
        with self._lock:
            return self._scene_for(image).vehicles

    def get_street_lamps(self, image: np.ndarray) -> np.ndarray | None:
        """The street lamps of ``image``'s scene as the vehicle reader found them, or ``None``.

        One row to a lamp: the left, top, right and bottom of its post in
        pixels, the column and row of its head, and a score
        (:func:`~surmentations.perception.vehicles.lamp_posts`). ``None``
        when the provider has no vehicle reader; an empty array for a scene
        in which it found none, or for a reader that does not look for them.
        """
        with self._lock:
            return self._scene_for(image).street_lamps

    def get_signs(self, image: np.ndarray) -> np.ndarray | None:
        """The road signs of ``image``'s scene as the vehicle reader found them, or ``None``.

        One row to a sign: its left, top, right and bottom in pixels and a
        score. ``None`` when the provider has no vehicle reader; an empty
        array where it found none or does not look for them.
        """
        with self._lock:
            return self._scene_for(image).signs

    def set_lit_air(self, image: np.ndarray, air: np.ndarray | None) -> bool:
        """Note the light that hangs in the air of one picture of a scene.

        ``air`` is ``(h, w, 4)`` float32, on any grid of the frame's shape.
        In its first three planes, for each line of sight, the light its
        lamps throw on the air along it, summed over its length, in the
        picture's own linear RGB times metres; in the fourth, how far off
        along the line that lit air is, in metres. Times an extinction
        coefficient the first is what fog, rain or snow in that air sends
        back to the lens, the cone under a street lamp and the wedge ahead
        of a headlamp, and the second is how much of it the same fog, rain
        or snow takes out again on the way.

        A transform that makes a night of a frame knows where its lamps
        are and notes this for the frame it returns; the transforms that
        put something in the air afterwards ask for it (:func:`lit_air`),
        and it follows the frame through them. It belongs to the picture
        and not to the scene: the same scene by day has none.

        Returns whether ``image`` is a frame this provider knows.
        """
        with self._lock:
            scene = self._find(image)
            alias = None if scene is None else scene.alias_of(image)
            if alias is None:
                return False
            alias.lit_air = air
            return True

    def get_labels(self, image: np.ndarray) -> np.ndarray | None:
        """What each pixel of ``image``'s scene is, or ``None`` with nothing to say.

        One :class:`~surmentations.perception.labels.SceneClass` per pixel,
        ``(H, W)`` uint8, read-only and shared like the depth. Made with the
        scene's depth, from the same frame, so a transform late in a
        pipeline is told about the scene and not about the fog over it.
        ``None`` when the provider has no label estimator and none were
        given through :meth:`remember`.
        """
        with self._lock:
            return self._scene_for(image).labels

    def get_geometry(self, image: np.ndarray) -> InstallationGeometry | None:
        """What is known of the camera's mounting for ``image``'s scene, or ``None``.

        The geometry the provider was given, or its estimator's reading of
        the frame. What it leaves open stays open here; :meth:`settle_camera`
        is what a transform calls.
        """
        with self._lock:
            return self._scene_for(image).geometry

    def remember(
        self,
        image: np.ndarray,
        *,
        labels: np.ndarray | None = None,
        geometry: InstallationGeometry | None = None,
        original: np.ndarray | None = None,
    ) -> None:
        """Tell the provider what is already known about ``image``, before a pipeline runs.

        ``labels`` are ground truth where a dataset has it: one
        :class:`~surmentations.perception.labels.SceneClass` per pixel,
        folded from the dataset's own classes. ``geometry`` is this frame's
        camera, for a dataset that mixes cameras whose mountings are known.
        Either takes the place of what an estimator would have made, and a
        geometry that leaves something open has it filled in by the
        provider's estimator if there is one.

        ``original`` is the same frame at the size it was taken at, where
        the pipeline works on a reduced copy. What names and finds things
        in a scene, its labels and its vehicle reader, is then run on the
        original, and what it finds is brought to the frame's pixels: a
        number plate or a lamp that is a few pixels in a frame 800 across
        is found in the photograph it was reduced from, and missed or
        misplaced in the reduction. Depth and the camera's geometry are
        still the frame's own. It must be the same picture: the same shape
        to within a fiftieth.

        Call it with the array the pipeline is about to be given: the scene
        is recognised by that array, as it is for depth.
        """
        with self._lock:
            scene = self._find(image)
            if scene is not None and original is not None:
                # Read again, from the original this time.
                self._scenes.remove(scene)
                scene = None
            if scene is None:
                self._scene_for(image, labels=labels, geometry=geometry, original=original)
                return
            if labels is not None:
                frame = _reference_frame(image)
                scene.labels = check_labels(labels, frame.shape[:2], source="remember()").view()
                scene.labels.flags.writeable = False
                self._read_vehicles(scene, frame)
            if geometry is not None:
                scene.geometry = geometry
            scene.fov = scene.pitch = scene.metric = None

    def settle_camera(
        self,
        image: np.ndarray,
        horizontal_fov_deg: float | None = None,
        pitch_deg: float | None = None,
    ) -> Camera | None:
        """The camera ``image``'s scene is seen through, the same for every caller.

        ``horizontal_fov_deg`` and ``pitch_deg`` are the caller's own draws.
        Where the scene's geometry says otherwise they are ignored. Where it
        leaves the field of view or the pitch open, the first caller's draw
        is kept for the scene, so that two transforms on one frame do not
        each see it through a camera of their own. A pitch drawn above a
        horizon that is known only as a bound is moved down to it.

        ``None`` when the provider has no geometry for the scene, or its
        field of view is neither known nor proposed; the caller then uses
        its own draws, as before there was a geometry to ask.
        """
        with self._lock:
            scene = self._scene_for(image)
            geometry = scene.geometry
            if geometry is None:
                return None
            height, width = _reference_frame(image).shape[:2]
            if scene.fov is None:
                known = geometry.horizontal_fov_deg
                scene.fov = horizontal_fov_deg if known is None else known
            if scene.fov is None:
                return None
            if scene.pitch is None:
                view = geometry.view(width, height, scene.fov, pitch_deg)
                scene.pitch = None if view is None else view.pitch_deg
            return Camera(
                width, height, scene.fov, scene.pitch, geometry.roll_deg, geometry.height_m
            )

    def settle_view(
        self, image: np.ndarray, horizontal_fov_deg: float, pitch_deg: float
    ) -> tuple[float, float]:
        """A transform's field of view and pitch, as :meth:`settle_camera` settles them.

        The two numbers a transform drew, replaced by the scene's camera
        where it has one: what a transform calls straight after drawing them.
        """
        camera = self.settle_camera(image, horizontal_fov_deg, pitch_deg)
        if camera is None:
            return horizontal_fov_deg, pitch_deg
        pitch = pitch_deg if camera.pitch_deg is None else camera.pitch_deg
        return camera.horizontal_fov_deg, pitch

    def horizon_row(self, image: np.ndarray) -> float | None:
        """Row of the horizon in ``image`` at its middle column, where the geometry fixes it.

        ``None`` when the provider has no geometry for the scene, the
        geometry leaves the tilt open or only bounds it, or no field of view
        has been settled yet. May be negative, or past the frame's last row.
        """
        with self._lock:
            geometry = self._scene_for(image).geometry
            if geometry is None or not geometry.fixes_tilt:
                return None
            camera = self.settle_camera(image)
            return None if camera is None else camera.horizon_row

    def settle_fov(self, image: np.ndarray, horizontal_fov_deg: float) -> float:
        """A transform's field of view, as :meth:`settle_camera` settles it.

        For a transform that draws a field of view and no pitch. Nothing is
        proposed for the pitch, which is left for a transform that draws one.
        """
        camera = self.settle_camera(image, horizontal_fov_deg)
        return horizontal_fov_deg if camera is None else camera.horizontal_fov_deg

    def _metric_of(self, scene: _Scene) -> tuple[float, float] | None:
        """The scale and offset that put ``scene``'s depth in metres, or ``None``.

        There are none without a camera whose height, field of view and
        pitch are all known and not merely proposed, a depth backend that
        gives its disparity, and enough ground to fit them to. Looked for
        once per scene.
        """
        if scene.metric is None:
            geometry, disparity = scene.geometry, scene.disparity
            height, width = scene.depth.shape
            camera = None if geometry is None else geometry.view(width, height, scene.fov)
            if geometry is not None and camera is None:
                # Its field of view is still open. A transform may yet
                # propose one, so this is not the last word on the scene.
                return None
            scene.metric = False
            if (
                camera is not None
                and disparity is not None
                and camera.height_m is not None
                and camera.pitch_deg is not None
            ):
                found = metric_scale(disparity, self._ground_of(scene, camera), camera)
                scene.metric = False if found is None else found
        return scene.metric if isinstance(scene.metric, tuple) else None

    def _ground_of(self, scene: _Scene, camera: Camera) -> np.ndarray:
        """Which of ``scene``'s pixels are ground, ``(H, W)`` bool.

        By the scene's labels where it has them. Without, by what faces up
        in the depth map as the camera is known to be tilted, which is what
        the wet road and the settled snow read too.
        """
        if scene.labels is not None:
            return class_weight(scene.labels, GROUND) > 0.5
        # Imported here: these import the compiled kernels, which nothing
        # else in this module needs.
        from surmentations.depth.normals import upward_facing
        from surmentations.utils.ground import level_ground_weight

        near, far = self.depth_range_m
        pseudo = to_pseudo_metric(scene.depth, far, near)
        assert camera.pitch_deg is not None
        facing = upward_facing(pseudo, camera.horizontal_fov_deg, camera.pitch_deg)
        level: np.ndarray = level_ground_weight(facing, 6.0) > 0.5
        return level

    def to_metric(self, relative_depth: np.ndarray, far_m: float | None = None) -> np.ndarray:
        """Relative depth in metres, through this provider's calibration.

        Real metres where the scene's camera is known well enough to measure
        from (:func:`~surmentations.perception.geometry.metric_scale`):
        the depth is then scaled so that the ground lies where a camera at
        that height and tilt would see it. Otherwise pseudo-metres, the
        nearest and farthest point of the scene placed at ``depth_range_m``.

        ``far_m`` overrides the far plane for one call, which is what
        ``RandomFog``'s ``depth_scale`` does, and is always in pseudo-metres:
        an explicit far plane is the caller's own calibration.
        """
        if far_m is None:
            with self._lock:
                held = [scene for scene in self._scenes if scene.depth is relative_depth]
                metric = self._metric_of(held[-1]) if held else None
                if metric is not None:
                    disparity = held[-1].disparity
                    assert disparity is not None
                    return metric_depth(disparity, *metric)
        near, far = self.depth_range_m
        return to_pseudo_metric(relative_depth, far if far_m is None else far_m, near)

    def get_metric_depth(self, image: np.ndarray) -> np.ndarray:
        """:meth:`get_depth` followed by :meth:`to_metric`."""
        return self.to_metric(self.get_depth(image))

    def describe(self, image: np.ndarray) -> dict[str, Any]:
        """What the provider knows about ``image``'s scene, for a log or a report.

        The geometry as read, the camera as settled so far, the scale and
        offset of metric depth if it is metric, and the share of the frame
        each scene class covers.
        """
        with self._lock:
            scene = self._scene_for(image)
            shares: dict[str, float] = {}
            if scene.labels is not None:
                counts = np.bincount(scene.labels.ravel(), minlength=len(SceneClass))
                shares = {
                    SceneClass(index).name: float(count) / scene.labels.size
                    for index, count in enumerate(counts)
                    if count
                }
            return {
                "geometry": scene.geometry,
                "camera": self.settle_camera(image),
                "metric": self._metric_of(scene),
                "label_shares": shares,
            }

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
                # Not a frame this provider has been asked about, yet. It
                # may be asked about what was made from it, and is then to
                # read the frame that came first and not this one.
                started_as = self._started_as(source)
                first = source if started_as is None else started_as
                self._lineage = [*self._lineage[-(_MAX_LINEAGE - 1) :]]
                self._lineage.append((_Alias(derived), weakref.ref(first)))
                return False
            before = scene.alias_of(source)
            scene.adopt(derived)
            after = scene.alias_of(derived)
            if before is not None and after is not None:
                # What hangs in the air of the one hangs in the other's.
                after.lit_air = before.lit_air
            return True

    def cache_info(self) -> DepthCacheInfo:
        """Hits, misses and scenes held — the way to check depth is being shared.

        A pipeline in which two depth-conditioned transforms fired on one
        image should show one miss and one hit for it, not two misses.
        """
        with self._lock:
            return DepthCacheInfo(self._hits, self._misses, len(self._scenes))

    def disk_cache_info(self) -> StoreInfo | None:
        """What the readings kept on disk have saved, or ``None`` where none are kept.

        Hits are frames whose reading was found there, and no network was
        run for; misses are frames read afresh; ``written`` and ``refused``
        say what became of those. Counted by this process: a ``DataLoader``
        worker has its own.
        """
        with self._lock:
            return None if self._store is None else self._store.info()


def settled_view(
    image: np.ndarray, horizontal_fov_deg: float, pitch_deg: float
) -> tuple[float, float]:
    """A field of view and a pitch, as whichever live provider knows the scene settles them.

    For a transform that draws a camera and has no provider of its own: the
    rain an infrared lamp lights, say. It is told what the provider of a
    transform earlier in the pipeline settled for this scene, and is given
    its own two numbers back when no provider holds the scene or has a
    geometry for it. Never estimates anything.

    A provider that holds the scene and has no geometry for it is passed
    over: with two providers alive on one frame, one of which knows the
    camera, the one that knows it answers.
    """
    for provider in list(_LIVE_PROVIDERS):
        with provider._lock:
            scene = provider._find(image)
            if scene is None or scene.geometry is None:
                continue
            return provider.settle_view(image, horizontal_fov_deg, pitch_deg)
    return horizontal_fov_deg, pitch_deg


def settled_fov(image: np.ndarray, horizontal_fov_deg: float) -> float:
    """A field of view, as whichever live provider knows the scene settles it.

    :func:`settled_view` for a transform that draws no pitch.
    """
    for provider in list(_LIVE_PROVIDERS):
        with provider._lock:
            scene = provider._find(image)
            if scene is None or scene.geometry is None:
                continue
            return provider.settle_fov(image, horizontal_fov_deg)
    return horizontal_fov_deg


def lit_air(image: np.ndarray) -> np.ndarray | None:
    """The light in the air of ``image``, as whichever live provider knows it; else ``None``.

    What :meth:`SharedDepthProvider.set_lit_air` noted for this picture or
    for the one it was made from. For a transform that puts rain, snow or
    fog in the air and has no provider of its own.
    """
    for provider in list(_LIVE_PROVIDERS):
        with provider._lock:
            scene = provider._find(image)
            alias = None if scene is None else scene.alias_of(image)
            if alias is not None and alias.lit_air is not None:
                return alias.lit_air
    return None


def carry_depth(source: np.ndarray, derived: np.ndarray) -> None:
    """Tell every live provider that ``derived`` shows ``source``'s scene.

    Called by :class:`~surmentations.base.CameraTransform` after each
    transform, so the scene survives a transform that has no provider of its
    own. A no-op when no provider holds depth for ``source``, which is the
    common case and costs one identity comparison per provider.
    """
    for provider in list(_LIVE_PROVIDERS):
        provider.carry(source, derived)
