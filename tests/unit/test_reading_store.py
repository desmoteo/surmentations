"""What a provider read of a frame, kept on disk and found again.

A ``SharedDepthProvider`` given ``disk_cache`` files what its networks make
of each frame, so that a frame seen again in another epoch, or another
run, is not read again. These tests hold it to what the provider's cache
in memory promises: the reading handed back is this frame's, made by these
estimators in this mode, or the frame is read afresh. Never a wrong answer.

The estimators here are counted stand-ins, so "read again" is a number.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

from surmentations import mode
from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.depth.store import ReadingStore, SceneReading, default_directory
from surmentations.perception import SceneClass, SceneLabelEstimator
from surmentations.perception.vehicles import VehicleView, VehicleViewEstimator

HEIGHT, WIDTH = 48, 64


class _Depth(DepthEstimator):
    """A depth that is the frame's own red channel, and a count of the times it was asked."""

    def __init__(self, name: str | None = "one") -> None:
        """Known by ``name``; ``None`` for an estimator that does not say what it is."""
        self.name = name
        self.calls = 0

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """The red channel in ``[0, 1]``, with a third of it not a float16."""
        self.calls += 1
        depth: np.ndarray = image[..., 0].astype(np.float32) / np.float32(255.0)
        return depth

    def signature(self) -> str | None:
        """Its name, where it has one."""
        return None if self.name is None else f"_Depth|{self.name}"


class _Labels(SceneLabelEstimator):
    """Road below the middle row, a vehicle in a box, sky above: counted."""

    def __init__(self) -> None:
        """Nothing asked yet."""
        self.calls = 0

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """The same scene whatever the frame."""
        self.calls += 1
        labels = np.full(image.shape[:2], int(SceneClass.SKY), dtype=np.uint8)
        labels[image.shape[0] // 2 :] = int(SceneClass.ROAD)
        labels[30:40, 10:30] = int(SceneClass.VEHICLE)
        return labels

    def signature(self) -> str | None:
        """Its class: it has no settings."""
        return "_Labels"


class _Reader(VehicleViewEstimator):
    """Two vehicles, one with a mask, a plate and a kind and one bare; a lamp and a sign."""

    def __init__(self) -> None:
        """Nothing asked yet."""
        self.calls = 0

    def estimate(self, image: np.ndarray, vehicles: np.ndarray) -> list[VehicleView]:
        """The same two views whatever the frame."""
        self.calls += 1
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 3:17] = True
        lamps = np.array([[12.0, 36.0, 1.5, 1.0, 0.8], [26.0, 36.0, 1.5, 1.0, 0.7]], np.float32)
        return [
            VehicleView((10, 30, 20, 10), 0.1, 0.9, lamps, mask, "car", (15.0, 37.0, 22.0, 39.0)),
            VehicleView((40, 28, 12, 9), 0.0, 0.0, np.zeros((0, 5), np.float32)),
        ]

    def street_lamps(self, image: np.ndarray) -> np.ndarray:
        """One lamp post."""
        return np.array([[50.0, 2.0, 53.0, 30.0, 51.0, 3.0, 0.6]], dtype=np.float32)

    def street_signs(self, image: np.ndarray) -> np.ndarray:
        """One sign."""
        return np.array([[5.0, 6.0, 9.0, 12.0, 0.4]], dtype=np.float32)

    def signature(self) -> str | None:
        """Its class: it has no settings."""
        return "_Reader"


def _frame(seed: int = 0) -> np.ndarray:
    """An 8-bit frame of the tests' size."""
    return np.random.default_rng(seed).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def _full(directory: Path, **given: object) -> tuple[SharedDepthProvider, list[object]]:
    """A provider that reads everything, keeping it in ``directory``, and its estimators."""
    estimators = [_Depth(), _Labels(), _Reader()]
    provider = SharedDepthProvider(
        estimators[0],  # type: ignore[arg-type]
        labels=estimators[1],  # type: ignore[arg-type]
        vehicles=estimators[2],  # type: ignore[arg-type]
        disk_cache=directory,
        **given,  # type: ignore[arg-type]
    )
    return provider, estimators


def _calls(estimators: list[object]) -> list[int]:
    """How often each estimator was asked."""
    return [estimator.calls for estimator in estimators]  # type: ignore[attr-defined]


def test_a_frame_read_once_is_not_read_again_by_another_provider(tmp_path: Path) -> None:
    """The second epoch, or the second run: a new provider and a new array
    of the same pixels. No network runs, and everything it hands back is
    what the first one made, to the bit."""
    first, asked = _full(tmp_path)
    frame = _frame()
    made = (
        first.get_depth(frame),
        first.get_labels(frame),
        first.get_vehicles(frame),
        first.get_street_lamps(frame),
        first.get_signs(frame),
    )
    assert _calls(asked) == [1, 1, 1]
    assert first.disk_cache_info() == (0, 1, 1, 0)

    second, asked_again = _full(tmp_path)
    again = frame.copy()
    assert np.array_equal(second.get_depth(again), made[0])
    assert second.get_depth(again).dtype == np.float32
    assert np.array_equal(second.get_labels(again), made[1])
    assert np.array_equal(second.get_street_lamps(again), made[3])
    assert np.array_equal(second.get_signs(again), made[4])
    views, kept = made[2], second.get_vehicles(again)
    assert views is not None and kept is not None and len(kept) == len(views) == 2
    for view, held in zip(views, kept, strict=True):
        assert held.box == view.box and held.kind == view.kind and held.plate == view.plate
        assert (held.front, held.rear) == (view.front, view.rear)
        assert np.array_equal(held.lamps, view.lamps) and held.lamps.dtype == view.lamps.dtype
        assert (held.mask is None) == (view.mask is None)
        if view.mask is not None:
            assert np.array_equal(held.mask, view.mask) and held.mask.dtype == np.bool_
    assert _calls(asked_again) == [0, 0, 0]
    assert second.disk_cache_info() == (1, 0, 0, 0)
    # In memory it is a scene like any other: asked again, it is a hit there.
    second.get_depth(again)
    assert second.cache_info()[:2] == (2, 1)
    assert second.disk_cache_info() == (1, 0, 0, 0)


def test_another_frame_another_mode_or_another_estimator_is_read_afresh(tmp_path: Path) -> None:
    """What a reading is filed under is the frame's pixels, the mode it was
    made in and what made it: change any, and it is not found."""
    first, _asked = _full(tmp_path)
    frame = _frame()
    first.get_depth(frame)

    other, asked = _full(tmp_path)
    changed = frame.copy()
    changed[HEIGHT - 1, WIDTH - 1, 2] ^= 1
    other.get_depth(changed)
    assert _calls(asked) == [1, 1, 1], "one bit of one pixel is another frame"

    was = mode.fast()
    try:
        mode.set_fast(not was)
        turned, asked = _full(tmp_path)
        turned.get_depth(frame.copy())
        assert _calls(asked) == [1, 1, 1], "a reading made in the other mode is not this one's"
    finally:
        mode.set_fast(was)

    renamed = SharedDepthProvider(_Depth("two"), disk_cache=tmp_path)
    depth_only = SharedDepthProvider(_Depth("one"), disk_cache=tmp_path)
    assert renamed.disk_cache_info() == (0, 0, 0, 0)
    renamed.get_depth(frame.copy())
    depth_only.get_depth(frame.copy())
    assert renamed.disk_cache_info() == (0, 1, 1, 0)
    assert depth_only.disk_cache_info() == (0, 1, 1, 0), "depth alone is not depth with labels"
    again = SharedDepthProvider(_Depth("one"), disk_cache=tmp_path)
    again.get_depth(frame.copy())
    assert again.disk_cache_info() == (1, 0, 0, 0)


def test_what_cannot_be_told_apart_is_not_kept(tmp_path: Path) -> None:
    """An estimator that does not say what it is, and a frame whose labels
    are the caller's own: neither is filed, and neither is looked for."""
    nameless = _Depth(None)
    provider = SharedDepthProvider(nameless, disk_cache=tmp_path)
    provider.get_depth(_frame())
    assert provider.disk_cache_info() == (0, 0, 0, 0)
    assert not list(tmp_path.rglob("*.npz"))

    given, asked = _full(tmp_path)
    frame = _frame()
    own = np.full((HEIGHT, WIDTH), int(SceneClass.OTHER), dtype=np.uint8)
    given.remember(frame, labels=own)
    assert np.array_equal(given.get_labels(frame), own)
    assert given.disk_cache_info() == (0, 0, 0, 0)
    assert not list(tmp_path.rglob("*.npz"))
    assert _calls(asked)[1] == 0


def test_a_file_that_is_not_a_reading_is_a_frame_to_read_again(tmp_path: Path) -> None:
    """Cut short, overwritten, or of another frame's shape: read afresh and
    replaced, and never an error or another frame's depth."""
    first, _asked = _full(tmp_path)
    frame = _frame()
    expected = first.get_depth(frame).copy()
    (file,) = tmp_path.rglob("*.npz")
    whole = file.read_bytes()
    for damaged in (whole[: len(whole) // 2], b"not a file of arrays", b""):
        file.write_bytes(damaged)
        provider, asked = _full(tmp_path)
        assert np.array_equal(provider.get_depth(frame.copy()), expected)
        assert _calls(asked) == [1, 1, 1]
        assert provider.disk_cache_info() == (0, 1, 1, 0)
        assert file.read_bytes() != damaged
    # A reading of another size under this frame's name.
    store = ReadingStore(tmp_path)
    small = SceneReading(np.zeros((4, 4), np.float32), None, None, None, None)
    assert store.save(file.stem, small)
    provider, asked = _full(tmp_path)
    assert np.array_equal(provider.get_depth(frame.copy()), expected)
    assert _calls(asked) == [1, 1, 1]


def test_a_store_keeps_a_reading_whole(tmp_path: Path) -> None:
    """Depth that is not a float16's worth is kept as it is, depth that is
    takes half the room, and what is ``None`` comes back ``None``."""
    store = ReadingStore(tmp_path)
    rng = np.random.default_rng(3)
    fine = rng.uniform(size=(HEIGHT, WIDTH)).astype(np.float32)
    coarse = fine.astype(np.float16).astype(np.float32)
    for name, depth in (("aa" + "0" * 30, fine), ("bb" + "0" * 30, coarse)):
        assert store.save(name, SceneReading(depth, None, None, None, None))
        held = store.load(name, (HEIGHT, WIDTH))
        assert held is not None and np.array_equal(held.depth, depth)
        assert held.depth.dtype == np.float32
        assert held.labels is None and held.vehicles is None
        assert held.street_lamps is None and held.signs is None
    sizes = {file.parent.name: file.stat().st_size for file in tmp_path.rglob("*.npz")}
    assert sizes["bb"] < 0.75 * sizes["aa"]
    # No vehicles read is not the same as none found.
    none_found = SceneReading(coarse, None, [], None, None)
    assert store.save("cc" + "0" * 30, none_found)
    held = store.load("cc" + "0" * 30, (HEIGHT, WIDTH))
    assert held is not None and held.vehicles == []
    assert store.load("dd" + "0" * 30, (HEIGHT, WIDTH)) is None
    assert store.info() == (3, 1, 3, 0)


def test_a_full_store_stops_growing_and_goes_on_serving(tmp_path: Path) -> None:
    """Past its limit nothing more is written and nothing is thrown out."""
    first, _asked = _full(tmp_path, disk_cache_max_gb=1e-9)
    kept, turned_away = _frame(1), _frame(2)
    first.get_depth(kept)
    first.get_depth(turned_away)
    assert first.disk_cache_info() == (0, 2, 1, 1)
    assert len(list(tmp_path.rglob("*.npz"))) == 1
    later, asked = _full(tmp_path, disk_cache_max_gb=1e-9)
    later.get_depth(kept.copy())
    assert _calls(asked) == [0, 0, 0]
    later.get_depth(turned_away.copy())
    assert _calls(asked) == [1, 1, 1]
    assert later.disk_cache_info() == (1, 1, 0, 1)
    with pytest.raises(ValueError, match="disk_cache_max_gb"):
        SharedDepthProvider(_Depth(), disk_cache=tmp_path, disk_cache_max_gb=-1.0)


def test_a_directory_that_cannot_be_written_costs_nothing_but_the_reading(tmp_path: Path) -> None:
    """A store is a saving and not a need: where it cannot write, the frame
    is read, used, and read again next time."""
    blocked = tmp_path / "a file where the directory should be"
    blocked.write_text("in the way")
    provider, asked = _full(blocked)
    frame = _frame()
    assert provider.get_depth(frame).shape == (HEIGHT, WIDTH)
    assert _calls(asked) == [1, 1, 1]
    assert provider.disk_cache_info() == (0, 1, 0, 1)


def test_a_worker_is_sent_where_the_readings_are_and_none_by_default(tmp_path: Path) -> None:
    """Pickled, as a ``DataLoader`` sends it to a worker, a provider keeps
    its directory and its limit and starts its own count. Without
    ``disk_cache`` nothing is kept anywhere."""
    provider, _asked = _full(tmp_path, disk_cache_max_gb=2.0)
    frame = _frame()
    provider.get_depth(frame)
    # A provider pickled here a line earlier, as a DataLoader pickles its
    # own for a worker: nothing from outside is unpickled.
    worker = pickle.loads(pickle.dumps(provider))
    assert worker.disk_cache_info() == (0, 0, 0, 0)
    worker.get_depth(frame.copy())
    assert worker.disk_cache_info() == (1, 0, 0, 0)
    assert worker._disk_cache_max_gb == 2.0

    plain = SharedDepthProvider(_Depth())
    plain.get_depth(frame.copy())
    assert plain.disk_cache_info() is None
    assert pickle.loads(pickle.dumps(plain)).disk_cache_info() is None


def test_the_usual_place_is_the_user_s_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``disk_cache=True`` is ``surmentations`` under the user's cache
    directory, or wherever ``SURMENTATIONS_CACHE`` says."""
    monkeypatch.delenv("SURMENTATIONS_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_directory() == tmp_path / "xdg" / "surmentations"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert default_directory() == Path.home() / ".cache" / "surmentations"
    monkeypatch.setenv("SURMENTATIONS_CACHE", str(tmp_path / "named"))
    assert default_directory() == tmp_path / "named"
    provider = SharedDepthProvider(_Depth(), disk_cache=True)
    provider.get_depth(_frame())
    assert len(list((tmp_path / "named").rglob("*.npz"))) == 1


def test_a_reduced_frame_is_read_from_the_reading_kept_for_its_original(tmp_path: Path) -> None:
    """A loader that runs the pipeline at the size the network sees: the
    frame at its own size was read once and kept; its reduction is read from
    that, with no network run and nothing new kept, every reading at the
    reduced frame's size."""
    first, asked = _full(tmp_path)
    original = _frame()
    first.get_depth(original)
    assert _calls(asked) == [1, 1, 1]

    second, asked_again = _full(tmp_path)
    again = original.copy()
    small = cv2.resize(again, (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_AREA)
    second.resized(small, again)
    assert _calls(asked_again) == [0, 0, 0], "a network ran for the reduced frame"
    assert second.disk_cache_info() == (1, 0, 0, 0)

    depth = second.get_depth(small)
    assert depth.shape == small.shape[:2]
    expected = cv2.resize(
        first.get_depth(original), (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_AREA
    )
    assert np.allclose(depth, expected, atol=1e-6)
    labels = second.get_labels(small)
    assert labels is not None and labels.shape == small.shape[:2]
    assert (labels == int(SceneClass.VEHICLE)).any()
    views = second.get_vehicles(small)
    assert views is not None and len(views) == 2
    assert views[0].box == (5, 15, 10, 5), "the vehicle's box is halved"
    assert np.allclose(views[0].lamps[:, :4], [[6.0, 18.0, 0.75, 0.5], [13.0, 18.0, 0.75, 0.5]])
    assert np.allclose(second.get_street_lamps(small)[0, :6], [25.0, 1.0, 26.5, 15.0, 25.5, 1.5])
    assert np.allclose(second.get_signs(small)[0, :4], [2.5, 3.0, 4.5, 6.0])
    assert second.disk_cache_info() == (1, 0, 0, 0), "nothing kept for the reduced frame"


def test_a_frame_that_is_not_the_original_reduced_is_refused(tmp_path: Path) -> None:
    provider, _asked = _full(tmp_path)
    with pytest.raises(ValueError, match="same picture"):
        provider.resized(np.zeros((HEIGHT // 2, WIDTH, 3), np.uint8), _frame())
