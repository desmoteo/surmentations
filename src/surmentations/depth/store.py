"""What a provider read of a frame, kept on disk for the next time the frame is seen.

A :class:`~surmentations.depth.provider.SharedDepthProvider` holds a
handful of scenes in memory, which is enough to share one reading between
the transforms of a pipeline and no use at all across the epochs of a
training run: every frame of a training set is read again every epoch. With
a provider that names the scene and reads its vehicles that reading is
some 170 ms a frame on a GPU, more than everything the transforms then do
with it. A :class:`ReadingStore` keeps each reading in a file, so that the
second epoch and every later one finds it there.

It follows the provider's own cache in what it promises:

* **Keyed on the frame, by what is in it.** A file is found by a hash of
  the frame's pixels, and of the photograph it was reduced from where the
  provider was told of one. In memory a sparse fingerprint is enough to
  notice a buffer overwritten; on disk the frame has to be told from every
  other frame there is, so all of it is hashed, which is a millisecond.
* **And on what read it.** The estimators' own descriptions of themselves
  (their ``signature()``), the mode the reading was made in
  (:mod:`surmentations.mode`), and the library's version. A reading made
  with another checkpoint, in half precision for a caller who now asks for
  exact, or by an older release, is not this frame's reading and is not
  handed over. An estimator that does not describe itself is not kept at
  all.
* **Never a wrong answer.** A file that cannot be read, or holds something
  other than this frame's reading, is a miss: the frame is read again and
  the file replaced.

The version is the release's and not the code's. Readings made while an
estimator's code is being changed under one version number, or by an
estimator of the caller's own whose signature did not change when its
output did, are stale and are not known to be: delete the directory.

What is kept is what the networks made and nothing else: the depth, the
labels, the vehicles, the street lamps and the signs. The camera's
geometry and the disparity are worked out again from those, which takes
three milliseconds and means a file never holds an old version of them.

Where it is kept is the caller's choice: a directory given by name, or the
user's cache directory (:func:`default_directory`), which is where the
networks' own weights already are. It is never used unless asked for.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

import surmentations
from surmentations import mode
from surmentations.perception.vehicles import VehicleView

# The layout of a file. A file of another layout is not looked for: the
# number is part of every key.
_FORMAT = 1


class StoreInfo(NamedTuple):
    """Counters from :meth:`ReadingStore.info`.

    The fields:

    * ``hits``: readings found on disk and used.
    * ``misses``: frames looked for and not found, or found unreadable.
    * ``written``: readings put there by this process.
    * ``refused``: readings not written because the store was full.
    """

    hits: int
    misses: int
    written: int
    refused: int


class SceneReading(NamedTuple):
    """What the networks made of one frame.

    The fields are a scene's as the provider holds them: ``depth`` as the
    estimator returned it, ``labels`` at the frame's size, ``vehicles``,
    ``street_lamps`` and ``signs`` in the frame's pixels; each of the last
    four ``None`` where the provider had nothing to read it with.
    """

    depth: np.ndarray
    labels: np.ndarray | None
    vehicles: list[VehicleView] | None
    street_lamps: np.ndarray | None
    signs: np.ndarray | None


def default_directory() -> Path:
    """Where readings are kept when no directory is named.

    ``$SURMENTATIONS_CACHE`` if it is set; otherwise ``surmentations``
    under the user's cache directory, ``$XDG_CACHE_HOME`` or ``~/.cache``,
    beside the weights the estimators download.
    """
    named = os.environ.get("SURMENTATIONS_CACHE")
    if named:
        return Path(named).expanduser()
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root).expanduser() / "surmentations"


def _packed(reading: SceneReading) -> dict[str, np.ndarray]:
    """A reading as the named arrays of one file."""
    depth = np.asarray(reading.depth)
    # Half the size where nothing is lost by it: a map made in half
    # precision is its own float16 copy.
    half = depth.astype(np.float16)
    exact = depth.dtype == np.float32 and np.array_equal(half.astype(np.float32), depth)
    arrays: dict[str, np.ndarray] = {"depth": half if exact else depth}
    if reading.labels is not None:
        arrays["labels"] = np.asarray(reading.labels)
    if reading.street_lamps is not None:
        arrays["street_lamps"] = np.asarray(reading.street_lamps)
    if reading.signs is not None:
        arrays["signs"] = np.asarray(reading.signs)
    if reading.vehicles is not None:
        views = reading.vehicles
        arrays["boxes"] = np.array([view.box for view in views], dtype=np.int64).reshape(-1, 4)
        arrays["ends"] = np.array(
            [(view.front, view.rear) for view in views], dtype=np.float64
        ).reshape(-1, 2)
        arrays["kinds"] = np.array([view.kind for view in views], dtype=np.str_)
        arrays["plates"] = np.array(
            [view.plate if view.plate is not None else (np.nan,) * 4 for view in views],
            dtype=np.float64,
        ).reshape(-1, 4)
        for index, view in enumerate(views):
            arrays[f"lamps_{index}"] = np.asarray(view.lamps)
            if view.mask is not None:
                arrays[f"mask_{index}"] = np.asarray(view.mask)
    return arrays


def _unpacked(arrays: dict[str, np.ndarray]) -> SceneReading:
    """The reading a file's arrays hold; raises ``KeyError`` for a file that is not one."""
    depth = arrays["depth"].astype(np.float32)
    vehicles: list[VehicleView] | None = None
    if "boxes" in arrays:
        vehicles = []
        ends, kinds, plates = arrays["ends"], arrays["kinds"], arrays["plates"]
        for index, box in enumerate(arrays["boxes"].tolist()):
            plate = plates[index]
            vehicles.append(
                VehicleView(
                    box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    front=float(ends[index, 0]),
                    rear=float(ends[index, 1]),
                    lamps=arrays[f"lamps_{index}"],
                    mask=arrays.get(f"mask_{index}"),
                    kind=str(kinds[index]),
                    plate=None
                    if bool(np.isnan(plate).any())
                    else (float(plate[0]), float(plate[1]), float(plate[2]), float(plate[3])),
                )
            )
    return SceneReading(
        depth, arrays.get("labels"), vehicles, arrays.get("street_lamps"), arrays.get("signs")
    )


class ReadingStore:
    """Readings of frames in a directory, one file a frame.

    Args:
        directory: where the files are; made when the first is written.
        max_gb: the most the directory may hold, in gigabytes, or ``None``
            for no limit. A reading at 512 pixels is about half a megabyte
            on a busy road, most of it the depth. Past the limit nothing
            more is written and nothing is thrown out: the store goes on
            serving what it has.

    Several processes may share a directory, as the workers of a
    ``DataLoader`` do: a file is written whole under another name and then
    given its own, so none is ever seen half written. Each process counts
    what it wrote itself against the limit, so with many writers the
    directory can pass it by what the others wrote meanwhile.
    """

    def __init__(self, directory: str | os.PathLike[str], max_gb: float | None = None) -> None:
        """Keep the directory and the limit; raises ``ValueError`` for a limit below 0."""
        if max_gb is not None and max_gb < 0.0:
            raise ValueError(f"max_gb must be 0 or more, or None, got {max_gb}")
        self.directory = Path(directory).expanduser()
        self.max_gb = max_gb
        self._bytes: int | None = None
        self._hits = self._misses = self._written = self._refused = 0

    def key(self, frame: np.ndarray, source: np.ndarray, signatures: Sequence[str]) -> str:
        """The name a reading of ``frame`` goes by: a hash of it and of what reads it.

        ``source`` is the photograph ``frame`` was reduced from, or
        ``frame`` itself; ``signatures`` what each estimator says of
        itself. The mode (:mod:`surmentations.mode`), the library's
        version and the file's layout are in it too.
        """
        digest = hashlib.blake2b(digest_size=16)
        described = (_FORMAT, surmentations.__version__, mode.fast(), tuple(signatures))
        digest.update(repr(described).encode())
        for picture in (frame,) if source is frame else (frame, source):
            digest.update(repr((picture.shape, picture.dtype.str)).encode())
            digest.update(np.ascontiguousarray(picture).data)
        return digest.hexdigest()

    def _path(self, key: str) -> Path:
        """The file a key's reading is in: under its first two characters, so no folder is vast."""
        return self.directory / key[:2] / f"{key}.npz"

    def load(self, key: str, shape: tuple[int, int]) -> SceneReading | None:
        """The reading kept under ``key`` for a frame of ``shape``, or ``None``.

        ``None`` for a file that is not there, cannot be read, or holds a
        depth or labels of another shape: each is a frame to read again.
        """
        try:
            with np.load(self._path(key), allow_pickle=False) as held:
                reading = _unpacked({name: held[name] for name in held.files})
        except (OSError, ValueError, KeyError, IndexError, EOFError, zipfile.BadZipFile):
            self._misses += 1
            return None
        fits = reading.depth.shape == tuple(shape) and (
            reading.labels is None or reading.labels.shape == tuple(shape)
        )
        if not fits:
            self._misses += 1
            return None
        self._hits += 1
        return reading

    def _held_bytes(self) -> int:
        """How much the directory holds: counted once, then kept up as this process writes."""
        if self._bytes is None:
            self._bytes = sum(
                entry.stat().st_size for entry in self.directory.glob("*/*.npz") if entry.is_file()
            )
        return self._bytes

    def save(self, key: str, reading: SceneReading) -> bool:
        """Keep ``reading`` under ``key``; ``False`` if the store is full or cannot be written.

        A store that cannot be written (a full disk, a directory that is
        not the caller's) is not an error: the reading was made and is
        used, and is made again next time.
        """
        if self.max_gb is not None and self._held_bytes() >= self.max_gb * 1e9:
            self._refused += 1
            return False
        path = self._path(key)
        unfinished = path.with_name(f"{path.stem}.{os.getpid()}.part")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(unfinished, "wb") as file:
                np.savez_compressed(file, **_packed(reading))  # type: ignore[arg-type]
            size = unfinished.stat().st_size
            os.replace(unfinished, path)
        except OSError:
            with contextlib.suppress(OSError):
                unfinished.unlink(missing_ok=True)
            self._refused += 1
            return False
        if self._bytes is not None:
            self._bytes += size
        self._written += 1
        return True

    def info(self) -> StoreInfo:
        """Hits, misses, readings written and readings refused, by this process."""
        return StoreInfo(self._hits, self._misses, self._written, self._refused)
