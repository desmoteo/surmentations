"""What an augmentation leaves legible, object by object, and the labels of what it did not.

Every transform in this library changes the picture and, with few
exceptions, leaves the labels as they were. A box round a car is still
there after fog has taken the car, after a lost slice has smeared it away,
or after a night has been made of the frame and the car stands in the dark
beyond the illuminator's reach. A network trained on that frame is told a
car is there, and learns to find cars in grey fog and in the dark.

:class:`LabelLegibility`, last in a pipeline, judges each labelled object
in the frame it receives against the frame the pipeline was given, and
does one of three things with the label of an object that can no longer
be made out (``docs/label_legibility_assessment.md`` is the design):

* **background**: where the object's pixels were replaced (tape, a cap of
  snow, a privacy mask, a leaf), the label goes, and a semantic mask's
  pixels become background. Only the transforms that replace pixels say
  so, exactly, so this is never a guess.
* **removed**, or **removed and blanked**: the label goes, and with
  ``"blank"`` the object is painted out of the picture with its
  surroundings, so that the background the network is told about is true.
* **suppressed**: the label goes from the labels and its object's region
  is returned under ``ignore_regions``, for a training loss to leave out:
  no positive there, and no negative either. A semantic mask's pixels
  become ``ignore_index``.

Every judgement short of "replaced" can only remove or suppress a label,
never make one background on a guess.

How the frame is judged
-----------------------
While a :class:`LabelLegibility` exists, each transform's call is noted
against the frame it was given (:func:`note`, from
:meth:`~surmentations.base.CameraTransform.apply_with_params`), and the
record follows the frame through the pipeline. A transform that knows what
it took reports it in planes, at a quarter of the frame's size
(:meth:`~surmentations.base.CameraTransform.legibility_planes`):

* ``hidden``: the share of each pixel replaced, from a privacy mask, tape
  or paint, a cap of snow, a leaf, frost, an insect on the housing;
* ``blur``: the spread of a point, in pixels, from a lens out of focus or
  the air's turbulence.

The rest is measured. Inside each object the finished frame is compared
with the original on where its edges are at the object's own scale: how
much of their pattern survives, and how strong they are against the
frame's noise. A person sees an edge that is a few times the noise
(Rose); the pattern that survives, at the resolution the blur leaves,
gives the cycles a person could still resolve across the object, and
Johnson's criteria the probability the task (``"detect"``,
``"recognise"``, ``"identify"``) can still be done. That is taken relative
to the clean frame's, so a clean frame keeps every label, and the dataset's
own small and faint objects are not questioned: augmentation can only take
labels away.
"""

from __future__ import annotations

import functools
import weakref
from collections import deque
from collections.abc import Callable
from typing import Any, ClassVar, Literal

import cv2
import numpy as np
from albumentations.core.transforms_interface import BaseTransformInitSchema
from albumentations.core.type_definitions import Targets
from pydantic import Field

from surmentations.base import CameraTransform
from surmentations.diagnostics import record_draw
from surmentations.utils.photometry import luminance, to_linear_unit

#: Cycles across an object's smaller dimension at which each task is done
#: half the time (Johnson, 1958; the 50% points still used with the
#: target transfer probability function).
TASK_CYCLES: dict[str, float] = {"detect": 1.0, "recognise": 4.0, "identify": 6.4}

# The planes are kept at this share of the frame's size.
_PLANE_SCALE = 4
# How many frames' records are kept: a DataLoader worker has one in
# flight, a notebook a few.
_RECENT = 32
# An octave of the edges is resolved where at least this share of the clean
# frame's edge energy at that octave is still seen: the edges stand clear of
# the noise, and in the same pattern.
_RESOLVED = 0.5
# Below this probability on the clean frame an object is too small or too
# faint for the measure to say anything: its label is the annotator's, and
# kept.
_LEAST_CLEAN_PROBABILITY = 0.05
# The outline counts: the object is judged with a margin of this share of
# its smaller side round it, and no more than the second, in pixels.
_MARGIN_SHARE = 0.25
_MOST_MARGIN = 32
# The frames are compared in log light, offset by this much linear light so
# that black is finite: about a level and a half of an 8-bit frame.
_LOG_OFFSET = 0.002
# The least noise a frame is taken to have, in log light: half a level of an
# 8-bit frame at mid grey, so that a clean object's ratio is finite.
_NOISE_FLOOR = 0.01
# A Gaussian noise's deviation from its mean absolute value.
_ABS_TO_SIGMA = 1.2533
# Rose's criterion as it is applied here: an edge is seen at this many
# times the noise. Rose's 5 is for a uniform disc against noise counted
# pixel by pixel; signal and noise here are both in an edge map blurred to
# its octave, where the noise is already averaged over that much.
_ROSE = 3.0
# Fewer visible pixels than this and an object is too small to judge: it
# is kept, unless it is hidden.
_LEAST_PIXELS = 16
# How many pixels of blur sigma make one period: a reported blur caps the
# cycles across an object, whatever the edges say.
_PERIOD_PER_SIGMA = 2.5
# A keypoint is judged on a square this many pixels across.
_KEYPOINT_SQUARE = 7

# The filters alive: while there is one, every transform's call is noted.
_FILTERS: weakref.WeakSet[LabelLegibility] = weakref.WeakSet()


def _fingerprint(image: np.ndarray) -> tuple[Any, ...]:
    """A cheap fingerprint of an array's contents, so a frame changed in place is not mistaken."""
    frame = image if image.ndim < 4 else image[0]
    step = max(1, max(frame.shape[:2]) // 16)
    return (image.shape, image.dtype.str, frame[::step, ::step].tobytes())


class _Record:
    """What the pipeline has done to one frame since it was given it."""

    __slots__ = ("aliases", "blur", "hidden", "original")

    def __init__(self, original: np.ndarray) -> None:
        """Start the record of ``original``, with nothing yet taken from it."""
        self.original = original
        self.hidden: np.ndarray | None = None
        self.blur: np.ndarray | None = None
        self.aliases: list[tuple[weakref.ref[np.ndarray], tuple[Any, ...]]] = []
        self.alias(original)

    def alias(self, image: np.ndarray) -> None:
        """Note that ``image`` shows this frame, further down the pipeline."""
        self.aliases.append((weakref.ref(image), _fingerprint(image)))

    def add(self, planes: dict[str, Any], shape: tuple[int, int]) -> None:
        """Fold what one transform took into the planes.

        A plane is ``(h, w)``, ``(h, w, channels)`` (reduced to its mean
        over the channels) or a scalar, on any grid of the frame's shape.
        Shares hidden combine as independent covers do; blurs add in
        quadrature.
        """
        small = (max(1, shape[1] // _PLANE_SCALE), max(1, shape[0] // _PLANE_SCALE))
        for name, value in planes.items():
            plane = np.asarray(value, dtype=np.float32)
            if plane.ndim == 3:
                plane = plane.mean(axis=2)
            if plane.ndim == 0:
                plane = np.full(small[::-1], float(plane), dtype=np.float32)
            elif plane.shape != small[::-1]:
                resized = cv2.resize(plane, small, interpolation=cv2.INTER_AREA)
                plane = np.asarray(resized, dtype=np.float32)
            if name == "hidden":
                plane = np.clip(plane, 0.0, 1.0)
                self.hidden = (
                    plane if self.hidden is None else 1.0 - (1.0 - self.hidden) * (1.0 - plane)
                )
            elif name == "blur":
                plane = np.maximum(plane, 0.0)
                self.blur = plane if self.blur is None else np.sqrt(self.blur**2 + plane**2)
            else:
                raise KeyError(f"no such legibility plane: {name!r}")


_RECORDS: deque[_Record] = deque(maxlen=_RECENT)


def tracking() -> bool:
    """Whether the transforms' calls are being noted: while a :class:`LabelLegibility` exists."""
    return len(_FILTERS) > 0


def record_of(image: np.ndarray) -> _Record | None:
    """The record of the frame ``image`` shows; ``None`` for a frame no transform was noted on."""
    fingerprint = None
    for record in reversed(_RECORDS):
        for ref, noted in record.aliases:
            if ref() is image:
                fingerprint = _fingerprint(image) if fingerprint is None else fingerprint
                if noted == fingerprint:
                    return record
    return None


def note(transform: Any, params: dict[str, Any], source: np.ndarray, derived: np.ndarray) -> None:
    """Note that ``transform`` turned ``source`` into ``derived``, and what it took from legibility.

    The first note on a frame makes it the record's original. A transform
    reports its planes through ``legibility_planes``; one that reports
    none is measured by :class:`LabelLegibility` instead.
    """
    record = record_of(source)
    if record is None:
        record = _Record(source)
        _RECORDS.append(record)
    planes = transform.legibility_planes(params, source, derived)
    if planes:
        record.add(planes, source.shape[:2])
    if derived is not source:
        record.alias(derived)


def task_probability(cycles: np.ndarray | float, needed: float) -> np.ndarray | float:
    """The probability a task needing ``needed`` cycles is done with ``cycles`` across the object.

    The target transfer probability function used with Johnson's criteria:
    ``(N/N50)^E / (1 + (N/N50)^E)``, ``E = 2.7 + 0.7 N/N50``.
    """
    ratio = np.maximum(np.asarray(cycles, dtype=np.float64) / max(needed, 1e-6), 0.0)
    power = ratio ** (2.7 + 0.7 * ratio)
    return power / (1.0 + power)


def _grey(image: np.ndarray) -> np.ndarray:
    """A frame's luminance in log light, ``(H, W)`` float32.

    In log light a change of lighting that varies slowly across the frame,
    a night made of a day or an infrared lamp's fall-off, is an offset that
    varies slowly, and the edges of what things are made of stay where and
    as strong as they were. The noise of a dark part is as large as it is
    against the little light there.
    """
    linear = to_linear_unit(image)[0]
    if linear.ndim == 3 and linear.shape[2] >= 3:
        grey = luminance(linear[..., :3])
    else:
        grey = linear.reshape(linear.shape[0], linear.shape[1], -1)[..., 0]
    logged: np.ndarray = np.log(np.maximum(grey, 0.0) + _LOG_OFFSET).astype(np.float32)
    return logged


def _edges(grey: np.ndarray, scale: float) -> np.ndarray:
    """The edges of ``grey`` at ``scale`` pixels: the gradient's magnitude after a blur to it.

    A magnitude, so that what is compared is where the edges are and how
    strong, not which side is the brighter: a night made of a day, or an
    infrared frame, turns a red bus dark and its glass bright and leaves
    every edge where it was, and the bus is as legible as it was.
    """
    blurred = cv2.GaussianBlur(grey, (0, 0), sigmaX=max(0.7, scale / 2.0))
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude: np.ndarray = np.sqrt(gx * gx + gy * gy)
    return magnitude


@functools.lru_cache(maxsize=64)
def _edge_noise_gain(scale: float) -> float:
    """The root mean square, in :func:`_edges` at ``scale``, of a pixel noise of deviation 1.

    The level an edge map stands at where there is nothing but noise:
    what is taken off each edge before it is compared, and what an edge
    must stand clear of to be seen.
    """
    noise = np.random.default_rng(0).standard_normal((128, 128)).astype(np.float32)
    inner = _edges(noise, scale)[24:-24, 24:-24]
    return float(np.sqrt((inner * inner).mean()))


def _local_noise(grey: np.ndarray, spread: float) -> np.ndarray:
    """The grain of ``grey`` about each pixel: the mean size of what a 3x3 median removes, nearby.

    Taken from the mean absolute residual, which for Gaussian noise is
    0.8 of its deviation, over a Gaussian window of ``spread`` pixels. Fine
    texture counts as grain too, on the clean frame and the augmented one
    alike. Never below :data:`_NOISE_FLOOR`.
    """
    residual = np.abs(grey - cv2.medianBlur(np.ascontiguousarray(grey, dtype=np.float32), 3))
    local = cv2.GaussianBlur(residual, (0, 0), sigmaX=spread) * np.float32(_ABS_TO_SIGMA)
    floored: np.ndarray = np.maximum(local, np.float32(_NOISE_FLOOR))
    return floored


def _seen_at(
    clean: np.ndarray, grey: np.ndarray, visible: np.ndarray
) -> tuple[float, float] | None:
    """How much of the clean frame's edges at this level each frame shows, or ``None`` for none.

    Both frames' edges are measured with each pixel's noise taken off: noise
    raises an edge map everywhere, and most in the dark, where in log light
    it is largest, and left on would read as edges where the clean frame has
    none. Of the clean frame's edge energy, the share where the edges stand
    at least Rose's multiple clear of the noise: on the clean frame, and on
    the augmented one times the cosine of the two maps, the pattern that
    survives. ``None`` where the clean frame has no edges above its grain
    at this level: nothing to judge by.
    """
    gain = np.float32(_edge_noise_gain(1.0))
    noise_before = _local_noise(clean, 2.0)[visible] * gain
    noise_after = _local_noise(grey, 2.0)[visible] * gain
    before = np.sqrt(np.maximum(_edges(clean, 1.0)[visible] ** 2 - noise_before**2, 0.0))
    after = np.sqrt(np.maximum(_edges(grey, 1.0)[visible] ** 2 - noise_after**2, 0.0))
    weight = before * before
    energy = float(weight.sum())
    if energy <= 1e-9:
        return None
    pattern = float((before * after).sum()) / np.sqrt(
        energy * max(float((after * after).sum()), 1e-12)
    )
    seen_before = float((weight * (before >= _ROSE * noise_before)).sum()) / energy
    seen_after = float((weight * (after >= _ROSE * noise_after)).sum()) / energy
    return seen_before, float(np.clip(pattern, 0.0, 1.0)) * seen_after


def _octaves(
    clean: np.ndarray, grey: np.ndarray, visible: np.ndarray, min_dim: float
) -> list[tuple[float, float]]:
    """What each frame shows of the clean edges, octave by octave, from 2 px to the object's size.

    A pyramid: each octave halves both frames, so that the edges are always
    measured at one pixel's scale, and an octave ``k`` is detail of a
    period of ``2 * 2**k`` pixels of the frame. A pair for each octave,
    clean and augmented; ``(-1, -1)`` for one with nothing to judge by.
    """
    octaves: list[tuple[float, float]] = []
    shown = visible.astype(np.float32)
    level = 0
    while 2 * 2**level <= min_dim and min(clean.shape) >= 4:
        mask = shown > 0.5
        if int(mask.sum()) < _LEAST_PIXELS // 2:
            break
        seen = _seen_at(clean, grey, mask)
        octaves.append((-1.0, -1.0) if seen is None else seen)
        clean = cv2.pyrDown(clean).astype(np.float32)
        grey = cv2.pyrDown(grey).astype(np.float32)
        shown = cv2.pyrDown(shown).astype(np.float32)
        level += 1
    return octaves


def _resolved_cycles(octaves: list[tuple[float, float]], which: int, min_dim: float) -> float:
    """The cycles a person resolves across the object in one frame: 0 clean, 1 augmented.

    At the finest octave resolved (:data:`_RESOLVED`), whatever the coarser
    ones show: an object's detail may be all fine, as a pattern's is, and
    average out to nothing at the coarse octaves. Noise does not make an
    octave resolved: the edges are counted with the noise taken off, where
    the clean ones are, and only where they stand clear of it. With no
    octave resolved, the cycles of the best one, in proportion to what it
    shows.
    """
    best = (0.0, -1)
    for level, pair in enumerate(octaves):
        seen = pair[which]
        if seen >= _RESOLVED:
            return min_dim / (2.0 * 2**level)
        if seen > best[0]:
            best = (seen, level)
    seen, level = best
    if level < 0:
        return 0.0 if octaves else min_dim / 2.0
    return min_dim / (2.0 * 2**level) * seen / _RESOLVED


class _Judge:
    """Scores the objects of one frame against the frame the pipeline was given."""

    def __init__(
        self,
        original: np.ndarray | None,
        image: np.ndarray,
        hidden: np.ndarray | None,
        blur: np.ndarray | None,
        needed: float,
    ) -> None:
        """Hold the frame and its clean original in log grey, and the planes at full size."""
        height, width = image.shape[:2]
        self.shape = (height, width)
        self.needed = needed
        self.grey = _grey(image)
        self.clean = None if original is None else _grey(original)
        full = (width, height)
        self.hidden = (
            np.zeros((height, width), dtype=np.float32)
            if hidden is None
            else cv2.resize(hidden, full, interpolation=cv2.INTER_LINEAR)
        )
        self.blur = None if blur is None else cv2.resize(blur, full, interpolation=cv2.INTER_LINEAR)
        #: What each object scored, in the order judged: for the record, and
        #: for anyone tuning the thresholds.
        self.details: list[dict[str, float]] = []

    def rect(self, x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
        """Pixel bounds of a box, clipped to the frame."""
        height, width = self.shape
        return (
            max(0, int(np.floor(x0))),
            max(0, int(np.floor(y0))),
            min(width, int(np.ceil(x1))),
            min(height, int(np.ceil(y1))),
        )

    def score(
        self, region: np.ndarray | None, bounds: tuple[int, int, int, int]
    ) -> tuple[float, float]:
        """How much of an object is hidden, and the probability the task can still be done on it.

        ``bounds`` is the object's pixel rectangle; ``region`` a boolean
        mask of its own pixels over the frame, or ``None`` for the whole
        rectangle. The second value is relative to the clean frame: 1
        where nothing was taken, whatever the object's own difficulty.
        """
        x0, y0, x1, y1 = bounds
        if x1 <= x0 or y1 <= y0:
            return 0.0, 1.0
        inside = np.ones((y1 - y0, x1 - x0), dtype=bool) if region is None else region[y0:y1, x0:x1]
        hidden = float(self.hidden[y0:y1, x0:x1][inside].mean()) if inside.any() else 0.0
        if self.clean is None:
            return hidden, 1.0
        min_dim = float(min(x1 - x0, y1 - y0))
        # The object with a margin round it, so that its outline counts,
        # less what was replaced.
        margin = min(_MOST_MARGIN, int(np.ceil(_MARGIN_SHARE * min_dim)))
        height, width = self.shape
        X0, Y0 = max(0, x0 - margin), max(0, y0 - margin)
        X1, Y1 = min(width, x1 + margin), min(height, y1 + margin)
        visible = self.hidden[Y0:Y1, X0:X1] < 0.5
        if region is not None:
            grown = cv2.dilate(
                region[Y0:Y1, X0:X1].astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)
            )
            visible &= grown.astype(bool)
        if int(visible.sum()) < _LEAST_PIXELS:
            return hidden, 1.0
        octaves = _octaves(self.clean[Y0:Y1, X0:X1], self.grey[Y0:Y1, X0:X1], visible, min_dim)
        clean_cycles = _resolved_cycles(octaves, 0, min_dim)
        cycles = _resolved_cycles(octaves, 1, min_dim)
        if self.blur is not None and inside.any():
            sigma = float(self.blur[y0:y1, x0:x1][inside].mean())
            if sigma > 0.0:
                cycles = min(cycles, min_dim / (_PERIOD_PER_SIGMA * sigma))
        clean_probability = float(task_probability(clean_cycles, self.needed))
        probability = (
            1.0
            if clean_probability < _LEAST_CLEAN_PROBABILITY
            else min(1.0, float(task_probability(cycles, self.needed)) / clean_probability)
        )
        self.details.append(
            {
                "min_dim_px": min_dim,
                "hidden": hidden,
                "cycles_before": clean_cycles,
                "cycles_after": cycles,
                "probability": probability,
                **{
                    f"{name}_{2 * 2**k}px": value
                    for k, pair in enumerate(octaves)
                    for name, value in zip(("clean", "seen"), pair, strict=True)
                    if value >= 0
                },
            }
        )
        return hidden, probability


def _blank(
    image: np.ndarray,
    regions: list[tuple[np.ndarray | None, tuple[int, int, int, int]]],
    protected: np.ndarray,
) -> np.ndarray:
    """``image`` with each region painted out by its surroundings, except where ``protected``.

    The fill is a normalised blur of what is round the region and not in
    it, so the object's own colour does not survive as a blob; and pixels
    of the objects whose labels are kept (``protected``) are left as they
    are, so that painting out one object does not damage its neighbour.
    """
    out = image.copy()
    height, width = image.shape[:2]
    for region, (x0, y0, x1, y1) in regions:
        if x1 <= x0 or y1 <= y0:
            continue
        sigma = max(3.0, min(x1 - x0, y1 - y0) / 2.0)
        margin = int(np.ceil(3 * sigma))
        X0, Y0 = max(0, x0 - margin), max(0, y0 - margin)
        X1, Y1 = min(width, x1 + margin), min(height, y1 + margin)
        object_here = np.zeros((Y1 - Y0, X1 - X0), dtype=bool)
        object_here[y0 - Y0 : y1 - Y0, x0 - X0 : x1 - X0] = (
            True if region is None else region[y0:y1, x0:x1]
        )
        patch = out[Y0:Y1, X0:X1].astype(np.float32)
        around = (~object_here).astype(np.float32)
        weight = cv2.GaussianBlur(around, (0, 0), sigmaX=sigma)
        spread = cv2.GaussianBlur(
            patch * (around[..., None] if patch.ndim == 3 else around), (0, 0), sigmaX=sigma
        )
        fill = (
            spread / np.maximum(weight, 1e-6)[..., None]
            if patch.ndim == 3
            else spread / (np.maximum(weight, 1e-6))
        )
        painted = object_here & ~protected[Y0:Y1, X0:X1] & (weight > 1e-3)
        patch[painted] = fill[painted]
        if np.issubdtype(image.dtype, np.integer):
            info = np.iinfo(image.dtype)
            out[Y0:Y1, X0:X1] = np.clip(np.round(patch), info.min, info.max).astype(image.dtype)
        else:
            out[Y0:Y1, X0:X1] = patch.astype(image.dtype)
    return out


class LabelLegibility(CameraTransform):
    """Keep, remove, blank or suppress each label by whether its object can still be made out.

    Put it last in the pipeline, after every transform that changes the
    picture. It changes the picture only where asked to blank an object.

    Args:
        task: what a label is for: ``"detect"``, ``"recognise"`` or
            ``"identify"`` (Johnson's 1, 4 and 6.4 cycles across the
            object), or a number of cycles. A plate reader needs
            identification; a detector that names classes, recognition.
        keep_at: a label is kept where the probability the task can still
            be done, relative to the clean frame, is at least this.
        gone_at: a label is made background where at least this share of
            its object's pixels were replaced: the object is not there.
        on_doubt: what to do with a label kept by neither rule.
            ``"ignore"`` suppresses it: it goes from the labels, and its
            region is returned under ``ignore_regions`` for the loss to
            leave out, and a semantic mask's pixels become
            ``ignore_index``. ``"drop"`` removes it, which a loss reads as
            background. ``"blank"`` removes it and paints the object out
            with its surroundings, so that background is true. ``"keep"``
            leaves everything as it was.
        ignore_index: the value a semantic mask's suppressed pixels take.
        background_value: the value a semantic mask's background has.
        p: probability of applying, 1 by default.

    The call's result has one key more, ``ignore_regions``: an ``(n, 4)``
    float32 array of ``x_min, y_min, x_max, y_max`` in pixels, the regions
    of the labels suppressed (boxes, instance masks, keypoints), empty
    unless ``on_doubt`` is ``"ignore"``. Geometric transforms after this
    one do not move it, which is one reason this goes last.

    What each kind of label gets:

    ============ ======================== ==========================================
    Label        Object replaced          Object doubtful, by ``on_doubt``
    ============ ======================== ==========================================
    box          removed                  suppressed, removed, blanked, or kept
    mask         pixels made background   ``ignore_index``, background, or kept
    masks        emptied                  emptied and suppressed, emptied, or kept
    keypoints    removed                  suppressed, removed, or kept
    ============ ======================== ==========================================
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)
    #: After the link: judged on the finished picture.
    pipeline_stage: ClassVar[int] = 13

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``LabelLegibility`` as Albumentations validates them."""

        task: str | float
        keep_at: float = Field(ge=0.0, le=1.0)
        gone_at: float = Field(gt=0.0, le=1.0)
        on_doubt: Literal["ignore", "drop", "blank", "keep"]
        ignore_index: int
        background_value: int

    def __init__(
        self,
        task: str | float = "recognise",
        keep_at: float = 0.5,
        gone_at: float = 0.9,
        on_doubt: Literal["ignore", "drop", "blank", "keep"] = "ignore",
        ignore_index: int = 255,
        background_value: int = 0,
        p: float = 1.0,
    ) -> None:
        """Keep the policy; while this filter exists, every transform's call is noted."""
        super().__init__(p=p)
        self.task = task
        self.keep_at = keep_at
        self.gone_at = gone_at
        self.on_doubt = on_doubt
        self.ignore_index = ignore_index
        self.background_value = background_value
        _ = self.needed_cycles
        _FILTERS.add(self)

    @property
    def needed_cycles(self) -> float:
        """Cycles across the object the task needs, from ``task``."""
        if isinstance(self.task, str):
            try:
                return TASK_CYCLES[self.task]
            except KeyError as error:
                raise ValueError(
                    f"task must be one of {sorted(TASK_CYCLES)} or a number of cycles, "
                    f"got {self.task!r}"
                ) from error
        return float(self.task)

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        """The picture, and every kind of label on it."""
        return {
            **super().targets,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
        }

    def _action(self, hidden: float, probability: float) -> str:
        """``"background"``, ``"keep"``, or what ``on_doubt`` says for the rest."""
        if hidden >= self.gone_at:
            return "background"
        if probability >= self.keep_at or self.on_doubt == "keep":
            return "keep"
        return self.on_doubt

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Judge every labelled object in the frame and decide what each label gets.

        Returns, for the boxes, instance masks and keypoints, which to keep;
        for a semantic mask, the mask as it will be; the regions suppressed;
        and the regions to paint out of the picture.
        """
        image = data["image"]
        record = record_of(image)
        judge = _Judge(
            None if record is None else record.original,
            image,
            None if record is None else record.hidden,
            None if record is None else record.blur,
            self.needed_cycles,
        )
        height, width = judge.shape
        counts = {"keep": 0, "background": 0, "ignore": 0, "drop": 0, "blank": 0}
        blank: list[tuple[np.ndarray | None, tuple[int, int, int, int]]] = []
        ignored: list[tuple[int, int, int, int]] = []
        protected = np.zeros((height, width), dtype=bool)

        def decide(
            region: np.ndarray | None, bounds: tuple[int, int, int, int], listed: bool = True
        ) -> str:
            """The object's action; a suppressed one's region is listed unless ``listed`` is off."""
            action = self._action(*judge.score(region, bounds))
            counts[action] += 1
            x0, y0, x1, y1 = bounds
            if action == "keep":
                if region is None:
                    protected[y0:y1, x0:x1] = True
                else:
                    protected[region] = True
            elif action == "blank":
                blank.append((region, bounds))
            elif action == "ignore" and listed:
                ignored.append(bounds)
            return action

        drawn: dict[str, Any] = {}

        bboxes = data.get("bboxes")
        if bboxes is not None and len(bboxes):
            box_actions = [
                decide(
                    None,
                    judge.rect(box[0] * width, box[1] * height, box[2] * width, box[3] * height),
                )
                for box in np.asarray(bboxes, dtype=np.float64)
            ]
            drawn["box_keep"] = np.array([a == "keep" for a in box_actions], dtype=bool)

        mask = data.get("mask")
        if isinstance(mask, np.ndarray) and mask.ndim == 2:
            new_mask = mask.copy()
            for value in np.unique(mask):
                if value in (self.background_value, self.ignore_index):
                    continue
                count, parts, stats, _centres = cv2.connectedComponentsWithStats(
                    (mask == value).astype(np.uint8), connectivity=8
                )
                for part in range(1, count):
                    x, y, w, h, _area = (int(v) for v in stats[part])
                    region = parts == part
                    # Suppressed in the mask itself, with `ignore_index`.
                    action = decide(region, (x, y, x + w, y + h), listed=False)
                    if action == "ignore":
                        new_mask[region] = self.ignore_index
                    elif action != "keep":
                        new_mask[region] = self.background_value
            drawn["new_mask"] = new_mask

        masks = data.get("masks")
        if masks is not None and len(masks):
            keep = []
            for one in np.asarray(masks):
                region = one > 0
                if not region.any():
                    keep.append(True)
                    continue
                rows, cols = np.nonzero(region)
                bounds = (
                    int(cols.min()),
                    int(rows.min()),
                    int(cols.max()) + 1,
                    int(rows.max()) + 1,
                )
                keep.append(decide(region, bounds) == "keep")
            drawn["masks_keep"] = np.array(keep, dtype=bool)

        keypoints = data.get("keypoints")
        if keypoints is not None and len(keypoints):
            half = _KEYPOINT_SQUARE // 2
            keep = []
            for point in np.asarray(keypoints, dtype=np.float64):
                x, y = int(round(point[0])), int(round(point[1]))
                keep.append(
                    decide(None, judge.rect(x - half, y - half, x + half + 1, y + half + 1))
                    == "keep"
                )
            drawn["kp_keep"] = np.array(keep, dtype=bool)

        drawn["blank"] = blank
        drawn["protected"] = protected
        drawn["ignore_regions"] = np.array(ignored, dtype=np.float32).reshape(-1, 4)
        scored = judge.details
        record_draw(
            original_known=record is not None,
            objects=sum(counts.values()),
            kept=counts["keep"],
            background=counts["background"],
            suppressed=counts["ignore"],
            removed=counts["drop"],
            blanked=counts["blank"],
            cycles_before_p50=(
                float(np.median([d["cycles_before"] for d in scored])) if scored else 0.0
            ),
            cycles_after_p50=(
                float(np.median([d["cycles_after"] for d in scored])) if scored else 0.0
            ),
        )
        return drawn

    def apply_with_params(
        self, params: dict[str, Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Apply, and hand back the regions suppressed under ``ignore_regions``."""
        result = super().apply_with_params(params, *args, **kwargs)
        regions = params.get("ignore_regions")
        result["ignore_regions"] = (
            np.zeros((0, 4), dtype=np.float32) if regions is None else regions
        )
        return result

    def apply(
        self,
        img: np.ndarray,
        blank: list[Any] | None = None,
        protected: np.ndarray | None = None,
        **params: Any,
    ) -> np.ndarray:
        """The picture, with the objects to blank painted out by their surroundings."""
        if not blank:
            return img
        if protected is None:
            protected = np.zeros(img.shape[:2], dtype=bool)
        return _blank(img, blank, protected)

    def apply_to_bboxes(
        self, bboxes: np.ndarray, box_keep: np.ndarray | None = None, **params: Any
    ) -> np.ndarray:
        """The boxes kept."""
        if box_keep is None or len(bboxes) == 0:
            return bboxes
        kept: np.ndarray = bboxes[box_keep]
        return kept

    def apply_to_mask(
        self, mask: np.ndarray, new_mask: np.ndarray | None = None, **params: Any
    ) -> np.ndarray:
        """The semantic mask, the objects judged gone or doubtful relabelled."""
        return mask if new_mask is None else new_mask

    def apply_to_masks(
        self, masks: np.ndarray, masks_keep: np.ndarray | None = None, **params: Any
    ) -> np.ndarray:
        """The instance masks, those not kept emptied."""
        if masks_keep is None or len(masks) == 0:
            return masks
        out = np.array(masks, copy=True)
        out[~masks_keep] = 0
        return out

    def apply_to_keypoints(
        self, keypoints: np.ndarray, kp_keep: np.ndarray | None = None, **params: Any
    ) -> np.ndarray:
        """The keypoints kept."""
        if kp_keep is None or len(keypoints) == 0:
            return keypoints
        kept: np.ndarray = keypoints[kp_keep]
        return kept
