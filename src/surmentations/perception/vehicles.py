"""Which end of each vehicle a picture shows, and where its lamps are.

A road's spray lies behind its vehicles and their lamps are white at one
end and red at the other, so an effect that draws either has to know which
end of each vehicle faces the camera. Colour alone says it poorly: a rear in
shade shows no red, a red car is red all over, an amber indicator is on the
front too, and none of it says *where* a headlamp is.

So this is asked of a network, once per scene, as the scene's labels are.
The shipped estimator is an open-vocabulary detector, OWLv2 (Minderer,
Gritsenko and Houlsby, 2023), asked for two things by name, "a car
headlight" and "a car tail light". What it finds on a vehicle says both
which end is seen, by which of the two it is surer of, and where the lamps
are, by where it found them. On the 22 single vehicles of this
repository's sample scenes, labelled by eye, it told front from rear every
time; the lens test of :func:`surmentations.utils.road.vehicle_facing` was
right for 12 of them, wrong for 3 and silent on 7. That is four scenes and
no benchmark: it says the one is much the better, not how good it is.

**Why a detector and not a classifier.** A classifier of views (CLIP,
asked "the front of a car" against "the rear of a car") did nearly as well
at which end is seen, 21 of the 22, and says nothing of where the lamps
are. Networks
trained on car keypoints say both and better, and the ones published with
weights are under licences this library cannot take on (OpenPifPaf's, on
ApolloCar3D, is AGPL). OWLv2 is Apache-2.0.

**The cost** is the largest of anything a provider runs: on a laptop GPU,
90 ms for every sixteen vehicles in half precision and 300 ms in full. It
is paid once per scene, only by a provider given this estimator, and not
at all for a scene with no vehicle in it.

References:

* Minderer, M., Gritsenko, A., Houlsby, N. (2023). Scaling Open-Vocabulary
  Object Detection. NeurIPS. https://arxiv.org/abs/2306.09683
* Radford, A., et al. (2021). Learning Transferable Visual Models From
  Natural Language Supervision. ICML. https://arxiv.org/abs/2103.00020
* Song, X., et al. (2019). ApolloCar3D: A Large 3D Car Instance
  Understanding Benchmark for Autonomous Driving. CVPR.
  https://arxiv.org/abs/1811.12222
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import cv2
import numpy as np

from surmentations import mode

# What the detector is asked for, in this order: the first is a front's and
# the second a rear's.
LAMP_PROMPTS = ("a car headlight", "a car tail light")
# A lamp is a small part of a vehicle: a box wider or taller than this share
# of its outline is the bumper, the windscreen or the whole car.
_LAMP_SHARE = 0.35
# A lamp found counts when it scores this share of the best of its kind on
# its vehicle.
_NEXT_TO_BEST = 0.3
# The two of a pair are at least this share of the outline's width apart,
# and within this share of its height of one level.
_PAIR_APART = 0.3
_PAIR_LEVEL = 0.25
# How far outside its outline, as a share of the outline's size, a lamp's
# middle may lie: a network's outline is a little inside the vehicle.
_OUTLINE_SLACK = 0.08

_NO_CUDA_MESSAGE = (
    "{cls} requires a CUDA GPU: it runs a detection network over the vehicles of every "
    "scene, which is too slow on a CPU to fit in a training-time DataLoader budget. No CUDA "
    "device was found (torch.cuda.is_available() is False). Vehicle views are optional: "
    "leave `vehicles` out of the SharedDepthProvider and the spray and the lamps fall back "
    "to what they read from colour."
)


class VehicleView(NamedTuple):
    """What a picture shows of one vehicle: which end, and where its lamps are.

    The fields:

    * ``box``: the vehicle's outline in the frame, as left, top, width and
      height in pixels. It is what a vehicle is known by: the outline is a
      connected piece of the scene's vehicle label.
    * ``front``: how sure the reading is that a headlamp is seen on it, 0
      to 1.
    * ``rear``: the same of a tail lamp.
    * ``lamps``: the lamps found of the kind it is surer of, at most two,
      one to a row: column, row, half width and half height in pixels of the
      frame, and the score. The pair, where a pair was found; one lamp
      where the other is hidden; none where none was found.
    """

    box: tuple[int, int, int, int]
    front: float
    rear: float
    lamps: np.ndarray

    @property
    def facing(self) -> float:
        """Which end is seen and how surely: above 0 the rear, below 0 the front."""
        return self.rear - self.front


def vehicle_outlines(
    vehicles: np.ndarray, min_pixels: int, most: int
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """The outlines in a vehicle mask, largest first: each one's box and its own mask in it.

    An outline is a connected piece of the mask, 8-connected, of at least
    ``min_pixels``; at most ``most`` are returned. Vehicles that touch in
    the picture are one outline: a semantic mask does not tell them apart.
    """
    mask = (np.asarray(vehicles) > 0.5).astype(np.uint8)
    count, parts, stats, _centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    order = sorted(range(1, count), key=lambda index: -int(stats[index, cv2.CC_STAT_AREA]))
    found = []
    for index in order[:most]:
        left, top, wide, tall, area = (int(value) for value in stats[index])
        if area < min_pixels:
            break
        found.append(
            ((left, top, wide, tall), parts[top : top + tall, left : left + wide] == index)
        )
    return found


def lamps_of(
    box: tuple[int, int, int, int],
    outline: np.ndarray,
    found: np.ndarray,
    least_score: float = 0.05,
) -> VehicleView:
    """One vehicle's view from the boxes a detector found on it.

    ``found`` has a row for each: left, top, right, bottom in pixels of the
    frame, the kind (0 a headlamp, 1 a tail lamp) and the score. ``outline``
    is the vehicle's own mask within ``box``.

    A lamp is a small thing on the vehicle: boxes that are a large part of
    the outline, or whose middle is off it, are dropped, and of boxes one
    inside another the surer is kept. The vehicle shows the end whose lamps
    score higher. Its lamps are the two of that kind that make the best
    pair, apart across the vehicle and near one level, which is what keeps
    out a number plate and a badge; or the single best, where there is no
    such pair.
    """
    left, top, wide, tall = box
    none = np.zeros((0, 5), dtype=np.float32)
    lamps: list[list[tuple[float, float, float, float, float]]] = [[], []]
    slack_x, slack_y = _OUTLINE_SLACK * wide + 1.0, _OUTLINE_SLACK * tall + 1.0
    grown = cv2.dilate(
        outline.astype(np.uint8),
        np.ones((2 * int(slack_y) + 1, 2 * int(slack_x) + 1), np.uint8),
    )
    for x0, y0, x1, y1, kind, score in sorted(found.tolist(), key=lambda row: -row[5]):
        if score < least_score:
            continue
        half_wide, half_tall = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
        if 2.0 * half_wide > _LAMP_SHARE * wide or 2.0 * half_tall > _LAMP_SHARE * tall:
            continue
        x, y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        column, row = int(x) - left, int(y) - top
        if not (0 <= column < wide and 0 <= row < tall) or not grown[row, column]:
            continue
        kept = lamps[int(kind)]
        if any(abs(x - at[0]) <= at[2] and abs(y - at[1]) <= at[3] for at in kept):
            continue
        kept.append((x, y, half_wide, half_tall, score))
    scores = [max((lamp[4] for lamp in kind), default=0.0) for kind in lamps]
    if max(scores) <= 0.0:
        return VehicleView(box, 0.0, 0.0, none)
    seen = [lamp for lamp in lamps[int(scores[1] > scores[0])]]
    seen = [lamp for lamp in seen if lamp[4] >= _NEXT_TO_BEST * seen[0][4]]
    best, pair = 0.0, [seen[0]]
    for first in range(len(seen)):
        for second in range(first + 1, len(seen)):
            one, other = seen[first], seen[second]
            if abs(one[0] - other[0]) < _PAIR_APART * wide:
                continue
            if abs(one[1] - other[1]) > _PAIR_LEVEL * tall:
                continue
            if one[4] + other[4] > best:
                best, pair = one[4] + other[4], sorted([one, other])
    return VehicleView(box, float(scores[0]), float(scores[1]), np.array(pair, dtype=np.float32))


def check_views(views: Any, *, source: str) -> list[VehicleView]:
    """``views`` as a list of :class:`VehicleView`, or a clear error.

    Raises ``TypeError`` when an entry is not one.
    """
    views = list(views)
    for view in views:
        if not isinstance(view, VehicleView):
            raise TypeError(
                f"{source} gave a {type(view).__name__} where a VehicleView was expected; "
                "an estimator of vehicle views returns one VehicleView per outline."
            )
    return views


class VehicleViewEstimator(ABC):
    """Interface for whatever says which end of each vehicle a frame shows.

    Subclass it to plug in another network, or a dataset's own annotation
    of headings.
    """

    @abstractmethod
    def estimate(self, image: np.ndarray, vehicles: np.ndarray) -> list[VehicleView]:
        """The view of each vehicle of one ``(H, W, 3)`` frame.

        ``vehicles`` is the ``(H, W)`` mask of the pixels labelled vehicle.
        A view is returned for each outline the estimator read, known by
        its box (:func:`vehicle_outlines`); an outline it did not read is
        simply absent, and whoever asks falls back to what colour says.
        """


class OpenVocabularyVehicleEstimator(VehicleViewEstimator):
    """Vehicle views from an open-vocabulary detector asked for lamps, the shipped default.

    Each outline of the vehicle mask is cut out of the frame with a little
    of what surrounds it, and the cuts are laid out as a square mosaic,
    ``grid`` to a side, which the detector sees as one picture: sixteen
    vehicles cost one pass and not sixteen. It is asked for a headlight and
    for a tail light (:data:`LAMP_PROMPTS`), and what it finds in each
    cell is that vehicle's (:func:`lamps_of`).

    Args:
        model_name: the checkpoint: anything ``transformers`` loads as a
            zero-shot object detector. The default is OWLv2 on a ViT-B/16,
            Apache-2.0, about 620 MB.
        device: where to run it. CUDA only, for the reason
            :class:`~surmentations.depth.estimator.DepthAnythingV2Estimator`
            gives.
        grid: vehicles to a side of the mosaic. Four, the default, shows
            the network each vehicle at 240 pixels, which is enough for it
            to find a lamp; five is a third faster on a busy road and finds
            more that are not lamps.
        max_vehicles: the most outlines read in one frame, largest first.
        min_pixels: the smallest outline read. Below a few hundred pixels a
            lamp is under a pixel, and there is nothing to find.
        min_score: the least the detector is believed at.
        half_precision: run the network in float16. ``None`` follows the
            mode (:mod:`surmentations.mode`): half in fast mode, full in
            exact mode. Half is four times as fast and moves a lamp by a
            tenth of a pixel.
    """

    def __init__(
        self,
        model_name: str = "google/owlv2-base-patch16-ensemble",
        device: str = "cuda",
        grid: int = 4,
        max_vehicles: int = 32,
        min_pixels: int = 150,
        min_score: float = 0.05,
        half_precision: bool | None = None,
    ) -> None:
        """Load the checkpoint onto the GPU; see the class docstring for the arguments.

        Raises ``RuntimeError`` where torch has no CUDA device and
        ``ValueError`` for a grid or a count below 1. torch and transformers
        are imported here and not at module level.
        """
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(_NO_CUDA_MESSAGE.format(cls=type(self).__name__))
        if grid < 1 or max_vehicles < 1:
            raise ValueError(
                f"grid and max_vehicles must be at least 1, got {grid} and {max_vehicles}"
            )
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.model_name = model_name
        self.grid = int(grid)
        self.max_vehicles = int(max_vehicles)
        self.min_pixels = int(min_pixels)
        self.min_score = float(min_score)
        self.half_precision = half_precision
        self.device = torch.device(device)
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device).eval()
        )
        size = getattr(self._processor.image_processor, "size", None)
        side = size.get("height") if isinstance(size, dict) else getattr(size, "height", None)
        self._side = int(side or 960)

    def estimate(self, image: np.ndarray, vehicles: np.ndarray) -> list[VehicleView]:
        """The view of each vehicle outline of one ``(H, W, 3)`` frame, largest first.

        A float frame is taken as ``[0, 1]`` and quantised to 8 bits first.
        """
        import torch

        height, width = image.shape[:2]
        if image.dtype != np.uint8:
            image = np.clip(np.rint(image.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
        outlines = vehicle_outlines(vehicles, self.min_pixels, self.max_vehicles)
        views: list[VehicleView] = []
        cell = self._side // self.grid
        for start in range(0, len(outlines), self.grid * self.grid):
            batch = outlines[start : start + self.grid * self.grid]
            canvas = np.full((self._side, self._side, 3), 114, dtype=np.uint8)
            placed = []
            for slot, ((left, top, wide, tall), _outline) in enumerate(batch):
                # With a little of what is round it, which is how the
                # network knows a car when it sees one.
                pad = int(0.15 * max(wide, tall)) + 2
                x0, y0 = max(0, left - pad), max(0, top - pad)
                x1, y1 = min(width, left + wide + pad), min(height, top + tall + pad)
                scale = (cell - 8) / max(x1 - x0, y1 - y0)
                cut = cv2.resize(
                    image[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )
                at_y, at_x = (slot // self.grid) * cell + 4, (slot % self.grid) * cell + 4
                canvas[at_y : at_y + cut.shape[0], at_x : at_x + cut.shape[1]] = cut
                placed.append((at_x, at_y, cut.shape[1], cut.shape[0], scale, x0, y0))
            inputs = self._processor(
                text=[list(LAMP_PROMPTS)], images=canvas, return_tensors="pt"
            ).to(self.device)
            half = mode.fast() if self.half_precision is None else self.half_precision
            with (
                torch.inference_mode(),
                torch.autocast(self.device.type, dtype=torch.float16, enabled=half),
            ):
                output = self._model(**inputs)
            found = self._processor.post_process_grounded_object_detection(
                output, threshold=self.min_score, target_sizes=[(self._side, self._side)]
            )[0]
            boxes = found["boxes"].float().cpu().numpy().reshape(-1, 4)
            kinds = found["labels"].cpu().numpy().reshape(-1, 1).astype(np.float32)
            scores = found["scores"].float().cpu().numpy().reshape(-1, 1)
            middle_x, middle_y = (
                0.5 * (boxes[:, 0] + boxes[:, 2]),
                0.5 * (boxes[:, 1] + boxes[:, 3]),
            )
            for (box, outline), (at_x, at_y, wide, tall, scale, x0, y0) in zip(
                batch, placed, strict=True
            ):
                inside = (
                    (middle_x >= at_x)
                    & (middle_x < at_x + wide)
                    & (middle_y >= at_y)
                    & (middle_y < at_y + tall)
                )
                # From the mosaic back to the frame.
                in_frame = (boxes[inside] - [at_x, at_y, at_x, at_y]) / scale + [x0, y0, x0, y0]
                rows = np.hstack([in_frame, kinds[inside], scores[inside]])
                views.append(lamps_of(box, outline, rows, self.min_score))
        return views
