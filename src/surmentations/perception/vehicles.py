"""Which end of each vehicle a picture shows, and where its lamps are.

A road's spray lies behind its vehicles and their lamps are white at one
end and red at the other, so an effect that draws either has to know which
end of each vehicle faces the camera. Colour alone says it poorly: a rear in
shade shows no red, a red car is red all over, an amber indicator is on the
front too, and none of it says *where* a headlamp is.

So this is asked of a network, once per scene, as the scene's labels are.
The shipped estimator is an open-vocabulary detector, OWLv2 (Minderer,
Gritsenko and Houlsby, 2023). It is asked first for the vehicles
themselves, over the whole frame, because a semantic mask runs touching
vehicles together: three cars nose to tail are one long outline to it, and
the boxes found on that outline share it out among them. Then each vehicle
is looked at for two things by name, "a car headlight" and "a car tail
light". What is found on a vehicle says both which end is seen, by which
of the two the network is surer of, and where the lamps are, by where it
found them. On the 22 single vehicles of this
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

**The cost** is the largest of anything a provider runs: one pass of the
network over the frame and one for every sixteen vehicles, each about 65 ms
on a laptop GPU in half precision and 280 ms in full. It is paid once per
scene, only by a provider given this estimator, and not at all for a scene
with no vehicle in it.

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
# What it is asked for first, over the whole frame: the vehicles themselves,
# so that those the scene's labels run together can be told apart. And what
# each is called, the heavy ones among them being slower and wider.
VEHICLE_PROMPTS = ("a car", "a van", "a truck", "a bus", "a motorcycle")
VEHICLE_KINDS = ("car", "van", "truck", "bus", "motorcycle")
HEAVY_KINDS = frozenset({"truck", "bus"})
# And, in the same look at the whole frame, the street lamps: a photograph
# taken by day has them off, and a frame made into a night one is lit by
# them. A post is a box this many times taller than wide, at least, and
# this share of the frame's height; a lamp's head is a box no taller than
# this many times its width, found in the top of a post's.
STREET_LAMP_PROMPTS = ("a street light",)
# And the road signs, and on each vehicle its number plate: the things on a
# road made to send light back where it came from, which under a camera's
# own lamp are the brightest things in the frame.
SIGN_PROMPTS = ("a traffic sign",)
PLATE_PROMPTS = ("a license plate",)
# A sign is believed at this score, and is no larger than this share of the
# frame; a plate at this score, no wider than this share of its vehicle and
# wider than it is tall.
_SIGN_SCORE = 0.2
_SIGN_LARGEST = 0.06
_PLATE_SCORE = 0.12
_PLATE_WIDEST = 0.6
_POST_SCORE = 0.2
_POST_TALL = 1.6
_POST_LEAST = 0.05
_HEAD_FLAT = 1.25
_HEAD_IN_TOP = 0.1
# A vehicle found is believed at this score; at least this share of its box
# is labelled vehicle; and at least this share of it is the outline it is
# taken to be part of.
_VEHICLE_SCORE = 0.2
_VEHICLE_FILL = 0.3
_OUTLINE_FILL = 0.25
# Two boxes are one vehicle when they overlap by this much of their union,
# or when the smaller lies this far inside the larger and is at least this
# share of its size. A box that holds the middles of two smaller ones is a
# group of vehicles and not one.
_SAME_VEHICLE = 0.5
_INSIDE = 0.8
_ALIKE = 0.35
_GROUP_OF = 0.6
# A pixel of an outline outside every box found on it goes to the nearest,
# up to this share of that box's size away; further off it is another
# vehicle, which the detector did not find.
_BESIDE = 0.25
# The smallest piece of an outline that is kept as a vehicle of its own.
_FEWEST_PIXELS = 20
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
      height in pixels.
    * ``front``: how sure the reading is that a headlamp is seen on it, 0
      to 1.
    * ``rear``: the same of a tail lamp.
    * ``lamps``: the lamps found of the kind it is surer of, at most two,
      one to a row: column, row, half width and half height in pixels of the
      frame, and the score. The pair, where a pair was found; one lamp
      where the other is hidden; none where none was found.
    * ``mask``: which pixels of the box are the vehicle, ``(height, width)``
      bool. ``None`` says the vehicle is the whole connected piece of the
      scene's vehicle label that has this box. With a mask, a piece of the
      label can be more than one vehicle: touching vehicles are one piece
      to a semantic mask, and an estimator that can tell them apart says
      so here.
    * ``kind``: what the vehicle is, where that is known: one of
      :data:`VEHICLE_KINDS`, or empty.
    * ``plate``: its number plate where one was found, as left, top, right
      and bottom in pixels of the frame; ``None`` otherwise.
    """

    box: tuple[int, int, int, int]
    front: float
    rear: float
    lamps: np.ndarray
    mask: np.ndarray | None = None
    kind: str = ""
    plate: tuple[float, float, float, float] | None = None

    @property
    def read(self) -> bool:
        """Whether anything was found on it: an unread vehicle says nothing of its ends."""
        return self.front > 0.0 or self.rear > 0.0

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


def _one_each(rows: list[tuple[float, ...]], groups: bool = True) -> list[tuple[float, ...]]:
    """Of boxes a detector found, those that are each one thing, surest first.

    A row is left, top, right, bottom, kind, score and area. Dropped are
    the same thing found twice, a box that holds the middles of two smaller
    ones, which is a group, and most of a thing found again as a thing.
    ``groups`` false keeps the second kind: a row of lamp posts down a
    road stands, in the picture, inside the box of the nearest.
    """
    rows = sorted(rows, key=lambda row: -row[5])

    def shared(one: tuple[float, ...], other: tuple[float, ...]) -> float:
        """The area two boxes have in common."""
        across = min(one[2], other[2]) - max(one[0], other[0])
        down = min(one[3], other[3]) - max(one[1], other[1])
        return max(across, 0.0) * max(down, 0.0)

    # The same thing twice, first: or a thing found twice would count as
    # two when the next step asks what a box holds.
    once: list[tuple[float, ...]] = []
    for row in rows:
        if all(
            shared(row, other) <= _SAME_VEHICLE * (row[6] + other[6] - shared(row, other))
            for other in once
        ):
            once.append(row)
    single = []
    for row in once:
        held = sum(
            1
            for other in once
            if other is not row
            and other[6] < _GROUP_OF * row[6]
            and row[0] <= 0.5 * (other[0] + other[2]) <= row[2]
            and row[1] <= 0.5 * (other[1] + other[3]) <= row[3]
        )
        if held < 2 or not groups:
            single.append(row)
    kept: list[tuple[float, ...]] = []
    for row in single:
        if not any(
            shared(row, other) > _INSIDE * min(row[6], other[6])
            and min(row[6], other[6]) > _ALIKE * max(row[6], other[6])
            for other in kept
        ):
            kept.append(row)
    return kept


def lamp_posts(found: np.ndarray, frame_height: int) -> np.ndarray:
    """The street lamps among the boxes a detector found, one row to a lamp.

    ``found`` has a row for each box: left, top, right, bottom, anything,
    and the score. A detector asked for a street light answers with the
    whole lamp, post and arm and head, and with the head alone. A post is
    a tall box; its heads are the flat boxes at the very top of it, two for
    a column with two arms, or the middle of its top where there is none.

    A row of the result is one head: its post's left, top, right and
    bottom, the column and row of the head, and the score. The foot of the post, where
    it stands on the ground, is the middle of its bottom.
    """
    posts: list[tuple[float, ...]] = []
    heads: list[tuple[float, float, float]] = []
    for x0, y0, x1, y1, _kind, score in found.reshape(-1, 6).tolist():
        wide, tall = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
        if score < _POST_SCORE:
            continue
        if tall >= _POST_TALL * wide and tall >= _POST_LEAST * frame_height:
            posts.append((x0, y0, x1, y1, 0.0, score, wide * tall))
        elif tall <= _HEAD_FLAT * wide:
            heads.append((0.5 * (x0 + x1), 0.5 * (y0 + y1), score))
    lamps = []
    for x0, y0, x1, y1, _kind, score, _area in _one_each(posts, groups=False):
        # A head is at the top of its post and no lower: a flat box further
        # down the post's is the head of the next lamp along the road.
        slack = 0.15 * (x1 - x0)
        mine = [
            head
            for head in heads
            if x0 - slack <= head[0] <= x1 + slack
            and y0 - slack <= head[1] <= y0 + _HEAD_IN_TOP * (y1 - y0)
        ]
        # Both heads of a column with two arms; the middle of the top
        # where the detector boxed none.
        mine.sort(key=lambda head: -head[2])
        for head_x, head_y, _head_score in mine[:2] or [(0.5 * (x0 + x1), y0, 0.0)]:
            lamps.append((x0, y0, x1, y1, head_x, head_y, score))
    return np.array(lamps, dtype=np.float32).reshape(-1, 7)


def separate_vehicles(found: np.ndarray, vehicles: np.ndarray) -> np.ndarray:
    """The boxes a detector found that are each one vehicle, surest first.

    ``found`` has a row for each box: left, top, right, bottom in pixels of
    the frame, the kind (an index into :data:`VEHICLE_KINDS`) and the
    score. A detector asked for cars answers with more than cars: the same
    car twice, a row of parked cars as one, a hoarding. Kept are the boxes
    it is sure enough of that lie mostly on the scene's vehicle label, less
    those that hold two smaller ones, and of boxes on one vehicle the
    surest.
    """
    mask = (np.asarray(vehicles) > 0.5).astype(np.uint8)
    height, width = mask.shape
    total = cv2.integral(mask)
    rows: list[tuple[float, ...]] = []
    for x0, y0, x1, y1, kind, score in found.tolist():
        left, top = int(np.clip(x0, 0, width - 1)), int(np.clip(y0, 0, height - 1))
        right, bottom = int(np.clip(x1, left + 1, width)), int(np.clip(y1, top + 1, height))
        inside = total[bottom, right] - total[top, right] - total[bottom, left] + total[top, left]
        area = float((right - left) * (bottom - top))
        if score >= _VEHICLE_SCORE and inside >= _VEHICLE_FILL * area:
            rows.append((float(left), float(top), float(right), float(bottom), kind, score, area))
    kept = _one_each(rows)
    return np.array([row[:6] for row in kept], dtype=np.float32).reshape(-1, 6)


def split_outlines(
    vehicles: np.ndarray, found: np.ndarray
) -> list[tuple[tuple[int, int, int, int], np.ndarray, str]]:
    """The vehicles of a vehicle mask, largest first: each one's box, its mask in it, and its kind.

    A connected piece of the mask with one vehicle found on it, or none, is
    one vehicle. One with several found on it (``found``, as
    :func:`separate_vehicles` returns it) is shared out among them. A pixel
    inside one box is that vehicle's; inside several, it is the vehicle's
    whose box reaches lowest in the picture, which is the nearest and
    stands in front of the others; just outside them all, the nearest's.
    What lies well away from every box is a vehicle the detector did not
    find, and is kept as one of its own, of no known kind.
    """
    mask = (np.asarray(vehicles) > 0.5).astype(np.uint8)
    count, parts, stats, _centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = found.reshape(-1, 6)
    out: list[tuple[tuple[int, int, int, int], np.ndarray, str]] = []
    for index in range(1, count):
        left, top, wide, tall, area = (int(value) for value in stats[index])
        if area < _FEWEST_PIXELS:
            continue
        own = parts[top : top + tall, left : left + wide] == index
        rows, columns = np.nonzero(own)
        on_it = []
        for x0, y0, x1, y1, kind, _score in boxes.tolist():
            if not (left <= 0.5 * (x0 + x1) < left + wide and top <= 0.5 * (y0 + y1) < top + tall):
                continue
            inside = (
                (columns + left >= x0)
                & (columns + left < x1)
                & (rows + top >= y0)
                & (rows + top < y1)
            )
            if int(inside.sum()) >= _OUTLINE_FILL * (x1 - x0) * (y1 - y0):
                on_it.append((x0, y0, x1, y1, int(kind), inside))
        if len(on_it) < 2:
            kind = VEHICLE_KINDS[on_it[0][4]] if on_it else ""
            out.append(((left, top, wide, tall), own, kind))
            continue
        # Whose each pixel is: the lowest-reaching box it is inside, or the
        # nearest box it is just outside of.
        whose = np.full(len(rows), -1)
        lowest = np.full(len(rows), -np.inf)
        nearest = np.full(len(rows), np.inf)
        beside = np.full(len(rows), -1)
        for number, (x0, y0, x1, y1, _kind, inside) in enumerate(on_it):
            take = inside & (y1 > lowest)
            whose[take], lowest[take] = number, y1
            off_x = np.maximum(np.maximum(x0 - (columns + left), columns + left - x1), 0.0)
            off_y = np.maximum(np.maximum(y0 - (rows + top), rows + top - y1), 0.0)
            off = np.maximum(off_x / max(x1 - x0, 1.0), off_y / max(y1 - y0, 1.0))
            closer = off < nearest
            nearest[closer], beside[closer] = off[closer], number
        outside = whose < 0
        whose[outside & (nearest <= _BESIDE)] = beside[outside & (nearest <= _BESIDE)]
        for number, (_x0, _y0, _x1, _y1, kind, _inside) in enumerate(on_it):
            mine = whose == number
            if int(mine.sum()) >= _FEWEST_PIXELS:
                out.append((*_boxed(rows[mine], columns[mine], left, top), VEHICLE_KINDS[kind]))
        rest = np.zeros((tall, wide), dtype=np.uint8)
        rest[rows[whose < 0], columns[whose < 0]] = 1
        pieces, piece_of, piece_stats, _middles = cv2.connectedComponentsWithStats(
            rest, connectivity=8
        )
        for piece in range(1, pieces):
            if int(piece_stats[piece, cv2.CC_STAT_AREA]) >= _FEWEST_PIXELS:
                piece_rows, piece_columns = np.nonzero(piece_of == piece)
                out.append((*_boxed(piece_rows, piece_columns, left, top), ""))
    out.sort(key=lambda one: -int(one[1].sum()))
    return out


def _boxed(
    rows: np.ndarray, columns: np.ndarray, left: int, top: int
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """Pixels given by row and column within an outline's box, as a box in the frame and a mask."""
    first_row, first_column = int(rows.min()), int(columns.min())
    tall, wide = int(rows.max()) - first_row + 1, int(columns.max()) - first_column + 1
    mask = np.zeros((tall, wide), dtype=bool)
    mask[rows - first_row, columns - first_column] = True
    return (left + first_column, top + first_row, wide, tall), mask


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


def plate_of(
    box: tuple[int, int, int, int], found: np.ndarray
) -> tuple[float, float, float, float] | None:
    """A vehicle's number plate among the boxes a detector found on it, or ``None``.

    ``found`` has a row for each: left, top, right, bottom in pixels of the
    frame, anything, and the score. A plate is a small thing wider than it
    is tall, within the vehicle's box: the surest such.
    """
    left, top, wide, tall = box
    best: tuple[float, tuple[float, float, float, float]] | None = None
    for x0, y0, x1, y1, _kind, score in found.reshape(-1, 6).tolist():
        across, down = x1 - x0, y1 - y0
        inside = left <= 0.5 * (x0 + x1) <= left + wide and top <= 0.5 * (y0 + y1) <= top + tall
        if score < _PLATE_SCORE or not inside or across <= down or across > _PLATE_WIDEST * wide:
            continue
        if best is None or score > best[0]:
            best = (score, (x0, y0, x1, y1))
    return None if best is None else best[1]


def street_signs(found: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The road signs among the boxes a detector found, a row to each.

    ``found`` has a row for each box: left, top, right, bottom, anything,
    and the score. Kept are those it is sure of that are no larger than a
    sign is in a frame, each once. A row of the result is left, top, right,
    bottom and score.
    """
    height, width = shape
    rows: list[tuple[float, ...]] = []
    for x0, y0, x1, y1, _kind, score in found.reshape(-1, 6).tolist():
        area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
        if score >= _SIGN_SCORE and 0.0 < area <= _SIGN_LARGEST * height * width:
            rows.append((x0, y0, x1, y1, 0.0, score, area))
    kept = _one_each(rows, groups=False)
    return np.array([(*row[:4], row[5]) for row in kept], dtype=np.float32).reshape(-1, 5)


def scaled_view(
    view: VehicleView, across: float, down: float, shape: tuple[int, int]
) -> VehicleView:
    """A view read in one size of a frame, in the pixels of another.

    ``across`` and ``down`` are the new frame's width and height over the
    old one's, and ``shape`` the new frame's. For a provider whose reader
    looks at a photograph at the size it was taken at and whose pipeline
    works on a reduced copy.
    """
    left, top, wide, tall = view.box
    new_left = int(np.clip(round(left * across), 0, shape[1] - 1))
    new_top = int(np.clip(round(top * down), 0, shape[0] - 1))
    new_wide = int(np.clip(round((left + wide) * across) - new_left, 1, shape[1] - new_left))
    new_tall = int(np.clip(round((top + tall) * down) - new_top, 1, shape[0] - new_top))
    lamps = view.lamps * np.array([across, down, across, down, 1.0], dtype=np.float32)
    mask = view.mask
    if mask is not None:
        mask = (
            cv2.resize(mask.astype(np.uint8), (new_wide, new_tall), interpolation=cv2.INTER_AREA)
            > 0
        )
        if not mask.any():
            mask[new_tall // 2, new_wide // 2] = True
    plate = view.plate
    if plate is not None:
        plate = (plate[0] * across, plate[1] * down, plate[2] * across, plate[3] * down)
    return view._replace(
        box=(new_left, new_top, new_wide, new_tall), lamps=lamps, mask=mask, plate=plate
    )


def check_views(views: Any, *, source: str) -> list[VehicleView]:
    """``views`` as a list of :class:`VehicleView`, or a clear error.

    Raises ``TypeError`` when an entry is not one.
    """
    views = list(views)
    for view in views:
        if not isinstance(view, VehicleView):
            raise TypeError(
                f"{source} gave a {type(view).__name__} where a VehicleView was expected; "
                "an estimator of vehicle views returns one VehicleView per vehicle."
            )
        if view.mask is not None and view.mask.shape != (view.box[3], view.box[2]):
            raise ValueError(
                f"{source} gave a vehicle a mask of shape {view.mask.shape} in a box "
                f"{view.box[2]} wide and {view.box[3]} tall; a view's mask is its box's size."
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
        A view is returned for each vehicle the estimator has something to
        say of: which pixels are it, where it can tell touching vehicles
        apart, and which end it shows, where it read that. A vehicle left
        out, or returned unread, is read by colour by whoever asks.
        """

    def street_lamps(self, image: np.ndarray) -> np.ndarray:
        """The street lamps of one ``(H, W, 3)`` frame, a row to each, where the reader finds them.

        As :func:`lamp_posts` returns them: the post's box, its head and
        the score. None here; an estimator that can find them says so.
        """
        return np.zeros((0, 7), dtype=np.float32)

    def street_signs(self, image: np.ndarray) -> np.ndarray:
        """The road signs of one ``(H, W, 3)`` frame, a row to each, where the reader finds them.

        As :func:`street_signs` returns them: left, top, right, bottom and
        score. None here; an estimator that can find them says so.
        """
        return np.zeros((0, 5), dtype=np.float32)


class OpenVocabularyVehicleEstimator(VehicleViewEstimator):
    """Vehicle views from an open-vocabulary detector asked for lamps, the shipped default.

    It looks twice. First at the whole frame, for the vehicles themselves
    (:data:`VEHICLE_PROMPTS`): a semantic mask runs touching vehicles
    together, and the boxes found on each connected piece of it share the
    piece out among them (:func:`separate_vehicles`,
    :func:`split_outlines`). Then at the vehicles one by one. Each is cut
    out of the frame with a little of what surrounds it, and the cuts are
    laid out as a square mosaic, ``grid`` to a side, which the detector
    sees as one picture: sixteen vehicles cost one pass and not sixteen.
    There it is asked for a headlight and for a tail light
    (:data:`LAMP_PROMPTS`), and what it finds in each cell is that
    vehicle's (:func:`lamps_of`).

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
        max_vehicles: the most vehicles read for their lamps in one frame,
            largest first.
        min_pixels: the smallest vehicle read. Below a few hundred pixels a
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
        pictures = self._processor.image_processor
        self._mean = torch.tensor(pictures.image_mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(pictures.image_std, device=self.device).view(1, 3, 1, 1)
        self._words: dict[tuple[str, ...], Any] = {}
        self._seen: tuple[Any, np.ndarray] | None = None

    def _find(self, picture: np.ndarray, prompts: tuple[str, ...], least: float) -> np.ndarray:
        """What the detector finds in a square picture, one row to a box.

        Left, top, right and bottom in the picture's own pixels, the kind,
        which is the index of the prompt, and the score.

        The picture is brought to the network's size and normalised here,
        on the GPU, and the prompts are put into tokens once: the
        checkpoint's own preprocessing does both on the CPU for every pass.
        """
        import torch

        side = picture.shape[0]
        if side != self._side:
            shrink = cv2.INTER_AREA if side > self._side else cv2.INTER_LINEAR
            picture = cv2.resize(picture, (self._side, self._side), interpolation=shrink)
        if prompts not in self._words:
            self._words[prompts] = self._processor(
                text=[list(prompts)], return_tensors="pt", padding="max_length"
            ).to(self.device)
        pixels = torch.from_numpy(picture).to(self.device).permute(2, 0, 1).float().unsqueeze(0)
        pixels = (pixels / 255.0 - self._mean) / self._std
        half = mode.fast() if self.half_precision is None else self.half_precision
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=torch.float16, enabled=half),
        ):
            output = self._model(pixel_values=pixels, **self._words[prompts])
        found = self._processor.post_process_grounded_object_detection(
            output, threshold=least, target_sizes=[(side, side)]
        )[0]
        boxes = found["boxes"].float().cpu().numpy().reshape(-1, 4)
        kinds = found["labels"].cpu().numpy().reshape(-1, 1).astype(np.float32)
        scores = found["scores"].float().cpu().numpy().reshape(-1, 1)
        return np.hstack([boxes, kinds, scores]).astype(np.float32)

    def _whole_frame(self, image: np.ndarray) -> np.ndarray:
        """What stands in a whole 8-bit frame: its vehicles and its street lamps, as found.

        One pass, asked for both, and kept for the frame it was made for:
        the vehicles and the lamps of one frame are asked for one after
        the other.
        """
        height, width = image.shape[:2]
        key = (image.shape, hash(image[:: max(1, height // 32), :: max(1, width // 32)].tobytes()))
        if self._seen is None or self._seen[0] != key:
            whole = np.full((max(height, width), max(height, width), 3), 114, dtype=np.uint8)
            whole[:height, :width] = image
            least = min(_VEHICLE_SCORE, _POST_SCORE, _SIGN_SCORE)
            asked = VEHICLE_PROMPTS + STREET_LAMP_PROMPTS + SIGN_PROMPTS
            self._seen = (key, self._find(whole, asked, least))
        return self._seen[1]

    def street_signs(self, image: np.ndarray) -> np.ndarray:
        """The road signs of one ``(H, W, 3)`` frame (:func:`street_signs`).

        From the same look at the whole frame that finds its vehicles.
        """
        if image.dtype != np.uint8:
            image = np.clip(np.rint(image.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
        found = self._whole_frame(image)
        signs = found[found[:, 4] == len(VEHICLE_PROMPTS) + len(STREET_LAMP_PROMPTS)]
        return street_signs(signs, image.shape[:2])

    def street_lamps(self, image: np.ndarray) -> np.ndarray:
        """The street lamps of one ``(H, W, 3)`` frame (:func:`lamp_posts`).

        From the same look at the whole frame that finds its vehicles.
        """
        if image.dtype != np.uint8:
            image = np.clip(np.rint(image.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
        found = self._whole_frame(image)
        return lamp_posts(found[found[:, 4] == len(VEHICLE_PROMPTS)], image.shape[0])

    def estimate(self, image: np.ndarray, vehicles: np.ndarray) -> list[VehicleView]:
        """The view of each vehicle of one ``(H, W, 3)`` frame, largest first.

        A float frame is taken as ``[0, 1]`` and quantised to 8 bits first.
        Every vehicle of the mask comes back, with its own pixels; the
        ``max_vehicles`` largest of at least ``min_pixels`` are read for
        their lamps, and the rest come back unread.
        """
        height, width = image.shape[:2]
        if image.dtype != np.uint8:
            image = np.clip(np.rint(image.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
        if not np.any(vehicles):
            return []
        # The vehicles first, in the whole frame, to tell apart the ones
        # the labels run together.
        found = self._whole_frame(image)
        found = found[found[:, 4] < len(VEHICLE_PROMPTS)]
        apart = split_outlines(vehicles, separate_vehicles(found, vehicles))
        none = np.zeros((0, 5), dtype=np.float32)
        views = [VehicleView(box, 0.0, 0.0, none, mask, kind) for box, mask, kind in apart]
        readable = [
            index
            for index, (_box, mask, _kind) in enumerate(apart)
            if mask.sum() >= self.min_pixels
        ][: self.max_vehicles]
        cell = self._side // self.grid
        for start in range(0, len(readable), self.grid * self.grid):
            batch = readable[start : start + self.grid * self.grid]
            canvas = np.full((self._side, self._side, 3), 114, dtype=np.uint8)
            placed = []
            for slot, index in enumerate(batch):
                left, top, wide, tall = apart[index][0]
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
            lamps = self._find(canvas, LAMP_PROMPTS + PLATE_PROMPTS, self.min_score)
            middle_x, middle_y = (
                0.5 * (lamps[:, 0] + lamps[:, 2]),
                0.5 * (lamps[:, 1] + lamps[:, 3]),
            )
            for index, (at_x, at_y, wide, tall, scale, x0, y0) in zip(batch, placed, strict=True):
                inside = (
                    (middle_x >= at_x)
                    & (middle_x < at_x + wide)
                    & (middle_y >= at_y)
                    & (middle_y < at_y + tall)
                )
                # From the mosaic back to the frame.
                rows = lamps[inside].copy()
                rows[:, :4] = (rows[:, :4] - [at_x, at_y, at_x, at_y]) / scale + [x0, y0, x0, y0]
                box, mask, kind = apart[index]
                seen = lamps_of(box, mask, rows[rows[:, 4] < len(LAMP_PROMPTS)], self.min_score)
                plate = plate_of(box, rows[rows[:, 4] >= len(LAMP_PROMPTS)])
                views[index] = seen._replace(mask=mask, kind=kind, plate=plate)
        return views
