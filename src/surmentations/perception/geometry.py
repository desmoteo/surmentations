"""How the camera is mounted: its field of view, its tilt, its height.

Every effect that happens *in the scene* needs the camera's geometry. Rain
falls along world vertical, which is not the image's vertical once the
camera is tilted. Level ground is what faces world up. A wet road is a
mirror about the horizon, and how much it reflects goes by the angle it is
seen at. Fog on the ground is a layer so many metres deep.

Until there was this module each transform drew a field of view and a pitch
of its own, at random, from a range: two transforms on one frame disagreed
with each other, and both with the picture. The horizon a wet road needs was
fitted from the depth map alone and came out forty to three hundred rows
off, because a monocular depth map is right in the order of its distances
and wrong in their proportion.

An :class:`InstallationGeometry` says what is known about the mounting. It
may be **given**, which is the right source for a fixed camera: one mast,
one geometry, every frame. Or it may be **estimated** from the frame by a
:class:`GeometryEstimator`. Either way a
:class:`~surmentations.depth.provider.SharedDepthProvider` holds it per
scene, and every transform that asks about that scene gets the same
:class:`Camera`.

What one picture can and cannot tell
------------------------------------
:class:`GroundPlaneGeometryEstimator` recovers what the picture actually
holds, and leaves the rest unknown instead of inventing it.

* **The horizon, and the camera's roll.** On level ground the reciprocal of
  distance is a linear function of position in the picture, and a depth
  network's raw output *is* a reciprocal distance, up to a scale and an
  offset nobody knows. So a plane is fitted to the disparity of the pixels
  labelled as ground, and the horizon is the line on which that plane
  reaches the disparity of the sky, which is the disparity of infinity.
  Then one fact the fit does not know is applied: the ground is never seen
  above the horizon, so the horizon is no lower than the top of the ground.
  On the sample scenes this lands within about forty rows of the horizon
  where the horizon can be judged by eye.
* **Not the field of view.** A plane looks like a plane through any lens.
  The field of view comes from whoever knows the lens. Otherwise one is
  drawn for the scene from a range of plausible lenses, by a rule that
  gives the same frame the same lens every time, and every transform is
  given that one. The *pitch* then follows from the horizon, so whatever
  field of view is taken, the tilt is the one that puts the horizon where
  the picture has it.
* **The height, roughly.** A car is about a metre and a half high and a
  person a little more. How much of the way from its foot to the horizon a
  thing of known height reaches gives the camera's height over the ground.
  It is good to a few tens of per cent, which is far better than no scale
  at all, and far worse than a tape measure: give the height if you have it.
* **Nothing, without a sky.** With no sky in the frame nothing marks the
  disparity of infinity. The fit then gives the row at which the ground
  would reach the farthest thing in view, and the horizon is at that row or
  above it: a bound (:attr:`InstallationGeometry.horizon_is_bound`), under
  which a transform's own draw of the pitch stands unless it breaks it.

Metres
------
With a height, level ground is at a known distance at every row, and the
depth map can be scaled to it: :func:`metric_scale` fits the two numbers
that turn disparity into metres. That replaces the near and far distances a
provider is otherwise told by hand.

Conventions
-----------
Pitch is positive when the camera looks down. Roll is positive when the
horizon runs downhill to the right in the picture. The horizon's position
is a fraction of the frame's height, measured from the top at the middle
column, and may be negative: a camera pitched well down has its horizon
above the frame.
"""

from __future__ import annotations

import math
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import NamedTuple

import cv2
import numpy as np

from surmentations.perception.labels import GROUND, PAVED, SceneClass, class_weight

# Share of the frame that has to be paving before the plane is fitted to
# paving alone; below it, to any ground.
_ENOUGH_PAVING = 0.03
# Fewest ground pixels a plane is fitted to.
_FEWEST_GROUND_PIXELS = 400
# Pixels the plane is fitted to at most: a sample, drawn the same way each time.
_FIT_SAMPLE = 4000
# Passes of the reweighted fit, and how many spreads off a pixel may lie
# before it counts for half.
_FIT_PASSES = 8
_FIT_OUTLIER_SPREADS = 2.5
# Share of the frame that has to be sky for its disparity to be trusted.
_ENOUGH_SKY = 0.005
# A row holding less ground than this share of the frame's width is not the
# top of the ground, and ground that reaches this near the top of the frame
# fills the view.
_GROUND_ROW_SHARE = 0.01
_GROUND_TO_THE_TOP = 0.03

# Height, and length and width on the ground, of what the camera's height is
# judged against, in metres.
_PERSON_SIZE_M = (1.7, 0.4, 0.5)
_VEHICLE_SIZE_M = (1.55, 4.4, 1.8)
# Share of its bounding box a thing has to fill to be one thing, the least
# share of the frame it may cover (and the fewest pixels, on a small frame),
# and how far its width may be from the width its height predicts, as a
# ratio's logarithm.
_OBJECT_BOX_FILL = 0.55
_OBJECT_LEAST_SHARE = 1.2e-4
_OBJECT_FEWEST_PIXELS = 12
_OBJECT_WIDTH_TOLERANCE = 0.25
# Fewest things the height is taken from.
_FEWEST_OBJECTS = 3

# The longer side, in pixels, of the copy the geometry is read from. A
# plane through the ground, the row its top is on and the outlines of the
# cars on it are all there at this size, and a megapixel frame took ten
# times as long to say the same.
_WORK_SIDE = 384

# The farthest distance a calibrated depth map reports, as a multiple of the
# nearest. Beyond a hundred to one the difference between a pixel's
# disparity and infinity's is within the network's own noise.
_FAR_OVER_NEAR = 100.0


def focal_px(width: int, horizontal_fov_deg: float) -> float:
    """Focal length in pixels of a lens that sees ``horizontal_fov_deg`` across ``width``."""
    return 0.5 * width / math.tan(math.radians(horizontal_fov_deg) / 2.0)


class Camera(NamedTuple):
    """One frame's view through an :class:`InstallationGeometry`, with nothing left open.

    ``pitch_deg`` and ``height_m`` are ``None`` where nothing is known and
    nothing was proposed. Rows and columns are in pixels of a
    ``width`` x ``height`` frame.
    """

    width: int
    height: int
    horizontal_fov_deg: float
    pitch_deg: float | None
    roll_deg: float
    height_m: float | None

    @property
    def focal_px(self) -> float:
        """Focal length in pixels."""
        return focal_px(self.width, self.horizontal_fov_deg)

    @property
    def horizon_row(self) -> float | None:
        """Row of the horizon at the middle column; ``None`` with no pitch."""
        if self.pitch_deg is None:
            return None
        return 0.5 * self.height - self.focal_px * math.tan(math.radians(self.pitch_deg))

    def horizon_rows(self, columns: np.ndarray) -> np.ndarray:
        """Row of the horizon at each of ``columns``, which a rolled camera tilts."""
        row = self.horizon_row
        if row is None:
            raise ValueError("this camera's pitch is not known, so neither is its horizon")
        slope = math.tan(math.radians(self.roll_deg))
        return row + (np.asarray(columns, dtype=np.float64) - 0.5 * self.width) * slope

    def below_horizon(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """How far below the horizon each pixel is, in pixels, square to the horizon."""
        return (np.asarray(rows, dtype=np.float64) - self.horizon_rows(columns)) * math.cos(
            math.radians(self.roll_deg)
        )

    def ground_depth_per_metre(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Distance of level ground at each pixel, per metre of camera height.

        Along the optical axis, which is what a depth map holds. Level ground
        ``h`` below a camera of focal length ``f`` pitched down by ``p`` is at

            z = h * f / (drop * cos p)

        where ``drop`` is how far below the horizon the pixel is. Infinite on
        the horizon and above it.
        """
        if self.pitch_deg is None:
            raise ValueError("this camera's pitch is not known, so its ground is not either")
        drop = self.below_horizon(rows, columns)
        scale = self.focal_px / math.cos(math.radians(self.pitch_deg))
        with np.errstate(divide="ignore"):
            return np.where(drop > 0.0, scale / np.maximum(drop, 1e-9), np.inf)


@dataclass(frozen=True)
class InstallationGeometry:
    """What is known about how a camera is mounted. Anything may be left out.

    Args:
        horizontal_fov_deg: the lens's horizontal field of view.
        pitch_deg: how far the camera is tilted down from level.
        roll_deg: how far it is rolled about its axis; positive when the
            horizon runs downhill to the right in the picture.
        height_m: the camera's height over the ground it looks at.
        horizon: where the horizon crosses the middle column, as a fraction
            of the frame's height from the top. An estimate gives this in
            place of ``pitch_deg``, because the horizon is what a picture
            shows and the pitch depends on a field of view it does not.
            Ignored when ``pitch_deg`` is given.
        horizon_is_bound: the horizon is at ``horizon`` *or above it*: the
            camera is tilted down at least that far, and perhaps further.

    For a fixed camera, give what you know::

        InstallationGeometry(horizontal_fov_deg=92.0, pitch_deg=24.0, height_m=6.5)
    """

    horizontal_fov_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float = 0.0
    height_m: float | None = None
    horizon: float | None = None
    horizon_is_bound: bool = False

    def __post_init__(self) -> None:
        """Refuse a geometry no camera has."""
        fov = self.horizontal_fov_deg
        if fov is not None and not 1.0 <= fov <= 179.0:
            raise ValueError(f"horizontal_fov_deg must be in [1, 179], got {fov}")
        if self.pitch_deg is not None and not -90.0 < self.pitch_deg < 90.0:
            raise ValueError(f"pitch_deg must be in (-90, 90), got {self.pitch_deg}")
        if not -90.0 < self.roll_deg < 90.0:
            raise ValueError(f"roll_deg must be in (-90, 90), got {self.roll_deg}")
        if self.height_m is not None and not self.height_m > 0.0:
            raise ValueError(f"height_m must be positive, got {self.height_m}")

    @property
    def fixes_tilt(self) -> bool:
        """Whether the camera's tilt is known, and not left open or merely bounded.

        True of a geometry given a pitch, or a horizon that is not a bound:
        the horizon of a frame seen through it is then a fact about the
        frame, and not a consequence of whatever pitch a transform drew.
        """
        return self.pitch_deg is not None or (
            self.horizon is not None and not self.horizon_is_bound
        )

    def filled_from(self, other: InstallationGeometry) -> InstallationGeometry:
        """This geometry, with what it leaves open taken from ``other``.

        What is given outranks what is estimated: a pitch here is kept and
        ``other``'s horizon ignored.
        """
        knows_tilt = self.pitch_deg is not None or self.horizon is not None
        return replace(
            self,
            horizontal_fov_deg=(
                other.horizontal_fov_deg
                if self.horizontal_fov_deg is None
                else self.horizontal_fov_deg
            ),
            horizon=self.horizon if knows_tilt else other.horizon,
            horizon_is_bound=self.horizon_is_bound if knows_tilt else other.horizon_is_bound,
            roll_deg=other.roll_deg if self.roll_deg == 0.0 and not knows_tilt else self.roll_deg,
            height_m=other.height_m if self.height_m is None else self.height_m,
        )

    def view(
        self,
        width: int,
        height: int,
        horizontal_fov_deg: float | None = None,
        pitch_deg: float | None = None,
    ) -> Camera | None:
        """The :class:`Camera` for a ``width`` x ``height`` frame.

        ``horizontal_fov_deg`` and ``pitch_deg`` are a caller's proposals
        for what this geometry leaves open, and are ignored where it does
        not. The pitch comes, in order, from ``pitch_deg`` given here at
        construction; from the horizon, through the field of view; and from
        the proposal, which a horizon that is only a bound may push further
        down. ``None`` when no field of view is known or proposed.
        """
        fov = self.horizontal_fov_deg
        fov = horizontal_fov_deg if fov is None else fov
        if fov is None:
            return None
        pitch = self.pitch_deg
        if pitch is None and self.horizon is not None:
            focal = focal_px(width, fov)
            implied = math.degrees(math.atan((0.5 - self.horizon) * height / focal))
            if not self.horizon_is_bound:
                pitch = implied
            elif pitch_deg is not None:
                pitch = max(pitch_deg, implied)
        elif pitch is None:
            pitch = pitch_deg
        return Camera(width, height, float(fov), pitch, self.roll_deg, self.height_m)


class GeometryEstimator(ABC):
    """Interface for whatever works an :class:`InstallationGeometry` out of a frame."""

    @abstractmethod
    def estimate(
        self, image: np.ndarray, disparity: np.ndarray | None, labels: np.ndarray | None
    ) -> InstallationGeometry:
        """What can be told of the mounting from one ``(H, W, 3)`` frame.

        ``disparity`` is the reciprocal of distance up to a scale and an
        offset, ``(H, W)``, larger for nearer; ``None`` from a depth backend
        that cannot give one. ``labels`` are the frame's
        :class:`~surmentations.perception.labels.SceneClass` map, or
        ``None`` with no label estimator.
        """


def fit_ground_disparity(
    disparity: np.ndarray, ground: np.ndarray
) -> tuple[float, float, float] | None:
    """The plane through the ground's disparity: ``a * x + b * y + c``.

    ``x`` and ``y`` are fractions of the frame's width and height, so the
    plane means the same at any resolution. Fitted to a fixed sample of the
    pixels where ``ground`` is set, reweighted over a few passes so that
    what stands on the ground and was labelled as ground does not tilt it.
    ``None`` with too little ground, or ground that does not get nearer
    down the frame.
    """
    height, width = disparity.shape
    rows, columns = np.nonzero(ground)
    if rows.size < _FEWEST_GROUND_PIXELS:
        return None
    if rows.size > _FIT_SAMPLE:
        # The same sample for the same frame: an estimate that moved with a
        # seed would make two runs disagree about where the horizon is.
        keep = np.random.default_rng(0).choice(rows.size, _FIT_SAMPLE, replace=False)
        rows, columns = rows[keep], columns[keep]
    x = (columns + 0.5) / width
    y = (rows + 0.5) / height
    values = disparity[rows, columns].astype(np.float64)
    design = np.stack([x, y, np.ones_like(x)], axis=1)
    weights = np.ones_like(values)
    plane = np.zeros(3)
    for _ in range(_FIT_PASSES):
        plane, *_ = np.linalg.lstsq(design * weights[:, None], values * weights, rcond=None)
        residual = values - design @ plane
        spread = 1.4826 * float(np.median(np.abs(residual))) + 1e-9
        weights = 1.0 / (1.0 + (residual / (_FIT_OUTLIER_SPREADS * spread)) ** 2)
    if not plane[1] > 1e-6:
        return None
    return float(plane[0]), float(plane[1]), float(plane[2])


def top_of_ground(ground: np.ndarray, slope: float = 0.0) -> float:
    """The highest row that holds ground, as a fraction of the frame's height.

    Measured at the middle column along lines of ``slope`` rows per column,
    which is the horizon's own slope under a rolled camera: the ground then
    reaches higher on one side of the frame than the other, and it is the
    line through its top that the horizon cannot be below. Zero when the
    ground reaches the top of the frame, or there is none.
    """
    height, width = ground.shape
    rows, columns = np.nonzero(ground)
    if rows.size == 0:
        return 0.0
    levelled = np.rint(rows - (columns - 0.5 * width) * slope).astype(np.int64)
    lowest = int(levelled.min())
    held = np.flatnonzero(
        np.bincount(levelled - lowest) >= max(3.0, _GROUND_ROW_SHARE * width)
    )
    if held.size == 0:
        return 0.0
    top = float(held[0] + lowest) / height
    return 0.0 if top <= _GROUND_TO_THE_TOP else top


def camera_height_from_objects(labels: np.ndarray, camera: Camera) -> float | None:
    """The camera's height in metres, from the vehicles and people in view.

    A thing of height ``H`` standing on level ground reaches a known share
    of the way from its foot to the horizon: all of it for a camera at
    height ``H``, half for one at ``2 H``. That is the whole of it for a
    person. A car seen from above also shows its roof and bonnet, so its
    top edge is the *far* edge of its roof, further off than its foot by
    the car's own length; each car is therefore tried at every heading, and
    the heading kept is the one at which its width in the picture comes out
    right as well. Things cut by the frame's edge, run together with their
    neighbours, or fitting no heading are left out, and the median of the
    rest is the answer.

    ``None`` with fewer than three usable things, or a camera whose pitch is
    not known. Good to a few tens of per cent.
    """
    if camera.pitch_deg is None:
        return None
    height, width = labels.shape
    focal = camera.focal_px
    pitch = math.radians(camera.pitch_deg)
    fewest = max(_OBJECT_FEWEST_PIXELS, _OBJECT_LEAST_SHARE * height * width)

    def depression(row: float, column: float) -> float:
        """Angle below the horizontal of the ray through a pixel, in radians.

        From how far below the horizon the pixel is: the horizon is the
        reference for every angle here, which is what makes the height
        independent of an error in the pitch to first order.
        """
        drop = float(camera.below_horizon(np.array(row), np.array(column)))
        return pitch + math.atan(drop / focal - math.tan(pitch))

    found: list[float] = []
    for scene_class, (tall, long, wide) in (
        (SceneClass.PERSON, _PERSON_SIZE_M),
        (SceneClass.VEHICLE, _VEHICLE_SIZE_M),
    ):
        mask = (labels == int(scene_class)).astype(np.uint8)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for index in range(1, count):
            left, top, box_w, box_h, area = (int(value) for value in stats[index])
            cut = left <= 1 or top <= 1 or left + box_w >= width - 1 or top + box_h >= height - 1
            if cut or area < fewest or area < _OBJECT_BOX_FILL * box_w * box_h:
                continue
            middle = left + 0.5 * box_w
            at_foot, at_top = depression(top + box_h, middle), depression(top, middle)
            if at_foot <= 0.01 or at_top <= 0.0 or at_top >= at_foot:
                continue
            best: tuple[float, float] | None = None
            for heading in np.radians(np.arange(0.0, 91.0, 5.0)):
                reach = long * math.cos(heading) + wide * math.sin(heading)
                across = long * math.sin(heading) + wide * math.cos(heading)
                # tan(at_top) = (h - H) / (D + reach), tan(at_foot) = h / D.
                over = (tall + reach * math.tan(at_top)) / max(
                    1.0 - math.tan(at_top) / math.tan(at_foot), 1e-3
                )
                distance = over / math.tan(at_foot)
                along_axis = distance * math.cos(pitch) + over * math.sin(pitch)
                predicted = focal * across / max(along_axis, 1e-3)
                error = abs(math.log(predicted / box_w))
                if best is None or error < best[0]:
                    best = (error, over)
            if best is not None and best[0] < _OBJECT_WIDTH_TOLERANCE:
                found.append(best[1])
    if len(found) < _FEWEST_OBJECTS:
        return None
    return float(np.median(found))


def metric_scale(
    disparity: np.ndarray, ground: np.ndarray, camera: Camera
) -> tuple[float, float] | None:
    """The two numbers that turn a disparity map into metres: ``k / (disparity - t)``.

    Level ground is at a known distance at every pixel once the camera's
    height, pitch and field of view are (:meth:`Camera.ground_depth_per_metre`),
    and disparity is a scale ``k`` times the reciprocal of distance plus an
    offset ``t``. So the ground's disparity is regressed on the reciprocal
    of its known distance: the slope is ``k`` and the intercept ``t``, the
    disparity of infinity, which needs no sky in the frame to find.

    ``None`` without a height or a pitch, with too little ground, or when
    the ground's disparity does not grow with nearness.
    """
    if camera.height_m is None or camera.pitch_deg is None:
        return None
    rows, columns = np.nonzero(ground)
    if rows.size < _FEWEST_GROUND_PIXELS:
        return None
    if rows.size > _FIT_SAMPLE:
        keep = np.random.default_rng(0).choice(rows.size, _FIT_SAMPLE, replace=False)
        rows, columns = rows[keep], columns[keep]
    distance = camera.height_m * camera.ground_depth_per_metre(rows + 0.5, columns + 0.5)
    usable = np.isfinite(distance)
    if int(usable.sum()) < _FEWEST_GROUND_PIXELS:
        return None
    nearness = 1.0 / distance[usable]
    values = disparity[rows[usable], columns[usable]].astype(np.float64)
    design = np.stack([nearness, np.ones_like(nearness)], axis=1)
    weights = np.ones_like(values)
    line = np.zeros(2)
    for _ in range(_FIT_PASSES):
        line, *_ = np.linalg.lstsq(design * weights[:, None], values * weights, rcond=None)
        residual = values - design @ line
        spread = 1.4826 * float(np.median(np.abs(residual))) + 1e-9
        weights = 1.0 / (1.0 + (residual / (_FIT_OUTLIER_SPREADS * spread)) ** 2)
    scale, offset = float(line[0]), float(line[1])
    if not scale > 1e-9:
        return None
    return scale, offset


def metric_depth(disparity: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Depth in metres from a disparity map and :func:`metric_scale`'s two numbers.

    Capped at a hundred times the nearest distance in the frame: past that
    the difference between a pixel's disparity and infinity's is inside the
    network's noise, and the sky's distance is whatever the noise says.
    """
    nearest = max(float(disparity.max()) - offset, 1e-6)
    floor = nearest / _FAR_OVER_NEAR
    return (scale / np.maximum(disparity - offset, floor)).astype(np.float32)


class GroundPlaneGeometryEstimator(GeometryEstimator):
    """The horizon, the roll and the height, from the ground's own shape.

    The estimate described in the module docstring: a plane through the
    ground's disparity, the horizon where it meets the sky's, never lower
    than the top of the ground; the camera's height from the vehicles and
    people standing on it. It needs scene labels to know which pixels are
    ground and which are sky, and a depth backend that can give its
    disparity; without either it returns a field of view and the height it
    was given, and nothing more.

    Args:
        horizontal_fov_deg: the lens's field of view, if known. A picture
            does not tell it.
        fov_range_deg: the fields of view a lens that is not known may
            have. Each scene is given one from this range, the same one
            whenever the same frame is seen again: an unknown lens is one
            lens, not a new one for every transform or every epoch.
        height_m: the camera's height, if known. Left out, it is estimated
            from the things in view when there are enough of them.
        estimate_height: whether to estimate a height that was not given.
            A height puts every depth-conditioned effect in real metres,
            scene by scene; turn this off to keep the provider's own
            ``depth_range_m`` for every frame alike.
        max_roll_deg: the largest roll believed. A fit that says more is a
            road running across the frame on a slope, not a tilted camera,
            and is taken as no roll.
    """

    def __init__(
        self,
        horizontal_fov_deg: float | None = None,
        height_m: float | None = None,
        fov_range_deg: tuple[float, float] = (48.0, 78.0),
        estimate_height: bool = True,
        max_roll_deg: float = 8.0,
    ) -> None:
        """Keep what is known of the camera; see the class docstring for the arguments."""
        # Validated by the same rules a geometry is.
        InstallationGeometry(horizontal_fov_deg=horizontal_fov_deg, height_m=height_m)
        narrow, wide = fov_range_deg
        InstallationGeometry(horizontal_fov_deg=narrow)
        InstallationGeometry(horizontal_fov_deg=wide)
        if narrow > wide:
            raise ValueError(f"fov_range_deg must be (narrow, wide), got {fov_range_deg}")
        self.horizontal_fov_deg = horizontal_fov_deg
        self.height_m = height_m
        self.fov_range_deg = (float(narrow), float(wide))
        self.estimate_height = estimate_height
        self.max_roll_deg = float(max_roll_deg)

    def _field_of_view(self, image: np.ndarray) -> float:
        """The field of view this frame is taken to have.

        The one given, or one from ``fov_range_deg`` chosen by the frame's
        own content: a sparse grid of its pixels seeds the draw, so the same
        frame gets the same lens in every process and every run.
        """
        if self.horizontal_fov_deg is not None:
            return self.horizontal_fov_deg
        step_y, step_x = max(1, image.shape[0] // 16), max(1, image.shape[1] // 16)
        seed = zlib.crc32(np.ascontiguousarray(image[::step_y, ::step_x]).tobytes())
        return float(np.random.default_rng(seed).uniform(*self.fov_range_deg))

    def estimate(
        self, image: np.ndarray, disparity: np.ndarray | None, labels: np.ndarray | None
    ) -> InstallationGeometry:
        """The geometry of one frame; see the class docstring.

        What cannot be told is left as ``None``: with no labels, no
        disparity or no ground in view, that is everything but the field of
        view and a height given at construction.
        """
        fov = self._field_of_view(image)
        given = InstallationGeometry(horizontal_fov_deg=fov, height_m=self.height_m)
        if disparity is None or labels is None:
            return given
        # On a reduced copy: everything read below is a fraction of the
        # frame or an angle, and means the same at any size.
        longer = max(labels.shape)
        if longer > _WORK_SIDE:
            size = (
                max(8, round(labels.shape[1] * _WORK_SIDE / longer)),
                max(8, round(labels.shape[0] * _WORK_SIDE / longer)),
            )
            labels = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
            disparity = cv2.resize(disparity, size, interpolation=cv2.INTER_AREA)
        height, width = labels.shape
        paving = class_weight(labels, PAVED) > 0.5
        ground = paving
        if float(paving.mean()) < _ENOUGH_PAVING:
            ground = class_weight(labels, GROUND) > 0.5
        plane = fit_ground_disparity(disparity, ground)
        if plane is None:
            return given
        across, down, constant = plane
        sky = labels == int(SceneClass.SKY)
        has_sky = float(sky.mean()) >= _ENOUGH_SKY
        # With no sky, the farthest thing in view stands in for infinity,
        # and the horizon found is the lowest it can be.
        infinity = float(np.median(disparity[sky])) if has_sky else float(disparity.min())
        horizon = (infinity - constant - 0.5 * across) / down
        slope = -across * height / (down * width)
        roll = math.degrees(math.atan(slope))
        if abs(roll) > self.max_roll_deg:
            roll = slope = 0.0
        horizon = min(horizon, top_of_ground(ground, slope))
        estimate = replace(
            given, horizon=float(horizon), horizon_is_bound=not has_sky, roll_deg=roll
        )
        if self.height_m is None and self.estimate_height:
            camera = replace(estimate, horizon_is_bound=False).view(width, height)
            if camera is not None:
                estimate = replace(estimate, height_m=camera_height_from_objects(labels, camera))
        return estimate
