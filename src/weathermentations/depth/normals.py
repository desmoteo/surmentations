"""Surface normals from a depth map.

Snow settles on surfaces that face **up**. Not on surfaces that are near the
bottom of the frame — that is a proxy for "ground" which fails on everything
interesting: a wall at the bottom of the frame gets snow it should not have,
a car roof or a rooftop in the upper half gets none when it should be the
most obvious snow in the shot.

The real quantity is the angle between a surface and vertical, and a depth
map is enough to recover it. Back-projecting each pixel through a pinhole
camera gives a 3-D point,

    X = (u - cx) * z / f,   Y = (v - cy) * z / f,   Z = z

and the surface normal is the cross product of that surface's two tangent
vectors, which are just the horizontal and vertical derivatives of ``P``.
The accumulation weight is then ``max(0, n . up)`` — Lambert's cosine law
applied to falling snow instead of to light, which is the same geometry:
what lands on a surface per unit area scales with the cosine of the angle
between the surface normal and the fall direction.

Caveats worth being honest about, since this is an estimate and not a
measurement:

* Monocular depth is *relative* and noisy. Its absolute scale is a guess
  (see ``depth/scale.py``), so the normals are reliable in direction but not
  in magnitude, and the depth is smoothed before differencing — raw
  gradients of a learned depth map are dominated by its own noise.
* Depth discontinuities (an object's silhouette against what is behind it)
  are not surfaces, and differencing across one invents a surface that joins
  the two sides. That used to be left to luck, and the luck ran out in two
  visible ways. Where the far side is *above* the near one — a treeline
  against the sky — the invented surface recedes upward exactly as ground
  does, so it collected a bright rim of snow. And because the depth was
  blurred first, every object's outline was smeared into a band several
  pixels wide that read as "not ground": the wet-road darkening stopped short
  of every car and left it standing in a dry halo.

  So derivatives are **one-sided** — each pixel takes its tangent from
  whichever neighbour is on its own side of the jump — the depth is blurred
  by one pixel rather than three, and a pixel both of whose neighbours are a
  large step away is treated as lying on an edge rather than on a surface.
  Measured on the reference photo, the snow weight along the skyline fell
  from 0.34 to 0.01, and the road beside a car, which had come out up to
  10.9% brighter than open road in the same row, is now within 5.4%. That is
  halved, not gone: what remains is the softness of the depth map itself,
  which is upsampled from a few hundred pixels and cannot place a silhouette
  to the pixel.

  Snapping the result to the *image's* edges with a guided filter was tried
  and rejected on that same measurement: it scored worse than the original
  (23.5%), because wherever the photograph has contrast that the geometry
  does not — lane markings, a painted kerb — the filter copies it into the
  mask.
* Nothing here models shelter: the underside of a bridge faces up and will
  be given snow it would never receive in reality.
"""

from __future__ import annotations

import cv2
import numpy as np

from weathermentations import mode

# Assumed horizontal field of view when none is supplied. A typical
# surveillance/dashcam lens sits near here, and the normal direction is only
# weakly sensitive to it.
DEFAULT_HORIZONTAL_FOV_DEG = 60.0


# Baseline, in pixels, of the one-sided differences. Two rather than one:
# monocular depth is noisy pixel to pixel, and a longer baseline averages that
# down without smearing the depth itself across edges the way a blur does.
_DIFFERENCE_STEP_PX = 2


_Planes = tuple[np.ndarray, np.ndarray, np.ndarray]


def _one_sided_tangent(point: _Planes, axis: int, step: int) -> _Planes:
    """Derivative of the 3-D ``point`` field along ``axis`` that does not cross depth jumps.

    Every pixel has two neighbours along an axis and so two candidate
    tangents. On a smooth surface they agree. At a silhouette one of them
    spans the jump to whatever is behind, and the other stays on the pixel's
    own surface — which is recognisably the one with the smaller change in
    depth. Taking that one gives each side of an edge its own normal instead
    of both sides sharing a fictitious one.

    ``point`` and the result are ``(x, y, z)`` as three separate planes: every
    step downstream is a per-component expression, and keeping the components
    apart is several times faster than interleaving them in one array.
    """
    size = point[2].shape[axis]
    if size < 2:
        return (np.zeros_like(point[0]), np.zeros_like(point[1]), np.zeros_like(point[2]))
    step = max(1, min(step, (size - 1) // 2))

    def along(start: int | None, stop: int | None) -> tuple[slice, ...]:
        index = [slice(None), slice(None)]
        index[axis] = slice(start, stop)
        return tuple(index)

    def delta(plane: np.ndarray) -> np.ndarray:
        # delta[i] = plane[i + step] - plane[i]: the forward difference at i
        # and the backward difference at i + step.
        difference: np.ndarray = plane[along(step, None)] - plane[along(None, -step)]
        return difference

    depth_delta = delta(point[2])
    use_forward: np.ndarray | None = None
    if size > 2 * step:
        use_forward = np.abs(depth_delta[along(step, None)]) <= np.abs(
            depth_delta[along(None, -step)]
        )

    def tangent(difference: np.ndarray) -> np.ndarray:
        result = np.empty((point[2].shape), dtype=np.float32)
        # Within `step` of either border only one neighbour exists.
        result[along(None, step)] = difference[along(None, step)]
        result[along(-step, None)] = difference[along(-step, None)]
        if use_forward is not None:
            result[along(step, -step)] = np.where(
                use_forward, difference[along(step, None)], difference[along(None, -step)]
            )
        return result

    return (tangent(delta(point[0])), tangent(delta(point[1])), tangent(depth_delta))


# Relative change in depth across the difference baseline above which a
# pixel is taken to sit *on* a silhouette rather than on a surface. A ground
# plane only reaches this within a dozen rows of its own horizon, where it is
# edge-on and carries nothing anyway; the ramp between a treetop and the sky
# behind it exceeds it several times over.
_SILHOUETTE_RELATIVE_JUMP = 0.2

# The longer side, in pixels, the depth network works at, and how much larger
# than that a frame has to be before fast mode reads its normals there.
_DEPTH_WORK_PX = 518
_REDUCE_ABOVE = 1.5


def _normals_and_jump(
    depth_m: np.ndarray, horizontal_fov_deg: float, smooth_sigma_px: float
) -> tuple[_Planes, np.ndarray]:
    """Unit normals, as planes, and the relative depth change each was computed across."""
    height, width = depth_m.shape[:2]
    focal = float(width / (2.0 * np.tan(np.deg2rad(horizontal_fov_deg) / 2.0)))
    cx, cy = width / 2.0, height / 2.0

    # Smoothed only as much as it must be. The gradient of a learned depth
    # map is noisy, but every pixel of blur also spreads each silhouette into
    # a ramp that one-sided differences cannot step around. One pixel is the
    # measured compromise: none at all left row-to-row ripple in the result
    # eleven times the original's, visible as scanlines inside a dark shadow.
    depth: np.ndarray = depth_m.astype(np.float32)
    if smooth_sigma_px > 0.1:
        depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=smooth_sigma_px)

    # float32 scalars throughout: a float64 one would promote every plane.
    u = (np.arange(width, dtype=np.float32).reshape(1, width) - np.float32(cx)) / np.float32(focal)
    v = (np.arange(height, dtype=np.float32).reshape(height, 1) - np.float32(cy)) / np.float32(
        focal
    )
    point: _Planes = (u * depth, v * depth, depth)

    ux, uy, uz = _one_sided_tangent(point, axis=1, step=_DIFFERENCE_STEP_PX)
    vx, vy, vz = _one_sided_tangent(point, axis=0, step=_DIFFERENCE_STEP_PX)

    # n = tangent_u x tangent_v
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    # Orient consistently toward the camera, or half the frame comes out
    # inverted depending on which way the cross product happened to fall.
    # The test has to be against the *view direction* (the camera sits at the
    # origin, so a visible surface satisfies n . P < 0), not against the sign
    # of n_z: a horizontal ground plane under a level camera has the normal
    # (0, -1, 0), whose z component is exactly zero, so an n_z test leaves
    # precisely the most important surface in the scene arbitrarily signed.
    facing = nx * point[0] + ny * point[1] + nz * point[2]
    scale = np.where(facing > 0.0, np.float32(-1.0), np.float32(1.0)) / np.maximum(
        length, np.float32(1e-8)
    )
    scale[length <= 1e-8] = 0.0

    jump = np.maximum(np.abs(uz), np.abs(vz)) / np.maximum(depth, np.float32(1e-6))
    return (nx * scale, ny * scale, nz * scale), jump


def surface_normals(
    depth_m: np.ndarray,
    horizontal_fov_deg: float = DEFAULT_HORIZONTAL_FOV_DEG,
    smooth_sigma_px: float = 1.0,
) -> np.ndarray:
    """``(H, W, 3)`` unit surface normals in camera coordinates, from metric depth.

    Camera convention: ``+x`` right, ``+y`` down, ``+z`` into the scene. A
    surface facing the camera therefore has a normal near ``(0, 0, -1)``, and
    a horizontal ground plane has one near ``(0, -1, 0)``.
    """
    normals, _jump = _normals_and_jump(depth_m, horizontal_fov_deg, smooth_sigma_px)
    return np.stack(normals, axis=-1).astype(np.float32)


def depth_work_size(height: int, width: int) -> tuple[int, int] | None:
    """``(width, height)`` to read a frame's depth at, or ``None`` for as it is.

    The depth network works at about 518 pixels on the longer side and its
    map is interpolated up to the frame. Anything read off the map alone,
    the lie of the ground for one, is as well read at that size and enlarged,
    at a fraction of the pixels. Fast mode does so for frames more than half
    as large again; exact mode (:mod:`weathermentations.mode`) never does.
    """
    longest = max(height, width)
    if not mode.fast() or longest <= _REDUCE_ABOVE * _DEPTH_WORK_PX:
        return None
    scale = _DEPTH_WORK_PX / longest
    return max(2, round(width * scale)), max(2, round(height * scale))


def upward_facing(
    depth_m: np.ndarray,
    horizontal_fov_deg: float = DEFAULT_HORIZONTAL_FOV_DEG,
    camera_pitch_deg: float = 0.0,
    smooth_sigma_px: float = 1.0,
) -> np.ndarray:
    """``(H, W)`` in ``[0, 1]``: how strongly each pixel's surface faces upward.

    This is ``max(0, n . up)``, the cosine factor that decides how much
    falling snow a surface collects per unit of its projected area. 1 is a
    perfectly horizontal surface, 0 is vertical or overhanging.

    ``camera_pitch_deg`` is positive when the camera is tilted **down**,
    which is the usual mounting for a surveillance camera and rotates world
    "up" away from the image's vertical axis.

    In fast mode (:mod:`weathermentations.mode`) a large frame's normals are
    read at the resolution the depth map was estimated at and the result is
    enlarged. The depth network works at about 518 pixels and its map is
    interpolated up from there, so the differences taken between neighbours
    at 1024 pixels are differences of interpolated values.
    """
    height, width = depth_m.shape[:2]
    size = depth_work_size(height, width)
    if size is not None:
        scale = size[0] / width
        small = cv2.resize(depth_m.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        # The smoothing is a distance in the scene, so it shrinks with the frame.
        sigma = smooth_sigma_px * scale
        facing = _upward_facing(small, horizontal_fov_deg, camera_pitch_deg, sigma)
        enlarged = cv2.resize(facing, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.clip(enlarged, 0.0, 1.0).astype(np.float32)
    return _upward_facing(depth_m, horizontal_fov_deg, camera_pitch_deg, smooth_sigma_px)


def _upward_facing(
    depth_m: np.ndarray,
    horizontal_fov_deg: float,
    camera_pitch_deg: float,
    smooth_sigma_px: float,
) -> np.ndarray:
    """:func:`upward_facing` at the resolution it is given."""
    (_nx, ny, nz), jump = _normals_and_jump(depth_m, horizontal_fov_deg, smooth_sigma_px)

    # World up expressed in camera coordinates. With no pitch that is -y
    # (image y grows downward); pitching the camera down tilts it toward -z.
    pitch = np.deg2rad(camera_pitch_deg)
    cosine = ny * np.float32(-np.cos(pitch)) + nz * np.float32(-np.sin(pitch))

    # A pixel whose *smaller* depth step is still a large fraction of its
    # depth is not on a surface at all: it is on the blurred edge between two
    # of them, and has no orientation worth reading.
    on_surface = 1.0 / (1.0 + (jump / np.float32(_SILHOUETTE_RELATIVE_JUMP)) ** 4)
    facing_up: np.ndarray = (np.clip(cosine, 0.0, 1.0) * on_surface).astype(np.float32)

    # The choice between a pixel's two neighbours flips from one pixel to the
    # next along an edge, which leaves the outline of every object stepped.
    # A one-pixel blur of the *result* takes that off without moving the edge.
    if smooth_sigma_px > 0.1:
        facing_up = cv2.GaussianBlur(facing_up, (0, 0), sigmaX=smooth_sigma_px)
    return np.clip(facing_up, 0.0, 1.0).astype(np.float32)


def horizon_row(
    depth_m: np.ndarray,
    ground_weight: np.ndarray,
    min_weight: float = 0.3,
) -> float:
    """Image row where the ground plane recedes to infinity, from the depth map.

    A specular reflection in wet ground is not a flip about an arbitrary line.
    For a horizontal plane the outgoing ray's vertical angle is the incoming
    one mirrored about the horizontal, so the reflection seen at row ``y``
    comes from row ``2 * y_horizon - y``. Everything therefore hinges on where
    the horizon actually is, and the depth map knows: on a ground plane at
    height ``h`` under a camera of focal length ``f``,

        z = h * f / (y - y_horizon)   =>   1/z = (y - y_horizon) / (h * f)

    so **inverse** depth is linear in the row index and crosses zero at the
    horizon. Fitting that line over the ground pixels and reading off its root
    recovers the horizon without knowing ``h``, ``f`` or the camera pitch.

    Args:
        depth_m: metric (or pseudo-metric) depth.
        ground_weight: ``(H, W)`` in ``[0, 1]``, e.g. from :func:`upward_facing`.
        min_weight: pixels below this weight are not ground and are ignored.

    Returns:
        The horizon row, which is routinely **negative** — a camera pitched
        down puts the horizon above the top of the frame. Falls back to just
        above the topmost ground pixel when the fit is degenerate (too few
        ground pixels, or a ground plane so flat in inverse depth that the
        line has no usable slope).
    """
    height = depth_m.shape[0]
    mask = ground_weight >= min_weight
    if mask.sum() < 32:
        return -float(height)

    rows = np.broadcast_to(
        np.arange(height, dtype=np.float64).reshape(height, 1), depth_m.shape
    )[mask]
    inverse = 1.0 / np.maximum(depth_m[mask].astype(np.float64), 1e-6)

    row_spread = rows.var()
    if row_spread < 1e-6:
        return -float(height)
    slope = ((rows - rows.mean()) * (inverse - inverse.mean())).mean() / row_spread
    if slope <= 1e-12:
        # Inverse depth must *increase* down the frame for a ground plane. A
        # flat or inverted fit means whatever was masked is not one.
        top = int(np.argmax(mask.any(axis=1)))
        return float(top) - 1.0
    return float(rows.mean() - inverse.mean() / slope)
