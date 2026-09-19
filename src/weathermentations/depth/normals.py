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
* Depth discontinuities (an object's silhouette against a distant
  background) produce enormous fake gradients, which back-project to
  surfaces nearly edge-on to the camera. Those read as *vertical*, so they
  correctly receive no snow, but it is luck rather than design.
* Nothing here models shelter: the underside of a bridge faces up and will
  be given snow it would never receive in reality.
"""

from __future__ import annotations

import cv2
import numpy as np

# Assumed horizontal field of view when none is supplied. A typical
# surveillance/dashcam lens sits near here, and the normal direction is only
# weakly sensitive to it.
DEFAULT_HORIZONTAL_FOV_DEG = 60.0


def surface_normals(
    depth_m: np.ndarray,
    horizontal_fov_deg: float = DEFAULT_HORIZONTAL_FOV_DEG,
    smooth_sigma_px: float = 3.0,
) -> np.ndarray:
    """``(H, W, 3)`` unit surface normals in camera coordinates, from metric depth.

    Camera convention: ``+x`` right, ``+y`` down, ``+z`` into the scene. A
    surface facing the camera therefore has a normal near ``(0, 0, -1)``, and
    a horizontal ground plane has one near ``(0, -1, 0)``.
    """
    height, width = depth_m.shape[:2]
    # float() matters: np.tan returns a float64 scalar, and dividing the
    # float32 coordinate grids by it would promote the whole back-projection
    # to float64 — which cv2.Sobel then refuses to write into a CV_32F output.
    focal = float(width / (2.0 * np.tan(np.deg2rad(horizontal_fov_deg) / 2.0)))
    cx, cy = width / 2.0, height / 2.0

    # Smooth before differencing: the gradient of a learned depth map is
    # otherwise mostly the network's own high-frequency noise.
    depth: np.ndarray = depth_m.astype(np.float32)
    if smooth_sigma_px > 0.1:
        depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=smooth_sigma_px)

    u = np.arange(width, dtype=np.float32).reshape(1, width) - cx
    v = np.arange(height, dtype=np.float32).reshape(height, 1) - cy

    point_x = u * depth / focal
    point_y = v * depth / focal
    point_z = depth

    # Tangent vectors along the image axes. Sobel gives a slightly smoothed
    # central difference, which suits a noisy depth map better than a raw one.
    def d_du(field: np.ndarray) -> np.ndarray:
        return cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3)

    def d_dv(field: np.ndarray) -> np.ndarray:
        return cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=3)

    tangent_u = np.stack([d_du(point_x), d_du(point_y), d_du(point_z)], axis=-1)
    tangent_v = np.stack([d_dv(point_x), d_dv(point_y), d_dv(point_z)], axis=-1)

    normals = np.cross(tangent_u, tangent_v)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 1e-8)

    # Orient consistently toward the camera, or half the frame comes out
    # inverted depending on which way the cross product happened to fall.
    # The test has to be against the *view direction* (the camera sits at the
    # origin, so a visible surface satisfies n . P < 0), not against the sign
    # of n_z: a horizontal ground plane under a level camera has the normal
    # (0, -1, 0), whose z component is exactly zero, so an n_z test leaves
    # precisely the most important surface in the scene arbitrarily signed.
    point = np.stack([point_x, point_y, point_z], axis=-1)
    facing = np.sum(normals * point, axis=-1, keepdims=True)
    flip = np.where(facing > 0.0, -1.0, 1.0).astype(np.float32)
    return (normals * flip).astype(np.float32)


def upward_facing(
    depth_m: np.ndarray,
    horizontal_fov_deg: float = DEFAULT_HORIZONTAL_FOV_DEG,
    camera_pitch_deg: float = 0.0,
    smooth_sigma_px: float = 3.0,
) -> np.ndarray:
    """``(H, W)`` in ``[0, 1]``: how strongly each pixel's surface faces upward.

    This is ``max(0, n . up)``, the cosine factor that decides how much
    falling snow a surface collects per unit of its projected area. 1 is a
    perfectly horizontal surface, 0 is vertical or overhanging.

    ``camera_pitch_deg`` is positive when the camera is tilted **down**,
    which is the usual mounting for a surveillance camera and rotates world
    "up" away from the image's vertical axis.
    """
    normals = surface_normals(depth_m, horizontal_fov_deg, smooth_sigma_px)

    # World up expressed in camera coordinates. With no pitch that is -y
    # (image y grows downward); pitching the camera down tilts it toward -z.
    pitch = np.deg2rad(camera_pitch_deg)
    up = np.array([0.0, -np.cos(pitch), -np.sin(pitch)], dtype=np.float32)

    return np.clip(normals @ up, 0.0, 1.0).astype(np.float32)
