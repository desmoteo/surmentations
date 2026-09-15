"""Relative -> pseudo-metric depth calibration.

Monocular relative depth (see :class:`~weathermentations.depth.estimator.
DepthEstimator`) has no absolute scale: it only says "farther" vs "nearer",
not "how far". Physical effect formulas (e.g. fog's Koschmieder's law) need
*some* distance unit for their coefficients to mean anything consistent
across images, so we map relative depth into "pseudo-meters" via a scale
factor — either supplied by the caller (if the true scene scale is known)
or a documented heuristic default.
"""

from __future__ import annotations

import numpy as np

# Heuristic default: treat the farthest point Depth Anything sees in a typical
# outdoor/street scene as roughly this many meters away. Not physically
# measured — a documented, overridable guess so effects have *a* consistent
# unit out of the box.
DEFAULT_MAX_DISTANCE_METERS = 50.0

# ...and the nearest. A camera does not sit against its scene: a bridge- or
# pole-mounted unit has several metres of air before the closest thing it can
# see. Mapping relative depth onto ``[0, max]`` instead put the nearest scene
# point at **zero metres**, which is not merely imprecise — it is a
# singularity for any inverse-square quantity.
#
# Measured consequence before this existed: ``NightIrMode``, whose illuminator
# irradiance goes as 1/z^2, saw a near:far ratio of **27 778x** on the
# reference highway frame (0.3 m after clamping, against 50 m). With a 3 m
# near plane that becomes 278x, which is what a real scene spanning 3-50 m
# actually gives. Auto-exposure could hide the absolute level but never the
# gradient, and the gradient is the part that reads as night footage.
DEFAULT_MIN_DISTANCE_METERS = 3.0


def to_pseudo_metric(
    relative_depth: np.ndarray,
    depth_scale: float | None = None,
    near_distance: float | None = None,
) -> np.ndarray:
    """Map ``[0, 1]`` relative depth (0=nearest, 1=farthest) to pseudo-meters.

    Linear between a near and a far plane, so the nearest scene point lands at
    ``near_distance`` rather than at zero. Every depth-conditioned effect reads
    through this function, so the near plane matters to all of them — but
    quadratically to the illuminator in ``camera/night_ir.py`` and only
    linearly to the Koschmieder extinction in fog, rain and snow.

    Args:
        relative_depth: output of a :class:`DepthEstimator`.
        depth_scale: pseudo-meters at ``relative_depth == 1.0``. Defaults to
            :data:`DEFAULT_MAX_DISTANCE_METERS` when ``None``.
        near_distance: pseudo-meters at ``relative_depth == 0.0``. Defaults to
            :data:`DEFAULT_MIN_DISTANCE_METERS` when ``None``. Set it to 0 to
            recover the old behaviour, though nothing physical wants that.
    """
    far = DEFAULT_MAX_DISTANCE_METERS if depth_scale is None else depth_scale
    near = DEFAULT_MIN_DISTANCE_METERS if near_distance is None else near_distance
    near = min(near, far)
    return near + relative_depth * (far - near)
