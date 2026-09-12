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


def to_pseudo_metric(relative_depth: np.ndarray, depth_scale: float | None = None) -> np.ndarray:
    """Map ``[0, 1]`` relative depth (0=nearest, 1=farthest) to pseudo-meters.

    Args:
        relative_depth: output of a :class:`DepthEstimator`.
        depth_scale: pseudo-meters at ``relative_depth == 1.0``. Defaults to
            :data:`DEFAULT_MAX_DISTANCE_METERS` when ``None``.
    """
    scale = DEFAULT_MAX_DISTANCE_METERS if depth_scale is None else depth_scale
    return relative_depth * scale
