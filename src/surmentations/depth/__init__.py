"""Monocular depth: the estimator, its scale in metres, and the provider that shares it.

One :class:`SharedDepthProvider` is built per pipeline and given to every
depth-conditioned transform, so that each frame's depth is estimated once.
"""

from surmentations.depth.estimator import DepthAnythingV2Estimator, DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.depth.scale import to_pseudo_metric

__all__ = [
    "DepthAnythingV2Estimator",
    "DepthEstimator",
    "SharedDepthProvider",
    "to_pseudo_metric",
]
