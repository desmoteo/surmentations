"""Shared, per-call depth caching for depth-conditioned transforms.

See ``plan.md`` §2.3 for the caching-strategy decision this implements:
depth is *not* cached across epochs (no stable per-image id is available
from a raw Albumentations call). Instead, one ``SharedDepthProvider``
instance is constructed by the user and passed into every depth-conditioned
transform's constructor, so depth is computed once per distinct image array
and reused by any other depth-conditioned transform chained after it in the
same ``Compose()`` call.
"""

from __future__ import annotations

import numpy as np

from weathermentations.depth.estimator import DepthEstimator


class SharedDepthProvider:
    """Wraps a :class:`DepthEstimator` with a ``maxsize=1`` per-call cache.

    The cache key is ``(id(image), image.shape)``. If an earlier transform
    in the pipeline happens to return a new array object (e.g. because it
    copies on write), this cache simply misses and depth is recomputed — a
    graceful performance degradation, not a correctness bug, per the
    explicitly accepted tradeoff in the caching-strategy decision.
    """

    def __init__(self, estimator: DepthEstimator) -> None:
        self._estimator = estimator
        self._cache_key: tuple[int, tuple[int, ...]] | None = None
        self._cache_value: np.ndarray | None = None

    def get_depth(self, image: np.ndarray) -> np.ndarray:
        key = (id(image), image.shape)
        if key != self._cache_key or self._cache_value is None:
            self._cache_value = self._estimator.estimate(image)
            self._cache_key = key
        return self._cache_value
