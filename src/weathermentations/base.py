"""Shared conventions for all weathermentations transforms.

Every effect in this library is an appearance-only change (fog, rain, snow,
lens dirt, water droplets, sun flare all leave scene geometry untouched), so
every transform subclasses Albumentations' ``ImageOnlyTransform``: masks,
bounding boxes and keypoints pass through ``Compose`` unmodified with no
extra code here.

Conventions used throughout this library (matching Albumentations >=1.4/2.x):

- Constructor parameters that describe a physical quantity are ``(min, max)``
  ranges, validated via a pydantic ``InitSchema`` nested class, the same
  pattern used by built-in transforms like ``RandomRain``/``RandomFog``.
  Declare one inline, the same way upstream Albumentations transforms do
  (a wrapper function returning ``Annotated[...]`` is *not* usable as a
  type annotation itself — mypy rejects a bare function call in annotation
  position)::

      class InitSchema(BaseTransformInitSchema):
          my_range: Annotated[
              tuple[float, float],
              AfterValidator(nondecreasing),
              AfterValidator(check_range_bounds(0.0, 1.0)),
          ]

  ``nondecreasing``/``check_range_bounds`` come from
  ``albumentations.core.pydantic``; ``AfterValidator`` from ``pydantic``.
- ``get_transform_init_args_names`` does **not** need to be overridden:
  ``BasicTransform`` auto-derives it by introspecting ``__init__`` across the
  MRO, which is sufficient for every transform in this library.
- Randomness must go through the transform's own RNG so pipeline-level
  seeding (``A.Compose(..., seed=...)``) reaches these transforms exactly
  like built-in ones: use ``self.py_random.uniform(...)`` etc. for scalar
  draws and ``self.random_generator.integers/.uniform/...`` (a
  ``numpy.random.Generator``) for array draws. Never call ``np.random``
  or ``random`` directly.
"""

from __future__ import annotations

from typing import Any

from albumentations.core.transforms_interface import ImageOnlyTransform

RangeF = tuple[float, float]
RangeI = tuple[int, int]


class WeatherTransform(ImageOnlyTransform):
    """Common base for every physically-grounded effect in this library.

    Subclasses implement ``apply`` and (when a parameter needs per-call
    randomization or depends on image shape) ``get_params_dependent_on_data``
    per the standard Albumentations ``ImageOnlyTransform`` contract.
    """


def require_depth_provider(depth_provider: Any, *, cls_name: str) -> Any:
    """Guard used by every depth-conditioned transform's ``__init__``.

    Albumentations 2.x validates constructor arguments per-class through a
    pydantic ``InitSchema`` wired up by its metaclass, which does not compose
    cleanly through an intermediate mixin base class. So depth-conditioned
    transforms (fog, rain, snow) each declare ``depth_provider`` directly as
    a field on their own ``InitSchema`` and call this helper at the start of
    ``__init__`` instead of inheriting shared ``__init__`` logic.

    A single :class:`~weathermentations.depth.provider.SharedDepthProvider`
    instance must be constructed once and passed to every depth-conditioned
    transform in a pipeline, so depth is shared rather than recomputed per
    transform (see ``depth/provider.py`` for the caching contract).
    """
    if depth_provider is None:
        raise ValueError(
            f"{cls_name} requires a `depth_provider` "
            "(weathermentations.depth.provider.SharedDepthProvider instance). "
            "Construct one SharedDepthProvider and pass the same instance to "
            "every depth-conditioned transform in your pipeline so depth is "
            "shared rather than recomputed per transform."
        )
    return depth_provider
