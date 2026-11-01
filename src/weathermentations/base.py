"""Shared conventions for all weathermentations transforms.

Every effect in this library is an appearance-only change (fog, rain, snow,
lens dirt, water droplets, sun flare all leave scene geometry untouched), so
every transform subclasses Albumentations' ``ImageOnlyTransform``: masks,
bounding boxes and keypoints pass through ``Compose`` unmodified with no
extra code here.

Conventions used throughout this library (matching Albumentations 2.x):

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

from typing import Any, ClassVar

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

from weathermentations.depth.provider import carry_depth
from weathermentations.diagnostics import instrument

RangeF = tuple[float, float]
RangeI = tuple[int, int]


class WeatherTransform(ImageOnlyTransform):
    """Common base for every physically-grounded effect in this library.

    Subclasses implement ``apply`` and (when a parameter needs per-call
    randomization or depends on image shape) ``get_params_dependent_on_data``
    per the standard Albumentations ``ImageOnlyTransform`` contract.

    Every subclass's ``get_params_dependent_on_data`` is wrapped automatically
    so that, when :func:`~weathermentations.diagnostics.recording` is active,
    the parameters it sampled are captured. Subclasses do not opt in and need
    know nothing about it; the wrapper is a single flag check when recording
    is off. See ``diagnostics.py`` for why the draws are worth recording
    separately from the rendered image.

    Two more things are handled here once rather than in sixteen places:

    * **Input is checked before anything is sampled.** Every effect is a
      statement about RGB radiance, so the image must be ``(H, W, 3)`` and
      ``uint8`` or floating point in ``[0, 1]``. Anything else used to get as
      far as the renderer and fail there — or not fail: a greyscale frame came
      back from several transforms with a *different shape*, and a 16-bit one
      came back crushed to white.
    * **The scene's depth follows the image through the pipeline.** A
      transform returns a new array, which a depth provider has never seen;
      the output is reported back so the next depth-conditioned transform
      reuses the depth already estimated for this scene instead of paying
      for another estimate. See ``depth/provider.py``.
    """

    #: Whether the output shows the same geometry as the input, pixel for
    #: pixel, so depth estimated for one is valid for the other. True of every
    #: appearance-only effect; a transform that moves pixels sets it False.
    preserves_scene_geometry: ClassVar[bool] = True

    #: Whether the effect is defined only for RGB. True of everything that
    #: models light; a pure resampling of pixel positions is indifferent to
    #: what the pixels hold and sets it False.
    requires_rgb: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Instrument each subclass's own ``get_params_dependent_on_data`` for diagnostics.

        Done once per class, when it is defined, so that every transform's
        draws can be recorded (:mod:`weathermentations.diagnostics`) without
        each one calling the recorder itself.
        """
        super().__init_subclass__(**kwargs)
        method = cls.__dict__.get("get_params_dependent_on_data")
        if method is not None and not getattr(
            method, "__weathermentations_instrumented__", False
        ):
            cls.get_params_dependent_on_data = instrument(method)

    def update_transform_params(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        # The first hook Albumentations calls once a transform has decided to
        # fire, and the last one before any parameter is sampled.
        """Check the input frame before anything is drawn from it.

        The first hook Albumentations calls once a transform fires: a
        transform that needs RGB refuses anything else here, with its own
        name in the message, rather than failing somewhere inside.
        """
        if self.requires_rgb:
            if "image" in data:
                check_image(data["image"], cls_name=type(self).__name__)
            elif "images" in data and len(data["images"]):
                check_image(data["images"][0], cls_name=type(self).__name__)
        return super().update_transform_params(params, data)

    def apply_with_params(
        self, params: dict[str, Any], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Apply, then tell the depth providers that the output shows the input's scene.

        Only for transforms that keep the scene's geometry, so that a
        depth-conditioned transform later in the pipeline reuses the depth
        estimated for the frame this one was given.
        """
        result = super().apply_with_params(params, *args, **kwargs)
        # Compared here rather than inside `apply`, because these are the two
        # arrays the pipeline itself holds: the one the depth provider was
        # asked about, and the one the next transform will ask about.
        # Albumentations may hand `apply` a contiguous copy of the first and
        # wrap what it returns in another.
        if self.preserves_scene_geometry:
            for key in ("image", "images"):
                source, derived = kwargs.get(key), result.get(key)
                if isinstance(source, np.ndarray) and isinstance(derived, np.ndarray):
                    carry_depth(source, derived)
        return result


def scene_of(data: dict[str, Any]) -> np.ndarray:
    """What a transform was called on: the ``(H, W, 3)`` image, or a clip.

    Albumentations passes a clip as ``images=``, an ``(N, H, W, 3)`` array,
    and applies one set of parameters to every frame of it. Transforms that
    sample their parameters *from* the picture — a depth map, an airlight
    colour — used to read ``data["image"]`` and so failed with a ``KeyError``
    on a clip, while the ones that did not sailed through.
    """
    if "image" in data:
        scene: np.ndarray = data["image"]
    elif "images" in data:
        scene = data["images"]
    else:
        raise KeyError("expected the data to contain `image` or `images`")
    return scene


def first_frame(scene: np.ndarray) -> np.ndarray:
    """One representative ``(H, W, 3)`` frame of :func:`scene_of`'s result."""
    return scene[0] if scene.ndim == 4 else scene


def check_image(image: Any, *, cls_name: str) -> None:
    """Reject input the effects cannot be meaningfully applied to.

    Raises:
        TypeError: if ``image`` is not a numpy array of a supported dtype.
        ValueError: if it is not an ``(H, W, 3)`` colour image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{cls_name} expects a numpy array, got {type(image).__name__}.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"{cls_name} expects an RGB image of shape (H, W, 3), got shape {image.shape}. "
            "The effects model colour radiance (wavelength-dependent scattering, "
            "illuminant casts, chroma subsampling), so there is no greyscale form "
            "of them; convert with cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) first."
        )
    if image.dtype != np.uint8 and not np.issubdtype(image.dtype, np.floating):
        raise TypeError(
            f"{cls_name} supports uint8 images and floating-point images in [0, 1], "
            f"got dtype {image.dtype}. Convert first, e.g. "
            "(image / 65535).astype(np.float32) for 16-bit input."
        )


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
