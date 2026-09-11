"""Monocular depth estimation backends.

:class:`DepthEstimator` is the interface every depth-conditioned effect
(fog, rain, snow) programs against; :class:`DepthAnythingV2Estimator` is the
shipped default implementation. Swap in a different backend (a different
checkpoint, a precomputed-depth loader, ...) by implementing the same
interface and passing it to :class:`~weathermentations.depth.provider.
SharedDepthProvider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

_NO_CUDA_MESSAGE = (
    "{cls} requires a CUDA GPU: depth-conditioned effects (fog/rain/snow) run "
    "monocular depth estimation on every call, which is far too slow on CPU to "
    "fit in a training-time DataLoader budget. No CUDA device was found "
    "(torch.cuda.is_available() is False). Options: (1) run on a machine with "
    "an NVIDIA GPU and a CUDA-enabled torch build, or (2) don't use "
    "depth-conditioned transforms (WaterDroplets/DirtSoil/SunFlare need no "
    "depth model and work on CPU)."
)


class DepthEstimator(ABC):
    """Interface for a monocular depth backend.

    Implementations must return **relative, unitless** depth: same
    ``(H, W)`` shape as the input, float32, where larger values mean
    farther from the camera and smaller values mean nearer (0 = nearest
    point in the scene). There is no absolute scale — callers that need a
    physical distance use :func:`weathermentations.depth.scale.to_pseudo_metric`
    to map this into pseudo-meters via an explicit or heuristic scale.
    """

    @abstractmethod
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate relative depth for an ``(H, W, C)`` uint8/float image."""


class DepthAnythingV2Estimator(DepthEstimator):
    """Depth Anything V2 (Small), the shipped default :class:`DepthEstimator`.

    CUDA-only by design (see module-level ``_NO_CUDA_MESSAGE``): raises a
    clear ``RuntimeError`` at construction time rather than silently falling
    back to a CPU inference path slow enough to blow the training-step time
    budget this library is built to respect.

    Note on sign convention: Depth Anything (V1 and V2) predicts a
    disparity-like quantity — larger raw output means *closer* to the
    camera, the opposite of this library's ``DepthEstimator`` contract. This
    class min-max normalizes the raw output per-image to ``[0, 1]`` and
    inverts it (``depth = 1 - normalized_disparity``) so ``estimate()``
    returns depth with larger = farther, as promised by the base class. This
    conversion is the first thing to re-check if a depth-conditioned effect
    ever looks inverted (e.g. fog denser close up instead of in the
    distance) against a real reference photo.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "cuda",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(_NO_CUDA_MESSAGE.format(cls=type(self).__name__))

        # Imported lazily: transformers is a heavy import and only needed once
        # a depth-conditioned transform is actually constructed.
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = torch.device(device)
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = (
            AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device).eval()
        )

    @torch.inference_mode()
    def estimate(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        raw_disparity = self._model(**inputs).predicted_depth  # (1, h', w'), larger = closer

        resized = torch.nn.functional.interpolate(
            raw_disparity.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        disparity = resized.squeeze().float().cpu().numpy()

        d_min, d_max = float(disparity.min()), float(disparity.max())
        if d_max - d_min < 1e-6:
            return np.zeros((height, width), dtype=np.float32)

        normalized_disparity = (disparity - d_min) / (d_max - d_min)
        depth = 1.0 - normalized_disparity  # invert: larger = farther, per DepthEstimator contract
        return depth.astype(np.float32)
