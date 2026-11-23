"""Monocular depth estimation backends.

:class:`DepthEstimator` is the interface every depth-conditioned effect
(fog, rain, snow) programs against; :class:`DepthAnythingV2Estimator` is the
shipped default implementation. Swap in a different backend (a different
checkpoint, a precomputed-depth loader, ...) by implementing the same
interface and passing it to :class:`~surmentations.depth.provider.
SharedDepthProvider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from surmentations import mode
from surmentations.engines import check_backend, compiled

# PIL's code for bicubic resampling, which is what the processor's
# `resample` holds.
_BICUBIC = 3
# The sizes the depth network's engine takes, each side, in pixels: least,
# usual and most, multiples of its 14-pixel patch. A frame the processor
# sizes outside them, wider than about two to one, runs in torch.
_ENGINE_SIDE = (126, 518, 1036)
_ENGINE_USUAL_LONG = 700
_PATCH = 14

_NO_CUDA_MESSAGE = (
    "{cls} requires a CUDA GPU: depth-conditioned effects (fog/rain/snow) run "
    "monocular depth estimation on every call, which is far too slow on CPU to "
    "fit in a training-time DataLoader budget. No CUDA device was found "
    "(torch.cuda.is_available() is False). Options: (1) run on a machine with "
    "an NVIDIA GPU and a CUDA-enabled torch build, or (2) don't use "
    "depth-conditioned transforms (WaterDroplets/DirtSoil/SunFlare need no "
    "depth model and work on CPU)."
    " A device of 'cpu', asked for by name, is honoured: slow, but enough to render a "
    "few pictures where there is no GPU, which is what the documentation's gallery does."
)


class DepthEstimator(ABC):
    """Interface for a monocular depth backend.

    Implementations must return **relative, unitless** depth: same
    ``(H, W)`` shape as the input, float32, where larger values mean
    farther from the camera and smaller values mean nearer (0 = nearest
    point in the scene). There is no absolute scale — callers that need a
    physical distance use :func:`surmentations.depth.scale.to_pseudo_metric`
    to map this into pseudo-meters via an explicit or heuristic scale.
    """

    @abstractmethod
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate relative depth for an ``(H, W, C)`` uint8/float image."""

    def disparity(self, relative_depth: np.ndarray) -> np.ndarray | None:
        """The disparity :meth:`estimate`'s depth was made from, or ``None``.

        Disparity is the reciprocal of distance up to a scale and an offset
        nobody knows, larger for nearer. It is what the camera's geometry is
        read from (:mod:`surmentations.perception.geometry`): on level
        ground it is a linear function of position in the picture, which a
        depth that has been normalised and reshaped is not. A backend that
        can undo its own reshaping returns it here; the default is ``None``,
        and the geometry is then not estimated.
        """
        return None

    def signature(self) -> str | None:
        """What this estimator is, as far as that decides its output; ``None`` if it cannot say.

        Two estimators with the same signature make the same map of the
        same frame. A provider that keeps its readings on disk
        (:mod:`surmentations.depth.store`) files each under it, so that a
        map made by another checkpoint or another setting is never taken
        for this one's. The default is ``None``, and what such an
        estimator makes is not kept: a subclass that wants it kept names
        its class, its weights and every setting its output depends on.
        """
        return None


class DepthAnythingV2Estimator(DepthEstimator):
    """Depth Anything V2 (Small), the shipped default :class:`DepthEstimator`.

    CUDA-only by design (see module-level ``_NO_CUDA_MESSAGE``): raises a
    clear ``RuntimeError`` at construction time rather than silently falling
    back to a CPU inference path slow enough to blow the training-step time
    budget this library is built to respect. A ``device`` of ``"cpu"`` asked
    for by name is the exception: it is honoured, for rendering a handful of
    pictures, as the documentation's gallery is on a runner with no GPU.

    Note on sign convention *and* on the shape of the conversion: Depth
    Anything (V1 and V2) predicts a **disparity**-like quantity — larger raw
    output means *closer* to the camera, and disparity is proportional to
    ``1/depth``, not to ``-depth``. So the raw output is min-max normalised
    per image and then **reciprocated**,
    ``depth = 1 / (disparity + disparity_offset)``, before being rescaled to
    ``[0, 1]``.

    Using a linear flip (``1 - disparity``) instead — an easy mistake — badly
    compresses the far field: everything past the mid-ground collapses into a
    narrow band of depth values, so distance-driven effects like fog stop
    separating the middle distance from the horizon and read as a flat wash.
    ``disparity_offset`` sets how far "the farthest thing in frame" is
    relative to the nearest: the depth ratio across the image is
    ``(1 + offset) / offset``, so 0.03 spans roughly a 34x range, which suits
    outdoor/driving scenes. Raise it for enclosed scenes with little depth
    range, lower it for long open vistas.

    ``backend`` is ``"torch"``, or ``"tensorrt"`` for the network compiled
    with TensorRT (:mod:`surmentations.engines`): built the first time, kept
    on disk, and with a signature of its own. The engine takes sides from
    126 to 1036 pixels; a frame the processor sizes outside them, wider than
    about two to one, is read by torch.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "cuda",
        disparity_offset: float = 0.03,
        half_precision: bool | None = None,
        backend: str = "torch",
    ) -> None:
        # Imported here, like transformers below. Every transform's base class
        # imports this package for the depth provider, so a module-level
        # import would make the lens and camera effects — which need no depth
        # model at all — require torch just to be imported.
        """Load the model onto the GPU; see the class docstring for the arguments.

        Raises ``RuntimeError`` where torch has no CUDA device: the network
        is too slow on a CPU for an augmentation. torch and transformers are
        imported here and not at module level, so that the effects that need
        no depth do not need them either.
        """
        import torch

        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(_NO_CUDA_MESSAGE.format(cls=type(self).__name__))
        if disparity_offset <= 0:
            raise ValueError("disparity_offset must be > 0 (it bounds the farthest depth)")
        self.model_name = model_name
        self.disparity_offset = disparity_offset
        self.half_precision = half_precision
        self.backend = check_backend(backend)
        self._engines: dict[bool, Any] = {}

        # Imported lazily: transformers is a heavy import and only needed once
        # a depth-conditioned transform is actually constructed.
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = torch.device(device)
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = (
            AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device).eval()
        )
        # Skip the processor's overhead where doing so changes nothing.
        self._direct = False
        self._direct = self._direct_matches_processor()

    def signature(self) -> str | None:
        """The checkpoint, the offset, the precision asked for and the kind of device.

        And the backend, where it is not torch.
        """
        return (
            f"{type(self).__name__}|{self.model_name}|{self.disparity_offset!r}"
            f"|{self.half_precision}|{self.device.type}"
            + ("" if self.backend == "torch" else f"|{self.backend}")
        )

    def _disparity(self, pixel_values: Any, half: bool) -> Any:
        """The network's disparity for ``pixel_values``, on the backend asked for."""
        import torch

        least, _usual, most = _ENGINE_SIDE
        fits = all(
            least <= side <= most and side % _PATCH == 0 for side in pixel_values.shape[-2:]
        )
        if self.backend == "torch" or not fits:
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=half):
                return self._model(pixel_values=pixel_values).predicted_depth
        engine = self._engines.get(half)
        if engine is None:
            sides = (least, _ENGINE_SIDE[1], most, _PATCH)
            wide = (least, _ENGINE_USUAL_LONG, most, _PATCH)
            engine = self._engines[half] = compiled(
                _for_export(self._model, pixel_values),
                {"pixel_values": pixel_values},
                ["predicted_depth"],
                name="depth",
                identity={"estimator": type(self).__name__, "model": self.model_name},
                half=half,
                dynamic={"pixel_values": {2: sides, 3: wide}},
            )
        return engine(pixel_values=pixel_values)["predicted_depth"]

    def _model_input_size(self, height: int, width: int) -> tuple[int, int] | None:
        """The size the model's own processor would resize a frame to.

        ``None`` when the processor is set up in a way this does not
        reproduce, which sends the frame through the processor itself.
        """
        processor = self._processor
        size = getattr(processor, "size", None)
        if isinstance(size, dict):
            target = (size.get("height"), size.get("width"))
        else:
            target = (getattr(size, "height", None), getattr(size, "width", None))
        reproducible = (
            getattr(processor, "do_resize", False)
            and getattr(processor, "do_rescale", False)
            and getattr(processor, "do_normalize", False)
            and not getattr(processor, "do_pad", False)
            and getattr(processor, "resample", None) == _BICUBIC
        )
        target_height, target_width = target
        if not reproducible or not target_height or not target_width:
            return None
        multiple = int(getattr(processor, "ensure_multiple_of", 1) or 1)
        scale_h, scale_w = target_height / height, target_width / width
        if getattr(processor, "keep_aspect_ratio", False):
            # "Scale as little as possible", as the processor puts it.
            if abs(1.0 - scale_w) < abs(1.0 - scale_h):
                scale_h = scale_w
            else:
                scale_w = scale_h
        new_height = round(scale_h * height / multiple) * multiple
        new_width = round(scale_w * width / multiple) * multiple
        if new_height <= 0 or new_width <= 0:
            return None
        return new_height, new_width

    def _preprocess_directly(self, image: np.ndarray) -> Any:
        """The processor's resize, rescale and normalise, without the processor.

        The same three steps by the same arithmetic: torch's antialiased
        bicubic resize of the uint8 frame on the CPU, which is what the
        processor calls, then the rescale and the normalisation on the GPU.
        What is left out is the processor's own bookkeeping, which was most of
        its time: 3 to 5 ms a frame against 1 to 2.

        Returns ``None`` for a frame or a processor this does not reproduce.
        Whether it does reproduce this processor is not assumed:
        :meth:`_direct_matches_processor` compares the two once, when the
        estimator is built.
        """
        import torch

        size = self._model_input_size(image.shape[0], image.shape[1])
        if size is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            return None
        processor = self._processor
        pixels = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        if size != (image.shape[0], image.shape[1]):
            pixels = torch.nn.functional.interpolate(
                pixels, size=size, mode="bicubic", align_corners=False, antialias=True
            )
        pixels = pixels.to(self.device).float()
        # The processor folds the rescale into the mean and the deviation and
        # divides once. Written as rescale, subtract, divide, the tensor was
        # the same to six decimals and the depth map not the same bytes.
        per_unit = 1.0 / float(processor.rescale_factor)
        mean = torch.tensor(processor.image_mean, device=self.device) * per_unit
        std = torch.tensor(processor.image_std, device=self.device) * per_unit
        return (pixels - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

    def _direct_matches_processor(self) -> bool:
        """Whether the direct preprocessing gives this processor's tensor, bit
        for bit.

        Tried on two frames, one that is resized down and one up. Another
        version of transformers may resize by other code (the older
        processors go through PIL); then the two differ, and the processor is
        used for every frame as before.
        """
        import torch

        rng = np.random.default_rng(0)
        try:
            for shape in ((700, 900, 3), (160, 96, 3)):
                frame = rng.integers(0, 256, shape, dtype=np.uint8)
                direct = self._preprocess_directly(frame)
                if direct is None:
                    return False
                reference = self._processor(images=frame, return_tensors="pt")["pixel_values"]
                reference = reference.to(self.device)
                if direct.shape != reference.shape or not torch.equal(direct, reference):
                    return False
        except Exception:
            return False
        return True

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Relative depth for one ``(H, W, 3)`` frame, ``(H, W)`` float32 in ``[0, 1]``.

        A float frame is taken as ``[0, 1]`` and quantised to 8 bits first,
        which is what the network's preprocessing assumes. The output is the
        network's disparity resized to the frame, normalised, inverted as
        1 / (disparity + disparity_offset) and rescaled to [0, 1].
        """
        import torch

        height, width = image.shape[:2]
        if image.dtype != np.uint8:
            # The image processor rescales by 1/255 on the assumption that it
            # was given 8-bit data. A float image in [0, 1] comes out of that
            # at ~0.002 everywhere, and the network is asked for the depth of
            # a black frame.
            image = np.clip(np.rint(image.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
        with torch.inference_mode():
            pixel_values = self._preprocess_directly(image) if self._direct else None
            if pixel_values is None:
                inputs = self._processor(images=image, return_tensors="pt").to(self.device)
                pixel_values = inputs["pixel_values"]
            half = mode.fast() if self.half_precision is None else self.half_precision
            # (1, h', w'), larger = closer
            raw_disparity = self._disparity(pixel_values, half).float()

            resized = torch.nn.functional.interpolate(
                raw_disparity.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
            # reshape, not squeeze: squeeze would also drop a height or width
            # of one and return the wrong rank for a single-row image.
            disparity = resized.reshape(height, width).float().cpu().numpy()

        d_min, d_max = float(disparity.min()), float(disparity.max())
        if d_max - d_min < 1e-6:
            return np.zeros((height, width), dtype=np.float32)

        normalized_disparity = (disparity - d_min) / (d_max - d_min)

        # disparity ~ 1/depth, so depth ~ 1/disparity. The offset keeps the
        # horizon (disparity -> 0) finite and sets the near/far depth ratio.
        depth = 1.0 / (normalized_disparity + self.disparity_offset)
        depth -= depth.min()
        depth /= max(float(depth.max()), 1e-6)
        return depth.astype(np.float32)

    def disparity(self, relative_depth: np.ndarray) -> np.ndarray | None:
        """The network's disparity, normalised to ``[0, 1]``, back from :meth:`estimate`'s depth.

        :meth:`estimate` normalises the disparity to ``[0, 1]``, takes
        ``1 / (disparity + disparity_offset)`` and rescales that to
        ``[0, 1]``; this is the same three steps backwards, so it is exact
        to the rounding of a float.
        """
        offset = self.disparity_offset
        nearest, farthest = 1.0 / (1.0 + offset), 1.0 / offset
        reciprocal = relative_depth.astype(np.float64) * (farthest - nearest) + nearest
        return (1.0 / reciprocal - offset).astype(np.float32)


def _for_export(model: Any, example: Any) -> Any:
    """Depth Anything as a module an export keeps the frame's size free in.

    Two parts are given in other words, computing the same:

    * The head enlarges its features to the frame with
      ``int(patch_height * patch_size)``, which an export takes as a number:
      the engine made from it gave every frame the depth of the first one's
      size. Here the head takes the size from the frame's own tensor.
    * The patches are cut by a convolution 14 pixels wide with a stride of
      14, which TensorRT 11 fails to compile in half precision when the
      frame's size changes. Here it is the matrix product it is: the
      patches laid out in rows, times the kernel.

    The module is checked against the original on ``example`` before it is
    used, so that a release of ``transformers`` that changes either part
    fails here and not in a reading.
    """
    import torch

    head = model.head
    patches = model.backbone.embeddings.patch_embeddings
    kernel = patches.projection

    def cut(pixel_values: Any) -> Any:
        """``Dinov2PatchEmbeddings.forward`` as a reshape and a matrix product."""
        batch, channels, height, width = pixel_values.shape
        side = kernel.kernel_size[0]
        rows, columns = height // side, width // side
        laid = pixel_values.reshape(batch, channels, rows, side, columns, side)
        laid = laid.permute(0, 2, 4, 1, 3, 5).reshape(batch, rows * columns, channels * side * side)
        weight = kernel.weight.reshape(kernel.out_channels, channels * side * side)
        return torch.nn.functional.linear(laid, weight, kernel.bias)

    def head_forward(hidden_states: list[Any], size: Any) -> Any:
        """``DepthAnythingDepthEstimationHead.forward``, enlarging to ``size``."""
        predicted = head.conv1(hidden_states[head.head_in_index])
        predicted = torch.nn.functional.interpolate(
            predicted, size=size, mode="bilinear", align_corners=True
        )
        predicted = head.activation1(head.conv2(predicted))
        predicted = head.activation2(head.conv3(predicted)) * head.max_depth
        return predicted.squeeze(dim=1)

    class _Depth(torch.nn.Module):
        """The model, its head enlarging to the frame's own size."""

        def __init__(self) -> None:
            """Hold the model."""
            super().__init__()
            self.model = model

        def forward(self, pixel_values: Any) -> Any:
            """The disparity for ``pixel_values``."""
            size = (pixel_values.shape[-2], pixel_values.shape[-1])
            original_head, original_cut = head.forward, patches.forward
            head.forward = lambda hidden, _h, _w: head_forward(hidden, size)
            patches.forward = cut
            try:
                return self.model(pixel_values=pixel_values).predicted_depth
            finally:
                head.forward, patches.forward = original_head, original_cut

    wrapped = _Depth().eval()
    # Checked in single precision, where the two must agree to the last bits
    # but the order of a sum: in half, twelve layers of rounding in another
    # order part them by more than any check could tell from a mistake.
    # And in single precision for both: cuDNN may run the original's
    # convolution in TensorFloat-32, which torch allows it by default and
    # does not allow the matrix product, and the two would part by that.
    tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode(False), torch.no_grad():
            pixels = example.detach().clone().float()
            ours = wrapped(pixels)
            theirs = model(pixel_values=pixels).predicted_depth
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
    if float((ours - theirs).abs().max()) > 1e-4 * max(float(theirs.abs().max()), 1e-6):
        raise RuntimeError(
            "Depth Anything is not the model this export was written for; "
            'use backend="torch"'
        )
    return wrapped

