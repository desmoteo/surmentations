"""The estimators' networks compiled with TensorRT: their fast backend.

Each estimator that runs a network on the GPU takes ``backend="tensorrt"``
(:data:`BACKENDS`). Its network, or the part of it that sees the picture,
is exported to ONNX under ``torch.autocast`` the first time it is asked
for, built into a TensorRT engine and kept on disk (:func:`engine_cache`);
after that it is loaded in a second. The export under autocast carries
PyTorch's own choices of precision into the graph, half precision for the
matrix products and the convolutions and single precision for the
normalisations and the softmax, and TensorRT 10 and later run a graph's
types as they are given. So the engine computes what the network computes
in fast mode, faster, and up to the rounding of half precision.

An estimator on this backend has a signature of its own: a reading made
with it is kept apart from one made with PyTorch
(:meth:`~surmentations.depth.estimator.DepthEstimator.signature`).

An engine is made for one GPU and one version of TensorRT. Everything that
changes it is in its name on disk: the network, its input shapes, the
precision, the GPU and its compute capability, and the versions of
TensorRT, torch and transformers. Building the three the shipped
estimators use took a minute, on a laptop GPU.

TensorRT is NVIDIA's, under NVIDIA's licence, and is an optional extra
(``pip install "surmentations[tensorrt]"``); nothing in the library needs it
unless this backend is asked for.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: The backends an estimator runs its network on.
BACKENDS = ("torch", "tensorrt")

# Where engines are kept, unless the environment says otherwise.
_CACHE_VARIABLE = "SURMENTATIONS_ENGINE_CACHE"
# The scratch memory TensorRT may use while it builds, in bytes.
_WORKSPACE = 2 << 30
# How engines are exported and built here: part of every engine's name, so
# that a change to it builds them again.
_EXPORT_VERSION = 2

_INSTALL_HINT = 'backend="tensorrt" needs TensorRT and ONNX: pip install "surmentations[tensorrt]"'


def engine_cache() -> Path:
    """Where built engines are kept.

    ``$SURMENTATIONS_ENGINE_CACHE``, else ``~/.cache/surmentations/engines``.
    """
    root = os.environ.get(_CACHE_VARIABLE)
    return Path(root) if root else Path.home() / ".cache" / "surmentations" / "engines"


def check_backend(backend: str) -> str:
    """``backend`` if it is one of :data:`BACKENDS` that can run here; raises otherwise.

    Raises:
        ValueError: for a backend that is not one of :data:`BACKENDS`.
        ImportError: for ``"tensorrt"`` without TensorRT or ONNX installed.
        RuntimeError: for ``"tensorrt"`` without a CUDA device.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "tensorrt":
        try:
            import onnx  # noqa: F401
            import tensorrt  # noqa: F401
        except ImportError as error:
            raise ImportError(_INSTALL_HINT) from error
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError('backend="tensorrt" needs a CUDA device')
    return backend


def _dtypes() -> dict[Any, Any]:
    """TensorRT's data types and torch's, one for one."""
    import tensorrt as trt
    import torch

    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }


class TensorRTEngine:
    """A built engine, called with torch tensors on the GPU and answering with them.

    Each call runs on a stream of the engine's own, after what the caller's
    stream has queued and before anything the caller queues next, so the
    tensors it returns are ready for the caller's stream to use.
    """

    def __init__(self, plan: Path) -> None:
        """Load the engine from ``plan``, the file :func:`compiled` built."""
        import tensorrt as trt
        import torch

        self.path = plan
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(plan.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"TensorRT could not load the engine in {plan}")
        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream()
        self._types = _dtypes()
        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        mode = trt.TensorIOMode.INPUT
        self.inputs = [n for n in names if self._engine.get_tensor_mode(n) == mode]
        self.outputs = [n for n in names if self._engine.get_tensor_mode(n) != mode]

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        """Run the engine on ``inputs``, by name; its outputs, by name.

        Each input is made contiguous and cast to the type the engine takes
        it in; the outputs are in the types the engine gives them.
        """
        import torch

        missing = set(self.inputs) - set(inputs)
        if missing:
            raise ValueError(f"the engine needs inputs {sorted(missing)}")
        caller = torch.cuda.current_stream()
        held = []
        for name in self.inputs:
            tensor = inputs[name]
            wanted = self._types[self._engine.get_tensor_dtype(name)]
            tensor = tensor.to(dtype=wanted).contiguous()
            held.append(tensor)
            self._context.set_input_shape(name, tuple(tensor.shape))
            self._context.set_tensor_address(name, tensor.data_ptr())
        outputs = {}
        for name in self.outputs:
            shape = tuple(self._context.get_tensor_shape(name))
            wanted = self._types[self._engine.get_tensor_dtype(name)]
            outputs[name] = torch.empty(shape, dtype=wanted, device=held[0].device)
            self._context.set_tensor_address(name, outputs[name].data_ptr())
        self._stream.wait_stream(caller)
        if not self._context.execute_async_v3(self._stream.cuda_stream):
            raise RuntimeError(f"TensorRT failed to run the engine in {self.path}")
        caller.wait_stream(self._stream)
        # The inputs and outputs must live until the engine is done with them.
        for tensor in (*held, *outputs.values()):
            tensor.record_stream(self._stream)
        return outputs


def _key(name: str, identity: Mapping[str, Any]) -> str:
    """The name an engine is kept under: ``name`` and a digest of all that changes it."""
    import tensorrt as trt
    import torch

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    described = json.dumps(
        {
            **identity,
            "export": _EXPORT_VERSION,
            "tensorrt": trt.__version__,
            "torch": torch.__version__,
            "transformers": transformers_version,
            "gpu": device.name,
            "capability": [device.major, device.minor],
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(described.encode()).hexdigest()[:16]
    return f"{name}-{digest}"


def compiled(
    module: Any,
    example: Mapping[str, Any],
    outputs: Sequence[str],
    *,
    name: str,
    identity: Mapping[str, Any],
    half: bool = True,
    dynamic: Mapping[str, Mapping[int, tuple[int, ...]]] | None = None,
    own_autocast: bool = False,
) -> TensorRTEngine:
    """``module`` as a TensorRT engine, built the first time and loaded from disk after.

    Args:
        module: the ``torch.nn.Module`` to compile, on the GPU, in eval
            mode, called with keyword arguments and returning a tensor or a
            tuple of them.
        example: an input of each name, on the GPU, of a shape it takes.
        outputs: the names to give what the module returns, in order.
        name: a short name for the engine's file.
        identity: what else decides what the module computes: the
            checkpoint, its settings. It is part of the key on disk.
        half: export under ``torch.autocast`` in float16, as fast mode runs
            the network; otherwise in single precision.
        dynamic: for each input with dimensions that change, each such
            dimension's ``(least, usual, most)``, and a fourth number where
            it only takes multiples of one: a vision transformer's side, in
            its patches. Others are fixed at the example's shape.
        own_autocast: the module applies autocast itself, to the parts it
            chooses, and is exported without it round the whole. ``half``
            still names the engine.
    """
    dynamic = dynamic or {}
    # Each axis's least, usual and most: the example's size where it does
    # not change. The key on disk takes these and not the example's size, so
    # the frame an engine is first built on does not name it.
    shapes = {
        key: [
            list(dynamic.get(key, {}).get(axis, (size,) * 3)[:3])
            for axis, size in enumerate(value.shape)
        ]
        for key, value in example.items()
    }
    stem = _key(name, {**identity, "half": half, "shapes": shapes, "outputs": list(outputs)})
    plan = engine_cache() / f"{stem}.plan"
    if not plan.exists():
        plan.parent.mkdir(parents=True, exist_ok=True)
        # One process builds and the rest wait for it: each DataLoader worker
        # makes its own estimators, and sixteen builds at once would each
        # take the minutes one does, and the GPU's memory between them.
        with _held(plan.with_suffix(".lock")):
            if not plan.exists():
                _make(module, example, outputs, plan, stem, half, own_autocast, dynamic, shapes)
    return TensorRTEngine(plan)


@contextlib.contextmanager
def _held(lock: Path) -> Any:
    """Hold ``lock`` while the block runs, against other processes; on Windows, nothing."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(lock, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _make(
    module: Any,
    example: Mapping[str, Any],
    outputs: Sequence[str],
    plan: Path,
    stem: str,
    half: bool,
    own_autocast: bool,
    dynamic: Mapping[str, Mapping[int, tuple[int, ...]]],
    shapes: Mapping[str, list[list[int]]],
) -> None:
    """Export ``module`` and build it into ``plan``."""
    with tempfile.TemporaryDirectory(dir=plan.parent) as work:
        graph = Path(work) / f"{stem}.onnx"
        _export(module, example, outputs, graph, half and not own_autocast, dynamic)
        serialised = _build(graph, shapes, tf32=half)
        # Written beside, then moved in place: a run that stops half way
        # leaves no half an engine behind for the next one to load.
        partial = Path(work) / f"{stem}.plan"
        partial.write_bytes(serialised)
        os.replace(partial, plan)


def _export(
    module: Any,
    example: Mapping[str, Any],
    outputs: Sequence[str],
    graph: Path,
    half: bool,
    dynamic: Mapping[str, Mapping[int, tuple[int, ...]]],
) -> None:
    """Export ``module`` to ONNX at ``graph``, under autocast when ``half``."""
    import torch

    shapes = None
    if dynamic:
        shapes = {
            key: {axis: _dim(f"{key}_{axis}", bounds) for axis, bounds in dynamic[key].items()}
            if key in dynamic
            else None
            for key in example
        }
    precision = torch.autocast("cuda", dtype=torch.float16) if half else contextlib.nullcontext()
    # Out of inference mode, which the estimators run in and an export
    # cannot trace through, with copies made out of it. An input with sides
    # that change is traced at their usual sizes and not at the example's:
    # two sides the same size in the example, a square frame, are taken by
    # the export to be one size always, and the engine refuses any other.
    with torch.inference_mode(False), torch.no_grad(), precision:
        example = {key: _usual(value, dynamic.get(key, {})) for key, value in example.items()}
        torch.onnx.export(
            module,
            (),
            str(graph),
            kwargs=dict(example),
            input_names=list(example),
            output_names=list(outputs),
            dynamic_shapes=shapes,
            dynamo=True,
            external_data=True,
            verbose=False,
        )


def _usual(tensor: Any, sides: Mapping[int, tuple[int, ...]]) -> Any:
    """A copy of ``tensor``, out of inference mode, its changing sides at their usual sizes.

    The values are drawn afresh where the shape changes: an export follows
    the shapes, and what is in the tensor does not matter to it.
    """
    import torch

    tensor = tensor.detach()
    if not sides:
        return tensor.clone()
    shape = list(tensor.shape)
    for axis, bounds in sides.items():
        shape[axis] = int(bounds[1])
    if tensor.is_floating_point():
        return torch.rand(shape, dtype=tensor.dtype, device=tensor.device)
    return torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)


def _dim(name: str, bounds: tuple[int, ...]) -> Any:
    """A dimension for ``torch.export``: ``(least, usual, most)``, or with a fourth, a step.

    With a step, a multiple of it: an export that sees a side divided by a
    patch would otherwise fix the side at the example's.
    """
    import torch

    least, _usual, most = bounds[:3]
    if len(bounds) > 3 and bounds[3] > 1:
        step = int(bounds[3])
        return step * torch.export.Dim(name, min=-(-least // step), max=most // step)
    return torch.export.Dim(name, min=least, max=most)


def _build(graph: Path, shapes: Mapping[str, list[list[int]]], tf32: bool) -> bytes:
    """Build the ONNX at ``graph`` into a serialised engine, with a profile for each input.

    Without ``tf32`` the single-precision products are made in single
    precision, as exact mode asks, and not in TensorFloat-32, which
    TensorRT would otherwise use for them.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(graph)):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not read the exported graph: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _WORKSPACE)
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        bounds = shapes.get(tensor.name)
        if bounds is None:
            continue
        least, usual, most = (tuple(b[i] for b in bounds) for i in range(3))
        profile.set_shape(tensor.name, least, usual, most)
    config.add_optimization_profile(profile)
    serialised = builder.build_serialized_network(network, config)
    if serialised is None:
        raise RuntimeError(f"TensorRT could not build an engine from {graph.name}")
    return bytes(serialised)


def returning(model: Any, field: str) -> Any:
    """``model`` as a module that takes ``pixel_values`` and returns its output's ``field``.

    For a ``transformers`` model whose output is a dataclass of tensors,
    which an export cannot hand on whole.
    """
    import torch

    class _Field(torch.nn.Module):
        """The model, returning one field of its output."""

        def __init__(self) -> None:
            """Hold the model."""
            super().__init__()
            self.model = model

        def forward(self, pixel_values: Any) -> Any:
            """The field of the model's output for ``pixel_values``."""
            return getattr(self.model(pixel_values=pixel_values), field)

    return _Field().eval()
