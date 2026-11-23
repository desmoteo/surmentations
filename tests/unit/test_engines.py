"""The TensorRT backend: an engine computes what its module does, is kept and takes its sizes.

Skipped without TensorRT, ONNX or a CUDA device, which is every CI runner.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from surmentations import engines


def test_an_unknown_backend_is_refused() -> None:
    with pytest.raises(ValueError, match="backend"):
        engines.check_backend("openvino")
    assert engines.check_backend("torch") == "torch"


torch = pytest.importorskip("torch")
pytest.importorskip("tensorrt")
pytest.importorskip("onnx")
if not torch.cuda.is_available():
    pytest.skip("the TensorRT backend needs a CUDA device", allow_module_level=True)


class _Small(torch.nn.Module):
    """A convolution, a normalisation and an enlargement back to the input's size."""

    def __init__(self) -> None:
        """Fixed weights, so that two builds compile the same thing."""
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.norm = torch.nn.GroupNorm(2, 8)
        self.out = torch.nn.Conv2d(8, 1, 1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Features at half the size, enlarged to the input's own."""
        size = (pixel_values.shape[-2], pixel_values.shape[-1])
        features = torch.relu(self.norm(self.conv(pixel_values)))
        features = torch.nn.functional.avg_pool2d(features, 2)
        features = torch.nn.functional.interpolate(features, size=size, mode="bilinear")
        return self.out(features)


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty engine cache of the test's own."""
    monkeypatch.setenv("SURMENTATIONS_ENGINE_CACHE", str(tmp_path))
    return tmp_path


def _compiled(half: bool, dynamic: dict | None = None) -> engines.TensorRTEngine:
    """The small module compiled, from an input 32 pixels square."""
    module = _Small().cuda().eval()
    example = {"pixel_values": torch.rand(1, 3, 32, 32, device="cuda")}
    return engines.compiled(
        module, example, ["out"], name="small", identity={"test": 1}, half=half, dynamic=dynamic
    )


@pytest.mark.parametrize("half", [False, True])
def test_an_engine_computes_what_its_module_does(cache: Path, half: bool) -> None:
    engine = _compiled(half)
    pixels = torch.rand(1, 3, 32, 32, device="cuda")
    module = _Small().cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=half):
        expected = module(pixels).float()
    got = engine(pixel_values=pixels)["out"].float()
    tolerance = 2e-2 if half else 1e-3
    assert float((got - expected).abs().max()) < tolerance


def test_an_engine_is_built_once_and_loaded_after(cache: Path) -> None:
    first = _compiled(half=True)
    built = first.path.stat().st_mtime_ns
    second = _compiled(half=True)
    assert second.path == first.path and second.path.stat().st_mtime_ns == built
    assert len(list(cache.glob("*.plan"))) == 1
    assert _compiled(half=False).path != first.path, "the precision is part of the name"


def test_a_dynamic_side_in_steps_takes_every_size_in_its_bounds(cache: Path) -> None:
    """A side that changes, in multiples of eight: the engine gives each size
    its own output, not the size it was exported at."""
    engine = _compiled(
        half=False, dynamic={"pixel_values": {2: (16, 32, 64, 8), 3: (16, 32, 64, 8)}}
    )
    module = _Small().cuda().eval()
    for rows, columns in ((16, 40), (48, 24), (64, 64)):
        pixels = torch.rand(1, 3, rows, columns, device="cuda")
        got = engine(pixel_values=pixels)["out"]
        assert tuple(got.shape) == (1, 1, rows, columns)
        with torch.no_grad():
            assert float((got - module(pixels)).abs().max()) < 1e-3


def test_a_square_first_input_does_not_tie_the_sides_together(cache: Path) -> None:
    """Built from a square input, an engine with two sides that change still
    takes them apart: an export traced at the input's own sizes took the
    two for one, and the engine refused every frame that was not square."""
    module = _Small().cuda().eval()
    square = {"pixel_values": torch.rand(1, 3, 32, 32, device="cuda")}
    engine = engines.compiled(
        module,
        square,
        ["out"],
        name="small",
        identity={"test": 2},
        half=False,
        dynamic={"pixel_values": {2: (16, 32, 64, 8), 3: (16, 48, 64, 8)}},
    )
    pixels = torch.rand(1, 3, 24, 56, device="cuda")
    assert tuple(engine(pixel_values=pixels)["out"].shape) == (1, 1, 24, 56)
