"""The composite in `docs/pipelines.md` must keep the guarantees it claims.

The document's whole argument is that exclusivity is *structural* — rain and
snow cannot co-occur because they are alternatives of one `OneOf`, and a sun
flare cannot land on an IR night frame because the two lighting branches are
alternatives of another. That is cheap to break: moving `SunFlare` from the
day branch into the shared tail would silently reintroduce flare-at-night and
nothing else in this suite would notice.

So the recipe is executed exactly as the document prints it, and the
properties are asserted rather than described.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.diagnostics import recording
from weathermentations.presets.pipeline import stage_of

DOC = Path(__file__).resolve().parents[2] / "docs" / "pipelines.md"

WEATHER = {"RandomRain", "RandomSnow", "RandomFog"}
DIRECT_SUN = {"SunFlare", "CastShadows"}
SAMPLES = 600


class _GroundPlane(DepthEstimator):
    def estimate(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        depth = np.clip(1.5 * 300.0 / np.maximum(rows - height / 2.0, 1e-3), 0.0, 50.0)
        return np.broadcast_to(depth / 50.0, (height, width)).astype(np.float32)


def _recipe() -> dict[str, Any]:
    """The document's first python block, executed as written."""
    blocks = re.findall(r"```python\n(.*?)```", DOC.read_text(), re.S)
    assert blocks, "docs/pipelines.md has no python block"
    namespace: dict[str, Any] = {}
    exec(blocks[0], namespace)  # noqa: S102 - the point is to run the documented code
    return namespace


@pytest.fixture(scope="module")
def drawn() -> list[set[str]]:
    """Which effects fired, per image, over many samples."""
    transform = _recipe()["surveillance"](SharedDepthProvider(_GroundPlane()))
    transform.set_random_seed(5)
    image = np.full((96, 96, 3), 120, dtype=np.uint8)
    samples = []
    for _ in range(SAMPLES):
        with recording() as log:
            transform(image=image)
        samples.append({record["t"] for record in log})
    return samples


def test_at_most_one_weather_state_per_image(drawn: list[set[str]]) -> None:
    """Rain, snow and fog are states of one atmosphere, not independent events."""
    both = [s for s in drawn if len(s & WEATHER) > 1]
    assert not both, f"{len(both)} images got several weather states at once, e.g. {both[0]}"


def test_night_infrared_never_coexists_with_direct_sunlight(drawn: list[set[str]]) -> None:
    """A frame the camera is lighting with its own IR illuminator has no sun in
    it, so it can carry neither a flare nor a hard cast shadow."""
    clashes = [s for s in drawn if "NightIrMode" in s and s & DIRECT_SUN]
    assert not clashes, f"{len(clashes)} night frames also got sunlight, e.g. {clashes[0]}"


def test_droplets_on_the_glass_require_wet_weather(drawn: list[set[str]]) -> None:
    """Rain or condensation puts water on the front element; clear weather does
    not, and applying it anyway teaches that lenses are randomly wet."""
    dry = [s for s in drawn if "WaterDroplets" in s and not s & WEATHER]
    assert not dry, f"{len(dry)} clear-weather frames got droplets on the lens"


def test_measured_frequencies_match_the_documented_priors(drawn: list[set[str]]) -> None:
    """The document publishes a frequency table; it has to stay true."""
    night = sum("NightIrMode" in s for s in drawn) / len(drawn)
    weather = sum(bool(s & WEATHER) for s in drawn) / len(drawn)
    assert night == pytest.approx(0.40, abs=0.07), f"night fraction {night:.2%}, designed 40%"
    assert weather == pytest.approx(0.12, abs=0.05), f"weather {weather:.2%}, designed 12%"


def test_the_shared_tail_is_in_physical_order() -> None:
    """`physical_order` cannot sort a pipeline built from containers, so the
    order inside each branch is written by hand — which makes it exactly the
    kind of thing that rots silently."""
    tail = _recipe()["_stream"](night=False, codec_p=0.15)
    stages = [stage_of(t) for t in tail]
    assert stages == sorted(stages), (
        "the shared tail is out of physical order: "
        + " -> ".join(f"{type(t).__name__}({s})" for t, s in zip(tail, stages, strict=True))
    )
