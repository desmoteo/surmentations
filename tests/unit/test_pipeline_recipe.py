"""The composite in `docs/pipelines.md` must keep the guarantees it claims.

The document's argument is that exclusivity is *structural* — rain and snow
cannot co-occur because they are alternatives of one `OneOf`, a sun flare
cannot land on an IR night frame because the lighting branches are
alternatives of another, and neither a flare nor a hard shadow can land on
a frame with rain or fog in it because they are built only into the states
with dry air. That is cheap to break: moving `SunFlare` from the dry states
into the shared tail would silently reintroduce flare-at-night and
flare-in-the-rain, and nothing else in this suite would notice.

Equally load-bearing, and easier to get wrong, is what must **not** be
exclusive. Water on the glass is correlated with rain, not implied by it: a
shielded lens stays dry in a downpour and dew forms under a clear sky. An
earlier version of the recipe forced droplets to require wet weather, which
is the same modelling error pointing the other way, so both directions are
asserted here.

The recipe is executed exactly as the document prints it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import recording
from surmentations.presets.pipeline import stage_of

DOC = Path(__file__).resolve().parents[2] / "docs" / "pipelines.md"

WEATHER = {"RandomRain", "RandomSnow", "RandomFog", "NightRain", "NightFog"}
STATES = (
    {"RandomRain", "NightRain"},
    {"RandomFog", "NightFog"},
    {"RandomSnow"},
)
DIRECT_SUN = {"SunFlare", "CastShadows"}
SAMPLES = 900


class _GroundPlane(DepthEstimator):
    """A level road under the lower half of the frame."""
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Depth of a level road under the lower half of the frame, and a far wall above it."""
        height, width = image.shape[:2]
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        depth = np.clip(1.5 * 300.0 / np.maximum(rows - height / 2.0, 1e-3), 0.0, 50.0)
        return np.broadcast_to(depth / 50.0, (height, width)).astype(np.float32)


def _recipe() -> dict[str, Any]:
    """The document's recipe block, executed as written.

    Selected by content rather than position: the document opens with a short
    import snippet, and indexing blindly into `blocks[0]` silently ran that
    instead the moment the page was reorganised.
    """
    blocks = [
        b
        for b in re.findall(r"```python\n(.*?)```", DOC.read_text(), re.S)
        if "def surveillance" in b
    ]
    assert blocks, "docs/pipelines.md no longer contains the recipe"
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
    """One state of the air and the ground per frame: the recipe has no arm
    for rain through fog, which the importable pipeline has."""
    both = [s for s in drawn if sum(bool(s & state) for state in STATES) > 1]
    assert not both, f"{len(both)} images got several weather states at once, e.g. {both[0]}"


def test_night_infrared_never_coexists_with_direct_sunlight(drawn: list[set[str]]) -> None:
    """A frame the camera is lighting with its own IR illuminator has no sun in
    it, so it can carry neither a flare nor a hard cast shadow."""
    clashes = [s for s in drawn if "NightIrMode" in s and s & DIRECT_SUN]
    assert not clashes, f"{len(clashes)} night frames also got sunlight, e.g. {clashes[0]}"


def test_the_sun_is_out_only_where_nothing_falls_or_hangs_in_the_air(
    drawn: list[set[str]],
) -> None:
    """Rain and fog come with cloud. Hard shadows and a flare used to be
    drawn on a share of every day frame, whatever the weather."""
    sunlit = [s for s in drawn if s & DIRECT_SUN]
    assert sunlit, "the sun is never out"
    clashes = [s for s in sunlit if s & {"RandomRain", "RandomFog"}]
    assert not clashes, f"{len(clashes)} frames with the sun out in rain or fog, e.g. {clashes[0]}"


def test_water_on_the_glass_is_correlated_with_weather_not_implied_by_it(
    drawn: list[set[str]],
) -> None:
    """Both directions have to occur, or the joint distribution is wrong.

    Droplets without weather are dew, residual drops after the rain stopped,
    or wind-blown spray. Weather without droplets is a recessed or shielded
    lens. Forbidding either is as wrong as drawing them independently.
    """
    droplets_no_weather = sum("WaterDroplets" in s and not s & WEATHER for s in drawn)
    weather_no_droplets = sum(bool(s & WEATHER) and "WaterDroplets" not in s for s in drawn)
    assert droplets_no_weather > 0, (
        "no frame got droplets under clear skies — dew and residual drops are real, "
        "and forcing droplets to require rain is the same error as drawing them "
        "independently, pointing the other way"
    )
    assert weather_no_droplets > 0, "no wet frame kept a dry lens; a shielded lens does"


def test_droplets_are_likelier_when_it_rains(drawn: list[set[str]]) -> None:
    """The correlation still has to be there, and in the right direction."""
    # Where rain is falling for certain: `RandomRain` is also a road left
    # wet with nothing falling, which has lower odds of its own, and by
    # name alone only the night's rain is told from it.
    rain = [s for s in drawn if "NightRain" in s]
    clear = [s for s in drawn if not s & WEATHER]
    assert len(rain) > 25 and clear
    p_wet_rain = sum("WaterDroplets" in s for s in rain) / len(rain)
    p_wet_clear = sum("WaterDroplets" in s for s in clear) / len(clear)
    assert p_wet_rain > 2 * p_wet_clear, (
        f"droplets given rain {p_wet_rain:.2%} vs given clear {p_wet_clear:.2%} — "
        "the recipe intends rain to make a wet lens several times likelier"
    )


def test_measured_frequencies_match_the_documented_priors(drawn: list[set[str]]) -> None:
    """The document publishes a frequency table; it has to stay true."""
    ir_night = sum("NightIrMode" in s for s in drawn) / len(drawn)
    dry = sum(not s & WEATHER for s in drawn) / len(drawn)
    wet = sum("RandomRain" in s for s in drawn) / len(drawn)
    assert ir_night == pytest.approx(0.30, abs=0.07), f"IR night {ir_night:.2%}, designed 30%"
    # The states are drawn by the square root of their shares of time: dry
    # air over a dry road on 44% of frames, a wet road with or without rain
    # on a third.
    assert dry == pytest.approx(0.44, abs=0.07), f"dry {dry:.2%}, designed 44%"
    assert wet == pytest.approx(0.34, abs=0.07), f"wet road {wet:.2%}, designed 34%"


def test_the_shared_tail_is_in_physical_order() -> None:
    """`physical_order` cannot sort a pipeline built from containers, so the
    order inside each branch is written by hand — which makes it exactly the
    kind of thing that rots silently."""
    tail = _recipe()["_stream"](lit_by_mains=False, codec_p=0.15)
    stages = [stage_of(t) for t in tail]
    assert stages == sorted(stages), (
        "the shared tail is out of physical order: "
        + " -> ".join(f"{type(t).__name__}({s})" for t, s in zip(tail, stages, strict=True))
    )
