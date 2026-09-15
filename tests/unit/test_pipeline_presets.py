"""The shipped pipelines must keep the guarantees `docs/pipelines.md` claims.

`test_pipeline_recipe.py` checks the version printed in the document; this
checks the importable one. They are meant to agree, and the properties below
are the reason either exists: the impossible combinations have to be
impossible structurally, and the merely-correlated ones have to stay
correlated rather than becoming implications.
"""

from __future__ import annotations

import numpy as np
import pytest

from weathermentations.depth.estimator import DepthEstimator
from weathermentations.depth.provider import SharedDepthProvider
from weathermentations.diagnostics import recording
from weathermentations.presets import surveillance, surveillance_lite, weather_state
from weathermentations.presets.pipeline import stage_of

WEATHER = {"RandomRain", "RandomSnow", "RandomFog"}
DIRECT_SUN = {"SunFlare", "CastShadows"}
SAMPLES = 700


class _GroundPlane(DepthEstimator):
    def estimate(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        depth = np.clip(1.5 * 300.0 / np.maximum(rows - height / 2.0, 1e-3), 0.0, 50.0)
        return np.broadcast_to(depth / 50.0, (height, width)).astype(np.float32)


def _draws(transform, samples: int = SAMPLES) -> list[set[str]]:
    """Which effects fired per frame, seeded **once**.

    Re-seeding per call correlates every transform's probability check, so
    frames come out with nearly all effects or nearly none — 27% empty and a
    variance of 7.25 effects/frame, against 8% and 1.81 seeded once. That is a
    real trap for callers and it would also make these assertions measure the
    wrong distribution.
    """
    transform.set_random_seed(5)
    image = np.full((96, 96, 3), 120, dtype=np.uint8)
    out = []
    for _ in range(samples):
        with recording() as log:
            transform(image=image)
        out.append({record["t"] for record in log})
    return out


@pytest.fixture(scope="module")
def full() -> list[set[str]]:
    return _draws(surveillance(SharedDepthProvider(_GroundPlane())))


def test_one_weather_state_at_most(full: list[set[str]]) -> None:
    assert not [s for s in full if len(s & WEATHER) > 1]


def test_infrared_night_never_carries_sunlight(full: list[set[str]]) -> None:
    assert not [s for s in full if "NightIrMode" in s and s & DIRECT_SUN]


def test_water_on_the_glass_is_correlated_not_implied(full: list[set[str]]) -> None:
    """Both directions must occur: dew under a clear sky, and a shielded lens
    staying dry in the rain."""
    assert sum("WaterDroplets" in s and not s & WEATHER for s in full) > 0
    assert sum(bool(s & WEATHER) and "WaterDroplets" not in s for s in full) > 0

    rain = [s for s in full if "RandomRain" in s]
    clear = [s for s in full if not s & WEATHER]
    p_rain = sum("WaterDroplets" in s for s in rain) / max(len(rain), 1)
    p_clear = sum("WaterDroplets" in s for s in clear) / max(len(clear), 1)
    assert p_rain > 3 * p_clear, f"droplets|rain {p_rain:.2%} vs |clear {p_clear:.2%}"


def test_lite_needs_no_depth_provider_and_renders() -> None:
    """Its entire point is running where there is no CUDA device."""
    transform = surveillance_lite()
    transform.set_random_seed(0)
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    assert transform(image=image)["image"].shape == image.shape

    drawn = _draws(transform, samples=200)
    fired = {name for s in drawn for name in s}
    assert not fired & (WEATHER | {"NightIrMode", "CastShadows"}), (
        f"surveillance_lite used depth-conditioned effects: {sorted(fired & WEATHER)}"
    )


def test_weather_state_rejects_probabilities_that_leave_no_clear_weather() -> None:
    """They are absolute probabilities, not weights; silently renormalising
    them would make `clear` vanish without telling anyone."""
    provider = SharedDepthProvider(_GroundPlane())
    with pytest.raises(ValueError, match="room for clear weather"):
        weather_state(provider, rain=0.6, snow=0.3, fog=0.2)


@pytest.mark.parametrize("build", [lambda: surveillance_lite()])
def test_pipelines_are_in_physical_order(build) -> None:
    """`physical_order` cannot sort containers, so the order is written by hand
    and has to be checked."""
    from weathermentations.presets.pipelines import _stream

    stages = [stage_of(t) for t in _stream(lit_by_mains=False, dirt=0.2, occluders=0.1, codec=0.1)]
    assert stages == sorted(stages), stages
