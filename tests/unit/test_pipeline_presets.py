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

WEATHER = {"RandomRain", "RandomSnow", "RandomFog", "NightRain"}
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


def test_pipelines_are_in_physical_order() -> None:
    """`physical_order` cannot sort containers, so the order is written by hand
    and has to be checked."""
    from weathermentations.presets.pipelines import _stream

    tail = _stream(
        lit_by_mains=False, dirt=0.2, occluders=0.1, codec=0.1, severity=2
    )
    stages = [stage_of(t) for t in tail]
    assert stages == sorted(stages), stages


def test_effects_per_frame_is_not_all_or_nothing() -> None:
    """Effects must fire roughly independently, not in lockstep.

    Albumentations seeds every child of a `Compose` from the same stream, and
    a `OneOf` that declines to fire consumes exactly one number — so a
    pipeline built from same-shaped `OneOf` gates never desynchronises and
    they all fire together whenever the shared draw falls below the smallest
    `p`. An earlier version picked each effect's severity with such a gate and
    turned an expected 2.4 effects per frame into 64% of frames with one or
    none and 12% with nine or more. The marginals stayed correct throughout,
    which is why only the joint distribution catches it.
    """
    drawn = _draws(surveillance_lite(), samples=900)
    counts = np.array([len(s) for s in drawn])
    # Independent Bernoulli draws give variance <= mean; lockstep firing sends
    # it far above. The bug measured 8.76 against a mean of 2.37.
    assert counts.var() < 1.6 * counts.mean(), (
        f"effects per frame: mean {counts.mean():.2f}, variance {counts.var():.2f} — "
        f"too clustered to be independent draws. Histogram: "
        f"{np.bincount(counts).tolist()}"
    )
    # Not eliminated, only made rare: the containers that pick the lighting
    # and weather state still share the stream, so a heavily stacked frame
    # turns up occasionally. Measured across three seeds it is 0.0-0.3% of
    # frames, against 12% when every effect was gated this way.
    heavy = (counts >= 8).mean()
    assert heavy < 0.015, (
        f"{heavy:.1%} of frames stacked 8+ effects; expected well under 1%. "
        f"Histogram: {np.bincount(counts).tolist()}"
    )


def test_night_rain_replaces_daylight_rain_under_infrared() -> None:
    """At night the rain is lit by the camera, so it is a different transform
    in a different place in the chain.

    `NightRain` follows the reflectance-to-infrared conversion because its
    streaks are already in infrared terms; `RandomRain` precedes the sensor
    because its streaks take the sky's radiance. A frame must get one or the
    other, never both, and the night one only ever alongside `NightIrMode`.
    """
    drawn = _draws(surveillance(SharedDepthProvider(_GroundPlane()), rain=0.12), samples=900)
    night_rain = [s for s in drawn if "NightRain" in s]
    assert night_rain, "NightRain never fired; the IR-night branch is not wired up"
    assert all("NightIrMode" in s for s in night_rain), (
        "NightRain fired on a frame with no infrared conversion, which is the "
        "one condition that makes it physically meaningful"
    )
    assert not any("RandomRain" in s for s in night_rain), (
        "a frame got both the daylight and the illuminated rain model"
    )


def test_pipelines_without_a_codec_stage_do_not_need_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``codec=0.0`` is the documented setting for frames decoded from video,
    and it has to work on a machine with no ffmpeg — `surveillance_lite` is
    sold as running anywhere. It used to construct the codec transform anyway,
    at probability zero, and the constructor is what demands the binary."""
    from weathermentations.camera import codec as codec_module

    monkeypatch.setattr(codec_module, "ffmpeg_path", lambda: None)

    lite = surveillance_lite(codec=0.0)
    lite.set_random_seed(0)
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    assert lite(image=image)["image"].shape == image.shape

    surveillance(SharedDepthProvider(_GroundPlane()), codec=0.0)

    # Asking for the codec stage without the binary still fails loudly, at
    # construction, rather than silently training without it.
    with pytest.raises(RuntimeError, match="ffmpeg"):
        surveillance_lite(codec=0.2)
