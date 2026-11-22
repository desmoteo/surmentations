"""The shipped pipelines must keep the guarantees `docs/pipelines.md` claims.

`test_pipeline_recipe.py` checks the version printed in the document; this
checks the importable one. They are meant to agree, and the properties below
are the reason either exists: the impossible combinations have to be
impossible structurally, and the merely-correlated ones have to stay
correlated rather than becoming implications.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest

from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import recording
from surmentations.presets import surveillance, surveillance_lite, weather_state
from surmentations.presets.pipeline import stage_of

WEATHER = {"RandomRain", "RandomSnow", "RandomFog", "NightRain", "NightSnow", "NightFog"}
# Which transforms render each weather state. Snow at night is two of them by
# design — the veil and settled snow before the infrared conversion, the
# lamp-lit flakes after it — so exclusivity is a property of states, not of
# transforms.
STATES = {
    "rain": {"RandomRain", "NightRain"},
    "snow": {"RandomSnow", "NightSnow"},
    "fog": {"RandomFog", "NightFog"},
}


def _states(fired: set[str]) -> set[str]:
    """The lighting states whose transforms include any of ``fired``."""
    return {state for state, transforms in STATES.items() if fired & transforms}
DIRECT_SUN = {"SunFlare", "CastShadows"}
SAMPLES = 700


class _GroundPlane(DepthEstimator):
    """A level road under the lower half of the frame."""
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Depth of a level road under the lower half of the frame, and a far wall above it."""
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
    """Which transforms fired on each of many draws of the full pipeline, as a fixture."""
    return _draws(surveillance(SharedDepthProvider(_GroundPlane())))


NIGHT_MADE = {"NightIrMode", "NightColorMode"}


def test_snow_goes_with_nothing_else_and_rain_goes_only_with_fog(full: list[set[str]]) -> None:
    """One state of the air and the ground per frame. Rain falls through
    fog, and that is one of the states; nothing else is two at once."""
    assert not [s for s in full if "snow" in _states(s) and len(_states(s)) > 1]
    together = [s for s in full if _states(s) == {"rain", "fog"}]
    assert together, "fog with rain falling through it never drawn"
    # The road is the rain's and the air is the fog's: no second veil of
    # streaks under the fog.
    assert not [s for s in together if "NightRain" in s]


def test_the_sun_is_out_only_over_dry_air(full: list[set[str]]) -> None:
    """Hard shadows and a flare need the sun's own beam. They used to be
    drawn in three day frames in ten whatever was falling."""
    sunlit = [s for s in full if s & DIRECT_SUN]
    assert sunlit, "no frame with the sun in it"
    assert not [s for s in sunlit if s & {"RandomRain", "RandomFog", "NightRain", "NightFog"}]
    assert not [s for s in sunlit if s & NIGHT_MADE]


def test_the_weather_is_drawn_by_its_shares_raised_to_a_power() -> None:
    """Matched to how often each state happens at 1, every state alike at
    0, and between them the rare states more often than they happen."""
    from surmentations.presets.pipelines import rebalanced, weather_shares

    real = weather_shares()
    assert sum(real.values()) == pytest.approx(1.0) and real["dry"] > 0.7
    assert real["rain_light"] > 10.0 * real["rain_heavy"] > 0.0
    assert real["wet_road"] > real["rain_light"], "a road is wet for longer than it rains"
    assert rebalanced(real, 1.0) == pytest.approx(real)
    even = rebalanced(real, 0.0)
    assert max(even.values()) == pytest.approx(min(even.values()))
    half = rebalanced(real, 0.5)
    assert sum(half.values()) == pytest.approx(1.0)
    assert real["dry"] > half["dry"] > even["dry"]
    assert real["fog_dense"] < half["fog_dense"] < half["fog_thin"] < half["dry"]
    # Fog belongs to the night more than to the day.
    assert weather_shares(hour="night")["fog_thin"] > 2.0 * weather_shares(hour="day")["fog_thin"]
    with pytest.raises(ValueError, match="prior_power"):
        rebalanced(real, 1.5)
    # The states are named in one order, and the arms are held in it.
    from surmentations.presets import WEATHER_STATES

    assert tuple(real) == WEATHER_STATES
    state = weather_state(SharedDepthProvider(_GroundPlane()))
    assert len(state.transforms) == len(WEATHER_STATES)
    weights = np.array([arm.p for arm in state.transforms])
    assert weights / weights.sum() == pytest.approx(
        list(rebalanced(weather_shares(hour="day"), 0.5).values())
    )


def test_most_of_the_weather_is_mild_and_the_ground_outlasts_it() -> None:
    """A road still wet with nothing falling, snow lying under a clear sky
    and rain too light to see are most of the bad weather there is, and
    could not be drawn at all."""
    transform = surveillance(
        SharedDepthProvider(_GroundPlane()), day=1.0, lit_night=0.0, ir_night=0.0, codec=0.0
    )
    transform.set_random_seed(5)
    image = np.full((96, 96, 3), 120, dtype=np.uint8)
    rates, after_rain, lying = [], 0, 0
    for _ in range(600):
        with recording() as log:
            transform(image=image)
        fired = {record["t"]: record for record in log}
        if "RandomRain" in fired and "RandomFog" not in fired:
            rain = fired["RandomRain"]
            if rain["init"]["max_streaks"] == 0:
                after_rain += 1
                assert rain["drew"]["draw_wetness"], "a road left wet that is dry"
            else:
                rates.append(rain["drew"]["rain_rate_mm_h"])
        if "RandomSnow" in fired and fired["RandomSnow"]["init"]["max_flakes"] == 0:
            lying += 1
    assert after_rain > 30 and lying > 15
    rates_drawn = np.array(rates)
    assert len(rates_drawn) > 40
    assert np.median(rates_drawn) < 5.0, "most rain is light"
    assert rates_drawn.max() > 10.0, "and some of it is not"


def test_infrared_night_never_carries_sunlight(full: list[set[str]]) -> None:
    assert not [s for s in full if "NightIrMode" in s and s & DIRECT_SUN]


def test_water_on_the_glass_is_correlated_not_implied(full: list[set[str]]) -> None:
    """Both directions must occur: dew under a clear sky, and a shielded lens
    staying dry in the rain."""
    assert sum("WaterDroplets" in s and not s & WEATHER for s in full) > 0
    assert sum(bool(s & WEATHER) and "WaterDroplets" not in s for s in full) > 0

    # Where rain is falling, against where nothing is and the road is dry:
    # a road left wet is `RandomRain` too, and has its own, lower odds.
    transform = surveillance(
        SharedDepthProvider(_GroundPlane()),
        day=1.0, lit_night=0.0, ir_night=0.0, codec=0.0, rain=0.3, prior_power=1.0,
    )
    transform.set_random_seed(5)
    image = np.full((96, 96, 3), 120, dtype=np.uint8)
    rain, clear = [], []
    for _ in range(600):
        with recording() as log:
            transform(image=image)
        fired = {record["t"]: record for record in log}
        if "RandomRain" in fired and fired["RandomRain"]["init"]["max_streaks"] > 0:
            rain.append("WaterDroplets" in fired)
        elif not set(fired) & WEATHER:
            clear.append("WaterDroplets" in fired)
    assert len(rain) > 80 and len(clear) > 150
    p_rain, p_clear = float(np.mean(rain)), float(np.mean(clear))
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
    from surmentations.presets.pipelines import _stream

    tail = _stream(
        lit_by_mains=False, infrared=False, dirt=0.2, occluders=0.1, codec=0.1,
        faults=0.1, severity=2,
    )
    stages = [stage_of(t) for t in tail]
    assert stages == sorted(stages), stages
    assert type(tail[-1]).__name__ == "TransmissionFaults", (
        "the link loses packets of the stream the camera encoded, so it is last"
    )


def test_the_camera_s_two_settings_are_the_caller_s() -> None:
    """Wide dynamic range and the noise reducer are settings of a camera,
    and the two costliest stages of its chain: how often each shows is an
    argument, the noise reducer's a quarter as often by day as at night."""
    from surmentations.presets.pipelines import _stream

    def odds(**given: float) -> dict[str, dict[str, float]]:
        found: dict[str, dict[str, float]] = {}
        for light, lit, infrared in (("day", False, False), ("night", True, False)):
            tail = _stream(
                lit_by_mains=lit, infrared=infrared, dirt=0.2, occluders=0.1, codec=0.0,
                faults=0.0, severity=2, **given,
            )  # fmt: skip
            found[light] = {type(t).__name__: t.p for t in tail}
        return found

    usual = odds()
    assert usual["day"]["WdrToneMapping"] == usual["night"]["WdrToneMapping"] == 0.30
    assert usual["night"]["TemporalDenoiseArtifacts"] == pytest.approx(0.8)
    assert usual["day"]["TemporalDenoiseArtifacts"] == pytest.approx(0.2)
    off = odds(wdr=0.0, denoise=0.0)
    for light in ("day", "night"):
        assert off[light]["WdrToneMapping"] == 0.0
        assert off[light]["TemporalDenoiseArtifacts"] == 0.0
    # And they reach the chain from both pipelines.
    for pipeline in (
        surveillance(SharedDepthProvider(_GroundPlane()), codec=0.0, wdr=0.05, denoise=0.4),
        surveillance_lite(codec=0.0, wdr=0.05, denoise=0.4),
    ):
        seen: dict[str, set[float]] = {}

        def walk(node: object, seen: dict[str, set[float]] = seen) -> None:
            for child in getattr(node, "transforms", []):
                seen.setdefault(type(child).__name__, set()).add(round(float(child.p), 6))
                walk(child)

        walk(pipeline)
        assert seen["WdrToneMapping"] == {0.05}
        assert seen["TemporalDenoiseArtifacts"] <= {0.4, 0.1}
        assert 0.4 in seen["TemporalDenoiseArtifacts"]


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
    because its streaks take the sky's radiance. A frame must get one set of
    streaks or the other, never both, and the night one only ever alongside
    `NightIrMode`.

    Rain under infrared is split as snow is: `RandomRain` still comes first,
    for the veil and the wet road, which belong to the scene, and is built
    to draw no streaks of its own. It used to be left out altogether, and
    the lamp-lit rain fell on a road as dry as the photograph's.
    """
    transform = surveillance(SharedDepthProvider(_GroundPlane()), rain=0.12)
    transform.set_random_seed(5)
    image = np.full((96, 96, 3), 120, dtype=np.uint8)
    night_rain, day_rain = [], []
    for _ in range(900):
        with recording() as log:
            transform(image=image)
        fired = {record["t"]: record for record in log}
        if "NightRain" in fired:
            night_rain.append(fired)
        elif "RandomRain" in fired:
            day_rain.append(fired)
    assert night_rain, "NightRain never fired; the night branches are not wired up"
    assert all(NIGHT_MADE & set(fired) for fired in night_rain), (
        "NightRain fired on a frame that was not made into a night, which is "
        "the one condition that makes it physically meaningful"
    )
    assert any("NightIrMode" in fired for fired in night_rain)
    # In colour there is no lamp at the lens: the rain shows in the
    # street's light alone.
    in_colour = [fired for fired in night_rain if "NightColorMode" in fired]
    assert in_colour
    assert all(
        max(fired["NightRain"]["init"]["illuminator_gain_range"]) == 0.0 for fired in in_colour
    )
    assert all("RandomRain" in fired for fired in night_rain), (
        "lamp-lit rain with no veil and no chance of a wet road under it"
    )
    assert all(fired["RandomRain"]["init"]["max_streaks"] == 0 for fired in night_rain), (
        "a frame got both the daylight and the illuminated rain model's streaks"
    )
    assert any(fired["RandomRain"]["drew"]["draw_wetness"] for fired in night_rain)
    # By day the streaks are `RandomRain`'s own, wherever rain is falling
    # and no fog hides it; and a night that was made never has them.
    falling = [
        fired
        for fired in day_rain
        if not NIGHT_MADE & set(fired)
        and "RandomFog" not in fired
        and fired["RandomRain"]["init"]["rain_rate_range_mm_h"][1] > 2.0
    ]
    assert falling and all(fired["RandomRain"]["init"]["max_streaks"] > 0 for fired in falling)
    made = [fired for fired in day_rain if NIGHT_MADE & set(fired)]
    assert all(fired["RandomRain"]["init"]["max_streaks"] == 0 for fired in made)


def test_pipelines_without_a_codec_stage_do_not_need_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``codec=0.0`` is the documented setting for frames decoded from video,
    and it has to work on a machine with no ffmpeg — `surveillance_lite` is
    sold as running anywhere. It used to construct the codec transform anyway,
    at probability zero, and the constructor is what demands the binary."""
    from surmentations.camera import _inprocess
    from surmentations.camera import codec as codec_module

    monkeypatch.setattr(codec_module, "ffmpeg_path", lambda: None)
    # Nor PyAV, which would do in the binary's place.
    monkeypatch.setattr(_inprocess, "has_encoder", lambda encoder: False)

    lite = surveillance_lite(codec=0.0)
    lite.set_random_seed(0)
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    assert lite(image=image)["image"].shape == image.shape

    surveillance(SharedDepthProvider(_GroundPlane()), codec=0.0)

    # Asking for the codec stage without the binary still fails loudly, at
    # construction, rather than silently training without it.
    with pytest.raises(RuntimeError, match="ffmpeg"):
        surveillance_lite(codec=0.2)
    # Packet loss needs the binary too. Unset, it follows `codec` to zero,
    # which is why the calls above worked; asked for, it fails as loudly.
    with pytest.raises(RuntimeError, match="ffmpeg"):
        surveillance_lite(codec=0.0, faults=0.05)


def names(transform: A.Compose) -> set[str]:
    """Every class anywhere in a pipeline's tree, fired or not."""
    found: set[str] = set()

    def walk(node: object) -> None:
        found.add(type(node).__name__)
        for child in getattr(node, "transforms", []):
            walk(child)

    walk(transform)
    return found


def test_packet_loss_is_on_by_default_only_where_the_codec_stage_is() -> None:
    assert "TransmissionFaults" in names(surveillance_lite())
    assert "TransmissionFaults" not in names(surveillance_lite(codec=0.0))
    assert "TransmissionFaults" not in names(surveillance_lite(faults=0.0))
    assert "TransmissionFaults" in names(surveillance_lite(codec=0.0, faults=0.2))


def test_occluders_glow_only_under_the_infrared_ring() -> None:
    """An insect on the dome is a silhouette by day and glows at night, and
    which one is decided by whether the camera's illuminator is on — not by a
    draw. The transform's default spans both, so left alone the pipelines put
    glowing shapes on daylit frames."""
    transform = surveillance(SharedDepthProvider(_GroundPlane()), occluders=0.6, codec=0.0)
    transform.set_random_seed(5)
    image = np.full((96, 96, 3), 120, dtype=np.uint8)

    day, night = [], []
    for _ in range(400):
        with recording() as log:
            transform(image=image)
        fired = {record["t"]: record for record in log}
        if "HousingOccluders" in fired:
            glow = fired["HousingOccluders"]["out"]["glow"]
            (night if "NightIrMode" in fired else day).append(glow)

    assert len(day) > 20 and len(night) > 20, (len(day), len(night))
    assert max(day) == 0.0, f"an occluder glowed at {max(day):.2f} with no illuminator"
    assert min(night) >= 0.6, f"an occluder under the IR ring glowed at only {min(night):.2f}"


def test_night_snow_is_lit_by_the_lamp_and_only_under_infrared() -> None:
    """Snow at night is split: its veil and settled snow belong to the scene
    and precede the infrared conversion, its falling flakes are lit by the
    camera and follow it. The two must come together, never on a day frame,
    and the scene half must not also draw the flakes."""
    drawn = _draws(surveillance(SharedDepthProvider(_GroundPlane()), snow=0.15), samples=600)
    night = [s for s in drawn if "NightSnow" in s]
    assert night, "NightSnow never fired; the night snow arms are not wired up"
    assert all("RandomSnow" in s and s & NIGHT_MADE for s in night)
    assert not any("NightRain" in s or "RandomRain" in s or "RandomFog" in s for s in night)

    day_snow = [s for s in drawn if "RandomSnow" in s and not s & NIGHT_MADE]
    assert day_snow and not any("NightSnow" in s for s in day_snow)


def test_snow_sits_on_the_housing_only_where_there_is_snow() -> None:
    """While it snows a cap builds on the housing and hangs into the view,
    and it is still there for a while after. Never without snow."""
    drawn = _draws(
        surveillance(SharedDepthProvider(_GroundPlane()), snow=0.2, snow_lying=0.2, codec=0.0),
        samples=500,
    )
    on_housing = [s for s in drawn if "SnowOnHousing" in s]
    assert on_housing, "snow never on the housing"
    assert all("RandomSnow" in s for s in on_housing)
    snowy = [s for s in drawn if "RandomSnow" in s]
    assert 0.05 < len(on_housing) / len(snowy) < 0.45


FALLING = {"RandomFog", "NightRain", "NightSnow", "NightFog"}


def test_insects_and_webs_are_lit_by_the_ring_and_kept_to_their_season() -> None:
    """Insects in the beam and a web across the housing show by the
    camera's own lamp, centimetres away: never by day and never in a night
    in colour. Insects fly where nothing falls; a web is there in rain and
    fog, which bead it; and neither goes with snow, which is not their
    season."""
    transform = surveillance(
        SharedDepthProvider(_GroundPlane()),
        insects=1.0, webs=1.0, codec=0.0,
        rain=0.2, fog=0.1, snow=0.1, snow_lying=0.1, prior_power=1.0,
    )  # fmt: skip
    drawn = _draws(transform, samples=500)
    flying = [s for s in drawn if "NightInsects" in s]
    webbed = [s for s in drawn if "SpiderWeb" in s]
    assert flying and webbed
    assert all("NightIrMode" in s for s in flying + webbed)
    assert not [s for s in flying if s & FALLING or "RandomSnow" in s]
    assert not [s for s in webbed if "RandomSnow" in s]
    assert any(s & {"NightRain", "NightFog"} for s in webbed), "a web never beaded by rain or fog"
    # At p=1 each is certain in its own states, so it is missing only
    # from the others.
    infrared = [s for s in drawn if "NightIrMode" in s]
    assert all("NightInsects" in s for s in infrared if not s & FALLING and "RandomSnow" not in s)
    assert all("SpiderWeb" in s for s in infrared if "RandomSnow" not in s)


def test_heat_shimmer_is_drawn_only_over_dry_ground_by_day() -> None:
    """The air shimmers over ground the sun has made warmer than it: never
    at night, never under rain, fog or snow, and not over a wet road."""
    transform = surveillance(
        SharedDepthProvider(_GroundPlane()), shimmer=1.0, codec=0.0, prior_power=0.5
    )
    drawn = _draws(transform, samples=400)
    shimmering = [s for s in drawn if "HeatShimmer" in s]
    assert shimmering
    assert not [s for s in shimmering if s & (NIGHT_MADE | WEATHER)]
    dry_days = [s for s in drawn if not s & (NIGHT_MADE | WEATHER)]
    assert all("HeatShimmer" in s for s in dry_days)


def test_insects_webs_and_shimmer_at_zero_are_not_in_the_pipeline() -> None:
    """Left out, not built and never fired: a caller who turns one off has
    the pipeline's draws as they were before it was added."""
    provider = SharedDepthProvider(_GroundPlane())
    three = {"NightInsects", "SpiderWeb", "HeatShimmer"}
    assert three <= names(surveillance(provider))
    assert not three & names(surveillance(provider, insects=0.0, webs=0.0, shimmer=0.0))


def test_water_on_the_glass_follows_the_lighting_state() -> None:
    """Two things about water on the glass are decided by the light, not
    drawn. It glows only when the camera's own illuminator is on, and it is
    blurred by an iris that is wide open at night — the aperture is what sets
    how sharp a drop looks, so the same wet dome reads differently by day."""
    transform = surveillance(
        SharedDepthProvider(_GroundPlane()),
        codec=0.0, wet_clear=0.6, wet_rain=0.6, wet_snow=0.6, wet_fog=0.6,
    )
    transform.set_random_seed(5)
    image = np.full((96, 96, 3), 120, dtype=np.uint8)

    day, infrared = [], []
    for _ in range(300):
        with recording() as log:
            transform(image=image)
        fired = {record["t"]: record for record in log}
        if "WaterDroplets" in fired:
            water = fired["WaterDroplets"]
            sample = (water["out"]["glow"], water["drew"]["aperture_mm"])
            (infrared if "NightIrMode" in fired else day).append(sample)

    assert len(day) > 20 and len(infrared) > 20, (len(day), len(infrared))
    assert max(glow for glow, _ in day) == 0.0, "water glowed with no illuminator on"
    assert min(glow for glow, _ in infrared) >= 0.6
    assert min(aperture for _, aperture in infrared) >= 1.4, "a stopped-down iris at night"
    assert min(aperture for _, aperture in day) < 1.0, "the day branches never stop down"


def test_rain_at_night_is_metered_for_street_lighting() -> None:
    """The camera that is wide open for the water on its glass is the same
    one looking at the rain. Under street lighting its shutter runs at the
    slow end of what it can do, so night rain is long faint streaks; by day
    the same transform meters for daylight and mostly freezes the drops."""
    from surmentations.presets.pipelines import weather_state

    def shutters(night: bool) -> np.ndarray:
        transform = A.Compose(
            [weather_state(SharedDepthProvider(_GroundPlane()), rain=0.9, snow=0.0, fog=0.0,
                           wet_road=0.0, snow_lying=0.0, night=night)]
        )
        transform.set_random_seed(3)
        image = np.full((96, 96, 3), 120, dtype=np.uint8)
        drawn = []
        for _ in range(60):
            with recording() as log:
                transform(image=image)
            drawn += [r["drew"]["exposure_s"] for r in log if r["t"] == "RandomRain"]
        return np.array(drawn)

    day, night = shutters(night=False), shutters(night=True)
    assert len(day) > 30 and len(night) > 30
    assert np.median(night) >= 1 / 30
    assert np.median(day) < 1 / 100
