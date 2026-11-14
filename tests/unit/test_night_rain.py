"""Rain lit by the camera's own illuminator.

This transform exists because of a measurement: in daylight a physically
correct streak lifts the pixels it crosses by ~8 sRGB levels against 17 levels
of asphalt texture, so it cannot be seen (``docs/rain_visibility.md``). The
drop is not the problem — its illumination is. Lit by the camera instead of
the sky, the same drop is brighter than anything in the scene.

So the properties worth pinning are the ones that distinguish this from
``RandomRain``: brightness falls as 1/z², it adds light rather than occluding,
and it needs no depth provider.
"""

from __future__ import annotations

import numpy as np
import pytest

from surmentations.atmosphere import NightRain
from surmentations.diagnostics import recording
from surmentations.presets.pipeline import stage_of


def _night(size: int = 192) -> np.ndarray:
    """A dark frame, which is the only place this effect makes sense."""
    rng = np.random.default_rng(0)
    base = np.full((size, size, 3), 28, dtype=np.int16)
    base += rng.integers(-4, 5, base.shape, dtype=np.int16)
    base[int(size * 0.7) :] += 40  # a road the illuminator reaches
    return np.clip(base, 0, 255).astype(np.uint8)


def test_needs_no_depth_provider() -> None:
    """The drops are in front of the scene and the beam is a property of the
    lamp, so unlike every other atmospheric effect here this runs on CPU."""
    transform = NightRain(p=1.0)
    transform.set_random_seed(0)
    scene = _night()
    assert transform(image=scene)["image"].shape == scene.shape


def test_streak_brightness_falls_as_inverse_square() -> None:
    """The whole mechanism. A drop at 0.5 m receives a hundred times what one
    at 5 m does, which is what makes near streaks saturate and far ones vanish
    — and what daylight rain, lit uniformly by the sky, cannot do."""
    transform = NightRain(
        p=1.0,
        distance_range_m=(0.3, 8.0),
        illuminator_range_m=(40.0, 40.0),  # far reach, so the 1/z^2 term is alone
    )
    transform.set_random_seed(2)
    with recording() as log:
        transform(image=_night())
    streaks = log[0]["out"]["streaks"]
    distance = streaks["distance"]
    alpha = streaks["alpha"]
    assert isinstance(distance, dict) and isinstance(alpha, dict)
    # Brightest streaks must be the near ones: the max opacity should be many
    # times the median, which only happens if distance is driving it.
    assert alpha["max"] > 5.0 * max(alpha["p50"], 1e-6), (
        f"alpha max {alpha['max']:.4f} vs p50 {alpha['p50']:.4f} — too flat in "
        "distance for an inverse-square illuminator"
    )


def test_it_adds_light_rather_than_occluding() -> None:
    """A drop scattering the lamp's light back does not hide what is behind
    it; at these opacities there is nothing behind it to hide. So the frame
    can only get brighter."""
    transform = NightRain(p=1.0, backscatter_range=(0.0, 0.0))
    scene = _night()
    for seed in range(4):
        transform.set_random_seed(seed)
        out = transform(image=scene)["image"].astype(np.int16)
        assert (out >= scene.astype(np.int16) - 1).all(), "night rain darkened pixels"


def test_gain_and_rate_both_raise_visible_streak_energy() -> None:
    """Two knobs, both monotone: how hard the lamp drives the drops, and how
    many drops there are."""
    import cv2

    scene = _night(256)

    def streak_energy(**kwargs: object) -> float:
        transform = NightRain(p=1.0, backscatter_range=(0.0, 0.0), **kwargs)  # type: ignore[arg-type]
        values = []
        for seed in range(3):
            transform.set_random_seed(seed)
            delta = transform(image=scene)["image"].astype(np.float32) - scene
            grey = delta.mean(axis=2)
            values.append(float(np.abs(grey - cv2.GaussianBlur(grey, (0, 0), 2)).mean()))
        return float(np.mean(values))

    low_gain = streak_energy(illuminator_gain_range=(1.0, 1.0), rain_rate_range_mm_h=(40.0, 40.0))
    high_gain = streak_energy(illuminator_gain_range=(6.0, 6.0), rain_rate_range_mm_h=(40.0, 40.0))
    light = streak_energy(illuminator_gain_range=(4.0, 4.0), rain_rate_range_mm_h=(10.0, 10.0))
    heavy = streak_energy(illuminator_gain_range=(4.0, 4.0), rain_rate_range_mm_h=(90.0, 90.0))

    assert high_gain > 2.0 * low_gain, f"gain barely mattered: {low_gain:.3f} -> {high_gain:.3f}"
    assert heavy > 1.5 * light, f"rate barely mattered: {light:.3f} -> {heavy:.3f}"


def test_backscatter_is_separable_and_brightens_the_whole_beam() -> None:
    """The unresolved drops the beam lights scatter straight back — the wall
    of light that makes an IR camera useless in heavy rain. It has to be
    switchable, because it is the part that destroys contrast."""
    scene = _night()
    off = NightRain(p=1.0, backscatter_range=(0.0, 0.0))
    on = NightRain(p=1.0, backscatter_range=(0.25, 0.25))
    off.set_random_seed(5)
    on.set_random_seed(5)
    assert on(image=scene)["image"].mean() > off(image=scene)["image"].mean() + 5.0


def test_runs_after_the_infrared_conversion() -> None:
    """The streaks are already in infrared terms — the lamp lit them and the
    sensor saw them — so passing them through the scene's reflectance-to-IR
    conversion would convert them twice."""
    assert stage_of(NightRain) > stage_of_night_ir()


def stage_of_night_ir() -> int:
    """The pipeline stage ``NightIrMode`` belongs to."""
    from surmentations.camera import NightIrMode

    return stage_of(NightIrMode)


def test_rejects_a_negative_illuminator_gain() -> None:
    with pytest.raises(Exception, match="illuminator_gain_range"):
        NightRain(p=1.0, illuminator_gain_range=(-1.0, 2.0))
