"""NightInsects: streaks whose length, beading and brightness follow from where the insect is.

Each claim of the module's docstring as a measurement on
:func:`~weathermentations.atmosphere.night_insects.insect_streaks`, and the
transform held to adding light and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

from weathermentations.atmosphere import NightInsects
from weathermentations.atmosphere.night_insects import insect_streaks
from weathermentations.utils.illuminator import lamp_falloff

SETTINGS = {
    "width": 640,
    "height": 480,
    "focal": 600.0,
    "aperture_m": 1.5e-3,
    "focus_m": 20.0,
    "exposure_s": 0.03,
    "gain": 1.0,
}


def _one(**insect: float) -> dict[str, np.ndarray]:
    """The discs of one insect; what is not given is a 10 mm moth 0.5 m away at 1 m/s."""
    values = {
        "distance_m": 0.5,
        "length_m": 0.01,
        "speed_m_s": 1.0,
        "wing_beat_hz": 40.0,
        "wing_share": 0.5,
    }
    settings = dict(SETTINGS)
    for key, value in insect.items():
        (settings if key in settings else values)[key] = value
    arrays = {key: np.array([value]) for key, value in values.items()}
    return insect_streaks(np.random.default_rng(0), **settings, **arrays)


def _length(discs: dict[str, np.ndarray]) -> float:
    """Distance along the path from the first disc's centre to the last's, in pixels."""
    steps = np.hypot(np.diff(discs["x"].astype(np.float64)), np.diff(discs["y"].astype(np.float64)))
    return float(steps.sum())


def _night() -> np.ndarray:
    """A dark, slightly textured frame."""
    return np.random.default_rng(0).integers(10, 60, (240, 320, 3), dtype=np.uint8)


def test_a_streak_is_as_long_as_the_insect_flies_in_the_exposure() -> None:
    # 1 m/s for 30 ms at 0.5 m through a 600 px lens: 36 px, a little more
    # along the bow.
    assert _length(_one()) == pytest.approx(600.0 * 1.0 * 0.03 / 0.5, rel=0.12)
    assert _length(_one(exposure_s=0.06)) == pytest.approx(2.0 * _length(_one()), rel=0.1)
    assert _length(_one(distance_m=0.25)) == pytest.approx(2.0 * _length(_one()), rel=0.1)


def test_a_faster_insect_returns_the_same_light_over_a_longer_streak() -> None:
    slow, fast = _one(speed_m_s=0.5), _one(speed_m_s=3.0)
    assert _length(fast) > 4.0 * _length(slow)
    assert float(fast["weight"].sum()) == pytest.approx(float(slow["weight"].sum()), rel=0.1)


def test_an_insect_is_lit_by_its_own_distance_from_the_lamp() -> None:
    near, far = _one(distance_m=0.3), _one(distance_m=1.2)
    # Its light is its radiance times the pixels it covers, and both fall
    # with distance: the lamp's falloff, and an image a quarter the size.
    expected = (lamp_falloff(0.3) / lamp_falloff(1.2)) * (1.2 / 0.3) ** 2
    ratio = float(near["weight"].sum()) / float(far["weight"].sum())
    assert ratio == pytest.approx(float(expected), rel=0.1)


def test_a_near_insect_is_drawn_wider() -> None:
    assert _one(distance_m=0.2)["radius"][0] > 3.0 * _one(distance_m=1.0)["radius"][0]


def test_the_streak_has_one_lobe_a_wing_beat() -> None:
    def lobes(beat_hz: float) -> int:
        """Peaks in the light along a streak whose light is all in its wings."""
        weight = _one(wing_beat_hz=beat_hz, wing_share=1.0, speed_m_s=3.0)["weight"]
        inner = weight[1:-1]
        return int(((inner > weight[:-2]) & (inner >= weight[2:]) & (inner > weight.mean())).sum())

    # 100 Hz for 30 ms is three beats, 200 Hz six; a peak at either end of
    # the streak may be cut off.
    assert lobes(100.0) in (2, 3, 4)
    assert lobes(200.0) in (5, 6, 7)


def test_dry_wings_give_an_even_streak() -> None:
    weight = _one(wing_share=0.0)["weight"]
    assert float(weight.std() / weight.mean()) < 1e-5


def test_insects_and_dust_only_add_light() -> None:
    image = _night()
    for seed in range(5):
        transform = NightInsects(p=1.0, insect_count_range=(4, 8), dust_count_range=(5, 10))
        transform.set_random_seed(seed)
        out = transform(image=image)["image"]
        assert (out >= image).all()
        assert (out > image).any()


def test_nothing_in_the_beam_changes_nothing() -> None:
    image = _night()
    transform = NightInsects(p=1.0, insect_count_range=(0, 0), dust_count_range=(0, 0))
    transform.set_random_seed(0)
    assert np.array_equal(transform(image=image)["image"], image)


def test_dust_is_one_faint_disc_a_mote() -> None:
    transform = NightInsects(p=1.0, insect_count_range=(0, 0), dust_count_range=(7, 7))
    transform.set_random_seed(1)
    discs = transform.get_params_dependent_on_data({"shape": (240, 320, 3)}, {})["discs"]
    assert len(discs["x"]) == 7
    brightness = discs["weight"] / (np.pi * (discs["radius"] ** 2 + 1.0 / 12.0))
    # The lamp is at most about 4.5 times its strength at 10 cm.
    assert float(brightness.max()) <= 0.2 * 4.6


def test_the_same_seed_draws_the_same_insects() -> None:
    image = _night()
    frames = []
    for _ in range(2):
        transform = NightInsects(p=1.0)
        transform.set_random_seed(4)
        frames.append(transform(image=image)["image"])
    assert np.array_equal(frames[0], frames[1])
