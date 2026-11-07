"""SpiderWeb: a web in front of the lens, drawn as the discs its threads are seen as.

The claims the module's docstring makes about where a thread is, each as a
measurement: how wide and how bright it is drawn at a distance, what beading
does and does not change, and what the spider is that the threads are not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from weathermentations import mode
from weathermentations.lens import SpiderWeb
from weathermentations.lens.spider_web import _NEAREST_M, _glints, _orb_web, _pieces
from weathermentations.utils.illuminator import (
    REFERENCE_DISTANCE_M,
    RING_RADIUS_M,
    lamp_falloff,
    splat_discs,
)

WIDTH, HEIGHT, FOCAL = 640, 480, 600.0
APERTURE_M, FOCUS_M = 2.0e-3, 20.0


def _thread(distance_m: float, tilt_m: float = 0.0) -> dict[str, np.ndarray]:
    """One thread across the whole view at a distance, in pieces.

    Level with the lens's axis and across the line of sight; with ``tilt_m``
    its right end is that much further away than its left.
    """
    half = 0.8 * distance_m
    threads = {
        "start": np.array([[-half, 0.0, distance_m - tilt_m / 2.0]]),
        "end": np.array([[half, 0.0, distance_m + tilt_m / 2.0]]),
        "sag": np.zeros((1, 3)),
        "band": np.ones(1),
        "beads": np.ones(1),
    }
    return _pieces(threads)


def _drawn(pieces: dict[str, np.ndarray], **settings: float) -> tuple[np.ndarray, dict]:
    """The light a set of pieces puts in a frame, and the discs it was drawn from."""
    arguments = {
        "width": WIDTH,
        "height": HEIGHT,
        "focal": FOCAL,
        "aperture_m": APERTURE_M,
        "focus_m": FOCUS_M,
        "gain": 1.0,
        "bead_spacing_m": 2.0e-3,
        "bead_share": 0.0,
        "budget": 200_000,
    }
    arguments.update(settings)
    budget = int(arguments.pop("budget"))
    discs = _glints(np.random.default_rng(0), pieces, budget=budget, **arguments)
    lit = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    splat_discs(lit, discs["x"], discs["y"], discs["radius"], discs["weight"])
    return lit, discs


def _web(**settings: object) -> SpiderWeb:
    """The transform, always firing, with no spider and no loose threads unless asked."""
    arguments: dict[str, object] = {"spider_probability": 0.0, "stray_thread_count_range": (0, 0)}
    arguments.update(settings)
    return SpiderWeb(p=1.0, **arguments)


def _night() -> np.ndarray:
    """A dark frame with a little texture, as an infrared camera gives at night."""
    rng = np.random.default_rng(0)
    return rng.integers(10, 60, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def test_the_lamp_is_inverse_square_far_away_and_dark_at_the_glass() -> None:
    assert lamp_falloff(REFERENCE_DISTANCE_M) == pytest.approx(1.0)
    assert lamp_falloff(2.0) / lamp_falloff(1.0) == pytest.approx(0.25, rel=2e-3)
    assert lamp_falloff(0.0) == 0.0
    near = np.linspace(0.001, 0.1, 2000)
    brightest = near[np.argmax(lamp_falloff(near))]
    assert brightest == pytest.approx(RING_RADIUS_M / math.sqrt(2.0), rel=0.01)


def test_a_bare_thread_at_the_reference_distance_peaks_at_the_gain() -> None:
    for gain in (0.3, 1.7):
        lit, _ = _drawn(_thread(REFERENCE_DISTANCE_M), gain=gain)
        across = lit[:, WIDTH // 2 - 40 : WIDTH // 2 + 40].max(axis=0)
        assert across.mean() == pytest.approx(gain, rel=0.04)


def test_a_thread_is_drawn_as_wide_as_the_aperture_seen_from_it() -> None:
    lit, discs = _drawn(_thread(0.05))
    expected = 0.5 * APERTURE_M * FOCAL * (1.0 / 0.05 - 1.0 / FOCUS_M)
    assert np.allclose(discs["radius"], expected, rtol=0.02)
    column = lit[:, WIDTH // 2]
    width = int((column > 0.05 * column.max()).sum())
    assert width == pytest.approx(2.0 * expected, abs=2.0)


def test_the_near_end_of_a_tilted_thread_is_wider_and_brighter() -> None:
    lit, discs = _drawn(_thread(0.12, tilt_m=0.1))
    left, right = lit[:, 60:120], lit[:, WIDTH - 120 : WIDTH - 60]
    # Half as far from the lamp again on the right, which alone would be a
    # factor of two; but the far end is also seen more along its length,
    # which puts more silk in each pixel and gives some of it back.
    assert left.max() > 1.3 * right.max()
    assert (left > 0.05 * left.max()).sum() > 1.3 * (right > 0.05 * right.max()).sum()
    assert discs["radius"][discs["x"] < 100].mean() > discs["radius"][discs["x"] > 540].mean()


def test_beading_moves_a_threads_light_and_adds_none() -> None:
    _, bare = _drawn(_thread(0.1))
    beaded = np.mean(
        [
            _glints(
                np.random.default_rng(seed),
                _thread(0.1),
                width=WIDTH, height=HEIGHT, focal=FOCAL, aperture_m=APERTURE_M,
                focus_m=FOCUS_M, gain=1.0, bead_spacing_m=2.0e-3, bead_share=0.8,
                budget=200_000,
            )["weight"].sum()
            for seed in range(20)
        ]
    )  # fmt: skip
    assert beaded == pytest.approx(float(bare["weight"].sum()), rel=0.03)


def test_beads_merge_when_closer_than_the_aperture_and_stand_alone_when_further() -> None:
    def ripple(spacing_m: float) -> float:
        """How far the light along the thread's middle swings, as a share of its mean."""
        lit, _ = _drawn(_thread(0.1), bead_spacing_m=spacing_m, bead_share=1.0)
        along = lit[HEIGHT // 2 - 1 : HEIGHT // 2 + 2, 100 : WIDTH - 100].mean(axis=0)
        return float(along.std() / along.mean())

    assert ripple(0.3 * APERTURE_M) < 0.35
    assert ripple(3.0 * APERTURE_M) > 1.0


def test_a_tight_budget_draws_fewer_discs_holding_the_same_light() -> None:
    full, many = _drawn(_thread(0.1))
    capped, few = _drawn(_thread(0.1), budget=100)
    assert len(few["x"]) < len(many["x"]) / 2
    assert float(capped.sum()) == pytest.approx(float(full.sum()), rel=1e-3)


def test_nothing_nearer_than_the_glass_is_drawn() -> None:
    _, discs = _drawn(_thread(_NEAREST_M * 0.5))
    assert len(discs["x"]) == 0


def test_a_disc_holds_its_weight_whatever_its_size() -> None:
    radius = np.array([0.5, 1.0, 2.5, 7.0, 19.0], dtype=np.float32)
    x = np.array([40.3, 120.0, 200.7, 300.2, 450.5], dtype=np.float32)
    y = np.array([50.5, 60.1, 70.9, 150.0, 240.4], dtype=np.float32)
    weight = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    for index in range(len(radius)):
        one = slice(index, index + 1)
        lit = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        splat_discs(lit, x[one], y[one], radius[one], weight[one])
        # A disc under a pixel across is sampled at too few points to be
        # exact; from a pixel up it is within a few percent.
        tolerance = 0.25 if index == 0 else 0.03
        assert float(lit.sum()) == pytest.approx(float(weight[index]), rel=tolerance)


def test_discs_add_where_they_overlap() -> None:
    x = np.array([100.0, 100.0], dtype=np.float32)
    y = np.array([100.0, 100.0], dtype=np.float32)
    radius = np.array([6.0, 6.0], dtype=np.float32)
    weight = np.array([10.0, 10.0], dtype=np.float32)
    one = np.zeros((200, 200), dtype=np.float32)
    two = np.zeros((200, 200), dtype=np.float32)
    splat_discs(one, x[:1], y[:1], radius[:1], weight[:1])
    splat_discs(two, x, y, radius, weight)
    assert np.allclose(two, 2.0 * one)


def test_an_orb_web_has_a_free_zone_and_loses_spiral_to_damage() -> None:
    def spiral_length(damage: float) -> float:
        """Metres of capture spiral in a web, averaged over a few of them."""
        lengths = []
        for seed in range(6):
            web = _orb_web(np.random.default_rng(seed), 0.15, 24, 5e-3, damage)
            sticky = web["beads"] == 1.0
            lengths.append(np.linalg.norm(web["end"][sticky] - web["start"][sticky], axis=1).sum())
        return float(np.mean(lengths))

    web = _orb_web(np.random.default_rng(0), 0.15, 24, 5e-3, 0.0)
    sticky = web["beads"] == 1.0
    middle = 0.5 * (web["start"][sticky] + web["end"][sticky])
    assert np.linalg.norm(middle, axis=1).min() > 0.12 * 0.15
    assert spiral_length(0.8) < 0.8 * spiral_length(0.0)
    # Several metres of silk in a web a hand and a half across.
    assert 3.0 < spiral_length(0.0) < 30.0


def test_a_web_only_adds_light() -> None:
    image = _night()
    for seed in range(4):
        transform = _web()
        transform.set_random_seed(seed)
        out = transform(image=image)["image"]
        assert (out >= image).all()
        assert (out > image).mean() > 0.01


def test_a_web_is_white_whatever_colour_is_behind_it() -> None:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[..., 0] = 30
    transform = _web(illuminator_gain_range=(1.0, 1.0), distance_range_m=(0.06, 0.06))
    transform.set_random_seed(1)
    out = transform(image=image)["image"].astype(np.int16)
    bright = out.max(axis=2) > 200
    assert bright.any()
    assert np.abs(out[bright][:, 0] - out[bright][:, 2]).max() <= 30


def _with_spider(seed: int, **settings: object) -> tuple[SpiderWeb, dict, tuple[int, int] | None]:
    """A web with its spider near the lens, its parameters, and the pixel the body is over.

    ``None`` for the pixel when the hub, and the spider with it, is outside
    the frame; far enough outside, it is not drawn at all.
    """
    transform = _web(spider_probability=1.0, distance_range_m=(0.07, 0.09), **settings)
    transform.set_random_seed(seed)
    params = transform.get_params_dependent_on_data({"shape": (HEIGHT, WIDTH, 3)}, {})
    spider = params["spider"]
    if spider is None:
        return transform, params, None
    assert spider["patch"].max() > 0.99
    row, column = np.unravel_index(np.argmax(spider["patch"]), spider["patch"].shape)
    y, x = spider["y"] + int(row), spider["x"] + int(column)
    inside = 0 <= y < HEIGHT and 0 <= x < WIDTH
    return transform, params, (y, x) if inside else None


def test_the_spider_clips_under_the_lamp() -> None:
    image = _night()
    found = 0
    for seed in range(30):
        transform, params, body = _with_spider(
            seed, illuminator_gain_range=(1.0, 1.0), beam_falloff_range=(0.0, 0.0)
        )
        if body is not None:
            found += 1
            assert (transform.apply(image, **params)[body] == 255).all()
    assert found >= 3


def test_the_spider_hides_what_is_behind_it() -> None:
    image = _night()
    found = 0
    for seed in range(30):
        # No lamp, so no light of its own: only what it blocks.
        transform, params, body = _with_spider(
            seed, illuminator_gain_range=(0.0, 0.0), glare_range=(0.0, 0.0)
        )
        if body is not None:
            found += 1
            out = transform.apply(image, **params)
            assert (out[body] <= 1).all()
            assert (out <= image).all()
    assert found >= 3


def test_no_spider_without_asking() -> None:
    transform = _web(spider_probability=0.0)
    transform.set_random_seed(0)
    params = transform.get_params_dependent_on_data({"shape": (HEIGHT, WIDTH, 3)}, {})
    assert params["spider"] is None


def test_loose_threads_are_drawn_besides_the_web() -> None:
    def count(strays: int) -> int:
        """Discs drawn for a web with a given number of loose threads."""
        transform = _web(stray_thread_count_range=(strays, strays))
        transform.set_random_seed(3)
        params = transform.get_params_dependent_on_data({"shape": (HEIGHT, WIDTH, 3)}, {})
        return len(params["discs"]["x"])

    assert count(6) > count(0)


def test_the_same_seed_draws_the_same_web() -> None:
    image = _night()
    frames = []
    for _ in range(2):
        transform = SpiderWeb(p=1.0)
        transform.set_random_seed(11)
        frames.append(transform(image=image)["image"])
    assert np.array_equal(frames[0], frames[1])


def test_the_budget_is_kept() -> None:
    transform = _web(max_discs=2000, distance_range_m=(0.05, 0.05))
    transform.set_random_seed(0)
    params = transform.get_params_dependent_on_data({"shape": (HEIGHT, WIDTH, 3)}, {})
    # The beads are never thinned, and every piece keeps one disc of band.
    assert len(params["discs"]["x"]) < 3 * 2000 + 20_000


def test_fast_mode_differs_from_exact_by_the_glare_alone() -> None:
    rng = np.random.default_rng(2)
    image = rng.integers(10, 60, (960, 1280, 3), dtype=np.uint8)
    was = mode.fast()
    try:
        frames = {}
        for flag in (True, False):
            mode.set_fast(flag)
            transform = SpiderWeb(p=1.0, spider_probability=1.0)
            transform.set_random_seed(5)
            frames[flag] = transform(image=image)["image"].astype(np.int16)
        difference = np.abs(frames[True] - frames[False])
        # One level at most over twelve seeds when this was written.
        assert difference.max() <= 3
        assert difference.mean() < 0.1

        for flag in (True, False):
            mode.set_fast(flag)
            transform = SpiderWeb(p=1.0, spider_probability=1.0, glare_range=(0.0, 0.0))
            transform.set_random_seed(5)
            frames[flag] = transform(image=image)["image"].astype(np.int16)
        assert np.array_equal(frames[True], frames[False])
    finally:
        mode.set_fast(was)
