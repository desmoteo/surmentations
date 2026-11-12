"""LensTamper: paint, tape or a cover, each hiding the scene in its own way."""

from __future__ import annotations

import numpy as np
import pytest

from weathermentations.lens import LensTamper
from weathermentations.lens.tamper import spray_mask, tape_mask

SHAPE = (240, 320)


def _scene() -> np.ndarray:
    """A daylit frame with detail everywhere."""
    rng = np.random.default_rng(0)
    return rng.integers(40, 220, (*SHAPE, 3), dtype=np.uint8)


def _detail(image: np.ndarray) -> float:
    """Mean difference between neighbouring pixels: what is left of the scene."""
    return float(np.abs(np.diff(image.astype(np.float32), axis=1)).mean())


def test_paint_and_tape_cover_about_what_was_asked() -> None:
    for mask in (spray_mask, tape_mask):
        for coverage in (0.2, 0.6):
            covered = [
                float(mask(np.random.default_rng(seed), *SHAPE, coverage).mean())
                for seed in range(12)
            ]
            assert np.mean(covered) == pytest.approx(coverage, rel=0.45)


def test_tape_is_a_straight_strip_across_the_whole_frame() -> None:
    for seed in range(6):
        mask = tape_mask(np.random.default_rng(seed), *SHAPE, 0.2) > 0.5
        # A strip that crosses the frame touches two of its sides.
        sides = [mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()]
        assert sum(sides) >= 2


def test_spray_is_opaque_at_its_core_and_speckled_at_its_edge() -> None:
    mask = spray_mask(np.random.default_rng(1), 480, 640, 0.3)
    blocks = mask[: 480 // 16 * 16, : 640 // 16 * 16].reshape(30, 16, 40, 16).mean(axis=(1, 3))
    assert (blocks > 0.98).any()
    assert ((blocks > 0.2) & (blocks < 0.8)).sum() >= 8


def test_a_cover_takes_the_whole_frame_and_leaves_no_image() -> None:
    image = _scene()
    transform = LensTamper(p=1.0, kinds=("cover",))
    transform.set_random_seed(0)
    out = transform(image=image)["image"]
    assert _detail(out) < 0.05 * _detail(image)


def test_what_is_not_covered_is_untouched() -> None:
    image = _scene()
    for kind in ("spray", "tape"):
        transform = LensTamper(p=1.0, kinds=(kind,), coverage_range=(0.15, 0.15))
        transform.set_random_seed(2)
        params = transform.get_params_dependent_on_data({"shape": image.shape}, {})
        out = transform.apply(image, **params)
        clear = params["cover"] < 1e-4
        assert clear.mean() > 0.2
        assert np.array_equal(out[clear], image[clear])


def test_only_the_kinds_asked_for_are_drawn() -> None:
    from weathermentations.diagnostics import recording

    image = _scene()
    seen = set()
    for seed in range(12):
        transform = LensTamper(p=1.0, kinds=("tape", "cover"))
        transform.set_random_seed(seed)
        with recording() as log:
            transform(image=image)
        seen.add(log[0]["drew"]["kind"])
    assert seen == {"tape", "cover"}
    with pytest.raises(ValueError, match="kinds"):
        LensTamper(kinds=("glue",))


def test_under_the_lamp_pale_paint_glows_and_black_paint_does_not() -> None:
    night = np.full((*SHAPE, 3), 15, dtype=np.uint8)
    transform = LensTamper(p=1.0, kinds=("cover",), glow_range=(2.0, 2.0), opacity_range=(1.0, 1.0))
    transform.set_random_seed(0)
    params = transform.get_params_dependent_on_data({"shape": night.shape}, {})
    white = transform.apply(night, **{**params, "albedo": np.array([0.75, 0.75, 0.72], np.float32)})
    black = transform.apply(night, **{**params, "albedo": np.array([0.02, 0.02, 0.02], np.float32)})
    assert white.mean() > 240
    assert black.mean() < 70


def test_less_opaque_lets_more_of_the_scene_through() -> None:
    image = _scene()
    details = []
    for opacity in (1.0, 0.6):
        transform = LensTamper(
            p=1.0, kinds=("tape",), coverage_range=(1.0, 1.0), opacity_range=(opacity, opacity)
        )
        transform.set_random_seed(4)
        details.append(_detail(transform(image=image)["image"]))
    assert details[1] > 1.5 * details[0]
