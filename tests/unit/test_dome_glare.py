"""IrDomeGlare: a ring of light where the dome mirrors the LEDs, and how it is cut and broken up.

The ring's shape is measured on
:func:`~surmentations.lens.dome_glare.ring_glow`; the transform is held
to adding light, and to its two modes agreeing.
"""

from __future__ import annotations

import numpy as np
import pytest

from surmentations import mode
from surmentations.lens import IrDomeGlare
from surmentations.lens.dome_glare import ring_glow
from surmentations.utils.grids import pixel_grid

SIZE = 201
SHAPE = {
    "centre": (100.0, 100.0),
    "scale": 100.0,
    "radius": 0.6,
    "width": 0.08,
    "arc_towards": 0.0,
    "arc_share": 1.0,
    "led_count": 8,
    "led_contrast": 0.0,
    "led_phase": 0.0,
}


def _ring(**changes: float) -> np.ndarray:
    """The ring's glow on a square frame, with some of its shape changed."""
    yy, xx = pixel_grid(SIZE, SIZE)
    return ring_glow(yy, xx, **{**SHAPE, **changes})


def _around(samples: int = 720, **changes: float) -> np.ndarray:
    """The glow along the middle of the band, once round from the far side of angle zero."""
    angle = np.linspace(-np.pi, np.pi, samples, endpoint=False)
    xx = (100.0 + 60.0 * np.cos(angle)).astype(np.float32)
    yy = (100.0 + 60.0 * np.sin(angle)).astype(np.float32)
    return ring_glow(yy, xx, **{**SHAPE, **changes})


def _fixed(**settings: object) -> IrDomeGlare:
    """The transform with every range pinned, so that one thing can be varied."""
    arguments: dict[str, object] = {
        "intensity_range": (0.5, 0.5),
        "ring_radius_range": (0.5, 0.5),
        "ring_width_range": (0.1, 0.1),
        "offset_range": (0.0, 0.0),
        "arc_range": (1.0, 1.0),
        "led_contrast_range": (0.0, 0.0),
        "veil_range": (0.0, 0.0),
        "dirt_range": (0.0, 0.0),
    }
    arguments.update(settings)
    return IrDomeGlare(p=1.0, **arguments)


def _glow(transform: IrDomeGlare, seed: int = 0, shape: tuple[int, int] = (240, 320)) -> np.ndarray:
    """The light a transform adds, for one seed."""
    transform.set_random_seed(seed)
    glow: np.ndarray = transform.get_params_dependent_on_data({"shape": (*shape, 3)}, {})["glow"]
    return glow


def test_the_band_peaks_on_the_ring_and_is_dark_at_its_centre() -> None:
    glow = _ring()
    along = glow[100, 100:]
    assert int(np.argmax(along)) == pytest.approx(60, abs=1)
    assert along[0] < 1e-6
    assert along.max() == pytest.approx(1.0, abs=1e-3)


def test_a_whole_ring_is_seen_all_the_way_round_and_an_arc_is_not() -> None:
    assert (_around() > 0.99).all()
    quarter = _around(arc_share=0.25) > 0.5
    assert quarter.mean() == pytest.approx(0.28, abs=0.03)
    # And it is the part towards `arc_towards` that is kept: angle zero is
    # the middle of the sampled turn.
    assert quarter[len(quarter) // 2]
    assert not quarter[0]


def test_the_leds_show_as_that_many_lobes() -> None:
    def lobes(count: int) -> int:
        """Peaks in the glow round a ring with distinct LEDs."""
        glow = _around(led_count=count, led_contrast=0.8)
        return int(((glow > np.roll(glow, 1)) & (glow >= np.roll(glow, -1)) & (glow > 0.6)).sum())

    assert lobes(6) == 6
    assert lobes(12) == 12
    even = _around(led_contrast=0.0)
    assert float(even.max() - even.min()) < 0.02


def test_glare_only_adds_light() -> None:
    image = np.random.default_rng(0).integers(10, 60, (240, 320, 3), dtype=np.uint8)
    for seed in range(5):
        transform = IrDomeGlare(p=1.0)
        transform.set_random_seed(seed)
        out = transform(image=image)["image"]
        assert (out >= image).all()
        assert (out > image).mean() > 0.02


def test_the_veil_lifts_every_pixel_by_itself() -> None:
    without = _glow(_fixed())
    veiled = _glow(_fixed(veil_range=(0.07, 0.07)))
    assert np.allclose(veiled - without, 0.07, atol=1e-6)


def test_dirt_breaks_the_band_up_and_adds_no_light() -> None:
    clean = _glow(_fixed())
    dirty = [_glow(_fixed(dirt_range=(0.8, 0.8)), seed) for seed in range(8)]
    assert float(np.mean([d.sum() for d in dirty])) == pytest.approx(float(clean.sum()), rel=0.12)
    band = clean > 0.4
    assert float((dirty[0][band] / clean[band]).std()) > 0.1


def test_a_tilted_lens_moves_the_ring_off_centre() -> None:
    def centroid(offset: float) -> np.ndarray:
        """The glow's centre of light, in pixels from the frame's centre."""
        glow = _glow(_fixed(offset_range=(offset, offset)), seed=3)
        rows, columns = np.indices(glow.shape)
        total = glow.sum()
        return np.array(
            [(rows * glow).sum() / total - 119.5, (columns * glow).sum() / total - 159.5]
        )

    assert float(np.linalg.norm(centroid(0.0))) < 1.0
    assert float(np.linalg.norm(centroid(0.4))) > 20.0


def test_fast_mode_s_coarse_grid_is_within_a_level() -> None:
    image = np.random.default_rng(1).integers(10, 60, (480, 640, 3), dtype=np.uint8)
    was = mode.fast()
    try:
        for seed in range(4):
            frames = {}
            for flag in (True, False):
                mode.set_fast(flag)
                transform = IrDomeGlare(p=1.0, led_contrast_range=(0.7, 0.7))
                transform.set_random_seed(seed)
                frames[flag] = transform(image=image)["image"].astype(np.int16)
            assert np.abs(frames[True] - frames[False]).max() <= 2
    finally:
        mode.set_fast(was)
