"""IrCutFilterFault: infrared added to a colour picture, and what the camera can and cannot undo."""

from __future__ import annotations

import numpy as np
import pytest

from weathermentations.camera import IrCutFilterFault
from weathermentations.camera.ir_cut import unfiltered_side

SHAPE = (120, 160)


def _scene() -> np.ndarray:
    """Foliage on the left, grey paving on the right, blue sky above both."""
    image = np.zeros((*SHAPE, 3), dtype=np.uint8)
    image[:40] = (110, 150, 220)
    image[40:, :80] = (40, 110, 35)
    image[40:, 80:] = (120, 120, 120)
    return image


def _fault(**settings: object) -> IrCutFilterFault:
    """The fault with the filter fully open and every range pinned unless changed."""
    arguments: dict[str, object] = {
        "leak_range": (1.0, 1.0),
        "coverage_range": (1.0, 1.0),
        "white_balance_adaptation_range": (1.0, 1.0),
        "vegetation_gain_range": (0.5, 0.5),
        **settings,
    }
    return IrCutFilterFault(p=1.0, **arguments)


def _run(transform: IrCutFilterFault, image: np.ndarray, seed: int = 0) -> np.ndarray:
    """Apply a transform with a seed, as float."""
    transform.set_random_seed(seed)
    out: np.ndarray = transform(image=image)["image"].astype(np.float32)
    return out


def _saturation(pixels: np.ndarray) -> float:
    """Spread between the channels of a patch, relative to its level."""
    mean = pixels.reshape(-1, 3).mean(axis=0)
    return float((mean.max() - mean.min()) / mean.mean())


def test_colours_wash_out_and_foliage_most_of_all() -> None:
    image = _scene()
    out = _run(_fault(), image)
    foliage_before, foliage_after = image[60:, :60], out[60:, :60]
    assert _saturation(foliage_after) < 0.5 * _saturation(foliage_before)
    # Foliage is bright in the infrared and sky is not: one gains on the other.
    gain_foliage = foliage_after.mean() / foliage_before.mean()
    gain_sky = out[:30].mean() / image[:30].mean()
    assert gain_foliage > 1.3 * gain_sky


def test_foliage_turns_from_green_towards_pink() -> None:
    image = _scene()
    out = _run(_fault(), image)
    before = image[60:, :60].reshape(-1, 3).mean(axis=0)
    after = out[60:, :60].reshape(-1, 3).mean(axis=0)
    # Red was a third of green and blue less. The infrared is added to all
    # three, so both close on green: the leaf is no longer green but pale,
    # between pink and lavender as the white balance leaves it.
    assert after[0] / after[1] > 1.7 * before[0] / before[1]
    assert after[2] / after[1] > 1.7 * before[2] / before[1]


def test_exposure_holds_the_frame_s_level() -> None:
    image = _scene()
    out = _run(_fault(), image)
    assert out.mean() == pytest.approx(float(image.mean()), rel=0.12)


def test_white_balance_takes_the_cast_out_on_average_and_only_on_average() -> None:
    image = _scene()

    def red_over_blue(frame: np.ndarray) -> float:
        """Red over blue, of the whole frame's mean."""
        mean = frame.reshape(-1, 3).mean(axis=0)
        return float(mean[0] / mean[2])

    def out(adaptation: float) -> np.ndarray:
        """The frame from a camera that rebalances this far."""
        return _run(_fault(white_balance_adaptation_range=(adaptation, adaptation)), image)

    before = red_over_blue(image.astype(np.float32))
    assert red_over_blue(out(0.0)) > 1.1 * before
    assert red_over_blue(out(1.0)) == pytest.approx(before, rel=0.08)
    # What it cannot do is balance foliage and paving at once: with the
    # frame neutral again, the foliage is still redder than it was.
    foliage = out(1.0)[60:, :60].reshape(-1, 3).mean(axis=0)
    assert foliage[0] / foliage[1] > 1.3 * 40.0 / 110.0


def test_a_filter_stuck_half_way_leaves_half_the_picture_alone() -> None:
    for share in (0.3, 0.5, 0.8):
        side = unfiltered_side(*SHAPE, share, 0.2, 2.0)
        assert float((side > 0.5).mean()) == pytest.approx(share, abs=0.03)
    assert (unfiltered_side(*SHAPE, 1.0, 0.0, 5.0) == 1.0).all()
    soft = unfiltered_side(*SHAPE, 0.5, 0.0, 30.0)
    assert int(((soft > 0.05) & (soft < 0.95))[60].sum()) > 20


def test_more_leak_moves_the_picture_further() -> None:
    image = _scene()
    changes = [
        float(np.abs(_run(_fault(leak_range=(leak, leak)), image) - image).mean())
        for leak in (0.2, 0.6, 1.2)
    ]
    assert changes == sorted(changes)
    assert changes[0] > 0.5


def test_no_leak_is_no_fault() -> None:
    image = _scene()
    assert np.array_equal(_run(_fault(leak_range=(0.0, 0.0)), image), image.astype(np.float32))
