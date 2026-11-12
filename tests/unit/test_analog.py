"""AnalogVideoArtifacts: each fault of a composite link, on its own, measured."""

from __future__ import annotations

import numpy as np
import pytest

from weathermentations.camera import AnalogVideoArtifacts
from weathermentations.camera.analog import along_lines, composite_round_trip, displace_lines

SHAPE = (96, 128)
NOTHING: dict[str, object] = {
    "luma_bandwidth_range": (1.0, 1.0),
    "peaking_range": (0.0, 0.0),
    "chroma_bandwidth_range": (1.0, 1.0),
    "chroma_delay_range": (0.0, 0.0),
    "crosstalk_range": (0.0, 0.0),
    "ghost_strength_range": (0.0, 0.0),
    "hum_depth_range": (0.0, 0.0),
    "noise_range": (0.0, 0.0),
    "line_jitter_range": (0.0, 0.0),
    "comb_shift_range": (0.0, 0.0),
}


def _link(**faults: object) -> AnalogVideoArtifacts:
    """A link with every fault off but the ones given."""
    return AnalogVideoArtifacts(p=1.0, **{**NOTHING, **faults})


def _send(link: AnalogVideoArtifacts, image: np.ndarray, seed: int = 0) -> np.ndarray:
    """A frame through a link, as float."""
    link.set_random_seed(seed)
    out: np.ndarray = link(image=image)["image"].astype(np.float32)
    return out


def _edges() -> np.ndarray:
    """A grey frame with one vertical and one horizontal step in brightness."""
    image = np.full((*SHAPE, 3), 60, dtype=np.uint8)
    image[:, 64:] = 190
    image[48:, :32] = 190
    return image


def _colour_edge() -> np.ndarray:
    """Red on the left, cyan on the right, of the same brightness."""
    image = np.zeros((*SHAPE, 3), dtype=np.uint8)
    image[:, :64] = (200, 60, 60)
    image[:, 64:] = (40, 130, 130)
    return image


def test_a_perfect_link_changes_almost_nothing() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (*SHAPE, 3), dtype=np.uint8)
    out = _send(_link(), image)
    # Colour still goes through its own narrow filter, at its widest here.
    assert float(np.abs(out - image).mean()) < 6.0
    assert np.abs(_send(_link(), _edges()) - _edges()).max() <= 1.0


def test_the_cable_softens_along_the_line_and_not_down_the_frame() -> None:
    out = _send(_link(luma_bandwidth_range=(0.2, 0.2)), _edges())[..., 0]
    across = np.abs(np.diff(out[20], axis=0)).max()
    down = np.abs(np.diff(out[:, 10], axis=0)).max()
    assert across < 0.5 * 130.0
    assert down == pytest.approx(130.0, abs=2.0)


def test_peaking_rings_beside_an_edge() -> None:
    plain = _send(_link(luma_bandwidth_range=(0.4, 0.4)), _edges())[20, :, 0]
    peaked = _send(_link(luma_bandwidth_range=(0.4, 0.4), peaking_range=(1.0, 1.0)), _edges())[
        20, :, 0
    ]
    assert plain.max() <= 191.0
    assert peaked.max() > 195.0 and peaked.min() < 55.0


def test_colour_is_smeared_far_more_than_brightness() -> None:
    out = _send(_link(chroma_bandwidth_range=(0.05, 0.05)), _colour_edge())
    red_minus_green = out[20, :, 0] - out[20, :, 1]
    width = int(((red_minus_green < 130.0) & (red_minus_green > -80.0)).sum())
    assert width >= 6


def test_colour_arrives_late() -> None:
    def crossing(delay: float) -> int:
        """Column where red stops leading green, on the middle line."""
        link = _link(chroma_bandwidth_range=(0.3, 0.3), chroma_delay_range=(delay, delay))
        out = _send(link, _colour_edge())
        return int(np.argmax(out[48, :, 0] < out[48, :, 1]))

    assert crossing(0.04) - crossing(0.0) == pytest.approx(0.04 * 128, abs=1.5)


def test_fine_brightness_detail_is_taken_for_colour() -> None:
    # Grey stripes at the subcarrier's own pitch: four pixels to a cycle.
    across = np.arange(128)
    stripes = np.where(across % 4 < 2, 0.25, 0.75).astype(np.float32)
    yuv = np.zeros((*SHAPE, 3), dtype=np.float32)
    yuv[..., 0] = stripes[None, :]
    decoded = composite_round_trip(yuv, 0.25, 4.0, 0.0)
    assert float(np.abs(decoded[..., 1:]).max()) > 0.1
    flat = np.zeros_like(yuv)
    flat[..., 0] = 0.5
    assert float(np.abs(composite_round_trip(flat, 0.25, 4.0, 0.0)[4:-4, 8:-8, 1:]).max()) < 0.03


def test_a_colour_edge_leaves_dots_in_the_brightness_and_crosstalk_scales_them() -> None:
    image = _colour_edge()

    def dots(crosstalk: float) -> float:
        """Line-to-line change of brightness along the edge: the dots alternate."""
        link = _link(chroma_bandwidth_range=(0.1, 0.1), crosstalk_range=(crosstalk, crosstalk))
        luma = _send(link, image) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return float(np.abs(np.diff(luma[:, 56:72], axis=0)).mean())

    assert dots(0.0) < 0.5
    assert dots(1.0) > 3.0
    assert dots(0.5) == pytest.approx(0.5 * dots(1.0), rel=0.25)


def test_a_reflection_is_a_faint_copy_to_the_right() -> None:
    link = _link(ghost_strength_range=(0.3, 0.3), ghost_delay_range=(0.1, 0.1))
    out = _send(link, _edges())[20, :, 0]
    # The step at column 64 again, a tenth of the width on, 0.3 as high.
    second = float(out[80] - out[72])
    assert second == pytest.approx(0.3 * 130.0, rel=0.2)


def test_hum_is_a_bar_down_the_frame_and_noise_is_not() -> None:
    grey = np.full((*SHAPE, 3), 128, dtype=np.uint8)
    hum = _send(_link(hum_depth_range=(0.1, 0.1), hum_cycles_range=(1.0, 1.0)), grey)[..., 0]
    by_line = hum.mean(axis=1)
    assert by_line.max() - by_line.min() == pytest.approx(2 * 0.1 * 255, rel=0.1)
    assert float(hum.std(axis=1).max()) < 0.6
    noisy = _send(_link(noise_range=(0.05, 0.05)), grey)[..., 0]
    assert noisy.std() == pytest.approx(0.05 * 255, rel=0.15)


def test_noise_on_a_narrow_link_is_smeared_along_the_line() -> None:
    grey = np.full((*SHAPE, 3), 128, dtype=np.uint8)
    noisy = _send(_link(noise_range=(0.05, 0.05), luma_bandwidth_range=(0.2, 0.2)), grey)[..., 0]
    along = float(np.abs(np.diff(noisy, axis=1)).mean())
    down = float(np.abs(np.diff(noisy, axis=0)).mean())
    assert along < 0.6 * down


def test_the_two_fields_part_by_the_shift_asked_for() -> None:
    out = _send(_link(comb_shift_range=(0.03, 0.03)), _edges())[..., 0]

    def step(line: int) -> int:
        """Column of the vertical edge on a line."""
        return int(np.argmax(out[line] > 125.0))

    assert abs(step(21) - step(20)) == pytest.approx(0.03 * 128, abs=1.0)
    assert step(20) == step(22)


def test_lines_are_moved_by_what_they_are_told() -> None:
    ramp = np.broadcast_to(np.arange(64, dtype=np.float32)[None, :], (8, 64)).copy()
    shift = np.array([0, 1, 2, 3, -1, -2, 0.5, 0], dtype=np.float32)
    moved = displace_lines(ramp, shift)
    assert np.allclose(moved[:, 20:40], ramp[:, 20:40] - shift[:, None], atol=1e-4)
    assert np.array_equal(along_lines(ramp, 0.0), ramp)
