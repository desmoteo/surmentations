"""OnScreenOverlay: lettering that stays legible and in the frame, and masks that hide boxes."""

from __future__ import annotations

import re

import albumentations as A
import numpy as np
import pytest

from surmentations.camera import OnScreenOverlay
from surmentations.camera.overlay import _GLYPHS, GLYPH_ROWS, render_text, timestamp
from surmentations.presets.pipeline import physical_order, stage_of


def _scene(height: int = 240, width: int = 320) -> np.ndarray:
    """A mid-grey frame with a little noise."""
    return np.random.default_rng(0).integers(90, 140, (height, width, 3), dtype=np.uint8)


def _overlay(**settings: object) -> OnScreenOverlay:
    """Text always, no masks, unless changed."""
    arguments: dict[str, object] = {
        "timestamp_probability": 1.0,
        "label_probability": 1.0,
        "privacy_mask_count_range": (0, 0),
        **settings,
    }
    return OnScreenOverlay(p=1.0, **arguments)


def test_every_character_is_five_by_seven_and_no_two_are_alike() -> None:
    shapes = {glyph.shape for glyph in _GLYPHS.values()}
    assert shapes == {(7, 5)}
    drawn = {character: glyph.tobytes() for character, glyph in _GLYPHS.items()}
    assert len(set(drawn.values())) == len(drawn)
    assert all(glyph.any() for character, glyph in _GLYPHS.items() if character != " ")


def test_text_is_drawn_dot_for_dot_at_any_scale() -> None:
    one = render_text("CAM 07", 1)
    assert one.shape == (7, 6 * 6 - 1)
    three = render_text("CAM 07", 3)
    assert three.shape == (21, 3 * 35)
    assert np.array_equal(three[::3, ::3], one)
    assert np.array_equal(render_text("cam 07", 1), one)
    assert not render_text("#", 1).any()


def test_a_timestamp_is_a_date_and_a_time() -> None:
    rng = np.random.default_rng(0)
    for _ in range(40):
        text = timestamp(rng)
        assert re.search(r"\d{2}:\d{2}:\d{2}$", text)
        assert re.search(r"\d{4}", text)
        assert all(character in _GLYPHS for character in text)


@pytest.mark.parametrize("size", [(96, 128), (256, 256), (480, 640), (1080, 1920)])
def test_lettering_is_whole_dots_and_inside_the_frame_at_every_size(size: tuple[int, int]) -> None:
    for seed in range(6):
        overlay = _overlay()
        overlay.set_random_seed(seed)
        params = overlay.get_params_dependent_on_data({"shape": (*size, 3)}, {})
        assert isinstance(params["scale"], int) and params["scale"] >= 1
        image = np.full((*size, 3), 120, dtype=np.uint8)
        out = overlay.apply(image, **params)
        assert out.shape == image.shape
        changed = np.argwhere((out != image).any(axis=2))
        assert len(changed) > 20
        # In the corners: nothing is written across the middle of the frame.
        middle = out[size[0] // 3 : 2 * size[0] // 3]
        assert np.array_equal(middle, image[size[0] // 3 : 2 * size[0] // 3])


def test_the_largest_lettering_still_fits_across_the_frame() -> None:
    for size in ((240, 320), (480, 640), (720, 1280)):
        overlay = _overlay(text_height_range=(0.2, 0.2))
        for seed in range(8):
            overlay.set_random_seed(seed)
            params = overlay.get_params_dependent_on_data({"shape": (*size, 3)}, {})
            for text, _corner, _style in params["lines"]:
                assert render_text(text, params["scale"]).shape[1] <= size[1]


def test_the_text_can_be_read_back_from_the_frame() -> None:
    image = np.full((256, 256, 3), 110, dtype=np.uint8)
    overlay = _overlay(label_probability=0.0)
    overlay.set_random_seed(1)
    params = overlay.get_params_dependent_on_data({"shape": image.shape}, {})
    out = overlay.apply(image, **params)
    text = params["lines"][0][0]
    scale = params["scale"]
    white = out[..., 0] > 220
    rows, columns = np.nonzero(white)
    found = white[rows.min() : rows.min() + GLYPH_ROWS * scale, columns.min() : columns.max() + 1]
    expected = render_text(text, scale)
    first = np.nonzero(expected.any(axis=0))[0]
    assert np.array_equal(found, expected[:, first.min() : first.max() + 1])


def test_a_privacy_mask_hides_what_is_under_it() -> None:
    image = _scene()
    overlay = _overlay(
        timestamp_probability=0.0, label_probability=0.0, privacy_mask_count_range=(1, 1)
    )
    for seed in range(8):
        overlay.set_random_seed(seed)
        params = overlay.get_params_dependent_on_data({"shape": image.shape}, {})
        (left, top, right, bottom), fill, block = params["masks"][0]
        out = overlay.apply(image, **params)
        inside = out[top:bottom, left:right]
        outside = out.copy()
        outside[top:bottom, left:right] = image[top:bottom, left:right]
        assert np.array_equal(outside, image)
        if fill is None:
            # Blocks: every pixel of a block is the same.
            corner = inside[:block, :block]
            assert (corner == corner[0, 0]).all()
        else:
            assert (inside == round(fill * 255)).all()


def test_a_box_under_a_mask_is_dropped_and_one_beside_it_is_kept() -> None:
    image = _scene()
    overlay = _overlay(
        timestamp_probability=0.0,
        label_probability=0.0,
        privacy_mask_count_range=(1, 1),
        privacy_mask_size_range=(0.4, 0.4),
    )
    overlay.set_random_seed(2)
    params = overlay.get_params_dependent_on_data({"shape": image.shape}, {})
    left, top, right, bottom = params["masks"][0][0]
    height, width = image.shape[:2]
    under = [(left + 4) / width, (top + 4) / height, (right - 4) / width, (bottom - 4) / height, 0]
    half = [(left - 40) / width, top / height, (left + 40) / width, bottom / height, 1]
    kept = overlay.apply_to_bboxes(np.array([under, half]), shape=image.shape, **params)
    assert kept.shape == (1, 5) and kept[0, 4] == 1
    lenient = _overlay(hidden_box_coverage=1.5)
    assert len(lenient.apply_to_bboxes(np.array([under, half]), shape=image.shape, **params)) == 2


def test_boxes_survive_a_whole_pipeline_unless_hidden() -> None:
    image = _scene()
    pipeline = A.Compose(
        [_overlay()],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        seed=0,
    )
    out = pipeline(image=image, bboxes=[(100, 100, 160, 150)], labels=["car"])
    assert [tuple(round(v) for v in box) for box in out["bboxes"]] == [(100, 100, 160, 150)]
    assert out["labels"] == ["car"]


def test_who_draws_it_decides_where_in_the_chain_it_goes() -> None:
    from surmentations.camera import AnalogVideoArtifacts, SensorNoise

    camera, recorder = OnScreenOverlay(drawn_by="camera"), OnScreenOverlay(drawn_by="recorder")
    link, sensor = AnalogVideoArtifacts(), SensorNoise()
    assert stage_of(sensor) < stage_of(camera) < stage_of(link) < stage_of(recorder)
    ordered = physical_order([recorder, link, camera, sensor])
    assert ordered == [sensor, camera, link, recorder]


def test_nothing_to_draw_changes_nothing() -> None:
    image = _scene()
    overlay = _overlay(timestamp_probability=0.0, label_probability=0.0)
    overlay.set_random_seed(0)
    assert np.array_equal(overlay(image=image)["image"], image)
