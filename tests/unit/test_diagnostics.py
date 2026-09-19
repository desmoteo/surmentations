"""The draw recorder is library code, so it needs the same guarantees as the rest.

Its whole value is that the numbers it reports are the numbers the renderer
actually used — a recorder that drifts from the transform is worse than none,
because it produces confident wrong answers.
"""

from __future__ import annotations

import json

import numpy as np

from weathermentations.camera import AcFlicker, ExposureTransient
from weathermentations.diagnostics import recording, write_jsonl
from weathermentations.lens import SunFlare


def _image() -> np.ndarray:
    return np.full((64, 96, 3), 120, dtype=np.uint8)


def test_records_one_entry_per_transform_call() -> None:
    transform = AcFlicker(p=1.0)
    with recording() as log:
        for seed in range(3):
            transform.set_random_seed(seed)
            transform(image=_image())
    assert len(log) == 3
    assert {r["t"] for r in log} == {"AcFlicker"}


def test_records_nothing_outside_the_context() -> None:
    transform = AcFlicker(p=1.0)
    transform.set_random_seed(0)
    transform(image=_image())  # not recorded

    with recording() as log:
        transform.set_random_seed(1)
        transform(image=_image())
    assert len(log) == 1

    transform.set_random_seed(2)
    transform(image=_image())  # not recorded either
    assert len(log) == 1


def test_recorded_draws_match_what_the_renderer_used() -> None:
    """The point of the whole module: a recorded value must be the value that
    actually drove the render, not a re-draw from the same distribution."""
    transform = ExposureTransient(p=1.0)
    transform.set_random_seed(7)
    with recording() as log:
        transform(image=_image())

    record = log[0]
    stops = record["drew"]["exposure_error_stops"]
    gain = record["out"]["exposure_gain"]
    assert gain == round(2.0**stops, 6)


def test_draws_are_reported_inside_their_declared_ranges() -> None:
    transform = ExposureTransient(p=1.0)
    with recording() as log:
        for seed in range(12):
            transform.set_random_seed(seed)
            transform(image=_image())

    for record in log:
        low, high = record["init"]["exposure_error_stops_range"]
        assert low <= record["drew"]["exposure_error_stops"] <= high


def test_large_arrays_are_summarised_not_stored() -> None:
    """A parameter dictionary holds megabytes of masks and per-particle arrays.
    Storing them would make the log unreadable and unusable; the summary is
    what makes a sweep of thousands of frames analysable at all."""
    transform = SunFlare(p=1.0, flare_position=(0.5, 0.2))
    transform.set_random_seed(0)
    with recording() as log:
        transform(image=_image())

    encoded = json.dumps(log[0])
    assert len(encoded) < 4000, "a record should stay small enough to read in bulk"

    ghosts = log[0]["out"]["ghosts"]
    # Long per-item lists reduce to a count rather than their contents.
    assert isinstance(ghosts, dict) and "n" in ghosts


def test_write_jsonl_round_trips(tmp_path) -> None:
    transform = AcFlicker(p=1.0)
    with recording() as log:
        transform.set_random_seed(0)
        transform(image=_image())

    path = tmp_path / "draws.jsonl"
    write_jsonl(log, path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == len(log)
    assert json.loads(lines[0])["t"] == "AcFlicker"


def test_recording_is_reentrant_and_restores_state() -> None:
    transform = AcFlicker(p=1.0)
    with recording() as outer:
        transform.set_random_seed(0)
        transform(image=_image())
        with recording() as inner:
            transform.set_random_seed(1)
            transform(image=_image())
        assert len(inner) == 1
        # The inner block must not have stolen the outer block's records.
        assert len(outer) == 1
        transform.set_random_seed(2)
        transform(image=_image())
    assert len(outer) == 2
