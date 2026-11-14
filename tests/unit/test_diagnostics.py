"""The draw recorder is library code, so it needs the same guarantees as the rest.

Its whole value is that the numbers it reports are the numbers the renderer
actually used — a recorder that drifts from the transform is worse than none,
because it produces confident wrong answers.
"""

from __future__ import annotations

import json

import numpy as np

from surmentations.camera import AcFlicker, ExposureTransient
from surmentations.diagnostics import recording, write_jsonl
from surmentations.lens import SunFlare


def _image() -> np.ndarray:
    """A flat grey frame."""
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


def test_array_summaries_have_the_same_shape_whatever_the_array_size() -> None:
    """A field's schema must not depend on how many particles were drawn.

    Summarising a small array as a bare list and a large one as a dict made
    ``out.flakes.radius.p50`` work on a heavy frame and raise ``TypeError`` on
    a light one — and the light frames are precisely the ones an audit of
    "is severity 1 doing anything at all" has to read.
    """
    from surmentations.diagnostics import _summarise

    keys = {"n", "shape", "min", "p50", "max", "mean"}
    for size in (1, 3, 4, 5, 1000):
        summary = _summarise(np.arange(size, dtype=np.float32))
        assert isinstance(summary, dict), f"size {size} was not summarised as a dict"
        assert keys <= set(summary), f"size {size} is missing {keys - set(summary)}"
        assert summary["n"] == size

    # Small arrays additionally keep their values; large ones do not.
    assert _summarise(np.array([1.0, 2.0]))["values"] == [1.0, 2.0]
    assert "values" not in _summarise(np.arange(1000, dtype=np.float32))


def test_every_depth_conditioned_transform_records_its_draws() -> None:
    """The log's value is that local variables reach it.

    Eight of fifteen transforms recorded no draws at all. For four of them
    that cost nothing — they hand every sampled parameter straight to the
    renderer, so the params dict *is* the draw list. For the four that derive
    something on the way, it meant the one chain worth auditing could not be
    read: fog logged ``exp(-beta z)`` but never ``beta``, so its
    visibility → extinction → transmission derivation had to be run backwards
    out of the transmission field, and could not be closed exactly.
    """
    from surmentations.atmosphere.fog import RandomFog
    from surmentations.depth.provider import SharedDepthProvider
    from surmentations.lens import DirtSoil, HousingOccluders, SpiderWeb
    from surmentations.scene import CastShadows
    from tests.unit.test_randomization import _GroundPlaneDepth

    provider = SharedDepthProvider(_GroundPlaneDepth())
    scene = np.random.default_rng(0).integers(0, 256, (96, 128, 3), dtype=np.uint8)

    expected = {
        "RandomFog": {"beta", "koschmieder_visibility_m"},
        "CastShadows": {"shadow_count", "penumbra_px"},
        "HousingOccluders": {"insect_count", "web_count"},
        "DirtSoil": {"smudge_count", "deposits"},
        "SpiderWeb": {"hub_distance_m", "radial_count", "disc_count", "spider"},
    }
    builders = {
        "RandomFog": lambda: RandomFog(depth_provider=provider, p=1.0),
        "CastShadows": lambda: CastShadows(depth_provider=provider, p=1.0),
        "HousingOccluders": lambda: HousingOccluders(p=1.0, insect_count_range=(2, 4)),
        "DirtSoil": lambda: DirtSoil(p=1.0),
        "SpiderWeb": lambda: SpiderWeb(p=1.0),
    }

    for name, build in builders.items():
        transform = build()
        transform.set_random_seed(0)
        with recording() as log:
            transform(image=scene)
        assert log, f"{name} produced no record at all"
        drew = log[0].get("drew", {})
        missing = expected[name] - set(drew)
        assert not missing, f"{name} does not record {sorted(missing)}"
