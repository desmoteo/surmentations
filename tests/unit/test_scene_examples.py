"""The per-scene examples have to stay in step with the scenes and the docs.

Nothing here renders. These check the bookkeeping that goes stale without
anyone noticing: a photograph added without its licence, a page that links an
image nobody generated, a README pointing at a file that has moved.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from docs.generate_examples import EFFECTS
from docs.generate_scene_examples import (
    FOG_VISIBILITY,
    KOSCHMIEDER,
    PIPELINE_DRAWS,
    SCENES,
    Scene,
    _overrides,
    _sources,
    _why_not,
)

REPO = Path(__file__).resolve().parents[2]
SCENES_OUT = REPO / "docs" / "examples" / "scenes"

# Markdown the repository publishes, and so has to keep its links alive.
PAGES = sorted(
    [
        REPO / "README.md",
        REPO / "sample_scenes" / "SOURCES.md",
        *(REPO / "docs").glob("*.md"),
        *SCENES_OUT.rglob("README.md"),
    ]
)
_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def test_every_photograph_is_configured_and_credited() -> None:
    """A scene needs three things: the file, what the transforms are told
    about it, and who made it under which licence. The last is not optional
    for the CC BY and BY-SA ones, and their derivatives are published here."""
    on_disk = {path.stem for path in (REPO / "sample_scenes").glob("*.jpg")}
    assert on_disk == set(SCENES)
    sources = _sources()
    assert set(sources) == set(SCENES)
    for name, source in sources.items():
        assert source["author"] and source["licence"], name
        assert source["url"].startswith("https://commons.wikimedia.org/wiki/File:"), name


@pytest.mark.parametrize("name", sorted(SCENES))
def test_a_scene_page_shows_what_was_rendered(name: str) -> None:
    """Each applicable effect has its image, each skipped one is named with
    its reason, and the credit is on the page."""
    scene, folder = SCENES[name], SCENES_OUT / name
    page = (folder / "README.md").read_text()
    source = _sources()[name]
    assert source["author"] in page and source["licence"] in page

    assert (folder / "original.jpg").is_file()
    for effect in EFFECTS:
        image = folder / "effects" / f"{effect}.jpg"
        if _why_not(effect, scene) is None:
            assert image.is_file(), f"{name}: {effect} was not rendered"
            assert f"effects/{effect}.jpg" in page
        else:
            assert not image.exists(), f"{name}: {effect} does not apply but has an image"
            assert f"`{effect}`" in page, f"{name}: {effect} was skipped without saying so"
    assert len(list((folder / "pipelines").glob("*.jpg"))) == PIPELINE_DRAWS
    assert not [path for path in folder.rglob("*.jpg") if path.name not in page], (
        f"{name}: an image on disk is not shown on the page"
    )


@pytest.mark.parametrize("page", PAGES, ids=lambda path: str(path.relative_to(REPO)))
def test_published_links_resolve(page: Path) -> None:
    broken = []
    for target in _LINK.findall(page.read_text()):
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            continue
        path = (page.parent / target.split("#")[0]).resolve()
        if not path.exists():
            broken.append(target)
    assert not broken, f"{page.relative_to(REPO)} links to files that do not exist: {broken}"


def test_effects_are_left_out_where_they_do_not_apply() -> None:
    day = Scene((5.0, 100.0), 10.0)
    night = Scene((5.0, 100.0), 10.0, night=True)
    indoors = Scene((2.0, 40.0), 5.0, sheltered=True)

    far = Scene((20.0, 2000.0), 5.0)

    # A hundred metres is too short a path for the air to show.
    assert {e for e in EFFECTS if _why_not(e, day)} == {"night_fog", "source_glare", "heat_shimmer"}
    assert {e for e in EFFECTS if _why_not(e, far)} == {"night_fog", "source_glare"}
    assert {e for e in EFFECTS if _why_not(e, night)} == {
        "sun_flare", "cast_shadows", "fog", "ir_cut_filter_fault", "heat_shimmer",
    }  # fmt: skip
    assert {e for e in EFFECTS if _why_not(e, indoors)} == {
        "sun_flare", "cast_shadows", "rain", "snow", "night_rain", "night_snow",
        "night_fog", "source_glare", "wet_ground", "snow_cover",
        "frost_on_glass", "snow_on_housing", "foliage_intrusion", "night_foliage", "ground_fog",
        "heat_shimmer",
    }  # fmt: skip


def test_weather_is_scaled_to_the_scene() -> None:
    """Severity fixes extinction per metre. Over a three-kilometre view that
    is a blank frame, so fog is set by visibility and the precipitation
    veils are capped — and a short scene is left exactly as severity has it."""
    valley = Scene((25.0, 3000.0), 0.0)
    low, high = _overrides("fog", valley)["beta_range"]
    assert low < KOSCHMIEDER / (FOG_VISIBILITY * 3000.0) < high
    assert _overrides("fog", valley)["depth_scale"] is None

    capped = _overrides("rain", valley)["veil_beta_range"]
    assert capped[1] == pytest.approx(KOSCHMIEDER / (1.5 * 3000.0))
    assert "veil_beta_range" in _overrides("snow", valley)

    crossing = Scene((12.0, 30.0), 70.0, sky_in_frame=False)
    assert "veil_beta_range" not in _overrides("rain", crossing)
    assert _overrides("rain", crossing)["camera_pitch_range_deg"] == (70.0, 70.0)
    # No sky in frame, so the dirt is lit by more than the camera can see.
    assert _overrides("dirt_soil", crossing)["ambient_light_range"][0] > 1.0
    assert "ambient_light_range" not in _overrides("dirt_soil", valley)
