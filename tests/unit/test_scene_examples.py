"""The per-scene examples have to stay in step with the scenes and the docs.

Nothing here renders. These check the bookkeeping that goes stale without
anyone noticing: a photograph added without its licence, a page that links an
image nobody generated, a README pointing at a file that has moved.

The pictures are not in the repository: CI renders and publishes them, and
the pages link to them there. So the pages are checked always, and the
pictures on disk only where the gallery has been rendered, which the gallery
workflow does before it runs these.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from docs.gallery import local
from docs.generate_examples import EFFECTS
from docs.generate_scene_examples import (
    COMPARED,
    FOG_VISIBILITY,
    PIPELINE_DRAWS,
    RAIN_VISIBILITY_FLOOR,
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
        *(REPO / "docs" / "transforms").glob("*.md"),
        *SCENES_OUT.rglob("README.md"),
    ]
)
_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def _rendered(folder: Path) -> bool:
    """Whether the pictures of a scene's folder have been rendered here."""
    return (folder / "original.webp").is_file()


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

    rendered = _rendered(folder)
    assert "/original.webp)" in page
    for effect in EFFECTS:
        image = folder / "effects" / f"{effect}.webp"
        if _why_not(effect, scene) is None:
            assert f"effects/{effect}.webp)" in page, f"{name}: {effect} is not on the page"
            assert image.is_file() or not rendered, f"{name}: {effect} was not rendered"
        else:
            assert f"effects/{effect}.webp)" not in page
            assert not image.exists(), f"{name}: {effect} does not apply but has an image"
            assert f"`{effect}`" in page, f"{name}: {effect} was skipped without saying so"
    assert len(re.findall(r"/pipelines/\w+_\d+\.webp\)", page)) == PIPELINE_DRAWS
    # The pipeline that takes a depth provider is drawn a second time over
    # one that read the scene; the other has nothing to compare.
    twice = "surveillance" if not (scene.night or scene.sheltered) else None
    again = re.findall(r"/pipelines_aware/\w+_\d+\.webp\)", page)
    assert len(again) == (PIPELINE_DRAWS if twice else 0)
    if rendered:
        assert len(list((folder / "pipelines_aware").glob("*.webp"))) == len(again)
        assert not [path for path in folder.rglob("*.webp") if path.name not in page], (
            f"{name}: an image on disk is not shown on the page"
        )


@pytest.mark.parametrize("name", sorted(SCENES))
def test_a_scene_page_compares_what_the_provider_s_reading_changes(name: str) -> None:
    """What a provider with labels and geometry read off the scene is shown,
    and every effect that depends on it is there twice: an effect's second
    picture exists exactly when the effect applies and reads the scene."""
    scene, folder = SCENES[name], SCENES_OUT / name
    page = (folder / "README.md").read_text()
    assert "/reading.webp)" in page
    assert (folder / "reading.webp").is_file() or not _rendered(folder)
    for row in ("Field of view", "Horizon", "Camera pitch below level", "Camera height", "Depth"):
        assert f"| {row} |" in page, f"{name}: the page does not say what was read for {row!r}"
    for effect in EFFECTS:
        image = folder / "aware" / f"{effect}.webp"
        if effect in COMPARED and _why_not(effect, scene) is None:
            assert image.is_file() or not _rendered(folder), (
                f"{name}: {effect} was not rendered a second time"
            )
            assert f"aware/{effect}.webp)" in page and f"`{effect}` without" in page
        else:
            assert f"aware/{effect}.webp)" not in page
            assert not image.exists(), f"{name}: {effect} has a second picture it should not"
    assert set(COMPARED) <= set(EFFECTS)


@pytest.mark.parametrize("page", PAGES, ids=lambda path: str(path.relative_to(REPO)))
def test_published_links_resolve(page: Path) -> None:
    """A link to a file of the repository finds it. A link to a published
    picture is a WebP, and finds it too where the gallery has been rendered."""
    broken = []
    for target in _LINK.findall(page.read_text()):
        published = local(target.split("#")[0])
        if published is not None and published.suffix:
            # The folder's pictures are all rendered together, or none are.
            rendered = any(published.parent.glob("*.webp")) and (
                published.parent.name != "examples"
                or (published.parent / "00_base_scene.webp").is_file()
            )
            if published.suffix != ".webp" or (rendered and not published.is_file()):
                broken.append(target)
            continue
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            continue
        if re.search(r"\.(png|jpe?g|webp)$", target, re.IGNORECASE) and "sample_scenes" not in str(
            (page.parent / target).resolve()
        ):
            # A picture the documentation draws is published, not kept here.
            broken.append(target)
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
        "night_color",
    }  # fmt: skip
    assert {e for e in EFFECTS if _why_not(e, indoors)} == {
        "sun_flare", "cast_shadows", "rain", "snow", "night_rain", "night_snow",
        "night_fog", "source_glare", "wet_ground", "snow_cover",
        "frost_on_glass", "snow_on_housing", "foliage_intrusion", "night_foliage", "ground_fog",
        "heat_shimmer",
    }  # fmt: skip


def test_weather_is_scaled_to_the_scene() -> None:
    """Severity fixes extinction per metre. Over a three-kilometre view that
    is a blank frame, so fog is set by visibility against the scene's depth
    and the precipitation veils are kept from closing its far end. The
    transforms measure that depth; the gallery only says how many of it."""
    valley = Scene((25.0, 3000.0), 0.0)
    for fog in ("fog", "night_fog"):
        low, high = _overrides(fog, valley)["visibility_range_scene"]
        assert low < FOG_VISIBILITY < high
    assert _overrides("fog", valley)["depth_scale"] is None
    assert "beta_range" not in _overrides("fog", valley)

    assert _overrides("rain", valley)["min_visibility_scene"] == RAIN_VISIBILITY_FLOOR
    assert _overrides("snow", valley)["min_visibility_scene"] > 0.0

    crossing = Scene((12.0, 30.0), 70.0, sky_in_frame=False)
    # The same setting on a thirty-metre scene: nothing about it is the scene's.
    assert _overrides("rain", crossing)["min_visibility_scene"] == RAIN_VISIBILITY_FLOOR
    assert _overrides("rain", crossing)["camera_pitch_range_deg"] == (70.0, 70.0)
    # No sky in frame, so the dirt is lit by more than the camera can see.
    assert _overrides("dirt_soil", crossing)["ambient_light_range"][0] > 1.0
    assert "ambient_light_range" not in _overrides("dirt_soil", valley)
