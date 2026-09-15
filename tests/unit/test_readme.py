"""The README's configuration reference must name parameters that exist.

Documentation drifts silently: nothing fails when a parameter is renamed and
the table that describes it is not. That has already cost this project twice
in other forms — a gallery override holding a stale copy of a library
default, and five published images from a naming the generator had dropped —
so the parameter names the README hands a reader are checked here against the
signatures they claim to describe.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from weathermentations.atmosphere import RandomFog, RandomRain, RandomSnow
from weathermentations.camera import (
    AcFlicker,
    ExposureTransient,
    NightIrMode,
    RollingShutter,
    TemporalDenoiseArtifacts,
    VideoCodecArtifacts,
    WdrToneMapping,
)
from weathermentations.lens import DirtSoil, HousingOccluders, SunFlare, WaterDroplets
from weathermentations.scene import CastShadows

README = Path(__file__).resolve().parents[2] / "README.md"

TRANSFORMS = {
    cls.__name__: cls
    for cls in (
        RandomFog, RandomRain, RandomSnow,
        WaterDroplets, DirtSoil, SunFlare, HousingOccluders,
        CastShadows, NightIrMode, AcFlicker, RollingShutter,
        ExposureTransient, WdrToneMapping, TemporalDenoiseArtifacts,
        VideoCodecArtifacts,
    )
}

# Named in the reference but not constructor arguments: module-level helpers
# and concepts the prose refers to.
NOT_PARAMETERS = {"p", "self"}


def _section(title: str) -> str:
    """The README section under `title`, at whatever heading level it uses.

    Matched by text rather than by a fixed `##`, so restructuring the document
    does not silently disable the check — which is the exact failure mode this
    file exists to catch, and one a brittle matcher would reintroduce.
    """
    text = README.read_text()
    start = re.search(rf"^#+\s+{re.escape(title)}\s*$", text, re.M)
    assert start, f"README has no section titled {title!r}"
    rest = text[start.end() :]
    nxt = re.search(r"^#+\s+\S", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


def _config_reference_rows() -> list[tuple[str, list[str]]]:
    """(transform name, parameter names it claims) for each table row."""
    section = _section("Configuration reference")
    rows = []
    for line in section.splitlines():
        match = re.match(r"\|\s*`(\w+)`\s*\|(.*)\|", line)
        if not match or match.group(1) not in TRANSFORMS:
            continue
        names = re.findall(r"`([a-z_][a-z0-9_]*)`", match.group(2))
        rows.append((match.group(1), names))
    return rows


def test_configuration_reference_covers_every_transform() -> None:
    documented = {name for name, _ in _config_reference_rows()}
    missing = set(TRANSFORMS) - documented
    assert not missing, f"README's configuration reference omits: {sorted(missing)}"


@pytest.mark.parametrize("name,claimed", _config_reference_rows(), ids=lambda v: str(v))
def test_configuration_reference_names_real_parameters(name: str, claimed: list[str]) -> None:
    signature = inspect.signature(TRANSFORMS[name].__init__).parameters
    unknown = [c for c in claimed if c not in signature and c not in NOT_PARAMETERS]
    assert not unknown, (
        f"README documents {unknown} for {name}, which its constructor does not take. "
        f"Available: {sorted(set(signature) - NOT_PARAMETERS)}"
    )


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_effect_tables_mark_the_depth_requirement_correctly(name: str) -> None:
    """A reader assembling a CPU-only pipeline relies on the "Needs depth"
    column, and a GPU requirement discovered at runtime is a bad surprise."""
    text = README.read_text()
    row = re.search(rf"^\|\s*`{name}`\s*\|.*\|\s*(yes|no)\s*\|\s*$", text, re.M)
    assert row, f"{name} has no row in the effect tables, or no depth column"
    needs_depth = "depth_provider" in inspect.signature(TRANSFORMS[name].__init__).parameters
    assert (row.group(1) == "yes") == needs_depth, (
        f"{name} {'needs' if needs_depth else 'does not need'} a depth provider, "
        f"but the effect table says {row.group(1)!r}"
    )


HOW_IT_WORKS = README.parent / "docs" / "how_it_works.md"


def test_how_it_works_diagrams_are_well_formed() -> None:
    """A broken mermaid block renders as a wall of raw text on GitHub, and
    nothing else in the suite would notice."""
    text = HOW_IT_WORKS.read_text()
    assert text.count("```") % 2 == 0, "unbalanced code fences"

    blocks = re.findall(r"```mermaid\n(.*?)```", text, re.S)
    assert len(blocks) >= 15, f"expected the diagrams to still be there, found {len(blocks)}"

    for index, block in enumerate(blocks, start=1):
        header = block.strip().splitlines()[0]
        assert re.match(r"^(flowchart|sequenceDiagram|graph)\b", header), (
            f"diagram {index} starts with {header!r}, which mermaid will not parse"
        )
        # Labels holding brackets or parentheses must be quoted, or the parser
        # takes them as shape syntax and gives up on the whole diagram.
        for line in block.splitlines():
            for label in re.findall(r"\[([^\"\[\]]*?)\]", line):
                assert not any(c in label for c in "()[]{}"), (
                    f"diagram {index} has an unquoted label needing quotes: {line.strip()!r}"
                )


def test_how_it_works_is_linked_from_the_readme() -> None:
    """It is the entry point for understanding the models; an unlinked
    document is one nobody reads."""
    assert "docs/how_it_works.md" in README.read_text()
