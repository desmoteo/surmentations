"""The native kernels are plain C that relies on the compiler to vectorise, at -O3.

That reliance failed twice while the kernels were written: one loop was left
scalar when inlined into its caller, and again when a test program called it
with constants, each time four times slower and with nothing to show for it
but a timing. Timings on a CI runner are noise. Whether a loop was vectorised
is not: the compiler says so, and this asks it, for every loop the source
marks ``SM_VECTORISED`` and every instruction-set tier of the build.

GCC only, since the remarks are GCC's; with another compiler it skips.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native" / "sm_kernels.c"


def _hatch_build():
    """The build hook module, imported from the repository root."""
    spec = importlib.util.spec_from_file_location("hatch_build", ROOT / "hatch_build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gcc(build) -> list[str] | None:
    """The compiler command if it is GCC, whose vectorisation report the tests read; else ``None``.
    """
    try:
        compiler = build.find_compiler().split()
        version = subprocess.run([*compiler, "--version"], capture_output=True, text=True)
    except (build.NativeBuildError, OSError):
        return None
    return compiler if "Free Software Foundation" in version.stdout else None


def test_every_marked_loop_is_vectorised_in_every_tier() -> None:
    if not SOURCE.exists():
        pytest.skip("not a source checkout")
    build = _hatch_build()
    compiler = _gcc(build)
    if compiler is None:
        pytest.skip("needs GCC, whose vectoriser remarks this reads")

    marked = {
        number
        for number, line in enumerate(SOURCE.read_text().splitlines(), start=1)
        if line.lstrip().startswith("SM_VECTORISED for")
    }
    assert len(marked) >= 7, "the marks have gone from sm_kernels.c"

    for tier, tier_flags in build.tiers():
        result = subprocess.run(
            [*compiler, *build.COMMON_FLAGS, *tier_flags, f"-DSM_TIER={tier}"]
            + ["-fopt-info-vec-optimized", "-c", str(SOURCE), "-o", "/dev/null"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        remark = r"sm_kernels\.c:(\d+):\d+: optimized: loop vectorized"
        vectorised = {int(match.group(1)) for match in re.finditer(remark, result.stderr)}
        missing = sorted(marked - vectorised)
        assert not missing, f"tier {tier}: loops at lines {missing} were not vectorised"


def test_the_build_never_enables_fast_math() -> None:
    """Every tier must round as numpy does; these flags would change that."""
    build = _hatch_build()
    flags = set(build.COMMON_FLAGS) | {flag for _, extra in build.tiers() for flag in extra}
    assert "-ffp-contract=off" in flags
    assert not flags & {"-ffast-math", "-Ofast", "-funsafe-math-optimizations", "-mfma"}


def test_every_command_ends_at_o3_whatever_it_is_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A compiler or extra flags that ask for less are overruled: GCC and Clang
    take the last -O, and the build puts -O3 last on every command."""
    build = _hatch_build()
    commands: list[list[str]] = []
    monkeypatch.setattr(build, "_run", lambda command: commands.append(command))
    monkeypatch.setattr(build, "find_compiler", lambda: "cc -O0")
    monkeypatch.setattr(build.os, "replace", lambda *_args: None)
    build.build_native(output_dir=tmp_path, extra_flags=["-O1", "-Os"])
    assert commands, "the build ran no command"
    for command in commands:
        levels = [flag for flag in command if re.fullmatch(r"-O\w*", flag)]
        assert levels[-1] == "-O3", f"ends at {levels[-1]}: {' '.join(command)}"


def test_the_built_library_was_compiled_at_o3() -> None:
    """The library in the package records its command lines (ELF only): every
    compile in it ended at -O3."""
    import shutil
    import sys

    library = ROOT / "src" / "surmentations" / "_native" / build_name()
    if not library.exists() or not sys.platform.startswith("linux") or not shutil.which("readelf"):
        pytest.skip("needs the library built on Linux, and readelf")
    dump = subprocess.run(
        ["readelf", "-p", ".GCC.command.line", str(library)], capture_output=True, text=True
    ).stdout
    lines = [line for line in dump.splitlines() if "-O" in line]
    if not lines:
        pytest.skip("this library was built before its command lines were recorded")
    for line in lines:
        levels = re.findall(r"(?<!\S)-O\w*", line)
        assert levels[-1] == "-O3", line


def build_name() -> str:
    """The library's file name on this platform, as the build hook names it."""
    return _hatch_build().library_name()
