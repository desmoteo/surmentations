"""The native kernels are plain C that relies on the compiler to vectorise.

That reliance failed twice while the kernels were written: one loop was left
scalar when inlined into its caller, and again when a test program called it
with constants, each time four times slower and with nothing to show for it
but a timing. Timings on a CI runner are noise. Whether a loop was vectorised
is not: the compiler says so, and this asks it, for every loop the source
marks ``WM_VECTORISED`` and every instruction-set tier of the build.

GCC only, since the remarks are GCC's; with another compiler it skips.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native" / "wm_kernels.c"


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
        if line.lstrip().startswith("WM_VECTORISED for")
    }
    assert len(marked) >= 7, "the marks have gone from wm_kernels.c"

    for tier, tier_flags in build.tiers():
        result = subprocess.run(
            [*compiler, *build.COMMON_FLAGS, *tier_flags, f"-DWM_TIER={tier}"]
            + ["-fopt-info-vec-optimized", "-c", str(SOURCE), "-o", "/dev/null"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        remark = r"wm_kernels\.c:(\d+):\d+: optimized: loop vectorized"
        vectorised = {int(match.group(1)) for match in re.finditer(remark, result.stderr)}
        missing = sorted(marked - vectorised)
        assert not missing, f"tier {tier}: loops at lines {missing} were not vectorised"


def test_the_build_never_enables_fast_math() -> None:
    """Every tier must round as numpy does; these flags would change that."""
    build = _hatch_build()
    flags = set(build.COMMON_FLAGS) | {flag for _, extra in build.tiers() for flag in extra}
    assert "-ffp-contract=off" in flags
    assert not flags & {"-ffast-math", "-Ofast", "-funsafe-math-optimizations", "-mfma"}
