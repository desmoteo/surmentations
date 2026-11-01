"""Builds the optional native kernels (``native/*.c``) into the package.

Two ways in:

* as hatchling's build hook, when a wheel or an editable install is made;
* run directly, ``python hatch_build.py``, to build in place in a checkout.

The library is optional. Without a C compiler the build goes on and the
package runs its numpy code, which gives the same pictures, more slowly; set
``WEATHERMENTATIONS_BUILD_NATIVE=require`` to make a failed build an error,
or ``=0`` to skip it and build a pure-Python wheel.

``wm_kernels.c`` is compiled once per instruction-set tier, each with its own
flags, and ``wm_dispatch.c`` picks a tier when the library loads; nothing but
the tier's own object is compiled for more than the platform's baseline, so
one binary runs on every CPU of the architecture. See docs/native_kernels.md.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "native"
PACKAGE_DIR = ROOT / "src" / "weathermentations" / "_native"
LIBRARY_STEM = "_wmkernels"

# -ffp-contract=off: no fused multiply-add, so every tier and every compiler
# rounds as numpy does. The three after it change no result and are what the
# vectoriser needs. -fno-math-errno lets sqrtf be an instruction, not a call.
# -fno-trapping-math lets a clamp followed by a float-to-int conversion be
# computed for every lane (it matters only to code that reads the
# floating-point exception flags). -fno-thread-jumps stops GCC turning a
# chain of clamps into branches before the vectoriser sees it: with jump
# threading on, the same loop was vectorised in one function and left scalar
# in another, four times slower. -funroll-loops is worth 8 to 18% on the
# colour encode's table lookups and nothing elsewhere, for 40 KB of binary.
#
# Tried and left out, with the numbers in docs/native_kernels.md: -Ofast
# (9% at most, on sprites only, and every tier and compiler then rounds its
# own way, which is the one thing the build promises not to do), fused
# multiply-add (4% at most, same objection), and -fstrict-overflow, which
# GCC already applies at -O3: the machine code is identical with it.
COMMON_FLAGS = [
    "-O3",
    "-std=c11",
    "-fPIC",
    "-fvisibility=hidden",
    "-ffp-contract=off",
    "-fno-math-errno",
    "-fno-trapping-math",
    "-fno-thread-jumps",
    "-funroll-loops",
    "-Wall",
    "-Wextra",
]


class NativeBuildError(RuntimeError):
    pass


def library_name() -> str:
    return LIBRARY_STEM + (".dylib" if sys.platform == "darwin" else ".so")


def tiers() -> list[tuple[str, list[str]]]:
    """``(name, extra flags)`` per tier, baseline first."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return [("sse2", []), ("avx2", ["-mavx2"])]
    if machine in ("aarch64", "arm64"):
        return [("neon", [])]
    return [("portable", [])]


def find_compiler() -> str:
    for candidate in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if candidate and shutil.which(candidate.split()[0]):
            return candidate
    raise NativeBuildError("no C compiler found (set CC, or install gcc or clang)")


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise NativeBuildError(f"{' '.join(command)}\n{result.stderr.strip()}")


def build_native(
    output_dir: Path = PACKAGE_DIR, extra_flags: list[str] | None = None
) -> Path:
    """Compile the kernels into ``output_dir`` and return the library's path."""
    if sys.platform == "win32":
        raise NativeBuildError("the native kernels are not built on Windows yet")
    compiler = find_compiler().split()
    flags = COMMON_FLAGS + (extra_flags or [])
    built = tiers()
    base = built[0][0]
    output = output_dir / library_name()
    with tempfile.TemporaryDirectory() as scratch:
        objects = []
        for name, tier_flags in built:
            obj = str(Path(scratch) / f"kernels_{name}.o")
            _run(
                [*compiler, *flags, *tier_flags, f"-DWM_TIER={name}", "-c"]
                + [str(SOURCES / "wm_kernels.c"), "-o", obj]
            )
            objects.append(obj)
        dispatch = str(Path(scratch) / "dispatch.o")
        defines = [f"-DWM_BASE={base}"] + (["-DWM_HAVE_AVX2"] if len(built) > 1 else [])
        _run([*compiler, *flags, *defines, "-c", str(SOURCES / "wm_dispatch.c"), "-o", dispatch])
        output_dir.mkdir(parents=True, exist_ok=True)
        # Link to a temporary name and rename, so a process that has the old
        # library mapped keeps a whole file.
        staging = str(output) + ".new"
        _run([*compiler, "-shared", *flags, *objects, dispatch, "-lm", "-o", staging])
        os.replace(staging, output)
    return output


def platform_tag() -> str:
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ImportError:  # run directly, without hatchling installed
    BuildHookInterface = object  # type: ignore[assignment,misc]


class NativeKernelsHook(BuildHookInterface):  # type: ignore[type-arg]
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        mode = os.environ.get("WEATHERMENTATIONS_BUILD_NATIVE", "auto").lower()
        if self.target_name != "wheel" or mode in ("0", "off", "false", "no"):
            return
        try:
            library = build_native()
        except NativeBuildError as error:
            if mode == "require":
                raise
            self.app.display_warning(
                f"weathermentations: native kernels not built, using numpy only.\n{error}"
            )
            return
        # The library is git-ignored, which hatchling takes to mean "leave it
        # out"; an artifact is included all the same.
        build_data["artifacts"].append(str(library.relative_to(ROOT)))
        # ctypes, not the C API: the binary depends on the platform and on no
        # Python version.
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{platform_tag()}"


if __name__ == "__main__":
    try:
        print(build_native())
    except NativeBuildError as error:
        raise SystemExit(f"native kernels not built: {error}") from error
