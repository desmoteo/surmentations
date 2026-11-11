"""The optional compiled kernels, and what to do without them.

A few loops in this library cost far more in Python than the arithmetic in
them: one call per snowflake, one per rain streak, and three passes over the
image to turn linear light back into code values. ``native/wm_kernels.c``
holds those loops in C; this module loads the library it is built into and
offers each kernel as a function that returns ``None`` when the library is
missing or switched off, so every caller keeps its numpy code and falls back
to it.

The two paths give the same pictures. The colour kernels are exact. The
particle kernels match the numpy code to float rounding (see
``tests/unit/test_native_kernels.py`` for the bounds), which moves an 8-bit
output by one level in a few pixels at most. In fast mode
(:mod:`weathermentations.mode`) the sprite kernel also trades its divisions
for multiplications, within the bounds in ``tests/unit/test_modes.py``.

Environment:

``WEATHERMENTATIONS_NATIVE``
    ``0`` runs the numpy code even when the library is there; ``require``
    makes a missing library an ``ImportError`` instead of a silent fallback.
``WEATHERMENTATIONS_ISA``
    Names the instruction-set tier to use (``sse2`` or ``avx2`` on x86-64),
    for comparing them on one machine. Every tier gives the same result.

How the library is built, and why it is C through ctypes, is in
``docs/native_kernels.md``.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "available",
    "decode_u8",
    "disabled",
    "enabled",
    "encode_f32",
    "encode_u8",
    "render_flakes",
    "render_streaks",
    "set_enabled",
    "set_tier",
    "mirror_rows",
    "tier",
    "tiers",
    "wet_composite",
    "wet_reflection",
]

# Must match WM_ABI_VERSION in native/wm_kernels.h. A library left over from
# an older checkout is ignored, not called with the wrong arguments.
_ABI_VERSION = 4

_pointer = ctypes.c_void_p
_SIGNATURES: dict[str, tuple[Any, list[Any]]] = {
    "wm_abi_version": (ctypes.c_int, []),
    "wm_tiers": (ctypes.c_char_p, []),
    "wm_tier": (ctypes.c_char_p, []),
    "wm_select": (ctypes.c_int, [ctypes.c_char_p]),
    "wm_encode_u8": (ctypes.c_int, [_pointer, _pointer, ctypes.c_size_t, _pointer]),
    "wm_encode_f32": (ctypes.c_int, [_pointer, _pointer, ctypes.c_size_t, _pointer]),
    "wm_decode_u8": (None, [_pointer, _pointer, ctypes.c_size_t, _pointer]),
    "wm_render_flakes": (
        ctypes.c_int,
        [_pointer, ctypes.c_int, ctypes.c_int, ctypes.c_ssize_t]
        + [_pointer] * 8
        + [ctypes.c_int]
        + [ctypes.c_double] * 5
        + [ctypes.c_int],
    ),
    "wm_mirror_rows": (
        ctypes.c_int,
        [_pointer, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_double, _pointer, _pointer],
    ),
    "wm_wet_reflection": (None, [_pointer] * 2 + [ctypes.c_ssize_t] * 3 + [_pointer] * 5),
    "wm_wet_composite": (
        ctypes.c_int,
        [_pointer] * 8 + [ctypes.c_ssize_t] * 2 + [ctypes.c_float] * 4,
    ),
    "wm_render_streaks": (
        ctypes.c_int,
        [_pointer, ctypes.c_int, ctypes.c_int, _pointer, ctypes.c_ssize_t]
        + [_pointer] * 5
        + [ctypes.c_int]
        + [_pointer] * 9
        + [ctypes.c_int],
    ),
}


def _load() -> ctypes.CDLL | None:
    mode = os.environ.get("WEATHERMENTATIONS_NATIVE", "").lower()
    found = sorted(Path(__file__).parent.glob("_wmkernels.*"))
    found = [path for path in found if path.suffix in (".so", ".dylib")]
    library = None
    problem = "it was not built (run `python hatch_build.py` in a checkout)"
    if found:
        try:
            library = ctypes.CDLL(str(found[0]))
            for name, (result, arguments) in _SIGNATURES.items():
                function = getattr(library, name)
                function.restype = result
                function.argtypes = arguments
            if library.wm_abi_version() != _ABI_VERSION:
                library, problem = None, f"{found[0].name} is from another version; rebuild it"
        except (OSError, AttributeError) as error:
            library, problem = None, str(error)
    if library is None:
        if mode == "require":
            raise ImportError(f"weathermentations' native kernels are required but {problem}")
        return None
    wanted = os.environ.get("WEATHERMENTATIONS_ISA", "auto").lower()
    if library.wm_select(wanted.encode()) != 0:
        library.wm_select(b"auto")
    return library


_library = _load()
_enabled = _library is not None and os.environ.get("WEATHERMENTATIONS_NATIVE", "").lower() not in (
    "0",
    "off",
    "false",
    "no",
)


def available() -> bool:
    """Whether the compiled library was found, switched on or not."""
    return _library is not None


def enabled() -> bool:
    """Whether the kernels are in use."""
    return _enabled


def set_enabled(flag: bool) -> None:
    """Switch the kernels on or off; on has no effect without the library."""
    global _enabled
    _enabled = bool(flag) and _library is not None


@contextmanager
def disabled() -> Iterator[None]:
    """Run the numpy code inside the block, whatever is installed."""
    before = _enabled
    set_enabled(False)
    try:
        yield
    finally:
        set_enabled(before)


def tiers() -> tuple[str, ...]:
    """The instruction-set tiers the library was built with, baseline first."""
    if _library is None:
        return ()
    return tuple(_library.wm_tiers().decode().split(","))


def tier() -> str | None:
    """The tier in use, or ``None`` without the library."""
    return None if _library is None else _library.wm_tier().decode()


def set_tier(name: str) -> None:
    """Use tier ``name`` (``"auto"`` for the best the CPU has)."""
    if _library is None:
        raise RuntimeError("the native kernels are not available")
    if _library.wm_select(name.encode()) != 0:
        raise ValueError(f"tier {name!r} is not in this build or not supported by this CPU")


def _float32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


def _int32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.int32)


# ---------------------------------------------------------------- colour


def encode_u8(linear: np.ndarray, table: np.ndarray) -> np.ndarray | None:
    """``table[int(clip(linear, 0, 1) * 65535)]`` as uint8, in one pass.

    ``None`` when the kernels are off, or when ``linear`` holds a NaN, which
    is left to the numpy path to report as it always has.
    """
    if not _enabled or linear.dtype != np.float32:
        return None
    assert _library is not None
    source = np.ascontiguousarray(linear)
    out = np.empty(source.shape, dtype=np.uint8)
    bad = _library.wm_encode_u8(source.ctypes.data, out.ctypes.data, source.size, table.ctypes.data)
    return None if bad else out


def encode_f32(linear: np.ndarray, table: np.ndarray) -> np.ndarray | None:
    """As :func:`encode_u8`, through a float32 table."""
    if not _enabled or linear.dtype != np.float32:
        return None
    assert _library is not None
    source = np.ascontiguousarray(linear)
    out = np.empty(source.shape, dtype=np.float32)
    bad = _library.wm_encode_f32(
        source.ctypes.data, out.ctypes.data, source.size, table.ctypes.data
    )
    return None if bad else out


def decode_u8(image: np.ndarray, table: np.ndarray) -> np.ndarray | None:
    """``table[image]`` for a uint8 image and a 256-entry float32 table."""
    if not _enabled or image.dtype != np.uint8:
        return None
    assert _library is not None
    source = np.ascontiguousarray(image)
    out = np.empty(source.shape, dtype=np.float32)
    _library.wm_decode_u8(source.ctypes.data, out.ctypes.data, source.size, table.ctypes.data)
    return out


# ------------------------------------------------------------- particles


def render_flakes(
    canvas: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius: np.ndarray,
    blur_radius: np.ndarray,
    smear: np.ndarray,
    dir_x: np.ndarray,
    dir_y: np.ndarray,
    gain: np.ndarray | None,
    blades: int,
    blade_rotation: float,
    rim_gain: float,
    cats_eye: float,
    frame_centre: tuple[float, float],
    fast: bool = False,
) -> bool:
    """Max-blend every flake's sprite into ``canvas``; ``False`` if not done.

    ``fast`` replaces the per-pixel divisions by multiplications with a
    reciprocal: the sprites differ in the last bits of a float.
    """
    if (
        not _enabled
        or canvas.dtype != np.float32
        or canvas.ndim != 2
        or not canvas.flags.c_contiguous
        or not canvas.flags.writeable
    ):
        return False
    assert _library is not None
    count = len(x)
    columns = [_float32(v) for v in (x, y, radius, blur_radius, smear, dir_x, dir_y)]
    if any(column.shape != (count,) for column in columns):
        return False
    gains = None if gain is None else _float32(gain)
    failed = _library.wm_render_flakes(
        canvas.ctypes.data,
        canvas.shape[0],
        canvas.shape[1],
        count,
        *(column.ctypes.data for column in columns),
        None if gains is None else gains.ctypes.data,
        int(blades),
        float(blade_rotation),
        float(rim_gain),
        float(cats_eye),
        float(frame_centre[0]),
        float(frame_centre[1]),
        int(bool(fast)),
    )
    return failed == 0


def render_streaks(
    layer: np.ndarray,
    select: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    patch_w: np.ndarray,
    patch_h: np.ndarray,
    vertices: np.ndarray,
    thickness: np.ndarray,
    alpha: np.ndarray,
    col_offset: np.ndarray,
    col_rate: np.ndarray,
    row_offset: np.ndarray,
    row_rate: np.ndarray,
    phase: np.ndarray,
    blur_sigma: np.ndarray,
    beading: np.ndarray,
) -> bool:
    """Add the streaks listed in ``select`` to ``layer``.

    ``vertices`` is ``(N, V, 2)`` int32, patch-local, and every listed streak's
    vertices must lie inside its patch with room for half its thickness.
    ``False`` if nothing was drawn.
    """
    if (
        not _enabled
        or layer.dtype != np.float32
        or layer.ndim != 2
        or not layer.flags.c_contiguous
        or not layer.flags.writeable
    ):
        return False
    assert _library is not None
    chosen = _int32(select)
    corners = _int32(vertices)
    integers = [_int32(v) for v in (x0, y0, patch_w, patch_h)]
    widths = _int32(thickness)
    floats = [
        _float32(v)
        for v in (alpha, col_offset, col_rate, row_offset, row_rate, phase, blur_sigma)
    ]
    table = _float32(beading)
    failed = _library.wm_render_streaks(
        layer.ctypes.data,
        layer.shape[0],
        layer.shape[1],
        chosen.ctypes.data,
        len(chosen),
        *(column.ctypes.data for column in integers),
        corners.ctypes.data,
        corners.shape[1],
        widths.ctypes.data,
        *(column.ctypes.data for column in floats),
        table.ctypes.data,
        len(table),
    )
    return failed == 0


def wet_composite(
    dry: np.ndarray,
    sharp: np.ndarray,
    rough: np.ndarray,
    wetness: np.ndarray,
    puddles: np.ndarray | None,
    fresnel: np.ndarray,
    film_fresnel: np.ndarray,
    ratio: float,
    film: float,
    gloss: float,
) -> np.ndarray | None:
    """The wet road over ``dry``, ``(rows, width, 3)``; ``None`` if not done.

    The arithmetic of ``rain._apply_ground_wetness`` in fast mode, in one
    pass: the darkened surface, the mirror image weighted over it, and the
    blend with the dry frame by ``wetness``. ``fresnel`` and ``film_fresnel``
    hold one value a row.
    """
    if not _enabled or dry.dtype != np.float32 or dry.ndim != 3 or dry.shape[2] != 3:
        return None
    assert _library is not None
    rows, width = dry.shape[:2]
    images = [_float32(v) for v in (dry, sharp, rough)]
    soaked = _float32(wetness)
    pools = None if puddles is None else _float32(puddles)
    per_row = [_float32(fresnel).reshape(-1), _float32(film_fresnel).reshape(-1)]
    shapes_agree = (
        all(image.shape == dry.shape for image in images)
        and soaked.shape == (rows, width)
        and (pools is None or pools.shape == (rows, width))
        and all(values.shape == (rows,) for values in per_row)
    )
    if not shapes_agree:
        return None
    out = np.empty(dry.shape, dtype=np.float32)
    failed = _library.wm_wet_composite(
        out.ctypes.data,
        *(image.ctypes.data for image in images),
        soaked.ctypes.data,
        None if pools is None else pools.ctypes.data,
        *(values.ctypes.data for values in per_row),
        rows,
        width,
        float(ratio),
        # In double, as numpy's `1.0 - ratio` is, and rounded once on the way in.
        1.0 - float(ratio),
        float(film),
        float(gloss),
    )
    return None if failed else out


def mirror_rows(solid: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray] | None:
    """``(about_foot, in_reach)`` of ``rain._mirrored_rows`` from its ``solid`` mask.

    ``about_foot`` is the row each pixel is mirrored about the foot of what
    stands above it, ``in_reach`` 1 where that mirror applies and 0 where it
    does not, both ``(H, W)`` float32, the second before it is feathered.
    ``None`` if not done.
    """
    if not _enabled or solid.dtype != np.bool_ or solid.ndim != 2:
        return None
    assert _library is not None
    mask = np.ascontiguousarray(solid).view(np.uint8)
    about_foot = np.empty(mask.shape, dtype=np.float32)
    in_reach = np.empty(mask.shape, dtype=np.float32)
    failed = _library.wm_mirror_rows(
        mask.ctypes.data,
        mask.shape[0],
        mask.shape[1],
        float(limit),
        about_foot.ctypes.data,
        in_reach.ctypes.data,
    )
    return None if failed else (about_foot, in_reach)


def wet_reflection(
    source: np.ndarray,
    about_horizon: np.ndarray,
    off_frame: np.ndarray,
    about_foot: np.ndarray,
    under_object: np.ndarray | None,
    sky: np.ndarray,
) -> np.ndarray | None:
    """The wet road's mirror image for a band of rows; ``None`` if not done.

    ``source`` is the whole frame, ``(H, W, 3)``. ``about_horizon`` and
    ``off_frame`` hold one value for each row of the band, ``about_foot``
    and ``under_object`` one for each of its pixels; ``under_object`` is
    ``None`` where nothing in the band is mirrored about a foot.
    """
    usable = (
        _enabled
        and source.dtype == np.float32
        and source.ndim == 3
        and source.shape[2] == 3
        and sky.dtype == np.float32
        and sky.size == 3
    )
    if not usable:
        return None
    assert _library is not None
    height, width = source.shape[:2]
    frame = np.ascontiguousarray(source)
    per_row = [_float32(about_horizon).reshape(-1), _float32(off_frame).reshape(-1)]
    rows = len(per_row[0])
    foot = _float32(about_foot)
    under = None if under_object is None else _float32(under_object)
    shapes_agree = (
        len(per_row[1]) == rows
        and foot.shape == (rows, width)
        and (under is None or under.shape == (rows, width))
    )
    if not shapes_agree:
        return None
    out = np.empty((rows, width, 3), dtype=np.float32)
    _library.wm_wet_reflection(
        out.ctypes.data,
        frame.ctypes.data,
        height,
        width,
        rows,
        per_row[0].ctypes.data,
        per_row[1].ctypes.data,
        foot.ctypes.data,
        None if under is None else under.ctypes.data,
        np.ascontiguousarray(sky).reshape(-1).ctypes.data,
    )
    return out
