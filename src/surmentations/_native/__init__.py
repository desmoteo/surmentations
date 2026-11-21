"""The optional compiled kernels, and what to do without them.

A few loops in this library cost far more in Python than the arithmetic in
them: one call per snowflake, one per rain streak, and three passes over the
image to turn linear light back into code values. ``native/sm_kernels.c``
holds those loops in C; this module loads the library it is built into and
offers each kernel as a function that returns ``None`` when the library is
missing or switched off, so every caller keeps its numpy code and falls back
to it.

The two paths give the same pictures. The colour kernels are exact. The
particle kernels match the numpy code to float rounding (see
``tests/unit/test_native_kernels.py`` for the bounds), which moves an 8-bit
output by one level in a few pixels at most. In fast mode
(:mod:`surmentations.mode`) the sprite kernel also trades its divisions
for multiplications, within the bounds in ``tests/unit/test_modes.py``.

Environment:

``SURMENTATIONS_NATIVE``
    ``0`` runs the numpy code even when the library is there; ``require``
    makes a missing library an ``ImportError`` instead of a silent fallback.
``SURMENTATIONS_ISA``
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
    "dirt_composite",
    "dirt_emphasise",
    "disabled",
    "droplet_composite",
    "edge_preserving",
    "enabled",
    "encode_f32",
    "encode_u8",
    "fog_lost",
    "fog_veil",
    "ground_tile",
    "grow_alike",
    "headlamp_beams",
    "lamps_lit",
    "overcast_blend",
    "place_apart",
    "render_flakes",
    "render_streaks",
    "road_along",
    "road_places",
    "set_enabled",
    "set_tier",
    "snow_settled",
    "splat_discs",
    "spray_and_lamps",
    "spray_puffs",
    "street_air",
    "street_light",
    "mirror_rows",
    "night_colour_expose",
    "night_colour_light",
    "night_ir_expose",
    "night_ir_light",
    "nir_reflectance",
    "normals",
    "tier",
    "tiers",
    "veil_blend",
    "water_lobes",
    "wet_composite",
    "wheel_tracks",
    "wet_reflection",
]

# Must match SM_ABI_VERSION in native/sm_kernels.h. A library left over from
# an older checkout is ignored, not called with the wrong arguments.
_ABI_VERSION = 14

_pointer = ctypes.c_void_p
_SIGNATURES: dict[str, tuple[Any, list[Any]]] = {
    "sm_abi_version": (ctypes.c_int, []),
    "sm_tiers": (ctypes.c_char_p, []),
    "sm_tier": (ctypes.c_char_p, []),
    "sm_select": (ctypes.c_int, [ctypes.c_char_p]),
    "sm_encode_u8": (ctypes.c_int, [_pointer, _pointer, ctypes.c_size_t, _pointer]),
    "sm_encode_f32": (ctypes.c_int, [_pointer, _pointer, ctypes.c_size_t, _pointer]),
    "sm_decode_u8": (None, [_pointer, _pointer, ctypes.c_size_t, _pointer]),
    "sm_render_flakes": (
        ctypes.c_int,
        [_pointer, ctypes.c_int, ctypes.c_int, ctypes.c_ssize_t]
        + [_pointer] * 8
        + [ctypes.c_int]
        + [ctypes.c_double] * 5
        + [ctypes.c_int],
    ),
    "sm_edge_preserving": (
        ctypes.c_int,
        [_pointer, _pointer, ctypes.c_ssize_t, ctypes.c_ssize_t]
        + [ctypes.c_float, ctypes.c_float, ctypes.c_int],
    ),
    "sm_dirt_emphasise": (None, [_pointer] * 3 + [ctypes.c_ssize_t, ctypes.c_float]),
    "sm_dirt_composite": (
        None,
        [_pointer] * 6 + [ctypes.c_ssize_t] + [ctypes.c_float] * 4,
    ),
    "sm_veil_blend": (None, [_pointer] * 7 + [ctypes.c_ssize_t]),
    "sm_overcast_blend": (None, [_pointer] * 6 + [ctypes.c_ssize_t] + [ctypes.c_float] * 3),
    "sm_grow_alike": (None, [_pointer] * 7 + [ctypes.c_ssize_t] + [ctypes.c_float] * 5),
    "sm_place_apart": (
        ctypes.c_ssize_t,
        [_pointer] * 4 + [ctypes.c_ssize_t] * 2 + [ctypes.c_double, _pointer],
    ),
    "sm_water_lobes": (
        None,
        [_pointer] * 2
        + [ctypes.c_ssize_t] * 2
        + [_pointer] * 2
        + [ctypes.c_ssize_t, ctypes.c_int, _pointer]
        + [ctypes.c_double] * 2,
    ),
    "sm_droplet_composite": (None, [_pointer] * 5 + [ctypes.c_ssize_t]),
    "sm_splat_discs": (
        None,
        [_pointer] + [ctypes.c_ssize_t] * 2 + [_pointer] * 5 + [ctypes.c_ssize_t],
    ),
    "sm_fog_lost": (None, [_pointer] * 6 + [ctypes.c_ssize_t, ctypes.c_float]),
    "sm_fog_veil": (None, [_pointer] * 7 + [ctypes.c_ssize_t, _pointer]),
    "sm_ground_tile": (None, [_pointer] * 3 + [ctypes.c_ssize_t] * 2 + [_pointer]),
    "sm_snow_settled": (None, [_pointer] * 12 + [ctypes.c_ssize_t, _pointer]),
    "sm_nir_reflectance": (
        None,
        [_pointer] * 3 + [ctypes.c_ssize_t, _pointer] + [ctypes.c_float] * 3,
    ),
    "sm_night_ir_light": (None, [_pointer] * 8 + [ctypes.c_ssize_t] + [ctypes.c_float] * 3),
    "sm_night_ir_expose": (None, [_pointer] * 11 + [ctypes.c_ssize_t, _pointer]),
    "sm_night_colour_light": (None, [_pointer] * 8 + [ctypes.c_ssize_t, _pointer]),
    "sm_night_colour_expose": (None, [_pointer] * 7 + [ctypes.c_ssize_t, _pointer, _pointer]),
    "sm_normals": (None, [_pointer] * 7 + [ctypes.c_ssize_t] * 3),
    "sm_spray_puffs": (
        None,
        [_pointer] * 4 + [ctypes.c_ssize_t] * 2 + [_pointer, ctypes.c_ssize_t]
        + [ctypes.c_float] * 3,
    ),
    "sm_street_light": (
        None,
        [_pointer] * 3 + [ctypes.c_ssize_t, _pointer, ctypes.c_ssize_t, _pointer, ctypes.c_float],
    ),
    "sm_headlamp_beams": (
        None,
        [_pointer] * 5 + [ctypes.c_ssize_t, _pointer, ctypes.c_ssize_t, _pointer],
    ),
    "sm_road_along": (None, [_pointer] + [ctypes.c_ssize_t] * 2 + [ctypes.c_double] * 3),
    "sm_road_places": (None, [_pointer] * 7 + [ctypes.c_ssize_t] * 2 + [_pointer] * 3),
    "sm_wheel_tracks": (
        None,
        [_pointer] * 7 + [ctypes.c_ssize_t, _pointer, _pointer, ctypes.c_int32, _pointer, _pointer],
    ),
    "sm_lamps_lit": (None, [_pointer] * 3 + [ctypes.c_ssize_t]),
    "sm_spray_and_lamps": (
        None,
        [_pointer] * 5 + [ctypes.c_ssize_t, _pointer, ctypes.c_float],
    ),
    "sm_street_air": (
        None,
        [_pointer] * 7 + [ctypes.c_ssize_t, _pointer, ctypes.c_ssize_t, _pointer]
        + [ctypes.c_float] * 3,
    ),
    "sm_mirror_rows": (
        ctypes.c_int,
        [_pointer, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_double, _pointer, _pointer],
    ),
    "sm_wet_reflection": (None, [_pointer] * 2 + [ctypes.c_ssize_t] * 3 + [_pointer] * 5),
    "sm_wet_composite": (
        ctypes.c_int,
        [_pointer] * 8 + [ctypes.c_ssize_t] * 2 + [ctypes.c_float] * 4,
    ),
    "sm_render_streaks": (
        ctypes.c_int,
        [_pointer, ctypes.c_int, ctypes.c_int, _pointer, ctypes.c_ssize_t]
        + [_pointer] * 5
        + [ctypes.c_int]
        + [_pointer] * 9
        + [ctypes.c_int],
    ),
}


def _load() -> ctypes.CDLL | None:
    """Find, load and check the compiled library; ``None`` if there is none to use.

    A library whose ABI version is not this module's is not used, nor one that
    lacks an entry point; with ``SURMENTATIONS_NATIVE=require`` either, or no
    library at all, is an ``ImportError``. The tier is the one
    ``SURMENTATIONS_ISA`` names, or the library's own choice if it names
    none or one this CPU lacks. Whether the kernels are then used is decided
    separately, by ``SURMENTATIONS_NATIVE=0`` and :func:`set_enabled`.
    """
    mode = os.environ.get("SURMENTATIONS_NATIVE", "").lower()
    found = sorted(Path(__file__).parent.glob("_smkernels.*"))
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
            if library.sm_abi_version() != _ABI_VERSION:
                library, problem = None, f"{found[0].name} is from another version; rebuild it"
        except (OSError, AttributeError) as error:
            library, problem = None, str(error)
    if library is None:
        if mode == "require":
            raise ImportError(f"surmentations' native kernels are required but {problem}")
        return None
    wanted = os.environ.get("SURMENTATIONS_ISA", "auto").lower()
    if library.sm_select(wanted.encode()) != 0:
        library.sm_select(b"auto")
    return library


_library = _load()
_enabled = _library is not None and os.environ.get("SURMENTATIONS_NATIVE", "").lower() not in (
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
    return tuple(_library.sm_tiers().decode().split(","))


def tier() -> str | None:
    """The tier in use, or ``None`` without the library."""
    return None if _library is None else _library.sm_tier().decode()


def set_tier(name: str) -> None:
    """Use tier ``name`` (``"auto"`` for the best the CPU has)."""
    if _library is None:
        raise RuntimeError("the native kernels are not available")
    if _library.sm_select(name.encode()) != 0:
        raise ValueError(f"tier {name!r} is not in this build or not supported by this CPU")


def _float32(array: np.ndarray) -> np.ndarray:
    """A C-contiguous float32 view or copy of ``array``, for a kernel to read."""
    return np.ascontiguousarray(array, dtype=np.float32)


def _int32(array: np.ndarray) -> np.ndarray:
    """A C-contiguous int32 view or copy of ``array``, for a kernel to read."""
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
    bad = _library.sm_encode_u8(source.ctypes.data, out.ctypes.data, source.size, table.ctypes.data)
    return None if bad else out


def encode_f32(linear: np.ndarray, table: np.ndarray) -> np.ndarray | None:
    """As :func:`encode_u8`, through a float32 table."""
    if not _enabled or linear.dtype != np.float32:
        return None
    assert _library is not None
    source = np.ascontiguousarray(linear)
    out = np.empty(source.shape, dtype=np.float32)
    bad = _library.sm_encode_f32(
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
    _library.sm_decode_u8(source.ctypes.data, out.ctypes.data, source.size, table.ctypes.data)
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
    failed = _library.sm_render_flakes(
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
    failed = _library.sm_render_streaks(
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
    failed = _library.sm_wet_composite(
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
    failed = _library.sm_mirror_rows(
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
    _library.sm_wet_reflection(
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


def edge_preserving(
    image: np.ndarray, sigma_s: float, sigma_r: float, tabled: bool
) -> np.ndarray | None:
    """``cv2.edgePreservingFilter(image, flags=RECURS_FILTER, ...)``; ``None`` if not done.

    For an ``(H, W, 3)`` uint8 image. With ``tabled`` off it is OpenCV's
    arithmetic in OpenCV's order; ``tabled`` takes the filter's power
    function from a table of its 766 possible values, which is faster and
    moves a few pixels in a million by one level.
    """
    if not _enabled or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        return None
    assert _library is not None
    source = np.ascontiguousarray(image)
    out = np.empty(source.shape, dtype=np.uint8)
    failed = _library.sm_edge_preserving(
        source.ctypes.data,
        out.ctypes.data,
        source.shape[0],
        source.shape[1],
        float(sigma_s),
        float(sigma_r),
        int(bool(tabled)),
    )
    return None if failed else out


def _is_rgb_float32(image: np.ndarray) -> bool:
    """Whether ``image`` is ``(H, W, 3)`` float32, the layout the colour kernels take."""
    return image.dtype == np.float32 and image.ndim == 3 and image.shape[2] == 3


def dirt_emphasise(linear: np.ndarray, luma: np.ndarray, boost: float) -> np.ndarray | None:
    """``linear * (1 + boost * clip((luma - 0.35) / 0.65, 0, 1))[..., None]``.

    ``DirtSoil``'s highlight-weighted scene, in one pass and with numpy's
    float32 arithmetic, so the result is the expression's to the bit.
    ``None`` if not done.
    """
    if (
        not _enabled
        or not _is_rgb_float32(linear)
        or luma.dtype != np.float32
        or luma.shape != linear.shape[:2]
    ):
        return None
    assert _library is not None
    scene, weights = np.ascontiguousarray(linear), np.ascontiguousarray(luma)
    out = np.empty(scene.shape, dtype=np.float32)
    _library.sm_dirt_emphasise(
        out.ctypes.data, scene.ctypes.data, weights.ctypes.data, weights.size, float(boost)
    )
    return out


def dirt_composite(
    linear: np.ndarray,
    spread: np.ndarray,
    mask: np.ndarray,
    albedo: np.ndarray,
    surroundings: np.ndarray,
    forward: float,
    veil_strength: float,
    absorption: float,
    body: float,
) -> np.ndarray | None:
    """``DirtSoil``'s deposit over the frame, in one pass; ``None`` if not done.

    ``spread`` is the scene through the scatter blur, ``surroundings`` the
    three values of light arriving from outside the frame, already weighted.
    The arithmetic and its order are those of ``DirtSoil.apply``.
    """
    usable = (
        _enabled
        and _is_rgb_float32(linear)
        and _is_rgb_float32(spread)
        and _is_rgb_float32(albedo)
        and spread.shape == linear.shape
        and albedo.shape == linear.shape
        and mask.dtype == np.float32
        and mask.shape == linear.shape[:2]
        and surroundings.dtype == np.float32
        and surroundings.size == 3
    )
    if not usable:
        return None
    assert _library is not None
    images = [np.ascontiguousarray(v) for v in (linear, spread)]
    cover, colours = np.ascontiguousarray(mask), np.ascontiguousarray(albedo)
    around = np.ascontiguousarray(surroundings).reshape(-1)
    out = np.empty(linear.shape, dtype=np.float32)
    _library.sm_dirt_composite(
        out.ctypes.data,
        images[0].ctypes.data,
        images[1].ctypes.data,
        cover.ctypes.data,
        colours.ctypes.data,
        around.ctypes.data,
        cover.size,
        float(forward),
        float(veil_strength),
        float(absorption),
        float(body),
    )
    return out


def _is_plane_of(plane: np.ndarray, image: np.ndarray) -> bool:
    """Whether ``plane`` is float32 with one value for each pixel of ``image``."""
    return plane.dtype == np.float32 and plane.shape == image.shape[:2]


def veil_blend(
    linear: np.ndarray,
    transmission: np.ndarray,
    airlight: np.ndarray,
    alpha: np.ndarray | None = None,
    light: np.ndarray | None = None,
) -> np.ndarray | None:
    """A frame through a veil and then a layer of particles; ``None`` if not done.

    ``linear * t + airlight * (1 - t)`` with ``t`` the ``(H, W)``
    ``transmission``, and with ``alpha`` given
    ``out * (1 - alpha) + light * alpha`` after it, ``light`` three values
    or an ``(H, W, 3)`` field. One pass, with numpy's float32 arithmetic in
    numpy's order, so the result is those expressions' to the bit.
    """
    usable = (
        _enabled
        and _is_rgb_float32(linear)
        and _is_plane_of(transmission, linear)
        and airlight.dtype == np.float32
        and airlight.size == 3
    )
    lit_everywhere = None
    if usable and alpha is not None:
        usable = _is_plane_of(alpha, linear) and light is not None and light.dtype == np.float32
        if usable and light is not None and light.size != 3:
            usable = light.shape == linear.shape
            lit_everywhere = np.ascontiguousarray(light) if usable else None
    if not usable:
        return None
    assert _library is not None
    scene, through = np.ascontiguousarray(linear), np.ascontiguousarray(transmission)
    air = np.ascontiguousarray(airlight).reshape(-1)
    cover = None if alpha is None else np.ascontiguousarray(alpha)
    three = None
    if light is not None and lit_everywhere is None:
        three = np.ascontiguousarray(light).reshape(-1)
    out = np.empty(scene.shape, dtype=np.float32)
    _library.sm_veil_blend(
        out.ctypes.data,
        scene.ctypes.data,
        through.ctypes.data,
        air.ctypes.data,
        None if cover is None else cover.ctypes.data,
        None if three is None else three.ctypes.data,
        None if lit_everywhere is None else lit_everywhere.ctypes.data,
        through.size,
    )
    return out


def overcast_blend(
    linear: np.ndarray,
    gain: np.ndarray,
    grey: np.ndarray,
    sky: np.ndarray | None,
    cloud: np.ndarray,
    desaturate: float,
    cover_scale: float,
    mean_grey: float,
) -> np.ndarray | None:
    """The last step of ``overcast.apply_overcast`` in fast mode; ``None`` if not done.

    Each pixel scaled by its ``gain``, drawn towards its ``grey`` by
    ``desaturate``, covered with ``cloud`` where ``sky`` says, and clipped:
    the arithmetic of ``overcast._relight``, in one pass.
    """
    usable = (
        _enabled
        and _is_rgb_float32(linear)
        and _is_plane_of(gain, linear)
        and _is_plane_of(grey, linear)
        and (sky is None or _is_plane_of(sky, linear))
        and cloud.dtype == np.float32
        and cloud.size == 3
    )
    if not usable:
        return None
    assert _library is not None
    scene = np.ascontiguousarray(linear)
    gains, greys = np.ascontiguousarray(gain), np.ascontiguousarray(grey)
    covered = None if sky is None else np.ascontiguousarray(sky)
    tint = np.ascontiguousarray(cloud).reshape(-1)
    out = np.empty(scene.shape, dtype=np.float32)
    _library.sm_overcast_blend(
        out.ctypes.data,
        scene.ctypes.data,
        gains.ctypes.data,
        greys.ctypes.data,
        None if covered is None else covered.ctypes.data,
        tint.ctypes.data,
        gains.size,
        float(desaturate),
        float(cover_scale),
        float(mean_grey),
    )
    return out


def grow_alike(
    linear: np.ndarray,
    nearby_sum: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
    grown: np.ndarray,
    depth: np.ndarray | None,
    level: float,
    depth_tolerance: float,
    colour_tolerance: float,
    least_weight: float,
    some_weight: float,
) -> np.ndarray | None:
    """A mask grown over the pixels that look like it; ``None`` if not done.

    The end of ``sky.sky_to_depth_edge`` (with ``depth``) and of
    ``ground.fill_to_edges`` (without): each pixel compared with the mask's
    colour near it, ``nearby_sum / max(weight, least_weight)``, and the mask
    kept, or taken from ``grown`` where the pixel is alike and, with
    ``depth``, near the mask's depth ``level``. ``linear`` and
    ``nearby_sum`` are ``(H, W, 3)``, the rest ``(H, W)``, all float32. One
    pass with the numpy code's arithmetic in its order: the same mask.
    """
    planes = [weight, mask, grown] + ([] if depth is None else [depth])
    usable = (
        _enabled
        and _is_rgb_float32(linear)
        and _is_rgb_float32(nearby_sum)
        and nearby_sum.shape == linear.shape
        and all(_is_plane_of(plane, linear) for plane in planes)
    )
    if not usable:
        return None
    assert _library is not None
    colour, around = np.ascontiguousarray(linear), np.ascontiguousarray(nearby_sum)
    fields = [np.ascontiguousarray(plane) for plane in planes]
    out = np.empty(linear.shape[:2], dtype=np.float32)
    _library.sm_grow_alike(
        out.ctypes.data,
        colour.ctypes.data,
        around.ctypes.data,
        *(field.ctypes.data for field in fields[:3]),
        None if depth is None else fields[3].ctypes.data,
        out.size,
        float(level),
        float(depth_tolerance),
        float(colour_tolerance),
        float(least_weight),
        float(some_weight),
    )
    return out


def place_apart(
    x: np.ndarray,
    y: np.ndarray,
    footprint: np.ndarray,
    order: np.ndarray,
    count: int,
    gap: float,
) -> np.ndarray | None:
    """Indices of the candidates kept by greedy placement; ``None`` if not done.

    Candidates are taken in ``order`` and kept, while fewer than ``count``
    are, if every one already kept is at least ``(r_kept + r) * (1 + gap)``
    away, ``r`` being the ``footprint``. The arithmetic of
    ``water_droplets._sample_positions`` in double, so the same candidates.
    """
    if not _enabled:
        return None
    assert _library is not None
    columns = [np.ascontiguousarray(v, dtype=np.float64) for v in (x, y, footprint)]
    taken = np.ascontiguousarray(order, dtype=np.int64)
    if any(column.shape != columns[0].shape for column in columns) or taken.ndim != 1:
        return None
    kept = np.empty(max(0, min(int(count), len(taken))), dtype=np.int64)
    if kept.size == 0:
        return kept
    found = _library.sm_place_apart(
        *(column.ctypes.data for column in columns),
        taken.ctypes.data,
        len(taken),
        kept.size,
        float(gap),
        kept.ctypes.data,
    )
    return kept[:found]


def water_lobes(
    level: np.ndarray,
    weighted: np.ndarray,
    lobes: np.ndarray,
    harmonics: np.ndarray | None,
    orders: tuple[int, ...],
    wobble_radius: float,
    outline_level: float,
) -> bool:
    """Add every drop's lobe into ``level`` and ``weighted``; ``False`` if not done.

    The lobe loop of ``water_droplets._water_surface``, for its fast mode:
    the outline's harmonics are taken from the angle's sine and cosine by a
    recurrence, not from ``arctan2`` and ``sin``, so the fields agree with
    numpy's to float rounding and not to the bit.
    """
    usable = (
        _enabled
        and level.dtype == weighted.dtype == np.float32
        and level.ndim == 2
        and level.shape == weighted.shape
        and level.flags.c_contiguous
        and weighted.flags.c_contiguous
        and lobes.ndim == 2
        and lobes.shape[1] == 7
        and len(orders) <= 8
    )
    if not usable:
        return False
    assert _library is not None
    drops = np.ascontiguousarray(lobes, dtype=np.float64)
    shapes = None
    if harmonics is not None:
        shapes = np.ascontiguousarray(harmonics, dtype=np.float64)
        if shapes.shape != (len(drops), len(orders), 2):
            return False
    order_values = np.ascontiguousarray(orders, dtype=np.int32)
    _library.sm_water_lobes(
        level.ctypes.data,
        weighted.ctypes.data,
        level.shape[0],
        level.shape[1],
        drops.ctypes.data,
        None if shapes is None else shapes.ctypes.data,
        len(drops),
        len(orders),
        order_values.ctypes.data,
        float(wobble_radius),
        float(outline_level),
    )
    return True


def droplet_composite(
    base: np.ndarray,
    scattered: np.ndarray | None,
    fraction: np.ndarray | None,
    layers: np.ndarray | None,
) -> np.ndarray | None:
    """The end of ``WaterDroplets.apply`` in one pass; ``None`` if not done.

    ``base + (scattered - base) * fraction``, then ``out * (1 - layers[..., 3])
    + layers[..., :3]``: the condensation film and the drops over it, with
    numpy's arithmetic in its order. ``base`` and ``scattered`` are
    ``(H, W, 3)``, ``fraction`` ``(H, W)`` or ``(H, W, 1)`` and ``layers``
    ``(H, W, 4)``, all float32; either step can be left out with ``None``.
    """
    if not _enabled or not _is_rgb_float32(base):
        return None
    pixels = base.shape[0] * base.shape[1]
    film = scattered is not None and fraction is not None
    if film:
        assert scattered is not None and fraction is not None
        if not (
            _is_rgb_float32(scattered)
            and scattered.shape == base.shape
            and fraction.dtype == np.float32
            and fraction.size == pixels
        ):
            return None
    if layers is not None and (layers.dtype != np.float32 or layers.shape != (*base.shape[:2], 4)):
        return None
    assert _library is not None
    source = np.ascontiguousarray(base)
    light = np.ascontiguousarray(scattered) if film else None
    share = np.ascontiguousarray(fraction) if film else None
    drops = None if layers is None else np.ascontiguousarray(layers)
    out = np.empty(base.shape, dtype=np.float32)
    _library.sm_droplet_composite(
        out.ctypes.data,
        source.ctypes.data,
        None if light is None else light.ctypes.data,
        None if share is None else share.ctypes.data,
        None if drops is None else drops.ctypes.data,
        pixels,
    )
    return out


def splat_discs(
    total: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    radius: np.ndarray,
    weight: np.ndarray,
    order: np.ndarray,
) -> bool:
    """Add a field of discs of light into ``total``; ``False`` if not done.

    ``total`` is ``(H, W)`` float64 and is added to in place. Disc ``i`` is
    centred on ``(x[i], y[i])`` with ``radius[i]`` and holds ``weight[i]``,
    all float32; the discs are taken in ``order``, int64. The arithmetic of
    ``illuminator.splat_discs``, so the same field to the last bit.
    """
    if not _enabled:
        return False
    if total.dtype != np.float64 or total.ndim != 2 or not total.flags.c_contiguous:
        return False
    columns = [x, y, radius, weight]
    if any(c.dtype != np.float32 or c.ndim != 1 or len(c) != len(x) for c in columns):
        return False
    if order.dtype != np.int64 or order.ndim != 1:
        return False
    assert _library is not None
    columns = [np.ascontiguousarray(c) for c in columns]
    taken = np.ascontiguousarray(order)
    _library.sm_splat_discs(
        total.ctypes.data,
        total.shape[0],
        total.shape[1],
        *(c.ctypes.data for c in columns),
        taken.ctypes.data,
        len(taken),
    )
    return True


def _planes_alike(planes: list[np.ndarray]) -> bool:
    """Whether every one is a float32 array, contiguous, of the first one's shape."""
    shape = planes[0].shape
    return all(
        p.dtype == np.float32 and p.shape == shape and p.flags.c_contiguous for p in planes
    )


def street_air(
    air: np.ndarray,
    reach: np.ndarray,
    ray: tuple[np.ndarray, np.ndarray, np.ndarray],
    far: np.ndarray,
    rising: np.ndarray,
    lamps: np.ndarray,
    up: np.ndarray,
    nearest_squared: float,
    spill: float,
    forward: float,
) -> bool:
    """Add every lamp's light on the air into ``air`` and ``reach``; ``False`` if not done.

    The sum of ``night_light.street_air``: ``air`` and ``reach`` are added
    to in place. ``ray`` is the lines of sight as three planes, ``far``
    their lengths and ``rising`` their direction against ``up``, all
    float32 of one shape; ``lamps`` is ``(n, 4)`` float32. The arithmetic
    of ``night_light._air_of_lamp`` in its order, so the same maps to the
    last bit.
    """
    if not _enabled or not _planes_alike([air, reach, *ray, far, rising]):
        return False
    if lamps.dtype != np.float32 or lamps.ndim != 2 or lamps.shape[1] != 4:
        return False
    if up.dtype != np.float32 or up.size != 3:
        return False
    assert _library is not None
    heads, upright = np.ascontiguousarray(lamps), np.ascontiguousarray(up)
    _library.sm_street_air(
        air.ctypes.data,
        reach.ctypes.data,
        *(plane.ctypes.data for plane in ray),
        far.ctypes.data,
        rising.ctypes.data,
        air.size,
        heads.ctypes.data,
        len(heads),
        upright.ctypes.data,
        float(nearest_squared),
        float(spill),
        float(forward),
    )
    return True


def spray_puffs(
    spray: np.ndarray,
    puffs: np.ndarray,
    stands: np.ndarray,
    hidden: np.ndarray,
    depth: np.ndarray | None,
    feather: float,
    hides_wholly: float,
    stands_nearer: float,
) -> bool:
    """Add a table of puffs of spray into ``spray``; ``False`` if not done.

    What ``road._add_puff`` does for each row of ``puffs``, ``(n, 15)``
    float64, in the table's order. ``spray``, ``stands``, ``hidden`` and
    ``depth`` (which may be ``None``) are ``(H, W)`` float32. The same
    float32 arithmetic but for two things: divisions by a puff's own
    constants are multiplications by their reciprocals, and the Gaussian
    is a polynomial raised to a power, good to a few millionths of its
    value (``tests/unit/test_native_kernels.py`` holds the bound).
    """
    maps = [spray, stands, hidden] if depth is None else [spray, stands, hidden, depth]
    if not _enabled or spray.ndim != 2 or not _planes_alike(maps):
        return False
    if puffs.dtype != np.float64 or puffs.ndim != 2 or puffs.shape[1] != 15:
        return False
    assert _library is not None
    table = np.ascontiguousarray(puffs)
    _library.sm_spray_puffs(
        spray.ctypes.data,
        stands.ctypes.data,
        hidden.ctypes.data,
        None if depth is None else depth.ctypes.data,
        spray.shape[0],
        spray.shape[1],
        table.ctypes.data,
        len(table),
        float(feather),
        float(hides_wholly),
        float(stands_nearer),
    )
    return True


def normals(
    depth: np.ndarray, across: np.ndarray, down: np.ndarray, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Unit surface normals as three planes, and the depth step each was read across.

    ``None`` if not done. What ``normals._normals_and_jump`` computes after
    its blur, to the bit: ``depth`` is ``(H, W)`` float32 and ``across``
    and ``down`` each column's and each row's line of sight, float32.
    Differences are taken ``step`` pixels apart, so a frame has to be more
    than twice that each way.
    """
    if not _enabled or depth.dtype != np.float32 or depth.ndim != 2:
        return None
    height, width = depth.shape
    if step < 1 or min(height, width) <= 2 * step:
        return None
    if across.dtype != np.float32 or across.shape != (width,):
        return None
    if down.dtype != np.float32 or down.shape != (height,):
        return None
    assert _library is not None
    source = np.ascontiguousarray(depth)
    columns, rows = np.ascontiguousarray(across), np.ascontiguousarray(down)
    out = [np.empty((height, width), dtype=np.float32) for _ in range(4)]
    _library.sm_normals(
        *(plane.ctypes.data for plane in out),
        source.ctypes.data,
        columns.ctypes.data,
        rows.ctypes.data,
        height,
        width,
        int(step),
    )
    return out[0], out[1], out[2], out[3]


def _held(
    planes: list[np.ndarray | None], shape: tuple[int, ...]
) -> tuple[list[np.ndarray | None], list[int | None]] | None:
    """Optional float32 fields of one shape, made contiguous, and their addresses.

    ``None`` if one that is given is not float32 of ``shape``. The arrays
    are returned with the addresses so that they outlive the call they are
    passed to.
    """
    kept: list[np.ndarray | None] = []
    for plane in planes:
        if plane is None:
            kept.append(None)
            continue
        if plane.dtype != np.float32 or plane.shape != shape:
            return None
        kept.append(np.ascontiguousarray(plane))
    return kept, [None if plane is None else plane.ctypes.data for plane in kept]


def _floats(values: Any) -> np.ndarray:
    """Numbers as a contiguous float32 array, for a kernel's ``how``."""
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32).ravel())


def nir_reflectance(
    linear: np.ndarray,
    vegetation: np.ndarray | None,
    response: np.ndarray,
    gain: float,
    labelled: float,
    usual: float,
) -> np.ndarray | None:
    """``night_ir.near_infrared_reflectance`` in one pass; ``None`` if not done.

    ``linear`` is ``(H, W, 3)`` float32, ``vegetation`` ``(H, W)`` or
    ``None``, ``response`` the three weights and ``usual`` the mean of the
    frame's response. numpy's float32 arithmetic, but for the weighted sum
    of the three channels, which numpy leaves to its matrix product: to
    rounding, and used in fast mode.
    """
    if not _enabled or not _is_rgb_float32(linear):
        return None
    held = _held([vegetation], linear.shape[:2])
    if held is None:
        return None
    assert _library is not None
    scene, weights = np.ascontiguousarray(linear), _floats(response)
    out = np.empty(linear.shape[:2], dtype=np.float32)
    _library.sm_nir_reflectance(
        out.ctypes.data,
        scene.ctypes.data,
        held[1][0],
        out.size,
        weights.ctypes.data,
        float(gain),
        float(labelled),
        float(usual),
    )
    return out


def night_ir_light(
    reflectance: np.ndarray,
    exposure: np.ndarray,
    in_beam: np.ndarray | None,
    street: np.ndarray | None,
    beams: np.ndarray | None,
    night_sky: np.ndarray | None,
    ambient: float,
    floor_scale: float,
    sky_scale: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The infrared night's signal before its exposure, and the reflectance under a beam.

    ``None`` if not done. Planes of one shape, float32; all but the first
    two may be ``None``. ``NightIrMode.apply``'s arithmetic in its order.
    """
    if not _enabled or reflectance.dtype != np.float32 or reflectance.ndim != 2:
        return None
    held = _held([reflectance, exposure, in_beam, street, beams, night_sky], reflectance.shape)
    if held is None or held[1][1] is None:
        return None
    assert _library is not None
    signal = np.empty(reflectance.shape, dtype=np.float32)
    lifted = np.empty(reflectance.shape, dtype=np.float32)
    _library.sm_night_ir_light(
        signal.ctypes.data,
        lifted.ctypes.data,
        *held[1],
        signal.size,
        float(ambient),
        float(floor_scale),
        float(sky_scale),
    )
    return signal, lifted


def night_ir_expose(
    signal: np.ndarray,
    noise: np.ndarray,
    how: Any,
    retro: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    glare: np.ndarray | None = None,
    vehicle_lamps: tuple[np.ndarray, np.ndarray] | None = None,
    head_lamps: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray | None:
    """The infrared night from its exposure on, ``(H, W, 3)`` float32; ``None`` if not done.

    ``signal`` and ``noise`` are ``(H, W)`` float32. ``retro`` is the
    paint's return, the reflectance and the illuminator's exposure;
    ``glare`` a plane; ``vehicle_lamps`` the traffic's light and glow,
    ``(H, W, 3)`` each, and ``head_lamps`` the street's, a plane each.
    ``how`` is the level, the bloom, the grain's shot, read and gain, and
    the tint's three. ``NightIrMode.apply``'s arithmetic in its order.
    """
    if not _enabled or signal.dtype != np.float32 or signal.ndim != 2:
        return None
    shape = signal.shape
    planes = _held(
        [signal, noise, *(retro or (None,) * 3), glare, *(head_lamps or (None, None))], shape
    )
    fields = _held(list(vehicle_lamps or (None, None)), (*shape, 3))
    values = _floats(how)
    if planes is None or fields is None or planes[1][1] is None or values.size != 8:
        return None
    assert _library is not None
    at_signal, at_noise, at_retro, at_lifted, at_exposure, at_glare, at_head, at_glow = planes[1]
    out = np.empty((*shape, 3), dtype=np.float32)
    _library.sm_night_ir_expose(
        out.ctypes.data,
        at_signal,
        at_retro,
        at_lifted,
        at_exposure,
        at_glare,
        *fields[1],
        at_head,
        at_glow,
        at_noise,
        signal.size,
        values.ctypes.data,
    )
    return out


def night_colour_light(
    linear: np.ndarray,
    evened: np.ndarray | None,
    in_beam: np.ndarray | None,
    street: np.ndarray | None,
    beams: np.ndarray | None,
    night_sky: np.ndarray | None,
    how: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The colour night's signal before its exposure, and its brightness; ``None`` if not done.

    ``linear`` is ``(H, W, 3)`` float32 and the rest planes or ``None``.
    ``how`` is the sky's glow, the street's colour and the headlamps',
    three each, then the beam floor's scale, the night sky's, and the
    luminance's three weights. ``NightColorMode.apply``'s float32
    arithmetic, but for the luminance, which numpy leaves to its matrix
    product: to rounding, and used in fast mode.
    """
    if not _enabled or not _is_rgb_float32(linear):
        return None
    held = _held([evened, in_beam, street, beams, night_sky], linear.shape[:2])
    values = _floats(how)
    if held is None or values.size != 14:
        return None
    assert _library is not None
    scene = np.ascontiguousarray(linear)
    signal = np.empty(linear.shape, dtype=np.float32)
    brightness = np.empty(linear.shape[:2], dtype=np.float32)
    _library.sm_night_colour_light(
        signal.ctypes.data,
        brightness.ctypes.data,
        scene.ctypes.data,
        *held[1],
        brightness.size,
        values.ctypes.data,
    )
    return signal, brightness


def night_colour_expose(
    signal: np.ndarray,
    noise: np.ndarray,
    street_colour: np.ndarray,
    how: Any,
    vehicle_lamps: tuple[np.ndarray, np.ndarray] | None = None,
    head_lamps: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray | None:
    """The colour night from its exposure on, ``(H, W, 3)`` float32; ``None`` if not done.

    ``signal`` and ``noise`` are ``(H, W, 3)`` float32; ``vehicle_lamps``
    the traffic's light and glow, the same shape, and ``head_lamps`` the
    street's, a plane each, in ``street_colour``. ``how`` is the level,
    the bloom and the grain's shot, read and gain.
    ``NightColorMode.apply``'s arithmetic in its order.
    """
    if not _enabled or not _is_rgb_float32(signal):
        return None
    fields = _held([signal, noise, *(vehicle_lamps or (None, None))], signal.shape)
    planes = _held(list(head_lamps or (None, None)), signal.shape[:2])
    values, colour = _floats(how), _floats(street_colour)
    if fields is None or planes is None or fields[1][1] is None:
        return None
    if values.size != 5 or colour.size != 3:
        return None
    assert _library is not None
    at_signal, at_noise, at_light, at_glow = fields[1]
    out = np.empty(signal.shape, dtype=np.float32)
    _library.sm_night_colour_expose(
        out.ctypes.data,
        at_signal,
        at_light,
        at_glow,
        *planes[1],
        at_noise,
        signal.shape[0] * signal.shape[1],
        colour.ctypes.data,
        values.ctypes.data,
    )
    return out


def fog_lost(
    linear: np.ndarray, clipped: np.ndarray, one_way: np.ndarray, over: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The night fog before its halo: the radiance, what the fog removes, and its weight.

    ``None`` if not done. ``linear`` is ``(H, W, 3)`` float32, ``clipped``
    and ``one_way`` planes. ``linear * (1 + over * clipped)``, that times
    ``clipped * (1 - one_way)``, and the last summed over its channels:
    ``NightFog.apply``'s arithmetic to the bit.
    """
    if not _enabled or not _is_rgb_float32(linear):
        return None
    held = _held([clipped, one_way], linear.shape[:2])
    if held is None or None in held[1]:
        return None
    assert _library is not None
    scene = np.ascontiguousarray(linear)
    radiance = np.empty(linear.shape, dtype=np.float32)
    removed = np.empty(linear.shape, dtype=np.float32)
    weight = np.empty(linear.shape[:2], dtype=np.float32)
    _library.sm_fog_lost(
        radiance.ctypes.data,
        removed.ctypes.data,
        weight.ctypes.data,
        scene.ctypes.data,
        *held[1],
        weight.size,
        float(over),
    )
    return radiance, removed, weight


def fog_veil(
    radiance: np.ndarray,
    halo: np.ndarray,
    one_way: np.ndarray,
    how: Any,
    cones: np.ndarray | None = None,
    lit: np.ndarray | None = None,
    wall: np.ndarray | None = None,
) -> np.ndarray | None:
    """The night fog after its halo, ``(H, W, 3)`` float32; ``None`` if not done.

    ``radiance``, ``halo``, ``cones`` and ``lit`` are ``(H, W, 3)``
    float32, ``one_way`` and ``wall`` planes; the last three arguments may
    be ``None``. ``how`` is ten numbers: the halo's share, whether the
    light crosses the fog twice, the cones' level a channel, the veil's
    share, its lit part and its diffuse part a channel.
    ``NightFog.apply``'s arithmetic to the bit.
    """
    if not _enabled or not _is_rgb_float32(radiance):
        return None
    fields = _held([radiance, halo, cones, lit], radiance.shape)
    planes = _held([one_way, wall], radiance.shape[:2])
    values = _floats(how)
    if fields is None or planes is None or fields[1][1] is None or planes[1][0] is None:
        return None
    if values.size != 10:
        return None
    assert _library is not None
    at_radiance, at_halo, at_cones, at_lit = fields[1]
    out = np.empty(radiance.shape, dtype=np.float32)
    _library.sm_fog_veil(
        out.ctypes.data,
        at_radiance,
        at_halo,
        planes[1][0],
        at_cones,
        at_lit,
        planes[1][1],
        radiance.shape[0] * radiance.shape[1],
        values.ctypes.data,
    )
    return out


def snow_settled(
    level: np.ndarray,
    relief: np.ndarray,
    steady: np.ndarray,
    green: np.ndarray,
    grain: np.ndarray,
    luma: np.ndarray,
    around: np.ndarray,
    sky: np.ndarray,
    how: Any,
    lawn: np.ndarray | None = None,
    named: np.ndarray | None = None,
    patch: np.ndarray | None = None,
) -> np.ndarray | None:
    """Lying snow from the planes it is read from, ``(H, W)`` float32; ``None`` if not done.

    The end of ``RandomSnow``'s reading of where snow lies, in one pass
    where it was some twenty: all float32 planes of one shape, the last
    three optional. ``how`` is where relief begins and its span, the
    patchiness, the patches' contrast, the strength, the grain's depth
    and a canopy's cover. Fast mode's arithmetic to the bit.
    """
    if not _enabled or level.dtype != np.float32 or level.ndim != 2:
        return None
    held = _held(
        [level, relief, steady, green, lawn, named, patch, grain, luma, around, sky], level.shape
    )
    values = _floats(how)
    if held is None or values.size != 7:
        return None
    if any(at is None for at in held[1][:4] + held[1][7:]):
        return None
    assert _library is not None
    settled = np.empty(level.shape, dtype=np.float32)
    _library.sm_snow_settled(settled.ctypes.data, *held[1], settled.size, values.ctypes.data)
    return settled


def ground_tile(depth: np.ndarray, how: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Each pixel's place in ``ground_noise``'s tile, across and along; ``None`` if not done.

    Two ``(H, W)`` float32 maps for ``cv2.remap``, from ``depth``, ``(H,
    W)`` float32. ``how`` is the focal length, the pitch's cosine and
    sine, tile pixels a metre, the fold's span and its reciprocal, and the
    two offsets. ``ground.ground_noise``'s fast-mode arithmetic to the bit.
    """
    values = _floats(how)
    if not _enabled or depth.dtype != np.float32 or depth.ndim != 2 or values.size != 8:
        return None
    assert _library is not None
    source = np.ascontiguousarray(depth)
    across = np.empty(depth.shape, dtype=np.float32)
    along = np.empty(depth.shape, dtype=np.float32)
    _library.sm_ground_tile(
        across.ctypes.data,
        along.ctypes.data,
        source.ctypes.data,
        depth.shape[0],
        depth.shape[1],
        values.ctypes.data,
    )
    return across, along


def street_light(
    light: np.ndarray,
    points: np.ndarray,
    raised: np.ndarray,
    lamps: np.ndarray,
    up: np.ndarray,
    spill: float,
) -> bool:
    """Add every street lamp's light on every point into ``light``; ``False`` if not done.

    The sum of ``night_light.street_light``: ``points`` is ``(h, w, 3)``
    float32, ``light`` and ``raised`` ``(h, w)``, ``lamps`` ``(n, 4)`` and
    ``up`` three, all float32. numpy's arithmetic in its order, but for the
    cosine's power of one and a half, a root and a product here: the same
    map to the last bit of a float or one off it.
    """
    if not _enabled or points.dtype != np.float32 or points.ndim != 3 or points.shape[2] != 3:
        return False
    if not points.flags.c_contiguous or not _planes_alike([light, raised]):
        return False
    if light.shape != points.shape[:2]:
        return False
    if lamps.dtype != np.float32 or lamps.ndim != 2 or lamps.shape[1] != 4:
        return False
    if up.dtype != np.float32 or up.size != 3:
        return False
    assert _library is not None
    heads, upright = np.ascontiguousarray(lamps), np.ascontiguousarray(up)
    _library.sm_street_light(
        light.ctypes.data,
        points.ctypes.data,
        raised.ctypes.data,
        light.size,
        heads.ctypes.data,
        len(heads),
        upright.ctypes.data,
        float(spill),
    )
    return True


def headlamp_beams(
    light: np.ndarray,
    along_first: np.ndarray,
    along_second: np.ndarray,
    above: np.ndarray,
    low: np.ndarray,
    beams: np.ndarray,
    how: Any,
) -> bool:
    """Add every lit vehicle's dipped beams into ``light``; ``False`` if not done.

    The sum of ``night_light.headlamp_light``. The first five are float32
    planes of one shape: the light, each point's place along the ground's
    two axes, its height above the ground, and nought where it is well
    above any beam. ``beams`` is ``(n, 6)`` float32: a vehicle's front
    along the two axes, its heading along them, its scale and what the
    lens gets of its pool. ``how`` is a lamp's width and widening, half
    the lamps' spacing, each lamp's level, where it halves, its reach and
    its height. numpy's arithmetic, with a polynomial for its
    exponential: within a few millionths of a pool.
    """
    values = _floats(how)
    if not _enabled or not _planes_alike([light, along_first, along_second, above, low]):
        return False
    if beams.dtype != np.float32 or beams.ndim != 2 or beams.shape[1] != 6 or values.size != 7:
        return False
    assert _library is not None
    table = np.ascontiguousarray(beams)
    _library.sm_headlamp_beams(
        light.ctypes.data,
        along_first.ctypes.data,
        along_second.ctypes.data,
        above.ctypes.data,
        low.ctypes.data,
        light.size,
        table.ctypes.data,
        len(table),
        values.ctypes.data,
    )
    return True


def wheel_tracks(
    piece: np.ndarray,
    across: np.ndarray,
    width_px: np.ndarray,
    weight: np.ndarray,
    ragged: np.ndarray | None,
    worn: np.ndarray | None,
    edges: Any,
    use: np.ndarray,
    how: Any,
) -> np.ndarray | None:
    """The weight of the wheel tracks down each lane, ``(H, W)`` float32; ``None`` if not done.

    What ``road.wheel_tracks`` computes once its lanes' use is drawn.
    ``piece`` is ``(H, W)`` int32, each pixel's road; ``across``,
    ``width_px``, ``weight`` and the optional ``ragged`` and ``worn`` are
    float32 planes of that shape. ``edges`` is each road's lane lines, a
    sequence of float64 arrays with road 0's first; ``use`` is 64 float32.
    ``how`` is the band, the wander, half the wheels' spacing, the wear's
    floor and its gain. The numpy code's arithmetic to the bit.
    """
    if not _enabled or piece.dtype != np.int32 or piece.ndim != 2:
        return None
    held = _held([across, width_px, weight, ragged, worn], piece.shape)
    values = _floats(how)
    if held is None or None in held[1][:3] or values.size != 5:
        return None
    if use.dtype != np.float32 or use.size != 64 or len(edges) < 1:
        return None
    lines = [np.ascontiguousarray(road, dtype=np.float64).ravel() for road in edges]
    first = np.zeros(len(lines) + 1, dtype=np.int32)
    first[1:] = np.cumsum([len(road) for road in lines])
    every = np.ascontiguousarray(np.concatenate(lines)) if first[-1] else np.zeros(1)
    assert _library is not None
    roads = np.ascontiguousarray(piece)
    lanes_use = np.ascontiguousarray(use)
    out = np.empty(piece.shape, dtype=np.float32)
    _library.sm_wheel_tracks(
        out.ctypes.data,
        roads.ctypes.data,
        *held[1],
        out.size,
        every.ctypes.data,
        first.ctypes.data,
        len(lines) - 1,
        lanes_use.ctypes.data,
        values.ctypes.data,
    )
    return out


def lamps_lit(out: np.ndarray, lamp_light: np.ndarray, through: np.ndarray) -> bool:
    """Add the traffic's lamps, up to full scale, through the veil; ``False`` if not done.

    ``out += minimum(lamp_light, 1) * through[..., None]`` in place, on
    ``(H, W, 3)`` float32 fields and an ``(H, W)`` plane: the expression's
    arithmetic to the bit.
    """
    if not _enabled or not _is_rgb_float32(out) or not out.flags.c_contiguous:
        return False
    if not out.flags.writeable or not _is_rgb_float32(lamp_light) or lamp_light.shape != out.shape:
        return False
    if not _is_plane_of(through, out):
        return False
    assert _library is not None
    lamps, veil = np.ascontiguousarray(lamp_light), np.ascontiguousarray(through)
    _library.sm_lamps_lit(out.ctypes.data, lamps.ctypes.data, veil.ctypes.data, veil.size)
    return True


def spray_and_lamps(
    out: np.ndarray,
    spray: np.ndarray | None,
    lamps: tuple[np.ndarray, np.ndarray] | None,
    through: np.ndarray,
    lit: np.ndarray,
    glow_in_rain: float,
) -> bool:
    """Lay the spray over ``out`` and the lamps through it, in place; ``False`` if not done.

    ``RandomRain.apply`` between its road and its streaks. ``out`` is
    ``(H, W, 3)`` float32, ``spray`` and ``through`` planes, ``lamps`` the
    traffic's light and glow, ``(H, W, 3)`` each; ``lit`` is the light
    the spray is seen in, three float32. Either of ``spray`` and ``lamps``
    may be ``None``. The numpy code's arithmetic to the bit.
    """
    if not _enabled or not _is_rgb_float32(out) or not out.flags.c_contiguous:
        return False
    if not out.flags.writeable or lit.dtype != np.float32 or lit.size != 3:
        return False
    planes = _held([spray, through], out.shape[:2])
    fields = _held(list(lamps or (None, None)), out.shape)
    if planes is None or fields is None or planes[1][1] is None:
        return False
    assert _library is not None
    light = np.ascontiguousarray(lit).ravel()
    _library.sm_spray_and_lamps(
        out.ctypes.data,
        planes[1][0],
        *fields[1],
        planes[1][1],
        out.shape[0] * out.shape[1],
        light.ctypes.data,
        float(glow_in_rain),
    )
    return True


def road_along(
    height: int, width: int, centre: float, horizon: float, bend: float
) -> np.ndarray | None:
    """The way a road runs at every pixel, ``(H, W, 2)`` float32; ``None`` if not done.

    ``road.model_frame``'s ``along`` for a road that runs to ``centre`` on
    a horizon at row ``horizon`` with a ``bend``: the numpy code's double
    arithmetic to the bit.
    """
    if not _enabled or height < 1 or width < 1:
        return None
    assert _library is not None
    along = np.empty((height, width, 2), dtype=np.float32)
    _library.sm_road_along(
        along.ctypes.data, height, width, float(centre), float(horizon), float(bend)
    )
    return along


def road_places(
    shape: tuple[int, int],
    top: int,
    offset: np.ndarray,
    at: np.ndarray,
    whole: np.ndarray,
    below: np.ndarray,
    of_position: np.ndarray,
    firsts: np.ndarray,
    widths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """A road frame's carriageway, place across it and width in pixels; ``None`` if not done.

    Three ``shape`` arrays (int32, float32, float32), filled from row
    ``top`` down from ``offset`` (float64), ``at`` (int64) and ``whole``
    (bool), each ``(shape[0] - top, shape[1])``, ``below`` a float64 a row,
    and the three tables ``road.model_frame`` makes: each position's
    carriageway, and each carriageway's first place and width. Above
    ``top`` there is no road. The numpy code's arithmetic to the bit.
    """
    height, width = shape
    rows = height - top
    if not _enabled or rows < 1 or offset.shape != (rows, width):
        return None
    if offset.dtype != np.float64 or at.dtype != np.int64 or at.shape != offset.shape:
        return None
    if whole.dtype != np.bool_ or whole.shape != offset.shape or below.size != rows:
        return None
    if of_position.dtype != np.int32 or firsts.dtype != np.float64 or widths.dtype != np.float64:
        return None
    if len(firsts) != len(widths) or int(of_position.max()) >= len(firsts):
        return None
    if int(at.min()) < 0 or int(at.max()) >= len(of_position) or int(of_position.min()) < 0:
        return None
    assert _library is not None
    held = [np.ascontiguousarray(a) for a in (offset, at, whole)]
    under = np.ascontiguousarray(below, dtype=np.float64).ravel()
    tables = [np.ascontiguousarray(a) for a in (of_position, firsts, widths)]
    piece = np.zeros(shape, dtype=np.int32)
    across = np.zeros(shape, dtype=np.float32)
    span = np.ones(shape, dtype=np.float32)
    _library.sm_road_places(
        piece[top:].ctypes.data,
        across[top:].ctypes.data,
        span[top:].ctypes.data,
        *(a.ctypes.data for a in held),
        under.ctypes.data,
        rows,
        width,
        *(a.ctypes.data for a in tables),
    )
    return piece, across, span
