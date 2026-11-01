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
    "dirt_composite",
    "dirt_emphasise",
    "disabled",
    "droplet_composite",
    "edge_preserving",
    "enabled",
    "encode_f32",
    "encode_u8",
    "grow_alike",
    "overcast_blend",
    "place_apart",
    "render_flakes",
    "render_streaks",
    "set_enabled",
    "set_tier",
    "mirror_rows",
    "tier",
    "tiers",
    "veil_blend",
    "water_lobes",
    "wet_composite",
    "wet_reflection",
]

# Must match WM_ABI_VERSION in native/wm_kernels.h. A library left over from
# an older checkout is ignored, not called with the wrong arguments.
_ABI_VERSION = 11

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
    "wm_edge_preserving": (
        ctypes.c_int,
        [_pointer, _pointer, ctypes.c_ssize_t, ctypes.c_ssize_t]
        + [ctypes.c_float, ctypes.c_float, ctypes.c_int],
    ),
    "wm_dirt_emphasise": (None, [_pointer] * 3 + [ctypes.c_ssize_t, ctypes.c_float]),
    "wm_dirt_composite": (
        None,
        [_pointer] * 6 + [ctypes.c_ssize_t] + [ctypes.c_float] * 4,
    ),
    "wm_veil_blend": (None, [_pointer] * 7 + [ctypes.c_ssize_t]),
    "wm_overcast_blend": (None, [_pointer] * 6 + [ctypes.c_ssize_t] + [ctypes.c_float] * 3),
    "wm_grow_alike": (None, [_pointer] * 7 + [ctypes.c_ssize_t] + [ctypes.c_float] * 5),
    "wm_place_apart": (
        ctypes.c_ssize_t,
        [_pointer] * 4 + [ctypes.c_ssize_t] * 2 + [ctypes.c_double, _pointer],
    ),
    "wm_water_lobes": (
        None,
        [_pointer] * 2
        + [ctypes.c_ssize_t] * 2
        + [_pointer] * 2
        + [ctypes.c_ssize_t, ctypes.c_int, _pointer]
        + [ctypes.c_double] * 2,
    ),
    "wm_droplet_composite": (None, [_pointer] * 5 + [ctypes.c_ssize_t]),
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
    failed = _library.wm_edge_preserving(
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
    _library.wm_dirt_emphasise(
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
    _library.wm_dirt_composite(
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
    _library.wm_veil_blend(
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
    _library.wm_overcast_blend(
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
    _library.wm_grow_alike(
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
    found = _library.wm_place_apart(
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
    _library.wm_water_lobes(
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
    _library.wm_droplet_composite(
        out.ctypes.data,
        source.ctypes.data,
        None if light is None else light.ctypes.data,
        None if share is None else share.ctypes.data,
        None if drops is None else drops.ctypes.data,
        pixels,
    )
    return out
