"""Keeping freed frames, for the allocator to hand back.

Every numpy expression over a frame allocates a frame-sized array and frees
it a moment later. glibc's allocator, as it comes, gives blocks of that size
back to the operating system when they are freed and asks for them again at
the next expression, and each time the kernel has to map and zero the pages
anew. For transforms made of many short numpy passes that was a large share
of the time: ``SensorNoise`` at 512 pixels took 12.7 ms as the allocator
comes and 5.9 ms with it told to keep what is freed; ``SunFlare``,
``DirtSoil`` and ``WaterDroplets`` at 1024 pixels gained 6 to 10%. Transforms
that spend their time inside one OpenCV filter or one compiled kernel gain
nothing.

This is a setting of the process, not of this library, so it is not made on
import. Call :func:`keep_freed_memory` once where the transforms run, in a
``DataLoader``'s ``worker_init_fn`` for one::

    def worker_init_fn(worker_id):
        weathermentations.keep_freed_memory()

The cost is that the process holds on to the most memory it has needed
instead of shrinking back. In the measurements above the peak itself did not
move.

A pool of scratch buffers inside the compiled kernels was measured as well,
and gains nothing: see ``docs/native_kernels.md``.
"""

from __future__ import annotations

import ctypes

__all__ = ["keep_freed_memory"]

# <malloc.h>
_M_TRIM_THRESHOLD = -1
_M_MMAP_THRESHOLD = -3

# The largest block malloc will take from its own heap and not map separately:
# glibc's ceiling for the setting, and room for a float32 RGB frame of 1600
# pixels square.
_MMAP_THRESHOLD_BYTES = 32 << 20
# How much freed memory at the top of the heap is kept before any goes back.
_TRIM_THRESHOLD_BYTES = 1 << 30


def keep_freed_memory() -> bool:
    """Have glibc's allocator keep freed frames and reuse them.

    Sets ``M_MMAP_THRESHOLD`` and ``M_TRIM_THRESHOLD`` for the calling
    process; both are needed, and either alone measured slower than neither.
    Returns whether the settings were made: ``False`` where the C library is
    not glibc (musl, macOS, Windows), which changes nothing.
    """
    try:
        mallopt = ctypes.CDLL(None).mallopt
    except (OSError, AttributeError, TypeError):
        return False
    mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
    mallopt.restype = ctypes.c_int
    kept = mallopt(_M_MMAP_THRESHOLD, _MMAP_THRESHOLD_BYTES)
    held = mallopt(_M_TRIM_THRESHOLD, _TRIM_THRESHOLD_BYTES)
    return bool(kept and held)
