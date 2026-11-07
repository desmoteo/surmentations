"""Fast and exact arithmetic.

Some operations in this library have a faster form that is not quite the
same operation: a blur 58 pixels wide computed on a smaller copy of the
frame, a division replaced by a multiplication with the reciprocal, a sum in
float32 where the reference code happens to use float64. Each differs from
the reference by a stated, small amount, and none of them is a change to a
physical model.

**Fast mode**, the default, uses those forms. **Exact mode** computes what the
reference numpy and OpenCV code computes: use it to compare frames with ones
made by an earlier version, or when a skyline has to be reproduced to the
pixel. The bounds on the difference between the two are in
``docs/native_kernels.md`` and are asserted in ``tests/unit/test_modes.py``.

The mode is a property of the process::

    import surmentations

    surmentations.set_fast(False)      # exact from here on
    with surmentations.exact():        # or for one block
        frame = transform(image=image)["image"]

``SURMENTATIONS_FAST=0`` in the environment selects exact mode at import.
That is the form to use with a ``DataLoader`` whose workers are spawned, not
forked, since a spawned worker imports the library afresh and does not see a
``set_fast`` call made in the parent.

The mode is independent of whether the compiled kernels are in use
(:mod:`surmentations._native`): either mode runs with or without them.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

__all__ = ["exact", "fast", "scalar", "set_fast"]

_fast = os.environ.get("SURMENTATIONS_FAST", "").lower() not in ("0", "off", "false", "no")


def fast() -> bool:
    """Whether fast mode is on."""
    return _fast


def set_fast(flag: bool) -> None:
    """Switch fast mode on or off for this process."""
    global _fast
    _fast = bool(flag)


@contextmanager
def exact() -> Iterator[None]:
    """Run the block in exact mode, whatever the process is set to."""
    before = _fast
    set_fast(False)
    try:
        yield
    finally:
        set_fast(before)


def scalar(value: float) -> float | np.float64:
    """A scalar for arithmetic with a float32 array.

    numpy keeps an array float32 when it meets a Python ``float`` and widens
    it to float64 when it meets an ``np.float64``, which is what ``np.cos``
    and its kind return. Several per-pixel fields in the reference code are
    float64 for that reason alone. Fast mode passes the Python float, and the
    field stays float32 at half the memory traffic; exact mode passes the
    ``np.float64`` the reference code had.
    """
    return float(value) if _fast else np.float64(value)
