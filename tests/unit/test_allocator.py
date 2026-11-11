"""The allocator setting is the process's, so it is tried in one of its own."""

from __future__ import annotations

import subprocess
import sys


def test_the_allocator_setting_is_made_or_declined_without_harm() -> None:
    """glibc takes it; elsewhere it says no. Either way arrays still work."""
    code = (
        "import numpy as np, weathermentations as w;"
        "made = w.keep_freed_memory(); again = w.keep_freed_memory();"
        "a = np.ones((1024, 1024, 3), np.float32) * 2; del a;"
        "b = np.zeros((1024, 1024, 3), np.float32); print(made, again, float(b.sum()))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    made, again, total = out.stdout.split()
    assert made == again and made in ("True", "False")
    assert total == "0.0"
