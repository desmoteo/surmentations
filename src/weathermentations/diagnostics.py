"""Opt-in recording of the random draws a transform actually made.

Why this exists
---------------
Almost every quantity in this library is *derived*: a rain rate fixes the
Marshall-Palmer slope, which with the simulated volume fixes how many drops
are in frame, which with each drop's diameter and distance fixes its streak's
length, width and opacity. Looking at the output image tells you whether it
*seems* plausible. It does not tell you whether the chain that produced it is
self-consistent — whether the drop count really is what that rate implies, or
whether a sampled value even landed inside the range it was supposed to come
from.

So this records the draws themselves. Two mechanisms:

* Every transform's returned parameter dictionary is captured automatically
  and **summarised**, since most of it is megabytes of masks and per-particle
  arrays that are useless to read but very informative to reduce.
* Transforms additionally call :func:`record_draw` for physical quantities
  that are local variables and never reach the returned dictionary — the
  sampled rain rate, the exposure time, the aperture. Those are exactly the
  values needed to check the derived chain, and they would otherwise be
  invisible.

Recording is off by default and costs nothing when off: the wrapper checks a
single module-level flag.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

_ACTIVE: _Recorder | None = None

# Arrays are summarised rather than stored; this many entries are shown in
# full for short ones (a colour, a gain triple) where the values themselves
# are the interesting part.
_SMALL_ARRAY = 4


class _Recorder:
    """Collects one record per transform call."""

    def __init__(self) -> None:
        """Start with no records and nothing pending."""
        self.records: list[dict[str, Any]] = []
        self._pending: dict[str, Any] = {}

    def note(self, **values: Any) -> None:
        """Hold named values a transform drew, until its parameters are complete.

        A value given as a function is called now: a statistic of a whole
        frame is only computed when someone is recording.
        """
        for name, value in values.items():
            # A statistic of a whole frame is passed as a function, so that
            # it is computed here, when someone is recording, and not on
            # every frame of every training run.
            self._pending[name] = value() if callable(value) else value

    def commit(self, transform: Any, params: dict[str, Any]) -> None:
        """Record one call: the transform, its arguments, what it drew and what it returned."""
        record: dict[str, Any] = {"t": type(transform).__name__}
        record["init"] = _summarise(_init_args(transform))
        if self._pending:
            record["drew"] = _summarise(self._pending)
        record["out"] = _summarise(params)
        self.records.append(record)
        self._pending = {}


def _init_args(transform: Any) -> dict[str, Any]:
    """Constructor arguments, i.e. the ranges a draw should have come from."""
    try:
        names = transform.get_transform_init_args_names()
    except Exception:  # pragma: no cover - defensive; Albumentations internal
        return {}
    args = {}
    for name in names:
        value = getattr(transform, name, None)
        # The depth provider and other objects are not data.
        if isinstance(value, (int, float, str, bool, tuple, list, type(None))):
            args[name] = value
    return args


def _summarise(value: Any) -> Any:
    """Reduce a value to something compact enough to read a million of."""
    if isinstance(value, dict):
        return {k: _summarise(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return {"n": 0}
        flat = value.astype(np.float64).ravel()
        summary = {
            "n": int(value.size),
            "shape": list(value.shape),
            "min": round(float(flat.min()), 5),
            "p50": round(float(np.median(flat)), 5),
            "max": round(float(flat.max()), 5),
            "mean": round(float(flat.mean()), 5),
        }
        # A tiny array is cheap enough to keep in full, but it keeps the same
        # shape as every other one. Emitting a bare list here instead — which
        # this did — meant a field's schema changed with the particle count,
        # so `out.flakes.radius.p50` read fine on a heavy frame and raised
        # TypeError on a light one. That is the worst place for an
        # inconsistency: the low-severity frames are exactly the ones an audit
        # of "is severity 1 doing anything" has to read.
        if value.size <= _SMALL_ARRAY:
            summary["values"] = [round(float(x), 5) for x in flat]
        return summary
    if isinstance(value, (list, tuple)):
        # A list of per-particle tuples reduces to its length; a short tuple
        # (a range, a position) is worth keeping verbatim.
        if len(value) > _SMALL_ARRAY:
            return {"n": len(value)}
        return [_summarise(v) for v in value]
    if isinstance(value, (bool, str, type(None))):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    return str(type(value).__name__)


def record_draw(**values: Any) -> None:
    """Note physical quantities a transform sampled but does not return.

    A no-op unless recording is active, so transforms can call it
    unconditionally. A value that costs something to work out, the median of
    a depth map say, is passed as a function of no arguments and is called
    only when recording is active::

        record_draw(beam_falloff=falloff, depth_p50_m=lambda: float(np.median(depth_m)))
    """
    if _ACTIVE is not None:
        _ACTIVE.note(**values)


def instrument(method: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Wrap ``get_params_dependent_on_data`` so its result is recorded.

    Applied automatically by :class:`~weathermentations.base.WeatherTransform`
    to every subclass, so no transform has to opt in or even know about it.
    """

    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Call the method and, while a recorder is active, record what it returned."""
        params = method(self, *args, **kwargs)
        if _ACTIVE is not None:
            _ACTIVE.commit(self, params)
        return params

    wrapper.__weathermentations_instrumented__ = True  # type: ignore[attr-defined]
    return wrapper


@contextmanager
def recording() -> Iterator[list[dict[str, Any]]]:
    """Collect every transform call made inside the block.

    ::

        with recording() as log:
            transform(image=image)
        write_jsonl(log, "draws.jsonl")
    """
    global _ACTIVE
    previous = _ACTIVE
    recorder = _Recorder()
    _ACTIVE = recorder
    try:
        yield recorder.records
    finally:
        _ACTIVE = previous


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """One compact JSON object per line."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
