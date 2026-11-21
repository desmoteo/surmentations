"""Severity ladders and ready-made pipelines.

:data:`SEVERITY` and :data:`GROUND_SEVERITY` give each transform's arguments
at five levels; :func:`surveillance` and :func:`surveillance_lite` assemble
whole pipelines in physical order, and :func:`weather_state` draws the
weather they share, by the share of time each of its states holds
(:func:`weather_shares`, :func:`rebalanced`). :func:`read_source` says what
a frame already is, and :class:`BySource` sends it down the branch for
that.
"""

from surmentations.presets.pipeline import physical_order, stage_of
from surmentations.presets.pipelines import (
    WEATHER_STATES,
    rebalanced,
    surveillance,
    surveillance_lite,
    weather_shares,
    weather_state,
)
from surmentations.presets.severity import GROUND_SEVERITY, SEVERITY
from surmentations.presets.source import BySource, Source, read_source

__all__ = [
    "GROUND_SEVERITY",
    "SEVERITY",
    "WEATHER_STATES",
    "BySource",
    "Source",
    "physical_order",
    "read_source",
    "rebalanced",
    "stage_of",
    "surveillance",
    "surveillance_lite",
    "weather_shares",
    "weather_state",
]
