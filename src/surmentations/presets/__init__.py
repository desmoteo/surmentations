"""Severity ladders and ready-made pipelines.

:data:`SEVERITY` and :data:`GROUND_SEVERITY` give each transform's arguments
at five levels; :func:`surveillance` and :func:`surveillance_lite` assemble
whole pipelines in physical order, and :func:`weather_state` draws the
weather they share.
"""

from surmentations.presets.pipeline import physical_order, stage_of
from surmentations.presets.pipelines import (
    surveillance,
    surveillance_lite,
    weather_state,
)
from surmentations.presets.severity import GROUND_SEVERITY, SEVERITY

__all__ = [
    "GROUND_SEVERITY",
    "SEVERITY",
    "physical_order",
    "stage_of",
    "surveillance",
    "surveillance_lite",
    "weather_state",
]
