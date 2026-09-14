"""ImageNet-C-style severity presets (1-5), layered on top of the physical range API.

These are a convenience layer, **not** a separate code path: each entry is
just a dict of the same kwargs every transform's constructor already accepts.
Unpack one into a transform for a quick severity sweep::

    from weathermentations.lens import WaterDroplets
    from weathermentations.presets.severity import SEVERITY

    transform = WaterDroplets(p=1.0, **SEVERITY[3]["WaterDroplets"])

Severity controls each effect's *intensity given that it fires*; it
intentionally does not set ``p`` (the probability the effect fires at all),
which stays the caller's choice. Depth-conditioned transforms
(``RandomFog``/``RandomRain``/``RandomSnow``) still need a ``depth_provider``
passed separately — that is pipeline wiring, not a severity knob.

Size-like values are fractions of the image diagonal, matching the
transforms' own resolution-independent parameterisation.
"""

from __future__ import annotations

from typing import Any


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _build() -> dict[int, dict[str, dict[str, Any]]]:
    """Interpolate every effect's knobs from 'barely there' to 'extreme'.

    Writing the endpoints once and interpolating keeps the five levels
    monotonic by construction, which is what a severity ladder is for — and
    avoids the transcription mistakes that come with hand-writing 5 x 6
    dictionaries.
    """
    severities: dict[int, dict[str, dict[str, Any]]] = {}
    for level in range(1, 6):
        t = (level - 1) / 4.0
        severities[level] = {
            "WaterDroplets": {
                "droplet_count_range": (int(_lerp(8, 90, t)), int(_lerp(25, 220, t))),
                "droplet_radius_range": (_lerp(0.002, 0.004, t), _lerp(0.012, 0.038, t)),
                "condensation_coverage_range": (_lerp(0.04, 0.3, t), _lerp(0.15, 0.75, t)),
                "condensation_edge_softness_range": (_lerp(0.45, 0.25, t), _lerp(0.7, 0.5, t)),
                "condensation_milkiness_range": (_lerp(0.1, 0.4, t), _lerp(0.3, 0.8, t)),
                "droplet_defocus_range": (_lerp(0.05, 0.2, t), _lerp(0.2, 0.5, t)),
                "trail_probability": _lerp(0.1, 0.6, t),
            },
            "DirtSoil": {
                "smudge_count_range": (int(_lerp(2, 8, t)), int(_lerp(5, 18, t))),
                "coverage_range": (_lerp(0.02, 0.12, t), _lerp(0.08, 0.35, t)),
                "highlight_boost_range": (_lerp(1.0, 3.0, t), _lerp(3.0, 7.0, t)),
                "veil_strength_range": (_lerp(0.08, 0.4, t), _lerp(0.25, 0.8, t)),
                "water_spot_count_range": (0, int(_lerp(5, 30, t))),
            },
            "SunFlare": {
                "intensity_range": (_lerp(0.2, 1.2, t), _lerp(0.45, 1.9, t)),
                "ghost_count_range": (int(_lerp(1, 6, t)), int(_lerp(3, 11, t))),
            },
            "RandomFog": {
                # MOR = 3.912 / beta, so this spans roughly 800 m down to 30 m
                # of visibility.
                "beta_range": (_lerp(0.004, 0.07, t), _lerp(0.012, 0.13, t)),
                "scattering_gamma_range": (_lerp(0.8, 0.0, t), _lerp(2.0, 0.4, t)),
                "forward_scatter_range": (_lerp(0.0005, 0.003, t), _lerp(0.002, 0.008, t)),
            },
            "RandomRain": {
                "streak_density_range": (int(_lerp(400, 4000, t)), int(_lerp(1200, 9000, t))),
                "drop_diameter_range": (_lerp(0.002, 0.004, t), _lerp(0.003, 0.006, t)),
                "veil_beta_range": (_lerp(0.002, 0.03, t), _lerp(0.01, 0.09, t)),
                "wetness_probability": _lerp(0.15, 0.9, t),
            },
            "RandomSnow": {
                "flake_density_range": (int(_lerp(300, 2500, t)), int(_lerp(900, 5000, t))),
                "bokeh_strength_range": (_lerp(0.4, 1.2, t), _lerp(1.2, 3.0, t)),
                "veil_beta_range": (_lerp(0.004, 0.06, t), _lerp(0.02, 0.16, t)),
                "accumulation_strength_range": (_lerp(0.05, 0.45, t), _lerp(0.18, 0.75, t)),
            },
        }
    return severities


SEVERITY: dict[int, dict[str, dict[str, Any]]] = _build()
