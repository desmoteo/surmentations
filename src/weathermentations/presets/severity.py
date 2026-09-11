"""ImageNet-C-style severity presets (1-5), layered on top of the physical range API.

These are a convenience layer, **not** a separate code path: each entry is
just a dict of the same ``(min, max)`` range kwargs every transform's
constructor already accepts. Unpack one into a transform's constructor for a
quick severity sweep::

    from weathermentations.lens import WaterDroplets
    from weathermentations.presets.severity import SEVERITY

    transform = WaterDroplets(p=1.0, **SEVERITY[3]["WaterDroplets"])

Severity controls each effect's *intensity given that it fires* (density,
opacity, beta, ...); it intentionally does not set ``p`` (the probability the
effect fires at all), which stays the caller's choice. Depth-conditioned
transforms (``RandomFog``/``RandomRain``/``RandomSnow``) still need a
``depth_provider`` passed separately — that's pipeline wiring, not a severity
knob, so it's excluded from the preset dict here.
"""

from __future__ import annotations

from typing import Any

SEVERITY: dict[int, dict[str, dict[str, Any]]] = {
    1: {
        "WaterDroplets": {
            "droplet_count_range": (2, 5),
            "streak_probability": 0.1,
        },
        "DirtSoil": {
            "blob_count_range": (1, 3),
            "opacity_range": (0.1, 0.25),
        },
        "SunFlare": {
            "intensity_range": (0.2, 0.4),
            "ghost_count_range": (1, 2),
        },
        "RandomFog": {
            "beta_range": (0.005, 0.015),
        },
        "RandomRain": {
            "streak_density_range": (30, 80),
            "veil_beta_range": (0.002, 0.008),
        },
        "RandomSnow": {
            "flake_density_range": (50, 150),
            "accumulation_strength_range": (0.05, 0.15),
        },
    },
    2: {
        "WaterDroplets": {
            "droplet_count_range": (4, 10),
            "streak_probability": 0.2,
        },
        "DirtSoil": {
            "blob_count_range": (2, 5),
            "opacity_range": (0.2, 0.35),
        },
        "SunFlare": {
            "intensity_range": (0.35, 0.6),
            "ghost_count_range": (2, 4),
        },
        "RandomFog": {
            "beta_range": (0.01, 0.03),
        },
        "RandomRain": {
            "streak_density_range": (80, 150),
            "veil_beta_range": (0.005, 0.015),
        },
        "RandomSnow": {
            "flake_density_range": (150, 300),
            "accumulation_strength_range": (0.1, 0.25),
        },
    },
    3: {
        "WaterDroplets": {
            "droplet_count_range": (8, 18),
            "streak_probability": 0.3,
        },
        "DirtSoil": {
            "blob_count_range": (4, 8),
            "opacity_range": (0.3, 0.5),
        },
        "SunFlare": {
            "intensity_range": (0.5, 0.9),
            "ghost_count_range": (3, 6),
        },
        "RandomFog": {
            "beta_range": (0.03, 0.06),
        },
        "RandomRain": {
            "streak_density_range": (150, 300),
            "veil_beta_range": (0.01, 0.03),
        },
        "RandomSnow": {
            "flake_density_range": (300, 500),
            "accumulation_strength_range": (0.2, 0.4),
        },
    },
    4: {
        "WaterDroplets": {
            "droplet_count_range": (15, 30),
            "streak_probability": 0.4,
        },
        "DirtSoil": {
            "blob_count_range": (7, 14),
            "opacity_range": (0.4, 0.65),
        },
        "SunFlare": {
            "intensity_range": (0.8, 1.3),
            "ghost_count_range": (5, 8),
        },
        "RandomFog": {
            "beta_range": (0.06, 0.12),
        },
        "RandomRain": {
            "streak_density_range": (300, 550),
            "veil_beta_range": (0.03, 0.06),
        },
        "RandomSnow": {
            "flake_density_range": (500, 800),
            "accumulation_strength_range": (0.35, 0.55),
        },
    },
    5: {
        "WaterDroplets": {
            "droplet_count_range": (25, 45),
            "droplet_radius_range": (6.0, 30.0),
            "streak_probability": 0.5,
        },
        "DirtSoil": {
            "blob_count_range": (12, 22),
            "opacity_range": (0.55, 0.85),
        },
        "SunFlare": {
            "intensity_range": (1.2, 1.8),
            "ghost_count_range": (7, 11),
        },
        "RandomFog": {
            "beta_range": (0.12, 0.25),
        },
        "RandomRain": {
            "streak_density_range": (550, 900),
            "veil_beta_range": (0.06, 0.12),
        },
        "RandomSnow": {
            "flake_density_range": (800, 1200),
            "accumulation_strength_range": (0.5, 0.75),
        },
    },
}
