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

from weathermentations.atmosphere.rain import MIN_RAIN_RATE_MM_H


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
                # Level 1 must be mild, not absent: at the previous floor it
                # moved the frame by 0.94 of an 8-bit level, i.e. below the
                # quantisation it is rendered into.
                "smudge_count_range": (int(_lerp(3, 8, t)), int(_lerp(7, 18, t))),
                "coverage_range": (_lerp(0.05, 0.12, t), _lerp(0.14, 0.35, t)),
                "highlight_boost_range": (_lerp(1.5, 3.0, t), _lerp(3.5, 7.0, t)),
                "veil_strength_range": (_lerp(0.18, 0.4, t), _lerp(0.4, 0.8, t)),
                "water_spot_count_range": (0, int(_lerp(5, 30, t))),
            },
            "SunFlare": {
                # Level 1 has to be *mild*, not *absent*. The previous floor
                # of 0.2 changed the image by about a third of an 8-bit level
                # on average, which is invisible — a severity ladder whose
                # bottom rung does nothing wastes a rung and reads as a bug.
                "intensity_range": (_lerp(0.55, 1.3, t), _lerp(0.95, 2.1, t)),
                "ghost_count_range": (int(_lerp(2, 6, t)), int(_lerp(5, 11, t))),
                "dispersion_ring_probability": _lerp(0.3, 0.7, t),
                # Severity here doubles as "how neglected is this camera".
                # A scuffed dome, a saturating older sensor and a failing
                # IR-cut filter all arrive together in practice, because they
                # are all symptoms of age rather than independent choices.
                "scratch_count_range": (int(_lerp(0, 3, t)), int(_lerp(3, 10, t))),
                "scratch_strength_range": (_lerp(0.03, 0.1, t), _lerp(0.15, 0.4, t)),
                "sensor_smear_probability": _lerp(0.1, 0.45, t),
                "ir_leak_probability": _lerp(0.15, 0.5, t),
                "window_ghost_probability": _lerp(0.2, 0.5, t),
            },
            "RandomFog": {
                # MOR = 3.912 / beta, so this spans roughly 800 m down to 30 m
                # of visibility.
                "beta_range": (_lerp(0.004, 0.07, t), _lerp(0.012, 0.13, t)),
                "scattering_gamma_range": (_lerp(0.8, 0.0, t), _lerp(2.0, 0.4, t)),
                "forward_scatter_range": (_lerp(0.0005, 0.003, t), _lerp(0.002, 0.008, t)),
            },
            # Rain and snow are driven by their *precipitation rate*, which
            # is the single physical quantity severity means here: it sets
            # the drop/flake size distribution, how many are in frame, and
            # (via the visibility relations) how dense the veil is. Nothing
            # else needs a ladder, because everything else follows from it.
            "RandomRain": {
                # Rate only. Exposure time is a property of the *camera*, not
                # of the weather, and tying it to severity made the ladder
                # non-monotonic: a longer exposure draws longer and therefore
                # proportionally fainter streaks, so past the middle of the
                # ladder the extra rain was cancelled by the extra blur.
                #
                # Note these stay *ranges*, narrowed around the level rather
                # than collapsed to a point. A preset picks how bad the
                # weather is; it must not turn the transform deterministic,
                # or every image at a given severity gets identical rain.
                # Floor stays at RandomRain's own validated minimum, so level
                # 1 is "light rain" rather than drizzle.
                "rain_rate_range_mm_h": (
                    _lerp(MIN_RAIN_RATE_MM_H, 55.0, t),
                    _lerp(8.0, 100.0, t),
                ),
                "wetness_probability": _lerp(0.15, 0.9, t),
            },
            "RandomSnow": {
                # Water-equivalent rate: a few tenths of a mm/h is a light
                # flurry, 10 mm/h (roughly 100 mm/h of actual snow depth) is a
                # blizzard whose veil alone drops visibility to tens of metres.
                "snowfall_rate_range_mm_h": (_lerp(0.2, 6.0, t), _lerp(1.2, 14.0, t)),
                # Heavier falls tend to be the big wet aggregates, which are
                # also the lowest-density and so the largest on screen.
                "snow_density_range_g_cm3": (_lerp(0.07, 0.03, t), _lerp(0.14, 0.07, t)),
                "aperture_diameter_range_mm": (_lerp(3.0, 6.0, t), _lerp(9.0, 20.0, t)),
                "accumulation_strength_range": (_lerp(0.1, 0.6, t), _lerp(0.35, 0.95, t)),
            },
            # --- Surveillance-stream effects (plan.md 6.2). Severity here is
            # "how degraded is this camera and its link", which in practice
            # co-varies: a neglected site has a scuffed dome, an old sensor,
            # a failing IR-cut filter *and* a starved uplink.
            "AcFlicker": {
                # A 2% supply modulation is genuinely invisible, so level 1
                # started below the 8-bit quantisation. Raised to a lightly
                # ballasted lamp rather than a perfect one. Note the sinc
                # cancellation still zeroes individual frames whose exposure
                # lands on a whole flicker period — that is the physics, and
                # it is why this is measured over many seeds.
                "modulation_depth_range": (_lerp(0.06, 0.2, t), _lerp(0.18, 0.6, t)),
            },
            "RollingShutter": {
                "pan_shift_range_px": (_lerp(-3.0, -30.0, t), _lerp(3.0, 30.0, t)),
                "wobble_amplitude_range_px": (0.0, _lerp(1.5, 12.0, t)),
            },
            "ExposureTransient": {
                "exposure_error_stops_range": (_lerp(-0.4, -2.2, t), _lerp(0.4, 2.2, t)),
                "actual_temperature_range_k": (_lerp(4800, 2400, t), _lerp(6200, 9500, t)),
                "tint_error_range": (_lerp(-0.02, -0.16, t), _lerp(0.02, 0.16, t)),
            },
            "WdrToneMapping": {
                "compression_range": (_lerp(0.8, 0.3, t), _lerp(0.95, 0.6, t)),
                "halo_strength_range": (_lerp(0.05, 0.5, t), _lerp(0.35, 1.0, t)),
                "detail_gain_range": (_lerp(0.95, 1.2, t), _lerp(1.2, 2.0, t)),
            },
            "TemporalDenoiseArtifacts": {
                # Same correction as DirtSoil: level 1 was moving the frame by
                # 0.71 of an 8-bit level, which is not a mild setting but no
                # setting at all.
                "strength_range": (_lerp(0.35, 0.7, t), _lerp(0.65, 1.0, t)),
                "iterations_range": (1, int(_lerp(2, 4, t))),
                "edge_threshold_range": (_lerp(18.0, 35.0, t), _lerp(40.0, 90.0, t)),
                "chroma_blur_range": (_lerp(0.005, 0.008, t), _lerp(0.014, 0.035, t)),
            },
            "NightRain": {
                # Rate drives the count, and the illuminator gain drives how
                # hard the lamp lights each drop. Both move with severity
                # because heavier rain in a stronger beam is the same axis:
                # a downpour under a bright illuminator is the case that
                # whites out an IR camera.
                "rain_rate_range_mm_h": (
                    _lerp(MIN_RAIN_RATE_MM_H, 55.0, t),
                    _lerp(12.0, 110.0, t),
                ),
                "illuminator_gain_range": (_lerp(1.2, 3.5, t), _lerp(3.0, 8.0, t)),
                # The wall of light from the drops the beam lights but cannot
                # resolve. Kept well below the streaks at low severity: it
                # destroys contrast rather than adding texture, so it should
                # arrive as the rain gets genuinely bad rather than tinting
                # every night frame.
                "backscatter_range": (_lerp(0.01, 0.10, t), _lerp(0.06, 0.30, t)),
            },
            "NightSnow": {
                # The same water-equivalent ladder as RandomSnow, so a night
                # frame and a day frame at one severity are the same storm.
                "snowfall_rate_range_mm_h": (_lerp(0.2, 6.0, t), _lerp(1.2, 14.0, t)),
                "snow_density_range_g_cm3": (_lerp(0.07, 0.03, t), _lerp(0.14, 0.07, t)),
                # The glow from the flakes too small to resolve. It arrives
                # with the storm: negligible in a flurry, most of the picture
                # in a blizzard, which is when an IR camera goes blind.
                "backscatter_range": (_lerp(0.01, 0.14, t), _lerp(0.06, 0.38, t)),
            },
            "NightIrMode": {
                # Severity is how little light there is to work with: a
                # shorter illuminator reach and more gain, hence more noise.
                "illuminator_range_m": (_lerp(10.0, 2.5, t), _lerp(22.0, 7.0, t)),
                "gain_range": (_lerp(0.8, 1.4, t), _lerp(1.3, 3.0, t)),
                "read_noise_range": (_lerp(0.002, 0.012, t), _lerp(0.012, 0.055, t)),
                "ambient_range": (_lerp(0.05, 0.002, t), _lerp(0.2, 0.03, t)),
            },
            "VideoCodecArtifacts": {
                "crf_range": (int(_lerp(20, 34, t)), int(_lerp(28, 46, t))),
                "stale_block_probability": _lerp(0.05, 0.5, t),
                "stale_block_fraction_range": (_lerp(0.01, 0.08, t), _lerp(0.08, 0.35, t)),
            },
            "HousingOccluders": {
                # At least one insect at every level. A count range starting
                # at zero made "no occluders at all" a legal draw 28% of the
                # time at level 1 and 3% at level 5 - so a caller who set
                # p=1.0, and therefore explicitly asked for the effect, got a
                # bit-identical frame back. Whether the effect applies is
                # what `p` is for; the count should decide how much of it
                # there is, not whether there is any.
                "insect_count_range": (int(_lerp(1, 3, t)), int(_lerp(2, 8, t))),
                # Webs stay optional: they are seasonal and a housing can
                # easily carry insects with no web at all.
                "web_count_range": (0, int(_lerp(1, 3, t))),
                "opacity_range": (_lerp(0.3, 0.6, t), _lerp(0.6, 1.0, t)),
            },
            "CastShadows": {
                # At least two. A figure's shadow covers a fifth of what a
                # crown's does, so a single draw can be one thin figure on
                # the verge, which moved the frame by half an 8-bit level —
                # the "severity 1 that does nothing" this ladder is tested
                # against.
                "shadow_count_range": (int(_lerp(2, 3, t)), int(_lerp(3, 9, t))),
                # Harder light means deeper shadows *and* crisper edges, which
                # is why these two move together rather than independently.
                "direct_fraction_range": (_lerp(0.25, 0.6, t), _lerp(0.55, 0.92, t)),
                "penumbra_range": (_lerp(0.004, 0.001, t), _lerp(0.03, 0.008, t)),
            },
        }
    return severities


SEVERITY: dict[int, dict[str, dict[str, Any]]] = _build()
