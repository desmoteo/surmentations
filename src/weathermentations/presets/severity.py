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

The ground is a second ladder
-----------------------------
How wet a road is, and how much snow lies on it, are set by the weather's
*history* and not by its rate at the moment of the frame. So they have their
own ladder, :data:`GROUND_SEVERITY`, with a level 0 the main one lacks — a
dry road, bare ground — and are combined freely with it::

    # A blizzard that has only just started.
    RandomSnow(depth_provider=provider,
               **SEVERITY[5]["RandomSnow"], **GROUND_SEVERITY[0]["RandomSnow"])
    # A few flakes over a town already deep in snow.
    RandomSnow(depth_provider=provider,
               **SEVERITY[1]["RandomSnow"], **GROUND_SEVERITY[5]["RandomSnow"])

A ground level states the ground's condition, so it sets the probability of
drawing it to one (or to zero at level 0). Leave the ground entry out and the
transform draws the condition itself, over its whole range.
"""

from __future__ import annotations

from typing import Any

from weathermentations.atmosphere.rain import MIN_RAIN_RATE_MM_H


def _lerp(a: float, b: float, t: float) -> float:
    """``a`` at ``t = 0``, ``b`` at ``t = 1``, in a straight line between."""
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
                # Calibrated for a far-focused camera, where the aperture
                # dissolves most small drops and a film only redistributes
                # the scene's own light. Both make water on the glass far
                # gentler per drop than a renderer that draws crisp beads
                # and a white veil, so the ladder needs more of it: at the
                # counts that suited the old renderer, level 1 moved a frame
                # by 1.0 level behind an open iris and level 5 by 15.
                "droplet_count_range": (int(_lerp(20, 160, t)), int(_lerp(50, 380, t))),
                "droplet_radius_range": (_lerp(0.002, 0.004, t), _lerp(0.014, 0.04, t)),
                "condensation_coverage_range": (_lerp(0.08, 0.5, t), _lerp(0.25, 0.95, t)),
                "condensation_edge_softness_range": (_lerp(0.45, 0.25, t), _lerp(0.7, 0.5, t)),
                "condensation_milkiness_range": (_lerp(0.2, 0.6, t), _lerp(0.4, 0.95, t)),
                "trail_probability": _lerp(0.1, 0.6, t),
            },
            "DirtSoil": {
                # Level 1 must be mild, not absent: at the previous floor it
                # moved the frame by 0.94 of an 8-bit level, i.e. below the
                # quantisation it is rendered into.
                "smudge_count_range": (int(_lerp(3, 8, t)), int(_lerp(7, 18, t))),
                "coverage_range": (_lerp(0.05, 0.2, t), _lerp(0.14, 0.5, t)),
                # The even film of settled dust between the patches. Absent
                # at the clean end, and at the dirty end what takes the
                # contrast out of the whole frame.
                "dust_film_range": (_lerp(0.0, 0.12, t), _lerp(0.06, 0.4, t)),
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
                #
                # The state of the ground is not on this ladder. It was:
                # the road went from usually dry at level 1 to soaked at
                # level 5, which made a downpour on a dry road and a wet
                # road under a few drops impossible to ask for, and both are
                # ordinary — the first seconds of a storm, and the hour
                # after one. See `GROUND_SEVERITY`.
                "rain_rate_range_mm_h": (
                    _lerp(MIN_RAIN_RATE_MM_H, 80.0, t),
                    _lerp(12.0, 150.0, t),
                ),
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
                # Settled snow is not on this ladder either: how much lies on
                # the ground is how long it has been snowing and how cold the
                # ground was, not how hard it is snowing now.
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
            "SensorNoise": {
                # Gain is the whole of it: 4-10 dB is a daylight frame whose
                # grain shows only in flat areas, 24-36 dB a night frame that
                # is mostly grain. Full well and read noise are the sensor's
                # and stay at the transform's own ranges.
                # The floor is 4 dB, not 0: at unity gain the grain moved the
                # reference frame by 0.98 of a level, under what it is
                # quantised into.
                "gain_range_db": (_lerp(4.0, 24.0, t), _lerp(10.0, 36.0, t)),
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
            "NightFog": {
                # The same span as RandomFog's beta, as visibility: mist at
                # level 1, 30-55 m at level 5.
                # Interpolated in the logarithm: visibility is a ratio, and a
                # straight line from 330 m to 30 m spends three levels in
                # fog too thin to see over a 50 m scene.
                "visibility_range_m": (330.0 * (30.0 / 330.0) ** t, 900.0 * (55.0 / 900.0) ** t),
            },
            "SourceGlare": {
                "intensity_range": (_lerp(0.05, 0.25, t), _lerp(0.12, 0.6, t)),
                "ghost_count_range": (int(_lerp(1, 4, t)), int(_lerp(3, 8, t))),
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
            "TransmissionFaults": {
                # How bad the link is. Loss carries it; with it goes how far
                # the scene has moved since the picture the decoder falls
                # back on, and how often it has no such picture at all —
                # both grow with the length of the outage.
                "packet_loss_range": (_lerp(0.02, 0.2, t), _lerp(0.06, 0.5, t)),
                "scene_motion_range": (_lerp(0.004, 0.03, t), _lerp(0.015, 0.08, t)),
                "missing_reference_probability": _lerp(0.05, 0.5, t),
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
            "SpiderWeb": {
                # Severity is how near the web is. Nearer, it takes more of
                # the frame, its discs are wider and the lamp lights it
                # harder, all at once; the gain climbs with it so that the
                # bottom rung is a faint lace and the top one a white-out.
                "distance_range_m": (_lerp(0.12, 0.04, t), _lerp(0.22, 0.09, t)),
                "illuminator_gain_range": (_lerp(0.2, 0.5, t), _lerp(0.45, 1.5, t)),
            },
            "NightInsects": {
                # More of them, and a brighter lamp to show them by. Where
                # they fly and how fast is theirs and does not change.
                "insect_count_range": (int(_lerp(3, 10, t)), int(_lerp(6, 40, t))),
                "dust_count_range": (int(_lerp(4, 15, t)), int(_lerp(12, 80, t))),
                "illuminator_gain_range": (_lerp(1.0, 1.5, t), _lerp(2.0, 4.0, t)),
            },
            "IrDomeGlare": {
                # A brighter reflection over a dirtier dome: more light in
                # the ring and more of it spread as a veil.
                "intensity_range": (_lerp(0.03, 0.3, t), _lerp(0.1, 0.9, t)),
                "veil_range": (_lerp(0.0, 0.02, t), _lerp(0.005, 0.08, t)),
            },
            "FrostOnGlass": {
                # Frost that has had longer to grow: further in from the rim
                # and thicker where it is.
                "coverage_range": (_lerp(0.15, 0.7, t), _lerp(0.35, 1.1, t)),
                "thickness_range": (_lerp(0.3, 1.5, t), _lerp(0.8, 3.5, t)),
            },
            "SnowOnHousing": {
                # A deeper cap, and more stuck to the glass.
                "cap_depth_range": (_lerp(0.04, 0.3, t), _lerp(0.12, 0.6, t)),
                "patch_count_range": (0, int(_lerp(1, 6, t))),
            },
            "LensTamper": {
                # More of the lens covered, more thoroughly.
                "coverage_range": (_lerp(0.05, 0.5, t), _lerp(0.2, 1.0, t)),
                "opacity_range": (_lerp(0.6, 0.9, t), _lerp(0.9, 1.0, t)),
            },
            "FoliageIntrusion": {
                # More branches, reaching further in.
                "branch_count_range": (1, int(_lerp(1, 5, t))),
                "reach_range": (_lerp(0.15, 0.5, t), _lerp(0.35, 1.0, t)),
            },
            "IrCutFilterFault": {
                # More infrared reaching more of the sensor, and a camera
                # that takes less of the cast back out.
                "leak_range": (_lerp(0.15, 0.9, t), _lerp(0.35, 1.6, t)),
                "white_balance_adaptation_range": (_lerp(0.8, 0.2, t), _lerp(1.0, 0.6, t)),
            },
            "DemosaicArtifacts": {
                # A sharper lens on a coarser sensor, and a processor that
                # hides less of the result.
                "optical_blur_range_px": (_lerp(0.5, 0.0, t), _lerp(0.8, 0.2, t)),
                "sensor_scale_range": (_lerp(1.0, 1.6, t), _lerp(1.2, 2.6, t)),
                "chroma_filter_range_px": (0, 1 if t < 0.5 else 0),
            },
            "AnalogVideoArtifacts": {
                # A longer cable into a cheaper receiver: less bandwidth,
                # more leakage, more noise, a stronger reflection.
                "luma_bandwidth_range": (_lerp(0.75, 0.2, t), _lerp(0.95, 0.4, t)),
                "crosstalk_range": (_lerp(0.1, 0.7, t), _lerp(0.4, 1.0, t)),
                "noise_range": (_lerp(0.004, 0.03, t), _lerp(0.012, 0.07, t)),
                "ghost_strength_range": (_lerp(0.0, 0.1, t), _lerp(0.06, 0.35, t)),
                "hum_depth_range": (_lerp(0.0, 0.03, t), _lerp(0.02, 0.12, t)),
            },
            "FocusShift": {
                # Focused nearer and nearer: the blur of everything far is
                # the aperture over that distance.
                "focus_distance_range_m": (_lerp(0.2, 0.03, t), _lerp(0.35, 0.06, t)),
            },
            "CameraVibration": {
                # A wider swing behind a longer shutter.
                "amplitude_range": (_lerp(0.0006, 0.004, t), _lerp(0.0015, 0.012, t)),
                "exposure_time_range": (_lerp(0.004, 0.02, t), _lerp(0.01, 0.04, t)),
            },
            "OnScreenOverlay": {
                # Larger lettering and more of the frame blanked. Whether
                # there is a timestamp at all stays the caller's.
                "text_height_range": (_lerp(0.02, 0.05, t), _lerp(0.03, 0.08, t)),
                "privacy_mask_count_range": (int(_lerp(0, 2, t)), int(_lerp(1, 5, t))),
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


def _build_ground() -> dict[int, dict[str, dict[str, Any]]]:
    """The state of the ground, 0 (dry, bare) to 5 (soaked, deep in snow)."""
    ground: dict[int, dict[str, dict[str, Any]]] = {
        0: {
            "RandomRain": {"wetness_probability": 0.0},
            "RandomSnow": {"accumulation_probability": 0.0},
        }
    }
    for level in range(1, 6):
        t = (level - 1) / 4.0
        ground[level] = {
            "RandomRain": {
                "wetness_probability": 1.0,
                # Wet ground's albedo relative to dry: 0.9 is damp, 0.5
                # soaked. The mirror follows from it, so a damp road is
                # darker and nothing else, and a soaked one reflects.
                "wetness_darkening_range": (_lerp(0.82, 0.4, t), _lerp(0.9, 0.55, t)),
                # Standing water, which needs the film: none on a damp road
                # whatever this says, and up to half of a soaked one.
                "puddle_coverage_range": (_lerp(0.0, 0.3, t), _lerp(0.1, 0.55, t)),
            },
            "RandomSnow": {
                "accumulation_probability": 1.0,
                # A dusting that leaves the road showing through, up to a
                # coat that hides everything level.
                "accumulation_strength_range": (_lerp(0.1, 0.8, t), _lerp(0.3, 1.0, t)),
            },
        }
    return ground


#: The ground's condition under rain or snow, independent of :data:`SEVERITY`.
GROUND_SEVERITY: dict[int, dict[str, dict[str, Any]]] = _build_ground()
