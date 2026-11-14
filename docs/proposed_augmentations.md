# Proposed augmentations

Effects the library did not have and should. All eight have since been
built and are kept here as the record of what was intended, each with a
note on what was done differently. Twelve more were built straight from a
shorter list without an entry here; they are named at the end.
Each entry says what the effect is, why it is missing, how it would be
modelled, where it sits in the imaging chain, and what would show that it
works, so that any one of them can be picked up cold.

They are in the order they should be built. The first two are cheap and
improve frames the shipped pipelines already produce. Three more ideas were
considered and rejected; they are at the end with the reasons, so they are
not proposed again.

| # | Working name | Chain stage | Needs depth | Moves pixels | Size |
| --- | --- | --- | --- | --- | --- |
| 1 | [`SensorNoise`](#1-sensornoise) — built | sensor | no | no | small |
| 2 | [Overcast lighting](#2-overcast-lighting-for-rain-and-snow) — built | atmosphere | no | no | small |
| 3 | [`SourceGlare`](#3-sourceglare) — built | optics | no | no | medium |
| 4 | [`NightFog`](#4-nightfog) — built | atmosphere | yes | no | medium |
| 5 | [`TransmissionFaults`](#5-transmissionfaults) — built | transmission | no | no | medium |
| 6 | [`OnScreenOverlay`](#6-onscreenoverlay) — built | after transmission | no | no | small |
| 7 | [`FocusShift` and `CameraVibration`](#7-focusshift-and-cameravibration) — built | optics | no | vibration does | medium |
| 8 | [`HeatShimmer`](#8-heatshimmer) — built | atmosphere | yes | yes | medium |

"Chain stage" refers to `scene -> atmosphere -> optics -> sensor -> ISP ->
transmission`, the order `presets/pipeline.py` sorts transforms into.

## What adding a transform involves

The same list for every entry below. `NightSnow` is the most recent transform
added this way and is the one to copy from.

1. The transform, subclassing `CameraTransform` (`base.py`). Parameters are
   physical quantities given as ranges and drawn per call. Anything that
   follows from them is derived, not exposed. Work in linear light
   (`utils/photometry.py`). Accept `uint8` and float input.
2. Export it from its subpackage's `__init__.py`.
3. A stage in `_STAGE_ORDER` (`presets/pipeline.py`).
4. A five-level entry in `SEVERITY` (`presets/severity.py`). Severity sets
   strength only: never `p`, and never a property of the camera.
5. A place in the pipelines (`presets/pipelines.py`), inside whichever
   lighting or weather branch makes it physically possible.
6. The gallery: append to `EFFECTS` in `docs/generate_examples.py`. Append,
   do not insert. Every effect draws from one generator in that order, so an
   insertion re-rolls the published image of everything after it.
7. `scripts/benchmark.py`, and the README's effect, parameter and timing
   tables.
8. Tests. Register the transform in `test_presets.py`,
   `test_severity_ladders.py`, `test_input_contract.py`, `test_readme.py`,
   and `test_randomization.py` if it draws per call; `test_gallery_configs.py`
   picks it up from `EFFECTS`. Then its own file, testing properties and not pixels: each test
   should name the wrong behaviour it rules out.
9. Regenerate deliberately: `python -m tests.visual_regression.generate_goldens`
   if it has a golden, `python docs/generate_examples.py`, and
   `python docs/generate_scene_examples.py`. Check that only the expected
   images changed.

A transform that moves pixels must also carry masks, boxes and keypoints, set
`preserves_scene_geometry = False`, and be tested for it. `RollingShutter`
(`camera/rolling_shutter.py`) is the model.

---

## 1. `SensorNoise`

> **Built**, as `camera/sensor_noise.py`. As specified, with two departures.
> `scene_illuminance_range_lux` was left out: gain is drawn directly, and
> the pipelines choose its range from the lighting state. And `NightIrMode`'s
> noise was not moved onto the shared helper, because doing so would re-roll
> its published images for no change in behaviour; `photon_transfer_noise` is
> there for it when that transform is next touched.

**What.** Shot noise, read noise and the chroma blotches a demosaiced
low-light frame carries, for cameras still in colour at night.

**Why it is missing.** Noise is modelled in exactly one place,
`camera/night_ir.py`. The street-lit night branch of `surveillance`
under-exposes the frame and shifts its colour and leaves it clean, and
`surveillance_lite` has no noise source at all. Real low-light colour footage
is dominated by noise; it is also what `TemporalDenoiseArtifacts` exists to
remove, so the pipelines currently denoise frames that have none.

**Model.** Undo the tone curve to linear, scale to photoelectrons with a full
well and a gain, then:

- shot noise, Poisson in electrons, so its variance equals the signal;
- read noise, Gaussian, independent of signal, added before gain;
- gain applied to both, which is why a gained-up dark frame is noisy;
- demosaicing: generate the noise on a Bayer mosaic and interpolate, which
  gives chroma noise a coarser grain than luma noise. This is the part a
  per-pixel Gaussian cannot reproduce and the one that makes it look like a
  camera.

`NightIrMode` already has the first three for a monochrome frame; lift them
into a shared helper and do not write them twice.

**Parameters.** `gain_range_db`, `full_well_range_e`, `read_noise_range_e`,
and `scene_illuminance_range_lux` to tie the gain to the light level the way
`RandomRain` meters its shutter.

**Stage.** Sensor, with `AcFlicker` and `RollingShutter`, before every ISP
stage. In the pipelines: the lit-night branch at high probability, the day
branch at low gain.

**Shows it works.** Noise variance rises linearly with signal in the shot
regime and is flat in the read regime. Chroma noise has a longer correlation
length than luma. Doubling gain at fixed scene light raises the noise floor.
A frame followed by `TemporalDenoiseArtifacts` loses texture it would have
kept without the noise.

**Open.** Whether to model fixed-pattern noise here (see the rejected list).

## 2. Overcast lighting for rain and snow

> **Built**, as `utils/overcast.py`, applied inside `RandomRain` and
> `RandomSnow` through `overcast_range`. The frame's level is not lowered:
> the transforms normalise it anyway, so the illuminance figures below were
> never needed and remain unchecked. The open question was answered by
> `utils/sky.py`, which finds sky as a featureless region at one depth that
> reaches the top of the frame (five of the six test scenes with open sky,
> none of the three without).

**What.** The light that comes with precipitation: a grey sky, lower
contrast and saturation, a dimmer frame.

**Why it is missing.** `RandomRain` and `RandomSnow` add drops, a veil and a
wet or white ground to whatever frame they are given. On a sunny photograph
the result is rain under a blue sky with hard shadows. With the distance
veil corrected to its true strength (see `rain_visibility.md`) daylight rain
is subtle, and the lighting is the strongest cue left that is not rendered.

**Model.** A term driven by the precipitation rate, since heavier rain means
deeper cloud:

- global illuminance falls with rate, from roughly a fifth of clear-sky
  light under thin overcast to a percent or two under a storm cloud (rough
  figures, to be checked against a source before they are coded); the
  auto-exposure stand-in the transforms already apply then hides most of the
  level change, so what remains visible is the next two;
- the direct-to-diffuse ratio falls to zero, which lowers contrast: pull
  luminance toward a blurred copy of itself, by an amount set by rate;
- the sky goes to cloud: desaturate and flatten pixels that the depth
  provider puts at the far plane and that are bright and blue;
- colour temperature rises a little (overcast light is bluer than sunlight)
  and saturation falls.

It cannot remove shadows that are already in the photograph. It softens a
sunny scene; it does not convert it. Say so in the docstring.

**Where.** Inside `RandomRain` and `RandomSnow` as a derived step, not as a
separate transform: it follows from the rate, and a separate knob would allow
a downpour in full sun, which is the thing being fixed. Give it an
`overcast_range` multiplier defaulting to 1 so that it can be turned off for
frames that are already overcast.

**Shows it works.** Sky saturation and global contrast fall monotonically
with rain rate. A frame that is already grey changes little. With the
multiplier at zero the output is what it is today.

**Open.** Sky detection without a segmentation model. Fog's `_sky_weight`
engaged on one of nine sample scenes; depth alone is not enough.

## 3. `SourceGlare`

> **Built**, as `lens/source_glare.py`, with sources found by
> `utils/sources.py`. `sun_flare.py` was not refactored: the transform holds
> a `SunFlare` and calls it once per source, which reuses every line of the
> optics without moving any. Strength comes from clipped area times
> `intensity_range`; there is no separate `over_range_range`. The open
> question stands: it costs one flare render per source.

**What.** Veiling glare, starbursts and ghosts from the lights actually in
the frame: headlights, street lamps, shop fronts.

**Why it is missing.** `SunFlare` has all the optics but takes one source at
a position the caller chooses (`flare_position`). At night the sources are in
the picture, there are several, and they are where the glare has to come
from. `WaterDroplets` and `DirtSoil` now scatter the scene's own light, but
only where there is water or dirt; a clean lens still flares.

**Model.** Find the sources: connected regions at or near clipping in linear
light, ranked by area times an assumed over-range factor, since a clipped
pixel is a light of unknown brightness (the film in `water_droplets.py`
assumes eight times full scale). For each of the brightest few, reuse
`SunFlare`'s components at a strength set by the source's estimated flux:
the glare veil, the diffraction starburst for the aperture's blade count,
and the ghost chain along the line through the image centre.

Refactor `sun_flare.py` so the per-source rendering takes a position and a
flux; `SunFlare` becomes the one-source case. Do not copy the code.

**Parameters.** `max_sources`, `over_range_range`, and the optical
parameters shared with `SunFlare`.

**Stage.** Optics, with `SunFlare`. In the pipelines: both night branches;
never together with `SunFlare`.

**Shows it works.** A frame with no near-clipped pixels is unchanged. Glare
energy scales with source area. Ghosts lie on the source-to-centre line, the
invariant `SunFlare` is already tested for.

**Open.** Cost: `SunFlare` takes tens of milliseconds at 512 px, and this
runs it per source.

## 4. `NightFog`

> **Built**, as `atmosphere/night_fog.py`. One transform, not the two-part
> split described under Stage: it sits after `NightIrMode` and attenuates the
> scene twice when the illuminator is on, which is the physics of a scene lit
> by the camera. The backscatter term is its own, not shared with
> `NightRain`; the two are three lines each. The open question stands.
>
> **Revised.** The model below — attenuation plus a glow, "neither of which
> is a uniform veil" — rendered as a dimmed frame with a blurred copy of
> itself on top, and read as blur. Fog at night does have a veil; it is
> uneven, as bright as the light around it. The transform now fades the
> scene towards a fog radiance estimated from the surfaces nearby, and
> keeps the glow for the lamps alone. The "mean radiance does not rise"
> check under *Shows it works* was wrong with it and is gone.

**What.** Fog at night: a glow round each lamp, and the wall of backscatter
from the camera's own illuminator.

**Why it is missing.** `RandomFog` is a daylight model. Its veil used to
take its colour from the brightest pixels in the frame, which on a real night
frame are the street lamps: measured, 17.8 times the frame's median. That is
fixed — the veil now takes the sky's radiance and stays dark at night — but
dark is all it is. Nothing lights the fog round a lamp.

**Model.** Two terms, neither of which is a uniform veil:

- **In-scene sources.** Each lamp lights the fog around it. Attenuate the
  scene by `exp(-beta z)` as by day, then add the scattered light as the
  source image convolved with a wide kernel whose width grows with optical
  depth, the single-scattering glow of a point source in fog. Dark parts of
  the frame far from any lamp stay dark.
- **The illuminator, under `NightIrMode`.** Backscatter along the line of
  sight, the integral of `beam(z) exp(-2 beta z) / z^2`, concentrated in the
  first metres. `NightRain` and `NightSnow` already carry this term for
  precipitation; share it.

**Parameters.** `beta_range` or `visibility_range_m`, the illuminator
parameters shared with the other night transforms.

**Stage.** The in-scene part is atmosphere and goes before the infrared
conversion; the backscatter is lit by the camera and goes after it, the same
split `RandomSnow` and `NightSnow` have. Needs depth.

**Shows it works.** Mean radiance does not rise except by the illuminator's
backscatter. A lamp's glow widens and its core dims as `beta` rises. A black
frame with the illuminator off stays black, the test `WaterDroplets` has.

**Open.** Lamps outside the frame light fog inside it and cannot be known.

## 5. `TransmissionFaults`

> **Built**, as `camera/transmission.py`, by the first route: a real H.264
> bitstream, packets removed, decoded by ffmpeg. Departures from the entry
> below. The single-image question is settled by sliding the image into
> place over a short clip (`scene_motion_range`), which gives concealment
> something wrong to copy; `images=` clips are still treated frame by
> frame, each as its own still, so a real previous frame is not used. The
> parameters gained `missing_reference_probability`,
> `concealment_probability` and `gap_passing_probability`, because whether
> a hole is smeared, left green or filled with garbage is decided by the
> receiver and not by the loss. In the pipelines `faults` defaults to 0.03
> and to 0 when `codec=0`. Not done: H.265, and moving objects in the
> reference.

**What.** What a lost packet does to a decoded stream: blocks that freeze,
smear or go grey-green, and frames torn part-way down.

**Why it is missing.** `VideoCodecArtifacts` re-encodes cleanly through
ffmpeg. That is compression. Loss is different and common on wireless and
congested links, and it is the fault operators actually report.

**Model.** Two honest routes:

- **Corrupt a real bitstream.** Encode a short clip, drop or damage slices or
  NAL units in the packetised stream, decode with error concealment on. This
  is exact and is what the codec transform's ffmpeg dependency already
  permits.
- **Emulate concealment.** On a macroblock grid, replace regions with the
  co-located blocks of a previous frame (freeze), or with motion-shifted
  copies (smear), from a damaged row to the next slice boundary.

Prefer the first. The second needs a previous frame, which a single image
does not have; with `images=` clips it does.

**Parameters.** `packet_loss_range`, `burst_length_range`, GOP length and
slice structure.

**Stage.** Transmission, after `VideoCodecArtifacts`. It needs ffmpeg, so
like the codec stage it must be left out of a pipeline entirely at
probability zero, not constructed and never fired.

**Shows it works.** Damage starts at a macroblock row and runs to a slice
boundary. It persists until the next keyframe in a clip. At zero loss the
output equals the codec transform's.

**Open.** A single frame has no temporal reference. Decide whether the
single-image case duplicates the frame into a clip or is simply unsupported.

## 6. `OnScreenOverlay`

**What.** The timestamp, camera name and privacy masks burned into most
surveillance footage.

**Why it is missing.** It is not weather or optics, so it was never in
scope. It is still in nearly every real frame, it covers corners where
objects enter, and a detector trained without it sees text as texture.

**Model.** Not physics, and the docs should say so. Render text in a
monospace bitmap face with an outline or a box, at the corners and positions
recorders use, with plausible content: a date and time, a channel name.
Privacy masks are filled rectangles or polygons, black or grey or pixelated.

**Parameters.** `timestamp_probability`, `label_probability`,
`privacy_mask_count_range`, position and size ranges.

**Stage.** Camera-side overlays are composited before encoding and pick up
compression artefacts; recorder-side overlays are added after decoding and
stay sharp. Support both with a flag. It keeps geometry, but privacy masks
hide objects, so decide what happens to boxes under a mask.

**Shows it works.** Overlays stay inside the frame at every resolution. Text
is legible at 256 px. A mask over a box either drops the box or marks it,
whichever is chosen, and is tested.

**Open.** Fonts. Shipping one adds a licence to track; rendering a small
built-in bitmap face avoids it.

**Built** as `camera/overlay.py`, with a 5 by 7 face of its own. A box
that privacy masks cover by `hidden_box_coverage` (0.9) of its area is
dropped; masks and keypoints are left alone. `drawn_by` chooses the stage,
which needed `stage_of` to read a stage from the instance: the transmission
stages moved up one to make room before the codec. Polygonal masks were
left out; they are rectangles.

## 7. `FocusShift` and `CameraVibration`

Two effects that are both blur and have nothing else in common.

**`FocusShift`.** A lens focused in visible light is slightly out of focus in
infrared unless it is IR-corrected, so the picture softens when the camera
switches modes at dusk. Focus also drifts with temperature and with a
loosened ring. Model it as a disc blur whose diameter follows from the
defocus and the aperture, scaled by depth where a provider is available,
since a focus error blurs near and far differently. The aperture-disc
convolution is in `water_droplets.py`; move it to `utils`.

Stage: optics. In the pipelines: under `NightIrMode` at a higher
probability.

**`CameraVibration`.** A pole in wind or a bracket next to traffic moves the
camera during the exposure. The result is a directional blur whose length
goes with shutter time, plus a small displacement of the whole frame. Model
it as a blur along a short, mostly angular trajectory, and a sub-pixel to
few-pixel shift. Tie the blur length to exposure the way `RandomRain` ties
streak length to it: no blur at 1/2000 s, a visible smear at 1/25 s.

Stage: sensor, with `RollingShutter`, and it interacts with it: a vibrating
rolling-shutter camera wobbles, which that transform already models. The
shift moves pixels, so annotations must be carried.

**Shows it works.** Focus blur grows with aperture and vanishes at the focus
distance. Vibration blur length is proportional to shutter time. Boxes and
keypoints move with the frame.

**Open.** Whether vibration belongs inside `RollingShutter`, which already
has a camera motion model.

**Built** as `lens/focus.py` and `camera/vibration.py`. `FocusShift` is
parametrised by where the lens is in fact focused, drawn evenly in
dioptres, and takes its depth provider optionally: without one the frame
gets the blur of a distant scene. The disc convolution is still
`water_droplets`' and was not moved. `CameraVibration` stayed separate
from `RollingShutter`: it treats the exposure as one instant for every
row, and says so. Its smear is the path's own distribution, not a box,
and stops growing once the exposure holds the whole swing.

## 8. `HeatShimmer`

**What.** Turbulence over hot ground: distant detail ripples and softens.

**Why it is missing.** It only matters over long paths, and the library's
default world is 3 to 50 m. The sample scenes now include views of 900 m to
6 km (`urban_road_day`, `mountain`, `wooded_hills`), where it is the main
thing degrading a long lens on a warm day.

**Model.** A random displacement field plus a blur, both growing with path
length: the tilt variance of a turbulent path goes roughly linearly with
distance for a given turbulence strength. Generate a smooth vector field of
correlated noise, scale it per pixel by depth, and warp; add a depth-scaled
blur for the scales smaller than a pixel. Strongest near the ground, where
the temperature gradient is.

**Parameters.** `turbulence_strength_range` (physically `Cn^2`), the path
height, and the field's correlation length.

**Stage.** Atmosphere. Needs depth, and it needs it calibrated: against the
default 3 to 50 m range it should do almost nothing, which is the correct
answer and a useful test. It moves pixels by fractions of a pixel to a few
pixels; decide whether annotations follow, and if they do not, bound the
displacement and say so.

**Shows it works.** Displacement grows with depth and is near zero in the
foreground. The mean displacement over the frame is zero: shimmer does not
shift the picture. On a clip the field changes between frames while its
statistics do not.

**Open.** A single frame shows shimmer only as distortion. Its signature is
motion, which needs the per-frame variation `images=` clips do not yet have.

**Built** as `atmosphere/heat_shimmer.py`, with the tilt and the blur from
the theory of seeing (`2.91 D^(-1/3) Cn2 L` and `lambda / r0`) in place of
a strength knob, so the test that it does nothing over 50 m is a
consequence and not a setting. A constant `Cn2` along the path: the
stronger turbulence near the ground was left out. Annotations do not
follow; the displacement is a pixel or two.

---

## Built without an entry

`SpiderWeb`, `NightInsects`, `IrDomeGlare`, `FrostOnGlass`,
`SnowOnHousing`, `LensTamper`, `FoliageIntrusion`, `IrCutFilterFault`,
`DemosaicArtifacts`, `AnalogVideoArtifacts`, `GroundFog` and
`LocalPlumes`. Each module's docstring is its entry: what the effect is
and the model, with the tests naming what would show it wrong.
[`how_it_works.md`](how_it_works.md) has a table of the one fact each is
built on. None of them, and none of the eight above from entry 6 on, has
been given a place in the ready-made pipelines: that needs a probability
for each, which is a decision about a deployment and not about physics.

## Considered and rejected

**Lens distortion.** Albumentations has it (`OpticalDistortion`,
`GridDistortion`), with annotation support. Nothing to add.

**Dust storms and smoke.** Rare in surveillance footage, and optically fog
with a tint: `RandomFog` with a coloured airlight covers most of it. Revisit
if a deployment in a dusty climate asks.

**Dead pixels and sensor dust.** Real, but fixed per camera: the same pixels
in every frame from that camera. A random augmentation that moves them
between samples teaches the opposite. They belong in a per-camera mask the
caller applies, and that is a dataset tool, not a transform. `SensorNoise`
should say whether it leaves room for one.

## Related work that is not a new augmentation

Defects in existing effects, found on the sample scenes, are listed in
[`examples/scenes/README.md`](examples/scenes/README.md). Two of them bear on
the entries above: sky detection (entry 2) and the daylight fog model at
night (entry 4).
