# surmentations

![surmentations: physically grounded weather, lens and camera augmentations](https://desmoteo.github.io/surmentations/assets/social_banner.webp)

Albumentations-compatible, physically-grounded augmentations for training
models that run on real camera feeds: surveillance cameras first, and any
camera that is bolted to something and left outdoors. The name is
"surveillance" and "augmentations" run together. Thirty-seven transforms,
in four groups:

- **weather and scene**: rain, snow and fog by day and under the camera's
  own lamp, fog lying on the ground, steam, heat shimmer, cast shadows,
  insects in the beam;
- **lens and housing**: water, dirt, flare and glare, frost, snow, a
  spider's web, a branch, a lens out of focus or tampered with;
- **sensor and processor**: night infrared and its filter, noise, flicker,
  rolling shutter, vibration, the mosaic, exposure, tone mapping,
  denoising;
- **the link**: a real codec, packet loss, an analog cable, and what the
  recorder writes over the picture.

The last three degrade a CCTV stream far more often than weather does.

Effects are parameterised by **physical quantities** — a rain rate in mm/h, a
drop diameter in millimetres, a shutter time in seconds — not opaque 0-1
"strength" dials, and everything that follows from those is derived rather
than exposed.

- **[`docs/how_it_works.md`](docs/how_it_works.md)** — what each effect
  models and why, with diagrams. Start here to decide which knobs matter.
- **[`docs/transforms/`](docs/transforms/README.md)** — one page per
  transform: the algorithm as a diagram, the model, every parameter with
  its default, sample output at five severities, and the papers behind it.
- **[`docs/rain_visibility.md`](docs/rain_visibility.md)** — why rain streaks
  are faint at surveillance resolutions, measured; and the drop-size
  distribution options.
- **[`docs/pipelines.md`](docs/pipelines.md)** — a worked composite whose
  effects cannot contradict each other, with measured frequencies and cost.
  Start here if you want a pipeline rather than a transform.
- **[`docs/road_surface.md`](docs/road_surface.md)** — what is laid out
  along a road: its lanes, read off the paint and the vehicles; wheel tracks
  in settled snow; water in the ruts and gutters; spray behind vehicles, by
  their heading and their speed; their lamps, lit.
- **[`docs/scene_understanding.md`](docs/scene_understanding.md)** — the
  things a provider can work out besides depth: what each pixel is, how
  the camera is mounted, and which end of each vehicle is seen. What they change in each effect, how the geometry
  is estimated and how far to trust it.
- **[`docs/physical_review.md`](docs/physical_review.md)** — a critical audit
  of whether the implementation lives up to that, including where it did not.
- **[`optimizations.md`](optimizations.md)** — where the CPU time actually
  goes, and an assessment of what is worth optimising.
- **[`docs/native_kernels.md`](docs/native_kernels.md)** — the optional
  compiled kernels: what was moved to C and why, what it bought, and why
  there is no hand-written SIMD.
- **[`docs/proposed_augmentations.md`](docs/proposed_augmentations.md)** —
  the effects that were proposed and have since been built, each with its
  model, what was done differently, and the checklist for adding another.
- **[`docs/site/`](docs/site/)** — the code reference, generated from the
  source: every module, class, function and C declaration. See
  [Development](#development) for building it.
- **[`docs/examples/scenes/`](docs/examples/scenes/README.md)** — every
  effect over eight photographs of different scenes, one page each, with
  the effects that read the scene shown side by side without and with scene
  labels, geometry and the vehicle reader.
- **[`plan.md`](plan.md)** — design record and roadmap.

---

## Install

```bash
pip install -e ".[dev]"   # editable install with lint/type/test tooling
```

Optional runtime dependencies:

| Need | Required for |
| --- | --- |
| CUDA GPU | the depth-conditioned effects (fog, rain, snow, night IR, cast shadows), and the optional scene labels |
| `ffmpeg` on PATH, or PyAV | `VideoCodecArtifacts` and `TransmissionFaults` |

Everything else runs on CPU with no extra setup.

`pip install "surmentations[codec]"` adds [PyAV](https://pyav.org), and
the two codec transforms then encode inside the process: an H.264 frame in
about 10 ms at 512 px where starting two ffmpeg processes takes 150 or
more. For H.264 the frame is the binary's to the byte, which is checked
once on each installation and not assumed; H.265 comes from another release
of the encoder and is used in fast mode only. PyAV's wheels carry their own
FFmpeg built with GPL components (libx264, libx265) and are about 130 MB
installed. `SURMENTATIONS_CODEC=ffmpeg` keeps to the binary.

If a C compiler (`cc`, `gcc` or `clang`) is present, installing also builds a
small native library that draws rain and snow and converts colour several
times faster. It is optional and changes no picture: without it the same
transforms run their numpy code. `python -c "from surmentations import
_native; print(_native.tier())"` prints the instruction set in use, or
`None` if it was not built; `python hatch_build.py` builds it in a checkout.
`SURMENTATIONS_BUILD_MARCH=native` adds a tier compiled with
`-march=native -mtune=native`, for the machine doing the build only.
See [`docs/native_kernels.md`](docs/native_kernels.md).

**Fast and exact mode.** By default a few operations use a faster form that
differs slightly from the reference arithmetic. The two that show: wide
blurs, the sky's edge in fog, rain and snow among them, are made on a
reduced copy of the frame, which is 5 to 7 times faster at 1024 px and
moves about 1% of pixels along the skyline; and the depth network runs in
half precision, which halves the depth estimate and moves depth-dependent
edges by a pixel here and there (`DepthAnythingV2Estimator(half_precision=False)`
keeps full precision).
`surmentations.set_fast(False)`, `with surmentations.exact():` or
`SURMENTATIONS_FAST=0` selects exact mode. The bounds are in
[`docs/native_kernels.md`](docs/native_kernels.md#fast-and-exact-mode).

**In a `DataLoader` worker**, `surmentations.keep_freed_memory()` tells
glibc's allocator to reuse freed frames instead of returning them to the
system. It made `SensorNoise` twice as fast at 512 px and several other
transforms 6 to 10% faster, at the cost of the worker not shrinking after a
peak. It is a process-wide setting, so it is yours to call.

---

## Quickstart

### Camera-local effects — no GPU

```python
import albumentations as A
from surmentations.lens import WaterDroplets, DirtSoil, SunFlare

transform = A.Compose([
    WaterDroplets(p=0.3),
    DirtSoil(p=0.2),
    SunFlare(p=0.1),
])

augmented = transform(image=image)["image"]
```

### Depth-conditioned effects — needs CUDA

Construct **one** `SharedDepthProvider` and pass that same instance to every
depth-conditioned transform, so depth is estimated once per image instead of
once per transform: a provider per transform pays the estimate again in
each (see [Performance](#performance) for what it costs). `provider.cache_info()`
reports hits and misses if you want to confirm it is happening.

```python
import albumentations as A
from surmentations.atmosphere import RandomFog, RandomRain, RandomSnow
from surmentations.depth import DepthAnythingV2Estimator, SharedDepthProvider

depth_provider = SharedDepthProvider(DepthAnythingV2Estimator())

transform = A.Compose([
    RandomFog(depth_provider=depth_provider, p=0.2),
    RandomRain(depth_provider=depth_provider, p=0.2),
    RandomSnow(depth_provider=depth_provider, p=0.1),
])
```

With no CUDA device, constructing `DepthAnythingV2Estimator` raises a
`RuntimeError` immediately with an actionable message, rather than silently
falling back to a CPU path slow enough to blow a training-step budget.

### What a provider can read — optional

Depth says how far away a pixel is. It does not say what the pixel is,
where the camera stands, or which way a car is facing. The same provider
can read each of those, once per scene, and every transform given that
provider uses whatever it has. Each reader is optional and independent of
the others, except that the vehicle reader needs the labels to say where
the vehicles are; a provider given none behaves as it always did.

| Reader | Argument | What it says | Shipped estimator | Cost per new scene |
| --- | --- | --- | --- | --- |
| Depth | first argument | how far each pixel is | `DepthAnythingV2Estimator` (Apache-2.0) | 11–15 ms |
| Scene labels | `labels=` | what each pixel is: road, sky, vegetation, vehicle, person … | `SegmentationLabelEstimator` (UperNet/ConvNeXt, MIT) | 33 ms |
| Installation geometry | `geometry=` | the camera's field of view, tilt, roll and height | `GroundPlaneGeometryEstimator`, or an `InstallationGeometry` you give | 5 ms |
| Vehicle views | `vehicles=` | each vehicle apart from its neighbours, which end of it is seen, where its lamps and its number plate are; and where the street lamps and the road signs stand | `OpenVocabularyVehicleEstimator` (OWLv2, Apache-2.0) | 130–200 ms |

```python
from surmentations.depth import DepthAnythingV2Estimator, SharedDepthProvider
from surmentations.perception import (
    GroundPlaneGeometryEstimator,
    InstallationGeometry,
    OpenVocabularyVehicleEstimator,
    SegmentationLabelEstimator,
)

depth_provider = SharedDepthProvider(
    DepthAnythingV2Estimator(),
    labels=SegmentationLabelEstimator(),          # road, sky, vegetation, vehicles ...
    geometry=GroundPlaneGeometryEstimator(),      # horizon, tilt and height, from the frame
    vehicles=OpenVocabularyVehicleEstimator(),    # which end of each vehicle, and its lamps
)

# A fixed camera whose mounting is known is better given than estimated:
geometry = InstallationGeometry(horizontal_fov_deg=92.0, pitch_deg=24.0, height_m=6.5)

# Where the pipeline works on a reduced frame, have the scene read from the
# photograph it was reduced from: a lamp or a plate is a few pixels otherwise.
depth_provider.remember(image, original=photograph)

# Or, in a training loader, run the pipeline at the size the network sees,
# with every reading taken from the one kept on disk for the photograph:
# nothing is estimated on the reduced frame, and no network runs.
depth_provider.resized(image, original=photograph)

# What the provider read off a frame:
labels = depth_provider.get_labels(image)        # (H, W) uint8 of SceneClass
camera = depth_provider.settle_camera(image)     # field of view, pitch, roll, height
views = depth_provider.get_vehicles(image)       # one VehicleView per vehicle: end, lamps, plate
lamps = depth_provider.get_street_lamps(image)   # one row per street lamp's head
signs = depth_provider.get_signs(image)          # one row per road sign
```

**Read once, across epochs.** A provider shares a reading between the
transforms of one pipeline call, but a training set comes back every epoch
and is read again every epoch: with all four readers that is about 170 ms
a frame on a GPU, more than the transforms then cost. `disk_cache` keeps
each reading in a file and finds it again, in 8 ms:

```python
depth_provider = SharedDepthProvider(
    DepthAnythingV2Estimator(),
    labels=SegmentationLabelEstimator(),
    vehicles=OpenVocabularyVehicleEstimator(),
    disk_cache="readings/",        # or True: ~/.cache/surmentations ($SURMENTATIONS_CACHE)
    disk_cache_max_gb=50,          # optional: past it, nothing more is written
)
depth_provider.disk_cache_info()   # hits, misses, written, refused
```

A reading is filed under a hash of the frame's own pixels, the estimators
that made it and their settings, fast or exact mode, and the library's
version, so a file is never handed back for a frame, a checkpoint or a
mode it was not made for; one that cannot be read is made again. About
half a megabyte a frame at 512 px. It is off unless asked for, the
workers of a `DataLoader` share the directory, and an estimator of your
own is kept once it has a `signature()` that says what it is.

What is read is read from the frame a pipeline starts with, whichever
transform asks first, and never from a frame with weather already drawn
over it.

**Faster networks: TensorRT.** Each of the three estimators takes
`backend="tensorrt"` (`pip install "surmentations[tensorrt]"`; TensorRT is
NVIDIA's, under its own licence). Its network, or for the vehicle reader
the part that sees the picture, is exported to ONNX under autocast and
built into a TensorRT engine the first time it runs, about a minute for
all three, and loaded from `~/.cache/surmentations/engines`
(`$SURMENTATIONS_ENGINE_CACHE`) after that. A reading made this way has a
signature of its own, so it is kept apart from one made with torch.

```python
depth_provider = SharedDepthProvider(
    DepthAnythingV2Estimator(backend="tensorrt"),
    labels=SegmentationLabelEstimator(backend="tensorrt"),
    vehicles=OpenVocabularyVehicleEstimator(backend="tensorrt"),
)
```

On 100 val2017 frames, fast mode, a laptop RTX PRO 1000 (Blackwell),
TensorRT 11.5, one frame at a time:

| Estimator | torch | TensorRT | | What changes |
| --- | ---: | ---: | ---: | --- |
| Depth | 15.1 ms | 7.4 ms | 2.0× | correlation with torch's map 0.99999 (least 0.99984) |
| Scene labels | 39.3 ms | 21.6 ms | 1.8× | 99.95% of pixels the same class (least 99.72%) |
| Vehicle reader | 39.1 ms | 28.1 ms | 1.4× | the same vehicles, street lamps and signs; 36 lamps for 35, moved 0.09 px at the median |
| A frame's whole reading | 139.5 ms | 90.3 ms | 1.55× | |

The vehicle reader's network alone goes from 62 to 44 ms a pass: torch
already runs its attention in fused kernels, and the matrix products are
at the card's half-precision rate either way. The engine for depth takes
sides of 126 to 1036 pixels; a frame sized outside them, wider than about
two to one, is read by torch.

**Labels and geometry.** With labels, a wet road stays off the vehicles on
it and snow settles by name as well as by shape; the sky is the sky; foliage
is bright in infrared night whatever colour it photographed as. With a
geometry, every transform sees the frame through one camera in place of a
field of view and a pitch drawn by each; a wet road mirrors about the true
horizon; and where the camera's height is known, depth is in metres measured
from it and `depth_range_m` is not needed. Each scene below with its labels
and the horizon the estimator found:

![what the provider reads off each sample scene](https://desmoteo.github.io/surmentations/assets/scene_understanding/labels_and_horizon.webp)

**Vehicle views.** A semantic mask runs touching vehicles together and
says nothing of which way they face. The vehicle reader tells them apart,
says which end of each is seen (green outlines are fronts, red are rears)
and finds its lamps (yellow):

![vehicles as the reader sees them](https://desmoteo.github.io/surmentations/assets/road_surface/vehicle_views.webp)

That is what `RandomRain` needs to lay spray behind each vehicle and light
its lamps (`spray_range`, `vehicle_lamp_range`, `traffic_speed_range_kmh`),
left as the road was and right with both:

![a soaked road without and with its traffic's spray and lamps](https://desmoteo.github.io/surmentations/assets/road_surface/wet_road_and_spray.webp)

and what `NightIrMode` and `NightColorMode` need to light a scene as a road
is lit at night: from its street lamps, each throwing a pool that falls off
with distance, and its traffic's lamps and beams, in place of one even
level of light. The same draw with the lamps turned off, which is a
daylight frame in grey, and with them:

![infrared night lit evenly, and lit by its lamps](https://desmoteo.github.io/surmentations/assets/road_surface/night_lamps.webp)

In colour, the same scene lit by the sky's glow alone, which is all a
provider with depth alone can do, and by sodium and by LED street lighting:

![a night in colour: unlit, sodium, LED](https://desmoteo.github.io/surmentations/assets/road_surface/night_colour.webp)

A night lit this way has lamps whose places are known, and what is in the
air shows in their light. `NightFog`, `NightRain` and `NightSnow` without
it (left) and with it (right):

![fog, rain and snow in the lamps' light](https://desmoteo.github.io/surmentations/assets/road_surface/lamp_cones.webp)

Under the camera's own infrared illuminator, the road signs and number
plates the reader finds send its light straight back, as they do on a real
camera. The reader's box says only where each is: its shape (a slanted
plate, a sign on its point) is taken from the picture, and its edge is
the haze a lens makes of something that bright. Without and with:

![signs and plates under infrared](https://desmoteo.github.io/surmentations/assets/road_surface/retroreflectors.webp)

All of these are evidence and not a verdict: an effect weighs what a reader
says against what it reads from depth and colour itself.
[`docs/scene_understanding.md`](docs/scene_understanding.md) has what each
estimator can and cannot tell, its cost and the licences of the networks;
[`docs/road_surface.md`](docs/road_surface.md) has the lanes, wheel tracks,
standing water, spray and lamps that are drawn along a road.

---

## The effects

Each transform has a page of its own under
[`docs/transforms/`](docs/transforms/README.md), with its algorithm, its
parameters, samples and references. The tables here are the overview.

### Weather and scene

| Transform | Models | Needs depth |
| --- | --- | --- |
| [`RandomFog`](docs/transforms/RandomFog.md) | Koschmieder extinction with per-channel scattering, the sky found and put at the far plane, and a veil the colour of the horizon sky, not of the brightest pixels | yes |
| [`RandomRain`](docs/transforms/RandomRain.md) | Marshall-Palmer (or Ulbrich gamma) drop sizes, Gunn-Kinzer fall speeds, Garg-Nayar streak photometry, a distance veil, and wet ground: dark surfaces darkened far more than pale ones, each object's reflection smeared down the road from its foot, Fresnel-weighted, with puddles that mirror the sky cleanly, and the vehicles on the road left as they were. At night, streaks lit by the lamps in the frame and a road that carries each lamp's long reflection | yes |
| [`RandomSnow`](docs/transforms/RandomSnow.md) | Gunn-Marshall sizes corrected for snow density, Locatelli-Hobbs fall speeds, and settled snow: level ground covered up to the foot of each object, clumps on tree foliage, lit by the light on the spot, so white by day and orange under sodium lamps | yes |
| [`NightRain`](docs/transforms/NightRain.md) | Rain lit by the camera's **own infrared illuminator**, brightness falling as 1/z² from the lamp. The one case where rain reads as streaks rather than haze — see [`docs/rain_visibility.md`](docs/rain_visibility.md). Needs no depth, so it runs on CPU | no |
| [`NightSnow`](docs/transforms/NightSnow.md) | Snow lit by the camera's own illuminator: each flake's brightness is set by *its* distance from the lamp, not by the scene behind it. `RandomSnow` before `NightIrMode` loses its flakes over anything distant; this is what puts them back. Needs no depth | no |
| [`NightFog`](docs/transforms/NightFog.md) | Fog lit by the lamps in the frame: everything fades with distance into a veil as bright and as coloured as the light around it, brightest near the lamps, and each lamp gets a halo. With the illuminator on, an even wall of backscatter across the beam | yes |
| [`CastShadows`](docs/transforms/CastShadows.md) | Shadows of people, vehicles and foliage on upward-facing surfaces, all thrown by one sun — a classic false-positive source. A loss of the direct beam, so texture survives and the shadow is bluer | yes |
| [`GroundFog`](docs/transforms/GroundFog.md) | A layer of fog on the ground: extinction falling exponentially with height, integrated in closed form along each pixel's ray from the camera's height to the point's, so the road is veiled and the roofs stand clear, the horizon closes and the sky above it does not; in banks and wisps laid out on the ground plane | yes |
| [`LocalPlumes`](docs/transforms/LocalPlumes.md) | Steam and exhaust: plumes standing at points of the scene, sized by metres at that distance, widening and thinning as they rise, leaning with the wind, shaded underneath by their own depth, and hiding only what is behind them | yes |
| [`HeatShimmer`](docs/transforms/HeatShimmer.md) | Turbulent air over a long path, from the theory of seeing: each pixel moved by a smooth random field scaled to the tilt `2.91 D^(-1/3) Cn2 L` and blurred by `λ/r0`. Nothing over a street's length, which is the correct answer | yes |
| [`NightInsects`](docs/transforms/NightInsects.md) | Insects in the infrared ring's beam: each a streak as long as it flies in the exposure, holding a fixed amount of light, beaded once a wing beat, lit by its own distance from the lamp; and motes of dust near the lens, one faint disc each | no |

### Lens and housing

| Transform | Models | Needs depth |
| --- | --- | --- |
| [`WaterDroplets`](docs/transforms/WaterDroplets.md) | Water on the glass of a camera focused far beyond it: every drop a lobe of one refracting surface, so drops merge; large drops individually shaped — lopsided and bottom-heavy when pinned, and as runners anything from a blunt slug to a head trailing a thread that breaks into beads; sharpness set by the lens aperture, not chosen; condensation as scattering of the scene's own light; lit by the infrared ring at night | no |
| [`DirtSoil`](docs/transforms/DirtSoil.md) | Grime as a *scattering* layer, not paint, with per-deposit albedos from grey dust through yellow pollen to brown spray. Lit from the whole hemisphere in front of the glass, so it glows under a sky the camera is not looking at; patches over an even film of settled dust | no |
| [`SunFlare`](docs/transforms/SunFlare.md) | Veiling glare, halo, coating-tinted ghost chain, diffraction starburst, dispersion corona, plus CCTV-specific dome scratches, smear and IR leak. The sun may be just outside the frame, leaving only its ghosts on the road | no |
| [`SourceGlare`](docs/transforms/SourceGlare.md) | The same optics on each compact light found in the frame (street lamps, headlights), with strength from the clipped area. Leaves a frame with no lights unchanged | no |
| [`HousingOccluders`](docs/transforms/HousingOccluders.md) | Insects and webs on the dome — hugely magnified and defocused, silhouetted by day, glowing under the IR ring | no |
| [`SpiderWeb`](docs/transforms/SpiderWeb.md) | An orb web a hand's width in front of the lens, lit by the infrared ring: frame, radii and capture spiral in a tilted plane, every thread drawn as the aperture's disc at its own distance, so the near side is bright and blurred wide and the beads of glue and dew read as strings of pearls; loose threads, and the spider at the hub, clipped white. Adds light and hides nothing, which is why a web is invisible by day | no |
| [`IrDomeGlare`](docs/transforms/IrDomeGlare.md) | The ring of LEDs mirrored in a dome that is dirty or badly sealed: a soft band, usually an arc because the lens is tilted off the dome's axis, lobed where the LEDs are few, patchy where the dome is dirty, over an even veil | no |
| [`FrostOnGlass`](docs/transforms/FrostOnGlass.md) | Ice ferns grown in from the rim, with branches at the lattice's sixty degrees, ahead of an even layer and with clear glass between them. Ice scatters and does not absorb: what it covers is replaced by a wide blur of the scene and the light the frost picks up | no |
| [`SnowOnHousing`](docs/transforms/SnowOnHousing.md) | A cap of snow hanging into the top of the view with an uneven lobed edge, and lumps stuck to the glass: opaque, a pale blue grey lit through from above, brightest along its thin edge, and clipped under the infrared ring | no |
| [`LensTamper`](docs/transforms/LensTamper.md) | Paint from a spray can (an opaque core thinning to speckle, with runs), strips of tape, or a cover over the housing; tape and cover pass a little light and no image. Hides the scene by design | no |
| [`FoliageIntrusion`](docs/transforms/FoliageIntrusion.md) | A leafy branch in front of the lens at its own distance, which sizes its leaves, blurs them through the aperture and smears them by the wind over the exposure. Every leaf has its own turn, its midrib and veins, and is lit on its face or lit through, shading those behind it; under the infrared ring each is as bright as it faces the lamp | no |
| [`FocusShift`](docs/transforms/FocusShift.md) | A lens focused too near, as infrared, temperature and a loose ring leave it: every point the aperture's disc, `A f abs(1/s - 1/z)` across, per pixel with depth and evenly without | optional |

### Sensor, ISP and transmission

| Transform | Models | Needs depth |
| --- | --- | --- |
| [`NightIrMode`](docs/transforms/NightIrMode.md) | IR-cut filter out: monochrome by *infrared* reflectance (foliage goes bright, which a luma conversion gets backwards), an illuminator falling as 1/z², gain-scaled noise; with a provider that reads the scene, lit as a road is at night: pools under its street lamps, the traffic's lamps and beams, and signs and plates sending the illuminator's light back | yes |
| [`NightColorMode`](docs/transforms/NightColorMode.md) | The same night as a colour camera sees it: lit from the scene's street lamps in their own colour, sodium's orange or an LED's white, with red tail lamps, white headlamps and their beams, and noise in each channel. With a provider that knows of no lamps, a night lit by the sky's glow alone | yes |
| [`AcFlicker`](docs/transforms/AcFlicker.md) | Mains banding. Cancels exactly when the exposure is a whole number of flicker periods — the basis of a camera's anti-flicker setting | no |
| [`RollingShutter`](docs/transforms/RollingShutter.md) | Row-dependent shear and wobble from camera motion during readout. One of the two transforms that move pixels, with `CameraVibration`, so masks, boxes and keypoints move with it | no |
| [`SensorNoise`](docs/transforms/SensorNoise.md) | Shot and read noise in photoelectrons, set by sensor gain, drawn on the Bayer mosaic and demosaiced, so the grain is coarse and coloured as in a real low-light frame | no |
| [`ExposureTransient`](docs/transforms/ExposureTransient.md) | AE/AWB caught mid-correction. White-balance error is the ratio of two blackbody colours, so it stays on the blue-orange axis real illuminants vary on | no |
| [`WdrToneMapping`](docs/transforms/WdrToneMapping.md) | Local tone mapping, with the black halos a cheap edge-preserving filter leaves | no |
| [`TemporalDenoiseArtifacts`](docs/transforms/TemporalDenoiseArtifacts.md) | 3D-NR's "watercolour" flattening and chroma blotching | no |
| [`VideoCodecArtifacts`](docs/transforms/VideoCodecArtifacts.md) | H.264/H.265 **encoded for real via ffmpeg**, so deblocking, 4:2:0 chroma and rate control come out correct rather than approximated. Optionally simulates stale macroblocks from a lost P-frame | no |
| [`TransmissionFaults`](docs/transforms/TransmissionFaults.md) | Packet loss, **decoded for real**: a short H.264 clip is encoded, packets are removed from the bitstream in bursts, and ffmpeg decodes what is left. Stale, smeared, green and garbage bands that start mid-row and stop at a slice boundary | no |
| [`IrCutFilterFault`](docs/transforms/IrCutFilterFault.md) | The day/night filter stuck open by day: near infrared added to all three channels, then exposure and white balance correcting what they can, which leaves foliage pale against a frame that is neutral on average; or stuck half-way, with its edge across the picture | no |
| [`DemosaicArtifacts`](docs/transforms/DemosaicArtifacts.md) | The frame put back through a Bayer mosaic and a cheap interpolator: false colour and zipped edges on detail as fine as the pixels, limited by the lens's blur and hidden in part by the processor's chroma filter | no |
| [`CameraVibration`](docs/transforms/CameraVibration.md) | The camera swinging on its mount: the frame added up over the path its pointing took during the exposure, so the smear grows with shutter time up to the whole swing, and displaced by the path's mean. Moves pixels: masks, boxes and keypoints follow | no |
| [`AnalogVideoArtifacts`](docs/transforms/AnalogVideoArtifacts.md) | A composite or analog-HD link: the frame **encoded onto a subcarrier and decoded**, so cross-colour and dot crawl are what a decoder leaves; with the cable's low-pass along the line, colour's narrow band and delay, a reflection, hum bars, line jitter and the comb of two fields | no |
| [`OnScreenOverlay`](docs/transforms/OnScreenOverlay.md) | Not physics: the timestamp, camera name and privacy masks that are on nearly every real frame, in a built-in bitmap face. A box that privacy masks cover is dropped | no |

---

## Configuration

### Parameter conventions

Three rules hold across every transform.

**Physical `(min, max)` ranges.** Constructors take ranges of physical
quantities — fog's extinction coefficient, a raindrop's diameter in
millimetres, an aperture's diameter and blade count, a shutter time in
seconds. Ranges are validated by a pydantic `InitSchema`, so an impossible
value fails at construction rather than at render.

**Derived quantities are not exposed.** A rain rate settles the drop-size
distribution, the particle count, and — through published visibility
relations (Atlas 1953 for rain) — how dense the veil is. These are not independent in reality, and
making them independent here is what allows a frame with a downpour's haze
and a drizzle's streaks.

**Sizes are resolution-independent.** Camera-local effects express sizes as
fractions of the image diagonal; effects with a real scene scale work in
metres and project through a stated field of view. A feature's *geometry* is
the same on a 256 px crop and a 4K frame.

### Configuration reference

The knobs that change the picture most, per transform. Every one is a
`(min, max)` range unless noted; see each module's docstring for the full set
and the physics behind it.

| Transform | Start with |
| --- | --- |
| `RandomFog` | `beta_range` (extinction, 1/m), `visibility_range_scene` (in its place: visibility as a multiple of the scene's depth), `scattering_gamma_range` (0 grey, ~1.3 blue haze, 4 Rayleigh), `depth_scale`, `vehicle_lamp_range` (off by default: the traffic's lamps lit through the fog, each with its halo; needs a vehicle reader) |
| `RandomRain` | `rain_rate_range_mm_h`, `drop_diameter_range_mm`, `dsd_shape_mu`, `scene_illuminance_range_lux` and `aperture_diameter_range_mm` (together they meter the shutter time), `exposure_time_range` (the shutter's limits), `wetness_probability`, `wetness_darkening_range`, `puddle_coverage_range`, `min_streak_width_px`, `lamp_glint_range` (night: how far past full scale the lamps lighting the rain were), `min_visibility_scene` (keeps the veil from hiding the far end of a long view), `spray_range` (off by default: mist trailed by the vehicles on a soaked road, with scene labels), `traffic_speed_range_kmh` (how fast that traffic is going, which is most of how much it throws up), `vehicle_lamp_range` (off by default: the traffic's head and tail lamps lit, with their glow and their streaks down the road), `distant_rain_range` (how much of its grain the rain beyond the drawn streaks keeps; 0 is the even veil) |
| `NightRain` | `illuminator_gain_range` (what makes streaks visible), `rain_rate_range_mm_h`, `illuminator_range_m`, `backscatter_range`, `exposure_time_range`, `backscatter_grain_range` (the grain of the unresolved drops' glow; 0 is an even wall), `lamp_cone_range` (how strongly it shows in the light of the scene's lamps, on a frame a night transform lit from them; 0 is not at all) |
| `NightSnow` | `snowfall_rate_range_mm_h`, `illuminator_gain_range`, `illuminator_range_m`, `backscatter_range`, `exposure_time_range`, `glare_range`, `backscatter_grain_range` (the speckle of the unresolved flakes' glow; 0 is an even wall), `lamp_cone_range` (how strongly it shows in the light of the scene's lamps, on a frame a night transform lit from them; 0 is not at all) |
| `NightFog` | `visibility_range_m` (read against the provider's depth range), `visibility_range_scene` (as a multiple of the scene's depth instead), `glow_radius_range`, `illuminator_backscatter_range` (infrared ring on; apply after `NightIrMode`), `lamp_cone_range` (how strongly it shows in the light of the scene's lamps, on a frame a night transform lit from them; 0 is not at all) |
| `RandomSnow` | `snowfall_rate_range_mm_h`, `snow_density_range_g_cm3`, `accumulation_probability`, `accumulation_strength_range`, `vehicle_cover_range` (0 moving traffic, 1 parked overnight), `min_flake_width_px`, `max_flakes`, `min_visibility_scene` (keeps the veil from hiding the far end of a long view), `wheel_track_range` (how far traffic has worn the settled snow into tracks, with scene labels), `distant_snow_range` (how much of its speckle the snow beyond the drawn flakes keeps; 0 is the even veil) |
| `CastShadows` | `casters` (person, vehicle, foliage), `shadow_count_range`, `shadow_scale_range`, `sun_elevation_range_deg`, `direct_fraction_range`, `penumbra_range`, `camera_pitch_deg` |
| `GroundFog` | `visibility_range_m` (at the ground), `layer_height_range_m` (how shallow the layer is), `camera_height_range_m`, `camera_pitch_range_deg`, `patchiness_range`, `patch_scale_range_m`, `depth_scale` (it shows along the distance) |
| `LocalPlumes` | `plume_count_range`, `source_width_range_m`, `rise_range_m`, `opacity_range`, `lean_range` (wind), `brightness_range` |
| `HeatShimmer` | `log10_cn2_range` (-14 mild, -12 hot ground), `horizontal_fov_range_deg` (a long lens), `aperture_diameter_range_mm`, `depth_scale` (it needs a path of a kilometre or so) |
| `NightInsects` | `insect_count_range`, `distance_range_m`, `flight_speed_range_m_s` and `exposure_time_range` (streak length), `wing_beat_range_hz` (lobes), `dust_count_range`, `illuminator_gain_range` |
| `WaterDroplets` | `droplet_count_range`, `droplet_radius_range`, `pane_width_range_mm` (sets the physical scale), `aperture_diameter_range_mm` (sets how sharp the water looks), `coalescence_range`, `condensation_coverage_range`, `condensation_milkiness_range`, `illuminator_glow_range` (infrared ring on) |
| `DirtSoil` | `coverage_range`, `dust_film_range` (even film between the patches), `ambient_light_range` (how much brighter the surroundings are than the frame), `veil_strength_range`, `highlight_boost_range`, `deposits` (which substances), `smudge_count_range` |
| `SunFlare` | `flare_position` (outside `[0, 1]` for a sun out of frame), `off_frame_probability`, `intensity_range`, `ghost_count_range`, `ghost_strength_range`, `aperture_blades_range`, `dispersion_ring_probability`, `scratch_count_range` |
| `SourceGlare` | `intensity_range` (stands for how far past full scale the lights were), `max_sources_range`, `ghost_count_range`, `streak_probability` |
| `HousingOccluders` | `insect_count_range`, `insect_size_range`, `web_count_range`, `defocus_range`, `glow_range` (0 by day, >0 under the IR ring) |
| `SpiderWeb` | `distance_range_m` (hub to lens: nearer is larger, wider-blurred and brighter), `web_radius_range_m`, `tilt_range_deg`, `mesh_spacing_range_mm`, `bead_spacing_range_mm` (against `aperture_diameter_range_mm`: band or pearls), `bead_share_range`, `damage_range`, `illuminator_gain_range`, `stray_thread_count_range`, `spider_probability` |
| `IrDomeGlare` | `intensity_range`, `ring_radius_range`, `arc_range` (whole ring or crescent), `offset_range` (lens tilt), `led_contrast_range`, `veil_range`, `dirt_range` |
| `FrostOnGlass` | `coverage_range` (how far in from the rim), `thickness_range`, `feather_range` (ferns or an even layer), `crystal_scale_range`, `ambient_light_range`, `illuminator_glow_range` (infrared ring on) |
| `SnowOnHousing` | `cap_depth_range`, `lumpiness_range`, `patch_count_range`, `brightness_range` (by day), `glow_range` (infrared ring on) |
| `LensTamper` | `kinds` (spray, tape, cover), `coverage_range`, `opacity_range`, `shade_range` (by day), `glow_range` (infrared ring on) |
| `FoliageIntrusion` | `branch_count_range`, `distance_range_m`, `leaf_length_range_mm`, `reach_range`, `sway_speed_range_m_s`, `illuminator_gain_range` (0 by day; at night also set `leaf_light_range` to 0), `beam_falloff_range` |
| `FocusShift` | `focus_distance_range_m` (where the lens is in fact focused; nearer is worse), `aperture_diameter_range_mm`, `depth_provider` (optional) |
| `NightIrMode` | `illuminator_range_m`, `beam_falloff_range`, `gain_range` (noise, not level), `read_noise_range`, `vegetation_gain_range`, `street_lamp_range` (how much light each street lamp throws under itself: the scene is lit in pools from the lamps the provider knows of, not evenly), `vehicle_lamp_range` (the traffic's lamps and beams, lit where the provider's vehicle reader found them; nothing without one), `retroreflector_range` (what road paint, and the signs and number plates the reader finds, send back of the illuminator's light; 0 is none) |
| `NightColorMode` | `lamp_temperature_range_k` (the street lighting's colour: 1900 to 2200 K is sodium, 3000 to 5700 K an LED), `street_lamp_range`, `vehicle_lamp_range`, `sky_glow_range` (all the light there is where the provider knows of no lamps), `target_level_range`, `gain_range` and `read_noise_range` (noise, in each channel) |
| `AcFlicker` | `mains_frequency_hz` (50 or 60), `readout_time_range_s` (sets the band *count*), `exposure_time_range_s` (sets the depth), `pulse_exponent_range`, `modulation_depth_range` |
| `RollingShutter` | `pan_shift_range_px`, `wobble_amplitude_range_px`, `wobble_cycles_range` |
| `SensorNoise` | `gain_range_db` (the knob: how few electrons make a full-scale picture), `full_well_range_e`, `read_noise_range_e` |
| `ExposureTransient` | `exposure_error_stops_range`, `assumed_temperature_range_k`, `actual_temperature_range_k` |
| `WdrToneMapping` | `compression_range`, `detail_gain_range`, `halo_strength_range` |
| `TemporalDenoiseArtifacts` | `strength_range`, `iterations_range`, `edge_threshold_range`, `chroma_blur_range` |
| `VideoCodecArtifacts` | `codec` (`"h264"` / `"h265"`), `crf_range`, `stale_block_probability` |
| `TransmissionFaults` | `packet_loss_range`, `burst_length_range`, `scene_motion_range` (how far the picture moved since the frame the decoder falls back on: what makes loss visible), `missing_reference_probability`, `concealment_probability` (a decoder that does not conceal shows green), `slices_range`, `frames_since_keyframe_range` |
| `IrCutFilterFault` | `leak_range`, `coverage_range` (1 is fully open; less leaves the filter's edge in the picture), `white_balance_adaptation_range` |
| `DemosaicArtifacts` | `optical_blur_range_px` (a sharp lens is the worst case), `sensor_scale_range` (a coarser sensor, wider fringes), `chroma_filter_range_px` |
| `CameraVibration` | `amplitude_range`, `frequency_range_hz`, `exposure_time_range` (the smear is what the camera moved meanwhile), `direction_range_deg` |
| `AnalogVideoArtifacts` | `luma_bandwidth_range` (cable length), `chroma_bandwidth_range`, `crosstalk_range` (decoder quality), `ghost_strength_range`, `hum_depth_range`, `noise_range`, `comb_shift_range` |
| `OnScreenOverlay` | `timestamp_probability`, `label_probability`, `text_height_range`, `privacy_mask_count_range`, `hidden_box_coverage`, `drawn_by` (camera: before the codec; recorder: after the link) |

Two of these are worth singling out because their effect is not what the name
suggests. `AcFlicker`'s `readout_time_range_s` matters more than
`modulation_depth_range`: band *count* is what makes flicker legible, and two
deep bands read as an exposure difference. `NightIrMode`'s `gain_range` scales
the **noise**, not the signal — the auto-exposure has already placed the
level, and gain is one of the levers it used to get there.

### Ready-made pipelines

Listing every effect with an independent probability produces frames that
could not exist — a sun flare under an IR night sky, hard shadows in the
rain, a downpour under a roof. Two pipelines that cannot are importable
directly:

```python
from surmentations.presets import surveillance, surveillance_lite

transform = surveillance(provider, codec=0.0)   # needs CUDA
transform = surveillance_lite()                 # CPU only, no night
```

**`severity` is the dial for how degraded the camera's own chain is** (1-5,
default 1) and the first thing to change if the output looks too heavy. It
defaults low because a pipeline stacks several effects and per-effect
severity compounds — level 2 on each of three effects is not an ordinary
frame.

**The weather is drawn by how often it happens, and at the strength it
has.** `surveillance` picks one state of the air and the ground per frame
out of ten: dry, a road still wet after the rain, rain at three strengths,
fog thin, dense or with drizzle, snow falling, snow lying. Each is drawn
by the share of time it holds (`rain`, `snow`, `fog`, `wet_road`,
`snow_lying`) raised to `prior_power`: 1 for frames in the proportions a
camera sees, which are four fifths dry, and 0.5, the default, to be shown
more of what is rare. Most rain is light and most fog is thin;
`weather_severity` draws all of it at one level for a pipeline of storms.
Hard shadows, a sun flare and heat shimmer are drawn only where the air
is dry; insects in the beam and a web across the housing only in an
infrared night, which is what lights them, and never with snow.

The ten states by day, each drawn alone on one scene
([`docs/pipelines.md`](docs/pipelines.md) has them at night too):

![the ten states of the air and the ground the pipeline draws](https://desmoteo.github.io/surmentations/assets/pipelines/states.webp)

**It looks at the frame first.** A frame taken at night is not made into a
night again, and one under a roof, which a provider with scene labels can
tell, gets no weather (`read_source=False` turns this off).

Measured at 512 px on one core with `codec=0.0`: `surveillance` 35 ms at
the median and 57 at the 90th percentile on a provider with depth alone,
and a little over twice that on one that also reads labels, geometry and
vehicles, whose first sight of each frame is another 170 ms on the GPU.
It was 12 before the weather was drawn on more than half of the frames
instead of an eighth; `prior_power=1.0` halves it, with four frames in
five dry. [`docs/pipelines.md`](docs/pipelines.md#cost) has what each
argument and each kind of frame costs.

The pipelines hold the transforms they were designed with, and three of
the sixteen added since: `NightInsects`, `SpiderWeb` and `HeatShimmer`, at
`insects=0.25`, `webs=0.10` and `shimmer=0.12`, each only in the states
that can have it ([`docs/pipelines.md`](docs/pipelines.md#insects-webs-and-heat-shimmer)).
The other thirteen are not in them: each needs a probability that says how
often it happens on *your* cameras, which is a fact about a deployment.
Add the ones you want with [`physical_order`](#composition-order), which
knows where each belongs.

A few arguments are worth knowing beside `severity`. `ground_severity`
(0-5) fixes how wet the road is under rain and how much snow has settled;
left unset, each frame draws it over the whole range, independently of how
hard it is raining or snowing. `faults` is the probability of packet-loss
damage (`TransmissionFaults`), 0.03 by default and 0 when `codec=0.0`, so
that setting still runs without ffmpeg. `wdr` (0.30) and `denoise` (0.8 at
night, a quarter of it by day) are how often two settings of a camera's
own processing show, wide dynamic range and the noise reducer: both are 0
or 1 for one camera, and they are the two costliest stages of its chain.

`surveillance_lite` renders no night at all — nothing in it can relight a
daylit frame, and tinting one does not approximate night. Its `night`
fraction only raises the mains-flicker and denoiser artefacts. Use
`surveillance` if you need night as a domain: it has two, a night in colour
under street lighting (`NightColorMode`) and an infrared one, and for a
single camera one of `lit_night` and `ir_night` should be 0. Every
probability is a keyword argument, because the defaults are estimates for a
lowland site in central Europe on a mixed camera fleet and will not match
your footage. [`docs/pipelines.md`](docs/pipelines.md) explains what each
encodes and which to change first.

> **Seed once, not per frame.** `set_random_seed` before every call correlates
> every transform's probability check, so frames come out with nearly all
> effects or nearly none — 27% of frames untouched against 8%. Set the seed on
> the `Compose` once and stream samples through it.

### Severity presets

For ImageNet-C-style sweeps, unpack a preset instead of picking ranges by
hand. Levels are 1-5 and every ladder is tested for monotonicity.

```python
from surmentations.presets import SEVERITY
from surmentations.lens import WaterDroplets

transform = WaterDroplets(p=1.0, **SEVERITY[3]["WaterDroplets"])
```

The ground under rain and snow has a ladder of its own, `GROUND_SEVERITY`,
from 0 (dry road, bare ground) to 5 (soaked, buried). How wet a road is and
how much snow lies on it follow how long the weather has lasted, not how
hard it is coming down, so the two are combined freely:

```python
from surmentations.presets import GROUND_SEVERITY, SEVERITY

# A blizzard that has only just started: heavy fall, bare ground.
RandomSnow(depth_provider=provider, **SEVERITY[5]["RandomSnow"],
           **GROUND_SEVERITY[0]["RandomSnow"])
# The morning after: a few flakes over deep snow.
RandomSnow(depth_provider=provider, **SEVERITY[1]["RandomSnow"],
           **GROUND_SEVERITY[5]["RandomSnow"])
```

Without a ground entry the transform draws the ground's state itself, over
the whole range. The pipelines do the same unless given `ground_severity`.

### Composition order

These model stages of one imaging chain — scene → atmosphere → optics →
sensor → ISP → codec — and applying them out of order produces images that
could not exist (codec blocking underneath lens blur means the lens blurred
an already-compressed image). `physical_order` sorts a list into it:

```python
from surmentations.presets import physical_order

transform = A.Compose(physical_order([
    VideoCodecArtifacts(p=0.4),
    RandomRain(depth_provider=provider, p=0.3),
    DirtSoil(p=0.2),
]))
```

It sorts a *list*; it does not build the `Compose`, so you keep control of
probabilities and seeding. Sorting is stable, so transforms sharing a stage
keep the order you gave them.

Two things it knows that are easy to get wrong by hand. What the camera
lights itself (`NightRain`, `NightSnow`, `NightFog`, `NightInsects`,
`SpiderWeb`, `IrDomeGlare`) goes **after** `NightIrMode`: that light is
already in the sensor's terms, and converted with the scene it would be
lit by the depth of whatever is behind it. And `OnScreenOverlay` goes
before the codec when the camera draws it and after the link when the
recorder does, by its `drawn_by`.

The transforms that sit on the housing and can be lit either way
(`WaterDroplets`, `HousingOccluders`, `FrostOnGlass`, `SnowOnHousing`,
`LensTamper`, `FoliageIntrusion`) sort with the optics. Under the infrared
ring, give them their glow or illuminator gain and place them after
`NightIrMode` yourself, as the pipelines do for water.

### Seeding

Randomness goes through each transform's own RNG, so pipeline-level seeding
reaches these transforms exactly as it does built-in ones. Build the
pipeline with this library's `Compose`, `OneOf` and `Sequential`, which
the ready-made pipelines use:

```python
from surmentations.base import Compose

transform = Compose([...], seed=42)
```

Albumentations' own containers hand the one seed to every child, and each
child builds its generators from it, so siblings draw the same numbers:
two transforms at `p=0.3` fire together or not at all, for ever. These
containers give each child a seed of its own, drawn from the seed and its
place, so the same seed still gives the same pictures.

### Labels an effect leaves illegible

Every effect changes the picture and leaves the labels as they were, so a
box round a car is still there after fog or a night has taken the car.
`LabelLegibility`, last in a pipeline, judges each labelled object against
the frame the pipeline was given and deals with the label of one that can
no longer be made out:

```python
from surmentations.base import Compose
from surmentations.legibility import LabelLegibility
from surmentations.presets import surveillance

pipeline = surveillance(provider)
transform = Compose(
    [*pipeline.transforms, LabelLegibility(task="recognise", on_doubt="ignore")],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
)
out = transform(image=image, bboxes=boxes, labels=labels)
out["bboxes"], out["ignore_regions"]   # kept boxes; regions for the loss to leave out
```

- **Where the object's pixels were replaced** (tape, a cap of snow, a
  privacy mask, a leaf), its label goes and a semantic mask's pixels become
  background. The transforms that replace pixels say where, exactly.
- **Where it can no longer be made out** (fog, the dark, noise, a smear),
  `on_doubt` decides. `"ignore"`, the default, suppresses the label: it
  leaves the labels and its region comes back under `ignore_regions`, for a
  loss to count neither as an object nor as background; a semantic mask's
  pixels become `ignore_index`. `"drop"` removes it; `"blank"` removes it
  and paints the object out with its surroundings; `"keep"` keeps it.
- **How it judges**: the finest detail still resolved across the object,
  octave by octave, where the clean frame's edges still stand clear of the
  noise in log light; Johnson's criteria for the task (1, 4 or 6.4 cycles
  to detect, recognise or identify) give a probability, relative to the
  clean frame's, and a label is kept from `keep_at` (0.5).

It adds about 13 ms a frame of eight objects on one core.
[`docs/label_legibility_assessment.md`](docs/label_legibility_assessment.md)
has the design.

---

## Performance

Milliseconds per call on **one core**, which is what a `DataLoader` worker
has: an Intel Core Ultra 9 285H performance core with OpenCV held to one
thread, fast mode, the compiled kernels built. Each figure is the mean over
four of the example scenes and two seeds of the least of two calls after a
warm-up, and the least of three such runs. Between runs most figures moved
by under 5%, but one more than doubled once; absolute numbers will differ
on your hardware, and the ordering is the part to plan around.

```bash
taskset -c 2 python scripts/benchmark_single_core.py run --out times.json
python scripts/benchmark.py --sizes 512 1024     # parameters and rendering apart
```

The sixteen transforms added after that run, `SpiderWeb` and the fifteen
that followed it, were timed the same way in two runs.

[`docs/native_kernels.md`](docs/native_kernels.md) has the same transforms
against the code before the native kernels, on this machine and on a Jetson
Orin NX, and what exact mode costs.

### CPU-only transforms — ms per image

| Transform | 512 px | 1024 px |
| --- | ---: | ---: |
| `AcFlicker` | 1.1 | 4.0 |
| `OnScreenOverlay` | 1.4 | 6.7 |
| `RollingShutter` | 2.0 | 7.7 |
| `ExposureTransient` | 2.2 | 8.6 |
| `CameraVibration` | 3.0 | 32.8 |
| `DemosaicArtifacts` | 3.0 | 14.2 |
| `IrDomeGlare` | 3.7 | 15.8 |
| `NightInsects` | 4.9 | 26.7 |
| `HousingOccluders` | 6.1 | 16.7 |
| `SensorNoise` | 6.1 | 27.5 |
| `DirtSoil` | 6.4 | 22.9 |
| `IrCutFilterFault` | 6.5 | 33.0 |
| `SunFlare` | 6.7 | 21.0 |
| `FocusShift`, without depth | 8.1 | 13.3 |
| `VideoCodecArtifacts`, with PyAV | 9.2 | 29.5 |
| `SpiderWeb` | 9.7 | 35.2 |
| `SourceGlare` | 11.6 | 38.4 |
| `LensTamper` | 12.1 | 54.8 |
| `NightSnow` | 12.9 | 45.7 |
| `TemporalDenoiseArtifacts` | 14.8 | 46.0 |
| `SnowOnHousing` | 14.9 | 52.3 |
| `AnalogVideoArtifacts` | 20.6 | 93.7 |
| `WaterDroplets` | 22.7 | 70.1 |
| `TransmissionFaults`, with PyAV | 23.2 | 79.3 |
| `WdrToneMapping` | 26.0 | 39.6 |
| `NightRain` | 26.1 | 63.0 |
| `FrostOnGlass` | 26.4 | 142.4 |
| `FoliageIntrusion` | 35.2 | 121.0 |
| `VideoCodecArtifacts`, through ffmpeg | 144.2 | 180.1 |
| `TransmissionFaults`, through ffmpeg | 171.7 | 279.4 |

The two codec transforms run ffmpeg as two subprocesses per call unless
PyAV is installed (see [Install](#install)), and then encode in process.

### Depth-conditioned transforms — ms per image

With depth already estimated: what every depth-conditioned transform costs
once one provider holds the frame's depth.

| Transform | 512 px | 1024 px |
| --- | ---: | ---: |
| `NightIrMode` | 10.1 | 42.8 |
| `LocalPlumes` | 14.1 | 74.3 |
| `HeatShimmer`, over a long view | 14.8 | 94.0 |
| `RandomFog` | 20.6 | 91.7 |
| `NightColorMode` | 21.7 | 87.4 |
| `CastShadows` | 20.8 | 46.9 |
| `NightFog` | 24.2 | 90.5 |
| `GroundFog` | 24.9 | 114.6 |
| `RandomRain`, dry road | 27.7 | 86.0 |
| `RandomSnow` | 59.9 | 161.7 |
| `RandomRain`, wet road | 87.0 | 197.0 |

`RandomRain` draws the wet road in 40% of frames by default
(`wetness_probability`), so it costs about 51 ms at 512 px and 130 ms at
1024 px on average. `NightIrMode` is timed here on a provider with depth
alone; lit from street lamps and vehicles on a provider that knows of them
it costs more, by the amount given in
[`docs/road_surface.md`](docs/road_surface.md).

The depth map itself is estimated once per new frame on the GPU: 12 and
16 ms at 512 and 1024 px (an RTX PRO 1000 Blackwell Laptop GPU, torch 2.11
+ CUDA 12.8, in fast mode, which runs the network in half precision). Every
depth-conditioned transform that shares one `SharedDepthProvider` reuses it:
chained in one pipeline, only the first pays. Constructing a provider per
transform pays it again in each.

A provider asked for [scene labels and geometry](docs/scene_understanding.md)
makes them with the depth, once per new frame: 33 ms for the labels at
either size, with the default network, and 5 to 6 ms for the geometry.

The saving holds along a whole pipeline, not only between adjacent
transforms: each transform here passes the scene on with its output, so depth
is estimated from the clean frame and reused by everything downstream —
`CastShadows` → `RandomFog` → `RandomRain` all firing on one frame is one
estimate, and so is `RandomFog` → `DirtSoil` → `NightIrMode`. Two things end
the chain, and the next depth-conditioned transform then estimates afresh:
`RollingShutter`, which moves pixels, and any transform from outside this
library, which cannot be assumed not to. If you know such a step keeps
geometry (a colour jitter, a blur), `provider.carry(before, after)` says so.

### Choosing a budget

A `DataLoader` worker must keep ahead of the GPU. Rough guidance, at 512 px:

- **Cheap enough to apply liberally** (under ~10 ms): `AcFlicker`,
  `OnScreenOverlay`, `RollingShutter`, `ExposureTransient`,
  `CameraVibration`, `DemosaicArtifacts`, `IrDomeGlare`, `NightInsects`,
  `HousingOccluders`, `SensorNoise`, `NightSnow`, `DirtSoil`,
  `IrCutFilterFault`, `SunFlare`, `FocusShift`, `SpiderWeb`, `NightIrMode`,
  and `VideoCodecArtifacts` with PyAV.
- **Moderate** (10-30 ms): `SourceGlare`, `LensTamper`, `LocalPlumes`,
  `TemporalDenoiseArtifacts`, `HeatShimmer`, `SnowOnHousing`,
  `AnalogVideoArtifacts`, `NightRain`, `WaterDroplets`, `GroundFog`,
  `WdrToneMapping`, `FrostOnGlass`, `TransmissionFaults` with PyAV,
  `CastShadows`, `RandomFog`, `NightFog`, and `RandomRain` on a dry road.
- **Expensive, keep `p` low**: `FoliageIntrusion`, `RandomSnow`,
  `RandomRain` on a wet road, and `VideoCodecArtifacts` and
  `TransmissionFaults` without PyAV.

Cost is paid only when `p` fires, so a pipeline of ten effects at `p=0.2`
costs about two of them per image, not ten. The depth estimate is the
exception: it is paid once if *any* depth-conditioned transform fires.

---

## Usage considerations

**These are per-image transforms.** Anything whose mechanism is genuinely
*temporal* — 3D-NR's motion trails, banding drifting between frames — is
rendered only as the single-frame state it leaves behind. Each module says so
where it applies rather than approximating silently. If you train on video and
need temporal coherence, you need to hold the draws fixed across a clip
yourself.

**`p` decides whether; parameters decide how strongly.** A transform
constructed with `p=1.0` will change the image. Ranges are not permitted to
include "no effect" as a legal draw — that was a bug twice, and both are now
guarded by tests. If you want an effect to sometimes not apply, lower `p`.

**Precipitation intensity is resolution-dependent, by design.** Geometry is
resolution-independent, but the amount of precipitation visible as *discrete
particles* rises with image size (roughly 6-8× over 256→1920 px), because the
cut deciding what is drawn individually rather than left to the veil is in
pixels — a bigger sensor really does resolve more drops. Total extinction is
unaffected (rendered particles carry ~0.3% of it). If you train multi-scale
and want the look held constant, `min_streak_width_px` and
`min_flake_width_px` are the knobs.

**Depth is pseudo-metric, not measured.** Monocular depth has no absolute
scale, so it is mapped onto a near and a far plane — 3 m and 50 m by default,
set in `depth/scale.py`. Effects that depend on distance *linearly* (fog,
rain, snow) tolerate a wrong guess; `NightIrMode`'s inverse-square illuminator
does not, and a scene whose real near plane is 10 m will read too contrasty
at the bottom of the frame.

The calibration belongs to the provider, so one setting reaches every
depth-conditioned transform:

```python
provider = SharedDepthProvider(DepthAnythingV2Estimator(), depth_range_m=(1.0, 12.0))
```

`RandomFog` can additionally override the far plane for itself with
`depth_scale`.

Or let the camera's height set the scale: a provider with an
[installation geometry](docs/scene_understanding.md#metres) whose height is
known, given or estimated, puts the depth in metres measured from the
camera, frame by frame, and `depth_range_m` is not used.

**Severity is extinction per metre, so it has to match the scene's scale.**
The presets suit the default 3-50 m. Calibrate a provider to a landscape and
the same fog or snow hides everything past a few hundred metres, and heavy
rain everything past a kilometre or two — which is what that weather does,
and leaves a blank frame. For long views, or a dataset that mixes scales,
set the air against the scene and not in metres: `visibility_range_scene`
on `RandomFog` and `NightFog` is visibility as a multiple of the scene's
depth, and `min_visibility_scene` on `RandomRain` and `RandomSnow` keeps
their veils from closing its far end. The transforms measure the scene's
depth themselves, as the distance of the farthest thing that is not sky, so
the setting holds whether the provider's distances are given or measured;
`docs/generate_scene_examples.py` uses 0.6 for fog and 1.5 and 1.0 for rain
and snow on scenes from thirty metres to six kilometres deep.

**The scene's own conditions are mostly not detected.** Nothing here knows
that a frame is under a roof, so snow settles indoors if you ask it to.
Three things are inferred, by heuristics that are documented as such:

- **Where the sky is** (`utils/sky.py`): featureless, at a single depth, and
  reached from the top of the frame. It misses sky seen through branches and
  can take a pale distant surface touching the top edge for sky.
- **What colour a veil is** (`utils/color.py`): the horizon sky's, read off
  the sky where there is one and inferred where there is not. On a real night
  frame fog, rain and snow therefore stay dark instead of taking the colour
  of the street lamps. They are still daylight models: fog at night does not
  glow round the lamps: `NightFog`, `NightRain` and `NightSnow` are the
  night models, and the pipelines use them.
- **What is paving** (`atmosphere/rain.py`): everything level that is not
  green vegetation. Bare soil and gravel still get a film of water.

Rain and snow also relight the frame towards overcast in step with their
rate (`overcast_range`). That softens a sunny scene; it cannot remove the
shadows it has.

**Clips.** Passed as `images=`, an `(N, H, W, 3)` array is treated the way
Albumentations treats it: one set of parameters for every frame. Depth is
estimated once, from the first frame, on the assumption that a clip is one
camera looking at one scene. Nothing *advances* between frames, though — rain
streaks sit in the same place in each — so this gives consistent conditions
across a clip, not motion.

**Annotations.** Nearly every effect changes appearance only, so masks,
boxes and keypoints pass through untouched. The exceptions:

- `RollingShutter` and `CameraVibration` move pixels and move their
  annotations with them: masks are resampled, boxes follow (for a shear,
  the box enclosing the sheared original), keypoints follow.
- `OnScreenOverlay` drops a box that its privacy masks cover by
  `hidden_box_coverage` (0.9) of its area, since what was in it is gone.
- `LensTamper`, `SnowOnHousing` and `FoliageIntrusion` hide part of the
  scene and leave annotations alone: whether an object behind a branch is
  still a positive is your decision.
- `HeatShimmer` and `AnalogVideoArtifacts` displace pixels by a pixel or
  two, a few at most, and do not move annotations.

**Input is RGB, `uint8` or float in `[0, 1]`.** Anything else — greyscale,
RGBA, 16-bit — is refused with an error before any parameter is sampled,
rather than rendered wrongly. The effects model colour radiance, so there is
no greyscale form of them; convert a mono frame to three channels first.
`RollingShutter` is the exception, since it only moves pixels: it takes any
channel count and bit depth.

**Compositing is in linear light.** If you chain these with other
augmentation libraries that work on gamma-encoded values, the ordering matters
more than usual — put colour-space-naive transforms outside this chain, not in
the middle of it.

**GPU memory.** One `DepthAnythingV2Estimator` holds a model on the GPU.
With multiple `DataLoader` workers each constructing their own, memory scales
with worker count — construct the provider inside the worker init, and size
`num_workers` accordingly.

---

## Effect gallery

Rendered by [`docs/generate_examples.py`](docs/generate_examples.py) on the
reference photo below. Each effect gets one image per severity level, so
`<effect>_0.webp` is severity 1 and `<effect>_4.webp` is severity 5. Each
variant also re-rolls the setup knobs severity leaves alone — sun position,
camera pitch, deposit mix, mains frequency — since those change the picture as
much as the weather does.

**Twice, where the scene's reading matters.** Nine rows are drawn a second
time, as `<effect>_aware_<n>.webp`, with a provider that reads the scene: its
[labels, its camera, its vehicles and its street lamps](#what-a-provider-can-read--optional).
Those are the rows whose picture depends on it, and each twin starts from
the same variant as its plain row. The plain picture is what any provider
gives; the second is what the same call gives once the provider knows where
the road, the traffic and the lamps are.

Each cell shows the **median of seven draws**, not the first: effects that can
legitimately draw themselves out of existence otherwise produce a cell the
reader cannot tell from a bug. The full sweep of 39 rows at five levels is on
[the gallery's site](https://desmoteo.github.io/surmentations/#examples).

None of these pictures is kept in git. CI renders them from the code on every
push to `main` and publishes them to GitHub Pages, and the pages here link to
them there; see [Development](#development).

The same effects over eight other scenes — a crossing from straight above, a
covered car park, a real night street, mountains across a valley — are in
[`docs/examples/scenes/`](docs/examples/scenes/README.md), one page per scene,
with the shipped pipelines run over each. That set is the place to look for
where the models stop fitting, and each page says what the transforms had to
be told about the scene to get there. Each page also shows what a provider
that reads the scene read off it, its labels with the horizon drawn across
them, and every effect that depends on that reading twice: without it and
with it, from the same parameters and the same seed.

![base](https://desmoteo.github.io/surmentations/examples/00_base_scene.webp)

### Weather, by level and with the scene read

The weather is what the library is for, so these are shown at three levels,
light, moderate and severe, and then with the scene read. Rain: with the
road and its traffic known, the vehicles trail spray by their heading and
speed and drive with their lamps on.

| Rain, severity 1 | severity 3 | severity 5 | severity 4, scene read: spray and lamps |
| --- | --- | --- | --- |
| ![rain, severity 1](https://desmoteo.github.io/surmentations/examples/rain_0.webp) | ![severity 3](https://desmoteo.github.io/surmentations/examples/rain_2.webp) | ![severity 5](https://desmoteo.github.io/surmentations/examples/rain_4.webp) | ![severity 4, scene read: spray and lamps](https://desmoteo.github.io/surmentations/examples/rain_aware_3.webp) |

Snow: with the road known, traffic has worn wheel tracks in what has settled.

| Snow, severity 1 | severity 3 | severity 5 | severity 4, scene read: wheel tracks |
| --- | --- | --- | --- |
| ![snow, severity 1](https://desmoteo.github.io/surmentations/examples/snow_0.webp) | ![severity 3](https://desmoteo.github.io/surmentations/examples/snow_2.webp) | ![severity 5](https://desmoteo.github.io/surmentations/examples/snow_4.webp) | ![severity 4, scene read: wheel tracks](https://desmoteo.github.io/surmentations/examples/snow_aware_3.webp) |

Fog needs depth and nothing else, and reads the same with either provider.

| Fog, severity 1 | severity 3 | severity 5 | Ground fog |
| --- | --- | --- | --- |
| ![fog, severity 1](https://desmoteo.github.io/surmentations/examples/fog_0.webp) | ![severity 3](https://desmoteo.github.io/surmentations/examples/fog_2.webp) | ![severity 5](https://desmoteo.github.io/surmentations/examples/fog_4.webp) | ![ground fog](https://desmoteo.github.io/surmentations/examples/ground_fog_2.webp) |

The ground has its own ladder, under light weather that does not change:

| Soaked road, light rain | scene read: spray and lamps | Deep snow, light fall | scene read: wheel tracks |
| --- | --- | --- | --- |
| ![soaked road, light rain](https://desmoteo.github.io/surmentations/examples/wet_ground_4.webp) | ![scene read: spray and lamps](https://desmoteo.github.io/surmentations/examples/wet_ground_aware_4.webp) | ![deep snow, light fall](https://desmoteo.github.io/surmentations/examples/snow_cover_4.webp) | ![scene read: wheel tracks](https://desmoteo.github.io/surmentations/examples/snow_cover_aware_4.webp) |

| Sun flare | Cast shadows | Plumes | Heat shimmer, through a long lens |
| --- | --- | --- | --- |
| ![sun flare](https://desmoteo.github.io/surmentations/examples/sun_flare_3.webp) | ![cast shadows](https://desmoteo.github.io/surmentations/examples/cast_shadows_3.webp) | ![plumes](https://desmoteo.github.io/surmentations/examples/local_plumes_3.webp) | ![heat shimmer, through a long lens](https://desmoteo.github.io/surmentations/examples/heat_shimmer_4.webp) |

### Night

A daylight frame made into a night one. With depth alone it is lit evenly;
with the scene read it is lit as a road is, from its street lamps and its
traffic, and under infrared its signs and plates send the illuminator's
light back.

| Night IR | Night IR, scene read | Night in colour, sky glow only | Night in colour, scene read |
| --- | --- | --- | --- |
| ![night ir](https://desmoteo.github.io/surmentations/examples/night_ir_3.webp) | ![night ir, scene read](https://desmoteo.github.io/surmentations/examples/night_ir_aware_3.webp) | ![night in colour, sky glow only](https://desmoteo.github.io/surmentations/examples/night_color_3.webp) | ![night in colour, scene read](https://desmoteo.github.io/surmentations/examples/night_color_aware_3.webp) |

What falls at night is lit by the camera, and with the scene read by the
scene's own lamps as well: a cone under each street lamp.

| Night rain | Night rain, scene read | Night snow | Night snow, scene read |
| --- | --- | --- | --- |
| ![night rain](https://desmoteo.github.io/surmentations/examples/night_rain_3.webp) | ![night rain, scene read](https://desmoteo.github.io/surmentations/examples/night_rain_aware_3.webp) | ![night snow](https://desmoteo.github.io/surmentations/examples/night_snow_3.webp) | ![night snow, scene read](https://desmoteo.github.io/surmentations/examples/night_snow_aware_3.webp) |

| Night fog, over a night photograph | Night fog, over this scene, its lamps known | Glare from lamps | Insects in the beam |
| --- | --- | --- | --- |
| ![night fog, over a night photograph](https://desmoteo.github.io/surmentations/examples/night_fog_3.webp) | ![night fog, over this scene, its lamps known](https://desmoteo.github.io/surmentations/examples/night_fog_aware_3.webp) | ![glare from lamps](https://desmoteo.github.io/surmentations/examples/source_glare_3.webp) | ![insects in the beam](https://desmoteo.github.io/surmentations/examples/night_insects_3.webp) |

### Glass and housing

| Water on the glass, iris open | iris stopped down | under the infrared ring | Dirt and soil |
| --- | --- | --- | --- |
| ![water on the glass, iris open](https://desmoteo.github.io/surmentations/examples/water_droplets_1.webp) | ![iris stopped down](https://desmoteo.github.io/surmentations/examples/water_droplets_2.webp) | ![under the infrared ring](https://desmoteo.github.io/surmentations/examples/night_droplets_3.webp) | ![dirt and soil](https://desmoteo.github.io/surmentations/examples/dirt_soil_2.webp) |

| Frost | Snow on the housing | The ring in the dome | Tampering |
| --- | --- | --- | --- |
| ![frost](https://desmoteo.github.io/surmentations/examples/frost_on_glass_2.webp) | ![snow on the housing](https://desmoteo.github.io/surmentations/examples/snow_on_housing_3.webp) | ![the ring in the dome](https://desmoteo.github.io/surmentations/examples/ir_dome_glare_2.webp) | ![tampering](https://desmoteo.github.io/surmentations/examples/lens_tamper_3.webp) |

| A spider's web, far | nearer | against the housing | The housing's own edge |
| --- | --- | --- | --- |
| ![a spider's web, far](https://desmoteo.github.io/surmentations/examples/spider_web_0.webp) | ![nearer](https://desmoteo.github.io/surmentations/examples/spider_web_2.webp) | ![against the housing](https://desmoteo.github.io/surmentations/examples/spider_web_4.webp) | ![the housing's own edge](https://desmoteo.github.io/surmentations/examples/housing_occluders_3.webp) |

| A branch | A branch under the infrared ring | Focus | Vibration |
| --- | --- | --- | --- |
| ![a branch](https://desmoteo.github.io/surmentations/examples/foliage_intrusion_3.webp) | ![a branch under the infrared ring](https://desmoteo.github.io/surmentations/examples/night_foliage_2.webp) | ![focus](https://desmoteo.github.io/surmentations/examples/focus_shift_3.webp) | ![vibration](https://desmoteo.github.io/surmentations/examples/camera_vibration_3.webp) |

### Sensor, processor and link

| Sensor noise | AC flicker | Rolling shutter | Exposure hunting |
| --- | --- | --- | --- |
| ![sensor noise](https://desmoteo.github.io/surmentations/examples/sensor_noise_3.webp) | ![ac flicker](https://desmoteo.github.io/surmentations/examples/ac_flicker_4.webp) | ![rolling shutter](https://desmoteo.github.io/surmentations/examples/rolling_shutter_3.webp) | ![exposure hunting](https://desmoteo.github.io/surmentations/examples/exposure_transient_3.webp) |

| Wide dynamic range | Temporal denoising | Demosaicing | Filter stuck open |
| --- | --- | --- | --- |
| ![wide dynamic range](https://desmoteo.github.io/surmentations/examples/wdr_tone_mapping_3.webp) | ![temporal denoising](https://desmoteo.github.io/surmentations/examples/denoise_artifacts_3.webp) | ![demosaicing](https://desmoteo.github.io/surmentations/examples/demosaic_artifacts_4.webp) | ![filter stuck open](https://desmoteo.github.io/surmentations/examples/ir_cut_filter_fault_3.webp) |

| Codec | Packet loss | Analog link | Overlay |
| --- | --- | --- | --- |
| ![codec](https://desmoteo.github.io/surmentations/examples/video_codec_3.webp) | ![packet loss](https://desmoteo.github.io/surmentations/examples/transmission_faults_3.webp) | ![analog link](https://desmoteo.github.io/surmentations/examples/analog_video_3.webp) | ![overlay](https://desmoteo.github.io/surmentations/examples/on_screen_overlay_2.webp) |

The `rain` and `snow` rows raise the ground with the weather, level for
level, because a row has one axis. `wet_ground` and `snow_cover` hold the
weather at level 1 and climb `GROUND_SEVERITY` alone: the two are separate
ladders, and `snow_cover_4` is a light fall over deep snow.

`night_rain`, `night_snow`, `spider_web`, `night_insects`, `ir_dome_glare`
and `night_foliage` are rendered over an IR night frame rather than the
daylit base scene, because what they draw is lit by the camera: on a sunny road it would be as invisible as `RandomRain`'s streaks,
which is the finding the transforms exist to answer. The `night_snow` row is
drawn over a road snow has settled on, level for level: `NightSnow` draws
the flakes the lamp lights and nothing else, and the snow on the ground is
`RandomSnow`'s, built with `max_flakes=0` and put before `NightIrMode`, which
is the order the pipelines use. `night_droplets` is `WaterDroplets`
over the same frame with `illuminator_glow_range` set, which is how the
pipelines build it at night. `night_fog` and `source_glare` act on the lights
in a frame, which the daylit reference does not have, so they are rendered
over the night street from `sample_scenes/`. `night_fog_aware` is the
exception: it is fog over this scene made a night of in colour, whose lamps
the provider knows, and is lit by them.

In the `_aware` rows, `rain` and `wet_ground` have `spray_range`,
`vehicle_lamp_range` and a motorway's `traffic_speed_range_kmh` set, which
are off by default and do nothing without the scene's labels; everything
else in them is what the defaults do once the provider reads the scene.
`night_color` without that reading is a night with no lamps in it, lit by
the sky's glow, which is the only night a provider with depth alone can
light.

`heat_shimmer` is rendered as through a 3 to 5 degree lens with the scene
taken to reach 6 km, since that is where shimmer is: through this
photograph's own wide lens it rightly does nothing. `ground_fog` and
`local_plumes` are given something like the photograph's real distances in
place of the provider's stand-in of a few tens of metres, because a layer
on the ground shows along the distance and a vent has a size.
`night_foliage` is `FoliageIntrusion` with the infrared ring on and no
daylight on the leaves. `lens_tamper` shows spray, tape and a cover in
turn along its row.

The `water_droplets` row alternates a stopped-down iris with an open one
(0.6, 2.4, 0.9, 2.8 and 0.5 mm). That is the largest single difference
between two frames of wet glass, and it is a property of the lens, not of the
weather: behind a 2.5 mm aperture most drops are narrower than the blur disc
and dissolve into soft patches of lost contrast.

---

## Auditing the physics

Most quantities here are *derived* — a rain rate fixes a size distribution,
which fixes a drop count, which fixes each streak's opacity. An image only
shows whether the result looks plausible, not whether that chain held, so the
draws behind each frame can be recorded and checked:

```bash
PYTHONPATH=. python scripts/diagnose.py --out diagnostics --seeds 3
```

That writes one frame plus one compact JSON record per call: the constructor
ranges, the physical quantities actually sampled, and a summary of every
parameter handed to the renderer. Recording is opt-in and costs one flag check
when off; transforms do not opt in, `CameraTransform` instruments every
subclass automatically.

```python
from surmentations.diagnostics import recording, write_jsonl

with recording() as log:
    augmented = transform(image=image)["image"]
write_jsonl(log, "draws.jsonl")
```

[`docs/physical_review.md`](docs/physical_review.md) is a critical assessment
built from such sweeps across four passes. Every derived chain that can be
checked closes to float precision — drop counts, flake sizes, flicker depth,
veil extinction — and the document is equally explicit about what did not hold
and what the failures had in common.

---

## Development

```bash
pip install -e ".[dev]"                      # also builds the native kernels
ruff check src tests docs scripts hatch_build.py
mypy src/surmentations
pytest tests -q
SURMENTATIONS_NATIVE=0 pytest tests -q   # the numpy path alone
SURMENTATIONS_FAST=0 pytest tests -q     # exact mode
python hatch_build.py                        # rebuild after editing native/*.c
```

Depth-conditioned tests skip automatically when no CUDA device is present.
Regenerate visual-regression goldens deliberately — only when a render
intentionally changes:

```bash
python -m tests.visual_regression.generate_goldens
```

The documentation's pictures are rendered, not kept: they are ignored by git,
and [`.github/workflows/gallery.yml`](.github/workflows/gallery.yml) renders
them on every push to `main` that touches the code, the docs or the
photographs, and publishes them with the code reference to
[GitHub Pages](https://desmoteo.github.io/surmentations/). Run it by hand from
the Actions tab to render a branch; it keeps the pictures as a build artifact
and publishes only `main`'s. To look at them locally, render them, in this
order, into the places the site serves them from:

```bash
python docs/generate_examples.py          # the severity gallery
python docs/generate_scene_examples.py    # every effect over sample_scenes/
python docs/make_scene_understanding_figures.py   # the labels and geometry guide's figures
python docs/make_road_surface_figures.py          # the road surface guide's figures
python docs/make_pipeline_figures.py              # the pipelines guide's figures
python docs/make_banner.py                        # the banner, from the severity gallery
```

They are quick on a CUDA GPU. Without one the networks run on the CPU, as
they do in CI: slowly, but to the same pictures within the difference
between half and full precision.

The code reference is a site generated from the source: Python by autodoc,
the C kernels by Hawkmoth through libclang, the shell scripts and CI
workflows from their header comments. Every module, class, function and C
declaration is documented in the source, and `tests/unit/test_documentation.py`
fails if one is not:

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs/site docs/site/_build/html
```

Two test files are worth knowing about because they guard properties rather
than pixels: `test_severity_ladders.py` checks every severity ladder is
monotonic and that level 1 is visible, and `test_gallery_configs.py` renders
the configuration the *gallery* builds — overrides included — and requires
each cell to change the image. The second exists because a stale override in
the gallery once discarded a library fix while every library test passed.

---

## License

MIT — see [LICENSE](LICENSE).

The photographs in [`sample_scenes/`](sample_scenes/) and the images derived
from them, shown on the pages under [`docs/examples/scenes/`](docs/examples/scenes/), are not
covered by it. They are from Wikimedia Commons under CC BY, CC BY-SA and CC0
licences, listed with their authors in
[`sample_scenes/SOURCES.md`](sample_scenes/SOURCES.md).
