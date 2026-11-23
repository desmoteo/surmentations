# Labels for what an augmentation leaves legible: an assessment

21 November 2026. Built on 23 November as
`surmentations.legibility.LabelLegibility`; "What was built" at the end
says where it differs from what is proposed here.

## The question

Every transform in this library changes the picture and, with three
exceptions, leaves the labels as they were. A box drawn round a car on a
clear day is still there after dense fog has taken the car, after a
packet loss has smeared it away, or after a night has been made of the
frame and the car stands in the dark beyond the illuminator's reach. A
network trained on that frame is told that a car is there and is
rewarded for finding one in grey fog or in the dark.

Can a reliable method decide, object by object, whether what a label
describes can still be made out after augmentation, and deal with those
that cannot, as an option the caller turns on and tunes?

## The short answer

Yes, for most of what the library does, provided the method is built to
err in one direction. There are three things a filter can do with a
label it doubts:

| Label after augmentation | What a network learns from it | Cost |
| --- | --- | --- |
| **Kept**, though the object can no longer be made out | to fire on fog, darkness, smears and noise | false positives in the field: the worst outcome |
| **Made background**, though the object can still be made out | to suppress real objects that are hard to see | missed detections in exactly the conditions augmentation is for |
| **Ignored**: no loss either way inside the object | nothing, from that object | some lost training signal |

So the method should keep what is legible, ignore what is doubtful, and
make background only what is certainly gone, because its pixels were
replaced. Under that rule the hard part, judging legibility, only has to
be roughly right to be safe: an error pushes a label into "ignored", not
into a lie.

How reliable each judgement can be depends on how the transform took the
object away:

* **Exactly, when pixels are replaced.** A privacy mask, tape, paint, a
  cap of snow, an opaque leaf or a slice of a stream lost in transit
  replaces what was there. The transform knows where, to the pixel.
  `OnScreenOverlay` already drops boxes this way.
* **Physically, with calibration, when the object is washed out or
  drowned.** Fog, veils, glare, darkness and sensor noise reduce contrast
  or signal by amounts the transforms already compute. There are
  established thresholds for when a contrast or a signal is too small to
  see; they need checking against people looking at this library's
  frames.
* **Approximately, when the object is smeared.** Defocus, a camera
  swinging on its mount and water on the glass blur by amounts the
  transforms know. Whether a blurred object can still be recognised has
  a well-tried empirical model, which is coarser.
* **Only by measurement, when the damage has no closed form.** Codec
  quantisation, demosaicking, analog links, tone mapping halos and noise
  reduction smears are best judged by comparing the frame with the
  original. That is the least reliable tier, and in practice the one
  least likely to make an object illegible.

## What the library does today

* `OnScreenOverlay` drops a box when privacy masks cover at least
  `hidden_box_coverage` of its area (`camera/overlay.py`). Masks and
  keypoints are left as they are.
* `RollingShutter` and `CameraVibration` move pixels, and move masks,
  boxes and keypoints with them.
* Every other transform is image-only.

Albumentations' own filtering (`BboxParams(min_visibility=...,
min_area=...)`, keypoints' `remove_invisible`) is geometric: it counts how
much of a box a crop or a rotation left in the frame. Nothing in it, or in
the robustness benchmarks built on it, asks whether what is left can be
seen. ImageNet-C and its detection counterparts (COCO-C, Cityscapes-C)
corrupt the images and keep every label at every severity, so a detector
is scored, at the highest levels, on objects no person could find.

## What "unreadable" can be made to mean

**The test is the annotator's.** A label is honest if a careful person
looking at the augmented frame alone would still draw it. That ties the
filter to what can be checked by hand, and it doesn't depend on the
network being trained, which must not be the judge: a filter that keeps
only what the model already sees teaches it nothing hard.

A transform can take an object away in four ways, and each has a
measure and an established threshold:

| How it goes | Measure | Threshold from the literature |
| --- | --- | --- |
| **Hidden**: pixels replaced | share of the object's area covered | occlusion conventions: CityPersons counts a person as reasonably visible from 65% of their area; KITTI grades occlusion in levels |
| **Washed out**: contrast lost to a veil | contrast retained, `c` | Koschmieder's 2% of contrast (5% for the WMO's optical range) as the limit of sight in fog |
| **Drowned**: too little signal for the noise | contrast-to-noise ratio | Rose's criterion: a signal about five times the noise to be seen at all |
| **Smeared**: too little resolution left | resolvable cycles across the object's smallest dimension | Johnson's criteria: about 1 cycle to detect, 4 to recognise, 6.4 to identify, at 50% probability |

The last two combine: blur lowers the contrast of fine detail, and noise
sets how much contrast is needed, so the finest detail still visible is
where the blurred, washed-out contrast falls to the noise threshold. The
number of such cycles across the object, against the task's requirement,
gives a probability that the task can be done, by the target transfer
probability function used with Johnson's criteria:

```
P(N) = (N / N50)^E / (1 + (N / N50)^E),   E = 2.7 + 0.7 · N / N50
```

These numbers were measured on people looking at military targets in
visible and thermal imagery. They are a defensible starting point and
not a calibration for this library's frames; see Validation.

**Relative to the original.** The filter should judge what the
augmentation took away, not the original's difficulty: a small, dim
object that an annotator labelled in the clean frame was legible to them.
The requirement is the smaller of the task's and the clean frame's own,
so augmentation can only remove labels, never question the dataset's.

## What each transform knows

Most transforms already compute, as part of drawing, the map a filter
needs. The table is the work to be done, transform by transform.

| Transform | How it takes objects away | What it already computes | Tier |
| --- | --- | --- | --- |
| `OnScreenOverlay` | privacy masks replace; text covers | the masks, the glyphs | exact (done for boxes) |
| `LensTamper` | paint, tape, a cover; little light, no image | coverage, opacity | exact |
| `SnowOnHousing` | an opaque cap and lumps | coverage | exact |
| `FoliageIntrusion` | leaves in front of the lens | leaf alpha; its light by infrared | exact for opaque leaves, contrast for lit ones |
| `FrostOnGlass` | replaces what it covers with a wide blur | coverage | exact |
| `TransmissionFaults` | lost and garbage slices | the damaged slices; a decode of the undamaged clip gives the map to the pixel | exact |
| `HousingOccluders` | silhouettes by day, glowing by infrared | coverage | exact by day, contrast at night |
| `RandomFog`, `GroundFog`, `NightFog` | contrast lost to the airlight | transmission per pixel | physical |
| `RandomRain`, `RandomSnow` | veil, streak and flake coverage, spray, snow settled on things | veil transmission, coverage, spray density, settled share | physical; settled snow on a vehicle is hidden |
| `LocalPlumes` | steam in front of things | plume opacity | physical |
| `DirtSoil`, condensation in `WaterDroplets` | scattered light added | coverage, the veil | physical |
| `IrDomeGlare`, `SunFlare`, `SourceGlare`, `SpiderWeb`, `NightInsects` | light added over the scene | the added light | physical: contrast falls under the added light; clipping measured |
| `NightIrMode`, `NightColorMode` | darkness beyond the light | irradiance per pixel, the sensor's gain | physical, with `SensorNoise` |
| `SensorNoise` | noise | electrons and read noise per pixel | physical |
| `ExposureTransient`, `WdrToneMapping` | clipping, halos | gain | clipping measured; halos measured |
| `FocusShift` | defocus | the disc's diameter per pixel | approximate (blur) |
| `CameraVibration`, `HeatShimmer` | smear over the exposure; turbulence | the path; the tilt and blur | approximate (blur) |
| drops in `WaterDroplets` | each drop a lens: refracted, blurred | coverage, each drop's disc | approximate |
| `VideoCodecArtifacts`, `DemosaicArtifacts`, `AnalogVideoArtifacts`, `TemporalDenoiseArtifacts` | quantisation, false colour, low-pass, smear | nothing usable | measured |
| `RollingShutter`, `CastShadows`, `AcFlicker`, `IrCutFilterFault` | shear, shade, bands, a colour cast | — | none needed: they move or tint, and leave objects legible |

## A method

**1. Each transform reports what it took, in four maps.** When the filter
is on, every transform that ran adds to a per-frame record of four
planes, at a quarter of the frame's size:

* `hidden`: the share of each pixel replaced, from 0 to 1;
* `contrast`: the share of the scene's local contrast that survives,
  multiplied in, transform by transform. A veil of transmission `t`
  over airlight `A` on a scene of local level `s` keeps
  `t·s / (t·s + (1 − t)·A)`; light added on top keeps
  `s / (s + added)`;
* `blur`: the spread of a point, in pixels, added in quadrature;
* `noise`: the noise, in the output's units, added in quadrature and
  scaled by later gains.

The rules are approximations: a veil before a gain, or a clip, is not
exactly a product. Clipping is the one that matters, and it is read from
the finished frame rather than predicted: where every channel is at full
scale or at zero, no contrast survives.

**2. Each object gets a score.** Over the object's mask, or its box where
there is no mask, the filter takes:

* the share hidden;
* over the part not hidden, the cycles resolvable across the object's
  smallest dimension, given its blur, contrast and noise, against the
  same in the clean frame. From that comes a probability that the task
  can be done.

A box is a coarse stand-in for an object: a car seen at an angle fills
half of its box. Where the provider has read the scene, the vehicle
reader's masks and the labels give the object's own pixels, and the
filter should use them.

**3. A policy turns scores into actions.**

| Score | Action, by default |
| --- | --- |
| hidden share ≥ `gone_at` (0.9) | background: the pixels aren't there |
| task probability ≥ `keep_at` (0.5) | keep |
| anything else | ignore |

What "background" and "ignore" mean depends on the kind of label:

| Label | Background | Ignore |
| --- | --- | --- |
| Box | dropped | moved to an ignore list, or kept with an ignore flag |
| Semantic mask | the object's pixels set to the background class | its pixels set to `ignore_index` (255 by convention) |
| Instance mask | dropped | kept, flagged as ignored, as COCO's `iscrowd` regions are |
| Keypoint | removed | marked labelled but not visible (COCO's `v = 1`) |

**4. What has no map is measured.** For the codec, the demosaicker, the
analog link, tone mapping and noise reduction, the filter compares the
finished frame with the original inside each object, on structure at the
object's own scale: the correlation of their gradient magnitudes after a
band-pass filter at about an eighth of the object's size. That ignores
changes of brightness and colour, which leave an object legible, and
falls with blocks, smears and false detail. It becomes a fifth plane
alongside the four, and it is the least trustworthy.

## How reliable this would be

| Tier | Decisions it makes | Reliability |
| --- | --- | --- |
| Exact | background for replaced pixels | as good as the transform's own geometry: no judgement involved |
| Physical | ignore for washed-out and dark objects | good once the thresholds are calibrated; the physics of each map is the transform's own |
| Approximate | ignore for smeared objects | fair: Johnson's criteria spread by ±20 to 25% around their 50% points |
| Measured | ignore where only the comparison speaks | weakest; reserved for effects that rarely take an object away |

Because every judgement short of "replaced" can only lead to "ignore",
the method is safe where it is least reliable: a poor threshold costs
training signal, not truth. The one judgement that writes a label,
"background", is the exact one.

## What can be tuned

All of it, and none of it runs unless asked for:

* **The task**: `"detect"`, `"recognise"` or `"identify"`, or a number of
  cycles. A plate reader needs identification; a person counter,
  detection.
* **`keep_at` and `gone_at`**: the probability at or above which a label
  is kept, and the hidden share at or above which it is background.
* **What to do in between**: `"ignore"`, `"drop"`, `"keep"`, or
  `"blank"` (below).
* **Per class**: thresholds, and the critical dimension (a person's width,
  a car's height).
* **Which effects count**: a caller can exclude, say, the codec from the
  judgement.
* **The measured fallback**: on, off, or on with its own threshold.

## Plumbing in Albumentations

**Carrying the maps.** Two ways:

* **A side record**, kept for each call the way `diagnostics.recording()`
  already collects draws, and read by a last transform
  (`LabelLegibility`). It needs no Albumentations internals. The record's
  maps are in the frame's geometry when they are written, so they stay
  right only if nothing moves pixels after them. This library's two
  geometric transforms can warp the record as they warp the labels, but
  a crop or a flip of Albumentations' own after them would not. The
  filter can check for that: each transform records a fingerprint of the
  frame it returned, and the filter refuses, or warns, if the frame it
  receives is not the last one recorded.
* **An extra mask target** (`additional_targets={"legibility": "mask"}`)
  carrying the planes. Every geometric transform, Albumentations' own
  included, would then warp them correctly. But an image-only transform
  that writes into one particular mask needs a hook into
  `BasicTransform._key2func`, which is private and could change between
  releases.

The side record first, with the ordering rule documented and checked;
the mask target later, if callers need geometric augmentation after the
camera's effects.

**Ignoring needs the training framework's help.** Semantic masks have
`ignore_index` everywhere. Detection frameworks differ: MMDetection takes
ignore regions; as far as I know Ultralytics' YOLO format and
torchvision's detection models have no such field, and a dropped box
there becomes background. For those, the `"blank"` action makes the
background true: it fills the object's pixels with a local blur of its
surroundings and drops the box, so the network is told "nothing here"
about a frame where that is so. It changes the picture, which is why it
is not the default.

## Cost

The maps are planes most transforms already compute, written out at a
quarter of the frame's size: a few milliseconds a frame at 512 px. The
per-object score is a handful of reductions over each object. The
measured fallback, a band-pass and a correlation per object, is a few
milliseconds more. None of it touches the GPU, and with the filter off
nothing is recorded.

## Validation

* **Per transform**: the four planes move the right way with severity
  (more fog, less contrast), and a frame with no effect scores every
  object as kept.
* **Against people**: a few hundred objects across the sample scenes,
  effects and severity levels, each marked by two people as clearly
  visible, doubtful or invisible. Fit `keep_at` and the contrast and noise
  thresholds to that, and report how often a kept label was marked
  invisible. That rate is the one that matters, and it should be near
  zero.
* **In training**: a detector trained with the filter off, with
  `"ignore"`, and with `"drop"`, evaluated on real degraded footage and on
  clean frames, counting false positives in fog, darkness and damaged
  frames separately.
* **In evaluation**: the same scores let a severity sweep count only
  legible objects, or report recall by legibility, so a model is not
  marked down at severity 5 for missing what nobody could see.

## Risks and open questions

* **People are not networks.** The thresholds come from human
  observers. A network may find objects a person cannot, especially in
  noise. The annotator test is still the right standard for a label, but
  ignoring more than needed costs exactly the hard examples augmentation
  is for. The defaults should lean towards keeping.
* **The composition is approximate**: products and quadrature sums are
  exact for linear steps, and the camera chain is not linear.
  Calibration absorbs some of this; clipping is measured to avoid the
  worst of it.
* **The original's labels aren't checked.** The filter judges only what
  augmentation took away.
* **Ignored pixels in semantic masks come out speckled** where a map
  crosses a threshold. They should be smoothed and closed so that ignore
  regions follow objects.
* **Should the filter feed back into the pipeline's odds?** A frame where
  most labels are ignored teaches little. The pipeline could draw such
  frames less often, but that changes the distribution the pipeline
  documents, so it should be a separate option.

## Recommendation

Build it, opt-in, in three steps, each useful alone:

1. **The exact tier.** The side record, the `hidden` plane from the
   transforms that replace pixels, the policy and its actions for every
   label type, and `OnScreenOverlay`'s rule folded in with its parameter
   kept. This alone ends the plainly false labels: boxes over tape, under
   a snow cap, in a lost slice.
2. **The physical tier.** `contrast` and `noise` from fog, veils, glare,
   the night modes and the sensor, with Koschmieder's and Rose's
   thresholds as defaults. This covers the commonest case of a
   misleading label: a day-labelled object left in the dark by a night
   made of the frame, or behind fog it cannot be seen through.
3. **Blur and the measured fallback**, with Johnson's criteria. Then
   calibrate all three against people, and run the training comparison.

## References

* Johnson, J. (1958). Analysis of image forming systems. *Image
  Intensifier Symposium*, Fort Belvoir. The cycles-across-target
  criteria for detection, recognition and identification.
* Vollmerhausen, R. H., Jacobs, E. & Driggers, R. G. (2004). New metric
  for predicting target acquisition performance. *Optical Engineering*
  43(11). The metric that succeeded Johnson's criteria; the probability
  function above is the one used with them in the U.S. Army's
  night-vision models.
* Rose, A. (1948). The sensitivity performance of the human eye on an
  absolute scale. *Journal of the Optical Society of America* 38(2).
* Koschmieder, H. (1924). Theorie der horizontalen Sichtweite. *Beiträge
  zur Physik der freien Atmosphäre* 12.
* Zhang, S., Benenson, R. & Schiele, B. (2017). CityPersons: a diverse
  dataset for pedestrian detection. *CVPR*.
* Geiger, A., Lenz, P. & Urtasun, R. (2012). Are we ready for autonomous
  driving? The KITTI vision benchmark suite. *CVPR*.
* Hendrycks, D. & Dietterich, T. (2019). Benchmarking neural network
  robustness to common corruptions and perturbations. *ICLR*.
* Michaelis, C. et al. (2019). Benchmarking robustness in object
  detection: autonomous driving when winter is coming. arXiv:1907.07484.
* Lin, T.-Y. et al. (2014). Microsoft COCO: common objects in context.
  *ECCV*. The `iscrowd` regions and keypoint visibility flags.

## What was built

23 November 2026. `LabelLegibility` (`src/surmentations/legibility.py`),
the exact tier and the measured tier, with these differences from the
method above:

- **Two planes, not four.** Transforms that replace pixels report
  `hidden` (`HousingOccluders`, `LensTamper`, `SnowOnHousing`,
  `FoliageIntrusion`, `FrostOnGlass`, `OnScreenOverlay`'s privacy masks);
  `FocusShift` and `HeatShimmer` report `blur`. Contrast and noise are not
  reported but measured: a veil, the dark and the grain all show in the
  measurement, and a reported contrast would be counted twice.
- **The measurement is the main tier, not a fallback.** Inside each object
  the finished frame is compared with the original in log light, so that
  a night made of a day, a change of light that varies slowly, does not
  read as lost detail. At each octave from 2 px to the object's size, the
  edges of both frames are taken with each pixel's own noise removed, and
  an octave is resolved where at least half the clean edge energy stands
  three times clear of the noise, in the clean pattern. The finest octave
  resolved gives the cycles across the object; Johnson's curve the
  probability, relative to the clean frame's. Two simpler measures were
  tried on 300 pipeline frames and dropped. Linear light suppressed a bus
  plainly seen under infrared. Edge magnitudes with the noise left on read
  the grain on a dark truck as edges the clean frame lacks, and suppressed
  the truck.
- **Suppression is the default** (`on_doubt="ignore"`), and a suppressed
  box leaves the labels and comes back in `ignore_regions` (pixel boxes),
  since most detection frameworks have no ignore flag on a box. The YOLOX
  bench carries them as labels of class −1 through its own augmentation,
  and its head leaves the anchors in them out of the objectness loss.
- **`"blank"` paints with the surroundings only**: a blur normalised over
  the pixels outside the object, never over the pixels of an object whose
  label is kept.
- **`TransmissionFaults` is measured, not exact.** A lost slice is
  suppressed, not made background; knowing which pixels a decoder filled
  needs a second decode of the clean clip.

On 300 val2017 frames through `surveillance` (2,243 boxes): 4.9% of boxes
suppressed, none made background (no frame was covered); 18.8% of the boxes
under 16 px across, 1.6% at 16 to 32 px, 0.5% above. By eye, the suppressed
large objects were gone: sofas under a decoder's green slab, an aircraft in
an infrared sky, kites in a black one; and the large ones kept were plain.
It costs about 13 ms a frame of eight objects on one core.

Calibration against people, the training comparison and an exact map for
lost slices are still to do.

