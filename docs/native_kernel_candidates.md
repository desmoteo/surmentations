# Which methods could take a C kernel written for the compiler to vectorise

19 November 2026. An assessment, and below it what was then built from it.

[`native_kernels.md`](native_kernels.md) answered this question for the
library as it stood before the road, the night lighting and the new
pipelines were written. This page asks it again of what has been added
since, and of what the pipelines now spend their time on. It keeps that
document's method and its findings: plain C compiled per instruction-set
tier, the compiler doing the vectorising, numpy kept as the definition and
the fallback, and no hand-written intrinsics.

## What was built

All six candidates, as twelve kernels, and the three things the assessment
said would help where a kernel could not. Measured on one pinned core,
OpenCV on one thread, AVX2 tier; "before" is the code as it stood when the
assessment was written.

| # | Candidate | Kernels | Before | After | Agreement with numpy |
| --- | --- | --- | ---: | ---: | --- |
| 1 | The lamps' light on the air | `street_air` | 154 ms | 5.9 ms (11.6 on the SSE2 tier) | to the bit |
| 2 | Spray puffs | `spray_puffs` | 44 ms, the motorway's spray; 13 of them the puffs | 18.5 ms; 1.0 the puffs | within 2e-6 of a density that is drawn from 0.004 up; the test allows 1e-5 of the largest |
| 3 | The night's compositing | `nir_reflectance`, `night_ir_light`, `night_ir_expose`, `night_colour_light`, `night_colour_expose` | 17.5 ms infrared, 30.7 ms colour, an `apply` at 800 x 480 | 11 to 14 ms, 13.8 ms | the exposure, lamps and grain to the bit; the light on the surfaces to rounding, fast mode only |
| 4 | Surface normals | `normals` | 12.3 ms at 640 px, 2.7 at 384 | 1.0 ms, 0.26 | to the bit |
| 5 | Fog at night | `fog_lost`, `fog_veil` | 30.7 ms, an `apply` at 800 x 480 | 9.0 ms | to the bit |
| 6 | Lying snow's arithmetic | `snow_settled`, `ground_tile` | 5.2 ms, and 4.6 ms for each field of ground noise | 0.5 ms, 0.9 ms | to the bit, fast mode's arithmetic |

What the estimates got wrong:

- **Candidate 1 was underestimated in the right direction and for the
  wrong reason.** The estimate was 3 to 6 ms and it is 5.9, but it takes
  nine points and not four or five: five by Simpson's rule were 2% off a
  fine sum on average, and at the worst line off by 40% of the frame's
  mean; nine are 0.4% and under 10%. The square roots that replace the
  sine, cosine and arctangent are what pays for the extra points.
- **Candidate 2 is 18 ms and not 3 to 5.** The puffs themselves are 1 ms
  now. The rest of the spray is three OpenCV blurs, a labelling of the
  vehicle mask and the placing of each wake, which were never the kernel's.
- **Candidate 3 saves less on the infrared night than on the colour one.**
  Its random draw, a float64 normal for every pixel, is 5 ms of what is
  left and decides the picture a seed gives.
- **Candidate 6 was half the size it was taken for**, 5 ms and not 8 to
  10; the ground's noise, which the assessment listed as not a candidate
  because it is a lookup, turned out to be a dozen numpy passes round one
  OpenCV lookup, and those passes are a kernel like any other.

And what was not a kernel:

| Suggested | Done | Before | After |
| --- | --- | ---: | ---: |
| Score the road model's trial pairs as one matrix | `fit_road_model` | 17 ms | 2.6 ms |
| One sort of a labelling's pixels, not a pass over the frame for each label | `road_cues`, `painted_lines` | 21 ms, 9 ms | 6 ms, 4 ms |
| `model_frame` on the rows below the horizon only | (numpy, not the one C pass suggested) | 17 ms | 7 ms |
| Call `wake_lies` once | `RandomRain` passes it to both users | 2 calls | 1 |
| The picture's thin bright things read once for the cues and the paint | `_thin_things` | 2 or 3 readings | 1 |

Reading a road went from 72 ms to 26 on the motorway and from 54 to 21 on
the city road, and gives the same road, lane for lane and bit for bit.

Whole frames, with a provider that has labels, geometry and the vehicle
reader, the least of three runs each, after the round that followed
([the ninth](native_kernels.md#a-ninth-round-lamps-tracks-and-the-roads-frame)):

| Frame | Estimated here | Before | After these six | After the ninth round |
| --- | ---: | ---: | ---: | ---: |
| Infrared night, city road | about 60 ms | 153 ms | 66 ms | 54 ms |
| Night in colour with fog, the same | about 95 ms | 183 ms | 70 ms | 63 ms |
| Wet road with spray and lamps, motorway | about 300 ms | 263 ms | 179 ms | 152 ms |

The third was estimated without the road's reading, which was then
reworked too. The others are in
[`pipelines.md`](pipelines.md#with-a-provider-that-reads-the-scene).

What this page left open has since been done: `wheel_tracks`,
`model_frame`, the lamps' pools and beams and the rain's lamps and spray
are kernels. Still open: `vegetation_weight` (a formula a pixel, 3 ms of
lying snow), and the labelling of the vehicle mask, which the road's and
the night's code still do many times a frame.

## The assessment as it was written

### The short answer

- **Six methods are worth a kernel.** Four are in code added this month
  and have never been through this treatment; two are older and were left
  at the time.
- **The gain is in frames made with a provider that reads the scene.** An
  infrared night with street lamps would go from about 170 ms to about 60,
  and spray on a busy road from about 55 ms to under 10. The pipeline on a
  depth-only provider would gain a few per cent: it has been through seven
  rounds of this already and what is left of it is inside OpenCV.
- **The largest single item is one I made slower this week.** The sum of
  the lamps' light along each line of sight (`street_air`) took 27 ms for
  25 lamps before the forward-scattering term was added and takes 154 ms
  now. On the city road it is 98 of an infrared night's 169 ms. It is the
  first thing to fix, and half of the fix is not C.
- **Most of the rest cannot use one.** Reading the road, finding plates
  and signs, and telling vehicles apart are connected components,
  thresholds and small OpenCV calls made once per object. Their cost is
  the interpreter and OpenCV, and there is no loop of arithmetic in them
  for a compiler to vectorise.

## What makes a method a candidate here

From what `native_kernels.md` measured, a loop is vectorised by GCC and
Clang with this library's flags when all of these hold:

| Condition | Why | What fails it |
| --- | --- | --- |
| The time is the method's own | A kernel cannot speed up an OpenCV filter that is already SIMD | Blurs, morphology, connected components, `bilateralFilter` |
| One call covers a whole frame or a whole field of objects | `ctypes` costs microseconds a call, and the win is the interpreter removed | A kernel called once per puff or per box |
| The loop body is arithmetic on neighbours at fixed offsets | That is what a vector lane is | Gathers through an index (`bincount`, a table lookup per pixel, a warp) |
| No branch on the data | A mispredicted branch is slower than numpy, and blocks the vectoriser | `if` on a pixel's value; written as `min`/`max` it is fine |
| No call to `exp`, `pow`, `sin`, `atan` per pixel | Without `-ffast-math`, which the build does not use so that every tier rounds as numpy does, a maths call is a scalar call | Needs a table, a recurrence, an algebraic identity, or a polynomial in fast mode only |
| float32 throughout the pixel loop | Half the lanes otherwise | A `double` per pixel, as in the mirror kernel before it was fixed |

The existing kernels already work round the fifth: `exp` is tabulated in
the sprite and denoise kernels, and the iris kernel takes an angle's sine
and cosine from a recurrence.

## How the figures were taken

One core, OpenCV on one thread, pinned. The city-road sample (800 × 480,
12 vehicles) and the motorway sample (640 × 640, 46 vehicles) with a
provider that has labels, geometry and the vehicle reader; the depth-only
pipeline at 512 px; and `street_air` on its own, on its 256 × 154 working
grid with 25 lamps, before and after this week's change. Times marked **[timed]** are the least
or the median of several calls. Times marked **[profiled]** are from
`cProfile`, which inflates code that makes many small Python calls by two
to three times and leaves numpy-bound code about right. Expected times are
my estimates from the pixel counts and the speed of the existing kernels
(0.2 to 1 ns per pixel-operation on the AVX2 tier), not measurements.

## Candidates, in the order I would take them

| # | Method | Where | Now | Why it is slow | After, estimated | Agreement |
| --- | --- | --- | ---: | --- | ---: | --- |
| 1 | `street_air` | `utils/night_light.py` | 154 ms for 25 lamps **[timed]**, 27 ms before this week; 98 ms on the city road **[profiled]** | Per lamp, eight passes of sine, cosine, clip and a 1.5th power over the grid: 25 lamps × 8 points × 39 000 pixels | 3 to 6 ms | fast mode: a rounding, if the angles are reworked as below |
| 2 | Spray puffs: `puff` and `cloud` in `vehicle_spray` | `utils/road.py` | 55 ms on the motorway **[profiled]**, about 20 real | One window of the frame per puff, 400 to 1 100 puffs, each a dozen numpy calls | 3 to 5 ms | exact, with `exp` tabulated as in the sprite kernel |
| 3 | The night's compositing: `NightIrMode.apply`, `NightColorMode.apply` | `camera/night_ir.py`, `night_color.py` | 19 ms a frame **[profiled]** | A dozen passes over the frame: reflectance, falloff, lamps, exposure, clip | 3 to 5 ms | exact; the noise draw stays in numpy |
| 4 | Surface normals: `_normals_and_jump`, `_upward_facing` | `depth/normals.py` | 5 ms at 384 px, 10 ms at full size **[profiled]** | Six numpy passes of differences, a cross product and a normalisation | about 1 ms | exact is possible; the order of the float32 sums has to be kept |
| 5 | Fog at night: `NightFog.apply` | `atmosphere/night_fog.py` | 10 to 15 ms of its 46 **[profiled]** | Eight passes over a three-channel frame round its two OpenCV blurs | 3 to 4 ms | exact, with `exp(-optical depth)` taken once in numpy or tabulated |
| 6 | Settled snow's arithmetic | `RandomSnow.get_params_dependent_on_data` | 8 to 10 ms of its 50 **[profiled]** | Canopy, relief, patches and grain as fifteen frame-sized expressions | 2 to 3 ms | exact |

### 1. The lamps' light on the air

This is the case the whole method was written for, and it wants the
method's first two steps before its third.

- **Algorithm first.** The eight-point sum was added to weigh each stretch
  of a line of sight by which way the light is going there. Nothing was
  timed when it went in, which is the mistake; the closed form it replaced
  cost 27 ms. Four points may be enough, lamps whose light cannot reach the
  frame can be skipped, and the 1.5th power is `x * sqrt(x)`.
- **Then the angles.** As written each point needs a sine and a cosine,
  and each line an arctangent at either end: scalar calls in C. All of
  them can be had from the geometry with square roots alone. The sine and
  cosine of the angle to a point `s` along a line that passes the lamp at
  `h` are `s` and `h` over `sqrt(s² + h²)`; the points in between follow
  by the angle-addition formula from constants. Only the width of the
  interval needs an arctangent, once per pixel and lamp, and a polynomial
  does it to 1e-5.
- **Then C.** What is left is square roots, multiplications and `min`/`max`
  over a grid, lamp by lamp: every condition in the table holds.

### 2. Spray puffs

The same shape as the snowflake and streak kernels, which went from 144 ms
to 17 and from 211 to 11: one call for the whole field in place of one
Python call per particle. Each puff is a Gaussian with two axes, weighted
by three maps at the same pixel (what stands in front, how sure, how far).
Those are lookups at the pixel's own index, not gathers, so the row loop
vectorises. The exponential is of a bounded argument and tabulates.

`vehicle_lamps` and `lamp_heads` draw their lamps the same way, a Gaussian
spot per lamp, and would use the same kernel: a few milliseconds more.

### 3 and 5. Compositing

`wet_composite`, `veil_blend` and `overcast_blend` are the precedent: nine
numpy expressions over the frame became one pass and went from 24 ms to
1.8. The night transforms and the night fog were written after those and
are still chains of numpy passes. Each is a fixed formula per pixel with
clamps, which is the easiest thing here to get vectorised and to hold to
numpy's bytes.

### 4. Normals

A stencil: differences with the neighbour on either side, the smaller one
kept, a cross product, a division by a length. It was left in numpy in the
earlier rounds because reading it on a smaller copy took most of its cost
away. With the reading now at 384 pixels it is 5 ms of every wet road,
every settled snow and every cast shadow, and it is the largest piece of
the depth-only pipeline that is the library's own arithmetic.

## Not candidates, and why

| Method | Time | What it is | What would help instead |
| --- | ---: | --- | --- |
| `read_road`: `fit_road_model`, `road_cues`, `model_frame`, `painted_lines` | about 95 ms on the motorway **[profiled]**, 40 to 60 real | Connected components, a principal axis per piece, 200 trial pairs scored in a Python loop, morphology | Score the pairs as one numpy matrix, as the lane comb now is (48 ms to 2). `model_frame`'s per-pixel offsets could be one C pass, for 5 ms |
| `sheeting_within`, `_outline_of` | 15 ms for 25 boxes **[profiled]** | Otsu thresholds, components and a convex hull on a window per box | Nothing in C. Skip boxes too small to matter |
| `vehicle_lights`, `road_mist`, `_outlines`, `small_level_areas`, `standing_objects` | 3 to 25 ms each | Connected components and their statistics | One labelling of the vehicle mask shared between them: the road's code labels it three or four times a frame |
| `wake_lies`, `spaced_speeds` | 7 ms each | A few dozen vehicles, a Python loop of small vectors | Call `wake_lies` once and pass it on; it is called twice |
| `far_field_texture` | 6.6 ms | A random draw and an OpenCV blur | The draw decides the picture a seed gives, as `SensorNoise`'s does |
| `ground_noise` | 3 to 4 ms a call | A noise tile looked up through each pixel's place on the ground | A gather: C would remove the numpy passes but not vectorise. 2 ms at most |
| `lens_glare`, the fog's lamp haloes, `scatter_psf`, `light_field` | 2 to 20 ms | OpenCV Gaussian blurs | The haloes are blurred over the whole frame and could be blurred on the lit pixels' bounding crop, as `lens_glare` already is |
| `TemporalDenoiseArtifacts`, `WdrToneMapping`, `WaterDroplets` | 10 to 15 ms each | Already a kernel, or OpenCV's `bilateralFilter`, or a disc blur | Listed under "Not done" in `native_kernels.md`; nothing has changed |
| `fill_to_edges`, `sky_to_depth_edge`, `apply_overcast` | 3 to 8 ms each | Already kernels round OpenCV calls | Nothing |

## What it would buy

Estimates, for the frames I measured:

| Frame | Now | With candidates 1 to 6 | Mostly from |
| --- | ---: | ---: | --- |
| Infrared night, city road, full provider | 169 ms **[timed]** | about 60 ms | `street_air` |
| Night in colour with fog, the same | 211 ms **[timed]** | about 95 ms | `street_air`, fog compositing |
| Wet road with spray and lamps, motorway | 330 ms **[timed]** | about 300 ms | spray puffs; the road reading is untouched |
| Depth-only pipeline at 512 px, mean | 44 ms **[timed]** | about 41 ms | normals, settled snow |

Restoring `street_air` to what it cost before this week, with no C at all,
is already about 80 ms of the first row.

## What each kernel has to come with

The rules the existing kernels follow, from `native_kernels.md`:

- The numpy code stays, as the definition and as what runs without a
  compiler or with `SURMENTATIONS_NATIVE=0`.
- A test holds the kernel to numpy: to the byte where the arithmetic is the
  same, to a stated bound where it is not, and then it is fast mode only.
- Every loop meant to vectorise is marked `SM_VECTORISED`, and
  `tests/unit/test_native_build.py` fails the build if the compiler did not
  vectorise it on any tier.
- Every tier gives the same pixels, and the hashes in `native/check` match
  between x86-64 and arm64.
- It is timed on one pinned core, old and new interleaved in one run.

## Limits of this assessment

- One scene per figure, and one machine. The lamp count sets `street_air`'s
  cost and the vehicle count sets the spray's; other scenes will differ in
  proportion.
- The "after" times are estimates. The existing kernels beat or missed
  their first estimates by a factor of two either way.
- Whether a loop is vectorised is the compiler's decision and was wrong
  three times in the first round for reasons nobody predicted. Each
  candidate needs its remark read on every tier before it is counted.
- I did not profile `NightColorMode.apply` separately from
  `NightIrMode.apply`; it is taken to cost the same.
- arm64 was not timed. The Jetson figures in `native_kernels.md` suggest
  the same ranking with a current GCC.
