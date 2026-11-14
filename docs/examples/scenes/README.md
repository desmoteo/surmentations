# Effects on other scenes

The severity gallery one level up shows every effect at five strengths over one
photograph. These are the same effects at one strength over eight scenes chosen
to be unlike it: a crossing from straight above, a covered car park, a real
night street, mountains across a valley.

Each scene's page says how the transforms were told about it: its distances,
the camera's pitch, whether it is night or under a roof, and which effects were
left out as not applying. Severity fixes extinction per metre, which suits a
street and blanks a landscape, so fog is set by visibility as a fraction of the
scene's depth and the veils of rain and snow are capped the same way.

Each page then shows what a provider with
[scene labels and geometry](../../scene_understanding.md) read off the scene
for itself, and every effect that depends on it twice, side by side: without
it and with it, from the same parameters and the same seed.

Regenerate with `python docs/generate_scene_examples.py`: quick on a CUDA GPU,
slow without one. CI renders the pictures and publishes them; they are not kept
in git. The inputs are in [`sample_scenes/`](../../../sample_scenes/), with their authors
and licences in [`SOURCES.md`](../../../sample_scenes/SOURCES.md). The images here
are derived from them and carry the same licences, not the library's.

| Scene | | What it stresses |
| --- | --- | --- |
| [`crosswalk_day`](crosswalk_day/README.md) | [![crosswalk_day](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/original.webp)](crosswalk_day/README.md) | near-vertical view of a crossing; people, cyclists, vehicles; no sky |
| [`parking_garage`](parking_garage/README.md) | [![parking_garage](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/original.webp)](parking_garage/README.md) | covered car park; pillars and a ceiling; short depth range |
| [`street_night`](street_night/README.md) | [![street_night](https://desmoteo.github.io/surmentations/examples/scenes/street_night/original.webp)](street_night/README.md) | real night street under sodium lamps |
| [`underpass_from_above`](underpass_from_above/README.md) | [![underpass_from_above](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/original.webp)](underpass_from_above/README.md) | road passing under a flyover, seen from slightly above; cars; foliage in the foreground |
| [`wooded_hills`](wooded_hills/README.md) | [![wooded_hills](https://desmoteo.github.io/surmentations/examples/scenes/wooded_hills/original.webp)](wooded_hills/README.md) | wooded hills and a village seen from a hilltop; very long depth range |
| [`mountain`](mountain/README.md) | [![mountain](https://desmoteo.github.io/surmentations/examples/scenes/mountain/original.webp)](mountain/README.md) | mountain face with snow seen across a valley; conifers in the foreground |
| [`urban_road_day`](urban_road_day/README.md) | [![urban_road_day](https://desmoteo.github.io/surmentations/examples/scenes/urban_road_day/original.webp)](urban_road_day/README.md) | urban road by day from overpass height; traffic, lamp posts, skyline |
| [`urban_street_wet`](urban_street_wet/README.md) | [![urban_street_wet](https://desmoteo.github.io/surmentations/examples/scenes/urban_street_wet/original.webp)](urban_street_wet/README.md) | city street from a first-floor balcony on a genuinely wet day |

## What these show that is not right

They are unretouched output, and some of it is wrong in ways worth knowing.
Where the provider's labels or geometry change the answer, the right-hand
picture of each pair shows it:

- **Wet ground is level ground that is not green.** Vegetation is left dry;
  bare soil, gravel and roofs seen from above still get a film of water. With
  labels it is also kept off what is named as a vehicle, a person, a tree or a
  lawn, and laid on what is named as paving.
- **A car's reflection is drawn straight below it.** That is right for a
  road running away from the camera and wrong looking across the traffic,
  where the reflection lies to one side. Tall things, such as walls and
  lamp posts, get no reflection of their own.
- **A reflection from outside the frame is a guess.** Where the mirrored point
  is above the picture the road reflects an estimate of the sky, which is
  right under open sky and wrong under the flyover in `underpass_from_above`.
- **Overcast light cannot remove shadows.** Rain and snow grey the sky and
  lower contrast, but the hard shadows of a sunny photograph stay.
- **Night fog and night rain know only the light they can see.** Fog is as
  bright as the surfaces around it suggest, and rain is lit by the lamps in
  the frame. A lamp outside it lights both and shows in neither.
- **How bright a lamp was is a guess.** A lamp is recorded at clipping. The
  strength of its halo in fog, of the streaks round it in rain and of its
  reflection in a wet road all rest on an assumed factor above that.
- **Packet loss has no moving objects.** `transmission_faults` conceals from a
  displaced copy of the same picture, so a stale band shows the scene
  shifted, never a car where it was two seconds ago.
- **Depth is relative and stretched to the range it is given.** The facades in
  `urban_street_wet` are placed further away than they are, so its fog is
  heavier than the stated visibility would make it. Where the estimator finds
  the camera's height, depth is in metres measured from it instead. Fog and
  the veils are set against the scene's depth as each provider has it, and
  the two still differ: the by-eye range spreads a scene evenly between its
  nearest and farthest point, where measured depth puts most of the frame
  near and a little of it very far, so the same fog leaves the foreground
  clearer and takes the skyline. What is in metres outright, an infrared
  lamp's reach or a ground fog's few metres of height, differs more.
- **Wet ground stops at a few hundred metres.** Beyond that, level ground is
  taken to be terrain; a distant wet road loses its sheen with it, unless it
  is labelled as a road.
- **Labels are a network's opinion.** It names the crossing in `crosswalk_day`,
  seen from straight above, a building; nothing is dried or bared on its
  word alone, which is why that scene's pairs hardly differ.
- **The camera's height is a rough estimate**, from the sizes of the vehicles
  and people in view, and every distance scales with it.
- **Flakes close to an infrared lamp are very large.** `night_snow` draws the
  nearest ones as long white bars.
