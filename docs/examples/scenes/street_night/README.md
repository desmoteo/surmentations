# street_night

Real night street under sodium lamps.

![original](https://desmoteo.github.io/surmentations/examples/scenes/street_night/original.webp)

Source: [Great Holm Milton Keynes at night.jpg](https://commons.wikimedia.org/wiki/File:Great_Holm_Milton_Keynes_at_night.jpg) by John Chryslar, CC0. Every image in this folder is derived from it and is available under the same licence.

## How the transforms were told about this scene

| | |
| --- | --- |
| Distance of nearest and farthest point | 3 m to 150 m (estimated by eye) |
| Camera pitch below level | -6 degrees |
| Lighting | night, open to the sky |
| Sky in frame | yes |
| Fog | visibility 0.6 x the scene's depth: about 90 m by eye |
| Rain and snow veils | visibility kept above 1.5 x and 1 x the scene's depth: about 225 m and 150 m |

## What a provider with scene labels and geometry read off it

The photograph and its label map, with the horizon the geometry estimator
found drawn across both. See
[scene understanding](../../../scene_understanding.md) for how each is made
and how far to trust it.

![labels and horizon](https://desmoteo.github.io/surmentations/examples/scenes/street_night/reading.webp)

| | |
| --- | --- |
| Field of view | 56°, assumed: a picture does not tell it |
| Horizon | 0.64 of the frame's height from the top |
| Camera pitch below level | -6° (by eye: -6°) |
| Roll | -1.4° |
| Camera height | not found: fewer than three usable vehicles or people in view |
| Depth | the by-eye range above: there is no height to measure from |
| Labels | foliage 41%, sky 35%, road 16%, structure 6%, pavement 1%, other 0% |

## Without and with it

Every effect that depends on what the provider knows, twice: on the left with
the plain provider and the by-eye values above, on the right with the provider
that read the scene. The parameters and the seed are the same, so what differs
is what the labels and the geometry changed.

`rain` and `wet_ground` have one thing more on the right: the traffic's spray
and its lamps (`spray_range`, `vehicle_lamp_range`), which are off by default
and which only a provider that knows the road and its vehicles can draw. That
provider also has a vehicle reader, which says which end of each vehicle is
seen and where its lamps are; see [the road surface](../../../road_surface.md).
With that reader `night_ir` and `night_color` are lit as a road is at night,
from its street lamps and its traffic's lamps and beams; on the left they are
lit evenly, as a daylight frame is. Under `night_ir` the road signs and number
plates the reader found send the illuminator's light back, and `night_rain` and
`night_snow`, drawn over that frame, show in the lamps' light. The reader looks
at each photograph at the size it was taken at, not at the reduced frame.

| `rain` without | `rain` with |
| --- | --- |
| ![`rain` without](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/rain.webp) | ![`rain` with](https://desmoteo.github.io/surmentations/examples/scenes/street_night/aware/rain.webp) |

| `snow` without | `snow` with |
| --- | --- |
| ![`snow` without](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/snow.webp) | ![`snow` with](https://desmoteo.github.io/surmentations/examples/scenes/street_night/aware/snow.webp) |

| `night_ir` without | `night_ir` with |
| --- | --- |
| ![`night_ir` without](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/night_ir.webp) | ![`night_ir` with](https://desmoteo.github.io/surmentations/examples/scenes/street_night/aware/night_ir.webp) |

| `night_rain` without | `night_rain` with |
| --- | --- |
| ![`night_rain` without](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/night_rain.webp) | ![`night_rain` with](https://desmoteo.github.io/surmentations/examples/scenes/street_night/aware/night_rain.webp) |

| `night_snow` without | `night_snow` with |
| --- | --- |
| ![`night_snow` without](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/night_snow.webp) | ![`night_snow` with](https://desmoteo.github.io/surmentations/examples/scenes/street_night/aware/night_snow.webp) |

| `night_fog` without | `night_fog` with |
| --- | --- |
| ![`night_fog` without](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/night_fog.webp) | ![`night_fog` with](https://desmoteo.github.io/surmentations/examples/scenes/street_night/aware/night_fog.webp) |

| `wet_ground` without | `wet_ground` with |
| --- | --- |
| ![`wet_ground` without](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/wet_ground.webp) | ![`wet_ground` with](https://desmoteo.github.io/surmentations/examples/scenes/street_night/aware/wet_ground.webp) |

| `snow_cover` without | `snow_cover` with |
| --- | --- |
| ![`snow_cover` without](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/snow_cover.webp) | ![`snow_cover` with](https://desmoteo.github.io/surmentations/examples/scenes/street_night/aware/snow_cover.webp) |

| `ground_fog` without | `ground_fog` with |
| --- | --- |
| ![`ground_fog` without](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/ground_fog.webp) | ![`ground_fog` with](https://desmoteo.github.io/surmentations/examples/scenes/street_night/aware/ground_fog.webp) |

| `local_plumes` without | `local_plumes` with |
| --- | --- |
| ![`local_plumes` without](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/local_plumes.webp) | ![`local_plumes` with](https://desmoteo.github.io/surmentations/examples/scenes/street_night/aware/local_plumes.webp) |

## The other effects at severity 3

One representative draw each: the median of seven by how much the frame changed.
These do not read the scene, and are the same with either provider.

| water_droplets | dirt_soil | housing_occluders |
| --- | --- | --- |
| ![water_droplets](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/water_droplets.webp) | ![dirt_soil](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/dirt_soil.webp) | ![housing_occluders](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/housing_occluders.webp) |

| ac_flicker | rolling_shutter | exposure_transient |
| --- | --- | --- |
| ![ac_flicker](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/ac_flicker.webp) | ![rolling_shutter](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/rolling_shutter.webp) | ![exposure_transient](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/exposure_transient.webp) |

| wdr_tone_mapping | denoise_artifacts | video_codec |
| --- | --- | --- |
| ![wdr_tone_mapping](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/wdr_tone_mapping.webp) | ![denoise_artifacts](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/denoise_artifacts.webp) | ![video_codec](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/video_codec.webp) |

| night_droplets | sensor_noise | source_glare |
| --- | --- | --- |
| ![night_droplets](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/night_droplets.webp) | ![sensor_noise](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/sensor_noise.webp) | ![source_glare](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/source_glare.webp) |

| transmission_faults | spider_web | night_insects |
| --- | --- | --- |
| ![transmission_faults](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/transmission_faults.webp) | ![spider_web](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/spider_web.webp) | ![night_insects](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/night_insects.webp) |

| ir_dome_glare | frost_on_glass | snow_on_housing |
| --- | --- | --- |
| ![ir_dome_glare](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/ir_dome_glare.webp) | ![frost_on_glass](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/frost_on_glass.webp) | ![snow_on_housing](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/snow_on_housing.webp) |

| lens_tamper | foliage_intrusion | demosaic_artifacts |
| --- | --- | --- |
| ![lens_tamper](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/lens_tamper.webp) | ![foliage_intrusion](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/foliage_intrusion.webp) | ![demosaic_artifacts](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/demosaic_artifacts.webp) |

| analog_video | focus_shift | camera_vibration |
| --- | --- | --- |
| ![analog_video](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/analog_video.webp) | ![focus_shift](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/focus_shift.webp) | ![camera_vibration](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/camera_vibration.webp) |

| on_screen_overlay | night_foliage |  |
| --- | --- | --- |
| ![on_screen_overlay](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/on_screen_overlay.webp) | ![night_foliage](https://desmoteo.github.io/surmentations/examples/scenes/street_night/effects/night_foliage.webp) |  |

Not rendered: `sun_flare`, `cast_shadows` — it needs the sun, and the scene is at night.

Not rendered: `fog` — it is the daylight model, an even veil lit by the sky. `night_fog` is fog lit by the lamps.

Not rendered: `ir_cut_filter_fault` — it is a fault of the colour picture by day, and the scene is at night.

Not rendered: `heat_shimmer` — it is turbulence over sunlit ground, which shows over a kilometre or so by day: this scene is too short, under a roof or at night.

Not rendered: `night_color` — it makes a night of a daylit frame, and this scene is a night already.

## `surveillance_lite` pipeline

4 consecutive draws. Each is a different combination of effects.
This pipeline takes no depth provider, so there is nothing to compare.

| draw 0 | draw 1 |
| --- | --- |
| ![draw 0](https://desmoteo.github.io/surmentations/examples/scenes/street_night/pipelines/surveillance_lite_0.webp) | ![draw 1](https://desmoteo.github.io/surmentations/examples/scenes/street_night/pipelines/surveillance_lite_1.webp) |

| draw 2 | draw 3 |
| --- | --- |
| ![draw 2](https://desmoteo.github.io/surmentations/examples/scenes/street_night/pipelines/surveillance_lite_2.webp) | ![draw 3](https://desmoteo.github.io/surmentations/examples/scenes/street_night/pipelines/surveillance_lite_3.webp) |
