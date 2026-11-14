# crosswalk_day

Near-vertical view of a crossing; people, cyclists, vehicles; no sky.

![original](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/original.webp)

Source: [Crosswalk of Market at Third, San Francisco.jpg](https://commons.wikimedia.org/wiki/File:Crosswalk_of_Market_at_Third,_San_Francisco.jpg) by Dllu, CC BY-SA 4.0. Every image in this folder is derived from it and is available under the same licence.

## How the transforms were told about this scene

| | |
| --- | --- |
| Distance of nearest and farthest point | 12 m to 30 m (estimated by eye) |
| Camera pitch below level | 70 degrees |
| Lighting | day, open to the sky |
| Sky in frame | no |
| Fog | visibility 0.6 x the scene's depth: about 18 m by eye |
| Rain and snow veils | visibility kept above 1.5 x and 1 x the scene's depth: about 45 m and 30 m |

## What a provider with scene labels and geometry read off it

The photograph and its label map, with the horizon the geometry estimator
found drawn across both. See
[scene understanding](../../../scene_understanding.md) for how each is made
and how far to trust it.

![labels and horizon](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/reading.webp)

| | |
| --- | --- |
| Field of view | 76°, assumed: a picture does not tell it |
| Horizon | -0.01 of the frame's height from the top, or above: there is no sky in the frame to mark it |
| Camera pitch below level | at least +27°, so the by-eye 70° stands |
| Roll | -1.8° |
| Camera height | not found: fewer than three usable vehicles or people in view |
| Depth | the by-eye range above: there is no height to measure from |
| Labels | structure 52%, road 38%, foliage 6%, pole 1%, person 1%, vehicle 1% |

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

| `fog` without | `fog` with |
| --- | --- |
| ![`fog` without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/fog.webp) | ![`fog` with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/aware/fog.webp) |

| `rain` without | `rain` with |
| --- | --- |
| ![`rain` without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/rain.webp) | ![`rain` with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/aware/rain.webp) |

| `snow` without | `snow` with |
| --- | --- |
| ![`snow` without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/snow.webp) | ![`snow` with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/aware/snow.webp) |

| `cast_shadows` without | `cast_shadows` with |
| --- | --- |
| ![`cast_shadows` without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/cast_shadows.webp) | ![`cast_shadows` with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/aware/cast_shadows.webp) |

| `night_ir` without | `night_ir` with |
| --- | --- |
| ![`night_ir` without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/night_ir.webp) | ![`night_ir` with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/aware/night_ir.webp) |

| `night_rain` without | `night_rain` with |
| --- | --- |
| ![`night_rain` without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/night_rain.webp) | ![`night_rain` with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/aware/night_rain.webp) |

| `night_snow` without | `night_snow` with |
| --- | --- |
| ![`night_snow` without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/night_snow.webp) | ![`night_snow` with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/aware/night_snow.webp) |

| `wet_ground` without | `wet_ground` with |
| --- | --- |
| ![`wet_ground` without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/wet_ground.webp) | ![`wet_ground` with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/aware/wet_ground.webp) |

| `snow_cover` without | `snow_cover` with |
| --- | --- |
| ![`snow_cover` without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/snow_cover.webp) | ![`snow_cover` with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/aware/snow_cover.webp) |

| `ground_fog` without | `ground_fog` with |
| --- | --- |
| ![`ground_fog` without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/ground_fog.webp) | ![`ground_fog` with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/aware/ground_fog.webp) |

| `local_plumes` without | `local_plumes` with |
| --- | --- |
| ![`local_plumes` without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/local_plumes.webp) | ![`local_plumes` with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/aware/local_plumes.webp) |

| `night_color` without | `night_color` with |
| --- | --- |
| ![`night_color` without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/night_color.webp) | ![`night_color` with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/aware/night_color.webp) |

## The other effects at severity 3

One representative draw each: the median of seven by how much the frame changed.
These do not read the scene, and are the same with either provider.

| water_droplets | dirt_soil | sun_flare |
| --- | --- | --- |
| ![water_droplets](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/water_droplets.webp) | ![dirt_soil](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/dirt_soil.webp) | ![sun_flare](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/sun_flare.webp) |

| housing_occluders | ac_flicker | rolling_shutter |
| --- | --- | --- |
| ![housing_occluders](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/housing_occluders.webp) | ![ac_flicker](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/ac_flicker.webp) | ![rolling_shutter](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/rolling_shutter.webp) |

| exposure_transient | wdr_tone_mapping | denoise_artifacts |
| --- | --- | --- |
| ![exposure_transient](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/exposure_transient.webp) | ![wdr_tone_mapping](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/wdr_tone_mapping.webp) | ![denoise_artifacts](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/denoise_artifacts.webp) |

| video_codec | night_droplets | sensor_noise |
| --- | --- | --- |
| ![video_codec](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/video_codec.webp) | ![night_droplets](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/night_droplets.webp) | ![sensor_noise](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/sensor_noise.webp) |

| transmission_faults | spider_web | night_insects |
| --- | --- | --- |
| ![transmission_faults](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/transmission_faults.webp) | ![spider_web](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/spider_web.webp) | ![night_insects](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/night_insects.webp) |

| ir_dome_glare | frost_on_glass | snow_on_housing |
| --- | --- | --- |
| ![ir_dome_glare](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/ir_dome_glare.webp) | ![frost_on_glass](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/frost_on_glass.webp) | ![snow_on_housing](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/snow_on_housing.webp) |

| lens_tamper | foliage_intrusion | ir_cut_filter_fault |
| --- | --- | --- |
| ![lens_tamper](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/lens_tamper.webp) | ![foliage_intrusion](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/foliage_intrusion.webp) | ![ir_cut_filter_fault](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/ir_cut_filter_fault.webp) |

| demosaic_artifacts | analog_video | focus_shift |
| --- | --- | --- |
| ![demosaic_artifacts](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/demosaic_artifacts.webp) | ![analog_video](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/analog_video.webp) | ![focus_shift](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/focus_shift.webp) |

| camera_vibration | on_screen_overlay | night_foliage |
| --- | --- | --- |
| ![camera_vibration](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/camera_vibration.webp) | ![on_screen_overlay](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/on_screen_overlay.webp) | ![night_foliage](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/effects/night_foliage.webp) |

Not rendered: `night_fog`, `source_glare` — it acts on artificial lights at night, and this is a daylit scene.

Not rendered: `heat_shimmer` — it is turbulence over sunlit ground, which shows over a kilometre or so by day: this scene is too short, under a roof or at night.

## `surveillance` pipeline

4 consecutive draws, each a different combination of effects:
on the left with the plain provider, on the right the same draw with the
provider that read the scene.

| draw 0 without | draw 0 with |
| --- | --- |
| ![draw 0 without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/pipelines/surveillance_0.webp) | ![draw 0 with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/pipelines_aware/surveillance_0.webp) |

| draw 1 without | draw 1 with |
| --- | --- |
| ![draw 1 without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/pipelines/surveillance_1.webp) | ![draw 1 with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/pipelines_aware/surveillance_1.webp) |

| draw 2 without | draw 2 with |
| --- | --- |
| ![draw 2 without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/pipelines/surveillance_2.webp) | ![draw 2 with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/pipelines_aware/surveillance_2.webp) |

| draw 3 without | draw 3 with |
| --- | --- |
| ![draw 3 without](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/pipelines/surveillance_3.webp) | ![draw 3 with](https://desmoteo.github.io/surmentations/examples/scenes/crosswalk_day/pipelines_aware/surveillance_3.webp) |
