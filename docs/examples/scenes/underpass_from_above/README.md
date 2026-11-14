# underpass_from_above

Road passing under a flyover, seen from slightly above; cars; foliage in the foreground.

![original](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/original.webp)

Source: [Cars on the Albany Road underpass in HK Central.jpg](https://commons.wikimedia.org/wiki/File:Cars_on_the_Albany_Road_underpass_in_HK_Central.jpg) by Benlisquare, CC BY-SA 4.0. Every image in this folder is derived from it and is available under the same licence.

## How the transforms were told about this scene

| | |
| --- | --- |
| Distance of nearest and farthest point | 8 m to 160 m (estimated by eye) |
| Camera pitch below level | 25 degrees |
| Lighting | day, open to the sky |
| Sky in frame | no |
| Fog | visibility 0.6 x the scene's depth: about 96 m by eye |
| Rain and snow veils | visibility kept above 1.5 x and 1 x the scene's depth: about 240 m and 160 m |

## What a provider with scene labels and geometry read off it

The photograph and its label map, with the horizon the geometry estimator
found drawn across both. See
[scene understanding](../../../scene_understanding.md) for how each is made
and how far to trust it.

![labels and horizon](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/reading.webp)

| | |
| --- | --- |
| Field of view | 61°, assumed: a picture does not tell it |
| Horizon | 0.25 of the frame's height from the top, or above: there is no sky in the frame to mark it |
| Camera pitch below level | at least +11°, so the by-eye 25° stands |
| Roll | -0.1° |
| Camera height | 6 m, from the vehicles and people in view |
| Depth | the by-eye range above: the height needs a pitch to measure with |
| Labels | foliage 43%, road 25%, structure 18%, vehicle 9%, pavement 2%, pole 1% |

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
| ![`fog` without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/fog.webp) | ![`fog` with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/aware/fog.webp) |

| `rain` without | `rain` with |
| --- | --- |
| ![`rain` without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/rain.webp) | ![`rain` with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/aware/rain.webp) |

| `snow` without | `snow` with |
| --- | --- |
| ![`snow` without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/snow.webp) | ![`snow` with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/aware/snow.webp) |

| `cast_shadows` without | `cast_shadows` with |
| --- | --- |
| ![`cast_shadows` without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/cast_shadows.webp) | ![`cast_shadows` with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/aware/cast_shadows.webp) |

| `night_ir` without | `night_ir` with |
| --- | --- |
| ![`night_ir` without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/night_ir.webp) | ![`night_ir` with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/aware/night_ir.webp) |

| `night_rain` without | `night_rain` with |
| --- | --- |
| ![`night_rain` without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/night_rain.webp) | ![`night_rain` with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/aware/night_rain.webp) |

| `night_snow` without | `night_snow` with |
| --- | --- |
| ![`night_snow` without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/night_snow.webp) | ![`night_snow` with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/aware/night_snow.webp) |

| `wet_ground` without | `wet_ground` with |
| --- | --- |
| ![`wet_ground` without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/wet_ground.webp) | ![`wet_ground` with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/aware/wet_ground.webp) |

| `snow_cover` without | `snow_cover` with |
| --- | --- |
| ![`snow_cover` without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/snow_cover.webp) | ![`snow_cover` with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/aware/snow_cover.webp) |

| `ground_fog` without | `ground_fog` with |
| --- | --- |
| ![`ground_fog` without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/ground_fog.webp) | ![`ground_fog` with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/aware/ground_fog.webp) |

| `local_plumes` without | `local_plumes` with |
| --- | --- |
| ![`local_plumes` without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/local_plumes.webp) | ![`local_plumes` with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/aware/local_plumes.webp) |

| `night_color` without | `night_color` with |
| --- | --- |
| ![`night_color` without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/night_color.webp) | ![`night_color` with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/aware/night_color.webp) |

## The other effects at severity 3

One representative draw each: the median of seven by how much the frame changed.
These do not read the scene, and are the same with either provider.

| water_droplets | dirt_soil | sun_flare |
| --- | --- | --- |
| ![water_droplets](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/water_droplets.webp) | ![dirt_soil](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/dirt_soil.webp) | ![sun_flare](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/sun_flare.webp) |

| housing_occluders | ac_flicker | rolling_shutter |
| --- | --- | --- |
| ![housing_occluders](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/housing_occluders.webp) | ![ac_flicker](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/ac_flicker.webp) | ![rolling_shutter](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/rolling_shutter.webp) |

| exposure_transient | wdr_tone_mapping | denoise_artifacts |
| --- | --- | --- |
| ![exposure_transient](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/exposure_transient.webp) | ![wdr_tone_mapping](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/wdr_tone_mapping.webp) | ![denoise_artifacts](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/denoise_artifacts.webp) |

| video_codec | night_droplets | sensor_noise |
| --- | --- | --- |
| ![video_codec](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/video_codec.webp) | ![night_droplets](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/night_droplets.webp) | ![sensor_noise](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/sensor_noise.webp) |

| transmission_faults | spider_web | night_insects |
| --- | --- | --- |
| ![transmission_faults](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/transmission_faults.webp) | ![spider_web](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/spider_web.webp) | ![night_insects](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/night_insects.webp) |

| ir_dome_glare | frost_on_glass | snow_on_housing |
| --- | --- | --- |
| ![ir_dome_glare](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/ir_dome_glare.webp) | ![frost_on_glass](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/frost_on_glass.webp) | ![snow_on_housing](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/snow_on_housing.webp) |

| lens_tamper | foliage_intrusion | ir_cut_filter_fault |
| --- | --- | --- |
| ![lens_tamper](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/lens_tamper.webp) | ![foliage_intrusion](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/foliage_intrusion.webp) | ![ir_cut_filter_fault](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/ir_cut_filter_fault.webp) |

| demosaic_artifacts | analog_video | focus_shift |
| --- | --- | --- |
| ![demosaic_artifacts](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/demosaic_artifacts.webp) | ![analog_video](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/analog_video.webp) | ![focus_shift](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/focus_shift.webp) |

| camera_vibration | on_screen_overlay | night_foliage |
| --- | --- | --- |
| ![camera_vibration](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/camera_vibration.webp) | ![on_screen_overlay](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/on_screen_overlay.webp) | ![night_foliage](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/effects/night_foliage.webp) |

Not rendered: `night_fog`, `source_glare` — it acts on artificial lights at night, and this is a daylit scene.

Not rendered: `heat_shimmer` — it is turbulence over sunlit ground, which shows over a kilometre or so by day: this scene is too short, under a roof or at night.

## `surveillance` pipeline

4 consecutive draws, each a different combination of effects:
on the left with the plain provider, on the right the same draw with the
provider that read the scene.

| draw 0 without | draw 0 with |
| --- | --- |
| ![draw 0 without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/pipelines/surveillance_0.webp) | ![draw 0 with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/pipelines_aware/surveillance_0.webp) |

| draw 1 without | draw 1 with |
| --- | --- |
| ![draw 1 without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/pipelines/surveillance_1.webp) | ![draw 1 with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/pipelines_aware/surveillance_1.webp) |

| draw 2 without | draw 2 with |
| --- | --- |
| ![draw 2 without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/pipelines/surveillance_2.webp) | ![draw 2 with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/pipelines_aware/surveillance_2.webp) |

| draw 3 without | draw 3 with |
| --- | --- |
| ![draw 3 without](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/pipelines/surveillance_3.webp) | ![draw 3 with](https://desmoteo.github.io/surmentations/examples/scenes/underpass_from_above/pipelines_aware/surveillance_3.webp) |
