# underpass_from_above

Road passing under a flyover, seen from slightly above; cars; foliage in the foreground.

![original](original.jpg)

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

![labels and horizon](reading.jpg)

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
| ![`fog` without](effects/fog.jpg) | ![`fog` with](aware/fog.jpg) |

| `rain` without | `rain` with |
| --- | --- |
| ![`rain` without](effects/rain.jpg) | ![`rain` with](aware/rain.jpg) |

| `snow` without | `snow` with |
| --- | --- |
| ![`snow` without](effects/snow.jpg) | ![`snow` with](aware/snow.jpg) |

| `cast_shadows` without | `cast_shadows` with |
| --- | --- |
| ![`cast_shadows` without](effects/cast_shadows.jpg) | ![`cast_shadows` with](aware/cast_shadows.jpg) |

| `night_ir` without | `night_ir` with |
| --- | --- |
| ![`night_ir` without](effects/night_ir.jpg) | ![`night_ir` with](aware/night_ir.jpg) |

| `night_rain` without | `night_rain` with |
| --- | --- |
| ![`night_rain` without](effects/night_rain.jpg) | ![`night_rain` with](aware/night_rain.jpg) |

| `night_snow` without | `night_snow` with |
| --- | --- |
| ![`night_snow` without](effects/night_snow.jpg) | ![`night_snow` with](aware/night_snow.jpg) |

| `wet_ground` without | `wet_ground` with |
| --- | --- |
| ![`wet_ground` without](effects/wet_ground.jpg) | ![`wet_ground` with](aware/wet_ground.jpg) |

| `snow_cover` without | `snow_cover` with |
| --- | --- |
| ![`snow_cover` without](effects/snow_cover.jpg) | ![`snow_cover` with](aware/snow_cover.jpg) |

| `ground_fog` without | `ground_fog` with |
| --- | --- |
| ![`ground_fog` without](effects/ground_fog.jpg) | ![`ground_fog` with](aware/ground_fog.jpg) |

| `local_plumes` without | `local_plumes` with |
| --- | --- |
| ![`local_plumes` without](effects/local_plumes.jpg) | ![`local_plumes` with](aware/local_plumes.jpg) |

| `night_color` without | `night_color` with |
| --- | --- |
| ![`night_color` without](effects/night_color.jpg) | ![`night_color` with](aware/night_color.jpg) |

## The other effects at severity 3

One representative draw each: the median of seven by how much the frame changed.
These do not read the scene, and are the same with either provider.

| water_droplets | dirt_soil | sun_flare |
| --- | --- | --- |
| ![water_droplets](effects/water_droplets.jpg) | ![dirt_soil](effects/dirt_soil.jpg) | ![sun_flare](effects/sun_flare.jpg) |

| housing_occluders | ac_flicker | rolling_shutter |
| --- | --- | --- |
| ![housing_occluders](effects/housing_occluders.jpg) | ![ac_flicker](effects/ac_flicker.jpg) | ![rolling_shutter](effects/rolling_shutter.jpg) |

| exposure_transient | wdr_tone_mapping | denoise_artifacts |
| --- | --- | --- |
| ![exposure_transient](effects/exposure_transient.jpg) | ![wdr_tone_mapping](effects/wdr_tone_mapping.jpg) | ![denoise_artifacts](effects/denoise_artifacts.jpg) |

| video_codec | night_droplets | sensor_noise |
| --- | --- | --- |
| ![video_codec](effects/video_codec.jpg) | ![night_droplets](effects/night_droplets.jpg) | ![sensor_noise](effects/sensor_noise.jpg) |

| transmission_faults | spider_web | night_insects |
| --- | --- | --- |
| ![transmission_faults](effects/transmission_faults.jpg) | ![spider_web](effects/spider_web.jpg) | ![night_insects](effects/night_insects.jpg) |

| ir_dome_glare | frost_on_glass | snow_on_housing |
| --- | --- | --- |
| ![ir_dome_glare](effects/ir_dome_glare.jpg) | ![frost_on_glass](effects/frost_on_glass.jpg) | ![snow_on_housing](effects/snow_on_housing.jpg) |

| lens_tamper | foliage_intrusion | ir_cut_filter_fault |
| --- | --- | --- |
| ![lens_tamper](effects/lens_tamper.jpg) | ![foliage_intrusion](effects/foliage_intrusion.jpg) | ![ir_cut_filter_fault](effects/ir_cut_filter_fault.jpg) |

| demosaic_artifacts | analog_video | focus_shift |
| --- | --- | --- |
| ![demosaic_artifacts](effects/demosaic_artifacts.jpg) | ![analog_video](effects/analog_video.jpg) | ![focus_shift](effects/focus_shift.jpg) |

| camera_vibration | on_screen_overlay | night_foliage |
| --- | --- | --- |
| ![camera_vibration](effects/camera_vibration.jpg) | ![on_screen_overlay](effects/on_screen_overlay.jpg) | ![night_foliage](effects/night_foliage.jpg) |

Not rendered: `night_fog`, `source_glare` — it acts on artificial lights at night, and this is a daylit scene.

Not rendered: `heat_shimmer` — it is turbulence over sunlit ground, which shows over a kilometre or so by day: this scene is too short, under a roof or at night.

## `surveillance` pipeline

4 consecutive draws, each a different combination of effects:
on the left with the plain provider, on the right the same draw with the
provider that read the scene.

| draw 0 without | draw 0 with |
| --- | --- |
| ![draw 0 without](pipelines/surveillance_0.jpg) | ![draw 0 with](pipelines_aware/surveillance_0.jpg) |

| draw 1 without | draw 1 with |
| --- | --- |
| ![draw 1 without](pipelines/surveillance_1.jpg) | ![draw 1 with](pipelines_aware/surveillance_1.jpg) |

| draw 2 without | draw 2 with |
| --- | --- |
| ![draw 2 without](pipelines/surveillance_2.jpg) | ![draw 2 with](pipelines_aware/surveillance_2.jpg) |

| draw 3 without | draw 3 with |
| --- | --- |
| ![draw 3 without](pipelines/surveillance_3.jpg) | ![draw 3 with](pipelines_aware/surveillance_3.jpg) |
