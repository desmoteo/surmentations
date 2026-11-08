# urban_road_day

Urban road by day from overpass height; traffic, lamp posts, skyline.

![original](original.jpg)

Source: [NFTA Metro Bus 18 @ Kensington Expressway overpass bridge, Buffalo, New York - 20220805.jpg](https://commons.wikimedia.org/wiki/File:NFTA_Metro_Bus_18_@_Kensington_Expressway_overpass_bridge,_Buffalo,_New_York_-_20220805.jpg) by Andre Carrotflower, CC BY-SA 4.0. Every image in this folder is derived from it and is available under the same licence.

## How the transforms were told about this scene

| | |
| --- | --- |
| Distance of nearest and farthest point | 8 m to 900 m (estimated by eye) |
| Camera pitch below level | -3 degrees |
| Lighting | day, open to the sky |
| Sky in frame | yes |
| Fog | visibility 0.6 x the scene's depth: about 540 m by eye |
| Rain and snow veils | visibility kept above 1.5 x and 1 x the scene's depth: about 1350 m and 900 m |

## What a provider with scene labels and geometry read off it

The photograph and its label map, with the horizon the geometry estimator
found drawn across both. See
[scene understanding](../../../scene_understanding.md) for how each is made
and how far to trust it.

![labels and horizon](reading.jpg)

| | |
| --- | --- |
| Field of view | 52°, assumed: a picture does not tell it |
| Horizon | 0.60 of the frame's height from the top |
| Camera pitch below level | -3° (by eye: -3°) |
| Roll | +0.4° |
| Camera height | 7 m, from the vehicles and people in view |
| Depth | in metres from the camera's height: 30 m to 1530 m |
| Labels | sky 34%, road 22%, foliage 20%, structure 12%, vehicle 5%, pole 3% |

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
With that reader `night_ir` lights the traffic's lamps too, as it does whenever
a provider has one.

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

| `heat_shimmer` without | `heat_shimmer` with |
| --- | --- |
| ![`heat_shimmer` without](effects/heat_shimmer.jpg) | ![`heat_shimmer` with](aware/heat_shimmer.jpg) |

## The other effects at severity 3

One representative draw each: the median of seven by how much the frame changed.
These do not read the scene, and are the same with either provider.

| water_droplets | dirt_soil | sun_flare |
| --- | --- | --- |
| ![water_droplets](effects/water_droplets.jpg) | ![dirt_soil](effects/dirt_soil.jpg) | ![sun_flare](effects/sun_flare.jpg) |

| housing_occluders | night_rain | ac_flicker |
| --- | --- | --- |
| ![housing_occluders](effects/housing_occluders.jpg) | ![night_rain](effects/night_rain.jpg) | ![ac_flicker](effects/ac_flicker.jpg) |

| rolling_shutter | exposure_transient | wdr_tone_mapping |
| --- | --- | --- |
| ![rolling_shutter](effects/rolling_shutter.jpg) | ![exposure_transient](effects/exposure_transient.jpg) | ![wdr_tone_mapping](effects/wdr_tone_mapping.jpg) |

| denoise_artifacts | video_codec | night_droplets |
| --- | --- | --- |
| ![denoise_artifacts](effects/denoise_artifacts.jpg) | ![video_codec](effects/video_codec.jpg) | ![night_droplets](effects/night_droplets.jpg) |

| sensor_noise | transmission_faults | spider_web |
| --- | --- | --- |
| ![sensor_noise](effects/sensor_noise.jpg) | ![transmission_faults](effects/transmission_faults.jpg) | ![spider_web](effects/spider_web.jpg) |

| night_insects | ir_dome_glare | frost_on_glass |
| --- | --- | --- |
| ![night_insects](effects/night_insects.jpg) | ![ir_dome_glare](effects/ir_dome_glare.jpg) | ![frost_on_glass](effects/frost_on_glass.jpg) |

| snow_on_housing | lens_tamper | foliage_intrusion |
| --- | --- | --- |
| ![snow_on_housing](effects/snow_on_housing.jpg) | ![lens_tamper](effects/lens_tamper.jpg) | ![foliage_intrusion](effects/foliage_intrusion.jpg) |

| ir_cut_filter_fault | demosaic_artifacts | analog_video |
| --- | --- | --- |
| ![ir_cut_filter_fault](effects/ir_cut_filter_fault.jpg) | ![demosaic_artifacts](effects/demosaic_artifacts.jpg) | ![analog_video](effects/analog_video.jpg) |

| focus_shift | camera_vibration | on_screen_overlay |
| --- | --- | --- |
| ![focus_shift](effects/focus_shift.jpg) | ![camera_vibration](effects/camera_vibration.jpg) | ![on_screen_overlay](effects/on_screen_overlay.jpg) |

| night_foliage |  |  |
| --- | --- | --- |
| ![night_foliage](effects/night_foliage.jpg) |  |  |

Not rendered: `night_fog`, `source_glare` — it acts on artificial lights at night, and this is a daylit scene.

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
