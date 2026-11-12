# street_night

Real night street under sodium lamps.

![original](original.jpg)

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

![labels and horizon](reading.jpg)

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

| `rain` without | `rain` with |
| --- | --- |
| ![`rain` without](effects/rain.jpg) | ![`rain` with](aware/rain.jpg) |

| `snow` without | `snow` with |
| --- | --- |
| ![`snow` without](effects/snow.jpg) | ![`snow` with](aware/snow.jpg) |

| `night_ir` without | `night_ir` with |
| --- | --- |
| ![`night_ir` without](effects/night_ir.jpg) | ![`night_ir` with](aware/night_ir.jpg) |

| `night_snow` without | `night_snow` with |
| --- | --- |
| ![`night_snow` without](effects/night_snow.jpg) | ![`night_snow` with](aware/night_snow.jpg) |

| `night_fog` without | `night_fog` with |
| --- | --- |
| ![`night_fog` without](effects/night_fog.jpg) | ![`night_fog` with](aware/night_fog.jpg) |

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

## The other effects at severity 3

One representative draw each: the median of seven by how much the frame changed.
These do not read the scene, and are the same with either provider.

| water_droplets | dirt_soil | housing_occluders |
| --- | --- | --- |
| ![water_droplets](effects/water_droplets.jpg) | ![dirt_soil](effects/dirt_soil.jpg) | ![housing_occluders](effects/housing_occluders.jpg) |

| night_rain | ac_flicker | rolling_shutter |
| --- | --- | --- |
| ![night_rain](effects/night_rain.jpg) | ![ac_flicker](effects/ac_flicker.jpg) | ![rolling_shutter](effects/rolling_shutter.jpg) |

| exposure_transient | wdr_tone_mapping | denoise_artifacts |
| --- | --- | --- |
| ![exposure_transient](effects/exposure_transient.jpg) | ![wdr_tone_mapping](effects/wdr_tone_mapping.jpg) | ![denoise_artifacts](effects/denoise_artifacts.jpg) |

| video_codec | night_droplets | sensor_noise |
| --- | --- | --- |
| ![video_codec](effects/video_codec.jpg) | ![night_droplets](effects/night_droplets.jpg) | ![sensor_noise](effects/sensor_noise.jpg) |

| source_glare | transmission_faults | spider_web |
| --- | --- | --- |
| ![source_glare](effects/source_glare.jpg) | ![transmission_faults](effects/transmission_faults.jpg) | ![spider_web](effects/spider_web.jpg) |

| night_insects | ir_dome_glare | frost_on_glass |
| --- | --- | --- |
| ![night_insects](effects/night_insects.jpg) | ![ir_dome_glare](effects/ir_dome_glare.jpg) | ![frost_on_glass](effects/frost_on_glass.jpg) |

| snow_on_housing | lens_tamper | foliage_intrusion |
| --- | --- | --- |
| ![snow_on_housing](effects/snow_on_housing.jpg) | ![lens_tamper](effects/lens_tamper.jpg) | ![foliage_intrusion](effects/foliage_intrusion.jpg) |

| demosaic_artifacts | analog_video | focus_shift |
| --- | --- | --- |
| ![demosaic_artifacts](effects/demosaic_artifacts.jpg) | ![analog_video](effects/analog_video.jpg) | ![focus_shift](effects/focus_shift.jpg) |

| camera_vibration | on_screen_overlay | night_foliage |
| --- | --- | --- |
| ![camera_vibration](effects/camera_vibration.jpg) | ![on_screen_overlay](effects/on_screen_overlay.jpg) | ![night_foliage](effects/night_foliage.jpg) |

Not rendered: `sun_flare`, `cast_shadows` — it needs the sun, and the scene is at night.

Not rendered: `fog` — it is the daylight model, an even veil lit by the sky. `night_fog` is fog lit by the lamps.

Not rendered: `ir_cut_filter_fault` — it is a fault of the colour picture by day, and the scene is at night.

Not rendered: `heat_shimmer` — it is turbulence over sunlit ground, which shows over a kilometre or so by day: this scene is too short, under a roof or at night.

## `surveillance_lite` pipeline

4 consecutive draws. Each is a different combination of effects.
This pipeline takes no depth provider, so there is nothing to compare.

| draw 0 | draw 1 |
| --- | --- |
| ![draw 0](pipelines/surveillance_lite_0.jpg) | ![draw 1](pipelines/surveillance_lite_1.jpg) |

| draw 2 | draw 3 |
| --- | --- |
| ![draw 2](pipelines/surveillance_lite_2.jpg) | ![draw 3](pipelines/surveillance_lite_3.jpg) |
