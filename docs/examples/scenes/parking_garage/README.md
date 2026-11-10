# parking_garage

Covered car park; pillars and a ceiling; short depth range.

![original](original.jpg)

Source: [Interior of Rupert Street parking garage - DSC 8051.jpg](https://commons.wikimedia.org/wiki/File:Interior_of_Rupert_Street_parking_garage_-_DSC_8051.jpg) by Tadeusz Hare, CC BY-SA 4.0. Every image in this folder is derived from it and is available under the same licence.

## How the transforms were told about this scene

| | |
| --- | --- |
| Distance of nearest and farthest point | 2 m to 45 m (estimated by eye) |
| Camera pitch below level | 4 degrees |
| Lighting | day, under a roof |
| Sky in frame | no |
| Fog | visibility 27 m (0.6 x the farthest point) |
| Rain and snow veils | capped so visibility stays above 67.5 m and 45 m |

## Effects at severity 3

One representative draw each: the median of seven by how much the frame changed.

| water_droplets | dirt_soil | fog |
| --- | --- | --- |
| ![water_droplets](effects/water_droplets.jpg) | ![dirt_soil](effects/dirt_soil.jpg) | ![fog](effects/fog.jpg) |

| housing_occluders | night_ir | ac_flicker |
| --- | --- | --- |
| ![housing_occluders](effects/housing_occluders.jpg) | ![night_ir](effects/night_ir.jpg) | ![ac_flicker](effects/ac_flicker.jpg) |

| rolling_shutter | exposure_transient | wdr_tone_mapping |
| --- | --- | --- |
| ![rolling_shutter](effects/rolling_shutter.jpg) | ![exposure_transient](effects/exposure_transient.jpg) | ![wdr_tone_mapping](effects/wdr_tone_mapping.jpg) |

| denoise_artifacts | video_codec | night_droplets |
| --- | --- | --- |
| ![denoise_artifacts](effects/denoise_artifacts.jpg) | ![video_codec](effects/video_codec.jpg) | ![night_droplets](effects/night_droplets.jpg) |

| sensor_noise |  |  |
| --- | --- | --- |
| ![sensor_noise](effects/sensor_noise.jpg) |  |  |

Not rendered: `sun_flare`, `rain`, `snow`, `cast_shadows`, `night_rain`, `night_snow` — the scene is under a roof.

Not rendered: `night_fog`, `source_glare` — it acts on artificial lights at night, and this is a daylit scene.

## `surveillance_lite` pipeline

4 consecutive draws. Each is a different combination of effects.

| draw 0 | draw 1 |
| --- | --- |
| ![draw 0](pipelines/surveillance_lite_0.jpg) | ![draw 1](pipelines/surveillance_lite_1.jpg) |

| draw 2 | draw 3 |
| --- | --- |
| ![draw 2](pipelines/surveillance_lite_2.jpg) | ![draw 3](pipelines/surveillance_lite_3.jpg) |
