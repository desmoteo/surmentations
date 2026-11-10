# crosswalk_day

Near-vertical view of a crossing; people, cyclists, vehicles; no sky.

![original](original.jpg)

Source: [Crosswalk of Market at Third, San Francisco.jpg](https://commons.wikimedia.org/wiki/File:Crosswalk_of_Market_at_Third,_San_Francisco.jpg) by Dllu, CC BY-SA 4.0. Every image in this folder is derived from it and is available under the same licence.

## How the transforms were told about this scene

| | |
| --- | --- |
| Distance of nearest and farthest point | 12 m to 30 m (estimated by eye) |
| Camera pitch below level | 70 degrees |
| Lighting | day, open to the sky |
| Sky in frame | no |
| Fog | visibility 18 m (0.6 x the farthest point) |
| Rain and snow veils | capped so visibility stays above 45 m and 30 m |

## Effects at severity 3

One representative draw each: the median of seven by how much the frame changed.

| water_droplets | dirt_soil | sun_flare |
| --- | --- | --- |
| ![water_droplets](effects/water_droplets.jpg) | ![dirt_soil](effects/dirt_soil.jpg) | ![sun_flare](effects/sun_flare.jpg) |

| fog | rain | snow |
| --- | --- | --- |
| ![fog](effects/fog.jpg) | ![rain](effects/rain.jpg) | ![snow](effects/snow.jpg) |

| housing_occluders | cast_shadows | night_ir |
| --- | --- | --- |
| ![housing_occluders](effects/housing_occluders.jpg) | ![cast_shadows](effects/cast_shadows.jpg) | ![night_ir](effects/night_ir.jpg) |

| night_rain | ac_flicker | rolling_shutter |
| --- | --- | --- |
| ![night_rain](effects/night_rain.jpg) | ![ac_flicker](effects/ac_flicker.jpg) | ![rolling_shutter](effects/rolling_shutter.jpg) |

| exposure_transient | wdr_tone_mapping | denoise_artifacts |
| --- | --- | --- |
| ![exposure_transient](effects/exposure_transient.jpg) | ![wdr_tone_mapping](effects/wdr_tone_mapping.jpg) | ![denoise_artifacts](effects/denoise_artifacts.jpg) |

| video_codec | night_snow | night_droplets |
| --- | --- | --- |
| ![video_codec](effects/video_codec.jpg) | ![night_snow](effects/night_snow.jpg) | ![night_droplets](effects/night_droplets.jpg) |

| sensor_noise | transmission_faults | wet_ground |
| --- | --- | --- |
| ![sensor_noise](effects/sensor_noise.jpg) | ![transmission_faults](effects/transmission_faults.jpg) | ![wet_ground](effects/wet_ground.jpg) |

| snow_cover |  |  |
| --- | --- | --- |
| ![snow_cover](effects/snow_cover.jpg) |  |  |

Not rendered: `night_fog`, `source_glare` — it acts on artificial lights at night, and this is a daylit scene.

## `surveillance` pipeline

4 consecutive draws. Each is a different combination of effects.

| draw 0 | draw 1 |
| --- | --- |
| ![draw 0](pipelines/surveillance_0.jpg) | ![draw 1](pipelines/surveillance_1.jpg) |

| draw 2 | draw 3 |
| --- | --- |
| ![draw 2](pipelines/surveillance_2.jpg) | ![draw 3](pipelines/surveillance_3.jpg) |
