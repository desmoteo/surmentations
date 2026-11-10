# street_night

Real night street under sodium lamps.

![original](original.jpg)

Source: [Great Holm Milton Keynes at night.jpg](https://commons.wikimedia.org/wiki/File:Great_Holm_Milton_Keynes_at_night.jpg) by John Chryslar, CC0. Every image in this folder is derived from it and is available under the same licence.

## How the transforms were told about this scene

| | |
| --- | --- |
| Distance of nearest and farthest point | 3 m to 150 m (estimated by eye) |
| Camera pitch below level | 5 degrees |
| Lighting | night, open to the sky |
| Sky in frame | yes |
| Fog | visibility 90 m (0.6 x the farthest point) |
| Rain and snow veils | capped so visibility stays above 225 m and 150 m |

## Effects at severity 3

One representative draw each: the median of seven by how much the frame changed.

| water_droplets | dirt_soil | fog |
| --- | --- | --- |
| ![water_droplets](effects/water_droplets.jpg) | ![dirt_soil](effects/dirt_soil.jpg) | ![fog](effects/fog.jpg) |

| rain | housing_occluders | night_ir |
| --- | --- | --- |
| ![rain](effects/rain.jpg) | ![housing_occluders](effects/housing_occluders.jpg) | ![night_ir](effects/night_ir.jpg) |

| night_rain | ac_flicker | rolling_shutter |
| --- | --- | --- |
| ![night_rain](effects/night_rain.jpg) | ![ac_flicker](effects/ac_flicker.jpg) | ![rolling_shutter](effects/rolling_shutter.jpg) |

| exposure_transient | wdr_tone_mapping | denoise_artifacts |
| --- | --- | --- |
| ![exposure_transient](effects/exposure_transient.jpg) | ![wdr_tone_mapping](effects/wdr_tone_mapping.jpg) | ![denoise_artifacts](effects/denoise_artifacts.jpg) |

| video_codec | night_snow | night_droplets |
| --- | --- | --- |
| ![video_codec](effects/video_codec.jpg) | ![night_snow](effects/night_snow.jpg) | ![night_droplets](effects/night_droplets.jpg) |

| sensor_noise |  |  |
| --- | --- | --- |
| ![sensor_noise](effects/sensor_noise.jpg) |  |  |

Not rendered: `sun_flare`, `cast_shadows` — it needs the sun, and the scene is at night.

Not rendered: `snow` — its settled snow is lit as by day, and comes out as a white floor under a black sky. `NightSnow` is the night model.

## `surveillance_lite` pipeline

4 consecutive draws. Each is a different combination of effects.

| draw 0 | draw 1 |
| --- | --- |
| ![draw 0](pipelines/surveillance_lite_0.jpg) | ![draw 1](pipelines/surveillance_lite_1.jpg) |

| draw 2 | draw 3 |
| --- | --- |
| ![draw 2](pipelines/surveillance_lite_2.jpg) | ![draw 3](pipelines/surveillance_lite_3.jpg) |
