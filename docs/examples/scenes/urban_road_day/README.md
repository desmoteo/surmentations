# urban_road_day

Urban road by day from overpass height; traffic, lamp posts, skyline.

![original](original.jpg)

Source: [NFTA Metro Bus 18 @ Kensington Expressway overpass bridge, Buffalo, New York - 20220805.jpg](https://commons.wikimedia.org/wiki/File:NFTA_Metro_Bus_18_@_Kensington_Expressway_overpass_bridge,_Buffalo,_New_York_-_20220805.jpg) by Andre Carrotflower, CC BY-SA 4.0. Every image in this folder is derived from it and is available under the same licence.

## How the transforms were told about this scene

| | |
| --- | --- |
| Distance of nearest and farthest point | 8 m to 900 m (estimated by eye) |
| Camera pitch below level | 10 degrees |
| Lighting | day, open to the sky |
| Sky in frame | yes |
| Fog | visibility 540 m (0.6 x the farthest point) |
| Rain and snow veils | capped so visibility stays above 1350 m and 900 m |

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

| sensor_noise |  |  |
| --- | --- | --- |
| ![sensor_noise](effects/sensor_noise.jpg) |  |  |

Not rendered: `night_fog`, `source_glare` — it acts on artificial lights at night, and this is a daylit scene.

## `surveillance` pipeline

4 consecutive draws. Each is a different combination of effects.

| draw 0 | draw 1 |
| --- | --- |
| ![draw 0](pipelines/surveillance_0.jpg) | ![draw 1](pipelines/surveillance_1.jpg) |

| draw 2 | draw 3 |
| --- | --- |
| ![draw 2](pipelines/surveillance_2.jpg) | ![draw 3](pipelines/surveillance_3.jpg) |
