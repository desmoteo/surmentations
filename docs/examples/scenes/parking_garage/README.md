# parking_garage

Covered car park; pillars and a ceiling; short depth range.

![original](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/original.webp)

Source: [Interior of Rupert Street parking garage - DSC 8051.jpg](https://commons.wikimedia.org/wiki/File:Interior_of_Rupert_Street_parking_garage_-_DSC_8051.jpg) by Tadeusz Hare, CC BY-SA 4.0. Every image in this folder is derived from it and is available under the same licence.

## How the transforms were told about this scene

| | |
| --- | --- |
| Distance of nearest and farthest point | 2 m to 45 m (estimated by eye) |
| Camera pitch below level | 4 degrees |
| Lighting | day, under a roof |
| Sky in frame | no |
| Fog | visibility 0.6 x the scene's depth: about 27 m by eye |
| Rain and snow veils | visibility kept above 1.5 x and 1 x the scene's depth: about 67.5 m and 45 m |

## What a provider with scene labels and geometry read off it

The photograph and its label map, with the horizon the geometry estimator
found drawn across both. See
[scene understanding](../../../scene_understanding.md) for how each is made
and how far to trust it.

![labels and horizon](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/reading.webp)

| | |
| --- | --- |
| Field of view | 68°, assumed: a picture does not tell it |
| Horizon | 0.52 of the frame's height from the top, or above: there is no sky in the frame to mark it |
| Camera pitch below level | at least -1°, so the by-eye 4° stands |
| Roll | +0.4° |
| Camera height | not found: fewer than three usable vehicles or people in view |
| Depth | the by-eye range above: there is no height to measure from |
| Labels | structure 58%, pavement 42%, lamp 0%, pole 0% |

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
| ![`fog` without](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/fog.webp) | ![`fog` with](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/aware/fog.webp) |

| `night_ir` without | `night_ir` with |
| --- | --- |
| ![`night_ir` without](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/night_ir.webp) | ![`night_ir` with](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/aware/night_ir.webp) |

| `local_plumes` without | `local_plumes` with |
| --- | --- |
| ![`local_plumes` without](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/local_plumes.webp) | ![`local_plumes` with](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/aware/local_plumes.webp) |

| `night_color` without | `night_color` with |
| --- | --- |
| ![`night_color` without](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/night_color.webp) | ![`night_color` with](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/aware/night_color.webp) |

## The other effects at severity 3

One representative draw each: the median of seven by how much the frame changed.
These do not read the scene, and are the same with either provider.

| water_droplets | dirt_soil | housing_occluders |
| --- | --- | --- |
| ![water_droplets](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/water_droplets.webp) | ![dirt_soil](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/dirt_soil.webp) | ![housing_occluders](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/housing_occluders.webp) |

| ac_flicker | rolling_shutter | exposure_transient |
| --- | --- | --- |
| ![ac_flicker](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/ac_flicker.webp) | ![rolling_shutter](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/rolling_shutter.webp) | ![exposure_transient](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/exposure_transient.webp) |

| wdr_tone_mapping | denoise_artifacts | video_codec |
| --- | --- | --- |
| ![wdr_tone_mapping](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/wdr_tone_mapping.webp) | ![denoise_artifacts](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/denoise_artifacts.webp) | ![video_codec](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/video_codec.webp) |

| night_droplets | sensor_noise | transmission_faults |
| --- | --- | --- |
| ![night_droplets](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/night_droplets.webp) | ![sensor_noise](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/sensor_noise.webp) | ![transmission_faults](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/transmission_faults.webp) |

| spider_web | night_insects | ir_dome_glare |
| --- | --- | --- |
| ![spider_web](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/spider_web.webp) | ![night_insects](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/night_insects.webp) | ![ir_dome_glare](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/ir_dome_glare.webp) |

| lens_tamper | ir_cut_filter_fault | demosaic_artifacts |
| --- | --- | --- |
| ![lens_tamper](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/lens_tamper.webp) | ![ir_cut_filter_fault](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/ir_cut_filter_fault.webp) | ![demosaic_artifacts](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/demosaic_artifacts.webp) |

| analog_video | focus_shift | camera_vibration |
| --- | --- | --- |
| ![analog_video](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/analog_video.webp) | ![focus_shift](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/focus_shift.webp) | ![camera_vibration](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/camera_vibration.webp) |

| on_screen_overlay |  |  |
| --- | --- | --- |
| ![on_screen_overlay](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/effects/on_screen_overlay.webp) |  |  |

Not rendered: `sun_flare`, `rain`, `snow`, `cast_shadows`, `night_rain`, `night_snow`, `wet_ground`, `snow_cover`, `frost_on_glass`, `snow_on_housing`, `foliage_intrusion`, `ground_fog`, `night_foliage` — the scene is under a roof.

Not rendered: `night_fog`, `source_glare` — it acts on artificial lights at night, and this is a daylit scene.

Not rendered: `heat_shimmer` — it is turbulence over sunlit ground, which shows over a kilometre or so by day: this scene is too short, under a roof or at night.

## `surveillance_lite` pipeline

4 consecutive draws. Each is a different combination of effects.
This pipeline takes no depth provider, so there is nothing to compare.

| draw 0 | draw 1 |
| --- | --- |
| ![draw 0](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/pipelines/surveillance_lite_0.webp) | ![draw 1](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/pipelines/surveillance_lite_1.webp) |

| draw 2 | draw 3 |
| --- | --- |
| ![draw 2](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/pipelines/surveillance_lite_2.webp) | ![draw 3](https://desmoteo.github.io/surmentations/examples/scenes/parking_garage/pipelines/surveillance_lite_3.webp) |
