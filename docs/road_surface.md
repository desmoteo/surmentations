# The road surface: lanes, wheel tracks, standing water and spray

Some of what happens on a road is laid out *along* it. Wheels wear two
tracks down every lane of a snowy road. Water stands in the same ruts and
along the kerb. A car on a wet road trails its spray behind it. None of
that can be drawn from a mask of where the road is: it needs the road's own
coordinates, and its lanes.

[`surmentations.utils.road`](../src/surmentations/utils/road.py) reads
both off the picture, and `RandomSnow` and `RandomRain` use them:

| | Argument | Default | Needs |
| --- | --- | --- | --- |
| Wheel tracks in settled snow | `RandomSnow(wheel_track_range=...)` | `(0.0, 0.9)` | scene labels |
| Water in the ruts and gutters | `RandomRain`, with its wet road | on | scene labels |
| A soaked road under cloud | `RandomRain`, with its wet road | on | nothing |
| Tail lamps reflected in the road | `RandomRain`, with its wet road | on | scene labels |
| Spray behind vehicles | `RandomRain(spray_range=...)` | `(0.0, 0.0)`: off | scene labels |

"Scene labels" means a provider built with `labels=`
([scene understanding](scene_understanding.md)): they say where the road
and its vehicles are. Without them there is no road to follow, and
everything in the table that needs them is simply not drawn.

```python
provider = SharedDepthProvider(DepthAnythingV2Estimator(), labels=SegmentationLabelEstimator())
snow = RandomSnow(depth_provider=provider, wheel_track_range=(0.2, 0.8))
rain = RandomRain(depth_provider=provider, wetness_probability=1.0, spray_range=(0.3, 0.8))
```

## Reading a road

```mermaid
flowchart TD
    L["scene labels: road, vehicles, people"] --> CUES
    IMG["the picture"] --> CUES["cues to the road's direction:<br/>painted streaks, the road mask's edges, vehicle outlines"]
    GEO["camera horizon and height, where known"] --> FIT
    CUES --> FIT["robust fit of one road model:<br/>u = centre + offset (v - horizon) + bend / (v - horizon)"]
    FIT --> OFF["offset: a position across the road for every pixel"]
    OFF --> CARR["carriageways: stretches of offset that are solid road"]
    CARR --> COMB["one comb of lane lines per carriageway"]
    PAINT["paint lying along the road"] --> COMB
    VEH["vehicles: position and width across the road"] --> COMB
    GEO --> COMB
    COMB --> TRACKS["wheel tracks, ruts, gutters"]
    FIT --> DIR["the road's direction at every pixel"]
    DIR --> SPRAY["spray plumes"]
    LAMPS["each vehicle's own lamps: red = seen from behind"] --> SPRAY
```

**One model for the whole road.** On level ground, every line that runs
along a road of constant curvature is, in the picture,

    u = centre + offset * (v - horizon) + bend / (v - horizon)

`horizon`, `centre` and `bend` are the road's, the same for every lane
line, kerb and wheel track; `offset` is each line's own. So `offset` is a
position across the road for every pixel, constant along a lane on a bend
as on a straight, and across two carriageways as across one
([`RoadModel`](../src/surmentations/utils/road.py)).

**Fitted to everything that points along the road.** Painted streaks, the
edges of the road's mask and the long way of each vehicle's outline are
each a short piece of line with a direction. None is trusted: a slip road,
the end of a kerb and a zebra crossing are in there too. Two cues make a
road, so many pairs are tried and the road kept is the one the largest
share of the cues lies along to within a few degrees; a straight road is
preferred unless a bend fits clearly better. Where the provider knows the
camera's horizon ([scene understanding](scene_understanding.md)) it is
used, which leaves two numbers to fit instead of three and is what keeps
the fit from bending a road to pass through anything.

**Carriageways.** Counted by offset, the road's mask is solid where there
is a carriageway and empty where there is a reserve.

**The lanes are one comb per carriageway**: a width and a phase, scored on
everything at once.

- *Paint lies on the comb's teeth.* The share of the paint that lies on a
  lane line, lowered for lane lines with no paint and by a small cost per
  line, so that a comb twice too fine does not win by having a tooth under
  everything.
- *Vehicles sit between the teeth*, in the middle of a lane.
- *A lane has a size.* Where the camera's height is known the road's width
  is known in metres, and a lane is 3.4 m give or take a quarter.
  Otherwise a lane is 1.7 times the width of what drives in it, within a
  wide margin.

**When no such road fits**, one running across the picture or seen from
straight above, the road is read off its mask alone, as before; and there
lanes are laid only where the paint or the vehicles vouch for them, never
guessed. Tracks ruled across a junction are worse than none.

Cues in yellow, paint in magenta, lane lines in cyan, tracks in orange:

![roads as read](assets/road_surface/lanes.jpg)

## Wheel tracks in snow

Two tracks per lane where traffic has worn the settled snow off, with snow
left between the wheels, between the lanes and along the verges. Nothing
about them is even, because ruled lines are a pedestrian crossing: each
lane carries its own share of the traffic, a track strays a little from
side to side, and it is worn deep in one stretch and nearly gone in the
next. Far down the road the tracks run together into a greyer road, as they
do to the eye. The road under them is wet and darker than it was dry.

`wheel_track_range` runs from untouched snow (0) through two ruts a tyre
wide (0.3) to lanes worn bare (1). Left, without; right, with:

![wheel tracks](assets/road_surface/wheel_tracks.jpg)

## The wet road

- **Water in the ruts and gutters.** A road drains to its kerbs and has two
  ruts in every lane. The noise the puddles are cut from is lowered there,
  so a little water is strips along the road and only a lot of it is
  everywhere. Puddles have an edge, their share is a share of the wet road,
  and rain roughens them in step with its rate.
- **The sky that soaked it.** A road does not get soaked under a clear
  sky. A film on the road now brings cloud with it whatever is falling
  (scaled by `overcast_range`, so a caller who says there is none is
  believed). A soaked road under light rain used to be lit by the sun,
  which is the light a wet road is least seen in.
- **Tail lamps.** Lit lamps on labelled vehicles are reflected in the road
  as street lamps are, as a long streak. By day only red ones count: a
  white point on a car in daylight is the sun on its glass.

## Spray

Each vehicle on a road with a film of water trails a low cloud of mist that
lies on the road behind it: densest at the wheels and no higher than they
are, as wide as the vehicle, then a little wider and up to about its roof,
thinning out over a length and a half.

- **Where it lies is the road's direction at the vehicle**, from the model
  above, turned to the vehicle's own outline where that is long and close
  to it: a car changing lanes trails its spray along its own line.
- **Which end is the tail** comes from each vehicle's own picture. Red
  lamps are tail lamps: it is seen from behind and driving away. White
  lamps at night are headlamps: it is coming on. A vehicle showing neither
  goes the way of the nearest one across the road that does, since the two
  sides of a road run opposite ways. With no lamp showing on a road, its
  direction is drawn.
- **How fast** each vehicle goes is drawn: each gets its own share of
  spray, and more than a third get none.
- **Some vehicles never get one**: a vehicle lying across the road, which
  is crossing, turning or parked; and any vehicle on a scrap of road under
  a twentieth of the frame, whose direction is a guess. Both used to
  shoot a jet of mist out sideways.
- The plume of a vehicle coming towards the camera is hidden by the
  vehicle.

Left, a soaked road; right, the same with `spray_range=(0.6, 0.6)`:

![wet road and spray](assets/road_surface/wet_road_and_spray.jpg)

It is off by default because of what one frame cannot say: a queue of
standing traffic gets plumes, and without tail lamps a plume may trail the
wrong way. It is a soft grey cloud and no more: there is no ballistic
sheet of water off the tyres and no side spray.

## Limits

- **Only with scene labels**, and only as good as they are. The network
  that names a crossing seen from straight above "building" leaves a piece
  of it that is read as a road running the wrong way. A road whose
  vehicles lie across it is caught; the rest is not.
- **The road is taken to be level and of one curvature.** A hill, an
  S-bend or a junction does not fit, and falls back to the mask.
- **Carriageways divided by a low barrier are often read as one road**,
  when the network labels the barrier as road. Lanes then keep one width
  and phase across both, and are right on one side and a little off on the
  other.
- **Lane counts are estimates.** Without a camera height they rest on the
  vehicles' widths; with one, on a height that is itself rough.
- **Paint on a wet or night road** is mostly glare and is ignored when
  there is too much of it to be paint.
- **A fixed camera's lanes do not move.** Lane lines given once per
  camera, as an installation geometry is, would be surer than any of
  this; that is not built.
- **Junctions, squares and car parks have no lanes.** A ragged mask gets
  no road; a tidy one is read as a road.
- **Touching vehicles are one vehicle** to a semantic mask, so dense
  traffic gives fewer, wider plumes.
- **A pale rim** can still show round vehicles on a dark wet road, where
  the tests that keep the road off the vehicle reach a pixel or two past
  it.
- The second and third roads of a frame, cut by its edge, have a rougher
  direction than the first.

## References

- Hartley, R., Zisserman, A. (2004). Multiple View Geometry in Computer
  Vision, 2nd ed. Cambridge University Press.
  [doi:10.1017/CBO9780511811685](https://doi.org/10.1017/CBO9780511811685)
  (lines through a vanishing point; the cross-ratio that keeps position
  across a road constant along a lane line)
- Lekner, J., Dorf, M. C. (1988). Why some things are darker when wet.
  Applied Optics 27(7), 1278–1280.
  [doi:10.1364/AO.27.001278](https://doi.org/10.1364/AO.27.001278)
