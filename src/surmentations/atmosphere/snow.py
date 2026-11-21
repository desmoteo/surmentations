"""Depth-conditioned snow: falling flakes plus depth-derived ground accumulation.

Physical model
--------------
Snow is parameterised in **physical units** — flake diameters in millimetres,
distances in metres, an exposure time in seconds — and projected through a
pinhole camera. That is what keeps the three things a flake's appearance
depends on locked together instead of being three independent tuning knobs:

* **Apparent size** ``= f * D / z``. A flake's distance ``z`` comes from a
  volume-uniform draw through the air in front of the lens (see
  ``utils/precipitation.py``), never from the depth map — a falling flake is
  *in front of* the scene, so what is behind it says nothing about how far
  away it is. Sizing flakes from the scene depth is what makes them look
  pasted onto surfaces: big over the near road, tiny over the distant sky.

* **Defocus**. A camera focused several metres out renders anything within a
  few centimetres of the lens as a large, very soft, translucent disc, well
  beyond the flake's true size. A defocused flake is not a Gaussian blob but
  an **image of the aperture**: a polygon with one side per blade,
  bright-rimmed from spherical aberration, and clipped into a cat's-eye lemon
  off-axis by optical vignetting.

* **Motion smear** ``= f * v * T / z``. This used to be a single global blur
  applied to the finished flake layer, which is wrong in the way that matters
  most: it smears near and far flakes by the *same* number of pixels, when
  the smear scales with distance exactly as the size does. Each flake now
  carries its own smear, and the ratio between smear and size is
  ``v T / D`` — independent of distance but inversely proportional to flake
  size, so small flakes read as short dashes and large ones stay blobby,
  which is the texture real snowfall has.

Snow falls *slowly*: aggregates reach only about 0.5-1.5 m/s, near enough
independent of size (Locatelli & Hobbs 1974 measure ``v ~ 1.1 D^0.12`` m/s
for unrimed dendrite aggregates, i.e. barely varying at all) because a
snowflake's drag and its mass grow together. That is roughly a fifth of a
raindrop's fall speed and is why snow streaks are short and rain streaks are
long at the same shutter speed.

Flakes also **flutter**: they are low-density plates and aggregates that
tumble and swing as they fall rather than dropping on a straight line, so
each one's smear direction deviates from the prevailing wind slant. Without
that, a field of flakes reads as a combed, unnaturally parallel texture.

Smearing a shape along a line is a **Minkowski sum**, which for a convex
aperture polygon is exact and free: the support distance of each edge simply
grows by the segment's extent projected onto that edge's normal. So the
smear costs no convolution — it is built into the sprite's own distance
field.

Accumulation
------------
Snow settles on surfaces that face **up**. Earlier versions brightened the
bottom band of the frame, which is a proxy for "ground" that fails on
anything interesting: a wall at the bottom of the frame got snow it should
not have, and a car roof or a rooftop higher up got none when it should be
the most obvious snow in the shot.

The depth map is enough to do this properly. Back-projecting it gives a
surface normal per pixel, and what lands on a surface goes as
``max(0, n . up)`` — Lambert's cosine law, applied to falling snow instead of
to light, since the geometry is the same. See ``depth/normals.py``, which
also documents where the estimate breaks down (it does not model shelter, so
the underside of a bridge collects snow it never would).

Four things had to be added before a covering looked like one:

* **Level ground is covered all over, up to the foot of what stands on it.**
  Estimated normals wander a few degrees, which left a road at half cover
  with a bare outline round every car.
* **Trees carry snow on top of their foliage, in clumps.** A canopy has no
  surface whose normal can be read, and reading one anyway covered trees in
  grey blotches the size of the depth map's errors. On foliage the snow
  goes on the parts that catch the light, which are the parts facing up.
* **Bare patches are patches of the ground.** Laid out in ground
  coordinates and with an edge, they are where wind and traffic cleared
  it. As soft screen-space noise they were a mist.
* **Snow is white because of what lights it.** It is shaded by the light
  on the patch it lies on, and that estimate has a floor: black asphalt
  says almost nothing about the light on it, and snow shaded by asphalt's
  own darkness came out grey. At night under sodium lamps the same snow is
  orange, so it takes the frame's colour cast when the frame has one.

The veil is the other half of "it is snowing": snow scatters some thirty
times more than rain at the same liquid-equivalent rate, so heavy snowfall
drops visibility hard, and that loss of contrast with distance is a stronger cue than the
flakes themselves.
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from surmentations import _native, mode
from surmentations.base import CameraTransform, first_frame, require_depth_provider, scene_of
from surmentations.depth.normals import upward_facing
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import record_draw
from surmentations.perception.labels import SceneClass, class_weight
from surmentations.utils.color import (
    TYPICAL_SCENE_ALBEDO,
    illuminant_radiance_linear,
    veil_radiance_linear,
)
from surmentations.utils.filters import wide_blur
from surmentations.utils.ground import (
    fill_to_edges,
    ground_noise,
    level_ground_weight,
    small_level_areas,
    vegetation_weight,
)
from surmentations.utils.overcast import apply_overcast, cloud_cover
from surmentations.utils.photometry import (
    from_linear_unit,
    luminance,
    srgb_to_linear,
    to_linear_unit,
)
from surmentations.utils.precipitation import (
    far_field_texture,
    focal_length_px,
    frustum_volume,
    gunn_marshall_slope,
    resolvable_fraction,
    sample_resolvable_particles,
    snow_extinction_coefficient,
    snow_number_density,
    snow_physical_diameter,
    width_cut_for_budget,
)
from surmentations.utils.road import read_road, wheel_tracks
from surmentations.utils.sky import (
    extinction_for_visibility,
    lift_sky,
    scene_extent_m,
    sky_to_depth_edge,
    sky_weight,
)

_ACCUMULATION_COLOR = np.array([235.0, 240.0, 245.0])

# Locatelli & Hobbs (1974) for unrimed aggregates: v = 1.1 * D^0.12 m/s with
# D in mm. The exponent is small enough that fall speed is nearly independent
# of flake size, which is the physically interesting part.
_SNOW_VELOCITY_COEFF = 1.1
_SNOW_VELOCITY_EXPONENT = 0.12


def snow_fall_speed(diameter_mm: np.ndarray | float) -> np.ndarray:
    """Terminal fall speed (m/s) of a snow aggregate ``diameter_mm`` across.

    Locatelli & Hobbs 1974. Unlike a raindrop — whose speed more than triples
    between 0.5 mm and 3 mm — a snow aggregate's speed barely moves, because
    its mass and its drag both scale with roughly the same power of size.
    """
    diameter = np.asarray(diameter_mm, dtype=np.float32)
    return _SNOW_VELOCITY_COEFF * np.maximum(diameter, 1e-3) ** _SNOW_VELOCITY_EXPONENT


def _smeared_aperture_distance(
    dy: np.ndarray,
    dx: np.ndarray,
    blades: int,
    blade_rotation: float,
    half_smear: float,
    smear_dir: tuple[float, float],
) -> np.ndarray:
    """Distance field of an aperture shape smeared along a motion segment.

    Returns a field that is < 1 inside the shape and 1 on its boundary, for
    the aperture swept along ``+-half_smear * smear_dir`` (both in units of
    the render radius).

    Sweeping a convex shape along a line segment is a Minkowski sum, and for
    a convex polygon expressed as an intersection of half-planes that sum has
    a closed form: each edge's supporting offset grows by the segment's
    extent projected onto that edge's normal. So a motion-smeared polygonal
    bokeh disc costs exactly as much to evaluate as an unsmeared one — no
    convolution, and no directional blur pass over the whole layer.
    """
    ux, uy = smear_dir

    if blades >= 3:
        angles = blade_rotation + 2.0 * np.pi * np.arange(blades) / blades
        base_support = np.cos(np.pi / blades)
        planes = []
        for angle in angles:
            # float64 scalars in the reference code, and so a float64 field;
            # fast mode keeps it float32, as the compiled kernel does.
            nx, ny = mode.scalar(np.cos(angle)), mode.scalar(np.sin(angle))
            support = mode.scalar(base_support + half_smear * abs(ux * nx + uy * ny))
            planes.append((dx * nx + dy * ny) / support)
        return np.max(np.stack(planes, axis=0), axis=0)

    # Circle swept along a segment is a capsule: distance to the segment.
    along = np.clip(dx * ux + dy * uy, -half_smear, half_smear)
    return np.sqrt((dx - along * ux) ** 2 + (dy - along * uy) ** 2)


def _paste_flake_sprite(
    canvas: np.ndarray,
    x: float,
    y: float,
    radius_px: float,
    blur_radius_px: float,
    smear_px: float,
    smear_dir: tuple[float, float],
    blades: int,
    blade_rotation: float,
    rim_gain: float,
    cats_eye: float,
    frame_centre: tuple[float, float],
    gain: float = 1.0,
) -> None:
    """Max-blend one flake's sprite into ``canvas``.

    ``radius_px`` is the flake's *geometric* projected radius and
    ``blur_radius_px`` half its circle of confusion. The rendered sprite is
    their quadrature sum, which is the standard approximation to convolving
    two discs, and the two limits it interpolates are the physically right
    ones: a flake near the focal plane renders at its own size and near-solid,
    while a flake against the front element renders as the aperture's image,
    far larger than the flake and correspondingly translucent.

    ``smear_px`` is the full length of the flake's motion streak over the
    exposure. ``gain`` scales the whole sprite, for a caller whose flakes are
    not all equally lit — by a lamp at the camera, say, rather than the sky.
    """
    render_radius = max(0.5, float(np.hypot(radius_px, blur_radius_px)))
    half_smear = 0.5 * smear_px / render_radius

    # The patch has to hold the disc plus the streak's reach in either
    # direction, or the sprite is clipped into a rectangle.
    reach = render_radius * (1.3 + half_smear)
    patch_size = max(3, int(round(reach * 2.0)) | 1)

    centre = patch_size / 2.0
    axis = (np.arange(patch_size, dtype=np.float32) - centre) / render_radius
    dy = axis.reshape(patch_size, 1)
    dx = axis.reshape(1, patch_size)

    dist = _smeared_aperture_distance(dy, dx, blades, blade_rotation, half_smear, smear_dir)

    # Cat's-eye: optical vignetting clips off-axis bokeh with a second
    # aperture displaced towards the frame centre, and the further off-axis
    # the flake sits the more of it is cut away.
    if cats_eye > 0.0:
        to_centre_x = frame_centre[0] - x
        to_centre_y = frame_centre[1] - y
        norm = float(np.hypot(to_centre_x, to_centre_y))
        if norm > 1e-3:
            dir_x, dir_y = to_centre_x / norm, to_centre_y / norm
            offset = cats_eye * min(1.0, norm / max(frame_centre[0], frame_centre[1]))
            clip = np.sqrt((dx - dir_x * offset) ** 2 + (dy - dir_y * offset) ** 2)
            dist = np.maximum(dist, clip)

    inside = np.clip(1.0 - dist, 0.0, 1.0)
    # Hard-ish edge (a defocus disc has a defined rim), then the aberration
    # profile: rim_gain > 1 brightens the edge, < 1 the centre.
    edge = np.clip(inside / 0.12, 0.0, 1.0)
    profile = 1.0 + (rim_gain - 1.0) * np.clip(1.0 - inside / 0.35, 0.0, 1.0)
    sprite: np.ndarray = edge * profile

    # Flux conservation. A flake reflects a fixed amount of light; defocus
    # and motion both only decide how wide an area it lands on, so the
    # sprite's peak is that flux divided by the area it now covers.
    #
    # Defocus spreads the flake's disc over the circle of confusion, so the
    # peak coverage is the ratio of the two areas — which is why a flake
    # pressed against the lens reads as a large *translucent* disc rather
    # than a large bright one, and why it fades smoothly to solid as the
    # flake approaches the focal plane.
    #
    # Motion smear spreads it further still, by the ratio of the swept area
    # to the static one: a disc swept along a segment gains 2*half_smear/pi.
    defocus_coverage = min(1.0, (radius_px / render_radius) ** 2)
    smear_area_ratio = 1.0 + 2.0 * half_smear / np.pi
    sprite = sprite * (gain * defocus_coverage / smear_area_ratio)

    ksize = max(1, int(render_radius * 0.12)) | 1
    if ksize > 1:
        sprite = cv2.GaussianBlur(sprite, (ksize, ksize), 0)

    h, w = canvas.shape[:2]
    x0 = int(round(x - centre))
    y0 = int(round(y - centre))
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(w, x0 + patch_size), min(h, y0 + patch_size)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
    src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)

    region = canvas[dst_y0:dst_y1, dst_x0:dst_x1]
    np.maximum(region, sprite[src_y0:src_y1, src_x0:src_x1], out=region)


def render_flake_field(
    canvas: np.ndarray,
    flakes: dict[str, np.ndarray],
    blades: int,
    blade_rotation: float,
    rim_gain: float,
    cats_eye: float,
    frame_centre: tuple[float, float],
    gain: np.ndarray | None = None,
) -> None:
    """Max-blend every flake of a field into ``canvas``.

    One call per flake to :func:`_paste_flake_sprite` is several thousand
    Python calls a frame, each of which spends more time setting numpy up
    than computing: a 15-pixel sprite cost what a 150-pixel one did. The
    compiled kernel draws the whole field in one call and is used when it is
    installed; the loop below is the same picture without it.
    """
    xs, ys = flakes["x"], flakes["y"]
    radii, blurs, smears = flakes["radius"], flakes["blur_radius"], flakes["smear"]
    dir_xs, dir_ys = flakes["dir_x"], flakes["dir_y"]
    if _native.render_flakes(
        canvas, xs, ys, radii, blurs, smears, dir_xs, dir_ys, gain,
        blades, blade_rotation, rim_gain, cats_eye, frame_centre, fast=mode.fast(),
    ):  # fmt: skip
        return
    for i in range(len(xs)):
        _paste_flake_sprite(
            canvas,
            float(xs[i]),
            float(ys[i]),
            float(radii[i]),
            float(blurs[i]),
            float(smears[i]),
            (float(dir_xs[i]), float(dir_ys[i])),
            blades,
            blade_rotation,
            rim_gain,
            cats_eye,
            frame_centre,
            gain=1.0 if gain is None else float(gain[i]),
        )


# Liquid-equivalent snowfall rate at which the light is most of the way to
# fully overcast. Snow of any note falls from a closed sky.
_OVERCAST_HALF_RATE_MM_H = 0.8

# Size of the patches snow is cleared from, in metres, and how hard their
# edge is: 1 is the noise as it comes, soft all over.
_BARE_PATCH_SCALE_M = 3.0
_BARE_PATCH_CONTRAST = 2.5

# Local contrast of luminance, as a fraction of its level, between which
# vegetation goes from lawn to canopy.
_CANOPY_RELIEF = (0.18, 0.36)

# Size of the unevenness in a thin covering, in metres, and its depth as a
# share of the covering, for the thinnest one. Deep snow is even.
_DUSTING_SCALE_M = 0.35
_DUSTING_GRAIN = 0.2

# The flake that stands for the veil's: the mean melted diameter weighted
# by what each flake takes out of a beam, in units of one over the
# distribution's slope. It sets how far the far snow's speckle is smeared.
_VEIL_FLAKE_SLOPES = 3.0
# How much of what is under settled snow it hides.
_SNOW_OPACITY = 0.95

# Widest level patch, as a fraction of the diagonal, taken for the top of a
# vehicle and not for ground.
_VEHICLE_WIDTH = 0.09

# What the scene's labels say about where snow lies, where the provider has
# them: what is a vehicle, what nothing settles on, what is a tree's crown
# and what is a lawn. And how soft a labelled region's edge is taken to be,
# as a fraction of the diagonal.
# What is on a road and is part of it for finding its sides: a road with its
# traffic cut out of it has a hole wherever a vehicle is.
_ON_THE_ROAD = (SceneClass.VEHICLE, SceneClass.PERSON)
# Wheel tracks. Half the width of one track as a share of its lane, and the
# share of the snow in it that is gone, and how much of a track is left
# where the road has been driven over least, each from the least traffic to
# the most; the scales, in metres, over which a track strays and over which it
# is worn more and less, and the octaves of the first; and how much darker
# the road is where snow has been worn off it, being wet.
# A rut drifts over tens of metres, as a driver's line does. It strayed
# over four, with finer octaves on top, and seen from a few tens of metres
# off, where four metres of road are a few rows of pixels, a thin track
# was a zigzag that read as hatching across the lanes.
_TRACK_BAND = (0.045, 0.2)
_TRACK_CLEARED = (0.45, 0.97)
_TRACK_WEAR_FLOOR = (0.2, 0.9)
_TRACK_RAGGED_M = 25.0
_TRACK_RAGGED_OCTAVES = 2
_TRACK_WORN_M = 20.0
_TRACK_WETTING = 0.35

_NAMED: dict[str, tuple[SceneClass, ...]] = {
    "vehicle": (SceneClass.VEHICLE,),
    "bare": (SceneClass.PERSON, SceneClass.SKY, SceneClass.WATER),
    "canopy": (SceneClass.FOLIAGE,),
    "lawn": (SceneClass.GRASS,),
}
_LABEL_EDGE = 0.003

# Share of a tree's lit foliage that carries snow when the ground is covered.
_CANOPY_COVER = 0.7

# The working sigma of the local-mean blur that canopy texture is measured
# against, in fast mode; see `wide_blur`. Coarser than its default of 8 so
# that the blur, at a hundredth of the diagonal, is reduced at all.
_LOCAL_MEAN_WORK_SIGMA = 4.0

# Least light settled snow is taken to be under, as a share of the frame's
# own estimate of its illuminant: at night, and by day.
_SHADING_FLOOR = (0.35, 0.9)

# Saturation of the frame's mean colour from which its light is taken to be
# coloured — sodium lamps — and at which settled snow takes all of the cast.
_CAST_SATURATION = (0.3, 0.6)

class RandomSnow(CameraTransform):
    """Physically-parameterised falling snow with depth-derived accumulation.

    With a provider that knows more than depth
    (:mod:`surmentations.perception`): the field of view and the pitch
    are the scene's camera's; the veil is over real metres where the
    camera's height is known; and snow settles by name as well as by shape,
    on a labelled vehicle as ``vehicle_cover_range`` says, on a lawn as on
    any ground, on a tree's crown as a dusting, and not on a person, the sky
    or open water. Without one, nothing here changes.

    Args:
        depth_provider: shared :class:`~surmentations.depth.provider.SharedDepthProvider`.
        flake_count_scale: multiplier on the flake count, which is otherwise
            *derived* rather than chosen: the number of flakes in frame is
            the Gunn-Marshall number density above
            ``flake_diameter_range_mm[0]`` times the volume of the simulated
            column, so it follows from the snowfall rate and the camera
            geometry instead of being a knob that can contradict them. Use
            this only to trade fidelity for speed, or for deliberate
            exaggeration.
        max_flakes: hard cap on the rendered flake count, so an extreme
            snowfall rate or a very deep column cannot make a single call
            unboundedly slow.
        flake_diameter_range_mm: truncation window for the flake size
            distribution, in **melted-equivalent** millimetres — the size of
            the water drop a flake would melt into, which is the unit every
            gauge-derived precipitation distribution is stated in. Sizes
            inside it follow the Gunn-Marshall exponential law rather than
            being uniform, so a frame gets many small flakes and a few large
            ones. The flake's real optical size is larger; see
            ``snow_density_g_cm3``.
        snow_density_g_cm3: bulk density of the falling snow. A snowflake is
            mostly air, so its physical diameter is
            ``D_melted * (rho_water / rho_snow)^(1/3)`` — about 2.3x at the
            default, and therefore five times the projected area. Lower
            (0.05) is fresh dry dendrites, higher (0.2) is wet compacted
            snow.
        min_flake_width_px: flakes projecting narrower than this are not
            drawn individually — they are what the veil represents. The cut
            is on *projected* width rather than on diameter, because a large
            distant flake and a small near one can subtend the same angle;
            cutting on diameter alone discards exactly the near flakes that
            carry the visible texture.
        snowfall_rate_mm_h: precipitation rate in mm/h of melted water
            equivalent, and the single physical intensity knob. It sets the
            *slope* of the size distribution (heavier snowfall lengthens the
            tail of large aggregates), the number density and hence how many
            flakes are in frame, and — through the visibility relation for
            dry snow — how dense the veil is.

            Worth knowing: past moderate rates the flake *count* saturates
            while sizes keep growing, because a flatter slope trades many
            small flakes for fewer large ones. Heavier snow therefore looks
            like bigger flakes and a thicker veil, not like more dots.
        distance_range_m: near and far bounds of the column of air in front
            of the lens that is simulated, in metres. Flakes fill it
            uniformly *by volume*, so most land near the far bound.

            The default stops at a couple of metres deliberately. At any
            sane resolution a flake further away than that is well under a
            pixel across, so drawing it individually costs a sprite to
            deposit a sub-pixel smudge; everything beyond this column is
            represented by the veil instead. That division of labour is why
            the counts here are in the hundreds rather than the tens of
            thousands a physically complete column would need.
        fall_speed_scale: multiplier on the Locatelli & Hobbs terminal
            velocity, for wind-driven snow that crosses the frame faster
            than it falls. 1 is still air.
        exposure_time_range: shutter time in seconds. Together with the fall
            speed this sets how far a flake travels during the exposure, and
            therefore how much it streaks.
        fall_angle_range: slant angle in degrees from vertical (wind).
        flutter_range: how far an individual flake's path deviates from that
            slant, in degrees. Snowflakes tumble and swing rather than
            falling on rails; 0 gives an unnaturally combed field.
        horizontal_fov_deg: camera field of view, used to project physical
            sizes and distances into pixels and to back-project the depth
            map into surface normals.
        aperture_diameter_range_mm: physical diameter of the lens aperture.
            This is what sets the bokeh: the circle of confusion is
            ``f * A * |1/z - 1/z_focus|``, so a wide aperture turns nearby
            flakes into large soft translucent discs while a small one (a
            typical surveillance camera) keeps the whole column sharp.
        focus_distance_m: what the camera is focused on. The default is far
            into the scene, which is the dashcam/surveillance case and the
            reason flakes on the front element are the defocused ones.
        min_visibility_scene: the least visibility the veil may leave, as a
            multiple of the scene's depth, or ``None`` for no limit. The
            veil follows from the precipitation rate, per metre, which is
            right and which over a long view hides the far end of the
            scene outright: 1.5 lets the veil thin the distance without
            ever closing it. The scene's depth is the distance of the
            farthest thing in it that is not sky.
        veil_beta_range: extinction coefficient for the snow veil. ``None``
            (the default) derives it from ``snowfall_rate_mm_h``, which is
            the physically consistent choice — the veil is the flakes too
            small to draw individually, so the two cannot be set apart
            without contradiction. Snow scatters roughly 7x more than rain at
            the same precipitation rate, so heavy snowfall drops visibility
            hard; this is the single biggest cue that it is *snowing* rather
            than that white dots have been pasted on.
        aperture_blades: number of lens aperture blades. Defocused flakes are
            images of the aperture, so they take its polygonal shape; 0 gives
            perfectly circular bokeh.
        rim_gain_range: brightness of a bokeh disc's rim relative to its
            centre (>1 = bright-rimmed, as most modern fast lenses render).
        cats_eye_strength: how strongly off-axis bokeh is clipped into
            lemon shapes by optical vignetting.
        accumulation_probability: chance of rendering settled snow. Not
            tied to the snowfall rate: what lies on the ground is how long
            it has been snowing and how cold the ground was, so a blizzard
            over bare tarmac and a flurry over deep snow are both ordinary.
            ``GROUND_SEVERITY`` in ``presets/severity.py`` is a ladder for
            this and the strength on their own, from bare to buried.
        accumulation_strength_range: how strongly upward-facing surfaces are
            pushed toward the settled-snow colour: 0.1 is a dusting the road
            shows through, 1 a coat that hides it. The default spans both.
        camera_pitch_deg: how far the camera is tilted **down** from level,
            which is how world "up" is oriented relative to the image. The
            usual mounting for a surveillance camera is a few degrees down.
        vehicle_cover_range: how much of the settled snow also lies on
            things small enough to be vehicles, drawn per call. 0 is moving
            traffic, which carries none; 1 is a car park after a night's
            snow. A level patch a car's width across is taken for a
            vehicle, since nothing else tells a roof from the road.
        accumulation_patchiness: how unevenly the settled snow lies. 0 is a
            perfectly even coat; higher leaves thin patches where it has
            blown clear or melted. Kept low by default: the patches are
            noise on the ground, not tyre tracks, and at high values a
            covered road reads as cloud.
        distant_snow_range: how much of its speckle the snow beyond the
            drawn flakes keeps, drawn per call. The flakes are those within
            ``distance_range_m``; the rest is a veil, and at 0 the veil is
            an even grey, as fog is, which it was before. At 1 it varies as
            a count of flakes does: a plain speckle just beyond the last
            flake drawn, where a pixel looks through a handful of them,
            finer and fainter with distance, and smooth where there are
            thousands. Each distance's share of it is in front of only
            what lies beyond that distance, so the snow has the scene's
            depth in it; and over that the far veil hangs in sheets. See
            :func:`~surmentations.utils.precipitation.far_field_texture`.
        wheel_track_range: how far traffic has worn the settled snow, drawn
            per call. Wheels run in the same two lines down every lane, and
            a snowed-in road that is driven on shows them before it shows
            anything else: 0 is untouched snow, 0.3 two dark ruts a tyre
            wide in each lane, 1 lanes worn bare with snow left between
            them and along the verges. The tracks follow the road, bends
            included, and pass under the vehicles on it, whose positions
            are where the lanes are read from
            (:mod:`surmentations.utils.road`). They are drawn only where
            the picture holds a road with two sides to follow, and only
            with scene labels on the
            provider (:mod:`surmentations.perception.labels`), which say
            where the road and its vehicles are: without them there is no
            road to follow and no tracks are drawn.
        accumulation_falloff: exponent applied to the cosine term. The
            cosine alone is the *rate* at which snow arrives, but whether it
            **stays** has a threshold — snow slides off anything much past
            45 degrees — so raising it to a power concentrates the coat on
            near-horizontal surfaces instead of giving a 60-degree wall half
            a covering. 1 disables the effect and uses the bare cosine.

        overcast_range: how much of the light that goes with this much
            snow the frame is given, drawn per call. Precipitation falls
            from cloud, so by default the scene is relit towards overcast in
            step with the rate: contrast and saturation fall and any sky in
            view turns to cloud. 1 is the full change for the rate, 0 leaves
            the light as photographed — right for frames that already are
            overcast. It cannot remove shadows the photograph has; it softens
            a sunny scene and does not convert it. See ``utils/overcast.py``.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``RandomSnow`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """
        depth_provider: SharedDepthProvider
        flake_count_scale: float = Field(ge=0.0, le=20.0)
        max_flakes: int = Field(ge=0, le=200000)
        flake_diameter_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 30.0)),
        ]
        snowfall_rate_range_mm_h: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.02, 100.0)),
        ]
        snow_density_range_g_cm3: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.005, 0.6)),
        ]
        min_flake_width_px: float = Field(ge=0.0, le=20.0)
        distance_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 200.0)),
        ]
        fall_speed_scale_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.05, 20.0)),
        ]
        exposure_time_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1e-5, 1.0)),
        ]
        fall_angle_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-60.0, 60.0)),
        ]
        flutter_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 90.0)),
        ]
        horizontal_fov_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(5.0, 175.0)),
        ]
        aperture_diameter_range_mm: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.1, 100.0)),
        ]
        focus_distance_range_m: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.2, 1000.0)),
        ]
        veil_beta_range: (
            Annotated[
                tuple[float, float],
                AfterValidator(nondecreasing),
                AfterValidator(check_range_bounds(0.0, 2.0)),
            ]
            | None
        )
        min_visibility_scene: float | None = Field(default=None, gt=0.0, le=100.0)
        aperture_blades_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(3, 12)),
        ]
        overcast_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        rim_gain_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.5, 2.0)),
        ]
        cats_eye_strength_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        accumulation_probability: float = Field(ge=0.0, le=1.0)
        accumulation_strength_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        camera_pitch_range_deg: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-89.0, 89.0)),
        ]
        vehicle_cover_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        accumulation_patchiness_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        wheel_track_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        distant_snow_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 2.0)),
        ]
        accumulation_falloff_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 12.0)),
        ]

    def __init__(
        self,
        depth_provider: SharedDepthProvider,
        flake_count_scale: float = 1.0,
        max_flakes: int = 9000,
        flake_diameter_range_mm: tuple[float, float] = (1.0, 15.0),
        snowfall_rate_range_mm_h: tuple[float, float] = (0.8, 8.0),
        snow_density_range_g_cm3: tuple[float, float] = (0.035, 0.11),
        min_flake_width_px: float = 1.5,
        distance_range_m: tuple[float, float] = (0.12, 5.0),
        fall_speed_scale_range: tuple[float, float] = (0.7, 1.8),
        exposure_time_range: tuple[float, float] = (0.001, 0.005),
        fall_angle_range: tuple[float, float] = (-12.0, 12.0),
        flutter_range: tuple[float, float] = (8.0, 30.0),
        horizontal_fov_range_deg: tuple[float, float] = (48.0, 78.0),
        aperture_diameter_range_mm: tuple[float, float] = (4.0, 14.0),
        focus_distance_range_m: tuple[float, float] = (10.0, 45.0),
        veil_beta_range: tuple[float, float] | None = None,
        min_visibility_scene: float | None = None,
        aperture_blades_range: tuple[int, int] = (5, 9),
        rim_gain_range: tuple[float, float] = (0.95, 1.45),
        cats_eye_strength_range: tuple[float, float] = (0.15, 0.7),
        accumulation_probability: float = 0.4,
        accumulation_strength_range: tuple[float, float] = (0.1, 1.0),
        camera_pitch_range_deg: tuple[float, float] = (0.0, 22.0),
        accumulation_patchiness_range: tuple[float, float] = (0.0, 0.3),
        wheel_track_range: tuple[float, float] = (0.0, 0.9),
        vehicle_cover_range: tuple[float, float] = (0.0, 0.6),
        accumulation_falloff_range: tuple[float, float] = (2.0, 5.0),
        overcast_range: tuple[float, float] = (0.7, 1.0),
        distant_snow_range: tuple[float, float] = (0.7, 1.2),
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring.

        Raises ``ValueError`` without a ``depth_provider``: the one
        :class:`~surmentations.depth.provider.SharedDepthProvider` of the
        pipeline must be passed in, so that depth is estimated once per frame.
        """
        super().__init__(p=p)
        self.depth_provider = require_depth_provider(depth_provider, cls_name=type(self).__name__)
        self.flake_count_scale = flake_count_scale
        self.max_flakes = max_flakes
        self.flake_diameter_range_mm = flake_diameter_range_mm
        self.snowfall_rate_range_mm_h = snowfall_rate_range_mm_h
        self.snow_density_range_g_cm3 = snow_density_range_g_cm3
        self.min_flake_width_px = min_flake_width_px
        self.distance_range_m = distance_range_m
        self.fall_speed_scale_range = fall_speed_scale_range
        self.exposure_time_range = exposure_time_range
        self.fall_angle_range = fall_angle_range
        self.flutter_range = flutter_range
        self.horizontal_fov_range_deg = horizontal_fov_range_deg
        self.aperture_diameter_range_mm = aperture_diameter_range_mm
        self.focus_distance_range_m = focus_distance_range_m
        self.veil_beta_range = veil_beta_range
        self.min_visibility_scene = min_visibility_scene
        self.aperture_blades_range = aperture_blades_range
        self.rim_gain_range = rim_gain_range
        self.cats_eye_strength_range = cats_eye_strength_range
        self.accumulation_probability = accumulation_probability
        self.accumulation_strength_range = accumulation_strength_range
        self.camera_pitch_range_deg = camera_pitch_range_deg
        self.accumulation_patchiness_range = accumulation_patchiness_range
        self.wheel_track_range = wheel_track_range
        self.vehicle_cover_range = vehicle_cover_range
        self.accumulation_falloff_range = accumulation_falloff_range
        self.overcast_range = overcast_range
        self.distant_snow_range = distant_snow_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the snow for one frame: the weather, the optics, every flake, and the ground.

        The number of flakes follows from the snowfall rate, the size cut-off
        and the volume of the column in view; past the budget the width
        threshold rises rather than the field being truncated. Returns the
        ``flakes`` with the aperture's shape (``blades``, ``cats_eye``,
        ``rim_gain``), the veil's transmission, and the ``settled`` snow laid
        on level ground, kept off foliage and sky.
        """
        scene = scene_of(data)
        height, width = params["shape"][:2]
        rng = self.random_generator

        relative_depth = self.depth_provider.get_depth(scene)
        depth_pm = self.depth_provider.to_metric(relative_depth)
        linear_scene, _scale, _is_integer = to_linear_unit(first_frame(scene))
        labels = self.depth_provider.get_labels(scene)
        sky = sky_to_depth_edge(
            sky_weight(linear_scene, relative_depth, labels), relative_depth, linear_scene
        )

        # Every physical quantity is drawn per call. A scalar here would make
        # a whole dataset share one snowfall rate, one snow type and one
        # camera geometry, which is the opposite of what an augmentation is
        # for.
        snowfall_rate = self.py_random.uniform(*self.snowfall_rate_range_mm_h)
        snow_density = self.py_random.uniform(*self.snow_density_range_g_cm3)
        horizontal_fov = self.py_random.uniform(*self.horizontal_fov_range_deg)
        focus_distance = self.py_random.uniform(*self.focus_distance_range_m)
        camera_pitch = self.py_random.uniform(*self.camera_pitch_range_deg)
        # Where the provider knows how the camera is mounted, the scene is
        # seen through that camera and not through this draw of one.
        horizontal_fov, camera_pitch = self.depth_provider.settle_view(
            scene, horizontal_fov, camera_pitch
        )
        fall_speed_scale = self.py_random.uniform(*self.fall_speed_scale_range)
        accumulation_falloff = self.py_random.uniform(*self.accumulation_falloff_range)
        vehicle_cover = self.py_random.uniform(*self.vehicle_cover_range)
        accumulation_patchiness = self.py_random.uniform(*self.accumulation_patchiness_range)

        focal = focal_length_px(width, height, horizontal_fov)

        # How many flakes belong in frame is a consequence of the weather and
        # the camera, not a free parameter: number density above the size
        # cut-off, times the volume of the column, times the fraction of
        # those that actually project wide enough to be worth drawing.
        slope = gunn_marshall_slope(snowfall_rate)
        size_scale = float(snow_physical_diameter(1.0, snow_density))
        density = snow_number_density(snowfall_rate, self.flake_diameter_range_mm[0])
        volume = frustum_volume(
            width, height, horizontal_fov, *self.distance_range_m
        )
        visible = resolvable_fraction(
            slope,
            *self.flake_diameter_range_mm,
            *self.distance_range_m,
            focal,
            self.min_flake_width_px,
            size_scale=size_scale,
        )
        population = density * volume * visible * self.flake_count_scale

        # When the budget binds, raise the width threshold rather than
        # truncating the field — the same rule rain follows, for the same
        # reason. Capping the count alone draws an arbitrary subset, so past
        # the cap a heavier snowfall stops looking heavier; at 1920 px and
        # severity 5 that was five draws in six, each losing about a fifth of
        # its flakes at random. Raising the cut keeps the flakes that are
        # drawn the widest ones and hands the rest to the veil, which is what
        # already represents everything too small to draw.
        width_cut = width_cut_for_budget(
            population,
            self.max_flakes,
            slope,
            *self.flake_diameter_range_mm,
            *self.distance_range_m,
            focal,
            self.min_flake_width_px,
            size_scale=size_scale,
        )
        count = int(min(self.max_flakes, round(population)))

        angle = self.py_random.uniform(*self.fall_angle_range)
        flutter = self.py_random.uniform(*self.flutter_range)
        aperture_m = self.py_random.uniform(*self.aperture_diameter_range_mm) * 1e-3
        exposure = self.py_random.uniform(*self.exposure_time_range)


        # Physical flake sizes and physical distances, projected through the
        # camera. Size, defocus and smear all derive from these two numbers,
        # which is what keeps them consistent with each other.
        melted_mm, distance_m = sample_resolvable_particles(
            rng,
            count,
            slope,
            *self.flake_diameter_range_mm,
            *self.distance_range_m,
            focal,
            width_cut,
            size_scale=size_scale,
        )
        count = len(melted_mm)
        # Gunn-Marshall is stated in melted-equivalent diameters. The flake
        # the camera sees is mostly air and therefore much larger.
        diameter_mm = snow_physical_diameter(melted_mm, snow_density)

        radius_px = 0.5 * focal * (diameter_mm * 1e-3) / distance_m

        # Thin-lens circle of confusion: c = f * A * |1/z - 1/z_focus|. This
        # is the actual defocus law rather than a "nearness" ramp, so the
        # aperture and the focus distance mean what they say — stopping down
        # genuinely deepens the depth of field, and a flake at the focal
        # distance genuinely renders sharp.
        blur_radius_px = (
            0.5
            * focal
            * aperture_m
            * np.abs(1.0 / distance_m - 1.0 / focus_distance)
        )

        # Distance travelled during the exposure, projected. Snow falls so
        # slowly that this is a short smear at typical shutter speeds — which
        # is exactly what distinguishes it from rain.
        speed = snow_fall_speed(diameter_mm) * fall_speed_scale
        smear_px = focal * speed * exposure / distance_m

        # Each flake tumbles, so its own path deviates from the prevailing
        # slant. Without this the field reads as combed.
        flake_angle = angle + rng.uniform(-flutter, flutter, count)
        angle_rad = np.deg2rad(flake_angle)

        flakes = {
            "x": rng.uniform(0.0, width, count).astype(np.float32),
            "y": rng.uniform(0.0, height, count).astype(np.float32),
            # Carried through for introspection and testing: it is the
            # quantity size, defocus and smear all derive from, and none of
            # them is a usable proxy for it on its own.
            "distance": distance_m.astype(np.float32),
            "radius": radius_px.astype(np.float32),
            "blur_radius": blur_radius_px.astype(np.float32),
            "smear": smear_px.astype(np.float32),
            # Direction of travel: the slant is measured from vertical, and
            # image y grows downward, so a falling flake moves along
            # (sin, cos).
            "dir_x": np.sin(angle_rad).astype(np.float32),
            "dir_y": np.cos(angle_rad).astype(np.float32),
        }

        draw_accumulation = self.py_random.random() < self.accumulation_probability
        accumulation_strength = (
            self.py_random.uniform(*self.accumulation_strength_range) if draw_accumulation else 0.0
        )

        settled = np.zeros((height, width), dtype=np.float32)
        road = on_road = None
        if draw_accumulation:
            # Lambert's cosine law for falling snow: what a surface collects
            # goes as the cosine of the angle between its normal and "up".
            cosine = upward_facing(
                depth_pm,
                horizontal_fov_deg=horizontal_fov,
                camera_pitch_deg=camera_pitch,
            )
            # The cosine is how fast snow *arrives*; staying put is a
            # threshold, since a coat slides off anything much steeper than
            # about 45 degrees. Without this a 60-degree wall keeps half a
            # covering and the whole frame washes flat white.
            as_read = level_ground_weight(cosine, accumulation_falloff)
            # A level patch the width of a car is a car. Whether snow
            # stays on it depends on whether it has been standing there.
            vehicles = small_level_areas(as_read, _VEHICLE_WIDTH)
            named: dict[str, np.ndarray] = {}
            if labels is not None:
                # What the network could name. A vehicle is a vehicle
                # whether or not its roof is an island of level surface,
                # and nothing settles on a person, the sky or open water.
                soften = max(1.0, _LABEL_EDGE * float(np.hypot(height, width)))
                for name, classes in _NAMED.items():
                    named[name] = class_weight(labels, classes, soften_px=soften)
                # The road and what is on it, by name, for the wheel tracks.
                on_road = class_weight(labels, _ON_THE_ROAD) > 0.5
                road = on_road | (labels == int(SceneClass.ROAD))
                vehicles = np.maximum(vehicles, named["vehicle"])
                as_read = as_read * (1.0 - named["bare"])
            # Without labels there is no road to follow. Level ground that
            # is not green was tried in its place: it takes in the pavements
            # and the verges, the islands of level surface that pass for
            # vehicles give lanes a third of their width, and the result was
            # a washboard across everything flat.
            level = fill_to_edges(as_read * (1.0 - vehicles), linear_scene)
            level = level + vehicle_cover * as_read * vehicles
            # Foliage has no surface to read a normal from, and from above a
            # canopy reads as level ground: covered evenly, it became a grey
            # smear with the tree gone. Grass is vegetation too and is
            # covered like any ground. What tells them apart is texture: a
            # canopy is light and shadow at every scale, a lawn is flat.
            sigma = 0.01 * float(np.hypot(height, width))
            luma = luminance(linear_scene)
            # The mean brightness around each pixel, which texture is measured
            # against. A blur of a hundredth of the diagonal, 14 pixels on a
            # 1024-pixel frame: in fast mode on a copy reduced to a working
            # sigma of 4, as lens grime is (`wide_blur`), 24 ms to 3.
            around = np.maximum(wide_blur(luma, sigma, work_sigma=_LOCAL_MEAN_WORK_SIGMA), 1e-4)
            relief = wide_blur(np.abs(luma / around - 1.0), 2.0 * sigma)
            # Green says vegetation by day. Under sodium lamps nothing is
            # green, and what is left to tell a tree by is that its "level
            # ground" comes and goes from one patch to the next, where a
            # road's is steady.
            steady = wide_blur(as_read, 2.0 * sigma)
            green = vegetation_weight(linear_scene)
            noise = None
            if accumulation_patchiness > 0.0:
                # Wind scours snow off exposed ground and it melts first
                # where the surface is warm: broad areas where it lies thin.
                noise = ground_noise(depth_pm, focal, camera_pitch, _BARE_PATCH_SCALE_M, rng)
            # A thin covering seen from a camera's distance is an even
            # greying of the ground: the snow lies in the grain of the
            # surface, far finer than a pixel. It is not quite even, though.
            # It is a little deeper in one square metre than the next, and
            # that unevenness, laid on the ground so it foreshortens with
            # the road, is what keeps a dusting from reading as a veil in
            # the air. (Cutting it into patches with edges was tried, and
            # looked like spilt paint at any scale.)
            grain = ground_noise(
                depth_pm, focal, camera_pitch, _DUSTING_SCALE_M, rng, octaves=4, uniform=True
            )
            thin = 1.0 - accumulation_strength
            done = None
            if mode.fast():
                # Everything from here to the snow that lies is a formula a
                # pixel: one pass where the compiled kernels are there, and
                # the twenty below where they are not.
                done = _native.snow_settled(
                    level,
                    relief,
                    steady,
                    green,
                    grain,
                    luma,
                    around,
                    sky,
                    [_CANOPY_RELIEF[0], float(np.ptp(_CANOPY_RELIEF)), accumulation_patchiness]
                    + [_BARE_PATCH_CONTRAST, accumulation_strength, _DUSTING_GRAIN * thin]
                    + [accumulation_strength * _CANOPY_COVER],
                    named.get("lawn"),
                    named.get("canopy"),
                    noise,
                )
            if done is not None:
                settled = done
            else:
                # `np.ptp` gives an np.float64, which would widen this field
                # and every one made from it down to the settled snow itself.
                relief_span = mode.scalar(np.ptp(_CANOPY_RELIEF))
                rough = np.clip((relief - _CANOPY_RELIEF[0]) / relief_span, 0.0, 1.0)
                # Comes and goes: a wall or the sky never reads as level at
                # all, and counting those drew snow along every skyline.
                unsteady = np.clip((0.85 - steady) / 0.25, 0.0, 1.0) * np.clip(
                    (steady - 0.15) / 0.2, 0.0, 1.0
                )
                canopy = rough * np.maximum(green, unsteady)
                if named:
                    # A tree is a canopy and a lawn is ground, by name, where
                    # texture and colour could only guess: by night, in leaf
                    # that is not green, on grass that is rough.
                    canopy = np.maximum(canopy * (1.0 - named["lawn"]), named["canopy"])
                ground = level * (1.0 - canopy)
                if noise is not None:
                    patches = np.clip((noise - 0.5) * _BARE_PATCH_CONTRAST + 0.5, 0.0, 1.0)
                    ground = ground * (1.0 - accumulation_patchiness * (1.0 - patches))
                uneven = 1.0 + _DUSTING_GRAIN * thin * (2.0 * grain - 1.0)
                settled = ground * accumulation_strength * uneven
                # On a canopy the snow lies on the clumps that stand proud,
                # which are the ones the light reaches.
                proud = np.clip((luma / around - 0.95) / 0.35, 0.0, 1.0)
                settled = settled + accumulation_strength * _CANOPY_COVER * canopy * proud
                # Nothing settles on the sky. Both steps above that look a
                # few pixels around — growing level ground to an object's
                # edge, and finding where "level" comes and goes — reach
                # past a roofline into the sky beside it, and drew a pale
                # line round the skyline.
                settled = settled * (1.0 - sky)

        if self.veil_beta_range is not None:
            veil_bounds = self.veil_beta_range
        else:
            beta = snow_extinction_coefficient(snowfall_rate)
            veil_bounds = (beta * 0.75, beta * 1.3)
        veil_beta = self.py_random.uniform(*veil_bounds)
        if self.min_visibility_scene is not None:
            # Never thicker than leaves the far end of this scene in view.
            floor = self.min_visibility_scene * scene_extent_m(depth_pm, sky)
            veil_beta = min(veil_beta, extinction_for_visibility(floor))

        record_draw(
            snowfall_rate_mm_h=snowfall_rate,
            snow_density_g_cm3=snow_density,
            melted_to_physical=size_scale,
            horizontal_fov_deg=horizontal_fov,
            focus_distance_m=focus_distance,
            camera_pitch_deg=camera_pitch,
            fall_speed_scale=fall_speed_scale,
            exposure_s=exposure,
            aperture_mm=aperture_m * 1e3,
            gm_slope=slope,
            focal_px=focal,
            volume_m3=volume,
            number_density_per_m3=density,
            resolvable_fraction=visible,
            population=population,
            width_cut_px=width_cut,
            count_requested=count,
            veil_beta=veil_beta,
            accumulation_falloff=accumulation_falloff,
            accumulation_patchiness=accumulation_patchiness,
            vehicle_cover=vehicle_cover,
        )

        drawn = {
            "flakes": flakes,
            "blade_rotation": self.py_random.uniform(0, 2 * np.pi),
            "blades": self.py_random.randint(*self.aperture_blades_range),
            "cats_eye": self.py_random.uniform(*self.cats_eye_strength_range),
            "rim_gain": self.py_random.uniform(*self.rim_gain_range),
            # The depth of air in front of each pixel for now: the veil is
            # made of it at the end, once its speckle is drawn.
            "veil_transmission": lift_sky(depth_pm, sky),
            "sky": sky,
            "overcast": self.py_random.uniform(*self.overcast_range)
            * cloud_cover(snowfall_rate, _OVERCAST_HALF_RATE_MM_H),
            "settled": settled.astype(np.float32),
            "accumulation_strength": accumulation_strength,
        }

        # Wheel tracks, last: what is drawn for them comes after every other
        # draw, so that a frame with none is the frame it was before there
        # were any.
        traffic = self.py_random.uniform(*self.wheel_track_range)
        cleared = None
        if road is not None and traffic > 0.0:
            carriageway = labels == int(SceneClass.ROAD) if labels is not None else road
            standing_on = np.zeros(road.shape, dtype=bool) if on_road is None else on_road
            named_vehicles = None if labels is None else labels == int(SceneClass.VEHICLE)
            camera = self.depth_provider.settle_camera(scene)
            read = read_road(
                linear_scene,
                carriageway,
                standing_on,
                named_vehicles,
                rng,
                horizon_row=self.depth_provider.horizon_row(scene),
                camera_height_m=None if camera is None else camera.height_m,
            )
            if read is not None:
                frame, lanes = read
                ragged = ground_noise(
                    depth_pm,
                    focal,
                    camera_pitch,
                    _TRACK_RAGGED_M,
                    rng,
                    octaves=_TRACK_RAGGED_OCTAVES,
                )
                worn = ground_noise(depth_pm, focal, camera_pitch, _TRACK_WORN_M, rng)
                low, high = _TRACK_BAND
                band = low + (high - low) * traffic
                low, high = _TRACK_WEAR_FLOOR
                floor = low + (high - low) * traffic
                tracks = wheel_tracks(frame, lanes, band, rng, ragged, worn, floor)
                # On the road, not on what stands on it: the lanes run
                # under the vehicles, and the snow on a roof has no ruts.
                standing = np.zeros(road.shape, np.float32) if on_road is None else on_road
                tracks = tracks * (1.0 - cv2.GaussianBlur(standing.astype(np.float32), (0, 0), 1.0))
                low, high = _TRACK_CLEARED
                cleared = tracks * np.float32(low + (high - low) * traffic)
                # How much snow went from each pixel, which is how much of
                # the wet road under it now shows.
                drawn["tracks"] = (cleared * np.clip(settled, 0.0, 1.0)).astype(np.float32)
                drawn["settled"] = (settled * (1.0 - cleared)).astype(np.float32)
        # The snow too far off to draw flake by flake, last of all the
        # draws: not an even grey but the speckle of a count of flakes, in
        # front of each thing by as much as there is air in front of it.
        distant = self.py_random.uniform(*self.distant_snow_range)
        distant_seed = self.py_random.randint(0, 2**31 - 1)
        thickness = np.float32(veil_beta) * drawn["veil_transmission"]
        if distant > 0.0 and veil_beta > 0.0:
            slope = gunn_marshall_slope(snowfall_rate)
            falls = float(snow_fall_speed(_VEIL_FLAKE_SLOPES / slope)) * fall_speed_scale
            thickness *= far_field_texture(
                drawn["veil_transmission"],
                np.random.default_rng(distant_seed),
                snow_number_density(snowfall_rate, 0.0),
                focal,
                self.distance_range_m[1],
                focal * falls * exposure,
                angle,
                distant,
            )
        np.negative(thickness, out=thickness)
        drawn["veil_transmission"] = np.exp(thickness, out=thickness).astype(
            np.float32, copy=False
        )
        record_draw(
            distant_snow=distant,
            wheel_tracks=traffic,
            wheel_tracks_drawn=cleared is not None,
            road_share=lambda: 0.0 if road is None else float(road.mean()),
        )
        return drawn

    def apply(
        self,
        img: np.ndarray,
        flakes: dict[str, np.ndarray],
        blade_rotation: float,
        blades: int,
        cats_eye: float,
        rim_gain: float,
        veil_transmission: np.ndarray,
        settled: np.ndarray,
        accumulation_strength: float,
        sky: np.ndarray | None = None,
        overcast: float = 0.0,
        tracks: np.ndarray | None = None,
        **params: Any,
    ) -> np.ndarray:
        """Relight the frame for overcast, veil it, lay the settled snow, and draw the flakes last.

        In linear light. Flakes take the illuminant's radiance and the veil
        the sky's; settled snow is lit by the scene's own brightness around
        it. The flakes are in the air near the lens and are drawn over
        everything else.
        """
        height, width = img.shape[:2]
        frame_centre = (width / 2.0, height / 2.0)

        flake_layer = np.zeros((height, width), dtype=np.float32)
        render_flake_field(
            flake_layer, flakes, blades, blade_rotation, rim_gain, cats_eye, frame_centre
        )

        linear, scale, is_integer = to_linear_unit(img)
        # Snow falls from cloud; relight the scene before anything is lit by it.
        linear = apply_overcast(linear, overcast, sky)

        # Flakes are ice lit by, and imaging, the same sky as the scene — so
        # they are composited toward the scene's own sky radiance, not toward
        # white. That makes them disappear into a bright overcast sky and
        # stand out against dark trees, exactly as real snow does, with no
        # special-casing.
        # A flake is a directly-lit object in front of the scene, so its
        # radiance is the *illuminant* — which the frame may not contain, if
        # no sky is in view. The veil is different: it is scattered skylight
        # accumulated along the path, so it saturates at the sky's own
        # recorded radiance and must not be lifted the same way, or a bright
        # sky blows out to flat white.
        flake_radiance = illuminant_radiance_linear(linear)
        airlight = veil_radiance_linear(linear, sky)

        # Heavy snow scatters hard: the veil is most of what says "snowing".
        # The veil now and the flakes at the very end, each one compiled
        # pass where there is one, with the arithmetic of the expressions
        # beside them in their order: the same floats.
        #
        # The order is that of the scene, from the far end to the lens. The
        # veil is the snow too far off to resolve; the snow that has settled
        # and the tracks worn in it are on the ground; the flakes are in the
        # air near the lens, in front of all of it. They used to be drawn
        # with the veil, and settled snow was then laid over them and lit by
        # a frame that had them in it.
        blended = _native.veil_blend(linear, veil_transmission, airlight)
        if blended is None:
            t_veil = veil_transmission[..., None]
            out = linear * t_veil + airlight[None, None, :] * (1.0 - t_veil)
        else:
            out = blended

        if accumulation_strength > 0.0:
            snow_albedo = srgb_to_linear(
                (_ACCUMULATION_COLOR / 255.0).reshape(1, 1, 3).astype(np.float32)
            )
            # Settled snow replaces a surface's *albedo*, not its
            # illumination. Compositing toward a constant white throws the
            # scene's lighting away and flattens everything it touches into
            # one dead tone; a real snow-covered road still has its shadows,
            # its bright and dull patches, and whatever the sky is doing to
            # it. So the snow is lit by a low-frequency estimate of the local
            # illumination: the light falling on a patch is about what the
            # patch sends back divided by a typical albedo, and snow returns
            # nearly all of it.
            #
            # That is an absolute estimate, and has to be. Normalising by the
            # frame's own mean instead, as this did, renders an averagely-lit
            # patch at full white whatever the frame — so at night settled
            # snow came out as a daylit floor under a black sky.
            illumination = (
                wide_blur(
                    luminance(out),
                    max(2.0, 0.02 * max(height, width)),
                    work_sigma=_LOCAL_MEAN_WORK_SIGMA,
                )
                / TYPICAL_SCENE_ALBEDO
            )
            # A surface darker than typical understates the light on it:
            # new asphalt, at an albedo of 0.05, says a quarter of the
            # truth, and snow shaded by that is grey. The frame's own
            # illuminant, read so that lamps do not count, puts a floor
            # under it.
            #
            # By day the floor is nearly all of the light, which costs the
            # snow most of the shadows that fall on it. A dark patch of
            # ground is either in shadow or dark, and nothing here can tell
            # which: reading the light off the brightest thing nearby was
            # tried, and stamped a pale disc round every headlight. At night
            # the light is pools under lamps, and the floor stays low or
            # snow glows where none reaches.
            ambient = float(luminance(airlight.reshape(1, 1, -1)).ravel()[0])
            daylight = float(np.clip((ambient - 0.1) / 0.2, 0.0, 1.0))
            low, high = _SHADING_FLOOR
            floor = (low + (high - low) * daylight) * ambient
            shading = np.clip(np.maximum(illumination, floor), 0.0, 1.15)[..., None]

            # White under white light. Under sodium lamps the whole frame is
            # orange and so is the snow.
            mean_colour = out.reshape(-1, out.shape[-1]).mean(axis=0)
            peak = max(float(mean_colour.max()), 1e-6)
            saturation = (peak - float(mean_colour.min())) / peak
            low, high = _CAST_SATURATION
            cast = float(np.clip((saturation - low) / (high - low), 0.0, 1.0))
            mean_luma = max(float(luminance(mean_colour.reshape(1, 1, -1)).ravel()[0]), 1e-6)
            tint = 1.0 + cast * (mean_colour / mean_luma - 1.0)

            if tracks is not None:
                # Where wheels have cleared the snow the road is wet with
                # it, and darker than it was dry.
                out = out * (1.0 - np.float32(_TRACK_WETTING) * tracks[..., None])
            # `settled` is already the share of each pixel under snow.
            coverage = np.clip(settled * _SNOW_OPACITY, 0.0, 1.0)[..., None]
            snow = snow_albedo * tint.reshape(1, 1, -1).astype(np.float32) * shading
            out = out * (1.0 - coverage) + snow * coverage

        # The flakes, last: in the air, in front of the ground and what lies on it.
        cover = np.clip(flake_layer, 0.0, 1.0)
        over = _native.veil_blend(
            out, np.ones(cover.shape, dtype=np.float32), airlight, cover, flake_radiance
        )
        if over is None:
            alpha = cover[..., None]
            out = out * (1.0 - alpha) + flake_radiance[None, None, :] * alpha
        else:
            out = over

        return from_linear_unit(out, scale, is_integer, img.dtype)
