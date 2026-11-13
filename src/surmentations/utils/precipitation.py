"""Precipitation microphysics shared by the rain and snow transforms.

Two things decide what a field of falling particles actually looks like, and
neither of them is a free tuning knob:

**How big the drops are.** Rain is not one drop size repeated. Marshall &
Palmer (1948) measured the drop-size distribution of stratiform rain and
found it exponential in diameter, ``N(D) = N0 exp(-Lambda D)``, with the
slope ``Lambda = 4.1 R^-0.21`` set entirely by the rain rate ``R`` in mm/h.
Heavier rain does not mean *more* drops of the same size so much as a
*shallower* slope — a longer tail of big drops. That is why drizzle reads as
a fine even mist and a downpour reads as a mix of fine texture and a few
heavy streaks, and it is why sampling drop sizes from a uniform range makes
every streak look like every other streak.

**How fast they fall.** A drop's terminal velocity is fixed by its size, and
so, given the exposure time, is the length of the streak it draws. Atlas &
Ulbrich's fit to Gunn & Kinzer's (1949) measurements,
``v = 9.40 (1 - exp(-1.57e3 D^1.15))`` with ``D`` in metres, tracks the
measured values within ~3% over the whole 0.5-5 mm range.

Coupling the two is what matters: a big drop is both *wider* and *faster*,
so it draws a longer streak, and its opacity ``alpha = D / (v T)`` follows
from the same two numbers rather than being chosen independently. Nothing
downstream gets to pick length, width and alpha separately, because in a
real photograph they are not separate.

**Where the particles are.** A falling drop hangs in the air *in front of*
the scene, so its distance has nothing to do with whatever the depth map
says is behind it. Sizing particles from the scene depth makes them look
pasted onto surfaces — big over the near road, tiny over the distant sky.
Instead each particle gets its own distance from the camera, sampled so the
particles fill the *volume* in the frustum: the count within distance ``d``
grows as ``d^3`` while apparent size falls as ``1/d``, which is what
produces a field of many tiny far particles punctuated by a few big near
ones instead of evenly-sized confetti.
"""

from __future__ import annotations

import cv2
import numpy as np

from surmentations import mode

# Marshall-Palmer slope: Lambda = MP_SLOPE_COEFF * R**MP_SLOPE_EXPONENT, in
# mm^-1 for a rain rate R in mm/h.
MP_SLOPE_COEFF = 4.1
MP_SLOPE_EXPONENT = -0.21

# Raindrops larger than this break up aerodynamically, so the measured
# distribution is truncated here rather than continuing its exponential tail.
MAX_RAINDROP_DIAMETER_MM = 6.0

# Gunn & Marshall (1958) slope for snow aggregates: 25.5 R^-0.48 cm^-1,
# i.e. 2.55 R^-0.48 mm^-1.
GM_SLOPE_COEFF = 2.55
GM_SLOPE_EXPONENT = -0.48


def marshall_palmer_slope(rain_rate_mm_h: float) -> float:
    """Exponential slope ``Lambda`` (mm^-1) of the drop-size distribution.

    Marshall & Palmer 1948. Heavier rain flattens the slope, which lengthens
    the tail of large drops.
    """
    return MP_SLOPE_COEFF * max(rain_rate_mm_h, 1e-3) ** MP_SLOPE_EXPONENT


#: Shape parameter of the gamma drop-size distribution (Ulbrich 1983). The
#: exponential of Marshall & Palmer is the ``mu = 0`` case; observed spectra
#: are consistently better fitted with ``mu`` around 2-4, which suppresses the
#: pile-up of very small drops an exponential predicts and moves the mode to a
#: finite diameter. Measured here at 50 mm/h, going from 0 to 3 raises the
#: share of drops above 2 mm from 11.5% to 14.8%.
#:
#: It changes the *statistics*, not how visible the rain is: the quantity that
#: decides that is the summed projected area, and at a fixed rain rate that
#: moves by 2.5% across the whole range of mu. See ``docs/rain_visibility.md``.
DEFAULT_DSD_SHAPE = 3.0


def gamma_dsd_slope(rain_rate_mm_h: float, shape_mu: float = DEFAULT_DSD_SHAPE) -> float:
    """Slope ``Lambda`` (mm^-1) of the gamma drop-size distribution.

    ``N(D) = N0 D^mu exp(-Lambda D)``. Pinned to Marshall & Palmer through the
    median volume diameter rather than by refitting: for a gamma distribution
    ``Lambda D0 = 3.67 + mu``, and the ``mu = 0`` case has to reproduce
    ``D0 = 3.67 / Lambda_MP``. So the same ``D0(R)`` serves every shape, and
    ``shape_mu = 0`` returns exactly :func:`marshall_palmer_slope`.
    """
    median_volume_diameter = 3.67 / marshall_palmer_slope(rain_rate_mm_h)
    return float((3.67 + shape_mu) / median_volume_diameter)


def _gamma_dsd_weights(diameter_mm: np.ndarray, slope: float, shape_mu: float) -> np.ndarray:
    """Unnormalised ``N(D)`` on a grid, guarding ``0^0`` at the origin."""
    safe = np.maximum(diameter_mm, 1e-6)
    return safe**shape_mu * np.exp(-slope * safe)


def sample_gamma_diameters(
    rng: np.random.Generator,
    count: int,
    slope: float,
    low: float,
    high: float,
    shape_mu: float = DEFAULT_DSD_SHAPE,
    grid: int = 1024,
) -> np.ndarray:
    """Sample ``N(D) ~ D^mu exp(-Lambda D)`` restricted to ``[low, high]``.

    The exponential case has a closed-form inverse CDF; the gamma one does not
    for arbitrary ``mu``, so the CDF is built on a grid and inverted by search.
    That is exact to the grid spacing, costs one ``searchsorted``, and unlike
    rejection sampling its cost does not depend on how much of the
    distribution the window cuts away — which matters because the window here
    routinely keeps only a few percent of it.
    """
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    return _diameters_from_uniform(rng.random(count), slope, low, high, shape_mu, grid)


def _diameters_from_uniform(
    u: np.ndarray, slope: float, low: float, high: float, shape_mu: float, grid: int = 1024
) -> np.ndarray:
    """The diameters :func:`sample_gamma_diameters` makes of the uniforms ``u``.

    Apart from the draw itself, so that a caller which can tell from a
    uniform that its particle will be discarded need not compute the rest.
    """
    if shape_mu <= 0.0:
        return _exponential_from_uniform(u, slope, low, high)
    edges = np.linspace(low, high, grid, dtype=np.float64)
    weights = _gamma_dsd_weights(edges, slope, shape_mu)
    cdf = np.cumsum(weights)
    total = cdf[-1]
    if total <= 0.0:  # pragma: no cover - degenerate slope/window
        return _exponential_from_uniform(u, slope, low, high)
    cdf /= total
    picked = np.searchsorted(cdf, u)
    return edges[np.clip(picked, 0, grid - 1)].astype(np.float32)


def _truncated_exponential(
    rng: np.random.Generator, count: int, slope: float, low: float, high: float
) -> np.ndarray:
    """Sample an exponential of rate ``slope`` restricted to ``[low, high]``.

    Both Marshall-Palmer (rain) and Gunn-Marshall (snow) are exponential in
    diameter and differ only in their slope, so one sampler serves both.

    Inverted analytically rather than rejection-sampled, so the cost does not
    depend on how much of the distribution the window cuts away. The lower
    bound is not a truncation of the model: particles below it are what the
    *veil* layer represents, so the cut is a division of labour between the
    two layers.
    """
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    return _exponential_from_uniform(rng.random(count), slope, low, high)


def _exponential_from_uniform(u: np.ndarray, slope: float, low: float, high: float) -> np.ndarray:
    """The inverse CDF of :func:`_truncated_exponential`, at the uniforms ``u``."""
    span = max(high - low, 1e-6)
    saturation = -np.expm1(-slope * span)
    values = low - np.log1p(-u * saturation) / slope
    return np.clip(values, low, high).astype(np.float32)


def gunn_marshall_slope(snowfall_rate_mm_h: float) -> float:
    """Exponential slope ``Lambda`` (mm^-1) of the snow aggregate size distribution.

    Gunn & Marshall 1958, the snow counterpart of Marshall-Palmer:
    ``Lambda = 25.5 R^-0.48`` cm^-1 for a precipitation rate ``R`` in mm/h of
    melted water equivalent, converted here to mm^-1.
    """
    return GM_SLOPE_COEFF * max(snowfall_rate_mm_h, 1e-3) ** GM_SLOPE_EXPONENT


def raindrop_terminal_velocity(diameter_mm: np.ndarray | float) -> np.ndarray:
    """Terminal fall speed in m/s for a raindrop ``diameter_mm`` across.

    Atlas & Ulbrich's fit to the Gunn & Kinzer (1949) measurements,
    ``v = 9.40 (1 - exp(-1.57e3 D^1.15))`` for ``D`` in metres. Matches the
    measured table within ~3% from 0.5 mm to 5 mm.
    """
    diameter_m = np.asarray(diameter_mm, dtype=np.float32) * 1e-3
    return 9.40 * (-np.expm1(-1.57e3 * diameter_m**1.15))


def focal_length_px(width: int, height: int, horizontal_fov_deg: float) -> float:
    """Pinhole focal length in pixels for a given horizontal field of view.

    Needed to turn a physical size (mm) at a physical distance (m) into an
    apparent size in pixels, which is the only way particle sizes can be
    resolution-independent *and* physically meaningful at the same time.
    """
    del height  # the horizontal FOV alone fixes the focal length
    half_fov = np.deg2rad(horizontal_fov_deg) / 2.0
    return float(width / (2.0 * np.tan(half_fov)))


def volume_uniform_distances(
    rng: np.random.Generator, count: int, near_m: float, far_m: float
) -> np.ndarray:
    """Distances (m) of ``count`` particles filling a frustum uniformly by volume.

    The frustum's cross-section grows as ``d^2``, so the number of particles
    closer than ``d`` grows as ``d^3``. Inverting that CDF gives
    ``d = (near^3 + u (far^3 - near^3))^(1/3)``, which puts most particles far
    away and a handful close — the distribution that makes a precipitation
    field read as a volume rather than as a flat scatter of equal dots.
    """
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    return _distances_from_uniform(rng.random(count), near_m, far_m)


def _distances_from_uniform(u: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    """The inverse CDF of :func:`volume_uniform_distances`, at the uniforms ``u``."""
    near = max(near_m, 1e-3)
    far = max(far_m, near * (1.0 + 1e-6))
    return np.cbrt(near**3 + u * (far**3 - near**3)).astype(np.float32)


# Gunn & Marshall intercept: N0 = 3.8e3 * R^-0.87 m^-3 mm^-1.
GM_INTERCEPT_COEFF = 3.8e3
GM_INTERCEPT_EXPONENT = -0.87


def snow_number_density(snowfall_rate_mm_h: float, min_diameter_mm: float = 1.0) -> float:
    """Snow aggregates per cubic metre larger than ``min_diameter_mm``.

    Integrating Gunn & Marshall's ``N(D) = N0 exp(-Lambda D)`` from
    ``min_diameter_mm`` upward gives ``(N0 / Lambda) exp(-Lambda D_min)``.

    This is what makes the rendered flake count a *consequence* of the
    weather rather than an independent knob that can silently contradict it:
    pick a snowfall rate and a column of air, and the number of flakes that
    belong in frame follows.
    """
    slope = gunn_marshall_slope(snowfall_rate_mm_h)
    intercept = GM_INTERCEPT_COEFF * max(snowfall_rate_mm_h, 1e-3) ** GM_INTERCEPT_EXPONENT
    return float(intercept / slope * np.exp(-slope * min_diameter_mm))


def frustum_volume(
    width: int, height: int, horizontal_fov_deg: float, near_m: float, far_m: float
) -> float:
    """Volume (m^3) of the camera frustum between ``near_m`` and ``far_m``.

    A rectilinear frustum's cross-section at distance ``z`` is
    ``z^2 * 4 tan(h/2) tan(v/2)``, so the volume integrates to
    ``4 tan(h/2) tan(v/2) (far^3 - near^3) / 3``.
    """
    tan_h = np.tan(np.deg2rad(horizontal_fov_deg) / 2.0)
    tan_v = tan_h * height / max(width, 1)
    near = max(near_m, 0.0)
    far = max(far_m, near)
    return float(4.0 * tan_h * tan_v * (far**3 - near**3) / 3.0)


# Marshall-Palmer intercept, m^-3 mm^-1. Unlike snow's, it does not vary
# with rate — that is the content of the original 1948 result.
MP_INTERCEPT = 8.0e3


def rain_number_density(
    rain_rate_mm_h: float, min_diameter_mm: float = 1.0, shape_mu: float = 0.0
) -> float:
    """Raindrops per cubic metre larger than ``min_diameter_mm``.

    ``shape_mu = 0`` is Marshall & Palmer and keeps their closed form. For a
    gamma distribution the intercept is not a constant of the model, so it is
    calibrated numerically against the rain rate itself:

        R = 1.885e-3 * integral of N(D) D^3 v(D) dD

    with ``D`` in mm and ``v`` from :func:`raindrop_terminal_velocity`. That
    is the definition of the rain rate rather than a fit, so the returned
    density carries the same rate whatever shape is asked for — which is the
    property the whole derived chain depends on.
    """
    if shape_mu <= 0.0:
        slope = marshall_palmer_slope(rain_rate_mm_h)
        return float(MP_INTERCEPT / slope * np.exp(-slope * min_diameter_mm))

    diameters = np.linspace(0.05, MAX_RAINDROP_DIAMETER_MM, 2048, dtype=np.float64)
    step = float(diameters[1] - diameters[0])
    speeds = raindrop_terminal_velocity(diameters)

    def _density_above(slope: float, mu: float) -> tuple[float, float]:
        """The rain rate the distribution's shape gives, and its share above ``min_diameter_mm``.

        Both per unit of the distribution's intercept, so that their ratio
        fixes the intercept for the rain rate asked for.
        """
        shape = _gamma_dsd_weights(diameters, slope, mu)
        rate = 1.885e-3 * float((shape * diameters**3 * speeds).sum() * step)
        return rate, float(shape[diameters >= min_diameter_mm].sum() * step)

    # Calibrated *relative to Marshall & Palmer at the same rate* rather than
    # absolutely. Integrating MP's own constants does not return the rain rate
    # they are labelled with -- (N0 = 8000, Lambda = 4.1 R^-0.21) comes out
    # 4% high at 100 mm/h and 20% high at 5 mm/h, because the intercept and
    # the slope relation were fitted separately. Calibrating the gamma path
    # absolutely would therefore put a 10% step at mu = 0 that is an artefact
    # of that inconsistency rather than a property of the shape. Taking the
    # ratio cancels it, so mu = 0 is exactly Marshall-Palmer and mu > 0
    # differs only by what the shape change actually does.
    reference_slope = marshall_palmer_slope(rain_rate_mm_h)
    reference_rate, reference_density = _density_above(reference_slope, 0.0)
    rate, density = _density_above(gamma_dsd_slope(rain_rate_mm_h, shape_mu), shape_mu)
    if rate <= 0.0 or reference_rate <= 0.0 or reference_density <= 0.0:  # pragma: no cover
        return 0.0
    marshall_palmer = MP_INTERCEPT / reference_slope * np.exp(-reference_slope * min_diameter_mm)
    return float(marshall_palmer * (density / reference_density) * (reference_rate / rate))


def width_cut_for_budget(
    population: float,
    budget: int,
    slope: float,
    d_min: float,
    d_max: float,
    near_m: float,
    far_m: float,
    focal_px: float,
    floor_width_px: float,
    size_scale: float = 1.0,
    shape_mu: float = 0.0,
) -> float:
    """Projected-width threshold (px) that keeps the rendered count within budget.

    A physically complete precipitation column holds far more particles than
    can be rasterised — hundreds of thousands in a few cubic metres, the
    overwhelming majority under a pixel wide. Simply truncating the count
    renders an arbitrary *subset* and makes apparent intensity depend on the
    render budget rather than on the weather, which shows up as a severity
    ladder that stops getting worse (or even reverses) once the cap binds.

    So the budget is converted into the width above which particles are drawn
    individually. Everything narrower is not discarded — it is exactly what
    the veil represents — and the particles that *are* drawn are always the
    most visible ones. Solved by bisection, since the resolvable fraction is
    monotonically decreasing in the threshold but has no closed-form inverse
    once distance is involved.
    """
    if budget <= 0 or population <= budget:
        return floor_width_px

    def count_at(width: float) -> float:
        """How many drops would be drawn with the width threshold at ``width``.

        The population scaled by the share that resolves at ``width`` over the
        share that resolves at the floor.
        """
        fraction = resolvable_fraction(
            slope, d_min, d_max, near_m, far_m, focal_px, width, size_scale, shape_mu=shape_mu
        )
        base = resolvable_fraction(
            slope,
            d_min,
            d_max,
            near_m,
            far_m,
            focal_px,
            floor_width_px,
            size_scale,
            shape_mu=shape_mu,
        )
        if base <= 0.0:
            return 0.0
        return population * fraction / base

    low, high = floor_width_px, max(floor_width_px * 2.0, 1.0)
    for _ in range(24):
        if count_at(high) <= budget:
            break
        high *= 2.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        if count_at(mid) > budget:
            low = mid
        else:
            high = mid
    return float(high)


def rain_extinction_coefficient(rain_rate_mm_h: float) -> float:
    """Extinction coefficient (1/m) of a rain-filled atmosphere.

    Atlas (1953, "Optical extinction by rainfall"): ``sigma = 0.25 R^0.63``
    per kilometre for ordinary (Bergeron-process) rain, the fit behind his
    visibility relation ``V = 11.6 R^-0.63`` km. At 40 mm/h that is 2.5 per
    km: a meteorological optical range, at the 2% contrast this library
    uses, of 1.5 km.

    Two independent routes land in the same place. Carbonneau's fit for
    optical links, 1.076 R^0.67 dB/km, is 0.248 R^0.67 per km. And summing
    geometric cross-sections over the Marshall-Palmer distribution this
    library samples its drops from gives ``pi N0 / Lambda^3``, 3.7 per km at
    40 mm/h, which is an upper bound: it counts the light diffracted forward
    into the lens as lost.

    This replaces ``V = 1.9 R^-0.63`` km, described here as the standard
    relation, which it is not: it is 8.2 times Atlas's extinction at every
    rate, and put visibility in 40 mm/h rain at 186 m. Against the default
    3-50 m depth range that passed for a heavy veil. Against real distances
    it hid everything beyond a couple of hundred metres.

    This is what ties the veil to the streaks. The two layers are not
    independent effects: the veil *is* the drops too small to resolve as
    individual streaks, so letting a caller set its density separately from
    the rain rate allows a frame with a downpour's haze and a drizzle's
    streaks. Deriving it means one physical knob controls both.
    """
    return float(0.25e-3 * max(rain_rate_mm_h, 1e-3) ** 0.63)


def snow_extinction_coefficient(snowfall_rate_mm_h: float) -> float:
    """Extinction coefficient (1/m) of a snow-filled atmosphere.

    From ``V = 0.6 / R`` km for dry snow (Rasmussen et al.), again with
    ``beta = 3.912 / V``.

    At equal liquid-equivalent rate this is some thirty times rain's
    extinction — 26x at 1 mm/h, 37x at 2.5 mm/h — because the same water is
    spread over flakes with far more cross-section than drops. That is why a
    snowfall which would be unremarkable as rain drops visibility so
    sharply, and why the veil — not the flakes — is the strongest cue that
    it is snowing.

    Treat the constant as an order of magnitude. Rasmussen et al. report
    visibility varying by a factor of 3 to 10 between storms at one rate,
    with crystal type and riming. (An earlier note here offered a 6x ratio
    to rain as a check on both relations. It only held because the rain
    figure was eight times too large.)
    """
    return float(3.912 / 0.6e3 * max(snowfall_rate_mm_h, 1e-3))


# Bulk density of freshly fallen snow, g/cm^3. Dendritic aggregates are mostly
# air: 0.05 for fresh dry dendrites, ~0.2 for wet compacted snow.
SNOW_DENSITY_G_PER_CM3 = 0.08


def snow_physical_diameter(
    melted_diameter_mm: np.ndarray | float, density_g_cm3: float
) -> np.ndarray:
    """Convert a melted-equivalent drop diameter to the aggregate's real size.

    This distinction is easy to miss and changes everything about how snow
    looks. Gunn & Marshall — like every gauge-derived precipitation
    distribution — is stated in **melted-equivalent** diameters: the size of
    the water drop a flake would become. But a snowflake is mostly air, so
    the thing the camera actually sees is far bigger:

        D_physical = D_melted * (rho_water / rho_snow)^(1/3)

    At a typical fresh-snow density of 0.08 g/cm^3 that is a factor of about
    2.3 in diameter, and therefore **five times the projected area**. Using
    the melted diameter as the optical size makes every flake far too small
    to resolve, which is exactly the "where did the snow go" failure: the
    distribution was right, the unit was wrong.
    """
    ratio = (1.0 / max(density_g_cm3, 1e-3)) ** (1.0 / 3.0)
    return np.asarray(melted_diameter_mm, dtype=np.float32) * ratio


def _truncated_exponential_survival(
    diameter_mm: np.ndarray, slope: float, low: float, high: float
) -> np.ndarray:
    """``P(D > diameter)`` for the truncated exponential used above."""
    span = max(high - low, 1e-6)
    normaliser = -np.expm1(-slope * span)
    clipped = np.clip(diameter_mm, low, high)
    tail = np.exp(-slope * (clipped - low)) - np.exp(-slope * span)
    survival = tail / max(normaliser, 1e-12)
    return np.where(diameter_mm <= low, 1.0, np.where(diameter_mm >= high, 0.0, survival))


def _dsd_survival(
    diameter_mm: np.ndarray, slope: float, low: float, high: float, shape_mu: float
) -> np.ndarray:
    """``P(D > diameter)`` for the truncated gamma, exponential when ``mu = 0``.

    Quadrature on the same grid the sampler inverts, so the two agree by
    construction: a diameter's sampling probability and its contribution to
    the resolvable fraction come from one description of the distribution.
    Keeping them in separate closed forms is how that pair drifts apart.
    """
    if shape_mu <= 0.0:
        return _truncated_exponential_survival(diameter_mm, slope, low, high)
    edges = np.linspace(low, high, 1024, dtype=np.float64)
    weights = _gamma_dsd_weights(edges, slope, shape_mu)
    cdf = np.cumsum(weights)
    cdf /= max(float(cdf[-1]), 1e-30)
    inside = np.interp(np.clip(diameter_mm, low, high), edges, cdf)
    survival = 1.0 - inside
    return np.where(diameter_mm <= low, 1.0, np.where(diameter_mm >= high, 0.0, survival))


def resolvable_fraction(
    slope: float,
    d_min: float,
    d_max: float,
    near_m: float,
    far_m: float,
    focal_px: float,
    min_width_px: float,
    size_scale: float = 1.0,
    samples: int = 512,
    shape_mu: float = 0.0,
) -> float:
    """Fraction of particles that project wider than ``min_width_px``.

    A particle is worth rasterising only if it covers something like a pixel;
    below that it deposits a sub-pixel smudge for the cost of a full sprite,
    and it belongs to the veil instead. But "resolvable" depends on diameter
    *and* distance together — a big drop far away and a small one close by can
    project to the same width — so cutting on diameter alone (which is what
    a pure size budget does) throws away exactly the near particles that carry
    the visible texture, and keeps far ones that cannot be seen.

    This integrates ``P(D > w0 z / f)`` over the volume-uniform distance
    distribution to get the true resolvable fraction, so the caller can both
    size its sample correctly and know how many resolvable particles really
    are in frame.
    """
    near = max(near_m, 1e-3)
    far = max(far_m, near * (1.0 + 1e-6))
    # Volume-uniform in distance means density proportional to z^2.
    z = np.linspace(near, far, samples, dtype=np.float64)
    weight = z**2
    weight /= weight.sum()

    required_mm = min_width_px * z / max(focal_px, 1e-6) * 1e3 / max(size_scale, 1e-6)
    survival = _dsd_survival(required_mm, slope, d_min, d_max, shape_mu)
    return float(np.sum(weight * survival))


def _least_uniform_to_resolve(
    slope: float,
    d_min: float,
    d_max: float,
    near_m: float,
    far_m: float,
    focal_px: float,
    min_width_px: float,
    size_scale: float,
    shape_mu: float,
    slices: int = 1024,
    grid: int = 1024,
) -> np.ndarray:
    """For each slice of the distance uniform, a diameter uniform too small to resolve.

    The distance sampler is increasing in its uniform, so the slice
    ``[k / slices, (k + 1) / slices)`` holds no distance nearer than the one
    at ``k / slices``. A particle that near has to be at least so wide in
    millimetres to project to ``min_width_px``; the diameter sampler is
    increasing in its uniform too, so there is a uniform at or below which
    its diameter is under that. Entry ``k`` is that uniform, taken with a
    margin of one part in ten thousand, a thousand times the float32 rounding
    of the width test this stands in front of: a candidate whose diameter
    uniform does not exceed its slice's entry cannot pass that test, and one
    that does is put to it as before.
    """
    nearest = _distances_from_uniform(np.arange(slices) / slices, near_m, far_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        needed_mm = min_width_px * nearest.astype(np.float64) / (focal_px * size_scale * 1e-3)
    needed_mm = np.nan_to_num(needed_mm * (1.0 - 1e-4), nan=0.0, posinf=np.inf, neginf=0.0)

    if shape_mu <= 0.0:
        # The inverse of `_exponential_from_uniform`.
        span = max(d_max - d_min, 1e-6)
        saturation = -np.expm1(-slope * span)
        inside = -np.expm1(-slope * (np.clip(needed_mm, d_min, d_max) - d_min)) / saturation
        least = np.where(needed_mm >= d_max, 1.0, inside - 1e-9)
    else:
        # `_diameters_from_uniform` returns the first grid diameter whose CDF
        # reaches the uniform, so a uniform up to the CDF of the last grid
        # diameter under the need gives a diameter under it.
        edges = np.linspace(d_min, d_max, grid, dtype=np.float64)
        cdf = np.cumsum(_gamma_dsd_weights(edges, slope, shape_mu))
        if cdf[-1] <= 0.0:  # pragma: no cover - degenerate slope/window
            return np.full(slices, -1.0)
        cdf /= cdf[-1]
        last_under = np.searchsorted(edges, needed_mm, side="left") - 1
        least = np.where(last_under >= 0, cdf[np.clip(last_under, 0, grid - 1)], -1.0)
    # Nothing is ruled out where the smallest particle already resolves.
    return np.where(needed_mm <= d_min, -1.0, least)


def sample_resolvable_particles(
    rng: np.random.Generator,
    count: int,
    slope: float,
    d_min: float,
    d_max: float,
    near_m: float,
    far_m: float,
    focal_px: float,
    min_width_px: float,
    size_scale: float = 1.0,
    shape_mu: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``count`` ``(diameter_mm, distance_m)`` pairs that actually resolve.

    Rejection sampling against the joint (size, distance) distribution, with
    the batch size set from the analytic acceptance rate so it normally
    finishes in one pass. The returned pairs are distributed exactly as the
    physical population is *conditioned on being resolvable*, so the field
    keeps the right mix of near-large and far-small particles — it simply
    does not spend the budget drawing the ones that cannot be seen.

    About one candidate in a hundred is kept, and most of the rest can be
    told from their two uniforms alone (:func:`_least_uniform_to_resolve`).
    The diameter and the distance are computed only for the candidates that
    test lets through: the same uniforms are drawn in the same order and the
    same pairs come back.
    """
    if count <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    acceptance = resolvable_fraction(
        slope,
        d_min,
        d_max,
        near_m,
        far_m,
        focal_px,
        min_width_px,
        size_scale,
        shape_mu=shape_mu,
    )
    acceptance = max(acceptance, 1e-4)

    least = _least_uniform_to_resolve(
        slope, d_min, d_max, near_m, far_m, focal_px, min_width_px, size_scale, shape_mu
    )

    diameters: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    found = 0
    for _ in range(8):
        batch = int(min(2_000_000, max(1024, (count - found) / acceptance * 1.4)))
        # Diameters' uniforms first, then distances': the order the two
        # samplers drew in when each candidate was computed in full.
        uniform_d = rng.random(batch)
        uniform_z = rng.random(batch)
        slices = (uniform_z * len(least)).astype(np.intp)
        possible = np.flatnonzero(uniform_d > least[slices])
        candidate_d = _diameters_from_uniform(uniform_d[possible], slope, d_min, d_max, shape_mu)
        candidate_z = _distances_from_uniform(uniform_z[possible], near_m, far_m)
        width = focal_px * (candidate_d * size_scale * 1e-3) / candidate_z
        keep = width > min_width_px
        if keep.any():
            diameters.append(candidate_d[keep])
            distances.append(candidate_z[keep])
            found += int(keep.sum())
        if found >= count:
            break

    if not diameters:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    all_d = np.concatenate(diameters)[:count]
    all_z = np.concatenate(distances)[:count]
    return all_d.astype(np.float32), all_z.astype(np.float32)


# The far field. Bands of distance double from one to the next, and the
# texture is worked out for this many of them at most; a band whose texture
# would vary by less than this share of its own veil is smooth, and so is
# everything beyond it.
_FAR_BANDS = 5
# Lit from the lens, the first band returns half the light and the second a
# quarter: the grain is theirs, and the rest is drawn even.
_LIT_BANDS = 2
_FAR_SMOOTH = 0.05
# The side of the tile a band's grain is made on in fast mode.
_FAR_TILE = 512
# The lens and the pixel spread a point over about this many pixels, as the
# sigma of a Gaussian: a drop far smaller than a pixel is seen through it.
_FAR_PSF_PX = 0.7
# A band with fewer particles than this to a pixel is counted out, particle
# by particle; one with more varies as a smooth quantity does.
_FAR_COUNTED = 0.75
# Of the particles in a volume, the share that makes its veil: extinction
# goes with the square of the diameter, and for an exponential distribution
# of sizes the effective number is a sixth of the count.
_FAR_EFFECTIVE = 1.0 / 6.0
# Rain and snow come in sheets. How much the far veil's thickness varies
# from one to the next, and how many sheets there are across the frame.
_SHEET_DEPTH = 0.22
_SHEET_CELLS = 5


def far_field_texture(
    depth_m: np.ndarray,
    rng: np.random.Generator,
    number_density: float,
    focal_px: float,
    near_m: float,
    fall_px_at_1m: float,
    angle_deg: float = 0.0,
    strength: float = 1.0,
    lit_from_lens: bool = False,
) -> np.ndarray:
    """How the veil of distant rain or snow varies across a frame, as a multiple of its mean.

    ``(H, W)`` float32, about 1: the optical depth of the veil in front of
    each pixel is its smooth value times this.

    A transform draws the particles near the lens one by one and leaves the
    rest to a veil, which is their average: the same grey in front of every
    pixel at one distance. It is the average of a countable number of
    things. Between the last particle drawn and the distance at which there
    are thousands in front of every pixel, the air holds a few, then tens,
    and their number varies from pixel to pixel as a count does, by its own
    square root. That variation is the fine streaked grain of rain against
    a dark doorway and the speckle of snow against a wood, and a veil
    without it is fog.

    It is drawn in bands of distance, each twice as deep as the one before,
    from ``near_m``, where the transform's own particles stop. In a band
    at distance ``z`` and of depth ``dz`` a pixel looks down a column of
    air ``(z / f)^2 dz`` in volume, holding
    ``number_density * (z / f)^2 * dz`` particles; the veil there varies by
    one over the square root of the effective number of them. The grain is
    as long as the band's particles fall during the exposure,
    ``fall_px_at_1m / z`` pixels along ``angle_deg`` from straight down,
    and as wide as the lens makes a point. Nearer bands are coarse and
    strong, far ones fine and faint, until there is nothing left to see.

    **It is in the scene.** Each band counts only for the part of it that
    lies in front of what the pixel shows (``depth_m``): there is no rain
    between the camera and a wall three metres off but what was drawn
    there, and the full depth of it in front of a hillside. A veil
    textured this way has the scene's own depth in it, where particles
    drawn over the frame have none.

    **And it comes in sheets**: over all of it, the far veil is thicker in
    one place than another by a fifth, in upright curtains a few to a
    frame.

    **Lit from the lens**, at night, by the camera's own lamp
    (``lit_from_lens``), what comes back from a band is not in proportion
    to its depth but to its depth over the square of its distance: the
    nearest band is half of it, and the texture is that band's. The
    scene's depth then plays no part, and ``depth_m`` gives only the
    frame's shape.

    ``number_density`` is in particles per cubic metre, of all sizes;
    ``strength`` scales the variation, 0 giving the smooth veil back.
    """
    height, width = depth_m.shape
    if strength <= 0.0 or number_density <= 0.0:
        return np.ones((height, width), dtype=np.float32)
    depth = depth_m.astype(np.float32, copy=False)
    reach = np.inf if lit_from_lens else float(depth.max())
    carried = np.zeros((height, width), dtype=np.float32)
    share = np.empty((height, width), dtype=np.float32)
    # In fast mode each band's grain is made on a tile and laid over the
    # frame, a different way round and from a different corner for each
    # band: a grain a pixel or two across has no shape to repeat, and the
    # scene's own depth, which is not tiled, is what gives it one.
    tile = (min(height, _FAR_TILE), min(width, _FAR_TILE)) if mode.fast() else (height, width)
    effective = number_density * _FAR_EFFECTIVE
    slant = np.deg2rad(angle_deg)
    front = near_m
    white: np.ndarray | None = None
    for _band in range(_LIT_BANDS if lit_from_lens else _FAR_BANDS):
        back = 2.0 * front
        if front >= reach:
            break
        middle = float(np.sqrt(front * back))
        per_pixel = effective * (middle / focal_px) ** 2 * (back - front)
        kernel = _grain_kernel(max(fall_px_at_1m / middle, 0.0), slant)
        sharp = float((kernel**2).sum())
        varies = float(np.sqrt(sharp / max(per_pixel, 1e-6)))
        if varies < _FAR_SMOOTH:
            break
        more: np.ndarray
        if per_pixel < _FAR_COUNTED:
            # Fewer than one to a pixel: counted out, each where it is.
            # Most pixels of a sparse band hold none and a few hold one,
            # which is a scatter of dots and nothing like an even grain.
            count = int(rng.poisson(per_pixel * tile[0] * tile[1]))
            places = rng.integers(0, tile[0] * tile[1], size=count)
            counted = np.bincount(places, minlength=tile[0] * tile[1]).astype(np.float32)
            more = np.asarray(
                cv2.filter2D(counted.reshape(tile), -1, kernel, borderType=cv2.BORDER_REFLECT)
            )
            more = more * np.float32(1.0 / per_pixel) - np.float32(1.0)
        else:
            # One field of noise for every band, moved for each: a band's
            # grain is its own length, so two bands cut from one field are
            # no more alike than two fields, and drawing a field is half
            # the cost of a band.
            if white is None:
                white = rng.standard_normal(tile, dtype=np.float32)
            shift = (int(rng.integers(tile[0])), int(rng.integers(tile[1])))
            moved = np.ascontiguousarray(np.roll(white, shift, axis=(0, 1)))
            grain = np.asarray(cv2.filter2D(moved, -1, kernel, borderType=cv2.BORDER_REFLECT))
            # A count is never below nothing and now and then well above
            # its mean: the variation is one-sided, as a log-normal's is.
            spread = float(np.sqrt(np.log1p(varies**2)))
            grain = grain * np.float32(spread / np.sqrt(sharp)) - np.float32(0.5 * spread**2)
            more = np.exp(grain, out=grain) - np.float32(1.0)
        more = _laid_over(more, (height, width), rng)
        if lit_from_lens:
            carried += more * np.float32(near_m * (1.0 / front - 1.0 / back))
        else:
            np.subtract(depth, np.float32(front), out=share)
            np.clip(share, 0.0, np.float32(back - front), out=share)
            share *= more
            carried += share
        front = back
    carried *= np.float32(strength)
    if lit_from_lens:
        carried += np.float32(1.0)
        return np.maximum(carried, 0.0, out=carried)
    texture = carried
    np.divide(texture, np.maximum(depth, np.float32(1e-3)), out=texture)
    texture += np.float32(1.0)
    if _SHEET_DEPTH > 0.0:
        small = (max(2, width // 8), max(2, height // 8))
        cells = (max(2, _SHEET_CELLS // 2), _SHEET_CELLS)
        coarse = np.tanh(rng.standard_normal(cells, dtype=np.float32))
        sheets = np.asarray(cv2.resize(coarse, small, interpolation=cv2.INTER_CUBIC))
        hung = np.asarray(cv2.resize(sheets, (width, height), interpolation=cv2.INTER_LINEAR))
        hung = hung * np.float32(_SHEET_DEPTH * min(strength, 1.0))
        # Of the veil beyond where the drawn particles stop, and by the
        # frame's own depth: a sheet hangs behind a near wall, not on it.
        np.divide(np.float32(near_m), np.maximum(depth, np.float32(1e-3)), out=share)
        np.subtract(np.float32(1.0), share, out=share)
        np.clip(share, 0.0, 1.0, out=share)
        share *= hung
        share += np.float32(1.0)
        texture *= share
    return np.maximum(texture, 0.0, out=texture)


def _laid_over(tile: np.ndarray, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A tile repeated over a frame of ``shape``, from a corner and a way round of its own.

    The tile itself where it is already the frame's size.
    """
    if tile.shape == shape:
        return tile
    if rng.integers(2):
        tile = tile[::-1]
    if rng.integers(2):
        tile = tile[:, ::-1]
    shift = (int(rng.integers(tile.shape[0])), int(rng.integers(tile.shape[1])))
    tile = np.ascontiguousarray(np.roll(tile, shift, axis=(0, 1)))
    over: np.ndarray = cv2.copyMakeBorder(
        tile, 0, shape[0] - tile.shape[0], 0, shape[1] - tile.shape[1], cv2.BORDER_WRAP
    )
    return over


def _grain_kernel(long_px: float, slant_rad: float) -> np.ndarray:
    """What one far particle looks like: a line as long as it falls, through the lens's blur.

    A float32 kernel that sums to 1, ``long_px`` long at ``slant_rad`` from
    straight down, widened by a Gaussian the size of the lens's own blur.
    """
    half = int(np.ceil(0.5 * long_px + 3.0 * _FAR_PSF_PX))
    size = 2 * half + 1
    ys, xs = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float32)
    along = xs * np.float32(np.sin(slant_rad)) + ys * np.float32(np.cos(slant_rad))
    aside = xs * np.float32(np.cos(slant_rad)) - ys * np.float32(np.sin(slant_rad))
    past = np.maximum(np.abs(along) - np.float32(0.5 * long_px), 0.0)
    kernel = np.exp(-0.5 * (aside**2 + past**2) / np.float32(_FAR_PSF_PX**2))
    assert kernel.shape == (size, size)
    result: np.ndarray = (kernel / kernel.sum()).astype(np.float32)
    return result
