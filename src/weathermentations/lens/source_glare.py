"""Lens glare from the lights that are in the frame.

``SunFlare`` has the optics — the veiling glare, the diffraction starburst,
the chain of ghosts along the line through the image centre — and puts them
on one source at a position the caller chooses. At night the sources are in
the picture, there are several, and that is where glare has to come from: a
headlight or a street lamp, each with its own flare.

So this finds the compact clipped regions of the frame
(:mod:`weathermentations.utils.sources`) and renders a ``SunFlare`` at each.
Nothing optical is reimplemented; the one new quantity is how strong each
flare is. A clipped pixel only says the light was *at least* full scale, so
strength is taken from the clipped area — a bigger blown region is a
brighter or nearer lamp — times ``intensity_range``, which stands for the
over-range no photograph records.

Artefacts of an ageing camera that ``SunFlare`` also carries (dome scratches,
CCD smear, infrared leak, the housing-window ghost) are left to it. They are
properties of one camera in sunlight, and repeating them per lamp would draw
the same scratch three times.

A frame with no compact light in it comes back unchanged. That is the right
answer for most daylit frames, and it means this transform, unlike the
others, can do nothing with ``p=1``.
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator

from weathermentations.base import WeatherTransform, first_frame, scene_of
from weathermentations.diagnostics import record_draw
from weathermentations.lens.sun_flare import SunFlare
from weathermentations.utils.photometry import to_linear_unit
from weathermentations.utils.sources import light_sources

# Clipped area, as a fraction of the frame, of a source that gets exactly the
# drawn intensity: a lamp about 1.5% of the frame's width across.
_REFERENCE_AREA = 2e-4


class SourceGlare(WeatherTransform):
    """Flare from each compact light in the frame.

    Args:
        max_sources_range: how many of the frame's lights flare, largest
            first, drawn per call. Each costs one ``SunFlare`` render.
        intensity_range: flare strength for a source of the reference size,
            in ``SunFlare``'s units, drawn per call. It stands for how far
            past full scale the lights really were, which is not recorded. A
            lamp is orders of magnitude dimmer than the sun those units were
            set for, so useful values are a fraction of ``SunFlare``'s.
            Larger clipped regions flare more, as the square root of their
            area, within a factor of two either way.
        ghost_count_range, halo_radius_range, aperture_blades_range,
            streak_probability: as for ``SunFlare``.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        max_sources_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1, 8)),
        ]
        intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 3.0)),
        ]
        ghost_count_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 20)),
        ]
        halo_radius_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.02, 1.0)),
        ]
        aperture_blades_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(3, 16)),
        ]
        streak_probability: float

    def __init__(
        self,
        max_sources_range: tuple[int, int] = (1, 3),
        intensity_range: tuple[float, float] = (0.08, 0.35),
        ghost_count_range: tuple[int, int] = (2, 6),
        halo_radius_range: tuple[float, float] = (0.04, 0.1),
        aperture_blades_range: tuple[int, int] = (5, 9),
        streak_probability: float = 0.75,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.max_sources_range = max_sources_range
        self.intensity_range = intensity_range
        self.ghost_count_range = ghost_count_range
        self.halo_radius_range = halo_radius_range
        self.aperture_blades_range = aperture_blades_range
        self.streak_probability = streak_probability
        # The optics. Everything that belongs to one ageing camera in
        # sunlight is switched off; see the module docstring.
        self._flare = SunFlare(
            p=1.0,
            ghost_count_range=ghost_count_range,
            halo_radius_range=halo_radius_range,
            aperture_blades_range=aperture_blades_range,
            streak_probability=streak_probability,
            # The rainbow ring needs a source as bright as the sun.
            dispersion_ring_probability=0.0,
            sensor_smear_probability=0.0,
            scratch_count_range=(0, 0),
            ir_leak_probability=0.0,
            window_ghost_probability=0.0,
        )

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        linear, _scale, _is_integer = to_linear_unit(first_frame(scene_of(data)))
        sources = light_sources(linear, self.py_random.randint(*self.max_sources_range))
        base = self.py_random.uniform(*self.intensity_range)

        # One generator for the whole transform, so a seed fixes every flare.
        flare = self._flare
        flare.py_random = self.py_random
        flare.random_generator = self.random_generator
        # The undecorated method: the decorated one would log each flare as
        # a `SunFlare` of its own, and there is no sun in this frame.
        sample = SunFlare.get_params_dependent_on_data.__wrapped__  # type: ignore[attr-defined]

        flares = []
        for source in sources:
            size = float(np.clip(np.sqrt(source.area / _REFERENCE_AREA), 0.5, 2.0))
            flare.flare_position = (source.x, source.y)
            flare.intensity_range = (base * size, base * size)
            flares.append(sample(flare, params, data))
        record_draw(
            sources=len(sources),
            source_area_max=max((s.area for s in sources), default=0.0),
            base_intensity=base,
        )
        return {"flares": flares}

    def apply(self, img: np.ndarray, flares: list[dict[str, Any]], **params: Any) -> np.ndarray:
        for flare in flares:
            img = self._flare.apply(img, **flare)
        return img
