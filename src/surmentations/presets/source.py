"""What a frame already is, before anything is done to it.

The ready-made pipelines turn a photograph into another hour and another
weather, and every branch of them takes the photograph to be a dry scene
out of doors by day. Given a frame taken at night, the infrared branch lit
a scene that was already lit; given a car park under a roof, the rain
branch rained in it. Neither is a hard case for a model to learn. Both are
frames that could not exist, which is what these pipelines are for ruling
out.

:func:`read_source` says which of the three a frame is, from the frame
itself and from the scene's labels where a provider has them, and
:class:`BySource` is the container that sends a frame down the branch built
for it.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from albumentations.core.composition import BaseCompose, TransformsSeqType

from surmentations.depth.provider import SharedDepthProvider
from surmentations.perception.labels import SceneClass
from surmentations.utils.photometry import luminance, to_linear_unit

# A frame taken at night: half of it darker than the first level, in linear
# light, and the whole of it darker on average than the second. A scene in
# deep shade by day (a road under a bridge and trees) has a median several
# times the first.
_NIGHT = (0.02, 0.08)
# What is only out of doors, and the share of a frame under which a scene
# with labels is taken to be under a roof.
_OUT_OF_DOORS = (
    SceneClass.SKY,
    SceneClass.ROAD,
    SceneClass.GRASS,
    SceneClass.FOLIAGE,
    SceneClass.TERRAIN,
    SceneClass.WATER,
)
_LEAST_OUT_OF_DOORS = 0.01


class Source(NamedTuple):
    """What a frame already is.

    The fields:

    * ``night``: it was taken after dark, and is lit by what lamps it has.
    * ``indoors``: it shows nothing that is only out of doors. Known only
      with the scene's labels; without them a frame is taken to be outside.
    """

    night: bool
    indoors: bool


def read_source(image: np.ndarray, labels: np.ndarray | None = None) -> Source:
    """Whether a frame was taken at night, and whether under a roof.

    ``image`` is one ``(H, W, 3)`` frame and ``labels`` its scene labels
    (:class:`~surmentations.perception.labels.SceneClass`), where there are
    any.

    * **Night** is read off the frame's own light: a night frame is mostly
      black with a few things lit, so half of it is nearly dark however
      bright its lamps are.
    * **Indoors** is read off the labels: no sky, no road, no vegetation,
      no open ground or water. A camera looking straight down at a paved
      square shows none of those either, and is taken for indoors; it is
      then given no weather, which is the safer mistake of the two.
    """
    linear, _scale, _is_integer = to_linear_unit(image)
    light = luminance(linear)
    night = float(np.median(light)) < _NIGHT[0] and float(light.mean()) < _NIGHT[1]
    indoors = False
    if labels is not None:
        outside = np.isin(labels, [int(one) for one in _OUT_OF_DOORS])
        indoors = float(outside.mean()) < _LEAST_OUT_OF_DOORS
    return Source(night, indoors)


class BySource(BaseCompose):
    """One of three branches, chosen by what the frame already is.

    Args:
        by_day: the branch for a frame taken out of doors by day, which is
            what every transform that changes the hour or the weather
            takes its frame to be.
        at_night: the branch for a frame taken at night. It is night
            already: nothing in it should light it again or put the sun in
            it.
        indoors: the branch for a frame taken under a roof, where there is
            no weather.
        depth_provider: the pipeline's provider, asked for the scene's
            labels; ``None``, or a provider without labels, and no frame is
            taken to be indoors.

    Nothing is drawn here: the same frame always goes the same way.
    """

    def __init__(
        self,
        by_day: Any,
        at_night: Any,
        indoors: Any,
        depth_provider: SharedDepthProvider | None = None,
        p: float = 1.0,
    ) -> None:
        """Keep the three branches and the provider."""
        branches: TransformsSeqType = [by_day, at_night, indoors]
        super().__init__(transforms=branches, p=p)
        self.depth_provider = depth_provider

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        """Send the frame down the branch for what it is."""
        if self.replay_mode:
            for transform in self.transforms:
                data = transform(**data)
            return data
        frame = data.get("image")
        if frame is None and "images" in data:
            frame = data["images"][0]
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            chosen = self.transforms[0]
        else:
            labels = None
            if self.depth_provider is not None:
                labels = self.depth_provider.get_labels(frame)
            source = read_source(frame, labels)
            chosen = self.transforms[2 if source.indoors else 1 if source.night else 0]
        data = chosen(force_apply=True, **data)
        self._track_transform_params(chosen, data)
        return data
