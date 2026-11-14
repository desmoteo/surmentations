"""What a scene holds and how the camera is mounted over it.

Depth says how far away each pixel is. This package answers the two
questions depth leaves open: what each pixel *is*
(:mod:`~surmentations.perception.labels`), and where the camera is that
took the picture (:mod:`~surmentations.perception.geometry`). Both are
optional, both are worked out once per scene and shared between transforms
by the same :class:`~surmentations.depth.provider.SharedDepthProvider` that
shares depth, and an effect given neither behaves as it did without them.

A third reading rests on the labels: which end of each vehicle is seen, and
where its lamps are (:mod:`~surmentations.perception.vehicles`).
"""

from surmentations.perception.geometry import (
    Camera,
    GeometryEstimator,
    GroundPlaneGeometryEstimator,
    InstallationGeometry,
)
from surmentations.perception.labels import (
    SceneClass,
    SceneLabelEstimator,
    SegmentationLabelEstimator,
    class_weight,
    classes_of,
)
from surmentations.perception.vehicles import (
    OpenVocabularyVehicleEstimator,
    VehicleView,
    VehicleViewEstimator,
)

__all__ = [
    "Camera",
    "GeometryEstimator",
    "GroundPlaneGeometryEstimator",
    "InstallationGeometry",
    "OpenVocabularyVehicleEstimator",
    "SceneClass",
    "SceneLabelEstimator",
    "SegmentationLabelEstimator",
    "VehicleView",
    "VehicleViewEstimator",
    "class_weight",
    "classes_of",
]
