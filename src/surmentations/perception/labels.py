"""What each pixel of a scene is: road, sky, vegetation, a vehicle.

Depth says how far away a pixel is and which way its surface faces. It does
not say what it is, and several effects turn on exactly that. Water lies as
a film on paving and not on a car's bonnet, which from two hundred metres is
the same smooth, level patch of a depth map. Snow buries a lawn and only
dusts a tree. The sky is where a veil takes its colour from. Foliage is
among the brightest things in a near-infrared picture and asphalt among the
darkest. Each of these was being guessed from colour, texture or the shape
of the depth map, and each guess has a scene that defeats it: nothing is
green under a sodium lamp.

A semantic segmentation network answers the question directly, once per
scene, and the answer is shared the way depth is
(:class:`~surmentations.depth.provider.SharedDepthProvider`).

A dozen classes, not a hundred and fifty
----------------------------------------
The effects need to tell a handful of things apart, so whatever a network
was trained to name is folded into :class:`SceneClass`. The folding is by
the *name* of each of the network's classes (:data:`CLASS_OF_NAME`), read
from the checkpoint's own label list, so any checkpoint trained on ADE20K or
Cityscapes can be used without a table of indices for it.

Evidence, not a verdict
-----------------------
The networks that are small enough to run inside an augmentation were
trained on photographs taken from eye level by day. A camera on a mast
looking straight down, or an infrared frame, is not what they saw: on the
sample scenes the bundled model names a road seen from directly above
"building". So no effect *requires* a class. Labels remove what plainly is
not ground (a vehicle, a person, a tree, the sky) and add what plainly is
(a road the normals read as uneven), and the geometry read from depth still
decides everything else. An effect given no labels behaves as it did before
there were any.

Licences
--------
The default checkpoint, ``openmmlab/upernet-convnext-tiny``, is published
under the MIT licence. The smaller and faster SegFormer checkpoints
(``nvidia/segformer-b0-finetuned-ade-512-512``) work here unchanged and are
under NVIDIA's licence for non-commercial use: whether that suits you is
yours to decide, which is why it is not the default.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from enum import IntEnum
from typing import Any

import cv2
import numpy as np

from surmentations import mode


class SceneClass(IntEnum):
    """What a pixel shows, as coarsely as the effects need to know it."""

    #: Anything not named below, and anything the network was unsure of.
    OTHER = 0
    #: Open sky.
    SKY = 1
    #: A carriageway: what vehicles drive on.
    ROAD = 2
    #: Paving that is not a carriageway: a pavement, a path, a floor.
    PAVEMENT = 3
    #: Bare ground: soil, sand, rock, a hillside.
    TERRAIN = 4
    #: Low vegetation that ground cover lies on like any ground: a lawn, a field.
    GRASS = 5
    #: Vegetation with no surface of its own: a tree's crown, a hedge, a shrub.
    FOLIAGE = 6
    #: Anything built that stands on the ground: a wall, a building, a bridge, a fence.
    STRUCTURE = 7
    #: A vehicle of any kind.
    VEHICLE = 8
    #: A person, or an animal.
    PERSON = 9
    #: Open water.
    WATER = 10
    #: A thin thing standing on the ground: a pole, a sign.
    POLE = 11
    #: A light fitting: a street lamp, a traffic light.
    LAMP = 12


#: What a film of water lies on.
PAVED: tuple[SceneClass, ...] = (SceneClass.ROAD, SceneClass.PAVEMENT)

#: What things stand on, and what an installation's ground plane is fitted to.
GROUND: tuple[SceneClass, ...] = (
    SceneClass.ROAD,
    SceneClass.PAVEMENT,
    SceneClass.TERRAIN,
    SceneClass.GRASS,
)

#: Vegetation of either kind.
VEGETATION: tuple[SceneClass, ...] = (SceneClass.GRASS, SceneClass.FOLIAGE)

#: What is never the ground under the camera, however level a depth map makes it.
NOT_GROUND: tuple[SceneClass, ...] = (
    SceneClass.SKY,
    SceneClass.FOLIAGE,
    SceneClass.VEHICLE,
    SceneClass.PERSON,
    SceneClass.POLE,
    SceneClass.LAMP,
    SceneClass.WATER,
)


def _names(scene_class: SceneClass, names: str) -> dict[str, SceneClass]:
    """``names``, separated by commas, each mapped to ``scene_class``."""
    return {name.strip(): scene_class for name in names.split(",")}


#: The class of each name a segmentation dataset uses, in lower case. ADE20K
#: lists synonyms after a comma ("road, route"); only the first is looked up.
#: A name that is not here is :attr:`SceneClass.OTHER`.
CLASS_OF_NAME: dict[str, SceneClass] = {
    **_names(SceneClass.SKY, "sky"),
    **_names(SceneClass.ROAD, "road, runway"),
    **_names(SceneClass.PAVEMENT, "sidewalk, pavement, floor, path, platform, parking"),
    # A mountainside, a hill and a rock face are named by the datasets too,
    # and are left as OTHER: they are not ground a camera stands over, and a
    # plane fitted to one puts the horizon in the sky.
    **_names(SceneClass.TERRAIN, "earth, sand, land, dirt track, terrain, ground"),
    **_names(SceneClass.GRASS, "grass, field"),
    **_names(SceneClass.FOLIAGE, "tree, plant, palm, flower, vegetation"),
    **_names(
        SceneClass.STRUCTURE,
        "building, wall, house, skyscraper, fence, railing, bridge, tower, column, ceiling, "
        "door, windowpane, stairs, stairway, step, awning, canopy, booth, hovel, pier, "
        "grandstand, bannister, tent, guard rail, tunnel, bench",
    ),
    **_names(
        SceneClass.VEHICLE,
        "car, bus, truck, van, minibike, bicycle, boat, ship, airplane, train, motorcycle, "
        "caravan, trailer",
    ),
    **_names(SceneClass.PERSON, "person, rider, animal"),
    **_names(SceneClass.WATER, "water, sea, river, lake, swimming pool, waterfall, fountain"),
    **_names(
        SceneClass.POLE, "pole, signboard, traffic sign, flag, bulletin board, trade name, ashcan"
    ),
    **_names(SceneClass.LAMP, "streetlight, traffic light, light, lamp, sconce, chandelier"),
}


def classes_of(names: Sequence[str]) -> np.ndarray:
    """The :class:`SceneClass` of each of a dataset's class names, ``(N,)`` uint8.

    ``names`` are a checkpoint's labels in index order. Synonyms after a
    comma and letter case are ignored; an unknown name is ``OTHER``.
    """
    table = np.zeros(len(names), dtype=np.uint8)
    for index, name in enumerate(names):
        key = str(name).split(",")[0].strip().lower()
        table[index] = int(CLASS_OF_NAME.get(key, SceneClass.OTHER))
    return table


def class_weight(
    labels: np.ndarray, classes: Iterable[SceneClass], soften_px: float = 0.0
) -> np.ndarray:
    """``(H, W)`` float32 in ``[0, 1]``: 1 where ``labels`` holds one of ``classes``.

    ``soften_px`` blurs the edge by that sigma. A label map has an outline a
    pixel wide and a network places it a few pixels off; a weight that is to
    multiply a picture wants neither the step nor the confidence.
    """
    table = np.zeros(256, dtype=np.float32)
    for scene_class in classes:
        table[int(scene_class)] = 1.0
    weight = table[labels]
    if soften_px > 0.1:
        weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=float(soften_px))
    return weight.astype(np.float32)


def check_labels(labels: Any, shape: tuple[int, int], *, source: str) -> np.ndarray:
    """``labels`` as an ``(H, W)`` uint8 map of :class:`SceneClass`, or a clear error.

    Raises ``ValueError`` when the shape is not the frame's or a value is
    not a class.
    """
    labels = np.asarray(labels)
    if labels.shape != tuple(shape):
        raise ValueError(
            f"{source} gave labels of shape {labels.shape} for a frame of shape {tuple(shape)}; "
            "scene labels are one SceneClass per pixel."
        )
    if labels.size and int(labels.max()) > max(SceneClass):
        raise ValueError(
            f"{source} gave a label of {int(labels.max())}; scene labels are SceneClass values, "
            f"0 to {int(max(SceneClass))}. Fold a dataset's own classes into them with "
            "classes_of()."
        )
    return labels.astype(np.uint8, copy=False)


class SceneLabelEstimator(ABC):
    """Interface for whatever names the pixels of a scene.

    Implementations return one :class:`SceneClass` per pixel: the frame's
    ``(H, W)`` shape, uint8. A loader of ground-truth masks is as good an
    implementation as a network, and better where there is one.
    """

    @abstractmethod
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """The :class:`SceneClass` of each pixel of an ``(H, W, 3)`` frame."""


_NO_CUDA_MESSAGE = (
    "{cls} requires a CUDA GPU: it runs a segmentation network on every scene, which is "
    "too slow on a CPU to fit in a training-time DataLoader budget. No CUDA device was "
    "found (torch.cuda.is_available() is False). Scene labels are optional: leave "
    "`labels` out of the SharedDepthProvider and every effect falls back to what it "
    "reads from depth and colour."
    " A device of 'cpu', asked for by name, is honoured: slow, but enough to render a "
    "few pictures where there is no GPU, which is what the documentation's gallery does."
)


class SegmentationLabelEstimator(SceneLabelEstimator):
    """Scene labels from a semantic segmentation checkpoint, the shipped default.

    Any checkpoint ``transformers`` loads as a semantic segmentation model
    and whose classes are named as ADE20K's or Cityscapes' are: its label
    list is folded into :class:`SceneClass` by name. The default is UperNet
    on a ConvNeXt-Tiny backbone trained on ADE20K, which knows indoor scenes
    as well as streets and is MIT-licensed; see the module docstring for the
    faster alternative and its licence.

    Args:
        model_name: the checkpoint.
        device: where to run it. CUDA, for the reason
            :class:`~surmentations.depth.estimator.DepthAnythingV2Estimator`
            gives, unless ``"cpu"`` is asked for by name.
        half_precision: run the network in float16. ``None`` follows the
            mode (:mod:`surmentations.mode`): half in fast mode, full in
            exact mode.
        min_confidence: a pixel whose winning class holds less than this
            share of the network's belief is ``OTHER``. The effects treat
            ``OTHER`` as "no opinion", which is the right reading of a
            network that has none.

    The frame is resized to the size the checkpoint was trained at, whatever
    its shape, as the checkpoint's own preprocessing does. The network's
    scores are folded into the dozen classes *before* they are enlarged to
    the frame, so a full-size frame costs thirteen planes and not a hundred
    and fifty.
    """

    def __init__(
        self,
        model_name: str = "openmmlab/upernet-convnext-tiny",
        device: str = "cuda",
        half_precision: bool | None = None,
        min_confidence: float = 0.35,
    ) -> None:
        """Load the checkpoint onto the GPU; see the class docstring for the arguments.

        Raises ``RuntimeError`` where torch has no CUDA device, and
        ``ValueError`` for a checkpoint none of whose classes is known. torch
        and transformers are imported here and not at module level.
        """
        import torch

        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(_NO_CUDA_MESSAGE.format(cls=type(self).__name__))
        if not 0.0 <= min_confidence < 1.0:
            raise ValueError(f"min_confidence must be in [0, 1), got {min_confidence}")
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        self.model_name = model_name
        self.half_precision = half_precision
        self.min_confidence = float(min_confidence)
        self.device = torch.device(device)
        processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = (
            AutoModelForSemanticSegmentation.from_pretrained(model_name).to(self.device).eval()
        )
        id2label = self._model.config.id2label
        names = [str(id2label[index]) for index in range(len(id2label))]
        table = classes_of(names)
        if not table.any():
            raise ValueError(
                f"none of the {len(names)} classes of {model_name} is one this library knows "
                "(it reads ADE20K's and Cityscapes' names); see CLASS_OF_NAME."
            )
        # One row per scene class, one column per class of the network: the
        # network's belief in each of its classes summed into ours.
        fold = np.zeros((len(SceneClass), len(names)), dtype=np.float32)
        fold[table, np.arange(len(names))] = 1.0
        self._fold = torch.from_numpy(fold).to(self.device)
        size = getattr(processor, "size", None)
        height = size.get("height") if isinstance(size, dict) else getattr(size, "height", None)
        width = size.get("width") if isinstance(size, dict) else getattr(size, "width", None)
        self._input_size = (int(height or 512), int(width or 512))
        per_unit = 1.0 / float(getattr(processor, "rescale_factor", 1.0 / 255.0))
        self._mean = (torch.tensor(processor.image_mean, device=self.device) * per_unit).view(
            1, 3, 1, 1
        )
        self._std = (torch.tensor(processor.image_std, device=self.device) * per_unit).view(
            1, 3, 1, 1
        )

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """The :class:`SceneClass` of each pixel of one ``(H, W, 3)`` frame, ``(H, W)`` uint8.

        A float frame is taken as ``[0, 1]`` and quantised to 8 bits first.
        """
        import torch

        height, width = image.shape[:2]
        if image.dtype != np.uint8:
            image = np.clip(np.rint(image.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
        # Reduced on the CPU by area, which is what a frame several times the
        # network's size needs, and is a few hundred microseconds.
        rows, columns = self._input_size
        shrink = cv2.INTER_AREA if height >= rows and width >= columns else cv2.INTER_LINEAR
        small = cv2.resize(image, (columns, rows), interpolation=shrink)
        with torch.inference_mode():
            pixels = torch.from_numpy(small).to(self.device).permute(2, 0, 1).unsqueeze(0)
            pixels = (pixels.float() - self._mean) / self._std
            half = mode.fast() if self.half_precision is None else self.half_precision
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=half):
                logits = self._model(pixel_values=pixels).logits
            belief = torch.softmax(logits.float(), dim=1)
            # Folded where the planes are small, then enlarged.
            ours = torch.einsum("kc,bchw->bkhw", self._fold, belief)
            ours = torch.nn.functional.interpolate(
                ours, size=(height, width), mode="bilinear", align_corners=False
            )
            confidence, winner = ours.max(dim=1)
            winner = torch.where(
                confidence >= self.min_confidence, winner, torch.zeros_like(winner)
            )
            labels: np.ndarray = winner.reshape(height, width).to(torch.uint8).cpu().numpy()
        return labels
