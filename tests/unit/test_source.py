"""What a frame already is decides what a pipeline may do to it.

A frame taken at night is not made into a night again, and it does not
rain under a roof: `surmentations.presets.source` reads which a frame is,
and `surveillance` sends it down the branch for that.
"""

from __future__ import annotations

import numpy as np

from surmentations.depth.estimator import DepthEstimator
from surmentations.depth.provider import SharedDepthProvider
from surmentations.diagnostics import recording
from surmentations.perception import SceneClass, SceneLabelEstimator
from surmentations.presets import read_source, surveillance

HEIGHT, WIDTH = 96, 96
WEATHER = {"RandomRain", "RandomSnow", "RandomFog", "NightRain", "NightSnow", "NightFog"}
NIGHT_MADE = {"NightIrMode", "NightColorMode"}
SUN = {"SunFlare", "CastShadows"}


class _Road(DepthEstimator):
    """A level road under the lower half of the frame."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Depth of a road under the lower half of the frame, and a far wall above it."""
        height, width = image.shape[:2]
        rows = np.arange(height, dtype=np.float32).reshape(height, 1)
        depth = np.clip(1.5 * 300.0 / np.maximum(rows - height / 2.0, 1e-3), 0.0, 50.0)
        return np.broadcast_to(depth / 50.0, (height, width)).astype(np.float32)


class _Named(SceneLabelEstimator):
    """Labels that are told, whatever the frame."""

    def __init__(self, labels: np.ndarray) -> None:
        """Keep the labels."""
        self.labels = labels

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """The labels."""
        return self.labels


def _day() -> np.ndarray:
    """A grey daylight frame."""
    return np.full((HEIGHT, WIDTH, 3), 120, dtype=np.uint8)


def _night() -> np.ndarray:
    """A frame that is nearly black, with a lamp and what it lights."""
    frame = np.full((HEIGHT, WIDTH, 3), 6, dtype=np.uint8)
    frame[20:24, 40:44] = 255
    frame[60:90, 20:70] = 40
    return frame


def _fired(transform: object, image: np.ndarray, samples: int = 250) -> list[set[str]]:
    """Which transforms fired on each of many draws."""
    transform.set_random_seed(7)  # type: ignore[attr-defined]
    drawn = []
    for _ in range(samples):
        with recording() as log:
            transform(image=image)  # type: ignore[operator]
        drawn.append({record["t"] for record in log})
    return drawn


def test_a_frame_says_whether_it_is_a_night_and_its_labels_whether_it_is_indoors() -> None:
    assert read_source(_day()) == (False, False)
    assert read_source(_night()) == (True, False)
    # In deep shade by day, a frame is dark and is not a night.
    shade = np.full((HEIGHT, WIDTH, 3), 55, dtype=np.uint8)
    assert not read_source(shade).night
    street = np.full((HEIGHT, WIDTH), int(SceneClass.ROAD), dtype=np.uint8)
    street[:30] = int(SceneClass.SKY)
    garage = np.full((HEIGHT, WIDTH), int(SceneClass.STRUCTURE), dtype=np.uint8)
    garage[60:] = int(SceneClass.PAVEMENT)
    assert not read_source(_day(), street).indoors
    assert read_source(_day(), garage).indoors
    # With no labels nothing says a frame is under a roof.
    assert not read_source(_day(), None).indoors


def test_a_night_frame_is_not_made_into_a_night_again_and_has_no_sun() -> None:
    options = {"codec": 0.0, "rain": 0.2, "fog": 0.1}
    transform = surveillance(SharedDepthProvider(_Road()), **options)
    by_day = _fired(transform, _day())
    assert any(s & NIGHT_MADE for s in by_day) and any(s & SUN for s in by_day)
    at_night = _fired(transform, _night())
    assert not [s for s in at_night if s & NIGHT_MADE], "a night lit a second time"
    assert not [s for s in at_night if s & SUN], "the sun in a frame taken at night"
    assert any(s & WEATHER for s in at_night), "a night frame has weather too"
    assert not [s for s in at_night if "RandomFog" in s], "daylight fog on a night frame"
    # Told not to look, it takes the night frame for a day one.
    blind = surveillance(SharedDepthProvider(_Road()), read_source=False, **options)
    assert any(s & NIGHT_MADE for s in _fired(blind, _night()))


def test_it_does_not_rain_under_a_roof() -> None:
    garage = np.full((HEIGHT, WIDTH), int(SceneClass.STRUCTURE), dtype=np.uint8)
    garage[60:] = int(SceneClass.PAVEMENT)
    provider = SharedDepthProvider(_Road(), labels=_Named(garage))
    transform = surveillance(provider, codec=0.0, rain=0.3, fog=0.2, snow=0.1)
    drawn = _fired(transform, _day())
    assert not [s for s in drawn if s & (WEATHER | NIGHT_MADE | SUN)]
    assert not [s for s in drawn if "WaterDroplets" in s]
    assert any(drawn), "the camera's own chain still runs"
