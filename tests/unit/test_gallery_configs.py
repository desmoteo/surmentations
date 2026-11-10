"""The documentation gallery must show effects that are actually visible.

Every other test here checks the library. Nothing checked the *configuration
the gallery builds*, and that gap hid two separate bugs that reached the
rendered images:

* `housing_occluders` drew its counts from a range starting at zero, so three
  of the five published frames were bit-identical to the input.
* `ac_flicker`'s variant override hardcoded `readout_time_range_s`, copied
  from an older library default. When that default was widened to get a
  legible band count, the override silently kept the gallery on the old
  range — so the fix never reached a single rendered frame, twice in a row,
  while the library's own tests passed throughout.

An override that duplicates a range the transform already owns is a copy
waiting to go stale, and the only thing that catches it is rendering what the
gallery will actually render.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from docs.generate_examples import EFFECTS, LEVELS, _build
from tests.unit.test_randomization import _GroundPlaneDepth
from weathermentations.depth.provider import SharedDepthProvider

# One 8-bit level of mean absolute change, the threshold used throughout this
# review for "this does something".
VISIBLE_LEVELS = 1.0
SEEDS = 8

# Needs an ffmpeg binary and two subprocess round-trips per render.
SKIP = {
    "video_codec",
    # Acts only on the lights in a frame. The gallery shows it over a night
    # street; over this daylit photograph it rightly does nothing.
    "source_glare",
}

REFERENCE_IMAGE = Path(__file__).resolve().parents[2] / "docs" / "examples" / "reference.webp"


@pytest.fixture(scope="module")
def scene() -> np.ndarray:
    """The reference photograph, which is what the gallery actually renders.

    Not the synthetic scene the golden tests use: this test asks whether a
    published frame is visible, so it has to ask it of the published frame's
    input. Effect strength genuinely depends on scene content — a
    multiplicative effect like AC flicker moves a bright photograph further
    than a darker synthetic one, and `DirtSoil` scatters rather than tints and
    so is near-invisible in flat light by design.
    """
    image = cv2.imread(str(REFERENCE_IMAGE))
    if image is None:  # pragma: no cover - the asset ships with the repo
        pytest.skip(f"reference image not found at {REFERENCE_IMAGE}")
    image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@pytest.mark.parametrize("name", [n for n in EFFECTS if n not in SKIP])
def test_gallery_configuration_renders_a_visible_effect(name: str, scene: np.ndarray) -> None:
    """Every published cell must change the image, at every severity.

    Averaged over seeds rather than asserted per seed: some effects can
    legitimately draw themselves out of existence — an exposure landing on a
    whole flicker period cancels AC banding exactly — and that is physics
    worth keeping. What must not happen is a *configuration* whose typical
    draw does nothing.
    """
    provider = SharedDepthProvider(_GroundPlaneDepth())
    rng = np.random.default_rng(1234)
    reference = scene.astype(np.float32)

    for index, level in enumerate(LEVELS):
        transform = _build(name, level, index, rng, provider)
        deltas = []
        for seed in range(SEEDS):
            transform.set_random_seed(seed)
            out = transform(image=scene)["image"].astype(np.float32)
            deltas.append(float(np.abs(out - reference).mean()))
        typical = float(np.median(deltas))
        assert typical >= VISIBLE_LEVELS, (
            f"{name} severity {level} as the gallery builds it changes the image by "
            f"{typical:.2f} levels (median of {SEEDS} seeds) — below the visible "
            f"threshold of {VISIBLE_LEVELS}. Values: {[round(d, 2) for d in deltas]}"
        )
