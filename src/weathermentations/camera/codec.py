"""Video-codec artefacts, produced by actually encoding rather than approximating.

Why not JPEG
------------
Augmentation pipelines reach for JPEG compression to stand in for "the image
was compressed". For a surveillance stream that is the wrong artefact. A
stream is H.264 or H.265 at a low bitrate, and it fails differently in ways
that matter:

* **A deblocking filter runs inside the codec.** H.264 smooths its own block
  boundaries as part of decoding, so the result is not the hard 8x8 grid JPEG
  leaves — it is blocky *and* smeared, with the smearing strongest exactly
  where the blocking would have been.
* **Chroma is subsampled 4:2:0** and quantised separately, so colour degrades
  faster than luminance and bleeds across edges. Saturated reds suffer worst.
  Measured on the reference photograph at CRF 30: luminance keeps 0.73 of its
  fine detail while the chroma channels keep 0.24 and 0.33. (This is a
  property of *natural* content — on synthetic images with uncorrelated
  channels, chroma carries as much real high-frequency structure as luma and
  the gap closes, which is why the unit tests do not assert it.)
* **Rate control decides the damage, not a quality dial.** The encoder spends
  its bits where its own model says they matter, so a busy frame is
  quantised harder than a flat one — the artefact is content-dependent in a
  way a fixed quality setting is not.

Rather than approximate any of that, this transform **encodes the frame with
ffmpeg and decodes it back**. Everything above then comes out correct for
free, including the parts an approximation gets wrong.

Inter-frame damage
------------------
The other half of what a stream does to an image has no single-frame
equivalent at all: when a P-frame is lost, the decoder keeps displaying stale
macroblocks from an older reference while the rest of the picture updates, so
regions of the frame are *out of date* rather than merely degraded.

That is simulated here by compositing macroblock-aligned regions from a
second, much harder-quantised encode of the same frame. It is labelled a
stand-in and not presented as the real thing: genuine staleness needs the
previous frame, which a single image does not have. What it does reproduce
faithfully is the *spatial signature* — sharp macroblock-aligned boundaries
between regions of differing quality, which is what a detector actually trips
over.

Cost
----
Two ffmpeg subprocesses per call, 150 ms or so, nearly all of it starting
them. That is expensive next to the array operations elsewhere in this
library, so keep ``p`` low rather than paying it on every sample, or install
PyAV (``pip install weathermentations[codec]``): the frame is then encoded in
this process, in 10 ms for H.264 at 512 pixels, and for H.264 it is the same
frame to the byte. See :mod:`weathermentations.camera._inprocess`.
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from typing import Annotated, Any

import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from weathermentations import mode
from weathermentations.base import WeatherTransform
from weathermentations.camera import _inprocess

# H.264 and H.265 both code in 16x16 macroblocks (H.265 calls them coding
# tree units and allows larger, but 16 is the common denominator and is what
# the visible block grid lands on).
MACROBLOCK_PX = 16

_CODECS = {"h264": "libx264", "h265": "libx265"}


@lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    """Location of the ffmpeg binary, or ``None`` if it is not installed."""
    return shutil.which("ffmpeg")


def codec_available(encoder: str = "libx264") -> bool:
    """Whether there is any way here to encode with ``encoder``.

    The ``ffmpeg`` binary, or PyAV in this process.
    """
    return ffmpeg_path() is not None or _inprocess.has_encoder(encoder)


# Whether PyAV has been seen to give the binary's frame on this installation,
# by encoder: absent until it has been tried.
_inprocess_is_binary: dict[str, bool] = {}


def reproduces_binary(encoder: str) -> bool:
    """Whether encoding in this process gives the ``ffmpeg`` binary's frame.

    Tried once per encoder on a small frame at two qualities and not assumed:
    it rests on two builds of FFmpeg agreeing. See
    :mod:`weathermentations.camera._inprocess`.
    """
    if encoder not in _inprocess_is_binary:
        probe = np.random.default_rng(0).integers(0, 256, (12, 16, 3), dtype=np.uint8)
        probe = np.ascontiguousarray(np.kron(probe, np.ones((4, 4, 1), dtype=np.uint8)))
        same = True
        for crf in (24, 38):
            ours = _inprocess.roundtrip(probe, encoder, crf)
            theirs = _binary_roundtrip(probe, encoder, crf)
            if ours is None or theirs is None or not np.array_equal(ours, theirs):
                same = False
        _inprocess_is_binary[encoder] = same
    return _inprocess_is_binary[encoder]


def use_inprocess(encoder: str) -> bool:
    """Whether a frame is to be encoded in this process and not by the binary.

    Where PyAV has ``encoder`` and one of: there is no binary; the mode is
    fast (:mod:`weathermentations.mode`); or PyAV gives the binary's frame
    here, so that it is the reference by another route.
    """
    if not _inprocess.has_encoder(encoder):
        return False
    return ffmpeg_path() is None or mode.fast() or reproduces_binary(encoder)


def _run_roundtrip(rgb: np.ndarray, encoder: str, crf: int) -> np.ndarray | None:
    """Encode one frame and decode it back. ``None`` on failure."""
    height, width = rgb.shape[:2]
    # Codecs need even dimensions for 4:2:0 chroma; pad and crop back.
    pad_h, pad_w = height % 2, width % 2
    if pad_h or pad_w:
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    out = None
    if use_inprocess(encoder):
        out = _inprocess.roundtrip(np.ascontiguousarray(rgb), encoder, crf)
    if out is None:
        out = _binary_roundtrip(rgb, encoder, crf)
    return None if out is None else np.ascontiguousarray(out[:height, :width])


def _binary_roundtrip(rgb: np.ndarray, encoder: str, crf: int) -> np.ndarray | None:
    """Encode one frame of even size with the ffmpeg binary and decode it back."""
    binary = ffmpeg_path()
    if binary is None:
        return None
    padded_h, padded_w = rgb.shape[:2]

    encode = [
        binary, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{padded_w}x{padded_h}", "-i", "pipe:0",
        "-c:v", encoder, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-frames:v", "1", "-f", "matroska", "pipe:1",
    ]
    decode = [
        binary, "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]

    try:
        encoded = subprocess.run(
            encode, input=rgb.tobytes(), capture_output=True, timeout=60, check=True
        ).stdout
        decoded = subprocess.run(
            decode, input=encoded, capture_output=True, timeout=60, check=True
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None

    expected = padded_h * padded_w * 3
    if len(decoded) < expected:
        return None
    return np.frombuffer(decoded[:expected], dtype=np.uint8).reshape(padded_h, padded_w, 3)


class VideoCodecArtifacts(WeatherTransform):
    """H.264/H.265 compression at a target quality, encoded for real.

    Args:
        codec: ``"h264"`` or ``"h265"``. H.265 holds detail better at the same
            bitrate but its artefacts are smoother and larger-scale, so the
            two do not look alike at matched quality.
        crf_range: constant-rate-factor range. Lower is better quality; 23 is
            the x264 default, and surveillance streams commonly run 28-38.
        stale_block_probability: chance of simulating a lost reference frame.
        stale_block_fraction_range: fraction of the frame left stale when that
            happens.
        stale_crf_delta_range: how much worse the stale regions are, in CRF
            points.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        codec: str
        crf_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 51)),
        ]
        stale_block_probability: float = Field(ge=0.0, le=1.0)
        stale_block_fraction_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]
        stale_crf_delta_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 40)),
        ]

    def __init__(
        self,
        codec: str = "h264",
        crf_range: tuple[int, int] = (26, 40),
        stale_block_probability: float = 0.25,
        stale_block_fraction_range: tuple[float, float] = (0.03, 0.25),
        stale_crf_delta_range: tuple[int, int] = (8, 18),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        if codec not in _CODECS:
            raise ValueError(f"codec must be one of {sorted(_CODECS)}, got {codec!r}")
        if not codec_available(_CODECS[codec]):
            raise RuntimeError(
                "VideoCodecArtifacts needs the `ffmpeg` binary on PATH or PyAV, which is "
                "how it produces real H.264/H.265 artefacts instead of approximating them. "
                "Install one (apt install ffmpeg / brew install ffmpeg, or "
                "pip install weathermentations[codec]), or use "
                "albumentations' ImageCompression if JPEG artefacts are acceptable — "
                "note they are a different artefact, not a weaker version of this one."
            )
        self.codec = codec
        self.crf_range = crf_range
        self.stale_block_probability = stale_block_probability
        self.stale_block_fraction_range = stale_block_fraction_range
        self.stale_crf_delta_range = stale_crf_delta_range

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        del data
        crf = self.py_random.randint(*self.crf_range)
        stale = self.py_random.random() < self.stale_block_probability
        return {
            "crf": crf,
            "stale_crf": min(51, crf + self.py_random.randint(*self.stale_crf_delta_range)),
            "stale_fraction": (
                self.py_random.uniform(*self.stale_block_fraction_range) if stale else 0.0
            ),
            "block_seed": self.py_random.randint(0, 2**31 - 1),
        }

    def apply(
        self,
        img: np.ndarray,
        crf: int,
        stale_crf: int,
        stale_fraction: float,
        block_seed: int,
        **params: Any,
    ) -> np.ndarray:
        is_integer = np.issubdtype(img.dtype, np.integer)
        rgb = img if is_integer else np.clip(img * 255.0, 0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)

        encoder = _CODECS[self.codec]
        compressed = _run_roundtrip(rgb, encoder, crf)
        if compressed is None:
            # ffmpeg vanished or failed mid-run. Returning the input unchanged
            # is the only honest option: silently substituting JPEG would put a
            # different artefact in the training set under this transform's name.
            return img

        if stale_fraction > 1e-3:
            stale = _run_roundtrip(rgb, encoder, stale_crf)
            if stale is not None:
                height, width = rgb.shape[:2]
                rows = (height + MACROBLOCK_PX - 1) // MACROBLOCK_PX
                cols = (width + MACROBLOCK_PX - 1) // MACROBLOCK_PX
                rng = np.random.default_rng(block_seed)
                # Whole macroblocks, because that is the granularity a decoder
                # actually holds stale — the hard block-aligned boundary is the
                # part a detector trips over.
                keep_stale = rng.random((rows, cols)) < stale_fraction
                mask = np.kron(keep_stale, np.ones((MACROBLOCK_PX, MACROBLOCK_PX), dtype=bool))
                mask = mask[:height, :width]
                compressed = np.where(mask[..., None], stale, compressed)

        if is_integer:
            return compressed.astype(img.dtype)
        return (compressed.astype(np.float32) / 255.0).astype(img.dtype)
