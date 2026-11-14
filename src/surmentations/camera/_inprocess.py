"""H.264 and H.265 inside this process, through PyAV where it is installed.

The codec transforms encode with the ``ffmpeg`` binary: two processes started
per frame, which is 120 ms of a 150 ms call before a pixel is compressed.
PyAV (``pip install surmentations[codec]``) binds the same FFmpeg
libraries into Python, and the round trip is then the encoder's own time: 10
ms for a 512-pixel H.264 frame.

**It is the same picture, where that can be had.** FFmpeg's command line and
PyAV share libswscale and libx264, and three settings make them agree to the
byte: the colour conversion's interpolation (the binary's is bicubic, PyAV's
default bilinear), one encoder thread (PyAV's default lets libx264 cut the
frame into slices, one per thread), and the binary's frame rate. With those,
75 of 75 H.264 round trips tried came back identical to the binary's, three
images at five sizes and five qualities, and so did the last frame of every
damaged stream. H.265 does not: the two carry different releases of libx265,
which make different decisions.

That equality rests on two builds of FFmpeg that this library controls
neither of, so it is never assumed. ``codec.reproduces_binary`` tries it
once per encoder on a small frame, and the callers encode here in exact mode
(:mod:`surmentations.mode`) only where it passed. In fast mode this
module is used whenever PyAV is there, H.265 included, and a frame is then
as much real H.265 as the binary's, from another release of the encoder.

**Decoding a damaged stream is the binary's in exact mode, always.** With
error concealment on, the two decoders gave the same frame every time on
x86-64; on arm64 they did not, by a level or two. With it off, a lost
slice shows whatever the decoder's frame buffer held: here that is the
zeroed, green hole every time, and from the binary sometimes
that and sometimes a stale picture, depending on which buffer it reused. 15
of 120 streams differed for that reason and no other. Neither is wrong, and
neither can be held to the other.
"""

from __future__ import annotations

import os
from fractions import Fraction
from functools import lru_cache
from typing import Any

import numpy as np

# The decoders the two encoders' streams are read back with.
_DECODERS = {"libx264": "h264", "libx265": "hevc"}

# What the ffmpeg binary assumes of raw frames piped to it.
_FRAME_RATE = 25


@lru_cache(maxsize=1)
def _av() -> Any:
    """The ``av`` module, or ``None`` where PyAV is not installed or is switched off."""
    if os.environ.get("SURMENTATIONS_CODEC", "").lower() == "ffmpeg":
        return None
    try:
        import av
        import av.video.reformatter  # noqa: F401  (not imported by `av` in every release)
    except Exception:  # an ImportError, or a wheel whose libraries do not load here
        return None
    return av


def available() -> bool:
    """Whether frames can be encoded in this process."""
    return _av() is not None


def has_encoder(encoder: str) -> bool:
    """Whether this PyAV's FFmpeg was built with ``encoder``."""
    av = _av()
    if av is None:
        return False
    try:
        av.codec.Codec(encoder, "w")
    except Exception:
        return False
    return True


def _to_yuv(av: Any, rgb: np.ndarray, index: int) -> Any:
    """An RGB frame as the 4:2:0 frame the binary would hand the encoder."""
    bicubic = av.video.reformatter.Interpolation.BICUBIC
    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
    frame = frame.reformat(format="yuv420p", interpolation=bicubic)
    frame.pts = index
    return frame


def _to_rgb(av: Any, frame: Any) -> np.ndarray:
    """A decoded frame as ``(H, W, 3)`` uint8 RGB, converted bicubically as the binary does."""
    bicubic = av.video.reformatter.Interpolation.BICUBIC
    rgb: np.ndarray = frame.reformat(format="rgb24", interpolation=bicubic).to_ndarray()
    return np.ascontiguousarray(rgb)


def _encode(
    av: Any, frames: np.ndarray, encoder: str, options: dict[str, str], keyframes: tuple[int, ...]
) -> list[Any]:
    """Encode ``(N, H, W, 3)`` uint8 frames; the packets the encoder wrote.

    On one thread, at the binary's frame rate, with the frames listed in
    ``keyframes`` forced to be intra-coded.
    """
    count, height, width = frames.shape[:3]
    context = av.CodecContext.create(encoder, "w")
    context.width, context.height, context.pix_fmt = width, height, "yuv420p"
    context.time_base = Fraction(1, _FRAME_RATE)
    context.framerate = Fraction(_FRAME_RATE, 1)
    # One thread: with more, libx264 cuts each frame into a slice a thread,
    # which is another stream and another picture. A DataLoader worker has
    # one core to give it in any case.
    context.thread_count = 1
    context.options = options
    packets = []
    for index in range(count):
        frame = _to_yuv(av, frames[index], index)
        if index in keyframes:
            frame.pict_type = av.video.frame.PictureType.I
        packets.extend(context.encode(frame))
    packets.extend(context.encode(None))
    return packets


def _decode(av: Any, packets: list[Any], decoder: str, options: dict[str, str]) -> list[Any]:
    """Decode packets on one thread; every frame the decoder put out.

    A packet the decoder gives up on is skipped, as the binary skips it.
    """
    context = av.CodecContext.create(decoder, "r")
    context.thread_count = 1
    if options:
        context.options = options
    frames = []
    for packet in [*packets, None]:
        try:
            frames.extend(context.decode(packet))
        except av.error.FFmpegError:
            # A damaged picture the decoder gave up on: the binary goes on to
            # the next one, and so does this.
            continue
    return frames


def roundtrip(rgb: np.ndarray, encoder: str, crf: int) -> np.ndarray | None:
    """Encode one ``(H, W, 3)`` uint8 frame of even size and decode it back.

    ``None`` if PyAV is missing, was built without ``encoder``, or failed.
    """
    av = _av()
    if av is None:
        return None
    options = {"crf": str(crf)}
    if encoder == "libx265":
        # libx265 reports every frame on stderr, and starts a thread for
        # every core of the machine to encode it with. Neither changes the
        # picture; a pool of one is the picture of the default.
        options["x265-params"] = "log-level=none:pools=1"
    try:
        packets = _encode(av, rgb[None], encoder, options, keyframes=())
        frames = _decode(av, packets, _DECODERS[encoder], {})
        return _to_rgb(av, frames[0]) if frames else None
    except (av.error.FFmpegError, ValueError, KeyError):
        return None


def encode_clip(frames: np.ndarray, crf: int, slices: int) -> bytes | None:
    """``transmission._encode_clip`` in this process: raw H.264, Annex B.

    The slices are the binary's to the byte. The stream is not: it opens
    with a message naming the libx264 release that wrote it, which is not a
    picture and is not put at risk of loss.
    """
    av = _av()
    if av is None:
        return None
    options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "crf": str(crf),
        "bf": "0",
        "g": str(len(frames) + 1),
        "sc_threshold": "0",
        "forced-idr": "1",
        "slices": str(slices),
    }
    try:
        packets = _encode(av, frames, "libx264", options, keyframes=(1,))
    except (av.error.FFmpegError, ValueError):
        return None
    return b"".join(bytes(packet) for packet in packets)


def decode_last_frame(
    stream: bytes, height: int, width: int, conceal: bool = True
) -> np.ndarray | None:
    """``transmission._decode_last_frame`` in this process."""
    av = _av()
    if av is None:
        return None
    options = {
        # Decode what is there and show it, as a player does: a damaged
        # picture, not a stopped one.
        "err_detect": "ignore_err",
        "ec": "guess_mvs+deblock" if conceal else "0",
        "flags": "+output_corrupt",
    }
    try:
        parser = av.CodecContext.create("h264", "r")
        packets = [*parser.parse(stream), *parser.parse(b"")]
        frames = _decode(av, packets, "h264", options)
        if not frames:
            return None
        last = _to_rgb(av, frames[-1])
    except (av.error.FFmpegError, ValueError):
        return None
    return last if last.shape == (height, width, 3) else None
