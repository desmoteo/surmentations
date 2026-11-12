"""Packet loss: what a damaged stream decodes to.

``VideoCodecArtifacts`` is compression — a stream that arrived whole. This is
the other thing a link does to a picture, and the one operators report: a
band of the frame that is out of date, smeared downwards, or filled with
blocks of the wrong colour. It is common on wireless and congested links,
where RTSP runs over UDP and a lost packet is simply gone.

Nothing here draws those artefacts. The transform **encodes a short H.264
clip, removes packets from the bitstream, and decodes what is left** with
the decoder's error concealment on, so what comes out is what a decoder
makes of that loss:

* **A slice whose packets are missing is concealed, or is not.** A decoder
  that conceals fills the hole from the last picture it has, shifted by its
  guess at the motion. On a still scene that is nearly invisible, which is
  the truth about packet loss and the reason it is noticed on moving things
  first. A decoder that does not conceal leaves the hole as the zeroed
  memory it is, and zero in YUV is green.
* **A slice that arrives with a gap in it decodes to garbage.** Everything
  after the gap is entropy-coded against state the decoder no longer has.
  It runs on, producing blocks of saturated colour, until it notices or the
  slice ends: damage starts mid-row and stops at a slice boundary.
* **With nothing to conceal from, it smears.** A decoder that joined the
  stream after an outage has no earlier picture, and fills a missing slice
  by stretching its neighbours' edges across it.
* **Damage persists.** Every later frame is predicted from the damaged one,
  so the fault stays, drifting and picking up corrections, until the next
  keyframe. The frame returned is the last of its group, however old the
  damage is by then.

What a single image cannot supply is the past. Concealment copies from an
earlier frame, and what makes that visible is the difference between then
and now. So the clip is built by sliding the picture: the reference frame
is the image displaced by ``scene_motion_range``, and it moves into place
over the group. That stands in for a camera that swayed or a scene that
moved; the last frame is the image where it was given, so labels still
fit. It does not put a car where the car was two seconds ago.

Cost
----
Two ffmpeg subprocesses and a clip of up to a handful of frames per call:
a few hundred milliseconds. Keep ``p`` low.
"""

from __future__ import annotations

import re
import subprocess
from typing import Annotated, Any

import cv2
import numpy as np
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from pydantic import AfterValidator, Field

from surmentations import mode
from surmentations.base import CameraTransform
from surmentations.camera import _inprocess
from surmentations.camera import codec as _codec
from surmentations.diagnostics import record_draw

# Payload bytes per packet. RTP over Ethernet carries about 1400; streams
# sent across tunnels and cellular links are configured lower.
_PACKET_BYTES = 1200

_START_CODE = b"\x00\x00\x00\x01"

# NAL unit types that carry picture data: a slice of an ordinary picture and
# a slice of a keyframe. Everything else is a parameter set or a message,
# which cameras resend and which is not put at risk here.
_SLICE_TYPES = (1, 5)


def split_nal_units(stream: bytes) -> list[bytes]:
    """The NAL units of an Annex B H.264 stream, without their start codes."""
    parts = re.split(b"\x00\x00\x01", stream)[1:]
    # A four-byte start code leaves its leading zero on the unit before it.
    return [
        part[:-1] if index < len(parts) - 1 and part.endswith(b"\x00") else part
        for index, part in enumerate(parts)
    ]


def _is_slice(unit: bytes) -> bool:
    """Whether a NAL unit carries picture data: a slice of an ordinary picture or a keyframe."""
    return bool(unit) and (unit[0] & 0x1F) in _SLICE_TYPES


def _starts_picture(unit: bytes) -> bool:
    """Whether a slice is the first of its picture.

    A slice header opens with ``first_mb_in_slice`` as an Exp-Golomb code,
    and the code for zero is a single set bit.
    """
    return len(unit) > 1 and bool(unit[1] & 0x80)


def packet_losses(
    rng: np.random.Generator, packet_count: int, loss: float, mean_burst: float
) -> np.ndarray:
    """Which of ``packet_count`` packets are lost, as a boolean array.

    Loss on a real link comes in bursts — a fade, a full queue — so packets
    are not dropped independently: a burst starts with a probability chosen
    to give the asked-for loss overall, and lasts a geometrically
    distributed number of packets with the given mean. At least one packet
    is always lost when ``loss`` is positive: whether the transform applies
    is what ``p`` is for.
    """
    lost = np.zeros(packet_count, dtype=bool)
    if packet_count == 0 or loss <= 0.0:
        return lost
    start = min(1.0, loss / max(mean_burst, 1.0))
    index = 0
    while index < packet_count:
        if rng.random() < start:
            length = int(rng.geometric(1.0 / max(mean_burst, 1.0)))
            lost[index : index + length] = True
            index += length
        else:
            index += 1
    if not lost.any():
        lost[int(rng.integers(packet_count))] = True
    return lost


def drop_packets(
    units: list[bytes],
    rng: np.random.Generator,
    loss: float,
    mean_burst: float,
    keep_reference: bool,
    pass_gaps: bool,
) -> tuple[list[bytes], int, int]:
    """Send a stream's NAL units over a lossy link.

    The first picture is the reference the decoder is taken to hold already.
    It is delivered whole, or with ``keep_reference`` false not at all; the
    pictures after it are cut into packets and lose some.

    A unit whose first packet is lost cannot be reassembled and is dropped.
    One that loses a later packet is cut short there, or with ``pass_gaps``
    is handed on with the gap closed up, which is what a receiver that does
    not check sequence numbers does and is where garbage comes from.

    Returns the surviving units and the counts of packets sent and lost.
    """
    pictures_seen = 0
    at_risk: list[int] = []
    kept: list[bytes | None] = []
    for unit in units:
        if not _is_slice(unit):
            kept.append(unit)
            continue
        if _starts_picture(unit):
            pictures_seen += 1
        if pictures_seen <= 1:
            kept.append(unit if keep_reference else None)
            continue
        at_risk.append(len(kept))
        kept.append(unit)

    counts = [max(1, -(-len(kept[i] or b"") // _PACKET_BYTES)) for i in at_risk]
    lost = packet_losses(rng, int(sum(counts)), loss, mean_burst)
    offset = 0
    for index, count in zip(at_risk, counts, strict=True):
        gone = lost[offset : offset + count]
        offset += count
        if not gone.any():
            continue
        unit = kept[index] or b""
        if gone[0]:
            kept[index] = None
        elif pass_gaps:
            kept[index] = b"".join(
                unit[i * _PACKET_BYTES : (i + 1) * _PACKET_BYTES]
                for i in range(count)
                if not gone[i]
            )
        else:
            kept[index] = unit[: int(np.argmax(gone)) * _PACKET_BYTES]
    return [unit for unit in kept if unit], int(lost.size), int(lost.sum())


def _encode_clip(frames: np.ndarray, crf: int, slices: int) -> bytes | None:
    """Encode ``(N, H, W, 3)`` frames as raw H.264, keyframes on the first two.

    The second frame is a keyframe so that the first can stand for "the
    previous group of pictures": something the decoder has, that the stream
    at risk does not depend on until a packet is lost.
    """
    if _encode_inprocess():
        stream = _inprocess.encode_clip(frames, crf, slices)
        if stream is not None:
            return stream
    return _binary_encode_clip(frames, crf, slices)


def _binary_encode_clip(frames: np.ndarray, crf: int, slices: int) -> bytes | None:
    """``_encode_clip`` with the ``ffmpeg`` binary; ``None`` if there is none or it fails."""
    binary = _codec.ffmpeg_path()
    if binary is None:
        return None
    count, height, width = frames.shape[:3]
    command = [
        binary, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "25",
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-crf", str(crf), "-bf", "0", "-g", str(count + 1), "-sc_threshold", "0",
        "-force_key_frames", "expr:eq(n,1)", "-forced-idr", "1",
        "-slices", str(slices), "-threads", "1",
        "-pix_fmt", "yuv420p", "-f", "h264", "pipe:1",
    ]  # fmt: skip
    try:
        return subprocess.run(
            command, input=frames.tobytes(), capture_output=True, timeout=120, check=True
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None


def _decode_last_frame(
    stream: bytes, height: int, width: int, conceal: bool = True
) -> np.ndarray | None:
    """Decode a possibly damaged stream and return its final frame."""
    if _decode_inprocess():
        frame = _inprocess.decode_last_frame(stream, height, width, conceal)
        if frame is not None:
            return frame
    return _binary_decode_last_frame(stream, height, width, conceal)


def _binary_decode_last_frame(
    stream: bytes, height: int, width: int, conceal: bool = True
) -> np.ndarray | None:
    """``_decode_last_frame`` with the ``ffmpeg`` binary; ``None`` if nothing decodes.

    It decodes what is there and shows it, as a player does, concealing
    losses from the previous picture when ``conceal`` is set.
    """
    binary = _codec.ffmpeg_path()
    if binary is None:
        return None
    command = [
        binary, "-hide_banner", "-loglevel", "quiet",
        # Decode what is there and show it, as a player does: a damaged
        # picture, not a stopped one.
        "-err_detect", "ignore_err", "-ec", "guess_mvs+deblock" if conceal else "0",
        "-flags", "+output_corrupt", "-threads", "1",
        "-f", "h264", "-i", "pipe:0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]  # fmt: skip
    try:
        decoded = subprocess.run(command, input=stream, capture_output=True, timeout=120).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    frame_bytes = height * width * 3
    count = len(decoded) // frame_bytes
    if count == 0:
        return None
    last = decoded[(count - 1) * frame_bytes : count * frame_bytes]
    return np.frombuffer(last, dtype=np.uint8).reshape(height, width, 3).copy()


# Whether PyAV's encoder has been seen to write the binary's slices here: None
# until it has been tried.
_inprocess_is_binary: bool | None = None


def _pictures(stream: bytes | None) -> list[bytes] | None:
    """A stream's NAL units without its messages, which name who wrote it."""
    if stream is None:
        return None
    return [unit for unit in split_nal_units(stream) if unit and (unit[0] & 0x1F) != 6]


def _reproduces_binary() -> bool:
    """Whether a clip encoded in this process is the binary's, slice for slice.

    As ``codec.reproduces_binary``, for this transform's encoder settings: a
    small clip is encoded both ways and the parameter sets and slices
    compared. The streams differ in one message, which names the libx264
    release that wrote them and is never put at risk of loss.
    """
    global _inprocess_is_binary
    if _inprocess_is_binary is None:
        probe = np.random.default_rng(0).integers(0, 256, (12, 16, 3), dtype=np.uint8)
        probe = np.ascontiguousarray(np.kron(probe, np.ones((8, 8, 1), dtype=np.uint8)))
        clip = _sliding_clip(probe, 3, (5.0, -2.0))
        ours = _pictures(_inprocess.encode_clip(clip, 28, 3))
        theirs = _pictures(_binary_encode_clip(clip, 28, 3))
        _inprocess_is_binary = ours is not None and ours == theirs
    return _inprocess_is_binary


def _encode_inprocess() -> bool:
    """As ``codec.use_inprocess``, for this transform's stream."""
    if not _inprocess.has_encoder("libx264"):
        return False
    return _codec.ffmpeg_path() is None or mode.fast() or _reproduces_binary()


def _decode_inprocess() -> bool:
    """Whether a damaged stream is decoded in this process.

    In fast mode, and where there is no binary. Not in exact mode: what an
    unconcealed loss shows is the decoder's own frame buffer, which is not
    the same in the two. See :mod:`surmentations.camera._inprocess`.
    """
    if not _inprocess.available():
        return False
    return _codec.ffmpeg_path() is None or mode.fast()


def _sliding_clip(rgb: np.ndarray, frames: int, shift_px: tuple[float, float]) -> np.ndarray:
    """``frames + 1`` copies of the image sliding into place.

    The first is displaced by the whole of ``shift_px`` and the last not at
    all, so the frame that comes out lines up with the one that went in.
    """
    height, width = rgb.shape[:2]
    clip = np.empty((frames + 1, height, width, 3), dtype=np.uint8)
    for index in range(frames + 1):
        remaining = (frames - index) / frames
        if remaining == 0.0:
            clip[index] = rgb
            continue
        matrix = np.array(
            [[1.0, 0.0, shift_px[0] * remaining], [0.0, 1.0, shift_px[1] * remaining]],
            dtype=np.float32,
        )
        clip[index] = cv2.warpAffine(
            rgb, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
    return clip


class TransmissionFaults(CameraTransform):
    """Packet loss on an H.264 stream, decoded for real.

    Args:
        packet_loss_range: share of the stream's packets that are lost,
            drawn per call on a log scale. 0.01 is a link with a problem,
            0.1 one that is barely usable. At least one packet is lost
            whenever this is positive.
        burst_length_range: mean number of consecutive packets lost each
            time loss occurs. 1 is independent loss; wireless fades run to
            several.
        frames_since_keyframe_range: how many frames into its group of
            pictures the returned frame is. Damage can strike any of them
            and stays until the next keyframe, so a larger number means
            more chances of damage, and older damage that later frames have
            partly drawn over.
        slices_range: slices each picture is coded in. A fault cannot cross
            a slice boundary within a picture, so this sets the height of
            the bands damage comes in: one slice, and a single lost packet
            can take the rest of the frame.
        crf_range: quality the stream is encoded at. See
            :class:`~surmentations.camera.codec.VideoCodecArtifacts`.
        scene_motion_range: how far the picture moved between the reference
            the decoder conceals from and the frame returned, as a fraction
            of the image diagonal. Concealment copies old picture into new,
            and this is how wrong the old picture is. At zero most loss is
            concealed without a trace, as on a still scene it is.
        missing_reference_probability: chance that the decoder has no
            earlier picture at all — it joined after an outage, or the loss
            took the previous keyframe too. Missing slices are then filled
            by smearing their surroundings and not by copying.
        concealment_probability: chance that the decoder conceals what it
            did not receive. Software players do; many embedded decoders
            and recorders do not, and show the missing region as flat
            green.
        gap_passing_probability: chance that the receiver hands the decoder
            a slice with a hole in it instead of cutting it off at the
            hole. This is what produces blocks of garbage.

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        """The arguments of ``TransmissionFaults`` as Albumentations validates them: each
        range ordered low to high and within the bounds given here.
        """
        packet_loss_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1e-4, 0.9)),
        ]
        burst_length_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1.0, 50.0)),
        ]
        frames_since_keyframe_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1, 30)),
        ]
        slices_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(1, 32)),
        ]
        crf_range: Annotated[
            tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 51)),
        ]
        scene_motion_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 0.25)),
        ]
        missing_reference_probability: float = Field(ge=0.0, le=1.0)
        concealment_probability: float = Field(ge=0.0, le=1.0)
        gap_passing_probability: float = Field(ge=0.0, le=1.0)

    def __init__(
        self,
        packet_loss_range: tuple[float, float] = (0.01, 0.15),
        burst_length_range: tuple[float, float] = (1.0, 4.0),
        frames_since_keyframe_range: tuple[int, int] = (1, 5),
        slices_range: tuple[int, int] = (1, 6),
        crf_range: tuple[int, int] = (24, 34),
        scene_motion_range: tuple[float, float] = (0.0, 0.03),
        missing_reference_probability: float = 0.15,
        concealment_probability: float = 0.65,
        gap_passing_probability: float = 0.5,
        p: float = 0.5,
    ):
        """Store the ranges; each argument is described in the class docstring.

        Raises ``RuntimeError`` where there is neither the ``ffmpeg`` binary
        nor PyAV: this transform damages and decodes a real H.264 stream.
        """
        super().__init__(p=p)
        if not _codec.codec_available("libx264"):
            raise RuntimeError(
                "TransmissionFaults needs the `ffmpeg` binary on PATH or PyAV: it damages a "
                "real H.264 bitstream and decodes it, and does not approximate the result. "
                "Install one (apt install ffmpeg / brew install ffmpeg, or "
                "pip install surmentations[codec])."
            )
        self.packet_loss_range = packet_loss_range
        self.burst_length_range = burst_length_range
        self.frames_since_keyframe_range = frames_since_keyframe_range
        self.slices_range = slices_range
        self.crf_range = crf_range
        self.scene_motion_range = scene_motion_range
        self.missing_reference_probability = missing_reference_probability
        self.concealment_probability = concealment_probability
        self.gap_passing_probability = gap_passing_probability

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Draw the link and the stream: how much is lost, in what bursts, and how it is decoded.

        Returns the packet ``loss`` and ``mean_burst``, the clip's ``frames``
        since a keyframe, ``slices`` and ``crf``, the scene's motion
        ``shift_px``, whether the decoder holds the reference, passes gaps on
        or conceals, and the ``link_seed`` that decides which packets go.
        """
        del data
        height, width = params["shape"][:2]
        low, high = self.packet_loss_range
        loss = float(np.exp(self.py_random.uniform(np.log(low), np.log(high))))
        motion = self.py_random.uniform(*self.scene_motion_range) * float(np.hypot(width, height))
        direction = self.py_random.uniform(0.0, 2.0 * np.pi)
        drawn = {
            "loss": loss,
            "mean_burst": self.py_random.uniform(*self.burst_length_range),
            "frames": self.py_random.randint(*self.frames_since_keyframe_range),
            "slices": self.py_random.randint(*self.slices_range),
            "crf": self.py_random.randint(*self.crf_range),
            "shift_px": (motion * float(np.cos(direction)), motion * float(np.sin(direction))),
            "keep_reference": self.py_random.random() >= self.missing_reference_probability,
            "pass_gaps": self.py_random.random() < self.gap_passing_probability,
            "conceal": self.py_random.random() < self.concealment_probability,
            "link_seed": self.py_random.randint(0, 2**31 - 1),
        }
        record_draw(
            packet_loss=loss,
            mean_burst=drawn["mean_burst"],
            frames_since_keyframe=drawn["frames"],
            slices=drawn["slices"],
            crf=drawn["crf"],
            scene_motion_px=motion,
            keep_reference=drawn["keep_reference"],
            pass_gaps=drawn["pass_gaps"],
            conceal=drawn["conceal"],
        )
        return drawn

    def apply(
        self,
        img: np.ndarray,
        loss: float,
        mean_burst: float,
        frames: int,
        slices: int,
        crf: int,
        shift_px: tuple[float, float],
        keep_reference: bool,
        pass_gaps: bool,
        conceal: bool,
        link_seed: int,
        **params: Any,
    ) -> np.ndarray:
        """Encode a short clip, drop packets from it, and decode the last frame.

        The frame is padded to even dimensions for 4:2:0 chroma and cropped
        back. If the codec fails the input is returned unchanged.
        """
        is_integer = np.issubdtype(img.dtype, np.integer)
        rgb = img if is_integer else np.clip(img * 255.0, 0, 255).astype(np.uint8)
        height, width = rgb.shape[:2]
        # 4:2:0 chroma needs even dimensions; pad and crop back.
        pad_h, pad_w = height % 2, width % 2
        if pad_h or pad_w:
            rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        rgb = np.ascontiguousarray(rgb)

        stream = _encode_clip(_sliding_clip(rgb, frames, shift_px), crf, slices)
        if stream is None:
            # ffmpeg vanished or failed. As with the codec transform, the
            # input unchanged is the only honest result.
            return img
        units, _sent, _lost = drop_packets(
            split_nal_units(stream),
            np.random.default_rng(link_seed),
            loss,
            mean_burst,
            keep_reference,
            pass_gaps,
        )
        damaged = _decode_last_frame(
            b"".join(_START_CODE + unit for unit in units), rgb.shape[0], rgb.shape[1], conceal
        )
        if damaged is None:
            return img
        damaged = damaged[:height, :width]
        if is_integer:
            return damaged.astype(img.dtype)
        return (damaged.astype(np.float32) / 255.0).astype(img.dtype)
