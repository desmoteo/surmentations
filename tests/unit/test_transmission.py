"""Packet loss is done to a real bitstream, so most of what is tested here is
the link: which packets go, and what is handed to the decoder. What the
decoder then draws is ffmpeg's, and is tested only for where it lands.
"""

from __future__ import annotations

import numpy as np
import pytest

from surmentations.camera.codec import MACROBLOCK_PX, ffmpeg_path
from surmentations.camera.transmission import (
    _PACKET_BYTES,
    _START_CODE,
    TransmissionFaults,
    _decode_last_frame,
    _encode_clip,
    _is_slice,
    _sliding_clip,
    _starts_picture,
    drop_packets,
    packet_losses,
    split_nal_units,
)
from surmentations.diagnostics import recording
from surmentations.presets.severity import SEVERITY

requires_ffmpeg = pytest.mark.skipif(ffmpeg_path() is None, reason="needs the ffmpeg binary")

SIZE = 192


def _scene() -> np.ndarray:
    """Textured enough that an encoded keyframe runs to many packets."""
    rng = np.random.default_rng(0)
    coarse = rng.integers(30, 220, size=(SIZE // 8, SIZE // 8, 3), dtype=np.uint8)
    image = np.kron(coarse, np.ones((8, 8, 1), dtype=np.uint8))
    noise = rng.integers(-12, 12, size=image.shape)
    return np.clip(image.astype(np.int32) + noise, 0, 255).astype(np.uint8)


def _stream(
    frames: int = 1, slices: int = 4, shift: tuple[float, float] = (0.0, 0.0), crf: int = 24
) -> bytes:
    """A raw H.264 stream of the test scene sliding into place over ``frames`` frames."""
    stream = _encode_clip(_sliding_clip(_scene(), frames, shift), crf=crf, slices=slices)
    assert stream is not None
    return stream


def _join(units: list[bytes]) -> bytes:
    """NAL units joined back into an Annex B stream."""
    return b"".join(_START_CODE + unit for unit in units)


# -------------------------------------------------------------------- link


def test_nal_units_are_split_on_both_start_codes() -> None:
    stream = b"\x00\x00\x00\x01\x67abc" + b"\x00\x00\x01\x68de" + b"\x00\x00\x00\x01\x65\x88xyz"
    assert split_nal_units(stream) == [b"\x67abc", b"\x68de", b"\x65\x88xyz"]


def test_loss_rate_and_burst_length_are_what_was_asked_for() -> None:
    """Loss is bursty without the bursts changing how much is lost: a mean
    burst of five must lose the same share as independent loss, in runs."""
    rng = np.random.default_rng(1)
    single = packet_losses(rng, 200_000, 0.05, 1.0)
    bursty = packet_losses(rng, 200_000, 0.05, 5.0)
    assert single.mean() == pytest.approx(0.05, rel=0.1)
    assert bursty.mean() == pytest.approx(0.05, rel=0.15)

    def mean_run(lost: np.ndarray) -> float:
        edges = np.diff(np.concatenate([[0], lost.astype(np.int8), [0]]))
        return float(lost.sum() / (edges == 1).sum())

    assert mean_run(bursty) > 3.0 * mean_run(single)


def test_a_positive_loss_always_loses_something() -> None:
    """`p` decides whether the transform applies. A loss rate so low that no
    packet happens to go must not hand back an undamaged stream."""
    for seed in range(50):
        assert packet_losses(np.random.default_rng(seed), 12, 1e-4, 1.0).sum() >= 1
    assert packet_losses(np.random.default_rng(0), 12, 0.0, 1.0).sum() == 0


@requires_ffmpeg
def test_the_reference_picture_is_never_cut_into() -> None:
    """Only the group of pictures being received is at risk. The reference is
    either all there or, for a decoder that joined late, all missing."""
    units = split_nal_units(_stream(frames=2))
    reference = [u for u in units if _is_slice(u)][:4]
    assert _starts_picture(reference[0]) and not any(map(_starts_picture, reference[1:]))

    rng = np.random.default_rng(0)
    kept, sent, lost = drop_packets(units, rng, 0.5, 2.0, keep_reference=True, pass_gaps=False)
    assert all(unit in kept for unit in reference)
    assert 0 < lost < sent

    gone, _, _ = drop_packets(units, rng, 0.5, 2.0, keep_reference=False, pass_gaps=False)
    assert not any(unit in gone for unit in reference)
    assert [u for u in gone if not _is_slice(u)] == [u for u in units if not _is_slice(u)], (
        "parameter sets are resent by the camera and must survive"
    )


@requires_ffmpeg
def test_a_gap_is_cut_off_or_closed_up_depending_on_the_receiver() -> None:
    units = split_nal_units(_stream(frames=1, slices=1, crf=8))
    at_risk = [u for u in units if _is_slice(u)][-1]
    assert len(at_risk) > 3 * _PACKET_BYTES, "the fixture needs a slice of several packets"

    class _OnePacketNearTheEnd:
        """Loses exactly one packet, the last but one of the stream."""

        def random(self) -> float:
            return 1.0

        def integers(self, count: int) -> int:
            return count - 2

    strict, _, _ = drop_packets(
        units, _OnePacketNearTheEnd(), 1e-9, 1.0, keep_reference=True, pass_gaps=False
    )  # type: ignore[arg-type]
    lenient, _, _ = drop_packets(
        units, _OnePacketNearTheEnd(), 1e-9, 1.0, keep_reference=True, pass_gaps=True
    )  # type: ignore[arg-type]
    total = sum(len(u) for u in units)
    assert sum(len(u) for u in lenient) == total - _PACKET_BYTES
    assert sum(len(u) for u in strict) < sum(len(u) for u in lenient)


# ----------------------------------------------------------------- decoder


@requires_ffmpeg
def test_an_undamaged_stream_decodes_to_the_picture() -> None:
    """With nothing lost this is the codec transform: the last frame of the
    clip is the image where it was given, compressed and nothing else."""
    scene = _scene()
    stream = _stream(frames=3, shift=(9.0, -4.0))
    kept, _, lost = drop_packets(
        split_nal_units(stream), np.random.default_rng(0), 0.0, 1.0, True, False
    )
    assert lost == 0
    out = _decode_last_frame(_join(kept), SIZE, SIZE)
    assert out is not None
    where_it_started = _sliding_clip(scene, 3, (9.0, -4.0))[0]

    def error(reference: np.ndarray) -> float:
        return float(np.abs(out.astype(np.float32) - reference).mean())

    assert error(scene) < 0.5 * error(where_it_started)


@requires_ffmpeg
def test_a_lost_slice_damages_its_own_band_and_no_other() -> None:
    """Damage stops at a slice boundary, and slices are whole rows of
    macroblocks. A fault that bled across the frame, or was soft-edged, would
    be something drawn and not something decoded."""
    units = split_nal_units(_stream(frames=1, slices=4))
    clean = _decode_last_frame(_join(units), SIZE, SIZE, conceal=False)
    slices = [i for i, u in enumerate(units) if _is_slice(u)]
    second_slice_of_the_second_picture = slices[5]
    damaged_units = [u for i, u in enumerate(units) if i != second_slice_of_the_second_picture]
    damaged = _decode_last_frame(_join(damaged_units), SIZE, SIZE, conceal=False)
    assert clean is not None and damaged is not None

    rows = np.abs(damaged.astype(np.float32) - clean).mean(axis=(1, 2)) > 4.0
    hit = np.flatnonzero(rows)
    assert hit.size > 0, "dropping a slice changed nothing"
    top, bottom = int(hit[0]), int(hit[-1]) + 1
    assert rows[top:bottom].all(), "the damage is one band"
    assert top % MACROBLOCK_PX == 0 and bottom % MACROBLOCK_PX == 0
    assert SIZE // 8 < bottom - top < SIZE // 2, "about one slice of four"
    assert 0 < top and bottom < SIZE


@requires_ffmpeg
def test_concealment_copies_the_old_picture_into_the_hole() -> None:
    """What makes loss visible is that the picture moved. The same lost slice
    is close to invisible over a still scene and plain over one that slid."""

    def damage(shift: tuple[float, float]) -> float:
        units = split_nal_units(_stream(frames=1, slices=4, shift=shift))
        slices = [i for i, u in enumerate(units) if _is_slice(u)]
        kept = [u for i, u in enumerate(units) if i != slices[5]]
        clean = _decode_last_frame(_join(units), SIZE, SIZE)
        out = _decode_last_frame(_join(kept), SIZE, SIZE)
        assert clean is not None and out is not None
        return float(np.abs(out.astype(np.float32) - clean).mean())

    still, moved = damage((0.0, 0.0)), damage((11.0, 0.0))
    assert moved > 4.0 * max(still, 0.05), (still, moved)


# --------------------------------------------------------------- transform


@requires_ffmpeg
def test_faults_are_reproducible_and_keep_the_frame_s_shape_and_type() -> None:
    scene = _scene()[:-1, :-3]  # odd on both axes: 4:2:0 needs padding
    transform = TransmissionFaults(p=1.0, packet_loss_range=(0.2, 0.2))
    transform.set_random_seed(4)
    first = transform(image=scene)["image"]
    transform.set_random_seed(4)
    again = transform(image=scene)["image"]
    assert first.shape == scene.shape and first.dtype == np.uint8
    assert np.array_equal(first, again)

    transform.set_random_seed(4)
    as_float = transform(image=(scene / 255.0).astype(np.float32))["image"]
    assert as_float.dtype == np.float32
    assert np.abs(as_float * 255.0 - first).mean() < 1.0


@requires_ffmpeg
def test_a_worse_link_damages_more_of_the_frame() -> None:
    scene = _scene()

    def damage(level: int) -> float:
        transform = TransmissionFaults(p=1.0, **SEVERITY[level]["TransmissionFaults"])
        total = 0.0
        for seed in range(8):
            transform.set_random_seed(seed)
            out = transform(image=scene)["image"].astype(np.float32)
            total += float(np.abs(out - scene).mean())
        return total / 8.0

    mild, severe = damage(1), damage(5)
    assert severe > 1.5 * mild, (mild, severe)


@requires_ffmpeg
def test_the_link_s_state_is_recorded() -> None:
    transform = TransmissionFaults(
        p=1.0, packet_loss_range=(0.05, 0.05), missing_reference_probability=1.0
    )
    with recording() as log:
        transform(image=_scene())
    drew = log[0]["drew"]
    assert drew["packet_loss"] == pytest.approx(0.05)
    assert drew["keep_reference"] is False
    assert 1 <= drew["frames_since_keyframe"] <= 5


def test_it_refuses_to_be_built_without_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """It could fall back to drawing blocks. It must not: that would put a
    different artefact in a training set under this transform's name."""
    from surmentations.camera import _inprocess
    from surmentations.camera import codec as codec_module

    monkeypatch.setattr(codec_module, "ffmpeg_path", lambda: None)
    # Nor PyAV, which would do in the binary's place.
    monkeypatch.setattr(_inprocess, "has_encoder", lambda encoder: False)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        TransmissionFaults()
