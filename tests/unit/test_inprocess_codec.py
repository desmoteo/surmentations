"""The codec transforms encoding in this process, through PyAV, against the ffmpeg binary.

PyAV is an optional extra (``pip install weathermentations[codec]``). Without
it most of this file skips, and what is left checks that its absence is
handled: nothing is encoded in process and the binary does the work.

With it, the contract is the one ``camera/_inprocess.py`` states. Where PyAV
has been seen to give the binary's frame, exact mode may use it and the
frame is the same to the byte; where it has not, exact mode keeps to the
binary. Fast mode uses it either way.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from weathermentations import mode
from weathermentations.camera import TransmissionFaults, VideoCodecArtifacts, _inprocess
from weathermentations.camera import codec as codec_module
from weathermentations.camera import transmission as transmission_module

HAS_BINARY = codec_module.ffmpeg_path() is not None
requires_av = pytest.mark.skipif(
    not _inprocess.available(), reason="PyAV is not installed (the `codec` extra)"
)
requires_both = pytest.mark.skipif(
    not (_inprocess.available() and HAS_BINARY), reason="needs PyAV and the ffmpeg binary"
)
requires_binary = pytest.mark.skipif(not HAS_BINARY, reason="needs the ffmpeg binary")


@pytest.fixture(autouse=True)
def _restore_mode():
    before = mode.fast()
    yield
    mode.set_fast(before)


def _scene(height: int = 144, width: int = 192, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    blocks = ((xx // 24) * 61 + (yy // 18) * 89) % 180
    image = blocks[..., None] + rng.integers(0, 60, (height, width, 3))
    return np.clip(image, 0, 255).astype(np.uint8)


def _without_pyav(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_inprocess, "has_encoder", lambda encoder: False)
    monkeypatch.setattr(_inprocess, "available", lambda: False)


# ----------------------------------------------------------- without PyAV


def test_without_pyav_nothing_is_encoded_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_inprocess, "_av", lambda: None)
    assert not _inprocess.available()
    assert not _inprocess.has_encoder("libx264")
    frame = _scene()
    assert _inprocess.roundtrip(frame, "libx264", 30) is None
    assert _inprocess.encode_clip(frame[None], 30, 2) is None
    assert _inprocess.decode_last_frame(b"", 144, 192) is None
    for flag in (True, False):
        mode.set_fast(flag)
        assert not codec_module.use_inprocess("libx264")
        assert not transmission_module._encode_inprocess()
        assert not transmission_module._decode_inprocess()


def test_the_environment_keeps_the_codec_to_the_binary() -> None:
    code = "from weathermentations.camera import _inprocess; print(_inprocess.available())"
    off = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "WEATHERMENTATIONS_CODEC": "ffmpeg"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert off.stdout.strip() == "False"


@requires_binary
def test_a_failed_encode_in_process_is_done_by_the_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _scene()
    reference = codec_module._binary_roundtrip(frame, "libx264", 30)
    monkeypatch.setattr(codec_module, "use_inprocess", lambda encoder: True)
    monkeypatch.setattr(_inprocess, "roundtrip", lambda *a: None)
    assert np.array_equal(codec_module._run_roundtrip(frame, "libx264", 30), reference)


# -------------------------------------------------------------- with PyAV


@requires_both
@pytest.mark.parametrize("size", [(144, 192), (97, 131), (64, 64), (360, 480)])
def test_h264_in_process_is_the_binary_s_frame(size: tuple[int, int]) -> None:
    """Bicubic colour conversion, one encoder thread and the binary's frame
    rate: with those the two are the same libraries doing the same thing.
    Where this installation's two builds do not agree, the transforms have
    found that out for themselves and there is nothing to hold PyAV to."""
    if not codec_module.reproduces_binary("libx264"):
        pytest.skip("this PyAV's libx264 does not encode as the ffmpeg binary's does")
    frame = _scene(*size)
    padded = np.pad(frame, ((0, size[0] % 2), (0, size[1] % 2), (0, 0)), mode="edge")
    for crf in (20, 30, 44):
        ours = _inprocess.roundtrip(np.ascontiguousarray(padded), "libx264", crf)
        theirs = codec_module._binary_roundtrip(padded, "libx264", crf)
        assert ours is not None and theirs is not None
        assert not np.array_equal(ours, padded), "the codec did nothing"
        assert np.array_equal(ours, theirs)


@requires_both
@pytest.mark.parametrize("fast", [True, False], ids=["fast", "exact"])
def test_the_h264_transform_does_not_depend_on_pyav(
    fast: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not codec_module.reproduces_binary("libx264"):
        pytest.skip("this PyAV's libx264 does not encode as the ffmpeg binary's does")
    transform = VideoCodecArtifacts(codec="h264", p=1.0, stale_block_probability=0.5)
    image = _scene(145, 193)
    mode.set_fast(fast)
    frames = []
    for seed in range(4):
        transform.set_random_seed(seed)
        frames.append(transform(image=image)["image"])
    _without_pyav(monkeypatch)
    for seed in range(4):
        transform.set_random_seed(seed)
        assert np.array_equal(transform(image=image)["image"], frames[seed])


@requires_both
def test_exact_mode_keeps_to_the_binary_where_pyav_was_seen_to_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(codec_module._inprocess_is_binary, "libx264", False)
    mode.set_fast(False)
    assert not codec_module.use_inprocess("libx264")
    mode.set_fast(True)
    assert codec_module.use_inprocess("libx264")


@requires_av
def test_without_the_binary_pyav_is_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codec_module, "ffmpeg_path", lambda: None)
    image = _scene()
    for flag in (True, False):
        mode.set_fast(flag)
        for transform in (VideoCodecArtifacts(p=1.0), TransmissionFaults(p=1.0)):
            transform.set_random_seed(3)
            out = transform(image=image)["image"]
            assert out.shape == image.shape and out.dtype == np.uint8
            assert not np.array_equal(out, image)


@requires_av
def test_h265_in_process_is_a_real_encode_and_a_quiet_one(capfd) -> None:
    """Not the binary's frame, which comes from another release of libx265,
    but a frame through H.265 all the same: close to its input, and closer
    at a better quality. libx265 says nothing on stderr about it."""
    if not _inprocess.has_encoder("libx265"):
        pytest.skip("this PyAV was built without libx265")
    frame = _scene()
    errors = []
    for crf in (20, 40):
        out = _inprocess.roundtrip(frame, "libx265", crf)
        assert out is not None and out.shape == frame.shape
        errors.append(np.abs(out.astype(np.int16) - frame).mean())
    assert 0.0 < errors[0] < errors[1] < 40.0
    assert "x265" not in capfd.readouterr().err


@requires_both
def test_a_clip_is_encoded_in_process_slice_for_slice() -> None:
    if not transmission_module._reproduces_binary():
        pytest.skip("this PyAV's libx264 does not encode as the ffmpeg binary's does")
    clip = transmission_module._sliding_clip(_scene(), 4, (6.0, -3.0))
    for crf, slices in ((26, 1), (32, 5)):
        ours = transmission_module._pictures(_inprocess.encode_clip(clip, crf, slices))
        theirs = transmission_module._pictures(
            transmission_module._binary_encode_clip(clip, crf, slices)
        )
        assert ours is not None and len(ours) > len(clip)
        assert ours == theirs


@requires_both
def test_exact_mode_decodes_a_damaged_stream_with_the_binary() -> None:
    """What an unconcealed loss shows is the decoder's own frame buffer, so
    the two decoders are not held to each other and exact mode uses one."""
    mode.set_fast(False)
    assert not transmission_module._decode_inprocess()
    mode.set_fast(True)
    assert transmission_module._decode_inprocess()


@requires_both
def test_exact_mode_transmission_faults_do_not_depend_on_pyav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not transmission_module._reproduces_binary():
        pytest.skip("this PyAV's libx264 does not encode as the ffmpeg binary's does")
    transform = TransmissionFaults(p=1.0)
    image = _scene(145, 193)
    mode.set_fast(False)
    frames = []
    for seed in range(8):
        transform.set_random_seed(seed)
        frames.append(transform(image=image)["image"])
    _without_pyav(monkeypatch)
    for seed in range(8):
        transform.set_random_seed(seed)
        assert np.array_equal(transform(image=image)["image"], frames[seed])


@requires_both
def test_a_concealed_loss_decodes_the_same_in_process() -> None:
    """With concealment on the two decoders agreed on every stream tried.
    Fast mode's frames differ from exact mode's only where it is off."""
    if not transmission_module._reproduces_binary():
        pytest.skip("this PyAV's libx264 does not encode as the ffmpeg binary's does")
    image = _scene()
    clip = transmission_module._sliding_clip(image, 4, (6.0, -3.0))
    stream = transmission_module._binary_encode_clip(clip, 28, 4)
    assert stream is not None
    for seed in range(5):
        units, _sent, lost = transmission_module.drop_packets(
            transmission_module.split_nal_units(stream),
            np.random.default_rng(seed),
            0.2,
            2.0,
            True,
            False,
        )
        assert lost > 0
        damaged = b"".join(transmission_module._START_CODE + unit for unit in units)
        ours = _inprocess.decode_last_frame(damaged, *image.shape[:2], True)
        theirs = transmission_module._binary_decode_last_frame(damaged, *image.shape[:2], True)
        assert ours is not None and theirs is not None
        assert np.array_equal(ours, theirs)
