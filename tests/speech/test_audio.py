"""audio_utils.py, audio_stream.py and audio_devices.py — resampling, the
microphone queue and device hot-swap detection (sounddevice is faked)."""

import json
import subprocess

import numpy as np
import pytest

import audio_devices
import audio_stream
from audio_devices import DeviceMonitor, default_of, identity
from audio_utils import StreamResampler, resample_int16
from config import AUDIO
from helpers.audio import voice
from helpers.fakes import FakeSoundDevice


# ── Resampling ─────────────────────────────────────────────────────────────────

def test_same_rate_is_passthrough():
    rs = StreamResampler(16000, 16000)
    assert rs.passthrough
    assert rs.process(b"\x01\x02") == b"\x01\x02"
    assert rs.flush() == b""
    rs.reset()


def test_streaming_matches_the_rate_ratio():
    signal = voice(1.0)                                  # 48 000 samples
    rs = StreamResampler(48000, 16000)
    out = b"".join(rs.process(c.tobytes()) for c in np.array_split(signal, 10))
    out += rs.flush()
    n = len(out) // 2
    assert abs(n - 16000) <= 16                          # filter delay only
    assert not rs.passthrough


def test_stream_and_oneshot_agree():
    signal = voice(0.5)
    rs = StreamResampler(48000, 16000)
    streamed = np.frombuffer(
        b"".join(rs.process(c.tobytes()) for c in np.array_split(signal, 5)) + rs.flush(),
        dtype=np.int16)
    oneshot = resample_int16(signal, 48000, 16000)
    assert oneshot.dtype == np.int16
    n = min(len(streamed), len(oneshot))
    assert np.corrcoef(streamed[:n].astype(float), oneshot[:n].astype(float))[0, 1] > 0.99


def test_resample_preserves_the_tone():
    t = np.arange(48000) / 48000
    tone = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    out = resample_int16(tone, 48000, 16000).astype(float)
    peak_hz = np.argmax(np.abs(np.fft.rfft(out))) * 16000 / len(out)
    assert abs(peak_hz - 440) < 5


def test_resample_shortcuts():
    empty = np.zeros(0, dtype=np.int16)
    assert resample_int16(empty, 48000, 16000) is empty
    same = np.ones(10, dtype=np.int16)
    assert resample_int16(same, 16000, 16000) is same


def test_reset_behaves_like_a_fresh_resampler():
    chunk = voice(0.1).tobytes()
    used = StreamResampler(48000, 16000)
    used.process(voice(0.3, seed=9).tobytes())          # leave history in the filter
    used.reset()
    after_reset = np.frombuffer(used.process(chunk), dtype=np.int16)
    fresh = np.frombuffer(StreamResampler(48000, 16000).process(chunk), dtype=np.int16)
    assert len(after_reset) == len(fresh)                # same filter delay again
    # int16 rounding may differ by an LSB; history would differ by thousands.
    assert np.max(np.abs(after_reset.astype(int) - fresh.astype(int))) <= 2


# ── AudioStream (fake PortAudio) ───────────────────────────────────────────────

@pytest.fixture
def sd(monkeypatch):
    fake = FakeSoundDevice()
    monkeypatch.setattr(audio_stream, "_sd", fake)
    return fake


def test_start_opens_a_raw_int16_stream(sd):
    stream = audio_stream.AudioStream(sample_rate=48000, chunk_size=4800, device_index=3)
    stream.start()
    kwargs = sd.streams[0].kwargs
    assert (kwargs["samplerate"], kwargs["blocksize"], kwargs["device"]) == (48000, 4800, 3)
    assert kwargs["dtype"] == "int16" and kwargs["channels"] == 1
    assert sd.streams[0].started and stream.is_running()


def test_double_start_is_ignored(sd):
    stream = audio_stream.AudioStream()
    stream.start()
    stream.start()
    assert len(sd.streams) == 1


def test_callback_queues_chunks_and_stop_sends_sentinel(sd):
    stream = audio_stream.AudioStream()
    stream.start()
    stream._audio_callback(b"\x01\x00" * 4, 4, None, None)
    stream.stop()
    assert stream.chunk_queue.get_nowait() == b"\x01\x00" * 4
    assert stream.chunk_queue.get_nowait() is None
    assert sd.streams[0].stopped and sd.streams[0].closed
    stream.stop()                                        # idempotent


def test_callback_ignored_when_not_running(sd):
    stream = audio_stream.AudioStream()
    stream._audio_callback(b"\x00\x00", 1, None, None)
    assert stream.chunk_queue.empty()


def test_full_queue_drops_the_oldest_chunk(sd):
    stream = audio_stream.AudioStream()
    stream.start()
    for i in range(stream.chunk_queue.maxsize + 3):
        stream._audio_callback(bytes([i % 256, 0]), 1, None, None)
    first = stream.chunk_queue.get_nowait()
    assert first == bytes([3, 0])                        # 0, 1, 2 were dropped


def test_restart_reinitialises_portaudio_and_keeps_the_device(sd):
    stream = audio_stream.AudioStream(device_index=2)
    stream.start()
    stream.restart()
    assert (sd.terminated, sd.initialized) == (1, 1)
    assert len(sd.streams) == 2 and sd.streams[1].kwargs["device"] == 2
    assert stream.chunk_queue.empty()                    # no None sentinel on restart


def test_restart_can_change_device(sd):
    stream = audio_stream.AudioStream(device_index=2)
    stream.start()
    stream.restart(device_index=None)
    assert sd.streams[1].kwargs["device"] is None


def test_restart_while_stopped_does_not_open(sd):
    stream = audio_stream.AudioStream()
    stream.restart(device_index=5)
    assert sd.streams == [] and stream.device_index == 5


def test_clear_queue(sd):
    stream = audio_stream.AudioStream()
    stream.start()
    for _ in range(5):
        stream._audio_callback(b"\x00\x00", 1, None, None)
    stream.clear_queue()
    assert stream.chunk_queue.empty()


def test_list_input_devices_filters_outputs(sd):
    assert audio_stream.list_input_devices() == [
        {"index": 1, "name": "USB Mic", "channels": 1},
        {"index": 2, "name": "Array Mic", "channels": 2},
    ]


def test_capture_rate_default():
    assert audio_stream.AudioStream().sample_rate == AUDIO["capture_rate"]


# ── Device inventory & hot-swap detection ──────────────────────────────────────

def dev(name, default=False, api="WASAPI", index=0):
    return {"index": index, "name": name, "hostapi": api, "channels": 1,
            "default_samplerate": 48000.0, "is_default": default}


def test_identity_ignores_index():
    assert identity(dev("Mic", index=1)) == identity(dev("Mic", index=7))


def test_default_of():
    assert default_of([dev("A"), dev("B", default=True)])["name"] == "B"
    assert default_of([dev("A")]) is None
    assert default_of(None) is None


class _Proc:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


@pytest.mark.parametrize("behaviour, expected", [
    (_Proc(0, json.dumps([dev("Mic", True)])), [dev("Mic", True)]),
    (_Proc(1, "", "PortAudio not initialized"), None),
    (_Proc(0, "not json"), None),
])
def test_snapshot_parses_the_child_query(monkeypatch, behaviour, expected):
    monkeypatch.setattr(audio_devices.subprocess, "run", lambda *a, **k: behaviour)
    assert audio_devices.snapshot() == expected


def test_snapshot_timeout(monkeypatch):
    def hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd="q", timeout=1)
    monkeypatch.setattr(audio_devices.subprocess, "run", hang)
    assert audio_devices.snapshot(timeout_s=1) is None


def test_monitor_reports_list_and_default_changes():
    lists, defaults = [], []
    mon = DeviceMonitor(on_default_change=lambda d, ds: defaults.append(d["name"]),
                        on_list_change=lambda ds: lists.append([d["name"] for d in ds]))
    old = [dev("Laptop Mic", True)]
    new = [dev("Laptop Mic"), dev("Headset", True)]
    mon._diff(old, new)
    assert lists == [["Laptop Mic", "Headset"]]
    assert defaults == ["Headset"]


def test_monitor_ignores_reindexing():
    calls = []
    mon = DeviceMonitor(on_default_change=lambda *a: calls.append(a),
                        on_list_change=lambda *a: calls.append(a))
    mon._diff([dev("Mic", True, index=0)], [dev("Mic", True, index=4)])
    assert calls == []


def test_monitor_without_a_default_does_not_fire():
    calls = []
    mon = DeviceMonitor(on_default_change=lambda *a: calls.append(a))
    mon._diff([dev("Mic", True)], [dev("Mic")])
    assert calls == []


def test_monitor_callback_errors_are_contained():
    def boom(*a):
        raise RuntimeError("ui bug")
    mon = DeviceMonitor(on_default_change=boom, on_list_change=boom)
    mon._diff([], [dev("Mic", True)])                    # must not raise


def test_monitor_thread_polls_until_stopped(monkeypatch):
    seq = iter([[dev("A", True)], None, [dev("A"), dev("B", True)]])
    monkeypatch.setattr(audio_devices, "snapshot", lambda *a, **k: next(seq, [dev("B", True)]))
    seen = []
    mon = DeviceMonitor(on_default_change=lambda d, ds: seen.append(d["name"]),
                        poll_interval_s=0.01)
    mon.start()
    import time
    deadline = time.monotonic() + 5
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    mon.stop()
    assert seen[0] == "B"
    assert mon.default_device["name"] == "B"
