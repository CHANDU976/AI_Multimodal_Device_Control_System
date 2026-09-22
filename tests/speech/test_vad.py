"""vad.py — segmentation of the microphone stream into utterances.

EnergyVAD runs on synthetic audio. SileroVAD runs against a fake ONNX
session that scores frames by energy, so its framing, alignment and layout
detection are tested without the model; the real model is exercised only
under the `models` marker.
"""

import numpy as np
import pytest

import vad as vad_mod
from config import SILERO, VAD as VAD_CFG
from helpers.audio import feed_all, noise, silence, voice
from helpers.fakes import FakeOnnxRuntime
from helpers.paths import require_real_file
from vad import EnergyVAD, Segment, SileroVAD, create_vad

RATE = 48000


def utterance(speech_s=1.5, lead_s=1.0, tail_s=1.5):
    return np.concatenate([noise(lead_s), voice(speech_s), noise(tail_s, seed=2)])


# ── EnergyVAD ──────────────────────────────────────────────────────────────────

def test_one_utterance_one_segment():
    segs = feed_all(EnergyVAD(), utterance())
    assert len(segs) == 1
    seg = segs[0]
    assert seg.pcm.dtype == np.int16
    # speech + preroll + hangover, give or take a frame or two.
    expected = 1.5 + VAD_CFG["preroll_ms"] / 1000 + VAD_CFG["hangover_ms"] / 1000
    assert abs(seg.duration_s - expected) < 0.15
    assert seg.duration_s == pytest.approx(len(seg.pcm) / RATE)
    assert not seg.truncated
    assert seg.peak_rms > seg.noise_floor * VAD_CFG["start_factor"]
    assert seg.speech_prob == 0.0                     # energy VAD has no probability


def test_two_utterances_two_segments():
    signal = np.concatenate([noise(1.0), voice(0.8), noise(1.2, seed=3), voice(0.8, seed=4),
                             noise(1.2, seed=5)])
    assert len(feed_all(EnergyVAD(), signal)) == 2


def test_silence_and_steady_noise_produce_nothing():
    assert feed_all(EnergyVAD(), silence(3.0)) == []
    assert feed_all(EnergyVAD(), noise(3.0, rms=500)) == []


def test_short_blip_is_discarded():
    blip = np.concatenate([noise(1.0), voice(0.08), noise(1.5, seed=2)])
    assert feed_all(EnergyVAD(), blip) == []


def test_runaway_segment_is_force_cut():
    vad = EnergyVAD(max_segment_s=2.0)
    segs = feed_all(vad, np.concatenate([noise(1.0), voice(5.0)]))
    assert segs[0].truncated
    assert segs[0].duration_s == pytest.approx(2.0, abs=0.05)


def test_flush_closes_speech_in_progress():
    vad = EnergyVAD()
    for chunk in np.array_split(np.concatenate([noise(1.0), voice(1.0)]), 20):
        assert vad.feed(chunk) is None
    assert vad.in_speech
    seg = vad.flush()
    assert seg is not None and not vad.in_speech
    assert vad.flush() is None


@pytest.mark.parametrize("size", [1000, 4800, 7777])
def test_chunk_size_does_not_change_the_result(size):
    reference = feed_all(EnergyVAD(), utterance())[0]
    other = feed_all(EnergyVAD(), utterance(), size=size)[0]
    assert len(other.pcm) == len(reference.pcm)


def test_accepts_bytes_and_arrays():
    as_bytes = feed_all(EnergyVAD(), utterance(), as_bytes=True)
    assert len(as_bytes) == 1


def test_noise_floor_adapts_and_reset_forgets_it():
    vad = EnergyVAD()
    assert vad.noise_floor == VAD_CFG["absolute_floor_rms"]
    feed_all(vad, noise(2.0, rms=800))
    assert vad.noise_floor > 400
    vad.reset()
    assert vad.noise_floor == VAD_CFG["absolute_floor_rms"]


def test_loud_room_raises_the_bar():
    # Speech barely above a loud fan: the relative threshold must not fire.
    quiet_voice = np.concatenate([noise(1.0, rms=2000), voice(1.0, amp=2500),
                                  noise(1.0, rms=2000, seed=2)])
    assert feed_all(EnergyVAD(), quiet_voice) == []


def test_empty_chunk():
    assert EnergyVAD().feed(np.zeros(0, dtype=np.int16)) is None


def test_segment_repr():
    seg = Segment(pcm=np.zeros(10, np.int16), duration_s=1.0, peak_rms=5, noise_floor=1,
                  truncated=True)
    assert "TRUNCATED" in repr(seg)


# ── SileroVAD (fake ONNX runtime) ──────────────────────────────────────────────

@pytest.fixture(params=["v5", "v6"])
def onnx(request, monkeypatch):
    fake = FakeOnnxRuntime(layout=request.param)
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake)
    return fake


def test_silero_detects_the_layout(onnx):
    vad = SileroVAD(model_path="fake.onnx")
    assert vad._layout == onnx.layout
    assert vad.frame_in == 1536 and vad.noise_floor == 0.0


def test_silero_segments_speech(onnx):
    segs = feed_all(SileroVAD(model_path="fake.onnx"), utterance())
    assert len(segs) == 1
    seg = segs[0]
    assert seg.speech_prob == pytest.approx(0.95, abs=0.01)
    assert abs(seg.duration_s - (1.5 + (SILERO["preroll_ms"] + SILERO["hangover_ms"]) / 1000)) < 0.2
    assert len(seg.pcm) % 1536 == 0                  # cut on 48 kHz model-frame boundaries
    assert onnx.calls > 0


def test_silero_ignores_noise(onnx):
    assert feed_all(SileroVAD(model_path="fake.onnx"), noise(3.0)) == []


def test_silero_reset(onnx):
    vad = SileroVAD(model_path="fake.onnx")
    vad.feed(voice(0.5))
    vad.reset()
    assert not vad.in_speech and vad._fifo16.size == 0 and vad._fifo48.size == 0


def test_silero_rejects_unknown_model_inputs(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeOnnxRuntime(layout="v9"))
    with pytest.raises(RuntimeError, match="Unrecognised Silero ONNX inputs"):
        SileroVAD(model_path="fake.onnx")


def test_silero_rejects_non_16k_multiple(onnx):
    with pytest.raises(ValueError, match="multiple of 16 kHz"):
        SileroVAD(sample_rate=44100, model_path="fake.onnx")


# ── create_vad fallback policy ─────────────────────────────────────────────────

def test_create_vad_prefers_silero(onnx):
    assert isinstance(create_vad(), SileroVAD)


def test_create_vad_falls_back_to_energy(monkeypatch):
    broken = FakeOnnxRuntime()
    broken.InferenceSession = lambda *a, **k: (_ for _ in ()).throw(OSError("no model"))
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", broken)
    assert isinstance(create_vad(), EnergyVAD)


def test_create_vad_required_raises(monkeypatch):
    broken = FakeOnnxRuntime()
    broken.InferenceSession = lambda *a, **k: (_ for _ in ()).throw(OSError("no model"))
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", broken)
    monkeypatch.setitem(vad_mod.SILERO_CFG, "required", True)
    with pytest.raises(OSError):
        create_vad()


def test_create_vad_disabled(monkeypatch):
    monkeypatch.setitem(vad_mod.SILERO_CFG, "enabled", False)
    assert isinstance(create_vad(), EnergyVAD)


# ── The real Silero model (Git LFS) ────────────────────────────────────────────

@pytest.mark.models
def test_real_silero_model_stays_quiet_on_noise():
    pytest.importorskip("onnxruntime")
    require_real_file(__import__("pathlib").Path(SILERO["model_path"]))
    vad = SileroVAD()
    assert feed_all(vad, np.concatenate([silence(1.0), noise(2.0)])) == []
