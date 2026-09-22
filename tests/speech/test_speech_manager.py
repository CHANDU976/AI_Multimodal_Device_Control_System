"""speech_manager.py — segment → denoise → VOSK → route → (Whisper) → event.

The manager is built for real; its engines are swapped for fakes so every
routing outcome, the language switch, the audio tee and device following can
be driven deterministically without a microphone or a model.
"""

import numpy as np
import pytest

import speech_manager as sm
from config import CONCURRENCY, VOSK_MODELS, command_grammar
from helpers.audio import voice
from vad import EnergyVAD, Segment
from vosk_engine import VoskResult
from whisper_engine import WhisperResult


class FakeVoskEngine:
    def __init__(self, result):
        self.result, self.calls, self.switched = result, [], []

    def recognize_segment(self, pcm16, grammar=None):
        self.calls.append((pcm16, grammar))
        return self.result

    def switch_model(self, path):
        self.switched.append(path)


class FakeWhisperEngine:
    def __init__(self, result):
        self.result, self.calls = result, []

    def transcribe(self, audio, language=None):
        self.calls.append((audio, language))
        return self.result


_KEEP = object()                      # mirrors AudioStream._KEEP: "same device"


class FakeStream:
    def __init__(self):
        self.restarts = []

    def restart(self, device_index=_KEEP):
        self.restarts.append(device_index)


def vosk_result(text, mean=0.95, lo=0.9, unk=0, dur=1.0):
    words = [{"word": w, "conf": mean, "start": 0.0, "end": 0.2} for w in text.split()]
    words += [{"word": "[unk]", "conf": 0.1}] * unk
    return VoskResult(event="command", text=text, is_final=True, words=words,
                      mean_confidence=mean, min_confidence=lo, duration_s=dur,
                      unk_count=unk, unk_ratio=unk / len(words) if words else 0.0)


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(sm, "create_vad", lambda: EnergyVAD())
    events = []
    mgr = sm.SpeechManager(callback=events.append)
    mgr.events = events
    return mgr


def segment(seconds=1.0, speech_prob=0.0):
    pcm = voice(seconds)
    return Segment(pcm=pcm, duration_s=seconds, peak_rms=5000.0, noise_floor=40.0,
                   wall_start=1_700_000_000.0, speech_prob=speech_prob)


# ── Routing outcomes ───────────────────────────────────────────────────────────

def test_confident_command_is_emitted_from_vosk(manager):
    manager._vosk_engine = FakeVoskEngine(vosk_result("open chrome"))
    manager._whisper_engine = FakeWhisperEngine(None)
    manager._handle_segment(segment(1.0, speech_prob=0.91))

    (event,) = manager.events
    assert (event.event_type, event.text, event.source) == ("command", "open chrome", "vosk")
    meta = event.metadata
    assert meta["denoised"] is False                      # no DFN3 loaded in tests
    assert meta["segment_s"] == 1.0 and meta["segment_start"] == 1_700_000_000.0
    assert meta["speech_prob"] == 0.91 and meta["mean_confidence"] == 0.95
    assert "routing_reason" in meta and "vosk_s" in meta
    assert manager._whisper_engine.calls == []
    assert manager.get_result(timeout=0.01) is event      # pull API mirrors the callback


def test_vosk_receives_16k_audio_and_the_command_grammar(manager):
    manager._vosk_engine = FakeVoskEngine(vosk_result("open chrome"))
    manager._handle_segment(segment(1.0))
    pcm16, grammar = manager._vosk_engine.calls[0]
    assert abs(len(pcm16) - 16000) <= 16                  # 48 kHz segment → 16 kHz
    assert grammar == command_grammar()


def test_unsure_decode_escalates_to_whisper_with_the_same_audio(manager):
    manager._vosk_engine = FakeVoskEngine(vosk_result("the report is due", mean=0.5, lo=0.4))
    manager._whisper_engine = FakeWhisperEngine(
        WhisperResult("The report is due Monday.", "en", [{"id": 0}], 0.8))
    manager._handle_segment(segment(2.0))

    (event,) = manager.events
    assert (event.event_type, event.source) == ("dictation", "whisper")
    assert event.text == "The report is due Monday."
    assert event.metadata["vosk_text"] == "the report is due"
    assert event.metadata["language"] == "en" and "whisper_s" in event.metadata
    sent_audio, lang = manager._whisper_engine.calls[0]
    assert sent_audio is manager._vosk_engine.calls[0][0]  # no second preprocessing pass
    assert lang is None


def test_oov_count_is_reported(manager):
    manager._vosk_engine = FakeVoskEngine(vosk_result("open", unk=2))
    manager._whisper_engine = FakeWhisperEngine(WhisperResult("open blender", "en", [], 0.1))
    manager._handle_segment(segment())
    assert manager.events[0].metadata["unk_count"] == 2


def test_whisper_error_becomes_an_error_event(manager):
    manager._vosk_engine = FakeVoskEngine(vosk_result("", mean=0.0, lo=0.0))
    manager._whisper_engine = FakeWhisperEngine(WhisperResult("", "", [], 0.1, error="timeout"))
    manager._handle_segment(segment())
    (event,) = manager.events
    assert event.event_type == "error" and event.metadata["error"] == "timeout"


def test_whisper_silence_emits_nothing(manager):
    manager._vosk_engine = FakeVoskEngine(vosk_result("", mean=0.0, lo=0.0))
    manager._whisper_engine = FakeWhisperEngine(WhisperResult("", "en", [], 0.1))
    manager._handle_segment(segment())
    assert manager.events == []


def test_escalation_without_whisper_is_dropped(manager):
    manager._vosk_engine = FakeVoskEngine(vosk_result("mumble", mean=0.2, lo=0.1))
    manager._whisper_engine = None
    manager._handle_segment(segment())
    assert manager.events == []


def test_processing_errors_are_reported_not_fatal(manager):
    class Broken:
        def recognize_segment(self, *a, **k):
            raise RuntimeError("kaldi exploded")
    manager._vosk_engine = Broken()
    manager._running.set()
    manager._segment_queue.put(segment())
    manager._segment_queue.put(None)
    manager._process_loop()
    (event,) = manager.events
    assert event.event_type == "error" and "kaldi exploded" in event.metadata["error"]


def test_user_callback_errors_are_contained(monkeypatch):
    monkeypatch.setattr(sm, "create_vad", lambda: EnergyVAD())

    def bad(event):
        raise RuntimeError("ui bug")
    mgr = sm.SpeechManager(callback=bad)
    mgr._emit(sm.SpeechEvent("command", "x", "vosk", "command"))
    assert mgr.get_result(timeout=0.01).text == "x"


# ── Language switching ─────────────────────────────────────────────────────────

def test_switch_to_hindi_drops_the_english_grammar(manager):
    manager._vosk_engine = FakeVoskEngine(None)
    manager.set_language("hi-in")
    assert manager._vosk_engine.switched == [VOSK_MODELS["hi-in"]]
    assert manager._whisper_language == "hi" and manager._grammar is None
    manager.set_language("en-us")
    assert manager._grammar == command_grammar() and manager._whisper_language == "en"


def test_language_before_start_is_remembered(manager):
    manager.set_language("en-us")
    assert manager._vosk_model_path == VOSK_MODELS["en-us"]


def test_unknown_language_is_rejected(manager):
    with pytest.raises(ValueError, match="Unknown language 'fr-fr'"):
        manager.set_language("fr-fr")


# ── Audio plumbing ─────────────────────────────────────────────────────────────

def test_tee_queue_feeds_vad_and_preview(manager):
    tee = manager._make_tee_queue()
    tee.put_nowait(b"chunk")
    assert manager._vad_queue.get_nowait() == b"chunk"
    assert manager._preview_queue.get_nowait() == b"chunk"


def test_tee_without_partials_skips_preview(monkeypatch):
    monkeypatch.setattr(sm, "create_vad", lambda: EnergyVAD())
    mgr = sm.SpeechManager(emit_partials=False)
    mgr._make_tee_queue().put_nowait(b"chunk")
    assert mgr._preview_queue.empty() and not mgr._vad_queue.empty()


def test_offer_drops_the_oldest_when_full():
    import queue
    q = queue.Queue(maxsize=2)
    for item in (1, 2, 3):
        sm.SpeechManager._offer(q, item)
    assert [q.get_nowait(), q.get_nowait()] == [2, 3]


def test_vad_loop_turns_chunks_into_segments(manager):
    manager._running.set()
    signal = np.concatenate([np.zeros(48000, np.int16), voice(1.0),
                             np.zeros(96000, np.int16)])
    for i in range(0, len(signal), 4800):
        manager._vad_queue.put(signal[i:i + 4800].tobytes())
    manager._vad_queue.put(None)
    manager._vad_loop()
    first = manager._segment_queue.get_nowait()
    assert isinstance(first, Segment)
    assert manager._segment_queue.get_nowait() is None      # end-of-stream marker


def test_partials_are_forwarded_finals_are_not(manager):
    manager._on_preview_result(VoskResult(event="partial", text="open chr", is_final=False))
    manager._on_preview_result(VoskResult(event="command", text="open chrome", is_final=True))
    manager._on_preview_result(VoskResult(event="partial", text="", is_final=False))
    assert [(e.event_type, e.text) for e in manager.events] == [("partial", "open chr")]


# ── Input devices ──────────────────────────────────────────────────────────────

def test_default_device_change_is_followed(manager):
    manager._audio_stream = FakeStream()
    manager._running.set()
    manager._on_default_device_change({"name": "Headset"}, [])
    assert len(manager._audio_stream.restarts) == 1
    (event,) = manager.events
    assert (event.event_type, event.text) == ("device", "Headset")
    assert event.metadata["reason"] == "default_changed"


def test_pinned_device_ignores_default_changes(manager):
    manager._audio_stream = FakeStream()
    manager._running.set()
    manager.set_input_device(3)
    manager._on_default_device_change({"name": "Headset"}, [])
    assert manager._audio_stream.restarts == [3]             # only the explicit pin
    assert manager.events == []
    manager.set_input_device(None)                           # back to following
    assert manager._follow_default


def test_device_list_change_event(manager):
    manager._on_device_list_change([{"name": "Mic"}])
    assert manager.events[0].metadata == {"reason": "list_changed", "devices": [{"name": "Mic"}]}


def test_no_device_monitor_means_no_devices(manager):
    assert manager.input_devices == []


def test_noise_floor_is_exposed(manager):
    assert manager.noise_floor > 0


def test_event_repr():
    assert "open chrome" in repr(sm.SpeechEvent("command", "open chrome", "vosk", "command"))


def test_queues_are_bounded(manager):
    assert manager._segment_queue.maxsize == CONCURRENCY["segment_queue_maxsize"]
    assert manager._output_queue.maxsize == CONCURRENCY["result_queue_maxsize"]
