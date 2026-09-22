"""whisper_engine.py — model loading policy, the worker request loop, the
engine's process management and DictationBuffer. faster-whisper is faked;
one slow test spawns a real worker process to prove the start-up error path."""

import queue
import sys
import types

import numpy as np
import pytest

import whisper_engine as we
from whisper_engine import DictationBuffer, WhisperEngine, WhisperResult, _SHUTDOWN


# ── WhisperResult ──────────────────────────────────────────────────────────────

def test_result_success_and_repr():
    ok = WhisperResult("hello", "en", [], 0.5)
    bad = WhisperResult("", "", [], 0.0, error="boom")
    assert ok.success and not bad.success
    assert "hello" in repr(ok) and "ERROR: boom" in repr(bad)


# ── _load_whisper_model ────────────────────────────────────────────────────────

@pytest.fixture
def faster_whisper(monkeypatch):
    created = []

    class WhisperModel:
        def __init__(self, what, **kwargs):
            created.append((what, kwargs))

    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    return created


def test_local_model_dir_is_preferred(tmp_path, faster_whisper):
    we._load_whisper_model(str(tmp_path), "small", True, "cpu", "int8")
    assert faster_whisper == [(str(tmp_path), {"device": "cpu", "compute_type": "int8"})]


def test_missing_dir_downloads_next_to_it(tmp_path, faster_whisper):
    target = tmp_path / "faster-whisper-small"
    we._load_whisper_model(str(target), "small", True, "cpu", "int8")
    what, kwargs = faster_whisper[0]
    assert what == "small" and kwargs["download_root"] == str(tmp_path)


def test_missing_dir_without_download_fails_clearly(tmp_path, faster_whisper):
    with pytest.raises(FileNotFoundError, match="WHISPER_FALLBACK_DOWNLOAD"):
        we._load_whisper_model(str(tmp_path / "nope"), "small", False, "cpu", "int8")


# ── _whisper_worker request loop (run in-process) ──────────────────────────────

class FakeModel:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        if self.fail:
            raise RuntimeError("decoder crashed")
        seg = types.SimpleNamespace(id=0, start=0.0, end=1.2, text=" hello there ",
                                    avg_logprob=-0.2, no_speech_prob=0.01)
        seg2 = types.SimpleNamespace(id=1, start=1.2, end=2.0, text=" friend",
                                     avg_logprob=-0.3, no_speech_prob=0.02)
        return iter([seg, seg2]), types.SimpleNamespace(language=kwargs["language"])


def run_worker(monkeypatch, requests, model=None, load_error=None, options=None):
    def load(*a, **k):
        if load_error:
            raise load_error
        return model
    monkeypatch.setattr(we, "_load_whisper_model", load)
    req, res = queue.Queue(), queue.Queue()
    for item in requests:
        req.put(item)
    we._whisper_worker("p", "small", False, req, res, options or {"language": "en"})
    out = []
    while not res.empty():
        out.append(res.get_nowait())
    return out


def test_worker_signals_ready_then_transcribes(monkeypatch):
    model = FakeModel()
    audio = (np.ones(1600) * 16384).astype(np.int16)
    out = run_worker(monkeypatch, [(audio, 16000, "hi"), _SHUTDOWN], model)
    assert out[0] == "READY"
    result = out[1]
    assert result.success and result.text == "hello there friend"
    assert result.language == "hi" and len(result.segments) == 2
    assert result.segments[0]["end"] == 1.2
    fed, kwargs = model.calls[0]
    assert fed.dtype == np.float32 and np.allclose(fed, 0.5)      # int16 → [-1, 1]
    assert kwargs["language"] == "hi" and kwargs["beam_size"] == 5


def test_worker_uses_configured_language_when_none_given(monkeypatch):
    model = FakeModel()
    run_worker(monkeypatch, [(np.zeros(10, np.int16), 16000, None), _SHUTDOWN], model,
               options={"language": "en", "beam_size": 2})
    assert model.calls[0][1]["language"] == "en" and model.calls[0][1]["beam_size"] == 2


def test_worker_reports_decode_errors_and_keeps_serving(monkeypatch):
    model = FakeModel(fail=True)
    out = run_worker(monkeypatch, [(np.zeros(10, np.int16), 16000, None),
                                   (np.zeros(10, np.int16), 16000, None), _SHUTDOWN], model)
    assert out[0] == "READY"
    assert [r.error for r in out[1:]] == ["decoder crashed"] * 2


def test_worker_reports_load_failure_without_ready(monkeypatch):
    out = run_worker(monkeypatch, [], load_error=OSError("model.bin missing"))
    assert len(out) == 1 and not out[0].success
    assert "Model load failed: model.bin missing" in out[0].error


def test_worker_resamples_non_16k_input(monkeypatch):
    seen = {}
    librosa = types.ModuleType("librosa")

    def resample(x, orig_sr, target_sr):
        seen.update(orig=orig_sr, target=target_sr)
        return x[::3]
    librosa.resample = resample
    monkeypatch.setitem(sys.modules, "librosa", librosa)
    model = FakeModel()
    run_worker(monkeypatch, [(np.zeros(4800, np.int16), 48000, None), _SHUTDOWN], model)
    assert seen == {"orig": 48000, "target": 16000}
    assert len(model.calls[0][0]) == 1600


def test_worker_without_librosa_still_transcribes(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "librosa", None)
    model = FakeModel()
    out = run_worker(monkeypatch, [(np.zeros(4800, np.int16), 48000, None), _SHUTDOWN], model)
    assert out[1].success and "librosa not installed" in caplog.text


# ── WhisperEngine ──────────────────────────────────────────────────────────────

def test_unknown_model_key():
    with pytest.raises(ValueError, match="Unknown model key 'huge'"):
        WhisperEngine("huge")


def test_default_options_come_from_config():
    engine = WhisperEngine("base")
    assert engine.model_path == we.WHISPER_MODELS["base"]
    assert engine.options["compute_type"] == we.WHISPER_CFG["compute_type"]
    assert set(engine.options) == {"language", "task", "device", "compute_type",
                                   "temperature", "beam_size", "best_of",
                                   "condition_on_previous_text"}


def test_transcribe_before_start():
    with pytest.raises(RuntimeError, match="not started"):
        WhisperEngine().transcribe(np.zeros(10, np.int16))


def test_transcribe_round_trip_and_timeout():
    engine = WhisperEngine()
    engine._request_q, engine._result_q = queue.Queue(), queue.Queue()
    engine._result_q.put(WhisperResult("ok", "en", [], 0.1))
    audio = np.zeros(10, np.int16)
    assert engine.transcribe(audio, language="hi").text == "ok"
    sent = engine._request_q.get_nowait()
    assert sent[0] is audio and sent[1:] == (16000, "hi")
    with pytest.raises(TimeoutError):
        engine.transcribe(audio, timeout=0.05)


def test_stop_without_start_is_safe():
    WhisperEngine().stop()


@pytest.mark.slow
def test_worker_process_load_failure_surfaces_at_start(tmp_path, monkeypatch):
    """Real spawn: a missing model with downloads disabled must fail start(),
    not hang or leave a zombie worker."""
    monkeypatch.setattr(we, "WHISPER_FALLBACK_DOWNLOAD", False)
    engine = WhisperEngine()
    engine.model_path = str(tmp_path / "missing-model")
    with pytest.raises(RuntimeError, match="load error"):
        engine.start(ready_timeout=120)
    engine.stop()
    assert not engine._worker.is_alive()


# ── DictationBuffer ────────────────────────────────────────────────────────────

def test_dictation_buffer_flushes_after_silence(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(we.time, "monotonic", lambda: now[0])
    buf = DictationBuffer(silence_timeout_s=2.0, silence_threshold=300)
    loud = (np.ones(4800) * 3000).astype(np.int16).tobytes()
    quiet = np.zeros(4800, np.int16).tobytes()
    assert buf.feed(loud) is False
    assert buf.feed(quiet) is False            # silence starts
    now[0] = 1.0
    assert buf.feed(quiet) is False
    now[0] = 1.5
    assert buf.feed(loud) is False             # speech resets the clock
    now[0] = 2.0
    assert buf.feed(quiet) is False
    now[0] = 4.1
    assert buf.feed(quiet) is True
    assert len(buf.to_numpy()) == 6 * 4800
    buf.reset()
    assert buf.to_numpy().size == 0
