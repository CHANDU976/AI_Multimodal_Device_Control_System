"""vosk_engine.VoskEngine — recogniser lifecycle, grammar probing, segment
decoding and the streaming loop, against a scripted fake of the vosk module."""

import queue
import time

import numpy as np
import pytest

import vosk_engine
from vosk_engine import VoskEngine


def make_engine(**kwargs):
    kwargs.setdefault("audio_queue", queue.Queue())
    kwargs.setdefault("input_rate", 16000)       # passthrough resampler: bytes in == bytes out
    return VoskEngine(model_path="/models/en-in", **kwargs)


def test_load_model_configures_the_preview_recogniser(fake_vosk):
    engine = make_engine()
    engine.load_model()
    assert fake_vosk.models == ["/models/en-in"]
    assert fake_vosk.log_level == vosk_engine.VOSK_CFG["log_level"]
    rec = fake_vosk.recognizers[0]
    assert rec.grammar is None and rec.words is True and rec.max_alternatives == 0
    assert rec.rate == 16000


def test_segment_decode_returns_scored_result(fake_vosk):
    fake_vosk.final = {"text": "open chrome", "result": [
        {"word": "open", "conf": 0.9}, {"word": "chrome", "conf": 0.8}]}
    engine = make_engine()
    pcm = np.zeros(16000, dtype=np.int16)             # 1 s at 16 kHz
    result = engine.recognize_segment(pcm)
    assert result.text == "open chrome" and result.is_final
    assert result.duration_s == pytest.approx(1.0)
    assert result.min_confidence == pytest.approx(0.8)


def test_grammar_recogniser_is_built_once_then_reset(fake_vosk):
    engine = make_engine()
    grammar = ["open", "chrome", "[unk]"]
    engine.recognize_segment(b"\x00\x00" * 160, grammar=grammar)
    engine.recognize_segment(b"\x00\x00" * 160, grammar=grammar)
    seg_recs = [r for r in fake_vosk.recognizers if r.grammar is not None]
    assert len(seg_recs) == 1 and seg_recs[0].grammar == grammar
    assert seg_recs[0].resets == 1                    # clean state for utterance 2
    assert engine._grammar_ok is True


def test_changing_grammar_rebuilds_the_recogniser(fake_vosk):
    engine = make_engine()
    engine.recognize_segment(b"\x00\x00", grammar=["a", "[unk]"])
    engine.recognize_segment(b"\x00\x00", grammar=["b", "[unk]"])
    engine.recognize_segment(b"\x00\x00", grammar=None)
    grammars = [r.grammar for r in fake_vosk.recognizers[1:]]
    assert grammars == [["a", "[unk]"], ["b", "[unk]"], None]


def test_model_without_dynamic_graph_falls_back_to_free_decoding(fake_vosk, caplog):
    fake_vosk.reject_grammar = True
    engine = make_engine()
    engine.recognize_segment(b"\x00\x00", grammar=["open", "[unk]"])
    assert engine._grammar_ok is False
    assert "rejected the runtime grammar" in caplog.text
    # The probe is permanent: no further grammar attempts on this model.
    engine.recognize_segment(b"\x00\x00", grammar=["other", "[unk]"])
    assert all(r.grammar is None for r in fake_vosk.recognizers)


def test_switch_model_hot_swaps_and_reprobes(fake_vosk):
    engine = make_engine()
    engine.load_model()
    engine._grammar_ok = False
    engine.switch_model("/models/hi-in")
    assert engine.model_path == "/models/hi-in"
    assert fake_vosk.models == ["/models/en-in", "/models/hi-in"]
    assert engine._grammar_ok is None and engine._seg_rec is None
    engine.switch_model("/models/hi-in")               # same model: no reload
    assert len(fake_vosk.models) == 2


def test_streaming_loop_emits_partials_and_finals(fake_vosk):
    results = []
    audio = queue.Queue()
    engine = make_engine(audio_queue=audio, result_callback=results.append)
    fake_vosk.partial = {"partial": "open chr"}
    engine.start()
    audio.put(b"\x01\x00" * 800)
    _wait(lambda: results)
    fake_vosk.accept = True
    fake_vosk.final = {"text": "open chrome", "result": [{"word": "open", "conf": 1.0},
                                                         {"word": "chrome", "conf": 1.0}]}
    audio.put(b"\x01\x00" * 800)
    _wait(lambda: any(r.is_final for r in results))
    engine.stop()
    assert results[0].event == "partial" and results[0].text == "open chr"
    final = next(r for r in results if r.is_final)
    assert final.text == "open chrome"
    assert not engine.is_running()


def test_preview_mode_drops_finals_but_reports_the_endpoint(fake_vosk):
    out = queue.Queue(maxsize=10)
    audio = queue.Queue()
    endpoints = []
    engine = make_engine(audio_queue=audio, result_queue=out, partials_only=True,
                         endpoint_callback=lambda: endpoints.append(True))
    fake_vosk.accept = True
    fake_vosk.final = {"text": "should be dropped"}
    engine.start()
    audio.put(b"\x01\x00" * 800)
    _wait(lambda: endpoints)
    engine.stop()
    assert out.empty()
    assert endpoints == [True]          # the utterance end still gets through


def test_full_result_queue_drops_the_oldest(fake_vosk):
    out = queue.Queue(maxsize=2)
    engine = make_engine(result_queue=out)
    from vosk_engine import VoskResult
    for i in range(4):
        engine._emit(VoskResult(event="partial", text=str(i), is_final=False))
    assert [out.get_nowait().text for _ in range(2)] == ["2", "3"]


def test_callback_errors_do_not_kill_the_loop(fake_vosk):
    def bad(result):
        raise RuntimeError("consumer bug")
    engine = make_engine(result_callback=bad)
    from vosk_engine import VoskResult
    engine._emit(VoskResult(event="partial", text="x", is_final=False))   # no raise
    engine.endpoint_callback = lambda: bad(None)
    engine._notify_endpoint()                                               # no raise


def test_context_manager(fake_vosk):
    with make_engine() as engine:
        assert engine.is_running()
    assert not engine.is_running()


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)
