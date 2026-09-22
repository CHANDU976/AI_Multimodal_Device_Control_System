"""calibrate.py — the routing-threshold calibration harness."""

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

import calibrate  # noqa: E402  (needs soundfile)
from helpers.audio import voice  # noqa: E402
from vosk_engine import parse_final  # noqa: E402


def test_normalize_for_comparison():
    assert calibrate._normalize("Open, Chrome!") == ["open", "chrome"]
    assert calibrate._normalize("don't STOP") == ["don't", "stop"]


def test_load_clip_downmixes_and_resamples(tmp_path):
    stereo = np.stack([voice(0.5), voice(0.5, seed=3)], axis=1)
    path = tmp_path / "clip.wav"
    sf.write(path, stereo, 48000, subtype="PCM_16")
    mono = calibrate._load_clip(path)
    assert mono.dtype == np.int16 and mono.ndim == 1
    assert abs(len(mono) - 8000) <= 8


def test_load_clip_16k_mono_untouched(tmp_path):
    path = tmp_path / "clip.wav"
    data = voice(0.25, rate=16000)
    sf.write(path, data, 16000, subtype="PCM_16")
    assert np.array_equal(calibrate._load_clip(path), data)


def test_describe_shows_unk_share():
    r = parse_final({"text": "open", "result": [{"word": "open", "conf": 0.9},
                                                {"word": "[unk]", "conf": 0.2}]})
    assert "unk 1/2" in calibrate._describe(r) and "'open'" in calibrate._describe(r)


class ScriptedEngine:
    """Returns a fixed decode per clip, free vs grammar-constrained."""
    free = {}
    grammar = {}

    def __init__(self, audio_queue=None):
        self.n = 0

    def load_model(self):
        pass

    def recognize_segment(self, pcm, grammar=None):
        table = self.grammar if grammar else self.free
        key = int(pcm[0])
        return parse_final(table[key])


def test_analyse_reports_and_suggests_thresholds(tmp_path, monkeypatch, capsys):
    for i, (text, conf) in enumerate([("open chrome", 0.97), ("scroll down", 0.93),
                                      ("volume up", 0.6)]):
        pcm = np.full(1600, i + 1, dtype=np.int16)          # sample value = clip id
        sf.write(tmp_path / f"clip_{i}.wav", pcm, 16000, subtype="PCM_16")
        (tmp_path / f"clip_{i}.txt").write_text(text if i < 2 else "mute", encoding="utf-8")
        words = [{"word": w, "conf": conf} for w in text.split()]
        ScriptedEngine.free[i + 1] = {"text": text, "result": words}
        ScriptedEngine.grammar[i + 1] = {"text": text, "result": words}
    monkeypatch.setattr(calibrate, "VoskEngine", ScriptedEngine)

    calibrate.analyse(tmp_path, denoise=False)
    out = capsys.readouterr().out
    assert "3 clips" in out
    assert out.count("OK") == 2 and out.count("WRONG") == 1
    assert "correct 2, wrong 1" in out
    assert "suggested  min_word_confidence ≈ 0.90" in out
    assert "wrong decodes    word conf: max 0.60" in out


def test_analyse_without_sidecars(tmp_path, monkeypatch, capsys):
    sf.write(tmp_path / "a.wav", np.full(1600, 1, np.int16), 16000, subtype="PCM_16")
    ScriptedEngine.free[1] = ScriptedEngine.grammar[1] = {"text": "hello"}
    monkeypatch.setattr(calibrate, "VoskEngine", ScriptedEngine)
    calibrate.analyse(tmp_path, denoise=False)
    assert "No .txt sidecars" in capsys.readouterr().out


def test_analyse_empty_folder_exits(tmp_path):
    with pytest.raises(SystemExit, match="No .wav files"):
        calibrate.analyse(tmp_path, denoise=False)
