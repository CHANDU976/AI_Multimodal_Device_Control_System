"""denoiser.py — DFN3 wrapper: optional by design, never loses audio.

torch and DeepFilterNet are replaced by minimal fakes, so the int16 <-> float
conversion, clipping, length handling, fallbacks and the torchaudio shim are
all tested without the 8 MB model or the torch stack.
"""

import contextlib
import sys
import types

import numpy as np
import pytest

import denoiser as dn
from denoiser import DeepFilterDenoiser
from helpers.audio import voice


class FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.arr, dim))

    def squeeze(self, dim):
        return FakeTensor(np.squeeze(self.arr, dim))

    def cpu(self):
        return self

    def numpy(self):
        return self.arr



@pytest.fixture
def fake_df(monkeypatch):
    """Install fake `df` / `df.enhance` / `torch` / `torchaudio` modules."""
    calls = {"enhance": [], "init": []}
    state = {"gain": 0.5, "sr": 48000, "fail": False, "trim": 0}

    def enhance(model, df_state, audio, pad=True, atten_lim_db=None):
        calls["enhance"].append({"pad": pad, "atten_lim_db": atten_lim_db,
                                 "shape": audio.arr.shape, "dtype": audio.arr.dtype})
        if state["fail"]:
            raise RuntimeError("CUDA out of memory")
        out = audio.arr * state["gain"]
        return FakeTensor(out[:, :out.shape[1] - state["trim"]] if state["trim"] else out)

    def init_df(**kwargs):
        calls["init"].append(kwargs)
        return "MODEL", types.SimpleNamespace(sr=lambda: state["sr"]), None

    df = types.ModuleType("df")
    df_enhance = types.ModuleType("df.enhance")
    df_enhance.enhance, df_enhance.init_df = enhance, init_df
    df.enhance = df_enhance
    torchaudio = types.ModuleType("torchaudio")
    torch_mod = types.ModuleType("torch")
    torch_mod.from_numpy, torch_mod.no_grad = FakeTensor, contextlib.nullcontext
    for name, mod in {"df": df, "df.enhance": df_enhance, "torch": torch_mod,
                      "torchaudio": torchaudio, "torchaudio.backend": None,
                      "torchaudio.backend.common": None}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return calls, state


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "config.ini").write_text("[df]\nsr = 48000\n")
    (tmp_path / "checkpoints").mkdir()
    return tmp_path


def live(model_dir, **kwargs):
    d = DeepFilterDenoiser(model_dir=str(model_dir), **kwargs)
    assert d.load() is True
    return d


# ── Loading and fallbacks ──────────────────────────────────────────────────────

def test_missing_model_dir_degrades_to_passthrough(tmp_path, caplog):
    d = DeepFilterDenoiser(model_dir=str(tmp_path / "absent"))
    assert d.load() is False and not d.available
    assert "fetch_deepfilternet_model.py" in caplog.text
    pcm = voice(0.5)
    assert d.enhance_int16(pcm) is pcm


def test_incomplete_download_is_named(tmp_path, caplog):
    DeepFilterDenoiser(model_dir=str(tmp_path)).load()
    assert "no config.ini" in caplog.text


def test_required_turns_failure_into_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="DeepFilterNet3 unavailable"):
        DeepFilterDenoiser(model_dir=str(tmp_path / "absent"), required=True).load()


def test_disabled_in_config(model_dir, fake_df):
    assert DeepFilterDenoiser(model_dir=str(model_dir), enabled=False).load() is False
    assert fake_df[0]["init"] == []


def test_load_passes_the_offline_options(model_dir, fake_df):
    d = live(model_dir, post_filter=True)
    init = fake_df[0]["init"][0]
    assert init["model_base_dir"] == str(model_dir)
    assert init["post_filter"] is True and init["log_file"] is None
    assert d.available and d.load() is True           # idempotent
    assert len(fake_df[0]["init"]) == 1


def test_sample_rate_mismatch_is_rejected(model_dir, fake_df, caplog):
    fake_df[1]["sr"] = 16000
    d = DeepFilterDenoiser(model_dir=str(model_dir))
    assert d.load() is False
    assert "model runs at 16000 Hz" in caplog.text


def test_torchaudio_shim_is_installed(model_dir, fake_df):
    live(model_dir)
    import torchaudio.backend.common as common
    assert hasattr(common, "AudioMetaData")


# ── Enhancement ────────────────────────────────────────────────────────────────

def test_enhance_scales_and_keeps_length(model_dir, fake_df):
    d = live(model_dir, atten_lim_db=10.0)
    pcm = voice(0.5)
    out = d.enhance_int16(pcm)
    assert out.dtype == np.int16 and len(out) == len(pcm)
    assert np.allclose(out, pcm * 0.5, atol=1)
    call = fake_df[0]["enhance"][0]
    assert call["pad"] is True and call["atten_lim_db"] == 10.0
    assert call["shape"] == (1, len(pcm)) and call["dtype"] == np.float32


def test_full_suppression_passes_none(model_dir, fake_df):
    live(model_dir, atten_lim_db=0.0).enhance_int16(voice(0.5))
    assert fake_df[0]["enhance"][0]["atten_lim_db"] is None


def test_output_is_clipped_not_wrapped(model_dir, fake_df):
    fake_df[1]["gain"] = 8.0
    loud = np.full(24000, 20000, dtype=np.int16)
    loud[::2] = -20000
    out = live(model_dir).enhance_int16(loud)
    assert out.max() == 32767 and out.min() == -32768


def test_short_buffers_skip_the_model(model_dir, fake_df):
    d = live(model_dir, min_duration_s=0.2)
    short = voice(0.1)
    assert d.enhance_int16(short) is short
    assert d.enhance_int16(np.zeros(0, np.int16)).size == 0
    assert fake_df[0]["enhance"] == []


def test_inference_failure_returns_raw_audio(model_dir, fake_df, caplog):
    fake_df[1]["fail"] = True
    pcm = voice(0.5)
    assert live(model_dir).enhance_int16(pcm) is pcm
    assert "using raw audio" in caplog.text


def test_short_model_output_is_not_padded_with_garbage(model_dir, fake_df):
    fake_df[1]["trim"] = 10
    pcm = voice(0.5)
    assert len(live(model_dir).enhance_int16(pcm)) == len(pcm) - 10


def test_close_releases_the_model(model_dir, fake_df):
    d = live(model_dir)
    d.close()
    assert not d.available
    pcm = voice(0.5)
    assert d.enhance_int16(pcm) is pcm


# ── torchaudio compatibility shim (in isolation) ───────────────────────────────

def test_shim_reuses_real_audiometadata_when_available(monkeypatch):
    ta = types.ModuleType("torchaudio")
    ta.AudioMetaData = type("AudioMetaData", (), {})
    for name, mod in {"torchaudio": ta, "torchaudio.backend": None,
                      "torchaudio.backend.common": None}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    dn._install_torchaudio_shim()
    assert sys.modules["torchaudio.backend.common"].AudioMetaData is ta.AudioMetaData
