"""
denoiser.py
DeepFilterNet3 speech-enhancement front-end.

Where this sits in the pipeline (per the HLD):

    Mic (48k) ─┬─ VAD ─→ downsample → 16k ──────────────────→ VOSK    (commands)
               └─ buffer voiced audio ─→ DFN3 ─→ downsample → Whisper (dictation)

Only the dictation branch is denoised. VOSK stays on the raw signal because it
is the latency-critical path; Whisper is the accuracy-critical path and is the
one that gains from clean audio. DFN3 also wants a whole utterance to be worth
its cost, which is exactly what the dictation buffer already holds.

DFN3 runs at 48 kHz — that is why AUDIO["capture_rate"] is 48000. Feeding it
upsampled 16 kHz audio would work mechanically but throws away the high band
the model was trained to use, so capture natively at 48 kHz.

Optional by design
------------------
If the `df` package or the model directory is missing, the denoiser degrades
to a pass-through and the pipeline keeps working. Set DENOISE["required"]=True
in config.py to make a missing DFN3 a hard startup failure instead.

Person A — Real-Time Engine + Audio Pipeline
"""

from __future__ import annotations
import path_setup  # noqa: F401 — adds project root to sys.path

import logging
import os
import sys
import time
import types

import numpy as np

from config import DENOISE

logger = logging.getLogger(__name__)

_INT16_SCALE = 32768.0


def _install_torchaudio_shim() -> None:
    """
    DeepFilterNet 0.5.6 does `from torchaudio.backend.common import AudioMetaData`,
    but torchaudio removed that module in 2.1.

    It is only used as a type annotation on df.io.load_audio() — a file-reading
    helper this pipeline never calls, since enhance() is handed tensors
    directly. Registering a stub keeps the import working instead of pinning
    the whole project back to torch 2.0 / torchaudio 2.0 for a type hint.
    """
    import torchaudio                       # noqa: PLC0415

    try:
        import torchaudio.backend.common    # noqa: PLC0415, F401
        return                              # torchaudio < 2.1 — nothing to do
    except ImportError:
        pass

    meta = getattr(torchaudio, "AudioMetaData", None)
    if meta is None:
        class meta:                         # noqa: N801 — stands in for the real class
            """Placeholder AudioMetaData; never instantiated by this pipeline."""

    backend = sys.modules.get("torchaudio.backend")
    if backend is None:
        backend = types.ModuleType("torchaudio.backend")
        sys.modules["torchaudio.backend"] = backend
        torchaudio.backend = backend

    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = meta
    backend.common = common
    sys.modules["torchaudio.backend.common"] = common
    logger.debug("Installed torchaudio.backend.common shim for DeepFilterNet.")


class DeepFilterDenoiser:
    """
    Wraps DeepFilterNet3 as a single `enhance_int16()` call.

    Parameters
    ----------
    model_dir : str
        Directory holding `config.ini` and `checkpoints/`. See
        fetch_deepfilternet_model.py.
    sample_rate : int
        Rate of the audio passed to enhance_int16(). Must match the model's
        own rate (48 kHz for DFN3) — mismatches raise at load time.
    atten_lim_db : float
        Attenuation limit. 0.0 applies the model's full suppression; 10.0
        removes at most 10 dB and mixes the remaining noise back in.

        Counter-intuitively, the limited setting is the better one for ASR.
        Full suppression starts removing speech along with the noise once the
        SNR drops, and Whisper hallucinates over the resulting near-silence.
        See the note in config.DENOISE for the measured numbers.
    post_filter : bool
        DFN's optional post-filter — slightly more suppression at the cost of
        some speech artefacts. Off by default, since artefacts hurt Whisper.
    required : bool
        True → raise if DFN3 cannot be loaded. False → log and pass through.
    min_duration_s : float
        Buffers shorter than this skip enhancement (model warm-up and STFT
        padding dominate, and there is nothing useful to denoise).

    Usage
    -----
        dn = DeepFilterDenoiser()
        dn.load()                          # once, at startup
        clean = dn.enhance_int16(noisy)    # per utterance
    """

    def __init__(
        self,
        model_dir: str = DENOISE["model_dir"],
        sample_rate: int = DENOISE["sample_rate"],
        atten_lim_db: float = DENOISE["atten_lim_db"],
        post_filter: bool = DENOISE["post_filter"],
        required: bool = DENOISE["required"],
        min_duration_s: float = DENOISE["min_duration_s"],
        enabled: bool = DENOISE["enabled"],
    ) -> None:
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self.atten_lim_db = atten_lim_db
        self.post_filter = post_filter
        self.required = required
        self.min_duration_s = min_duration_s
        self.enabled = enabled

        self._torch = None
        self._model = None
        self._df_state = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True once the model is loaded and enhance_int16() will actually denoise."""
        return self._model is not None

    def load(self) -> bool:
        """
        Load DFN3. Returns True if the denoiser is live, False if the pipeline
        should run without denoising. Safe to call more than once.
        """
        if self._model is not None:
            return True

        if not self.enabled:
            logger.info("DFN3 disabled in config — dictation audio will not be denoised.")
            return False

        try:
            self._load()
        except Exception as exc:
            message = (
                f"DeepFilterNet3 unavailable: {exc} — "
                f"dictation audio will be sent to Whisper un-denoised."
            )
            if self.required:
                raise RuntimeError(message) from exc
            logger.warning(message)
            self._model = None
            return False

        return True

    def _load(self) -> None:
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"model directory not found at '{self.model_dir}' — run "
                f"'python fetch_deepfilternet_model.py' to download it"
            )
        if not os.path.isfile(os.path.join(self.model_dir, "config.ini")):
            raise FileNotFoundError(
                f"'{self.model_dir}' has no config.ini — the download looks incomplete; "
                f"re-run 'python fetch_deepfilternet_model.py --force'"
            )

        import torch                          # noqa: PLC0415 — heavy, import on demand

        _install_torchaudio_shim()
        from df.enhance import init_df        # noqa: PLC0415

        t0 = time.perf_counter()
        # log_file=None keeps DFN from writing an enhance.log into Models/DFN3.
        model, df_state, _ = init_df(
            model_base_dir=self.model_dir,
            post_filter=self.post_filter,
            log_level="WARNING",
            log_file=None,
            config_allow_defaults=True,
        )

        model_sr = int(df_state.sr())
        if model_sr != self.sample_rate:
            raise ValueError(
                f"model runs at {model_sr} Hz but the pipeline feeds it "
                f"{self.sample_rate} Hz — set AUDIO['capture_rate'] and "
                f"DENOISE['sample_rate'] to {model_sr}"
            )

        self._torch = torch
        self._model = model
        self._df_state = df_state
        logger.info(
            "DFN3 loaded from %s in %.1fs (%d Hz, atten_lim=%.1f dB, post_filter=%s)",
            self.model_dir, time.perf_counter() - t0,
            model_sr, self.atten_lim_db, self.post_filter,
        )

    # ── Enhancement ────────────────────────────────────────────────────────────

    def enhance_int16(self, pcm: np.ndarray) -> np.ndarray:
        """
        Denoise a mono int16 buffer at self.sample_rate.

        Returns a new int16 array of the same length. Falls back to returning
        the input unchanged whenever the model is unavailable, the buffer is
        too short to be worth it, or inference fails — denoising is an
        enhancement, never a reason to lose a dictation.
        """
        if self._model is None or pcm.size == 0:
            return pcm

        if pcm.size / self.sample_rate < self.min_duration_s:
            return pcm

        torch = self._torch
        t0 = time.perf_counter()
        try:
            # df.enhance wants float [-1, 1] shaped [channels, samples]
            audio = torch.from_numpy(
                (pcm.astype(np.float32) / _INT16_SCALE)
            ).unsqueeze(0)

            from df.enhance import enhance   # noqa: PLC0415
            with torch.no_grad():
                clean = enhance(
                    self._model,
                    self._df_state,
                    audio,
                    pad=True,                       # keeps output length == input length
                    atten_lim_db=self.atten_lim_db or None,
                )

            out = clean.squeeze(0).cpu().numpy()
            out = np.clip(out * _INT16_SCALE, -_INT16_SCALE, _INT16_SCALE - 1).astype(np.int16)
        except Exception as exc:
            logger.warning("DFN3 enhancement failed (%s) — using raw audio.", exc)
            return pcm

        duration_s = pcm.size / self.sample_rate
        elapsed = time.perf_counter() - t0
        logger.debug(
            "DFN3 enhanced %.2fs of audio in %.2fs (RTF %.2f)",
            duration_s, elapsed, elapsed / duration_s if duration_s else 0.0,
        )
        # pad=True should preserve length, but never let a mismatch propagate.
        return out[:pcm.size] if out.size >= pcm.size else out

    def close(self) -> None:
        """Release the model (frees a few hundred MB of RAM)."""
        self._model = None
        self._df_state = None
        self._torch = None


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s: %(message)s")

    from audio_stream import AudioStream
    from config import AUDIO

    denoiser = DeepFilterDenoiser(required=True)
    denoiser.load()

    seconds = 5
    print(f"\nRecording {seconds}s at {AUDIO['capture_rate']} Hz — talk over some noise …")
    chunks: list[bytes] = []
    with AudioStream() as stream:
        target = seconds * AUDIO["capture_rate"] // AUDIO["chunk_size"]
        while len(chunks) < target:
            chunk = stream.chunk_queue.get()
            if chunk is None:
                break
            chunks.append(chunk)

    noisy = np.frombuffer(b"".join(chunks), dtype=np.int16)
    clean = denoiser.enhance_int16(noisy)

    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))

    print(f"\nnoisy RMS : {_rms(noisy):8.1f}")
    print(f"clean RMS : {_rms(clean):8.1f}")

    try:
        import soundfile as sf
        sf.write("dfn3_noisy.wav", noisy, AUDIO["capture_rate"])
        sf.write("dfn3_clean.wav", clean, AUDIO["capture_rate"])
        print("\nWrote dfn3_noisy.wav / dfn3_clean.wav — listen to compare.")
    except ImportError:
        print("\n(pip install soundfile to also dump wav files for listening)")
