"""
whisper_engine.py
Runs OpenAI Whisper in a separate process to avoid blocking the VOSK
real-time loop with heavy model inference.

Person B — Dictation + Intelligence Layer

Model loading priority
----------------------
1. Load from local .pt file  (e.g. models/whisper/base.pt)
2. If file missing and WHISPER_FALLBACK_DOWNLOAD=True → download from OpenAI
3. If file missing and WHISPER_FALLBACK_DOWNLOAD=False → raise FileNotFoundError
"""

from __future__ import annotations
import path_setup  # noqa: F401 — adds project root to sys.path

import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass

import numpy as np
import whisper

from config import (
    WHISPER as WHISPER_CFG,
    WHISPER_MODELS,
    WHISPER_DEFAULT_MODEL,
    WHISPER_FALLBACK_DOWNLOAD,
    AUDIO,
    CONCURRENCY,
)

logger = logging.getLogger(__name__)

# Sentinel that tells the worker to exit cleanly
_SHUTDOWN = "<<SHUTDOWN>>"


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class WhisperResult:
    text: str
    language: str
    segments: list[dict]
    duration_s: float       # inference wall-time
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def __repr__(self) -> str:
        if self.error:
            return f"WhisperResult(ERROR: {self.error})"
        return f"WhisperResult({self.text!r}  [{self.duration_s:.2f}s])"


# ── Model loader (used inside child process) ───────────────────────────────────

def _load_whisper_model(model_path: str, model_key: str, allow_download: bool):
    """
    Load Whisper preferring the local .pt file.

    Parameters
    ----------
    model_path : str
        Path to the local .pt file (e.g. models/whisper/base.pt).
    model_key : str
        Whisper size name ('base', 'tiny', ...) used as fallback download key.
    allow_download : bool
        If True and the file is missing, download from OpenAI CDN.

    Raises
    ------
    FileNotFoundError
        If the file is missing and allow_download is False.
    """
    import warnings
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

    if os.path.isfile(model_path):
        logger.info("[WhisperWorker] Loading local model: %s", model_path)
        return whisper.load_model(model_path)

    # ── File not found ─────────────────────────────────────────────────────────
    if allow_download:
        logger.warning(
            "[WhisperWorker] '%s' not found — downloading '%s' from OpenAI "
            "(this may take a while).",
            model_path,
            model_key,
        )
        return whisper.load_model(model_key)   # saves to ~/.cache/whisper

    raise FileNotFoundError(
        f"Whisper model not found at '{model_path}'. "
        f"Either place {model_key}.pt there or set WHISPER_FALLBACK_DOWNLOAD=True in config.py."
    )


# ── Worker (runs in child process) ────────────────────────────────────────────

def _whisper_worker(
    model_path: str,
    model_key: str,
    allow_download: bool,
    request_q: mp.Queue,
    result_q: mp.Queue,
    options: dict,
) -> None:
    """
    Long-lived worker: loads the Whisper model once and serves
    transcription requests until it receives the _SHUTDOWN sentinel.
    """
    try:
        model = _load_whisper_model(model_path, model_key, allow_download)
    except Exception as exc:
        result_q.put(WhisperResult(
            text="", language="", segments=[],
            duration_s=0.0, error=f"Model load failed: {exc}"
        ))
        return

    # Signal that the model is ready before entering the request loop
    result_q.put("READY")
    logger.info("[WhisperWorker] Ready. Waiting for audio requests …")

    while True:
        item = request_q.get()      # blocks until a request arrives

        if item == _SHUTDOWN:
            logger.info("[WhisperWorker] Shutdown signal received.")
            break

        audio_np, sample_rate = item
        audio_float = audio_np.astype(np.float32) / 32768.0    # int16 → [-1, 1] float

        # Resample if needed (Whisper requires 16 kHz)
        if sample_rate != 16000:
            try:
                import librosa
                audio_float = librosa.resample(
                    audio_float, orig_sr=sample_rate, target_sr=16000
                )
            except ImportError:
                logger.warning(
                    "librosa not installed — cannot resample %d Hz to 16000 Hz. "
                    "pip install librosa",
                    sample_rate,
                )

        t0 = time.perf_counter()
        try:
            raw = model.transcribe(
                audio_float,
                language=options.get("language", "en"),
                task=options.get("task", "transcribe"),
                fp16=options.get("fp16", False),
                temperature=options.get("temperature", 0.0),
                beam_size=options.get("beam_size", 5),
                best_of=options.get("best_of", 5),
                condition_on_previous_text=options.get("condition_on_previous_text", True),
            )
            result = WhisperResult(
                text=raw["text"].strip(),
                language=raw.get("language", ""),
                segments=raw.get("segments", []),
                duration_s=time.perf_counter() - t0,
            )
        except Exception as exc:
            result = WhisperResult(
                text="", language="", segments=[],
                duration_s=time.perf_counter() - t0,
                error=str(exc),
            )

        result_q.put(result)


# ── Engine (runs in main process) ─────────────────────────────────────────────

class WhisperEngine:
    """
    Manages the Whisper worker process and provides a synchronous
    `transcribe()` call that blocks until the result is available.

    Parameters
    ----------
    model_key : str
        Key into WHISPER_MODELS in config.py — 'base' or 'tiny'.
        Defaults to WHISPER_DEFAULT_MODEL ('base').
    sample_rate : int
        Sample rate of audio passed to transcribe(). 16000 is preferred.
    options : dict, optional
        Override Whisper transcription options from config.WHISPER.

    Examples
    --------
    engine = WhisperEngine()              # uses models/whisper/base.pt
    engine = WhisperEngine("tiny")        # uses models/whisper/tiny.pt
    """

    def __init__(
        self,
        model_key: str = WHISPER_DEFAULT_MODEL,
        sample_rate: int = AUDIO["sample_rate"],
        options: dict | None = None,
    ) -> None:
        if model_key not in WHISPER_MODELS:
            raise ValueError(
                f"Unknown model key '{model_key}'. "
                f"Valid keys: {list(WHISPER_MODELS.keys())}"
            )
        self.model_key  = model_key
        self.model_path = WHISPER_MODELS[model_key]
        self.sample_rate = sample_rate
        self.options = options or {
            k: WHISPER_CFG[k]
            for k in (
                "language", "task", "fp16",
                "temperature", "beam_size", "best_of",
                "condition_on_previous_text",
            )
        }
        self._request_q: mp.Queue | None = None
        self._result_q: mp.Queue | None = None
        self._worker: mp.Process | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self, wait_for_ready: bool = True, ready_timeout: float = 60.0) -> None:
        """
        Spawn the worker process (loads the Whisper model in the child).

        Parameters
        ----------
        wait_for_ready : bool
            Block until the worker signals the model is loaded.
            Recommended — prevents transcribe() from being called too early.
        ready_timeout : float
            Seconds to wait for the ready signal before raising TimeoutError.
        """
        if self._worker and self._worker.is_alive():
            logger.warning("WhisperEngine worker already running.")
            return

        logger.info(
            "Starting WhisperEngine — key='%s'  path='%s'  fallback=%s",
            self.model_key, self.model_path, WHISPER_FALLBACK_DOWNLOAD,
        )

        ctx = mp.get_context("spawn")   # safe on Windows + macOS + Linux
        self._request_q = ctx.Queue(maxsize=CONCURRENCY["result_queue_maxsize"])
        self._result_q  = ctx.Queue(maxsize=CONCURRENCY["result_queue_maxsize"])
        self._worker = ctx.Process(
            target=_whisper_worker,
            args=(
                self.model_path,
                self.model_key,
                WHISPER_FALLBACK_DOWNLOAD,
                self._request_q,
                self._result_q,
                self.options,
            ),
            name="WhisperWorker",
            daemon=True,
        )
        self._worker.start()
        logger.info("WhisperEngine worker started (PID %d).", self._worker.pid)

        if wait_for_ready:
            try:
                signal = self._result_q.get(timeout=ready_timeout)
            except Exception:
                raise TimeoutError(
                    f"WhisperEngine worker did not become ready within {ready_timeout}s. "
                    "Check model path or enable WHISPER_FALLBACK_DOWNLOAD in config.py."
                )
            if isinstance(signal, WhisperResult) and not signal.success:
                raise RuntimeError(f"WhisperEngine load error: {signal.error}")
            # signal == "READY" → good to go

    def stop(self) -> None:
        """Send shutdown sentinel and wait for the worker to exit."""
        if self._request_q:
            try:
                self._request_q.put(_SHUTDOWN, timeout=2)
            except Exception:
                pass
        if self._worker:
            self._worker.join(timeout=CONCURRENCY["whisper_process_timeout"])
            if self._worker.is_alive():
                logger.warning("Whisper worker did not exit cleanly — terminating.")
                self._worker.terminate()
        logger.info("WhisperEngine stopped.")

    # ── Transcription ──────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio: np.ndarray,
        timeout: float = CONCURRENCY["whisper_process_timeout"],
    ) -> WhisperResult:
        """
        Send a numpy int16 audio array to the worker and block until done.

        Parameters
        ----------
        audio : np.ndarray  (dtype=int16, shape=(N,))
            Raw PCM audio buffer at self.sample_rate Hz.
        timeout : float
            Seconds to wait before raising TimeoutError.
        """
        if self._request_q is None or self._result_q is None:
            raise RuntimeError("WhisperEngine is not started. Call start() first.")

        self._request_q.put((audio, self.sample_rate))
        try:
            return self._result_q.get(timeout=timeout)
        except Exception:
            raise TimeoutError(f"Whisper transcription timed out after {timeout}s.")

    # ── Context-manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "WhisperEngine":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


# ── Audio buffer collector ────────────────────────────────────────────────────

class DictationBuffer:
    """
    Accumulates raw PCM chunks during dictation mode and returns a single
    numpy array ready for Whisper when silence is detected.
    """

    def __init__(
        self,
        sample_rate: int = AUDIO["sample_rate"],
        silence_timeout_s: float = WHISPER_CFG["silence_timeout_s"],
        silence_threshold: float = 300.0,   # RMS amplitude (int16 scale)
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_timeout_s = silence_timeout_s
        self.silence_threshold = silence_threshold
        self._chunks: list[bytes] = []
        self._silent_since: float | None = None

    def reset(self) -> None:
        self._chunks.clear()
        self._silent_since = None

    def feed(self, chunk: bytes) -> bool:
        """
        Add a PCM chunk.
        Returns True when silence timeout is reached → caller should flush.
        """
        self._chunks.append(chunk)
        arr = np.frombuffer(chunk, dtype=np.int16)
        rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))

        if rms < self.silence_threshold:
            if self._silent_since is None:
                self._silent_since = time.monotonic()
            elif time.monotonic() - self._silent_since >= self.silence_timeout_s:
                return True     # silence timeout → flush to Whisper
        else:
            self._silent_since = None   # reset on speech activity

        return False

    def to_numpy(self) -> np.ndarray:
        """Concatenate all collected chunks into a single int16 array."""
        if not self._chunks:
            return np.zeros(0, dtype=np.int16)
        return np.frombuffer(b"".join(self._chunks), dtype=np.int16)


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    from audio_stream import AudioStream

    print(f"Model  : {WHISPER_MODELS[WHISPER_DEFAULT_MODEL]}")
    print(f"Fallback download: {WHISPER_FALLBACK_DOWNLOAD}\n")

    engine = WhisperEngine()        # default = base.pt
    engine.start(wait_for_ready=True)

    print("Recording 5 seconds of audio …")
    buf = DictationBuffer()
    with AudioStream() as stream:
        count = 0
        target = 5 * (AUDIO["sample_rate"] // AUDIO["chunk_size"])
        while count < target:
            chunk = stream.chunk_queue.get()
            if chunk is None:
                break
            buf.feed(chunk)
            count += 1

    audio_np = buf.to_numpy()
    print(f"Transcribing {len(audio_np) / AUDIO['sample_rate']:.1f}s …")
    result = engine.transcribe(audio_np)
    print("Result:", result)
    engine.stop()
