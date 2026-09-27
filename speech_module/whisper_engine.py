"""
whisper_engine.py
Runs Whisper (via faster-whisper / CTranslate2) in a separate process to
avoid blocking the VOSK real-time loop with heavy model inference.

Person B — Dictation + Intelligence Layer

Model loading priority
----------------------
1. Load from local CTranslate2 model dir  (e.g. Models/WHISPER/faster-whisper-small)
2. If dir missing and WHISPER_FALLBACK_DOWNLOAD=True → download from Hugging Face
3. If dir missing and WHISPER_FALLBACK_DOWNLOAD=False → raise FileNotFoundError
"""

from __future__ import annotations
import path_setup  # noqa: F401 — adds project root to sys.path

import logging
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass

import numpy as np

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
    # Appended after `error`, never before it: WhisperResult(..., "load failed")
    # must still mean a failed result, not a model name.
    model: str = ""         # e.g. "faster-whisper-small" — which model produced this

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def confidence(self) -> float | None:
        """
        Pseudo-confidence in [0, 1], derived from segments' avg_logprob (mean
        log-probability per decoded token). faster-whisper has no native 0..1
        confidence score the way VOSK does, so this exponentiates the mean
        log-prob into an approximate per-token probability — the same trick
        whisper.cpp's confidence display uses. None when there are no
        segments (empty or failed transcription), so a caller can't mistake
        "nothing to measure" for an actual zero.
        """
        if not self.segments:
            return None
        avg_logprob = sum(s.get("avg_logprob", 0.0) for s in self.segments) / len(self.segments)
        return max(0.0, min(1.0, math.exp(avg_logprob)))

    def __repr__(self) -> str:
        if self.error:
            return f"WhisperResult(ERROR: {self.error})"
        return f"WhisperResult({self.text!r}  [{self.duration_s:.2f}s])"


# ── Model loader (used inside child process) ───────────────────────────────────

def _load_whisper_model(model_path: str, model_key: str, allow_download: bool,
                        device: str, compute_type: str):
    """
    Load faster-whisper preferring the local CTranslate2 model directory.

    Parameters
    ----------
    model_path : str
        Path to the local model dir (e.g. Models/WHISPER/faster-whisper-small),
        holding model.bin + config.json + tokenizer.json + vocabulary.txt.
    model_key : str
        Whisper size name ('base', 'small', ...) used as fallback download key.
    allow_download : bool
        If True and the dir is missing, download from Hugging Face.
    device : str
        'cpu' or 'cuda'.
    compute_type : str
        CTranslate2 quantisation — 'int8' on CPU, 'float16' on GPU.

    Raises
    ------
    FileNotFoundError
        If the dir is missing and allow_download is False.
    """
    from faster_whisper import WhisperModel   # deferred: only the worker pays for it

    if os.path.isdir(model_path):
        logger.info("[WhisperWorker] Loading local model: %s", model_path)
        return WhisperModel(model_path, device=device, compute_type=compute_type)

    # ── Directory not found ────────────────────────────────────────────────────
    if allow_download:
        logger.warning(
            "[WhisperWorker] '%s' not found — downloading '%s' from Hugging Face "
            "(this may take a while).",
            model_path,
            model_key,
        )
        return WhisperModel(
            model_key, device=device, compute_type=compute_type,
            download_root=os.path.dirname(model_path),
        )

    raise FileNotFoundError(
        f"Whisper model not found at '{model_path}'. "
        f"Either place the faster-whisper-{model_key} model dir there or set "
        f"WHISPER_FALLBACK_DOWNLOAD=True in config.py."
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
        model = _load_whisper_model(
            model_path, model_key, allow_download,
            device=options.get("device", "cpu"),
            compute_type=options.get("compute_type", "int8"),
        )
    except Exception as exc:
        result_q.put(WhisperResult(
            text="", language="", segments=[],
            duration_s=0.0, model=os.path.basename(model_path.rstrip("/\\")) or model_key,
            error=f"Model load failed: {exc}"
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

        audio_np, sample_rate, language = item
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
            # transcribe() returns a lazy generator — decoding happens as the
            # segments are consumed, so the list() below is the actual work.
            segments_gen, info = model.transcribe(
                audio_float,
                language=language or options.get("language", "en"),
                task=options.get("task", "transcribe"),
                temperature=options.get("temperature", 0.0),
                beam_size=options.get("beam_size", 5),
                best_of=options.get("best_of", 5),
                condition_on_previous_text=options.get("condition_on_previous_text", True),
            )
            segments = [
                {
                    "id": s.id, "start": s.start, "end": s.end, "text": s.text,
                    "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                }
                for s in segments_gen
            ]
            result = WhisperResult(
                text=" ".join(s["text"].strip() for s in segments).strip(),
                language=info.language,
                segments=segments,
                duration_s=time.perf_counter() - t0,
                model=os.path.basename(model_path.rstrip("/\\")) or model_key,
            )
        except Exception as exc:
            result = WhisperResult(
                text="", language="", segments=[],
                duration_s=time.perf_counter() - t0,
                model=os.path.basename(model_path.rstrip("/\\")) or model_key,
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
        Key into WHISPER_MODELS in config.py — 'base' or 'small'.
        Defaults to WHISPER_DEFAULT_MODEL ('small').
    sample_rate : int
        Sample rate of audio passed to transcribe(). 16000 is preferred.
    options : dict, optional
        Override Whisper transcription options from config.WHISPER.

    Examples
    --------
    engine = WhisperEngine()              # uses Models/WHISPER/faster-whisper-small
    engine = WhisperEngine("base")        # uses Models/WHISPER/faster-whisper-base
    """

    def __init__(
        self,
        model_key: str = WHISPER_DEFAULT_MODEL,
        sample_rate: int = AUDIO["target_rate"],
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
                "language", "task", "device", "compute_type",
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
            # Loading the fp16 weights (145 MB base / 484 MB small) and
            # quantising to int8 in a freshly spawned interpreter takes a few
            # seconds, longer on a cold file cache. Say so — an unexplained
            # silent pause here reads as a hang and invites a Ctrl+C.
            logger.info(
                "Waiting for Whisper '%s' to load in the worker process "
                "(a few seconds; up to %.0fs on first run) …",
                self.model_key, ready_timeout,
            )
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
        language: str | None = None,
    ) -> WhisperResult:
        """
        Send a numpy int16 audio array to the worker and block until done.

        Parameters
        ----------
        audio : np.ndarray  (dtype=int16, shape=(N,))
            Raw PCM audio buffer at self.sample_rate Hz.
        timeout : float
            Seconds to wait before raising TimeoutError.
        language : str, optional
            Per-request override of the configured language (e.g. after a
            runtime language switch) — 'en', 'hi', ….
        """
        if self._request_q is None or self._result_q is None:
            raise RuntimeError("WhisperEngine is not started. Call start() first.")

        self._request_q.put((audio, self.sample_rate, language))
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
    numpy array when silence is detected.

    This is the HLD's "Buffer Voiced Audio" box, so it holds audio at the
    *capture* rate (48 kHz) — DFN3 runs on this buffer before anything is
    downsampled for Whisper. The RMS gate below is the frame-level VAD that
    decides when the utterance has ended.
    """

    def __init__(
        self,
        sample_rate: int = AUDIO["capture_rate"],
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
    from audio_utils import resample_int16
    from denoiser import DeepFilterDenoiser

    print(f"Model  : {WHISPER_MODELS[WHISPER_DEFAULT_MODEL]}")
    print(f"Fallback download: {WHISPER_FALLBACK_DOWNLOAD}\n")

    denoiser = DeepFilterDenoiser()
    denoiser.load()

    engine = WhisperEngine()        # default = faster-whisper-small
    engine.start(wait_for_ready=True)

    while True:
        print('press Y to continue anything else to quit..')
        user_choice = input().strip()
        if user_choice.lower() != "y":
            print("Exiting.")
            break
        
        buf = DictationBuffer()
        with AudioStream() as stream:
            print("Recording 7 seconds of audio …")
            count = 0
            target = 7 * (AUDIO["capture_rate"] // AUDIO["chunk_size"])
            while count < target:
                chunk = stream.chunk_queue.get()
                if chunk is None:
                    break
                buf.feed(chunk)
                count += 1
    
        # Same path the manager uses: buffer @48k → DFN3 @48k → downsample → Whisper
        audio_np = buf.to_numpy()
        print(f"Transcribing {len(audio_np) / AUDIO['capture_rate']:.1f}s "
              f"(denoise={'on' if denoiser.available else 'off'}) …")
        audio_np = denoiser.enhance_int16(audio_np)
        audio_np = resample_int16(audio_np, AUDIO["capture_rate"], AUDIO["target_rate"])
        result = engine.transcribe(audio_np)
        print("Result:", result)

    engine.stop()
