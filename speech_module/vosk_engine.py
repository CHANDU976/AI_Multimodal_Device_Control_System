"""
vosk_engine.py
Wraps VOSK for low-latency, continuous speech recognition.

Person A — Real-Time Engine + Audio Pipeline

Feeds raw PCM chunks from AudioStream into a VOSK recognizer and
emits partial_text / final_text via a callback or result queue.
"""

from __future__ import annotations
import path_setup  # noqa: F401 — adds project root to sys.path

import json
import queue
import threading
import logging
from dataclasses import dataclass, field
from typing import Callable

import vosk

from config import AUDIO, VOSK as VOSK_CFG, VOSK_MODELS, VOSK_DEFAULT_LANG, EVENT_TYPE

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class VoskResult:
    event: str          # EVENT_TYPE["partial"] or EVENT_TYPE["command"]
    text: str
    is_final: bool
    raw: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        tag = "FINAL" if self.is_final else "partial"
        return f"VoskResult({tag}: {self.text!r})"


# ── Engine ─────────────────────────────────────────────────────────────────────

class VoskEngine:
    """
    Continuous VOSK recognizer that runs in its own thread.

    Parameters
    ----------
    audio_queue : queue.Queue[bytes | None]
        Raw PCM chunk queue from AudioStream. Put None to signal shutdown.
    result_callback : callable, optional
        Called on the recognizer thread with each VoskResult.
        Use result_queue if you need thread-safe delivery instead.
    result_queue : queue.Queue, optional
        Alternative to result_callback — results are put() here.
    model_path : str
        Path to the VOSK model directory.
    """

    def __init__(
        self,
        audio_queue: queue.Queue[bytes | None],
        result_callback: Callable[[VoskResult], None] | None = None,
        result_queue: queue.Queue | None = None,
        model_path: str = VOSK_MODELS[VOSK_DEFAULT_LANG],
        sample_rate: int = AUDIO["sample_rate"],
    ) -> None:
        self.audio_queue = audio_queue
        self.result_callback = result_callback
        self.result_queue = result_queue
        self.model_path = model_path
        self.sample_rate = sample_rate

        self._model: vosk.Model | None = None
        self._recognizer: vosk.KaldiRecognizer | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the VOSK model (call once before start)."""
        vosk.SetLogLevel(VOSK_CFG["log_level"])
        logger.info("Loading VOSK model from: %s", self.model_path)
        self._model = vosk.Model(self.model_path)
        self._recognizer = vosk.KaldiRecognizer(self._model, self.sample_rate)
        self._recognizer.SetMaxAlternatives(VOSK_CFG["max_alternatives"])
        self._recognizer.SetWords(True)
        logger.info("VOSK model loaded.")

    def start(self) -> None:
        """Spin up the recognizer thread."""
        if self._model is None:
            self.load_model()
        self._running.set()
        self._thread = threading.Thread(
            target=self._recognition_loop,
            name="VoskEngine",
            daemon=True,
        )
        self._thread.start()
        logger.info("VoskEngine thread started.")

    def stop(self) -> None:
        """Signal the recognition loop to exit and join the thread."""
        self._running.clear()
        # Unblock the thread if it's waiting on the audio queue
        self.audio_queue.put(None)
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("VoskEngine stopped.")

    def is_running(self) -> bool:
        return self._running.is_set()

    # ── Recognition loop ───────────────────────────────────────────────────────

    def _recognition_loop(self) -> None:
        """Main loop: pull chunks → recognise → emit results."""
        logger.debug("VoskEngine recognition loop running.")
        while self._running.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if chunk is None:           # sentinel → exit
                break

            # Feed chunk to recognizer
            if self._recognizer.AcceptWaveform(chunk):
                # Final result for this utterance
                raw = json.loads(self._recognizer.Result())
                text = raw.get("text", "").strip()
                if text:
                    result = VoskResult(
                        event=EVENT_TYPE["command"],
                        text=text,
                        is_final=True,
                        raw=raw,
                    )
                    self._emit(result)
            else:
                # Partial result (still speaking)
                if VOSK_CFG["partial_results"]:
                    raw = json.loads(self._recognizer.PartialResult())
                    text = raw.get("partial", "").strip()
                    if text:
                        result = VoskResult(
                            event=EVENT_TYPE["partial"],
                            text=text,
                            is_final=False,
                            raw=raw,
                        )
                        self._emit(result)

        logger.debug("VoskEngine recognition loop exited.")

    def _emit(self, result: VoskResult) -> None:
        """Deliver a result to the callback and/or queue."""
        if self.result_callback:
            try:
                self.result_callback(result)
            except Exception as exc:
                logger.exception("result_callback raised: %s", exc)
        if self.result_queue is not None:
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    pass
                logger.warning("VoskEngine result_queue full old result dropped")

    # ── Context-manager support ────────────────────────────────────────────────

    def __enter__(self) -> "VoskEngine":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


# ── Quick smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from audio_stream import AudioStream

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    def on_result(r: VoskResult) -> None:
        tag = ">>>" if r.is_final else "..."
        print(f"{tag} {r.text}")

    audio = AudioStream()
    engine = VoskEngine(audio_queue=audio.chunk_queue, result_callback=on_result)
    engine.load_model()

    print("Speak into the microphone. Press Ctrl+C to stop.")
    audio.start()
    engine.start()
    try:
        threading.Event().wait()    # block until Ctrl+C
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        audio.stop()
