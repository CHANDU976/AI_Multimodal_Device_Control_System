"""
speech_manager.py
Main public API for the Speech Module.

Orchestrates:
  - AudioStream      (microphone capture)
  - VoskEngine       (real-time, low-latency recognition — thread)
  - Classifier       (command ↔ dictation decision)
  - WhisperEngine    (accurate transcription — separate process)
  - DictationBuffer  (accumulates audio for Whisper during dictation)

Threading model
---------------
  Main thread          : starts/stops everything; consumer can call .get_result()
  VoskEngine thread    : reads audio_queue → emits VoskResults → calls _on_vosk_result
  _dictation_thread    : drains audio_queue into DictationBuffer,
                         sends to Whisper, posts WhisperResult to output_queue
  WhisperWorker process: long-lived; transcribes on demand

Usage
-----
    from speech_manager import SpeechManager

    def handle(event):
        print(event)

    manager = SpeechManager(callback=handle)
    manager.start()
    # … runs until stopped …
    manager.stop()

Or pull-based:
    manager.start()
    while True:
        event = manager.get_result(timeout=1.0)
        if event:
            handle(event)
"""

from __future__ import annotations
import path_setup  # noqa: F401 — adds project root to sys.path

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from audio_stream import AudioStream
from vosk_engine import VoskEngine, VoskResult
from whisper_engine import WhisperEngine, DictationBuffer, WhisperResult
from classifier import Classifier, ClassificationResult
from config import AUDIO, MODE, EVENT_TYPE, CONCURRENCY, WHISPER as WHISPER_CFG

logger = logging.getLogger(__name__)


# ── Output event ───────────────────────────────────────────────────────────────

@dataclass
class SpeechEvent:
    """
    Unified output event from the SpeechManager.

    Attributes
    ----------
    event_type : str
        One of EVENT_TYPE values: 'partial', 'command', 'dictation', 'error'
    text : str
        Recognised or transcribed text.
    source : str
        'vosk' or 'whisper' — which engine produced this result.
    mode : str
        Current mode at the time of emission.
    metadata : dict
        Extra info (VOSK raw JSON, Whisper segments, etc.)
    """
    event_type: str
    text: str
    source: str
    mode: str
    metadata: dict

    def __repr__(self) -> str:
        return (
            f"SpeechEvent({self.event_type!r} | {self.source!r} | "
            f"mode={self.mode!r} | text={self.text!r})"
        )


# ── Manager ────────────────────────────────────────────────────────────────────

class SpeechManager:
    """
    High-level API. Initialise once, call start(), consume events.

    Parameters
    ----------
    callback : callable, optional
        Called on a background thread with each SpeechEvent.
        If None, use get_result() instead.
    vosk_model_path : str, optional
    whisper_model_name : str, optional
    """

    def __init__(
        self,
        callback: Callable[[SpeechEvent], None] | None = None,
        vosk_model_path: str | None = None,
        whisper_model_name: str | None = None,
    ) -> None:
        self._callback = callback
        self._vosk_model_path = vosk_model_path
        self._whisper_model_name = whisper_model_name

        # Shared queues
        self._audio_queue: queue.Queue[bytes | None] = queue.Queue(
            maxsize=CONCURRENCY["vosk_queue_maxsize"]
        )
        self._output_queue: queue.Queue[SpeechEvent] = queue.Queue(
            maxsize=CONCURRENCY["result_queue_maxsize"]
        )

        # Components
        self._audio_stream: AudioStream | None = None
        self._vosk_engine: VoskEngine | None = None
        self._whisper_engine: WhisperEngine | None = None
        self._classifier = Classifier()
        self._dictation_buffer = DictationBuffer()

        # Dictation state
        self._in_dictation = threading.Event()
        self._dictation_audio_q: queue.Queue[bytes | None] = queue.Queue()
        self._dictation_thread: threading.Thread | None = None

        # Running flag
        self._running = threading.Event()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Load models and start all engines."""
        logger.info("SpeechManager starting …")

        # ── Audio stream ───────────────────────────────────────────────────────
        self._audio_stream = AudioStream(
            **{k: AUDIO[k] for k in ("sample_rate", "channels", "chunk_size", "device_index")}
        )
        # Attach a second callback that tees audio to the dictation queue
        self._audio_stream.chunk_queue = self._make_tee_queue()

        # ── VOSK ───────────────────────────────────────────────────────────────
        kwargs = {"audio_queue": self._audio_queue, "result_callback": self._on_vosk_result}
        if self._vosk_model_path:
            kwargs["model_path"] = self._vosk_model_path
        self._vosk_engine = VoskEngine(**kwargs)
        self._vosk_engine.load_model()

        # ── Whisper ────────────────────────────────────────────────────────────
        wkwargs = {}
        if self._whisper_model_name:
            wkwargs["model_name"] = self._whisper_model_name
        self._whisper_engine = WhisperEngine(**wkwargs)
        self._whisper_engine.start()

        # ── Dictation thread ───────────────────────────────────────────────────
        self._dictation_thread = threading.Thread(
            target=self._dictation_loop,
            name="DictationThread",
            daemon=True,
        )

        # ── Go ─────────────────────────────────────────────────────────────────
        self._running.set()
        self._audio_stream.start()
        self._vosk_engine.start()
        self._dictation_thread.start()

        logger.info("SpeechManager ready.")

    def stop(self) -> None:
        """Stop all engines gracefully."""
        logger.info("SpeechManager stopping …")
        self._running.clear()

        if self._vosk_engine:
            self._vosk_engine.stop()
        if self._audio_stream:
            self._audio_stream.stop()
        if self._whisper_engine:
            self._whisper_engine.stop()

        # Unblock dictation thread
        self._dictation_audio_q.put(None)
        if self._dictation_thread:
            self._dictation_thread.join(timeout=5)

        logger.info("SpeechManager stopped.")

    def get_result(self, timeout: float = 0.1) -> SpeechEvent | None:
        """
        Pull the next event from the output queue (non-callback usage).
        Returns None on timeout.
        """
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def current_mode(self) -> str:
        return self._classifier.mode

    # ── VOSK callback (runs on VoskEngine thread) ──────────────────────────────

    def _on_vosk_result(self, vosk_result: VoskResult) -> None:
        if not vosk_result.text:
            return

        if not vosk_result.is_final:
            partial_text = vosk_result.text
            if self._classifier.mode == MODE["command"] and self._classifier.matches_dictation_trigger(partial_text):
                logger.info("Trigger detected on partial: %r", partial_text)
                self._classifier.force_dictation_mode()
                self._start_dictation_collection()
                return
            if self._classifier.mode == MODE["dictation"] and self._classifier.matches_dictation_stop(partial_text):
                logger.info("Stop phrase detected on partial: %r", partial_text)
                self._classifier.force_command_mode()
                self._stop_dictation_collection()
                return
            # Emit partial directly — no classification needed
            self._emit(SpeechEvent(
                event_type=EVENT_TYPE["partial"],
                text=vosk_result.text,
                source="vosk",
                mode=self._classifier.mode,
                metadata={"raw": vosk_result.raw},
            ))
            return

        # Final result — classify it
        clf_result: ClassificationResult = self._classifier.classify(vosk_result.text)

        if clf_result.event == EVENT_TYPE["dictation"] and clf_result.mode_changed:
            # Switched into dictation — start collecting audio for Whisper
            self._start_dictation_collection()

        elif clf_result.event == EVENT_TYPE["command"] and clf_result.mode_changed:
            # Switched back to command — signal dictation loop to flush
            self._stop_dictation_collection()

        if clf_result.event == EVENT_TYPE["command"] and clf_result.text:
            self._emit(SpeechEvent(
                event_type=EVENT_TYPE["command"],
                text=clf_result.text,
                source="vosk",
                mode=self._classifier.mode,
                metadata={"raw": vosk_result.raw},
            ))

    # ── Dictation control ──────────────────────────────────────────────────────

    def _start_dictation_collection(self) -> None:
        """Signal the dictation loop to start accumulating audio."""
        logger.debug("Dictation collection started.")
        self._dictation_buffer.reset()
        self._in_dictation.set()

    def _stop_dictation_collection(self) -> None:
        """Signal the dictation loop to flush its buffer to Whisper."""
        logger.debug("Dictation collection stopping — flushing to Whisper.")
        self._in_dictation.clear()
        self._dictation_audio_q.put(None)   # flush sentinel

    # ── Dictation loop (own thread) ────────────────────────────────────────────

    def _dictation_loop(self) -> None:
        """
        Waits for dictation to be active, accumulates audio from the
        tee queue, sends to Whisper, emits results.
        """
        logger.debug("DictationThread running.")
        while self._running.is_set():
            # Wait until dictation mode is active
            if not self._in_dictation.wait(timeout=0.2):
                continue

            # Drain the dictation audio queue
            while self._running.is_set():
                try:
                    chunk = self._dictation_audio_q.get(timeout=0.3)
                except queue.Empty:
                    # Check silence timeout
                    # (DictationBuffer.feed already tracks this)
                    continue

                if chunk is None:
                    # Flush sentinel received — send to Whisper now
                    break

                done = self._dictation_buffer.feed(chunk)
                if done:
                    logger.debug("Silence timeout — auto-flushing to Whisper.")
                    break

            # Transcribe whatever we collected
            audio_np = self._dictation_buffer.to_numpy()
            if len(audio_np) > 0 and self._whisper_engine:
                logger.info("Sending %d samples to Whisper …", len(audio_np))
                result: WhisperResult = self._whisper_engine.transcribe(audio_np)
                if result.success and result.text:
                    self._emit(SpeechEvent(
                        event_type=EVENT_TYPE["dictation"],
                        text=result.text,
                        source="whisper",
                        mode=self._classifier.mode,
                        metadata={
                            "language": result.language,
                            "duration_s": result.duration_s,
                            "segments": result.segments,
                        },
                    ))
                elif not result.success:
                    self._emit(SpeechEvent(
                        event_type=EVENT_TYPE["error"],
                        text="",
                        source="whisper",
                        mode=self._classifier.mode,
                        metadata={"error": result.error},
                    ))
            self._dictation_buffer.reset()

        logger.debug("DictationThread exited.")

    # ── Audio tee ──────────────────────────────────────────────────────────────

    def _make_tee_queue(self) -> queue.Queue:
        """
        Return a proxy queue whose put() forwards chunks to both
        the VOSK audio_queue and the dictation audio queue.
        """
        vosk_q = self._audio_queue
        dictation_q = self._dictation_audio_q

        class TeeQueue(queue.Queue):
            def put_nowait(self, item):  # type: ignore[override]
                # Forward to VOSK
                try:
                    vosk_q.put_nowait(item)
                except queue.Full:
                    try:
                        vosk_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        vosk_q.put_nowait(item)
                    except queue.Full:
                        pass
                # Forward to dictation buffer if in dictation mode
                if item is not None:
                    try:
                        dictation_q.put_nowait(item)
                    except queue.Full:
                        try:
                            dictation_q.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            dictation_q.put_nowait(item)
                        except queue.Full:
                            pass

        return TeeQueue(maxsize=CONCURRENCY["vosk_queue_maxsize"])

    # ── Output ─────────────────────────────────────────────────────────────────

    def _emit(self, event: SpeechEvent) -> None:
        """Put event in output queue and/or call the user callback."""
        try:
            self._output_queue.put_nowait(event)
        except queue.Full:
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._output_queue.put_nowait(event)
            except queue.Full:
                pass
            #logger.warning("Output queue full old event dropped")
        if self._callback:
            try:
                self._callback(event)
            except Exception as exc:
                logger.exception("User callback raised: %s", exc)

    # ── Context-manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "SpeechManager":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    def on_event(event: SpeechEvent) -> None:
        symbols = {
            EVENT_TYPE["partial"]: "?",
            EVENT_TYPE["command"]: ">_-",
            EVENT_TYPE["dictation"]: "...",
            EVENT_TYPE["error"]: "✗",
        }
        sym = symbols.get(event.event_type, "?")
        print(f"  {sym}  [{event.source}]  {event.text}")

    print("Speech Module — say something. Ctrl+C to exit.\n")
    print("  >_- = command   ... = dictation   ? = partial\n")

    with SpeechManager(callback=on_event) as manager:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nExiting …")
