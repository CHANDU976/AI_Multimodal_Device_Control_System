"""
speech_manager.py
Main public API for the Speech Module.

Pipeline
--------
    Mic (48k) → VAD → DFN3 → resample 16k → VOSK ─┬─ confident+short → COMMAND
                                                  └─ long or unsure  → WHISPER

The VAD is Silero (neural, speech-only triggering) with EnergyVAD as the
automatic fallback — see vad.create_vad().

Everything downstream of the VAD works on whole voiced segments rather than a
chunk stream. That is not a compromise — it is what each stage wants:

  * DFN3   has no streaming mode in its Python API (`enhance()` resets the
           model's recurrent state per call and pads each input), so a complete
           utterance is its best case and a chunk stream is its worst.
  * VOSK   decodes a full utterance with per-word confidences, which is what
           the router needs to decide whether to escalate.
  * Whisper reuses the *same* denoised, resampled buffer VOSK just saw, so
           escalation costs one extra inference and no extra preprocessing.

Denoising sits before VOSK because that is where it measurably helps: on this
stack DFN3 cut VOSK's WER by 6-11 points across SNRs, while Whisper — trained
on noisy data — showed no significant gain from it.

A second VOSK recognizer runs on the raw stream purely to emit live partials
for UI feedback. Its final results are discarded; the segment decode above is
authoritative. Disable it with SpeechManager(emit_partials=False).

Threading model
---------------
  Main thread           : start/stop; consumer calls .get_result() or gets callbacks
  AudioStream callback  : PortAudio thread → tees chunks to the VAD and preview queues
  VoskEngine thread     : preview partials only
  _vad_thread           : chunks → VAD → segment queue
  _process_thread       : segment → DFN3 → VOSK → route → maybe Whisper → events
  WhisperWorker process : long-lived; transcribes on demand

Usage
-----
    from speech_manager import SpeechManager

    def handle(event):
        print(event)

    with SpeechManager(callback=handle) as manager:
        ...

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
from dataclasses import dataclass, field
from typing import Callable

from audio_devices import DeviceMonitor
from audio_stream import AudioStream
from audio_utils import resample_int16
from denoiser import DeepFilterDenoiser
from router import Router, Decision
from vad import Segment, create_vad
from vosk_engine import VoskEngine, VoskResult
from whisper_engine import WhisperEngine, WhisperResult
from config import (
    AUDIO, EVENT_TYPE, CONCURRENCY, DEVICE_MONITOR, GRAMMAR, VOSK_MODELS,
    WHISPER_LANG, command_grammar,
)

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
        'vosk' or 'whisper' — which engine produced this text.
    mode : str
        Retained for API compatibility; mirrors event_type now that routing is
        per-utterance rather than a sticky mode.
    metadata : dict
        Confidence, routing reason, timings, denoise flag.
    """
    event_type: str
    text: str
    source: str
    mode: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"SpeechEvent({self.event_type!r} | {self.source!r} | "
                f"text={self.text!r})")


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
    emit_partials : bool
        Run the preview recognizer on raw audio for live partial text.
    """

    def __init__(
        self,
        callback: Callable[[SpeechEvent], None] | None = None,
        vosk_model_path: str | None = None,
        whisper_model_name: str | None = None,
        emit_partials: bool = True,
    ) -> None:
        self._callback = callback
        self._vosk_model_path = vosk_model_path
        self._whisper_model_name = whisper_model_name
        self._emit_partials = emit_partials

        # Queues. All bounded: audio arrives faster than a stalled consumer
        # drains, and dropping the oldest chunk beats unbounded growth.
        self._preview_queue: queue.Queue[bytes | None] = queue.Queue(
            maxsize=CONCURRENCY["vosk_queue_maxsize"]
        )
        self._vad_queue: queue.Queue[bytes | None] = queue.Queue(
            maxsize=CONCURRENCY["vosk_queue_maxsize"]
        )
        self._segment_queue: queue.Queue[Segment | None] = queue.Queue(
            maxsize=CONCURRENCY["segment_queue_maxsize"]
        )
        self._output_queue: queue.Queue[SpeechEvent] = queue.Queue(
            maxsize=CONCURRENCY["result_queue_maxsize"]
        )

        # Components
        self._audio_stream: AudioStream | None = None
        self._vosk_engine: VoskEngine | None = None
        self._whisper_engine: WhisperEngine | None = None
        self._denoiser = DeepFilterDenoiser()
        self._vad = create_vad()    # Silero if its model loads, energy fallback
        self._router = Router()

        # Input-device tracking. While no device is explicitly chosen
        # (device_index None) the capture stream follows the system default;
        # set_input_device() pins a specific device and stops the following.
        self._device_monitor: DeviceMonitor | None = None
        self._follow_default = AUDIO["device_index"] is None
        self._audio_lock = threading.Lock()   # serialises stream restarts vs stop()

        self._vad_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None
        self._running = threading.Event()

        # Command-vocabulary grammar for the authoritative segment decode.
        # None = free decoding (grammar disabled, or a language whose commands
        # the English vocabulary can't cover).
        self._grammar: list[str] | None = (
            command_grammar() if GRAMMAR["enabled"] else None
        )
        # Per-request Whisper language override; None = config default.
        self._whisper_language: str | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Load models and start all engines."""
        logger.info("SpeechManager starting …")

        # ── Audio stream (48 kHz — DFN3's native rate) ─────────────────────────
        self._audio_stream = AudioStream(
            sample_rate=AUDIO["capture_rate"],
            **{k: AUDIO[k] for k in ("channels", "chunk_size", "device_index")}
        )
        self._audio_stream.chunk_queue = self._make_tee_queue()

        # ── VOSK ───────────────────────────────────────────────────────────────
        kwargs = {
            "audio_queue": self._preview_queue,
            "result_callback": self._on_preview_result,
            "partials_only": True,
        }
        if self._vosk_model_path:
            kwargs["model_path"] = self._vosk_model_path
        self._vosk_engine = VoskEngine(**kwargs)
        self._vosk_engine.load_model()

        # ── DFN3 ───────────────────────────────────────────────────────────────
        # Before Whisper so a hard failure (DENOISE["required"]) surfaces
        # without first paying to spawn the Whisper process.
        self._denoiser.load()

        # ── Whisper ────────────────────────────────────────────────────────────
        wkwargs = {}
        if self._whisper_model_name:
            wkwargs["model_key"] = self._whisper_model_name
        self._whisper_engine = WhisperEngine(**wkwargs)
        self._whisper_engine.start()

        # ── Go ─────────────────────────────────────────────────────────────────
        self._running.set()
        self._vad.reset()
        self._vad_thread = threading.Thread(target=self._vad_loop, name="VADThread", daemon=True)
        self._process_thread = threading.Thread(
            target=self._process_loop, name="SegmentThread", daemon=True
        )
        self._vad_thread.start()
        self._process_thread.start()

        self._audio_stream.start()
        if self._emit_partials:
            self._vosk_engine.start()

        if DEVICE_MONITOR["enabled"]:
            self._device_monitor = DeviceMonitor(
                on_default_change=self._on_default_device_change,
                on_list_change=self._on_device_list_change,
            )
            self._device_monitor.start()

        logger.info("SpeechManager ready.")

    def stop(self) -> None:
        """Stop all engines gracefully."""
        logger.info("SpeechManager stopping …")
        self._running.clear()

        # Monitor first, so no device switch can restart the stream mid-stop.
        if self._device_monitor:
            self._device_monitor.stop()

        with self._audio_lock:
            if self._audio_stream:
                self._audio_stream.stop()
        if self._vosk_engine and self._emit_partials:
            self._vosk_engine.stop()

        # Unblock the worker threads, then let them drain.
        self._offer(self._vad_queue, None)
        self._offer(self._segment_queue, None)
        for t in (self._vad_thread, self._process_thread):
            if t:
                t.join(timeout=5)

        if self._whisper_engine:
            self._whisper_engine.stop()
        self._denoiser.close()

        logger.info("SpeechManager stopped.")

    def get_result(self, timeout: float = 0.1) -> SpeechEvent | None:
        """Pull the next event from the output queue. None on timeout."""
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def noise_floor(self) -> float:
        """Current VAD noise-floor estimate (int16 RMS) — useful for tuning."""
        return self._vad.noise_floor

    def set_language(self, lang: str) -> None:
        """
        Switch both recognisers to another language at runtime — the
        `set_language` control operation of the subprocess protocol.

        Hot-swaps the VOSK model between segments and steers Whisper via a
        per-request language override, so neither engine restarts. The English
        command grammar is dropped for non-English languages (its words would
        all be OOV in that model's lexicon) and restored on the way back.

        Parameters
        ----------
        lang : str
            A key of config.VOSK_MODELS — 'en-in', 'en-us', 'hi-in'.
        """
        if lang not in VOSK_MODELS:
            raise ValueError(
                f"Unknown language '{lang}'. Valid: {list(VOSK_MODELS.keys())}"
            )
        logger.info("Switching language to '%s' …", lang)
        if self._vosk_engine is not None:
            self._vosk_engine.switch_model(VOSK_MODELS[lang])
        else:
            self._vosk_model_path = VOSK_MODELS[lang]

        self._whisper_language = WHISPER_LANG.get(lang)
        use_grammar = GRAMMAR["enabled"] and (self._whisper_language or "en") == "en"
        self._grammar = command_grammar() if use_grammar else None
        logger.info("Language switched to '%s' (grammar %s).",
                    lang, "on" if self._grammar else "off")

    # ── Input devices ──────────────────────────────────────────────────────────

    @property
    def input_devices(self) -> list[dict]:
        """
        Latest snapshot of available input devices (refreshed every
        DEVICE_MONITOR poll): [{index, name, hostapi, channels,
        default_samplerate, is_default}, …]. The future device-picker UI
        reads this and calls set_input_device() with an index from it.
        """
        return self._device_monitor.devices if self._device_monitor else []

    def set_input_device(self, device_index: int | None) -> None:
        """
        Capture from a specific input device (index from `input_devices`),
        or pass None to return to following the system default.
        """
        self._follow_default = device_index is None
        with self._audio_lock:
            if self._audio_stream and self._running.is_set():
                self._audio_stream.restart(device_index=device_index)
        logger.info(
            "Input device %s.",
            "following system default" if self._follow_default
            else f"pinned to index {device_index}",
        )

    def _on_default_device_change(self, new_default: dict, devices: list[dict]) -> None:
        """DeviceMonitor thread: system default input changed — follow it."""
        if not self._follow_default:
            logger.info("Default input changed but a device is pinned — ignoring.")
            return
        with self._audio_lock:
            if not (self._audio_stream and self._running.is_set()):
                return
            # device_index stays None: the restart re-initialises PortAudio,
            # so "default" now resolves to the new device.
            self._audio_stream.restart()
        self._emit(SpeechEvent(
            event_type=EVENT_TYPE["device"],
            text=new_default["name"],
            source="audio",
            mode=EVENT_TYPE["device"],
            metadata={"reason": "default_changed", "device": new_default},
        ))

    def _on_device_list_change(self, devices: list[dict]) -> None:
        """DeviceMonitor thread: a device appeared or disappeared."""
        self._emit(SpeechEvent(
            event_type=EVENT_TYPE["device"],
            text="",
            source="audio",
            mode=EVENT_TYPE["device"],
            metadata={"reason": "list_changed", "devices": devices},
        ))

    # ── Preview partials (VoskEngine thread) ───────────────────────────────────

    def _on_preview_result(self, result: VoskResult) -> None:
        if result.is_final or not result.text:
            return
        self._emit(SpeechEvent(
            event_type=EVENT_TYPE["partial"],
            text=result.text,
            source="vosk",
            mode=EVENT_TYPE["partial"],
            metadata={"preview": True},
        ))

    # ── VAD loop ───────────────────────────────────────────────────────────────

    def _vad_loop(self) -> None:
        """Chunks → EnergyVAD → segment queue."""
        logger.debug("VADThread running.")
        while self._running.is_set():
            try:
                chunk = self._vad_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if chunk is None:
                break
            segment = self._vad.feed(chunk)
            if segment is not None:
                logger.info("VAD segment: %r", segment)
                self._offer(self._segment_queue, segment)

        # Close anything still open so a final utterance is not lost on stop().
        tail = self._vad.flush()
        if tail is not None:
            self._offer(self._segment_queue, tail)
        self._offer(self._segment_queue, None)
        logger.debug("VADThread exited.")

    # ── Segment processing ─────────────────────────────────────────────────────

    def _process_loop(self) -> None:
        """Segment → DFN3 → VOSK → route → command event or Whisper."""
        logger.debug("SegmentThread running.")
        while True:
            try:
                segment = self._segment_queue.get(timeout=0.3)
            except queue.Empty:
                if not self._running.is_set():
                    break
                continue
            if segment is None:
                break
            try:
                self._handle_segment(segment)
            except Exception as exc:
                logger.exception("Segment processing failed: %s", exc)
                self._emit(SpeechEvent(
                    event_type=EVENT_TYPE["error"], text="", source="pipeline",
                    mode=EVENT_TYPE["error"], metadata={"error": str(exc)},
                ))
        logger.debug("SegmentThread exited.")

    def _handle_segment(self, segment: Segment) -> None:
        t0 = time.perf_counter()

        # 1. Denoise the whole utterance at 48 kHz (DFN3's native rate).
        clean48 = self._denoiser.enhance_int16(segment.pcm)
        t_denoise = time.perf_counter() - t0

        # 2. Downsample once; both recognisers consume this same buffer.
        clean16 = resample_int16(clean48, AUDIO["capture_rate"], AUDIO["target_rate"])

        # 3. VOSK, with per-word confidences — grammar-constrained when enabled.
        t1 = time.perf_counter()
        vosk_result = self._vosk_engine.recognize_segment(clean16, grammar=self._grammar)
        t_vosk = time.perf_counter() - t1

        # 4. Route.
        decision: Decision = self._router.route(vosk_result)
        logger.info("%r", decision)

        meta = {
            "denoised": self._denoiser.available,
            # Epoch seconds of the segment's first sample. Word start/end
            # offsets below are relative to this, so the central module can
            # align words with gesture events on one timeline.
            "segment_start": round(segment.wall_start, 3),
            "segment_s": round(segment.duration_s, 2),
            "noise_floor": round(segment.noise_floor, 1),
            "vosk_text": vosk_result.text,
            **({"speech_prob": round(segment.speech_prob, 2)}
               if segment.speech_prob else {}),
            "words": vosk_result.words,     # [{word, start, end, conf}, …]
            "mean_confidence": round(vosk_result.mean_confidence, 3),
            "min_confidence": round(vosk_result.min_confidence, 3),
            "routing_reason": decision.reason,
            "denoise_s": round(t_denoise, 3),
            "vosk_s": round(t_vosk, 3),
        }
        if vosk_result.unk_count:
            meta["unk_count"] = vosk_result.unk_count

        if not decision.escalate:
            self._emit(SpeechEvent(
                event_type=EVENT_TYPE["command"], text=decision.text,
                source="vosk", mode=EVENT_TYPE["command"], metadata=meta,
            ))
            return

        # 5. Escalate: same denoised 16 kHz buffer, no extra preprocessing.
        if self._whisper_engine is None:
            return
        t2 = time.perf_counter()
        result: WhisperResult = self._whisper_engine.transcribe(
            clean16, language=self._whisper_language,
        )
        meta["whisper_s"] = round(time.perf_counter() - t2, 3)

        if result.success and result.text:
            meta.update(language=result.language, segments=result.segments)
            self._emit(SpeechEvent(
                event_type=EVENT_TYPE["dictation"], text=result.text,
                source="whisper", mode=EVENT_TYPE["dictation"], metadata=meta,
            ))
        elif not result.success:
            meta["error"] = result.error
            self._emit(SpeechEvent(
                event_type=EVENT_TYPE["error"], text="", source="whisper",
                mode=EVENT_TYPE["error"], metadata=meta,
            ))

    # ── Audio tee ──────────────────────────────────────────────────────────────

    def _make_tee_queue(self) -> queue.Queue:
        """
        Proxy queue whose put_nowait() forwards each chunk to both the VAD and
        the preview recognizer, dropping the oldest item when either backs up.
        """
        targets = [self._vad_queue]
        if self._emit_partials:
            targets.append(self._preview_queue)
        offer = self._offer

        class TeeQueue(queue.Queue):
            def put_nowait(self, item):  # type: ignore[override]
                for target in targets:
                    offer(target, item)

        return TeeQueue(maxsize=CONCURRENCY["vosk_queue_maxsize"])

    # ── Output ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _offer(q: queue.Queue, item) -> None:
        """Non-blocking put with drop-oldest on overflow."""
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass

    def _emit(self, event: SpeechEvent) -> None:
        """Put the event on the output queue and/or call the user callback."""
        self._offer(self._output_queue, event)
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
            EVENT_TYPE["error"]: "x",
        }
        sym = symbols.get(event.event_type, "?")
        if event.event_type == EVENT_TYPE["partial"]:
            print(f"  {sym}  {event.text}")
            return
        conf = event.metadata.get("mean_confidence")
        why = event.metadata.get("routing_reason", "")
        extra = f"   [conf {conf} | {why}]" if conf is not None else ""
        print(f"  {sym}  [{event.source}]  {event.text}{extra}")

    print("Speech Module — say something. Ctrl+C to exit.\n")
    print("  >_- = command   ... = dictation (escalated to Whisper)   ? = partial\n")

    with SpeechManager(callback=on_event) as manager:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nExiting …")
