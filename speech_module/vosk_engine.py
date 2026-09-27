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
import os
from dataclasses import dataclass, field
from typing import Callable

import vosk
import numpy as np

from audio_utils import StreamResampler
from config import AUDIO, VOSK as VOSK_CFG, VOSK_MODELS, VOSK_DEFAULT_LANG, EVENT_TYPE

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class VoskResult:
    event: str          # EVENT_TYPE["partial"] or EVENT_TYPE["command"]
    text: str
    is_final: bool
    raw: dict = field(default_factory=dict)

    # Populated for final results only. Confidences are VOSK's per-word 0..1
    # values, which exist only when max_alternatives == 0 — see config.VOSK.
    words: list[dict] = field(default_factory=list)
    mean_confidence: float = 0.0
    min_confidence: float = 0.0
    duration_s: float = 0.0

    # Grammar decodes only: how much of the utterance fell outside the command
    # vocabulary. Free decodes never produce [unk], so these stay 0 there.
    unk_count: int = 0
    unk_ratio: float = 0.0

    # Which VOSK model produced this result (e.g. "vosk-model-small-en-in-0.4").
    model: str = ""

    # True when the decode was constrained to config.command_grammar(), so the
    # text can only be phrases from it — the router's cue to apply its
    # phrase rules. False for free decodes, including a model that rejected
    # the grammar.
    constrained: bool = False

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def __repr__(self) -> str:
        tag = "FINAL" if self.is_final else "partial"
        if not self.is_final:
            return f"VoskResult({tag}: {self.text!r})"
        return (f"VoskResult(FINAL: {self.text!r}  "
                f"conf={self.mean_confidence:.2f}/min {self.min_confidence:.2f}  "
                f"{self.duration_s:.1f}s)")


def parse_final(raw: dict, duration_s: float = 0.0, model: str = "",
                constrained: bool = False) -> VoskResult:
    """
    Build a VoskResult from VOSK's final-result JSON.

    Expects the max_alternatives == 0 schema: {"text": ..., "result": [{word,
    conf, start, end}, ...]}. If "result" is absent (older VOSK, or SetWords
    off) confidences fall back to 1.0 so routing stays permissive rather than
    escalating everything to Whisper.

    [unk] tokens (grammar decodes) are counted in unk_count/unk_ratio but
    excluded from both the text and the confidence aggregates: a breath or a
    keyboard click mapping to one low-confidence [unk] should not put a
    literal "[unk]" in a command or drag min_confidence below threshold —
    whether the OOV share matters is the router's call.
    """
    words = raw.get("result") or []
    real_words = [w for w in words if w.get("word") != "[unk]"]
    unk_count = len(words) - len(real_words)
    if unk_count:
        text = " ".join(w["word"] for w in real_words)
    else:
        text = (raw.get("text") or "").strip()
    confs = [float(w["conf"]) for w in real_words if "conf" in w]
    return VoskResult(
        event=EVENT_TYPE["command"],
        text=text,
        is_final=True,
        raw=raw,
        words=words,
        mean_confidence=float(sum(confs) / len(confs)) if confs else (1.0 if text else 0.0),
        min_confidence=min(confs) if confs else (1.0 if text else 0.0),
        duration_s=duration_s,
        unk_count=unk_count,
        unk_ratio=unk_count / len(words) if words else 0.0,
        model=model,
        constrained=constrained,
    )


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
    input_rate : int
        Rate of the chunks arriving on `audio_queue` (mic capture rate).
    sample_rate : int
        Rate the recogniser runs at. Chunks are resampled from `input_rate`
        to this before AcceptWaveform().
    endpoint_callback : callable, optional
        Called on the recognizer thread, with no arguments, each time the
        streaming recognizer closes an utterance (a VOSK endpoint). In preview
        mode, where finals are dropped, this is the only end-of-utterance
        signal.
    """

    def __init__(
        self,
        audio_queue: queue.Queue[bytes | None],
        result_callback: Callable[[VoskResult], None] | None = None,
        result_queue: queue.Queue | None = None,
        model_path: str = VOSK_MODELS[VOSK_DEFAULT_LANG],
        sample_rate: int = AUDIO["target_rate"],
        input_rate: int = AUDIO["capture_rate"],
        partials_only: bool = False,
        endpoint_callback: Callable[[], None] | None = None,
    ) -> None:
        self.audio_queue = audio_queue
        self.result_callback = result_callback
        self.result_queue = result_queue
        self.endpoint_callback = endpoint_callback
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.input_rate = input_rate
        # Preview mode: emit partials for live feedback and drop finals, which
        # recognize_segment() produces more accurately from denoised audio.
        self.partials_only = partials_only

        # One long-lived resampler for the whole session — see StreamResampler
        # docstring for why this must not be per-chunk.
        self._resampler = StreamResampler(
            in_rate=input_rate, out_rate=sample_rate, channels=AUDIO["channels"]
        )

        self._model: vosk.Model | None = None
        self._recognizer: vosk.KaldiRecognizer | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

        # None = untested. Set on first grammar decode: small VOSK models carry
        # the dynamic graph runtime grammars need, but not every model does.
        self._grammar_ok: bool | None = None

        # Segment recognizer cache. Building a grammar recognizer makes Kaldi
        # re-log every vocabulary word the model lacks, so build once and
        # Reset() between utterances instead of constructing per segment.
        self._seg_lock = threading.Lock()
        self._seg_rec: vosk.KaldiRecognizer | None = None
        self._seg_grammar: list[str] | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the VOSK model (call once before start)."""
        vosk.SetLogLevel(VOSK_CFG["log_level"])
        logger.info("Loading VOSK model from: %s", self.model_path)
        self._model = vosk.Model(self.model_path)
        self._recognizer = self._make_recognizer()   # streaming preview: no grammar
        logger.info("VOSK model loaded.")

    def start(self) -> None:
        """Spin up the recognizer thread."""
        if self._model is None:
            self.load_model()
        self._resampler.reset()
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

    def switch_model(self, model_path: str) -> None:
        """
        Hot-swap to a different VOSK model (language change) without stopping
        the streaming loop. The new model is fully loaded before the swap, so
        recognition keeps running on the old one until the moment of exchange;
        an utterance in flight on the streaming recognizer at that moment is
        simply restarted from silence. Safe because both swapped references are
        single assignments and recognize_segment() builds recognizers fresh.
        """
        if model_path == self.model_path and self._model is not None:
            return
        logger.info("Switching VOSK model to: %s", model_path)
        model = vosk.Model(model_path)
        with self._seg_lock:
            self.model_path = model_path
            self._grammar_ok = None                    # re-probe on the new model
            self._model = model
            self._seg_rec = None                       # rebuilt on next segment
            self._recognizer = self._make_recognizer() # streaming preview: no grammar
        logger.info("VOSK model switched.")

    # ── Segment recognition (the authoritative path) ───────────────────────────

    def _make_recognizer(self, grammar: list[str] | None = None) -> vosk.KaldiRecognizer:
        """
        Build a recognizer, grammar-constrained when one is given and the
        model supports it. The first grammar attempt doubles as the probe:
        models without a dynamic graph raise, we log once and permanently
        fall back to free decoding.
        """
        if grammar and self._grammar_ok is not False:
            try:
                rec = vosk.KaldiRecognizer(
                    self._model, self.sample_rate,
                    json.dumps(self._sayable(grammar), ensure_ascii=False),
                )
                self._grammar_ok = True
            except Exception as exc:
                self._grammar_ok = False
                logger.warning(
                    "Model %s rejected the runtime grammar (%s) — "
                    "command decodes fall back to free recognition.",
                    self.model_path, exc,
                )
                rec = vosk.KaldiRecognizer(self._model, self.sample_rate)
        else:
            rec = vosk.KaldiRecognizer(self._model, self.sample_rate)
        rec.SetMaxAlternatives(VOSK_CFG["max_alternatives"])
        rec.SetWords(True)
        return rec

    def _sayable(self, phrases: list[str]) -> list[str]:
        """
        Drop grammar phrases that use a word this model's lexicon lacks. Left
        in, Kaldi deletes just the missing word — "open notepad" becomes
        "open", a real command the utterance never was. config.SPOKEN_FORMS
        supplies spellings the model can say instead.
        """
        find_word = getattr(self._model, "vosk_model_find_word", None)
        if find_word is None:                  # vosk without the lexicon API
            return phrases
        words = {w for p in phrases for w in p.split()} - {"[unk]"}
        missing = {w for w in words if find_word(w) < 0}
        if not missing:
            return phrases
        logger.info("Grammar: %s lacks %s — phrases using them dropped.",
                    os.path.basename(self.model_path), " ".join(sorted(missing)))
        return [p for p in phrases if not missing.intersection(p.split())]

    def recognize_segment(
        self,
        pcm16: "np.ndarray | bytes",
        grammar: list[str] | None = None,
    ) -> VoskResult:
        """
        Decode one complete utterance and return a final result with
        confidences. Thread-safe with respect to the streaming loop, which
        uses its own recognizer.

        The segment recognizer is cached and Reset() between utterances —
        equivalent to a fresh decoder each time, without re-triggering
        Kaldi's vocabulary warnings on every grammar build.

        Parameters
        ----------
        pcm16 : np.ndarray[int16] | bytes
            Audio already at self.sample_rate (16 kHz) — i.e. denoised and
            downsampled by the caller.
        grammar : list[str], optional
            Phrase list restricting the decode to a command vocabulary (see
            config.command_grammar()). Include "[unk]" so OOV speech shows up
            as [unk] tokens; the router escalates on those.
        """
        if self._model is None:
            self.load_model()

        data = pcm16.tobytes() if hasattr(pcm16, "tobytes") else pcm16
        duration_s = len(data) / 2 / self.sample_rate

        with self._seg_lock:
            if self._seg_rec is None or self._seg_grammar != grammar:
                self._seg_rec = self._make_recognizer(grammar)
                self._seg_grammar = list(grammar) if grammar else None
            else:
                # Reused recognizer: back to a clean decoder state, so the
                # previous utterance cannot bias this one.
                self._seg_rec.Reset()
            self._seg_rec.AcceptWaveform(data)
            raw = json.loads(self._seg_rec.FinalResult())
            constrained = bool(grammar) and self._grammar_ok is True
        result = parse_final(raw, duration_s=duration_s, model=os.path.basename(self.model_path),
                             constrained=constrained)
        logger.debug("Segment recognised: %r", result)
        return result

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

            # Downsample capture_rate → recogniser rate (HLD: "Downsample→16k")
            chunk = self._resampler.process(chunk)
            if not chunk:               # resampler still filling its filter
                continue

            # Feed chunk to recognizer
            if self._recognizer.AcceptWaveform(chunk):
                # Final result for this utterance. In preview mode these are
                # dropped — the authoritative decode happens per VAD segment on
                # denoised audio via recognize_segment().
                if self.partials_only:
                    self._recognizer.Result()      # consume, discard
                else:
                    result = parse_final(json.loads(self._recognizer.Result()),
                                          model=os.path.basename(self.model_path))
                    if result.text:
                        self._emit(result)
                self._notify_endpoint()
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
                            model=os.path.basename(self.model_path),
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

    def _notify_endpoint(self) -> None:
        """Tell endpoint_callback the streaming recognizer closed an utterance."""
        if self.endpoint_callback:
            try:
                self.endpoint_callback()
            except Exception as exc:
                logger.exception("endpoint_callback raised: %s", exc)

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
