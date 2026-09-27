"""
router.py
Decides what to do with a VOSK result: act on it as a command, or escalate
the same audio to Whisper.

    VOSK result ──┬── confident and short ──→ COMMAND   (act now, ~0 extra cost)
                  └── long or unsure ───────→ WHISPER   (accurate, ~1-3 s)

This replaces the old explicit mode switch ("start dictation" … "stop
dictation"). Escalation is decided per utterance from evidence VOSK already
provides, so short confident phrases stay on the fast path and anything VOSK
struggled with — or anything too long to be a command — gets the accurate
model.

Confidence source
-----------------
VOSK's per-word `conf` values, in [0, 1]. These exist only when
max_alternatives == 0; with alternatives enabled VOSK instead reports an
unnormalised path score that grows with utterance length and cannot be
thresholded. config.VOSK enforces the right setting.

Both a mean and a minimum are checked: mean catches a generally poor decode,
minimum catches a single mangled word in an otherwise clean sentence — which
for a command is usually the word that matters.

Grammar decodes
---------------
Confidence alone cannot be trusted there: VOSK must answer from the grammar,
and with few competing paths even a wrong answer scores ~1.0. So a grammar
decode must also BE a command — exactly one config.COMMAND_PHRASES entry,
politeness fillers aside, no [unk] anywhere. Everyday decoys, a bare opener
("open [unk]") and chains of loose phrases all go to Whisper. The command
text sent on is the canonical one ("open note pad" → "open notepad").

Person B — Dictation + Intelligence Layer
"""

from __future__ import annotations
import path_setup  # noqa: F401 — adds project root to sys.path

import logging
from dataclasses import dataclass

from classifier import Classifier
from config import ROUTING, EVENT_TYPE, EVERYDAY_PHRASES, FILLER_PHRASES, command_phrases
from vosk_engine import VoskResult

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    """Where an utterance should go, and why."""
    destination: str            # EVENT_TYPE["command"] | EVENT_TYPE["dictation"]
    reason: str                 # human-readable trigger, for logs and metadata
    text: str                   # VOSK text (may be superseded by Whisper)

    @property
    def escalate(self) -> bool:
        return self.destination == EVENT_TYPE["dictation"]

    def __repr__(self) -> str:
        arrow = "→ WHISPER" if self.escalate else "→ COMMAND"
        return f"Decision({arrow}  {self.reason}  text={self.text!r})"


class Router:
    """
    Stateless policy object. Thresholds come from config.ROUTING.

    Usage
    -----
        decision = Router().route(vosk_result)
        if decision.escalate:
            text = whisper.transcribe(audio)
    """

    def __init__(
        self,
        min_word_confidence: float = ROUTING["min_word_confidence"],
        min_mean_confidence: float = ROUTING["min_mean_confidence"],
        max_command_words: int = ROUTING["max_command_words"],
        max_command_duration_s: float = ROUTING["max_command_duration_s"],
        escalate_on_empty: bool = ROUTING["escalate_on_empty"],
    ) -> None:
        self.min_word_confidence = min_word_confidence
        self.min_mean_confidence = min_mean_confidence
        self.max_command_words = max_command_words
        self.max_command_duration_s = max_command_duration_s
        self.escalate_on_empty = escalate_on_empty
        # Explicit override on top of the automatic policy: saying a configured
        # trigger phrase forces Whisper even for a short, confident utterance.
        self._classifier = Classifier()
        # Grammar decodes: spoken command phrase → canonical text, and the
        # politeness that may wrap one (longest first, so "can you" beats "can").
        self._commands = command_phrases()
        self._fillers = sorted((tuple(f.split()) for f in FILLER_PHRASES),
                               key=len, reverse=True)

    def route(self, result: VoskResult) -> Decision:
        text = result.text.strip()

        if text and self._classifier.matches_dictation_trigger(text):
            return Decision(EVENT_TYPE["dictation"], "explicit dictation trigger", text)

        # ── Grammar decode: only a complete command phrase may run ─────────────
        # Any [unk] means part of the utterance was outside the grammar — an
        # unknown app after "open", or not a command at all.
        if result.constrained:
            if result.unk_count:
                return Decision(
                    EVENT_TYPE["dictation"],
                    f"{result.unk_count}/{len(result.words)} words out of vocabulary",
                    text,
                )
            if text:
                core = self._strip_fillers(text)
                command = self._commands.get(core)
                if command is None:
                    reason = ("everyday speech" if not core or core in EVERYDAY_PHRASES
                              else "not a single command phrase")
                    return Decision(EVENT_TYPE["dictation"], reason, text)
                text = command

        # ── Nothing recognised ─────────────────────────────────────────────────
        # The VAD found speech but VOSK produced no words, so something was
        # said that this model could not handle. Whisper is the better bet.
        if not text:
            if self.escalate_on_empty:
                return Decision(EVENT_TYPE["dictation"], "vosk returned nothing", "")
            return Decision(EVENT_TYPE["command"], "empty, escalation disabled", "")

        # ── Too long to be a command ───────────────────────────────────────────
        # Counted on the command itself — fillers already stripped above.
        word_count = len(text.split())
        if word_count > self.max_command_words:
            return Decision(
                EVENT_TYPE["dictation"],
                f"{word_count} words > {self.max_command_words}",
                text,
            )

        if result.duration_s > self.max_command_duration_s:
            return Decision(
                EVENT_TYPE["dictation"],
                f"{result.duration_s:.1f}s > {self.max_command_duration_s}s",
                text,
            )

        # ── Unsure ─────────────────────────────────────────────────────────────
        if result.min_confidence < self.min_word_confidence:
            return Decision(
                EVENT_TYPE["dictation"],
                f"weakest word {result.min_confidence:.2f} < {self.min_word_confidence}",
                text,
            )

        if result.mean_confidence < self.min_mean_confidence:
            return Decision(
                EVENT_TYPE["dictation"],
                f"mean conf {result.mean_confidence:.2f} < {self.min_mean_confidence}",
                text,
            )

        # ── Short and confident → act on it ────────────────────────────────────
        return Decision(
            EVENT_TYPE["command"],
            f"conf {result.mean_confidence:.2f}, {word_count}w",
            text,
        )

    def _strip_fillers(self, text: str) -> str:
        """
        The command inside the politeness: "can you open chrome please" →
        "open chrome". Stops as soon as what is left is a command phrase, and
        returns "" when the utterance was nothing but fillers ("thank you").
        """
        words = text.split()
        while words and " ".join(words) not in self._commands:
            for filler in self._fillers:
                n = len(filler)
                if tuple(words[:n]) == filler:
                    words = words[n:]
                    break
                if tuple(words[-n:]) == filler:
                    words = words[:-n]
                    break
            else:
                break                               # no filler at either edge
        return " ".join(words)


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    def make(text, mean, lo, dur):
        return VoskResult(event=EVENT_TYPE["command"], text=text, is_final=True,
                          mean_confidence=mean, min_confidence=lo, duration_s=dur)

    router = Router()
    cases = [
        ("open chrome",                       0.99, 0.98, 1.2),
        ("scroll down",                       0.95, 0.91, 0.9),
        ("open chrome",                       0.90, 0.42, 1.2),   # one bad word
        ("open chrome",                       0.60, 0.55, 1.2),   # poor overall
        ("the meeting has been moved to next tuesday afternoon at three",
                                              0.97, 0.95, 5.5),   # too long
        ("",                                  0.00, 0.00, 2.0),   # nothing heard
    ]
    for text, mean, lo, dur in cases:
        print(f"  {router.route(make(text, mean, lo, dur))}")
