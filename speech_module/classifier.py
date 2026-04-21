"""
classifier.py
Classifies VOSK-recognised text as either a voice COMMAND or a DICTATION
trigger, and manages the current mode state.

Person B — Dictation + Intelligence Layer
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from config import (
    DICTATION_TRIGGERS,
    DICTATION_STOP_WORDS,
    COMMAND_PREFIXES,
    MODE,
    EVENT_TYPE,
)

logger = logging.getLogger(__name__)


# ── Classification result ──────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """What the classifier decided about a piece of recognised text."""
    event: str              # EVENT_TYPE["command"] | EVENT_TYPE["dictation"]
    text: str               # cleaned text after stripping trigger words
    raw_text: str           # original VOSK text (for debugging)
    mode_changed: bool      # True if this result caused a mode switch
    new_mode: str           # MODE["command"] | MODE["dictation"]

    def is_command(self) -> bool:
        return self.event == EVENT_TYPE["command"]

    def is_dictation_start(self) -> bool:
        return self.event == EVENT_TYPE["dictation"] and self.mode_changed

    def __repr__(self) -> str:
        return (
            f"ClassificationResult("
            f"event={self.event!r}, "
            f"text={self.text!r}, "
            f"mode={'→' + self.new_mode if self.mode_changed else self.new_mode!r}"
            f")"
        )


# ── Classifier ────────────────────────────────────────────────────────────────

class Classifier:
    """
    Stateful text classifier that tracks whether the system is in
    command mode or dictation mode.

    State machine
    -------------
    COMMAND mode
      ├─ receives dictation trigger word  → switch to DICTATION mode
      └─ receives any other text          → classify as COMMAND

    DICTATION mode
      ├─ receives stop word               → switch back to COMMAND mode
      └─ receives any other text          → classify as DICTATION
    """

    def __init__(self, initial_mode: str = MODE["command"]) -> None:
        self._mode = initial_mode
        logger.info("Classifier initialised in '%s' mode.", self._mode)

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._mode

    def classify(self, raw_text: str) -> ClassificationResult:
        """
        Classify a piece of VOSK-recognised text.

        Parameters
        ----------
        raw_text : str
            The text string from VoskResult.text.

        Returns
        -------
        ClassificationResult
        """
        text = raw_text.strip().lower()
        mode_changed = False

        # ── In command mode ────────────────────────────────────────────────────
        if self._mode == MODE["command"]:
            if self._matches_any(text, DICTATION_TRIGGERS):
                # Strip the trigger phrase from the remainder
                remainder = self._strip_trigger(text, DICTATION_TRIGGERS)
                self._mode = MODE["dictation"]
                mode_changed = True
                logger.info("Mode → DICTATION (trigger: %r)", text)
                return ClassificationResult(
                    event=EVENT_TYPE["dictation"],
                    text=remainder,
                    raw_text=raw_text,
                    mode_changed=True,
                    new_mode=self._mode,
                )
            else:
                return ClassificationResult(
                    event=EVENT_TYPE["command"],
                    text=text,
                    raw_text=raw_text,
                    mode_changed=False,
                    new_mode=self._mode,
                )

        # ── In dictation mode ──────────────────────────────────────────────────
        else:
            if self._matches_any(text, DICTATION_STOP_WORDS):
                self._mode = MODE["command"]
                mode_changed = True
                logger.info("Mode → COMMAND (stop word: %r)", text)
                return ClassificationResult(
                    event=EVENT_TYPE["command"],
                    text="",
                    raw_text=raw_text,
                    mode_changed=True,
                    new_mode=self._mode,
                )
            else:
                return ClassificationResult(
                    event=EVENT_TYPE["dictation"],
                    text=raw_text.strip(),  # preserve original capitalisation
                    raw_text=raw_text,
                    mode_changed=False,
                    new_mode=self._mode,
                )

    def reset(self) -> None:
        """Force a reset back to command mode."""
        self._mode = MODE["command"]
        logger.info("Classifier reset to COMMAND mode.")

    def force_dictation_mode(self) -> None:
        """Directly set dictation mode."""
        self._mode = MODE["dictation"]

    def force_command_mode(self) -> None:
        """Directly set command mode."""
        self._mode = MODE["command"]

    def matches_dictation_trigger(self, text: str) -> bool:
        """Check whether text starts with a configured dictation trigger."""
        return self._matches_any(text, DICTATION_TRIGGERS)

    def matches_dictation_stop(self, text: str) -> bool:
        """Check whether text starts with a configured dictation stop phrase."""
        return self._matches_any(text, DICTATION_STOP_WORDS)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _matches_any(text: str, phrases: set[str]) -> bool:
        """True if `text` starts with or equals any phrase in the set."""
        normalized_text = Classifier._normalize_text(text)
        for phrase in phrases:
            normalized_phrase = Classifier._normalize_text(phrase)
            if (
                normalized_text == normalized_phrase
                or normalized_text.startswith(normalized_phrase + " ")
            ):
                return True
        return False

    @staticmethod
    def _strip_trigger(text: str, triggers: set[str]) -> str:
        """Remove the matched trigger phrase from the front of `text`."""
        normalized_text = Classifier._normalize_text(text)
        for phrase in triggers:
            normalized_phrase = Classifier._normalize_text(phrase)
            if normalized_text.startswith(normalized_phrase):
                return normalized_text[len(normalized_phrase):].strip()
        return text

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Lowercase and remove punctuation so trigger matching is tolerant."""
        lower = text.strip().lower()
        cleaned = re.sub(r"[^a-z0-9\s]", " ", lower)
        return " ".join(cleaned.split())

    @staticmethod
    def looks_like_command(text: str) -> bool:
        """
        Heuristic check (independent of mode) — useful for confidence
        scoring. Returns True if the text starts with a known command prefix.
        """
        lower = text.strip().lower()
        for prefix in COMMAND_PREFIXES:
            if lower == prefix or lower.startswith(prefix + " "):
                return True
        return False


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    clf = Classifier()
    test_phrases = [
        "open chrome",
        "start dictation",
        "hello this is some dictated text",
        "new paragraph",
        "stop dictation",
        "save file",
    ]
    for phrase in test_phrases:
        result = clf.classify(phrase)
        print(result)
