"""
intent_engine.py
Turns a recognised utterance into an IntentMatch — or None if the system
honestly does not know what was meant (better than guessing at a PC action).

Matching ladder, cheapest first:

    1. alias   — exact lookup in the learned-alias table (adaptation layer).
                 Phrases the user repeats end up here, so the common case is
                 one dict access.
    2. exact   — compiled-regex match against every catalog pattern; the
                 most specific pattern (most literal words) wins. Free
                 precision: a verbatim command never risks a model error.
    3. model   — the fine-tuned DistilBERT joint intent+slot model
                 (nlu/engine.py), which handles paraphrases and typos the
                 patterns never anticipated — and refuses ("none" class /
                 low confidence) rather than guessing.
    4. fuzzy   — difflib similarity, used ONLY when the neural model is not
                 available (not trained yet, or transformers missing), so
                 the system still works out of the box.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable

from central_module.config import NLP
from central_module.nlp.intents import (
    BY_NAME, CATALOG, DIRECTIONS, Intent, KEY_ALIASES,
)
from central_module.nlp.normalize import normalize

logger = logging.getLogger(__name__)

# Slot kind → regex fragment (named group filled in per pattern).
_SLOT_REGEX = {
    "app":       r"(?P<{name}>.+)",
    "text":      r"(?P<{name}>.+)",
    "number":    r"(?P<{name}>\d{{1,3}})",
    "count":     r"(?P<{name}>\d{{1,3}})",
    "direction": r"(?P<{name}>up|down|left|right|top|bottom)",
    "key":       r"(?P<{name}>[a-z0-9 ]+)",
}


def _validate(kind: str, value: str):
    """Canonicalize a captured slot; None = invalid, reject the match."""
    value = value.strip()
    if not value:
        return None
    if kind in ("number", "count"):
        # Exact matches arrive as pure digits (\d{1,3}); the fuzzy tail can
        # be anything ("60 sveven") and must be rejected, not crash int().
        if not value.isdigit():
            return None
        n = int(value)
        return n if 0 <= n <= 999 else None
    if kind == "direction":
        return value if value in DIRECTIONS else None
    if kind == "key":
        return KEY_ALIASES.get(value)
    return value                        # app / text: any non-empty string


@dataclass
class _Pattern:
    intent: Intent
    regex: re.Pattern
    literal_tokens: list[str]           # in order, slots excluded
    slot_kinds: dict                    # captured name → kind
    uses_light: bool                    # match against light form (text payload)
    fixed_text: str | None              # full phrase if the pattern has no slots
    tail_slot: str | None               # slot name if pattern = literals + slot


@dataclass
class IntentMatch:
    intent: Intent
    slots: dict = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "exact"               # exact | fuzzy | alias
    text: str = ""                      # normalized utterance that matched

    def __repr__(self) -> str:
        return (f"IntentMatch({self.intent.name} {self.slots} "
                f"conf={self.confidence:.2f} via {self.source})")


class IntentEngine:
    """
    parse(text) → IntentMatch | None.

    alias_provider : callable(normalized_phrase) → {"intent": ..., "slots": ...}
        Hook for the adaptation layer's learned aliases. Optional.
    """

    def __init__(
        self,
        alias_provider: Callable[[str], dict | None] | None = None,
        fuzzy_threshold: float = NLP["fuzzy_threshold"],
        fuzzy_penalty: float = NLP["fuzzy_penalty"],
    ) -> None:
        self._alias_provider = alias_provider
        self._threshold = fuzzy_threshold
        self._penalty = fuzzy_penalty
        self._patterns: list[_Pattern] = [
            self._compile(intent, p) for intent in CATALOG
            for p in intent.patterns
        ]
        self._neural = None
        if NLP["engine"] == "neural":
            try:
                from central_module.nlp.nlu.engine import NeuralNLU
                self._neural = NeuralNLU()
            except FileNotFoundError as exc:
                logger.warning("%s — using rule engine until then", exc)
            except ImportError as exc:
                logger.warning("neural NLU needs transformers (%s) — "
                               "using rule engine", exc)
            except Exception:
                logger.exception("neural NLU failed to load — using rule engine")

    @property
    def mode(self) -> str:
        return "neural" if self._neural else "rules"

    # ── Compilation ────────────────────────────────────────────────────────────

    @staticmethod
    def _compile(intent: Intent, pattern: str) -> _Pattern:
        parts: list[str] = []
        literals: list[str] = []
        kinds: dict = {}
        tokens = pattern.split()
        for tok in tokens:
            m = re.fullmatch(r"\{(\w+)\}", tok)
            if m:
                name = m.group(1)
                kind = intent.slots[name]
                kinds[name] = kind
                parts.append(_SLOT_REGEX[kind].format(name=name))
            else:
                literals.append(tok)
                parts.append(re.escape(tok))
        regex = re.compile(r"^" + r"\s+".join(parts) + r"$")
        is_tail = bool(kinds) and len(kinds) == 1 and "{" in tokens[-1]
        return _Pattern(
            intent=intent,
            regex=regex,
            literal_tokens=literals,
            slot_kinds=kinds,
            uses_light=("text" in kinds.values()),
            fixed_text=pattern if not kinds else None,
            tail_slot=next(iter(kinds)) if is_tail else None,
        )

    # ── Parsing ────────────────────────────────────────────────────────────────

    def parse(self, text: str) -> IntentMatch | None:
        light, cmd = normalize(text)
        if not cmd:
            return None

        # 1. Learned alias — the adaptation fast path.
        if self._alias_provider:
            alias = self._alias_provider(cmd)
            if alias and alias.get("intent") in BY_NAME:
                return IntentMatch(
                    intent=BY_NAME[alias["intent"]],
                    slots=dict(alias.get("slots", {})),
                    confidence=1.0, source="alias", text=cmd,
                )

        # 2. Exact pattern match, most literal words wins.
        best: IntentMatch | None = None
        best_score = -1
        for pat in self._patterns:
            m = pat.regex.match(light if pat.uses_light else cmd)
            if not m:
                continue
            slots = self._extract(pat, m.groupdict())
            if slots is None:
                continue
            score = len(pat.literal_tokens)
            if score > best_score:
                best_score = score
                best = IntentMatch(pat.intent, slots, 1.0, "exact", cmd)
        if best:
            return best

        # 3. Neural model — paraphrases and typos the patterns never saw.
        if self._neural:
            parsed = self._neural.parse(text)
            if parsed is not None:
                name, slots, conf = parsed
                return IntentMatch(BY_NAME[name], slots, conf, "model", cmd)
            if self._neural.trusted:
                # A properly trained model's refusal is the answer. Fuzzing
                # past it would resurrect exactly the false positives
                # ("delete xhatter" → delete) the model was trained to reject.
                return None

        # 4. Fuzzy fallback — rule engine, or an untrusted model's refusal.
        return self._fuzzy(light, cmd)

    @staticmethod
    def _extract(pat: _Pattern, groups: dict) -> dict | None:
        slots: dict = {}
        for name, kind in pat.slot_kinds.items():
            value = _validate(kind, groups.get(name, ""))
            if value is None:
                return None
            slots[name] = value
        return slots

    def _fuzzy(self, light: str, cmd: str) -> IntentMatch | None:
        best: IntentMatch | None = None
        best_ratio = self._threshold      # must beat the floor to count

        cmd_len = len(cmd.split())
        for pat in self._patterns:
            if pat.fixed_text is not None:
                # A fixed phrase may absorb misspellings, never payload:
                # "increase volume to 67" must not collapse into
                # "increase volume" with the level silently dropped.
                if abs(cmd_len - len(pat.fixed_text.split())) > 1:
                    continue
                if not self._tokens_agree(cmd, pat.fixed_text):
                    continue
                ratio = SequenceMatcher(None, cmd, pat.fixed_text).ratio()
                if ratio >= best_ratio:
                    best_ratio = ratio
                    best = IntentMatch(pat.intent, {}, ratio * self._penalty,
                                       "fuzzy", cmd)
            elif pat.tail_slot is not None and pat.literal_tokens:
                # "opun chrome" → prefix "opun" ≈ "open", tail = slot value.
                # Prefix and tail both come from the form this pattern
                # matches on, so payload slots keep their exact words.
                tokens = (light if pat.uses_light else cmd).split()
                k = len(pat.literal_tokens)
                if len(tokens) <= k:
                    continue
                ratio = SequenceMatcher(
                    None, " ".join(tokens[:k]),
                    " ".join(pat.literal_tokens)).ratio()
                if ratio < best_ratio:
                    continue
                value = _validate(pat.slot_kinds[pat.tail_slot],
                                  " ".join(tokens[k:]))
                if value is None:
                    continue
                best_ratio = ratio
                best = IntentMatch(pat.intent, {pat.tail_slot: value},
                                   ratio * self._penalty, "fuzzy", cmd)
        return best

    def _tokens_agree(self, cmd: str, fixed: str) -> bool:
        """
        Word-level sanity for fixed-phrase fuzzy: a good whole-string ratio
        can hide one entirely wrong word — "delete xhatter" scores 0.88
        against "delete that" yet means something else. When word counts
        line up, every word pair must individually clear the same fuzzy
        threshold the whole phrase does.
        """
        a, b = cmd.split(), fixed.split()
        if len(a) != len(b):
            return True                 # off-by-one merges/splits: let the
                                        # whole-string ratio decide alone
        return all(SequenceMatcher(None, x, y).ratio() >= self._threshold
                   for x, y in zip(a, b))


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = IntentEngine(alias_provider=lambda p: (
        {"intent": "open_app", "slots": {"app": "spotify"}}
        if p == "music time" else None))

    cases = [
        "open chrome",
        "could you open the calculator please",
        "opun chrome",                     # fuzzy prefix
        "volum up",                        # fuzzy fixed
        "set volume to fifty five",
        "scroll down three times",
        "type The report is due on Monday",
        "search for weather in bangalore",
        "press page up",
        "shut down",
        "music time",                      # learned alias
        "what is the meaning of life",     # → None
    ]
    for c in cases:
        print(f"  {c!r:45s} → {engine.parse(c)}")
