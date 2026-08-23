"""
adaptation.py
Learns from usage so the system gets faster and more personal over time.

What is learned, and how it feeds back in:

  * Alias promotion — a phrase the engine only fuzzy-matched ("opun chrome")
    that then *executed successfully* several times becomes an alias: an
    exact-lookup entry the IntentEngine checks first. The user's sloppy-but-
    consistent phrasing turns into a first-class command. Aliases that start
    failing (app uninstalled, meaning drifted) are demoted automatically.

  * Intent & app usage counts — which commands and apps the user actually
    uses. Surfaced in stats, and available to future UIs (suggestions,
    most-used ordering).

  * Unknown phrases — everything the system could not act on, counted. The
    top entries are exactly what the user most wants mapped as custom
    commands; suggestions() hands them to a future settings UI.

Persistence: usage.json, throttled (every ADAPTATION save_every records) and
flushed on shutdown, atomic like the preference store.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

from central_module.config import ADAPTATION, DATA_DIR, USAGE_FILE

logger = logging.getLogger(__name__)


class BehaviorModel:

    def __init__(self, path: str = USAGE_FILE) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._dirty = 0
        self._data: dict = {
            "intent_counts": {},    # intent name → executions
            "app_counts": {},       # spoken app name → launches
            "phrase_stats": {},     # phrase → {intent, slots, hits, fails, source}
            "aliases": {},          # phrase → {intent, slots}   (promoted)
            "unknown": {},          # phrase → count
        }
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as fh:
                stored = json.load(fh)
            if isinstance(stored, dict):
                for key in self._data:
                    if isinstance(stored.get(key), dict):
                        self._data[key] = stored[key]
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("usage data unreadable (%s) — starting fresh", exc)

    def save(self, force: bool = True) -> None:
        with self._lock:
            if not force and self._dirty < ADAPTATION["save_every"]:
                return
            os.makedirs(os.path.dirname(self._path) or DATA_DIR, exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, self._path)
            self._dirty = 0

    def _touch(self) -> None:
        self._dirty += 1
        self.save(force=False)

    # ── Recording ──────────────────────────────────────────────────────────────

    def record_intent(self, phrase: str, intent_name: str, slots: dict,
                      source: str, ok: bool) -> None:
        """Called after every executed (or failed) intent."""
        with self._lock:
            counts = self._data["intent_counts"]
            if ok:
                counts[intent_name] = counts.get(intent_name, 0) + 1

            stats = self._data["phrase_stats"].setdefault(phrase, {
                "intent": intent_name, "slots": slots, "source": source,
                "hits": 0, "fails": 0, "last": 0,
            })
            stats["last"] = int(time.time())
            if ok:
                stats["hits"] += 1
            else:
                stats["fails"] += 1

            self._maybe_promote(phrase, stats)
            self._maybe_demote(phrase, stats)
        self._touch()

    def _maybe_promote(self, phrase: str, stats: dict) -> None:
        """Reliable fuzzy/model phrase → exact alias (the adaptation payoff).
        For model matches this pins the user's habitual phrasings so they
        bypass inference entirely — personalization on top of the model."""
        if (stats["source"] in ("fuzzy", "model")
                and stats["hits"] >= ADAPTATION["alias_promote_hits"]
                and stats["hits"] > stats["fails"] * 2
                and phrase not in self._data["aliases"]):
            self._data["aliases"][phrase] = {
                "intent": stats["intent"], "slots": stats["slots"],
            }
            logger.info("adaptation: promoted %r → %s %s",
                        phrase, stats["intent"], stats["slots"])

    def _maybe_demote(self, phrase: str, stats: dict) -> None:
        if phrase in self._data["aliases"] and stats["fails"] > stats["hits"]:
            del self._data["aliases"][phrase]
            logger.info("adaptation: demoted alias %r (keeps failing)", phrase)

    def record_unknown(self, phrase: str) -> None:
        with self._lock:
            unknown = self._data["unknown"]
            unknown[phrase] = unknown.get(phrase, 0) + 1
            if len(unknown) > ADAPTATION["max_unknown"]:
                # Evict the rarest so the map stays bounded.
                victim = min(unknown, key=unknown.get)
                del unknown[victim]
        self._touch()

    def record_app(self, spoken: str) -> None:
        with self._lock:
            apps = self._data["app_counts"]
            apps[spoken] = apps.get(spoken, 0) + 1
        self._touch()

    # ── Feedback into the engine ───────────────────────────────────────────────

    def get_alias(self, phrase: str) -> dict | None:
        """IntentEngine's alias_provider — its step-1 exact lookup."""
        with self._lock:
            alias = self._data["aliases"].get(phrase)
            return dict(alias) if alias else None

    # ── Introspection ──────────────────────────────────────────────────────────

    def suggestions(self, n: int = 5) -> list[tuple[str, int]]:
        """Most frequent unrecognised phrases — prime custom-command material."""
        with self._lock:
            items = sorted(self._data["unknown"].items(),
                           key=lambda kv: kv[1], reverse=True)
            return items[:n]

    def summary(self) -> str:
        with self._lock:
            top_intents = sorted(self._data["intent_counts"].items(),
                                 key=lambda kv: kv[1], reverse=True)[:5]
            top_apps = sorted(self._data["app_counts"].items(),
                              key=lambda kv: kv[1], reverse=True)[:5]
            lines = [
                f"aliases learned : {len(self._data['aliases'])}",
                f"top intents     : {', '.join(f'{k}×{v}' for k, v in top_intents) or '—'}",
                f"top apps        : {', '.join(f'{k}×{v}' for k, v in top_apps) or '—'}",
                f"unknown phrases : {', '.join(f'{k!r}×{v}' for k, v in self.suggestions()) or '—'}",
            ]
            return "\n".join(lines)


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        model = BehaviorModel(os.path.join(tmp, "usage.json"))
        # Three successful fuzzy matches → alias appears.
        for _ in range(3):
            model.record_intent("opun chrome", "open_app",
                                {"app": "chrome"}, "fuzzy", ok=True)
        assert model.get_alias("opun chrome") == {
            "intent": "open_app", "slots": {"app": "chrome"}}
        # Failures overtake → alias demoted.
        for _ in range(4):
            model.record_intent("opun chrome", "open_app",
                                {"app": "chrome"}, "fuzzy", ok=False)
        assert model.get_alias("opun chrome") is None
        model.record_unknown("do a barrel roll")
        model.save()
        reloaded = BehaviorModel(os.path.join(tmp, "usage.json"))
        assert reloaded.suggestions()[0][0] == "do a barrel roll"
        print("adaptation smoke-test passed.")
        print(reloaded.summary())
