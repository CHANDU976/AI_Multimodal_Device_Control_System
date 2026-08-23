"""
preferences.py
JSON-backed user preference store.

Explicit settings the user (or a future settings UI) controls: language,
the gesture app's launch command, spoken-name → app mappings, gesture → action
mappings, custom command phrases. Learned data lives next door in
adaptation.py; the split keeps "what the user chose" separate from "what the
system inferred", so wiping learned behaviour never loses configuration.

Writes are atomic (temp file + os.replace) — a crash mid-save can't destroy
the previous file. All access is lock-guarded; callers may be on the event
thread or the executor.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading

from central_module.config import DATA_DIR, PREFERENCES_FILE

logger = logging.getLogger(__name__)

DEFAULTS: dict = {
    "language": "en-in",            # speech module VOSK_MODELS key
    "gesture": {
        "enabled": False,
        "command": [],              # argv of the external gesture app
        "cwd": None,
    },
    # Spoken name → command/path, e.g. "my editor": "code". Consulted before
    # the adapter's built-in table, so the user always wins.
    "app_aliases": {},
    # Gesture name (as reported by the gesture app) → spoken command it
    # should run through the central pipeline, e.g. "fist": "pause".
    "gesture_actions": {},
    # Custom phrase → canonical phrase, e.g. "battle stations": "open code".
    "custom_commands": {},
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class PreferenceStore:

    def __init__(self, path: str = PREFERENCES_FILE) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._data = copy.deepcopy(DEFAULTS)
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as fh:
                stored = json.load(fh)
            if isinstance(stored, dict):
                self._data = _deep_merge(DEFAULTS, stored)
        except FileNotFoundError:
            pass                                    # first run — defaults
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("preferences unreadable (%s) — using defaults", exc)

    def save(self) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(self._path) or DATA_DIR, exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, self._path)

    # ── Generic access (dotted keys: "gesture.command") ────────────────────────

    def get(self, key: str, default=None):
        with self._lock:
            node = self._data
            for part in key.split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return copy.deepcopy(node)

    def set(self, key: str, value, persist: bool = True) -> None:
        with self._lock:
            parts = key.split(".")
            node = self._data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        if persist:
            self.save()

    # ── App aliases ────────────────────────────────────────────────────────────

    def get_app_alias(self, spoken: str) -> str | None:
        return self.get(f"app_aliases.{spoken}")

    def learn_app_alias(self, spoken: str, command: str) -> None:
        """Remember how a spoken app name resolved, so next time is instant."""
        if self.get_app_alias(spoken) != command:
            logger.info("learned app alias: %r → %r", spoken, command)
            self.set(f"app_aliases.{spoken}", command)


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        store = PreferenceStore(os.path.join(tmp, "prefs.json"))
        store.set("language", "hi-in")
        store.learn_app_alias("my editor", "code")
        reloaded = PreferenceStore(os.path.join(tmp, "prefs.json"))
        assert reloaded.get("language") == "hi-in"
        assert reloaded.get_app_alias("my editor") == "code"
        assert reloaded.get("gesture.enabled") is False
        print("preferences smoke-test passed.")
