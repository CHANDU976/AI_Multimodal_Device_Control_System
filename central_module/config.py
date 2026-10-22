# config.py — Central configuration for the Central Module

import os
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_PKG_DIR)

# User data (preferences, learned behaviour) lives inside the package by
# default so it is easy to inspect during demos; point MMCTRL_DATA elsewhere
# for a real install. The directory is git-ignored.
DATA_DIR = os.environ.get("MMCTRL_DATA", os.path.join(_PKG_DIR, "data"))

PREFERENCES_FILE = os.path.join(DATA_DIR, "preferences.json")
USAGE_FILE = os.path.join(DATA_DIR, "usage.json")

# ── Child processes ────────────────────────────────────────────────────────────
# Each spec is consumed by supervisor.ChildSpec. argv uses the *current*
# interpreter so the children inherit the venv the central module runs in.
SPEECH_CHILD = {
    "enabled": True,
    "argv": [sys.executable, "-u", "-m", "central_module.runners.speech_runner"],
    "cwd": PROJECT_ROOT,
    "grace_s": 10.0,        # SpeechManager.stop() unloads three models
}

# The gesture module is an independent app with its own UI. Point "command"
# at it (e.g. ["C:/path/to/venv/python.exe", "C:/path/to/gesture_app/main.py"])
# via preferences.json → gesture.command, or leave it on the bundled stub.
# If it prints protocol lines on stdout (docs/ipc-protocol.md, e.g. a
# "gesture_fired" event) the central module routes them; any other output is
# just logged.
# Version of the child protocol (docs/ipc-protocol.md) this module reads.
IPC_PROTOCOL_VERSION = 1

GESTURE_CHILD = {
    "enabled": False,
    "argv": [sys.executable, "-u", "-m", "central_module.runners.gesture_runner"],
    "cwd": PROJECT_ROOT,
    "grace_s": 5.0,
}

# Restart policy shared by all children.
SUPERVISOR = {
    "max_restarts": 5,          # consecutive crashes before giving up
    "backoff_base_s": 1.0,      # first restart delay; doubles each crash …
    "backoff_cap_s": 30.0,      # … up to this cap
    "healthy_uptime_s": 60.0,   # alive this long resets the crash counter
    "poll_interval_s": 1.0,
}

# ── NLP / intent matching ──────────────────────────────────────────────────────
NLP = {
    # "neural" = learned aliases → exact patterns → DistilBERT joint model.
    # Falls back to "rules" (aliases → exact → fuzzy) automatically when the
    # trained model or transformers is unavailable.
    "engine": "neural",
    "model_dir": os.path.join(PROJECT_ROOT, "Models", "NLU", "joint-distilbert"),
    # Softmax probability the model's top intent must reach; below this the
    # utterance is refused and recorded, never guessed at.
    "intent_confidence": 0.70,
    # Dev-set intent accuracy a saved model must reach before its refusals
    # are treated as authoritative. Below this it is a sanity artifact and
    # the fuzzy rule fallback stays active behind it.
    "trusted_dev_acc": 0.90,
    # difflib ratio a phrase must reach to fuzzy-match a pattern. Below this
    # the utterance is "unknown" and only recorded for adaptation. 0.75 lets
    # short misheard prefixes through ("opun"→"open" is exactly 0.75) while
    # unrelated short words stay out ("yet"→"yes" is 0.67).
    "fuzzy_threshold": 0.75,
    # Fuzzy matches score ratio * this, so an exact match always outranks one.
    "fuzzy_penalty": 0.9,
}

# ── Actions ────────────────────────────────────────────────────────────────────
ACTIONS = {
    "scroll_steps": 3,          # wheel notches per plain "scroll down"
    "volume_steps": 2,          # volume-key presses per plain "volume up"
    "type_interval_s": 0.012,   # delay between typed characters
    "power_delay_s": 3,         # grace seconds before shutdown/restart fires
}

# Intents that must be spoken twice: the first utterance arms them, a
# following "yes/confirm" fires them, anything else cancels.
CONFIRM_REQUIRED = {"shutdown", "restart"}

CONFIRM_WORDS = {"yes", "confirm", "go ahead", "do it", "sure", "yeah"}
CANCEL_WORDS = {"no", "cancel", "never mind", "abort", "don't"}
PENDING_TIMEOUT_S = 15.0        # armed action expires after this long

# ── Dictation mode ─────────────────────────────────────────────────────────────
# Mirrors speech_module/config.py trigger sets. Central owns the mode: while
# ON, every final utterance is typed instead of parsed.
DICTATION_START = {
    "start dictation", "start dictating", "begin dictation",
    "dictation mode", "start typing mode",
}
DICTATION_STOP = {
    "stop dictation", "end dictation", "stop typing",
    "cancel dictation", "exit dictation",
}

# ── Language switching ─────────────────────────────────────────────────────────
# Spoken language name → speech module's VOSK_MODELS key.
LANGUAGES = {
    "english": "en-in",
    "indian english": "en-in",
    "american english": "en-us",
    "hindi": "hi-in",
}

# ── Adaptation ─────────────────────────────────────────────────────────────────
ADAPTATION = {
    # A fuzzy-matched phrase that executed successfully this many times is
    # promoted to an alias: from then on it matches instantly and exactly.
    "alias_promote_hits": 3,
    # Persist usage.json after this many recorded events (and on shutdown).
    "save_every": 10,
    # Cap on stored unknown phrases (oldest/rarest evicted first).
    "max_unknown": 200,
}
