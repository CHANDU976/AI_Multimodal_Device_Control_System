# config.py — Central configuration for the Speech Module

import os

# ── Base model directory (relative to project root) ────────────────────────────
_MODELS_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Models"))

# ── VOSK model paths — one per language ────────────────────────────────────────
VOSK_MODELS = {
    "en-in": os.path.join(_MODELS_ROOT, "VOSK", "vosk-model-small-en-in-0.4"),   # ← default
    "en-us": os.path.join(_MODELS_ROOT, "VOSK", "vosk-model-small-en-us-0.15"),
    "hi-in": os.path.join(_MODELS_ROOT, "VOSK", "vosk-model-small-hi-0.22"),
}

# Active VOSK model (change this to switch language)
VOSK_DEFAULT_LANG = "en-in"

# ── Whisper model paths ─────────────────────────────────────────────────────────
WHISPER_MODELS = {
    "base": os.path.join(_MODELS_ROOT, "WHISPER", "base.pt"),   # ← default
    "tiny": os.path.join(_MODELS_ROOT, "WHISPER", "tiny.pt"),
}

# Active Whisper model key
WHISPER_DEFAULT_MODEL = "base"

# Fallback: if the local .pt file is missing, download this model size from OpenAI
WHISPER_FALLBACK_DOWNLOAD = True   # set False to hard-fail instead

# ── Convenience aliases (used by engines) ──────────────────────────────────────
MODEL_PATHS = {
    "vosk":    VOSK_MODELS[VOSK_DEFAULT_LANG],
    "whisper": WHISPER_MODELS[WHISPER_DEFAULT_MODEL],
}

# ── Audio settings ─────────────────────────────────────────────────────────────
AUDIO = {
    "sample_rate": 16000,       # Hz — required by both VOSK and Whisper
    "channels": 1,              # mono
    "chunk_size": 4000,         # frames per read (250 ms at 16 kHz)
    "dtype": "int16",           # raw PCM format
    "device_index": None,       # None = auto-detect system default input device
}

# ── VOSK settings ──────────────────────────────────────────────────────────────
VOSK = {
    "log_level": -1,            # -1 = silent, 0 = errors, 1 = verbose
    "partial_results": True,    # emit partial text while speaking
    "max_alternatives": 1,
}

# ── Whisper settings ───────────────────────────────────────────────────────────
WHISPER = {
    "language": "en",
    "task": "transcribe",       # "transcribe" or "translate"
    "fp16": False,              # set True on CUDA GPU
    "temperature": 0.0,
    "beam_size": 5,
    "best_of": 5,
    "condition_on_previous_text": True,
    "chunk_length_s": 30,       # seconds of audio per Whisper call
    "silence_timeout_s": 2.0,   # stop collecting audio after N s of silence
}

# ── Classifier settings ────────────────────────────────────────────────────────
# Keywords that switch the engine into dictation mode
DICTATION_TRIGGERS = {
    "start dictation",
    "start dictating",
    "begin dictation",
    "dictation mode",
    "type this",
    "write this",
    "transcribe",
}

# Keywords that cancel dictation and return to command mode
DICTATION_STOP_WORDS = {
    "stop dictation",
    "end dictation",
    "stop typing",
    "cancel dictation",
}

# One-shot command prefixes (treated as commands regardless of surrounding text)
COMMAND_PREFIXES = {
    "open",
    "close",
    "search",
    "go to",
    "navigate",
    "scroll",
    "click",
    "press",
    "select",
    "copy",
    "paste",
    "undo",
    "redo",
    "save",
    "delete",
    "new",
    "switch",
    "minimize",
    "maximize",
    "exit",
    "quit",
    "run",
    "execute",
}

# ── Mode enum-like mapping ─────────────────────────────────────────────────────
MODE = {
    "command": "command",
    "dictation": "dictation",
}

# ── Output event types ─────────────────────────────────────────────────────────
EVENT_TYPE = {
    "partial": "partial",       # VOSK partial result (in-progress speech)
    "command": "command",       # final, classified as a voice command
    "dictation": "dictation",   # final, classified as dictation text
    "error": "error",
}

# ── Threading / multiprocessing ────────────────────────────────────────────────
CONCURRENCY = {
    "vosk_queue_maxsize": 100,      # audio chunk queue depth
    "result_queue_maxsize": 50,     # inter-process result queue depth
    "whisper_process_timeout": 30,  # seconds before Whisper worker is considered hung
}
