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

# Whisper language code per VOSK language key — keeps both engines in step
# when SpeechManager.set_language() switches at runtime.
WHISPER_LANG = {
    "en-in": "en",
    "en-us": "en",
    "hi-in": "hi",
}

# ── Whisper model paths (faster-whisper / CTranslate2 format) ──────────────────
# Each entry is a *directory* holding model.bin + config.json + tokenizer.json
# + vocabulary.txt, as published by Systran/faster-whisper-<size> on HF.
WHISPER_MODELS = {
    "base":  os.path.join(_MODELS_ROOT, "WHISPER", "faster-whisper-base"),
    "small": os.path.join(_MODELS_ROOT, "WHISPER", "faster-whisper-small"),   # ← default
}

# Active Whisper model key
WHISPER_DEFAULT_MODEL = "small"

# Fallback: if the local model dir is missing, download this size from Hugging Face
WHISPER_FALLBACK_DOWNLOAD = True   # set False to hard-fail instead

# ── DeepFilterNet3 model path ──────────────────────────────────────────────────
# Directory holding config.ini + checkpoints/ — populated by
# `python fetch_deepfilternet_model.py`
DFN3_MODEL_DIR = os.path.join(_MODELS_ROOT, "DFN3")

# ── Silero VAD model path ──────────────────────────────────────────────────────
SILERO_MODEL_PATH = os.path.join(_MODELS_ROOT, "SILERO", "silero_vad.onnx")

# ── Convenience aliases (used by engines) ──────────────────────────────────────
MODEL_PATHS = {
    "vosk":    VOSK_MODELS[VOSK_DEFAULT_LANG],
    "whisper": WHISPER_MODELS[WHISPER_DEFAULT_MODEL],
    "dfn3":    DFN3_MODEL_DIR,
}

# ── Audio settings ─────────────────────────────────────────────────────────────
# Two rates now, not one. The mic is captured at DeepFilterNet3's native
# 48 kHz; both recognisers are fed 16 kHz after resampling (see audio_utils).
AUDIO = {
    "capture_rate": 48000,      # Hz — mic capture; DFN3's native rate
    "target_rate": 16000,       # Hz — what VOSK and Whisper consume
    "channels": 1,              # mono
    "chunk_size": 4800,         # frames per read (100 ms at 48 kHz = 10 DFN3 hops)
    "dtype": "int16",           # raw PCM format
    "device_index": None,       # None = auto-detect system default input device
}

# ── Input-device monitoring ────────────────────────────────────────────────────
# Watches for changes of the system default input device and hot-swaps the
# capture stream onto it (only while AUDIO["device_index"] is None / no device
# was explicitly chosen). Each poll runs in a small subprocess because
# PortAudio only enumerates devices at initialisation — see audio_devices.py.
DEVICE_MONITOR = {
    "enabled": True,
    "poll_interval_s": 3.0,     # seconds between device queries
    "query_timeout_s": 10.0,    # kill a hung query (USB headset enumeration)
}

# ── DeepFilterNet3 denoiser ────────────────────────────────────────────────────
# Applied to the dictation branch only (buffered audio → DFN3 → Whisper).
# The VOSK command branch stays on raw audio to keep latency down.
DENOISE = {
    "enabled": True,
    "model_dir": DFN3_MODEL_DIR,
    "sample_rate": 48000,       # DFN3 is trained at 48 kHz — must equal capture_rate
    # Attenuation limit in dB. 0.0 = the model's full suppression; 10.0 removes
    # at most 10 dB of noise and mixes the rest back in.
    #
    # Measured on this stack (TTS speech + synthetic fan and pink noise, WER vs
    # ground truth): full suppression is *worse* than a limit at low SNR — it
    # strips speech content and Whisper hallucinates over the gaps (+3.3 pts at
    # 5 dB, +10.1 pts at 0 dB). A 10 dB limit matched or beat full suppression
    # at almost every SNR tested and never showed that failure mode.
    "atten_lim_db": 10.0,
    "post_filter": False,       # extra suppression, but adds speech artefacts
    "required": False,          # False = pass audio through if DFN3 can't load
    "min_duration_s": 0.2,      # skip enhancement for buffers shorter than this
}

# ── VOSK settings ──────────────────────────────────────────────────────────────
VOSK = {
    "log_level": -1,            # -1 = silent, 0 = errors, 1 = verbose
    "partial_results": True,    # emit partial text while speaking
    # MUST stay 0. Any value > 0 switches VOSK to an {"alternatives": [...]}
    # schema that has no top-level "text" key and replaces the per-word 0..1
    # "conf" values with an unnormalised path score. Routing needs those
    # per-word confidences, and every text reader here expects "text".
    "max_alternatives": 0,
}

# ── Whisper settings (faster-whisper) ──────────────────────────────────────────
WHISPER = {
    "language": "en",
    "task": "transcribe",       # "transcribe" or "translate"
    "device": "cpu",            # "cuda" on an NVIDIA GPU
    # int8 halves RAM vs fp32 and is the fastest CPU mode; accuracy loss is
    # negligible. Use "float16" on CUDA.
    "compute_type": "int8",
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

# ── VOSK command grammar ───────────────────────────────────────────────────────
# Restricting the per-segment decoder to a known vocabulary makes in-vocabulary
# commands dramatically more accurate, and gives routing a clean OOV signal:
# anything outside the vocabulary decodes as "[unk]" instead of being
# force-fitted onto the nearest-sounding command.
#
# Words the model's lexicon lacks are silently dropped from the grammar by
# Kaldi, and models without a dynamic graph reject grammars entirely — the
# engine probes for that at first use and falls back to free decoding.
#
# The streaming preview recognizer never uses the grammar; partials keep
# showing arbitrary speech.

# Things commands act on / with. Extend freely — this is the list the central
# module's set_vocabulary control message will replace at runtime later.
COMMAND_TARGETS = {
    # apps & surfaces
    "chrome", "browser", "notepad", "explorer", "terminal", "settings",
    "calculator", "spotify", "youtube", "google", "whatsapp", "word", "excel",
    # ui nouns
    "window", "tab", "file", "folder", "page", "menu", "desktop", "screen",
    "volume", "brightness", "music", "video", "process", "task", "manager",
    "text", "line", "word", "all",
    # directions & steps
    "up", "down", "left", "right", "next", "previous", "back", "forward",
    "top", "bottom", "home", "end",
    # small numbers
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "double", "half",
    # frequent glue words so natural phrasing survives the grammar
    "the", "a", "to", "on", "off", "in", "out", "this", "that", "please",
    "mute", "unmute", "play", "pause", "stop", "start", "enter", "escape",
    "space", "shift", "control",
}

GRAMMAR = {
    "enabled": True,
}


def command_grammar() -> list[str]:
    """
    Phrase list for VOSK's grammar-constrained command decode: every command
    word plus the mode-switch phrases (which must stay recognisable or the
    explicit dictation trigger could never fire), plus "[unk]" so
    out-of-vocabulary speech surfaces as [unk] tokens instead of being
    mis-recognised as a command.
    """
    phrases: set[str] = set()
    phrases |= COMMAND_PREFIXES | COMMAND_TARGETS
    phrases |= DICTATION_TRIGGERS | DICTATION_STOP_WORDS
    return sorted(phrases) + ["[unk]"]


# ── Voice activity detection / segmenter ───────────────────────────────────────
# Splits the mic stream into voiced segments. Everything downstream (DFN3,
# VOSK, Whisper) works on whole segments, which is also what DFN3 wants — its
# Python API has no streaming mode.
#
# Thresholds are relative to a running estimate of the room's noise floor
# rather than a fixed RMS, so the same numbers work in a quiet room and next to
# a fan. A fixed threshold is what made the old DictationBuffer unreliable.
VAD = {
    "frame_ms": 20,                 # energy is measured per 20 ms frame
    "noise_floor_percentile": 20,   # percentile of recent quiet frames = floor
    "noise_floor_window_s": 5.0,    # history length for that estimate
    "start_factor": 3.0,            # speech starts above floor * this
    "end_factor": 1.8,              # ...and ends below floor * this (hysteresis)
    "min_speech_ms": 200,           # discard blips shorter than this
    "hangover_ms": 600,             # trailing silence that closes a segment
    "preroll_ms": 300,              # audio retained from *before* onset
    "max_segment_s": 20.0,          # hard cap — force-close runaway segments
    "absolute_floor_rms": 40.0,     # never treat anything quieter than this as speech
}

# ── Silero VAD (preferred segmenter) ───────────────────────────────────────────
# A ~2 MB neural VAD (onnxruntime, sub-ms per frame) that fires on speech
# only — keyboard clatter, music and coughs that clear an energy threshold do
# not clear this one. Falls back to EnergyVAD when the model can't load,
# the same way DFN3 degrades to pass-through. Segment semantics (min speech,
# hangover, preroll, max length) mirror the VAD block above.
SILERO = {
    "enabled": True,
    "model_path": SILERO_MODEL_PATH,
    "start_prob": 0.60,             # frame is speech above this probability …
    "end_prob": 0.35,               # … and silence below this (hysteresis)
    "min_speech_ms": 200,
    "hangover_ms": 600,
    "preroll_ms": 300,
    "max_segment_s": 20.0,
    "required": False,              # True = fail hard instead of falling back
}

# ── Routing: VOSK result → act as command, or escalate to Whisper ─────────────
# "Command if confident, Whisper if long or unsure."
ROUTING = {
    "min_word_confidence": 0.70,    # weakest single word VOSK may report
    "min_mean_confidence": 0.85,    # mean per-word confidence over the utterance
    "max_command_words": 8,         # longer than this is prose, not a command
    "max_command_duration_s": 4.0,
    "escalate_on_empty": True,      # VOSK heard nothing → let Whisper try anyway
    # Share of [unk] tokens (grammar decode only) above which the utterance is
    # not a command. A third tolerates one stray [unk] — a breath or keyboard
    # click — alongside a two-word command, while "open <unknown app>" (1/2)
    # still escalates.
    "max_unk_ratio": 0.34,
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
    "device": "device",         # input device switched / device list changed
    "error": "error",
}

# ── Threading / multiprocessing ────────────────────────────────────────────────
CONCURRENCY = {
    "vosk_queue_maxsize": 100,          # audio chunk queue depth
    "segment_queue_maxsize": 8,         # voiced segments awaiting DFN3/VOSK/Whisper
    "result_queue_maxsize": 50,         # inter-process result queue depth
    "whisper_process_timeout": 30,      # seconds before Whisper worker is considered hung
}

# ── IPC message shaping ─────────────────────────────────────────────────────────
# VOSK's preview recognizer fires a partial result many times per utterance
# (roughly every audio chunk it decodes). Forwarding every one of those over
# IPC is wasteful — the central module only needs enough of them to keep a
# "live captions" UI feeling responsive, not the full stream. These knobs
# collapse consecutive partials down to the minimum that still looks live:
# identical text is dropped, and nothing goes out faster than the interval
# below, however fast VOSK is producing partials underneath.
#
# Final events (command / dictation / device / error) are already exactly
# one message per segment or per state change, so nothing here applies to
# them.
IPC = {
    "partial_dedupe": True,             # drop a partial identical to the last one sent
    "partial_min_interval_s": 0.15,     # minimum gap between partial IPC messages
}