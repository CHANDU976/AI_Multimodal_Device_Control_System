# config.py — Central configuration for the Speech Module

import itertools
import os
import re

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
# The per-segment VOSK decode is constrained to the phrases below. That makes
# in-vocabulary commands far more accurate — but a closed vocabulary also means
# VOSK always answers with *something* from it, and with few competing paths
# that answer scores ~1.0 however poor the acoustic match. A bag of loose words
# let "the weather is nice today" come out as "nine to" at confidence 1.0.
#
# So the grammar gives every kind of speech an honest place to land:
#
#   COMMAND_PHRASES      complete commands — action + target — grouped by the
#                        central_module intent they trigger. The ONLY way onto
#                        the fast path is decoding to exactly one of these,
#                        optionally wrapped in FILLER_PHRASES (router.py).
#   OPEN_ENDED_PREFIXES  command openers whose object the grammar can't list —
#                        a search, a website, an app it doesn't know. They let
#                        "open sesame" decode as "open [unk]" (→ Whisper)
#                        instead of being forced onto "open gmail".
#   FILLER_PHRASES       politeness around a command: "can you … please".
#   EVERYDAY_PHRASES     common speech that is not a command. Decoys: they can
#                        only ever send an utterance to Whisper, never run one.
#   "[unk]"              anything else.
#
# Small VOSK models lack some words ("notepad", "unmute", "f1"). SPOKEN_FORMS
# spells those with words the model has, and the router maps the decode back
# to the canonical word. Phrases a model still can't say are dropped whole by
# VoskEngine — left to Kaldi, it deletes just the missing word, turning
# "open notepad" into "open".
#
# Models without a dynamic graph reject grammars entirely; the engine probes
# for that at first use and falls back to free decoding. The streaming preview
# recognizer never uses the grammar, so partials keep showing arbitrary speech.

GRAMMAR = {
    "enabled": True,
}

# Values for the {slot} placeholders in COMMAND_PHRASES. Canonical spellings —
# SPOKEN_FORMS covers the ones a model can't say.
GRAMMAR_SLOTS = {
    # Every app name the Windows, Linux and macOS adapters can launch
    # (APP_COMMANDS in central_module/actions/*_adapter.py).
    "app": (
        "browser", "calc", "calculator", "chrome", "cmd", "code", "command prompt",
        "control panel", "edge", "excel", "explorer", "file explorer", "files",
        "finder", "firefox", "gmail", "google", "google chrome", "notepad", "notes",
        "outlook", "paint", "powerpoint", "powershell", "safari", "settings",
        "spotify", "task manager", "terminal", "visual studio code", "vs code",
        "whatsapp", "word", "youtube",
    ),
    "direction": ("up", "down", "left", "right", "top", "bottom"),
    # Key names from central KEY_ALIASES. Digits are spoken as words (the
    # central normaliser turns "five" into 5); the letter "a" is left out
    # because the normaliser drops it as an article.
    "key": (
        "alt", "backspace", "caps lock", "command", "control", "delete", "down",
        "down arrow", "end", "enter", "escape", "home", "insert", "left",
        "left arrow", "option", "page down", "page up", "print screen", "return",
        "right", "right arrow", "shift", "space", "spacebar", "super", "tab", "up",
        "up arrow", "windows", "windows key",
        "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
        "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
        "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ),
    "count": ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
    # Volume levels, in steps of five.
    "number": (
        "zero", "five", "ten", "fifteen", "twenty", "twenty five", "thirty",
        "thirty five", "forty", "forty five", "fifty", "fifty five", "sixty",
        "sixty five", "seventy", "seventy five", "eighty", "eighty five", "ninety",
        "ninety five", "one hundred",
    ),
    # Spoken language names — central_module/config.py LANGUAGES.
    "language": ("english", "indian english", "american english", "hindi"),
}

# Every command the fast path may act on, by central intent name. The catalog
# wording (central_module/nlp/intents.py) plus natural variants the central
# normaliser reduces to it ("close the window" → "close window").
# tests/speech/test_speech_config.py checks every phrase against the catalog.
COMMAND_PHRASES = {
    "open_app": ("open {app}", "open the {app}", "launch {app}", "start {app}",
                 "run {app}", "edit {app}"),
    "press_key": ("press {key}", "hit {key}", "tap {key}", "press key {key}"),
    "copy": ("copy", "copy this", "copy that"),
    "paste": ("paste", "paste it", "paste here"),
    "cut": ("cut", "cut this"),
    "select_all": ("select all", "select everything"),
    "undo": ("undo", "undo that"),
    "redo": ("redo", "redo that"),
    "save": ("save", "save file", "save this file", "save the file"),
    "delete": ("delete", "delete this", "delete that", "delete selection",
               "delete the selection"),
    "scroll": ("scroll {direction}", "scroll {direction} {count}",
               "scroll {direction} {count} times", "scroll to {direction}",
               "scroll to the {direction}"),
    "click": ("click", "left click", "click here"),
    "double_click": ("double click", "click twice"),
    "right_click": ("right click", "context menu"),
    "volume_up": ("volume up", "increase volume", "raise volume", "louder",
                  "turn up volume", "turn volume up", "turn up the volume",
                  "turn the volume up", "increase the volume", "raise the volume",
                  "volume up {count}", "volume up by {count}",
                  "increase volume by {count}", "raise volume by {count}",
                  "turn volume up by {count}"),
    "volume_down": ("volume down", "decrease volume", "lower volume", "quieter",
                    "turn down volume", "turn volume down", "turn down the volume",
                    "turn the volume down", "decrease the volume", "lower the volume",
                    "volume down {count}", "volume down by {count}",
                    "decrease volume by {count}", "lower volume by {count}",
                    "turn volume down by {count}"),
    "mute": ("mute", "mute volume", "mute the volume", "silence"),
    "unmute": ("unmute", "unmute volume", "unmute the volume", "sound on"),
    "set_volume": ("set volume to {number}", "set the volume to {number}",
                   "set volume {number}", "volume {number} percent",
                   "volume to {number}", "increase volume to {number}",
                   "raise volume to {number}", "decrease volume to {number}",
                   "lower volume to {number}"),
    "play_pause": ("play", "pause", "play pause", "resume", "play music", "pause music",
                   "stop music", "play the music", "pause the music", "stop the music"),
    "next_track": ("next track", "next song", "skip song", "skip track",
                   "skip the song"),
    "prev_track": ("previous track", "previous song", "last song", "go back song"),
    "minimize": ("minimize", "minimize window", "minimize the window", "minimize this"),
    "maximize": ("maximize", "maximize window", "maximize the window", "full screen"),
    "close_window": ("close window", "close the window", "close this window",
                     "close current window", "close the current window"),
    "switch_window": ("switch window", "switch windows", "next window", "change window",
                      "switch the window", "change the window"),
    "show_desktop": ("show desktop", "show the desktop", "go to desktop",
                     "go to the desktop"),
    "new_tab": ("new tab", "open new tab", "open a new tab"),
    "close_tab": ("close tab", "close the tab", "close this tab"),
    "next_tab": ("next tab", "switch tab", "the next tab"),
    "prev_tab": ("previous tab", "last tab", "the previous tab"),
    "browser_back": ("go back", "navigate back", "back page"),
    "browser_forward": ("go forward", "navigate forward", "forward page"),
    "refresh": ("refresh", "reload", "refresh page", "reload page", "refresh the page",
                "reload the page"),
    "screenshot": ("take screenshot", "take a screenshot", "screenshot",
                   "capture screen", "capture the screen"),
    "lock": ("lock screen", "lock the screen", "lock computer", "lock the computer",
             "lock my computer", "lock pc"),
    "brightness_up": ("brightness up", "increase brightness", "increase the brightness",
                      "brighter screen"),
    "brightness_down": ("brightness down", "decrease brightness",
                        "decrease the brightness", "dim screen", "dim the screen"),
    "sleep": ("go to sleep", "sleep computer", "suspend"),
    "shutdown": ("shut down", "shutdown", "shut down computer", "shut down the computer",
                 "power off", "turn off computer", "turn off the computer"),
    "restart": ("restart", "reboot", "restart computer", "restart the computer",
                "reboot computer", "reboot the computer"),
    "set_language": ("switch language to {language}", "change language to {language}",
                     "set language to {language}"),
    "start_dictation": ("start dictation", "start dictating", "begin dictation",
                        "dictation mode"),
    "stop_dictation": ("stop dictation", "end dictation", "stop typing",
                       "cancel dictation", "exit dictation"),
    # Answers to "are you sure?" — central CONFIRM_WORDS / CANCEL_WORDS.
    "confirm": ("yes", "yeah", "sure", "confirm", "go ahead", "do it"),
    "cancel": ("no", "don't", "cancel", "never mind", "abort"),
}

# Command openers whose object may be outside the grammar. Never a command on
# their own: "<opener> [unk]" always goes to Whisper for the free text.
OPEN_ENDED_PREFIXES = {
    "open", "open the", "launch", "start", "run", "edit",
    "press", "hit", "tap", "press key",
    "search", "search for", "search the web for", "search google for", "google",
    "look up", "go to", "open website", "visit", "type", "write",
    "switch language to", "change language to", "set language to",
}

# Politeness that may wrap a command; the router strips it before matching.
FILLER_PHRASES = {
    "please", "kindly", "hey", "hey computer", "okay", "ok",
    "can you", "could you", "would you", "will you", "i want to", "i want you to",
    "just", "now", "for me", "thanks", "thank you",
}

# Everyday speech that is not a command — what gets said near a microphone:
# greetings, reactions, hesitation, questions, mic checks. Deliberately
# includes near-misses of real commands ("shut up" / "shut down", "be quiet" /
# "quieter", "stop talking" / "stop typing") so those have a non-command
# landing spot. Must never contain a command phrase.
EVERYDAY_PHRASES = {
    # greetings
    "hello", "hi", "hi there", "hello there", "hey there", "good morning",
    "good afternoon", "good evening", "good night", "bye", "bye bye", "goodbye",
    "see you", "see you later", "see you soon", "take care", "nice to meet you",
    "welcome",
    # reactions
    "alright", "all right", "fine", "cool", "great", "nice", "good", "very good",
    "awesome", "perfect", "excellent", "amazing", "wow", "oh", "oh no", "oh my god",
    "oops", "hmm", "um", "uh", "ah", "aha", "i see", "got it", "i got it",
    "makes sense", "that's right", "exactly", "of course", "right", "true",
    "interesting", "really", "seriously", "no way", "come on", "that's it",
    "that's fine", "that's good", "not bad", "me too", "why not",
    # thanks and apologies
    "thank you so much", "thanks a lot", "you're welcome", "no problem",
    "no worries", "sorry", "i'm sorry", "excuse me", "pardon", "pardon me", "my bad",
    # hesitation
    "wait", "wait a second", "wait a minute", "hold on", "one second", "one moment",
    "just a moment", "give me a second", "let me think", "let me see", "let me check",
    "i don't know", "i'm not sure", "maybe", "maybe later", "not now", "later",
    "i think so", "i guess", "well", "anyway", "so", "and then", "what else",
    # questions
    "what", "why", "how", "when", "where", "who", "what is it", "what is this",
    "what is that", "what's up", "what's that", "what happened", "what do you mean",
    "what time is it", "what's the time", "what day is it", "how are you",
    "how is it going", "who is it", "who is this", "where is it", "where are you",
    "is it working", "does it work", "are you there", "are you listening",
    "can you hear me", "do you understand", "did you hear that", "are you sure",
    "is that right",
    # mic checks
    "test", "testing", "testing testing", "mic check", "check check", "hello hello",
    "testing one two three",
    # statements
    "i'm here", "i am here", "i'm back", "i'm done", "i'm ready", "let's go",
    "it's working", "it is working", "it's not working", "this is good",
    "try again", "one more time", "again", "come here", "look at this",
    "shut up", "be quiet", "calm down", "slow down", "sit down",
    "stop it", "stop that", "stop talking",       # vs "stop typing" / "stop music"
}

# Spellings for canonical words small VOSK models lack. The decode comes back
# spelled ("open note pad") and the router restores the canonical word.
SPOKEN_FORMS = {
    "notepad": ("note pad",),
    "powershell": ("power shell",),
    "firefox": ("fire fox",),
    "whatsapp": ("whats app", "what's app"),
    "cmd": ("c m d",),
    "backspace": ("back space",),
    "spacebar": ("space bar",),
    "unmute": ("un mute",),
    "f1": ("f one",), "f2": ("f two",), "f3": ("f three",), "f4": ("f four",),
    "f5": ("f five",), "f6": ("f six",), "f7": ("f seven",), "f8": ("f eight",),
    "f9": ("f nine",), "f10": ("f ten",), "f11": ("f eleven",), "f12": ("f twelve",),
}


def command_phrases() -> dict[str, str]:
    """
    Every spoken command phrase the grammar accepts → the canonical text to act
    on: slots expanded from GRAMMAR_SLOTS, SPOKEN_FORMS spellings included and
    mapped back ("open note pad" → "open notepad").
    """
    phrases: dict[str, str] = {}
    for templates in COMMAND_PHRASES.values():
        for template in templates:
            for text in _expand_slots(template):
                spellings = [(w, *SPOKEN_FORMS.get(w, ())) for w in text.split()]
                for spoken in itertools.product(*spellings):
                    phrases.setdefault(" ".join(spoken), text)
    return phrases


def _expand_slots(template: str) -> list[str]:
    names = re.findall(r"\{(\w+)\}", template)
    texts = []
    for values in itertools.product(*(GRAMMAR_SLOTS[n] for n in names)):
        text = template
        for name, value in zip(names, values):
            text = text.replace("{" + name + "}", value, 1)
        texts.append(text)
    return texts


def command_grammar() -> list[str]:
    """
    Phrase list for VOSK's grammar-constrained segment decode: every command
    phrase, the open-ended openers, fillers, everyday decoys and the mode-switch
    phrases (the explicit dictation trigger must stay recognisable), plus
    "[unk]" for everything else.
    """
    phrases = set(command_phrases())
    phrases |= OPEN_ENDED_PREFIXES | FILLER_PHRASES | EVERYDAY_PHRASES
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
    # Grammar decodes additionally must be exactly one COMMAND_PHRASES entry
    # (fillers aside) with no [unk] at all — see router.py.
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
# below, however fast VOSK is producing partials underneath. The one exception:
# when an utterance ends, a partial the interval held back is sent at once, so
# the last caption is never stale.
#
# Final events (command / dictation / device / error) are already exactly
# one message per segment or per state change, so nothing here applies to
# them.
IPC = {
    "partial_dedupe": True,             # drop a repeat of this utterance's last partial
    "partial_min_interval_s": 0.15,     # minimum gap between partial IPC messages
    # Envelope identity, shared with the gesture module (docs/ipc-protocol.md).
    "module": "speech",
    "protocol_version": 1,
}
