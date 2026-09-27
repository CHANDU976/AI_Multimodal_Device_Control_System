"""speech_module/config.py — invariants the pipeline silently depends on,
plus the cross-module contracts with central_module/config.py."""

import os
import sys

import pytest

import config as speech_config
from central_module import config as central_config
from central_module.nlp.intents import CATALOG, DIRECTIONS


def test_vosk_alternatives_must_stay_zero():
    # Any value > 0 switches VOSK to a schema without per-word 0..1 "conf"
    # values and without a top-level "text" — routing would break silently.
    assert speech_config.VOSK["max_alternatives"] == 0


def test_denoiser_runs_at_the_capture_rate():
    assert speech_config.DENOISE["sample_rate"] == speech_config.AUDIO["capture_rate"]


def test_rates_are_compatible_with_silero_and_resampling():
    capture, target = speech_config.AUDIO["capture_rate"], speech_config.AUDIO["target_rate"]
    assert capture % 16000 == 0            # SileroVAD requires a 16 kHz multiple
    assert target == 16000                 # VOSK and Whisper both consume 16 kHz
    assert speech_config.AUDIO["dtype"] == "int16" and speech_config.AUDIO["channels"] == 1


def test_chunk_is_whole_vad_frames():
    frame = speech_config.AUDIO["capture_rate"] * speech_config.VAD["frame_ms"] // 1000
    assert speech_config.AUDIO["chunk_size"] % frame == 0


def test_language_tables_agree():
    assert set(speech_config.WHISPER_LANG) == set(speech_config.VOSK_MODELS)
    assert speech_config.VOSK_DEFAULT_LANG in speech_config.VOSK_MODELS
    assert speech_config.WHISPER_DEFAULT_MODEL in speech_config.WHISPER_MODELS


def test_central_language_names_map_to_real_vosk_models():
    """central_module sends {"op": "set_language", "lang": X}; X must exist."""
    assert set(central_config.LANGUAGES.values()) <= set(speech_config.VOSK_MODELS)


def test_speech_stop_words_are_honoured_by_the_central_module():
    # Whatever the speech side treats as "stop dictation", central must too.
    assert speech_config.DICTATION_STOP_WORDS <= central_config.DICTATION_STOP


def test_model_paths_point_into_models_dir():
    models = os.path.normpath(speech_config._MODELS_ROOT)
    for path in (*speech_config.VOSK_MODELS.values(), *speech_config.WHISPER_MODELS.values(),
                 speech_config.DFN3_MODEL_DIR, speech_config.SILERO_MODEL_PATH):
        assert os.path.normpath(path).startswith(models)


def test_model_directories_exist_in_the_checkout():
    for path in (*speech_config.VOSK_MODELS.values(), *speech_config.WHISPER_MODELS.values()):
        assert os.path.isdir(path), path
    assert os.path.exists(speech_config.SILERO_MODEL_PATH)


def test_routing_thresholds_are_sane():
    r = speech_config.ROUTING
    assert 0 < r["min_word_confidence"] <= r["min_mean_confidence"] <= 1
    assert r["max_command_words"] >= 2 and r["max_command_duration_s"] > 0


def test_vad_hysteresis():
    assert speech_config.VAD["start_factor"] > speech_config.VAD["end_factor"] > 1
    assert speech_config.SILERO["start_prob"] > speech_config.SILERO["end_prob"]


def test_event_types_cover_the_runner_protocol():
    assert set(speech_config.EVENT_TYPE.values()) == {
        "partial", "command", "dictation", "device", "error"}


# ── VOSK command grammar ───────────────────────────────────────────────────────

def test_command_grammar():
    grammar = speech_config.command_grammar()
    assert grammar[-1] == "[unk]" and grammar.count("[unk]") == 1
    body = set(grammar[:-1])
    assert grammar[:-1] == sorted(body)
    for group in (set(speech_config.command_phrases()), speech_config.OPEN_ENDED_PREFIXES,
                  speech_config.FILLER_PHRASES, speech_config.EVERYDAY_PHRASES,
                  # Mode-switch phrases must stay recognisable under the grammar.
                  speech_config.DICTATION_TRIGGERS, speech_config.DICTATION_STOP_WORDS):
        assert group <= body


def test_only_command_phrases_can_run():
    # Decoys, fillers and openers exist to send speech to Whisper. One that is
    # also a command phrase would put that speech back on the fast path.
    commands = set(speech_config.command_phrases())
    for group in (speech_config.EVERYDAY_PHRASES, speech_config.FILLER_PHRASES,
                  speech_config.OPEN_ENDED_PREFIXES):
        assert not group & commands


def test_spoken_forms_decode_back_to_canonical_words():
    phrases = speech_config.command_phrases()
    assert phrases["open note pad"] == phrases["open notepad"] == "open notepad"
    assert phrases["press f five"] == "press f5"
    assert phrases["un mute the volume"] == "unmute the volume"
    assert all(" " not in word for word in speech_config.SPOKEN_FORMS)


def test_slots_expand_into_every_combination():
    scroll = speech_config._expand_slots("scroll {direction} {count} times")
    slots = speech_config.GRAMMAR_SLOTS
    assert len(scroll) == len(slots["direction"]) * len(slots["count"])
    assert "scroll down three times" in scroll
    assert speech_config._expand_slots("copy") == ["copy"]


# ── Grammar ↔ central_module contract ──────────────────────────────────────────
# A grammar phrase the central module doesn't understand is a fast-path command
# that does nothing; a catalog command missing from the grammar can never take
# the fast path. Both directions are pinned here.

# The central normaliser turns "double"/"twice" into 2, so these reach a fuzzy
# SINGLE click — central's bug, tracked as KNOWN_UNREACHABLE (strict xfail) in
# tests/central/test_intent_engine.py.
CENTRAL_KNOWN_UNREACHABLE = {"double click", "click twice"}


@pytest.fixture(scope="module")
def central_engine():
    from central_module.nlp.intent_engine import IntentEngine
    original = central_config.NLP["engine"]
    central_config.NLP["engine"] = "rules"      # exact matching needs no model
    try:
        yield IntentEngine()
    finally:
        central_config.NLP["engine"] = original


def test_every_command_phrase_runs_its_intent_in_the_central_module(central_engine):
    # Some answers are handled by the central manager before intent parsing.
    manager_words = {"confirm": central_config.CONFIRM_WORDS,
                     "cancel": central_config.CANCEL_WORDS,
                     "stop_dictation": central_config.DICTATION_STOP}
    wrong = []
    for intent, templates in speech_config.COMMAND_PHRASES.items():
        for template in templates:
            for text in speech_config._expand_slots(template):
                if text in CENTRAL_KNOWN_UNREACHABLE or text in manager_words.get(intent, ()):
                    continue
                match = central_engine.parse(text)
                if match is None or (match.intent.name, match.source) != (intent, "exact"):
                    wrong.append((intent, text, match))
    assert not wrong


def test_the_whole_central_catalog_is_in_the_grammar():
    phrases = speech_config.COMMAND_PHRASES
    for intent in CATALOG:
        for pattern in intent.patterns:
            if intent.name == "set_language":
                # Its "text" is a closed list of language names: full phrases.
                assert pattern.replace("{text}", "{language}") in phrases["set_language"]
            elif "{text}" in pattern:
                # Free text never runs from VOSK, but the opener must decode so
                # "search for [unk]" goes to Whisper instead of a wrong command.
                assert pattern.split(" {")[0] in speech_config.OPEN_ENDED_PREFIXES, pattern
            elif (intent.name, pattern) != ("press_key", "press the {key} key"):
                # (that one is unreachable centrally — see KNOWN_UNREACHABLE)
                assert pattern in phrases[intent.name], (intent.name, pattern)


def test_confirm_and_cancel_words_are_all_in_the_grammar():
    phrases = speech_config.COMMAND_PHRASES
    assert central_config.CONFIRM_WORDS <= set(phrases["confirm"])
    assert central_config.CANCEL_WORDS <= set(phrases["cancel"])


def test_slots_match_what_the_central_module_accepts():
    slots = speech_config.GRAMMAR_SLOTS
    assert set(slots["direction"]) == DIRECTIONS
    assert set(slots["language"]) == set(central_config.LANGUAGES)


@pytest.mark.parametrize("module, adapter", [
    pytest.param("windows_adapter", "WindowsAdapter", marks=pytest.mark.skipif(
        sys.platform != "win32", reason="WindowsAdapter binds user32.dll at import")),
    ("linux_adapter", "LinuxAdapter"),
    ("macos_adapter", "MacAdapter"),
])
def test_every_launchable_app_is_in_the_grammar(module, adapter):
    import importlib
    adapter_cls = getattr(importlib.import_module(f"central_module.actions.{module}"), adapter)
    assert set(adapter_cls.APP_COMMANDS) <= set(speech_config.GRAMMAR_SLOTS["app"])
