"""speech_module/config.py — invariants the pipeline silently depends on,
plus the cross-module contracts with central_module/config.py."""

import os

import config as speech_config
from central_module import config as central_config


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
    assert 0 < r["max_unk_ratio"] < 1


def test_vad_hysteresis():
    assert speech_config.VAD["start_factor"] > speech_config.VAD["end_factor"] > 1
    assert speech_config.SILERO["start_prob"] > speech_config.SILERO["end_prob"]


def test_command_grammar():
    grammar = speech_config.command_grammar()
    assert grammar[-1] == "[unk]" and grammar.count("[unk]") == 1
    body = grammar[:-1]
    assert body == sorted(body) and len(body) == len(set(body))
    # Mode-switch phrases must stay recognisable under the grammar.
    assert speech_config.DICTATION_TRIGGERS <= set(body)
    assert speech_config.DICTATION_STOP_WORDS <= set(body)
    assert speech_config.COMMAND_PREFIXES <= set(body)


def test_event_types_cover_the_runner_protocol():
    assert set(speech_config.EVENT_TYPE.values()) == {
        "partial", "command", "dictation", "device", "error"}
