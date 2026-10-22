"""The child-process protocol (docs/ipc-protocol.md), checked across modules.

The speech and gesture modules build their messages independently, and
neither imports the other or the central module. This file is where the
three are held to one contract: the same envelope, one version, no event
name that means two different things, the same payload keys for shared
types, and a central dispatcher that handles every type either module
can send. Each module's own tests cover what its payloads mean.
"""

import ast
import inspect
import json
import types
from pathlib import Path

import pytest

import gesture_events as ge
from central_module import config as central_config
from central_module import manager as manager_mod
from config import EVENT_TYPE, IPC
from speech_manager import SpeechEvent

ROOT = Path(__file__).resolve().parents[2]

ENVELOPE_KEYS = {"type", "module", "timestamp", "version", "payload"}

# Shared types: every module that sends one uses these payload keys.
# ready and heartbeat may add module-specific keys; fatal and pong may not.
SHARED_REQUIRED = {"ready": {"status"}, "heartbeat": {"status"},
                   "fatal": {"code", "message"}, "pong": set()}
SHARED_EXACT = {"fatal", "pong"}

# Types the speech side sends: SpeechEvent's types, plus the messages
# speech_runner builds itself (its own tests check those payloads).
SPEECH_RUNNER_TYPES = {"ready", "fatal", "pong", "devices", "error"}
SPEECH_TYPES = set(EVENT_TYPE.values()) | SPEECH_RUNNER_TYPES
SPEECH_PAYLOAD_KEYS = {"text", "source", "model", "confidence", "segment_start",
                       "metadata"}


def runner_constants():
    """speech_runner's MODULE and PROTOCOL_VERSION, read without importing it.

    Importing the runner redirects this process's stdout (it claims fd 1
    for the protocol), so its constants are read from the source instead.
    """
    tree = ast.parse((ROOT / "central_module/runners/speech_runner.py")
                     .read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in ("MODULE", "PROTOCOL_VERSION"):
                values[name] = ast.literal_eval(node.value)
    return values


def speech_messages():
    events = [
        SpeechEvent("command", "open chrome", "vosk", "command",
                    {"vosk_model": "vosk-model-small-en-in-0.4",
                     "mean_confidence": 0.95, "segment_start": 1790000000.0,
                     "words": [{"word": "open", "start": 0.1, "end": 0.3, "conf": 0.95}]}),
        SpeechEvent("dictation", "The report is due.", "whisper", "dictation",
                    {"whisper_model": "faster-whisper-small", "whisper_confidence": 0.8}),
        SpeechEvent("partial", "open chr", "vosk", "partial", {}),
        SpeechEvent("device", "USB Mic", "audio", "device", {"reason": "list_changed"}),
        SpeechEvent("error", "", "whisper", "error", {"error": "timeout"}),
    ]
    return [e.to_ipc() for e in events]


def gesture_messages():
    return [
        ge.ready(camera_index=0, semantic="ready", cursor_control="always"),
        ge.pointing_started(hand="right", cursor_control="always"),
        ge.pointing_stopped(reason=ge.REASON_HAND_LOST, hand="right"),
        ge.gesture_fired(gesture="fist", action="LEFT_CLICK", confidence=0.9),
        ge.drag_started(gesture="grip"),
        ge.drag_stopped(reason=ge.REASON_GESTURE, duration_sec=1.2),
        ge.heartbeat(pointing=True, dragging=False, fps=29.7),
        ge.fatal(code=ge.FATAL_TRACKING_LOOP, message="boom"),
        ge.pong(),
    ]


ALL = [("speech", m) for m in speech_messages()] + \
      [("gesture", m) for m in gesture_messages()]


# ── Envelope ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module, message", ALL,
                         ids=[f"{mod}:{msg['type']}" for mod, msg in ALL])
def test_every_message_uses_the_shared_envelope(module, message):
    assert set(message) == ENVELOPE_KEYS
    assert message["module"] == module
    assert message["version"] == central_config.IPC_PROTOCOL_VERSION
    assert isinstance(message["timestamp"], float)
    assert message["timestamp"] == round(message["timestamp"], 3)
    assert isinstance(message["payload"], dict)
    text = json.dumps(message, allow_nan=False)          # strict JSON
    assert json.loads(text) == message


def test_every_side_agrees_on_the_version_and_names():
    runner = runner_constants()
    assert runner["MODULE"] == IPC["module"] == "speech"
    assert ge.MODULE == "gesture"
    assert (runner["PROTOCOL_VERSION"] == IPC["protocol_version"]
            == ge.PROTOCOL_VERSION == central_config.IPC_PROTOCOL_VERSION)
    # The module names are the supervisor's child names, so a message's
    # "module" and the child it arrived from always agree.
    source = (inspect.getsource(manager_mod.CentralManager._build_children)
              + inspect.getsource(manager_mod.CentralManager._gesture_spec))
    assert 'name="speech"' in source and 'name="gesture"' in source


# ── Event types ────────────────────────────────────────────────────────────────

def test_only_shared_types_appear_in_both_modules():
    assert SPEECH_TYPES & set(ge.EVENT_TYPES) <= set(SHARED_REQUIRED)


def test_gesture_shared_types_use_the_shared_payload():
    for message in gesture_messages():
        kind = message["type"]
        if kind not in SHARED_REQUIRED:
            continue
        keys = set(message["payload"])
        assert SHARED_REQUIRED[kind] <= keys, kind
        if kind in SHARED_EXACT:
            assert keys == SHARED_REQUIRED[kind], kind
    assert ge.ready()["payload"]["status"] == "ready"
    assert ge.heartbeat(pointing=False, dragging=False)["payload"]["status"] == "running"


def test_speech_payloads_always_carry_every_key():
    for message in speech_messages():
        assert set(message["payload"]) == SPEECH_PAYLOAD_KEYS, message["type"]


def test_speech_from_a_segment_says_when_it_was_spoken():
    by_type = {m["type"]: m["payload"] for m in speech_messages()}
    command = by_type["command"]
    assert command["segment_start"] == 1790000000.0
    word = command["metadata"]["words"][0]
    assert command["segment_start"] + word["start"] == pytest.approx(1790000000.1)
    assert by_type["partial"]["segment_start"] is None
    assert by_type["device"]["segment_start"] is None


def test_source_is_never_an_envelope_key():
    # "source" means the recognition engine inside a speech payload; it must
    # not reappear at the top level with a different meaning.
    for _, message in ALL:
        assert "source" not in message


# ── The central side ───────────────────────────────────────────────────────────

def recording_manager():
    calls = []
    fake = types.SimpleNamespace(
        _verbose=False,
        _say=lambda text: calls.append(("say", text)),
        handle_text=lambda text, origin="speech": calls.append(("text", text)),
        _handle_gesture=lambda payload: calls.append(("gesture", payload.get("gesture"))),
        _track_gesture=lambda kind, msg, payload: None,
    )
    return fake, calls


@pytest.mark.parametrize("module, kind", sorted(
    [("speech", t) for t in SPEECH_TYPES] + [("gesture", t) for t in ge.EVENT_TYPES]))
def test_central_handles_every_type_a_module_can_send(module, kind, caplog):
    fake, _ = recording_manager()
    message = {"type": kind, "module": module, "timestamp": 1790000000.0,
               "version": central_config.IPC_PROTOCOL_VERSION, "payload": {}}
    manager_mod.CentralManager._dispatch(fake, module, message)
    assert "unhandled message type" not in caplog.text
    assert "protocol version" not in caplog.text


def test_central_reads_real_messages_from_both_modules():
    fake, calls = recording_manager()
    for module, message in ALL:
        manager_mod.CentralManager._dispatch(fake, module, message)
    assert ("text", "open chrome") in calls                # speech command
    assert ("text", "The report is due.") in calls         # speech dictation
    assert ("gesture", "fist") in calls                    # gesture_fired
    assert ("say", "gesture failed: boom") in calls        # gesture fatal
