"""runners/ and __main__ — the child-process protocol and the CLI, end to end.

Each test launches a real interpreter. The speech runner gets a stand-in
`speech_manager` injected before it imports, so the protocol (ready,
events, control ops, EOF handling, stdout hygiene) is tested without a
microphone or a single model.
"""

import json
import subprocess
import sys
import textwrap

import pytest

from helpers.paths import ROOT

pytestmark = pytest.mark.slow


def run_py(code, stdin="", timeout=60, env_extra=None):
    import os
    # Piped stdout on Windows defaults to cp1252, which cannot encode the
    # "»"/"✓" console marks; production runs attach a UTF-8 console instead.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **(env_extra or {})}
    return subprocess.run([sys.executable, "-c", code], input=stdin, cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8",
                          timeout=timeout, env=env)


def json_lines(stdout):
    lines = [line for line in stdout.splitlines() if line.strip()]
    parsed = [json.loads(line) for line in lines]      # every line MUST be JSON
    return parsed


# ── gesture_runner (stub child) ────────────────────────────────────────────────

GESTURE = ("import runpy; "
           "runpy.run_module('central_module.runners.gesture_runner', run_name='__main__')")


def test_gesture_stub_protocol():
    ops = "\n".join(json.dumps(o) for o in [
        {"op": "ping"}, {"op": "emit", "name": "fist", "confidence": "0.5"},
        {"op": "unknown"}]) + "\nnot json\n\n" + json.dumps({"op": "shutdown"}) + "\n" + \
        json.dumps({"op": "ping"}) + "\n"                 # after shutdown: ignored
    proc = run_py(GESTURE, ops)
    assert proc.returncode == 0, proc.stderr
    events = json_lines(proc.stdout)
    assert [(e["type"], e["module"], e["version"]) for e in events] == [
        ("ready", "gesture", 1), ("pong", "gesture", 1), ("gesture_fired", "gesture", 1)]
    fired = events[2]["payload"]
    assert (fired["gesture"], fired["confidence"], fired["executed"]) == ("fist", 0.5, False)


def test_gesture_stub_exits_when_parent_disappears():
    proc = run_py(GESTURE, stdin="")                     # immediate EOF
    assert proc.returncode == 0
    assert [e["type"] for e in json_lines(proc.stdout)] == ["ready"]


# ── speech_runner with a stand-in SpeechManager ────────────────────────────────

FAKE_SPEECH = textwrap.dedent("""
    import sys, types, runpy
    fake = types.ModuleType("speech_manager")

    class SpeechEvent:
        def __init__(self, event_type, text, source, metadata):
            self.event_type, self.text, self.source, self.metadata = \\
                event_type, text, source, metadata

        def to_ipc(self):                # the real shape is unit-tested in tests/speech
            return {"type": self.event_type, "module": "speech",
                    "timestamp": 1700000000.123, "version": 1,
                    "payload": {"text": self.text, "source": self.source,
                                "model": "vosk-model-small-en-in-0.4",
                                "confidence": 0.97, "segment_start": 1699999999.2,
                                "metadata": self.metadata}}

    class SpeechManager:
        def __init__(self, callback):
            self.cb = callback
            self.input_devices = [{"index": 1, "name": "USB Mic"}]
        def start(self):
            print("stray print from a native library")   # must NOT hit the protocol
            self.cb(SpeechEvent("command", "open chrome", "vosk", {"conf": 0.97}))
        def set_language(self, lang):
            if lang == "xx":
                raise ValueError("Unknown language 'xx'")
            sys.stderr.write(f"LANG={lang}\\n")
        def set_input_device(self, index):
            sys.stderr.write(f"DEVICE={index}\\n")
        def stop(self):
            sys.stderr.write("STOPPED\\n")

    fake.SpeechManager, fake.SpeechEvent = SpeechManager, SpeechEvent
    MODE = __MODE__
    if MODE == "import-fails":
        fake = None                      # makes `from speech_manager import …` fail
    elif MODE == "start-fails":
        def boom(self): raise RuntimeError("no microphone")
        SpeechManager.start = boom
    sys.modules["speech_manager"] = fake
    runpy.run_module("central_module.runners.speech_runner", run_name="__main__")
""")


def speech(mode, stdin):
    return run_py(FAKE_SPEECH.replace("__MODE__", repr(mode)), stdin)


def test_speech_runner_protocol():
    ops = [{"op": "ping"}, {"op": "set_language", "lang": "hi-in"},
           {"op": "set_language", "lang": "xx"}, {"op": "set_input_device", "index": 2},
           {"op": "get_devices"}, {"op": "frobnicate"}, {"op": "shutdown"}]
    stdin = "\n".join(json.dumps(o) for o in ops[:3]) + "\n{broken\n" + \
        "\n".join(json.dumps(o) for o in ops[3:]) + "\n"
    proc = speech("ok", stdin)
    assert proc.returncode == 0, proc.stderr
    events = json_lines(proc.stdout)                    # also proves stdout hygiene
    assert events[0] == {"type": "command", "module": "speech",       # to_ipc(), verbatim
                         "timestamp": 1700000000.123, "version": 1,
                         "payload": {"text": "open chrome", "source": "vosk",
                                     "model": "vosk-model-small-en-in-0.4",
                                     "confidence": 0.97, "segment_start": 1699999999.2,
                                     "metadata": {"conf": 0.97}}}
    # Messages the runner builds itself use the same envelope.
    for event in events[1:]:
        assert set(event) == {"type", "module", "timestamp", "version", "payload"}
        assert (event["module"], event["version"]) == ("speech", 1)
    assert [e["type"] for e in events[1:]] == ["ready", "pong", "error", "devices"]
    assert events[1]["payload"] == {"status": "ready"}
    assert events[2]["payload"] == {}
    error = events[3]["payload"]
    assert error["source"] == "runner" and error["metadata"]["op"] == "set_language"
    assert error["segment_start"] is None
    assert "Unknown language" in error["metadata"]["error"]
    assert events[4]["payload"] == {"devices": [{"index": 1, "name": "USB Mic"}]}
    assert len(events) == 5
    assert "stray print from a native library" in proc.stderr
    for marker in ("LANG=hi-in", "DEVICE=2", "bad control line", "unknown op", "STOPPED"):
        assert marker in proc.stderr


def test_speech_runner_shuts_down_on_parent_eof():
    proc = speech("ok", stdin="")
    assert proc.returncode == 0
    assert "STOPPED" in proc.stderr


def test_speech_runner_reports_import_failure():
    proc = speech("import-fails", stdin="")
    assert proc.returncode == 1
    fatal = json_lines(proc.stdout)
    assert fatal[0]["type"] == "fatal" and fatal[0]["module"] == "speech"
    assert fatal[0]["payload"]["code"] == "import_failed"
    assert "speech import failed" in fatal[0]["payload"]["message"]


def test_speech_runner_reports_start_failure():
    proc = speech("start-fails", stdin="")
    assert proc.returncode == 1
    fatal = json_lines(proc.stdout)[-1]
    assert fatal["type"] == "fatal"
    assert fatal["payload"] == {"code": "start_failed",
                                "message": "start failed: no microphone"}


# ── python -m central_module ───────────────────────────────────────────────────

CLI = textwrap.dedent("""
    import runpy, sys
    from central_module import config
    config.NLP["engine"] = "rules"        # deterministic: no DistilBERT load
    sys.argv = ["central_module"] + {args!r}
    runpy.run_module("central_module", run_name="__main__")
""")


def cli(args, stdin="", tmp_path=None):
    return run_py(CLI.format(args=args), stdin,
                  env_extra={"MMCTRL_DATA": str(tmp_path)} if tmp_path else None)


def test_cli_once_dry_run(tmp_path):
    proc = cli(["--once", "open chrome", "--dry-run"], tmp_path=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "[dry-run] open_app {'app': 'chrome'}" in proc.stdout
    usage = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
    assert usage["app_counts"] == {"chrome": 1}


def test_cli_once_unknown(tmp_path):
    proc = cli(["--once", "sing me a song", "--dry-run"], tmp_path=tmp_path)
    assert proc.returncode == 0
    assert "didn't understand" in proc.stdout


def test_cli_simulate(tmp_path):
    proc = cli(["--simulate", "--dry-run"], stdin="volume up\nstats\nquit\n", tmp_path=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "[dry-run] volume_up" in proc.stdout
    assert "aliases learned" in proc.stdout


def test_cli_help():
    proc = cli(["--help"])
    assert proc.returncode == 0
    for flag in ("--simulate", "--dry-run", "--once", "--no-speech", "--gesture-stub"):
        assert flag in proc.stdout
