"""runners/ and __main__ — the child-process protocol and the CLI, end to end.

Each test launches a real interpreter. The speech runner gets a stand-in
`speech_manager` injected before it imports, and the gesture runner a
stand-in engine and Gesture Studio, so the protocol (ready, events, control
ops, EOF handling, stdout hygiene) is tested without a microphone, a camera,
a window or a single model.
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


# ── gesture_stub (stand-in child) ──────────────────────────────────────────────

GESTURE = ("import runpy; "
           "runpy.run_module('central_module.runners.gesture_stub', run_name='__main__')")


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

    def report_inherited_stdin():
        import subprocess
        got = subprocess.run([sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        print("INHERITED_STDIN=" + got)
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
            if MODE == "inherit-check":
                report_inherited_stdin()
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


def test_speech_runner_keeps_its_control_pipe_private():
    # Native libraries and child processes must not see the control pipe as
    # stdin: on Windows a pending read on it blocks them until the parent
    # writes, which froze numpy imports and audio device queries.
    stdin = json.dumps({"op": "ping"}) + "\n" + json.dumps({"op": "shutdown"}) + "\n"
    proc = speech("inherit-check", stdin)
    assert proc.returncode == 0, proc.stderr
    assert "INHERITED_STDIN=''" in proc.stderr          # NUL, not our pipe
    assert "pong" in [e["type"] for e in json_lines(proc.stdout)]   # control still works


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


# ── gesture_runner (the real gesture child) ────────────────────────────────────
# Stand-ins for hand_cursor_2 and app are injected before the runner imports
# them; gesture_events is the real builder, so the wire format is the real one.

FAKE_GESTURE = textwrap.dedent("""
    import runpy, sys, threading, types
    MODE = __MODE__

    def report_inherited_stdin():
        import subprocess
        got = subprocess.run([sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        print("INHERITED_STDIN=" + got)
    sys.path.insert(0, "Mousegesture")
    import gesture_events as ge

    class Engine:
        def __init__(self, *, enable_monitor, verbose, event_sink):
            self.sink, self.status, self.closed = event_sink, {"error": None}, False
            if MODE == "slow-load":                       # the real engine takes ~30 s
                threading.Event().wait(1.0)
            self._running = False
        @property
        def running(self):
            return self._running
        def start(self, camera):
            if MODE == "camera-fails":
                self.status["error"] = "camera 7 is busy"
                return False
            print("CAMERA", camera)                       # stray print -> stderr
            if MODE == "inherit-check":
                report_inherited_stdin()
            self._running = True
            self.sink(ge.ready(camera_index=camera, semantic="ready"))
            self.sink(ge.pointing_started(hand="right"))
            if MODE == "crash":
                self.sink(ge.fatal(code=ge.FATAL_TRACKING_LOOP, message="boom"))
                self.status["error"] = "boom"
                self._running = False
            return True
        def close(self):
            print("ENGINE CLOSED")

    hc = types.ModuleType("hand_cursor_2")
    hc.HandTrackerEngine = Engine
    sys.modules["hand_cursor_2"] = hc

    class Studio:
        def __init__(self):
            self.jobs = []
            self.closed = threading.Event()
        def after(self, ms, fn):
            self.jobs.append(fn)
        def mainloop(self):
            # The user connects a camera: the app builds its engine with
            # event_sink=self.event_queue.put_nowait, as app.py does.
            self.event_queue.put_nowait(ge.ready(camera_index=0, semantic="loading"))
            self.event_queue.put_nowait(ge.gesture_fired(gesture="fist", action="LEFT_CLICK"))
            if MODE == "user-closes":
                print("WINDOW CLOSED BY USER")
                return
            while not self.closed.is_set():               # Tk's loop, cut down
                for fn in self.jobs[:]:
                    self.jobs.remove(fn)
                    fn()
                self.closed.wait(0.05)
        def _on_close(self):
            print("ON_CLOSE")
            self.closed.set()

    app = types.ModuleType("app")
    app.GestureStudio = Studio
    sys.modules["app"] = app
    sys.argv = ["gesture_runner"] + __ARGS__
    runpy.run_module("central_module.runners.gesture_runner", run_name="__main__")
""")


def gesture(mode, args, stdin):
    code = FAKE_GESTURE.replace("__MODE__", repr(mode)).replace("__ARGS__", repr(args))
    return run_py(code, stdin)


def gesture_live(mode, args, timeout=60):
    """Run the gesture runner the way central does: the control pipe stays
    open, and ping + shutdown are sent only once the engine reports ready or
    fatal. (A shutdown that arrives first is honoured before the camera opens.)"""
    import os
    code = FAKE_GESTURE.replace("__MODE__", repr(mode)).replace("__ARGS__", repr(args))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8")
    lines = []
    for line in proc.stdout:                          # until ready, fatal or exit
        lines.append(line)
        if json.loads(line)["type"] in ("ready", "fatal"):
            break
    if lines and json.loads(lines[-1])["type"] == "ready":
        try:
            proc.stdin.write(json.dumps({"op": "ping"}) + "\n" +
                             json.dumps({"op": "shutdown"}) + "\n")
            proc.stdin.flush()
        except OSError:                                # it already exited (crash)
            pass
    out, err = proc.communicate(timeout=timeout)
    return subprocess.CompletedProcess(proc.args, proc.returncode,
                                       "".join(lines) + out, err)


def test_gesture_runner_headless_streams_engine_events():
    proc = gesture_live("ok", ["--headless", "--camera", "2"])
    assert proc.returncode == 0, proc.stderr
    events = json_lines(proc.stdout)                    # also proves stdout hygiene
    assert all(e["module"] == "gesture" and e["version"] == 1 for e in events)
    assert [e["type"] for e in events] == ["ready", "pointing_started", "pong"]
    assert events[0]["payload"]["camera_index"] == 2
    assert "CAMERA 2" in proc.stderr and "ENGINE CLOSED" in proc.stderr


def test_gesture_runner_keeps_its_control_pipe_private():
    proc = gesture_live("inherit-check", ["--headless"])
    assert proc.returncode == 0, proc.stderr
    assert "INHERITED_STDIN=''" in proc.stderr
    assert "pong" in [e["type"] for e in json_lines(proc.stdout)]


def test_gesture_runner_headless_reports_a_camera_that_will_not_open():
    proc = gesture_live("camera-fails", ["--headless", "--camera", "7"])
    assert proc.returncode == 1
    (fatal,) = json_lines(proc.stdout)
    assert fatal["type"] == "fatal"
    assert fatal["payload"] == {"code": "camera_failed", "message": "camera 7 is busy"}
    assert "ENGINE CLOSED" in proc.stderr               # camera released anyway


def test_gesture_runner_headless_exits_nonzero_when_tracking_crashes():
    # The engine reports its own fatal; a non-zero exit gets it restarted.
    proc = gesture_live("crash", ["--headless"])
    assert proc.returncode == 1
    kinds = [e["type"] for e in json_lines(proc.stdout)]
    assert "fatal" in kinds                             # a reply to our ping may follow it


def test_gesture_runner_headless_does_not_open_the_camera_after_shutdown():
    # Central gave up on us while the engine was loading: leave the camera alone.
    proc = gesture("slow-load", ["--headless"], stdin=json.dumps({"op": "shutdown"}) + "\n")
    assert proc.returncode == 0, proc.stderr
    assert "CAMERA" not in proc.stderr and json_lines(proc.stdout) == []


def test_gesture_runner_gui_wires_the_app_and_closes_it_on_shutdown():
    proc = gesture("ok", [], json.dumps({"op": "shutdown"}) + "\n")
    assert proc.returncode == 0, proc.stderr
    events = json_lines(proc.stdout)
    assert [e["type"] for e in events] == ["ready", "gesture_fired"]
    assert events[1]["payload"]["gesture"] == "fist"
    assert "ON_CLOSE" in proc.stderr                    # the app's own close path


def test_gesture_runner_gui_exits_cleanly_when_the_user_closes_the_window():
    proc = gesture("user-closes", [], stdin="")
    assert proc.returncode == 0                         # clean: not restarted
    assert [e["type"] for e in json_lines(proc.stdout)] == ["ready", "gesture_fired"]


def test_gesture_runner_shuts_down_when_the_parent_disappears():
    proc = gesture("ok", [], stdin="")                  # immediate EOF
    assert proc.returncode == 0
    assert "ON_CLOSE" in proc.stderr


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
    for flag in ("--simulate", "--dry-run", "--once", "--no-speech", "--no-gesture",
                 "--gesture-headless", "--gesture-stub"):
        assert flag in proc.stdout
