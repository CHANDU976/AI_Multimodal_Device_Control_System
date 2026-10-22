"""manager.py — the decision maker: macros → confirmation → dictation →
intents → execution → learning, plus child-event routing."""

import functools

import pytest

from central_module import config, manager as manager_mod
from central_module.learning.adaptation import BehaviorModel
from central_module.learning.preferences import PreferenceStore
from helpers.fakes import recording_adapter_class

RecordingAdapter = recording_adapter_class()


class FakeSupervisor:
    def __init__(self, on_event=None):
        self.specs, self.sent = [], []
        self.alive = True
        self.started = self.stopped = False

    def add(self, spec):
        self.specs.append(spec)

    def send(self, name, obj):
        self.sent.append((name, obj))
        return self.alive

    def start_all(self):
        self.started = True

    def stop_all(self):
        self.stopped = True


@pytest.fixture
def make_manager(tmp_path, monkeypatch):
    prefs_path, usage_path = str(tmp_path / "prefs.json"), str(tmp_path / "usage.json")
    monkeypatch.setattr(manager_mod, "PreferenceStore",
                        functools.partial(PreferenceStore, prefs_path))
    monkeypatch.setattr(manager_mod, "BehaviorModel",
                        functools.partial(BehaviorModel, usage_path))
    monkeypatch.setattr(manager_mod, "ProcessSupervisor", FakeSupervisor)
    # App resolution must not depend on what this machine has on PATH.
    from central_module.actions import executor
    monkeypatch.setattr(executor.shutil, "which", lambda name: None)

    def _make(**kwargs):
        adapter = RecordingAdapter()
        monkeypatch.setattr(manager_mod, "get_adapter", lambda: adapter)
        mgr = manager_mod.CentralManager(**kwargs)
        mgr.adapter = adapter
        return mgr

    return _make


@pytest.fixture
def mgr(make_manager):
    return make_manager()


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(manager_mod.time, "monotonic", lambda: now[0])
    return now


def said(capsys):
    return [line[2:] for line in capsys.readouterr().out.splitlines() if line.startswith("» ")]


# ── Core pipeline ──────────────────────────────────────────────────────────────

def test_command_is_executed_and_learned(mgr, capsys):
    mgr.handle_text("open chrome")
    assert mgr.adapter.calls == [("open_app", "chrome-cmd")]
    assert said(capsys) == ["✓ opened chrome"]
    stats = mgr.behavior._data["phrase_stats"]["open chrome"]
    assert (stats["intent"], stats["hits"], stats["source"]) == ("open_app", 1, "exact")


def test_fuzzy_match_is_reported_with_its_source(mgr, capsys):
    mgr.handle_text("volum up")
    assert mgr.adapter.calls == [("volume_change", 2)]
    assert said(capsys) == ["✓ volume up  (via fuzzy)"]


def test_failed_action_is_reported_and_recorded(mgr, capsys):
    from central_module.actions.os_adapter import ActionError

    def fail():
        raise ActionError("LockWorkStation failed")
    mgr.adapter.lock = fail
    mgr.handle_text("lock screen")
    assert said(capsys) == ["✗ LockWorkStation failed"]
    assert mgr.behavior._data["phrase_stats"]["lock screen"]["fails"] == 1


def test_unknown_utterance_is_recorded_not_guessed(mgr, capsys):
    mgr.handle_text("do a barrel roll")
    assert mgr.adapter.calls == []
    assert said(capsys) == ["didn't understand 'do barrel roll'"]
    assert mgr.behavior.suggestions() == [("do barrel roll", 1)]


@pytest.mark.parametrize("text", ["", "   ", "?!"])
def test_empty_utterances_are_ignored(mgr, capsys, text):
    mgr.handle_text(text)
    assert said(capsys) == [] and mgr.adapter.calls == []


def test_one_bad_utterance_never_takes_the_manager_down(mgr, capsys, monkeypatch):
    def explode(text):
        raise RuntimeError("engine bug")
    monkeypatch.setattr(mgr.engine, "parse", explode)
    mgr.handle_text("open chrome")                 # must not raise
    assert "something went wrong" in said(capsys)[0]


def test_custom_command_expands_before_parsing(mgr):
    mgr.prefs.set("custom_commands", {"battle stations": "open code"})
    mgr.handle_text("Battle stations!")
    assert mgr.adapter.calls == [("open_app", "code")]


# ── Confirmation of destructive intents ────────────────────────────────────────

def test_shutdown_waits_for_yes(mgr, capsys, clock):
    mgr.handle_text("shut down")
    assert mgr.adapter.calls == []
    assert "say 'yes'" in said(capsys)[0]
    mgr.handle_text("yes")
    assert mgr.adapter.calls == [("power", "shutdown", 3)]


@pytest.mark.parametrize("word", sorted(config.CONFIRM_WORDS))
def test_every_confirm_word_fires(mgr, clock, word):
    mgr.handle_text("restart")
    mgr.handle_text(word)
    assert mgr.adapter.calls == [("power", "restart", 3)]


def test_cancel_word_disarms(mgr, capsys, clock):
    mgr.handle_text("restart")
    mgr.handle_text("cancel")
    mgr.handle_text("yes")                          # nothing armed any more
    assert mgr.adapter.calls == []
    assert "restart cancelled" in said(capsys)


def test_any_other_utterance_disarms_and_is_processed(mgr, capsys, clock):
    mgr.handle_text("shut down")
    mgr.handle_text("open chrome")
    assert mgr.adapter.calls == [("open_app", "chrome-cmd")]
    lines = said(capsys)
    assert "shutdown cancelled (no confirmation)" in lines
    mgr.handle_text("yes")
    assert ("power", "shutdown", 3) not in mgr.adapter.calls


def test_confirmation_expires(mgr, capsys, clock):
    mgr.handle_text("shut down")
    clock[0] += config.PENDING_TIMEOUT_S + 0.1
    mgr.handle_text("yes")
    assert mgr.adapter.calls == []


# ── Dictation mode ─────────────────────────────────────────────────────────────

def test_dictation_types_everything_until_stopped(mgr, capsys):
    mgr.handle_text("start dictation")
    mgr.handle_text("Open chrome is what I said")
    mgr.handle_text("stop dictation")
    mgr.handle_text("open chrome")
    assert mgr.adapter.calls == [("type_text", "Open chrome is what I said "),
                                 ("open_app", "chrome-cmd")]


@pytest.mark.parametrize("phrase", sorted(config.DICTATION_START))
def test_every_dictation_start_phrase(mgr, phrase):
    mgr.handle_text(phrase)
    assert mgr._dictation_on


def test_dictation_via_learned_alias_reaches_the_manager_intent(mgr):
    # Exact trigger phrases short-circuit before parsing; a learned alias is
    # how the start_dictation *intent* (the manager branch) gets exercised.
    mgr.behavior._data["aliases"]["begin typing"] = {"intent": "start_dictation", "slots": {}}
    mgr.handle_text("begin typing")
    assert mgr._dictation_on
    mgr.handle_text("end dictation")
    assert not mgr._dictation_on


def test_polite_trigger_phrase_still_switches_mode(mgr):
    mgr.handle_text("could you start dictating please")
    assert mgr._dictation_on


def test_dry_run_dictation_does_not_type(make_manager, capsys):
    mgr = make_manager(dry_run=True)
    mgr.handle_text("start dictation")
    mgr.handle_text("hello")
    assert mgr.adapter.calls == []
    assert "[dry-run] would type: 'hello'" in said(capsys)


def test_dictation_typing_failure_is_reported(mgr, capsys):
    from central_module.actions.os_adapter import ActionError

    def fail(text):
        raise ActionError("typing blocked")
    mgr.adapter.type_text = fail
    mgr.handle_text("start dictation")
    mgr.handle_text("hello")
    assert "typing failed: typing blocked" in said(capsys)


# ── Manager-level intents ──────────────────────────────────────────────────────

def test_language_switch_is_sent_to_speech_and_remembered(mgr, capsys):
    mgr.handle_text("switch language to hindi")
    assert mgr.supervisor.sent == [("speech", {"op": "set_language", "lang": "hi-in"})]
    assert mgr.prefs.get("language") == "hi-in"
    assert said(capsys) == ["language switched to hindi"]


def test_unknown_language_lists_the_options(mgr, capsys):
    mgr.handle_text("switch language to klingon")
    assert mgr.supervisor.sent == []
    assert "unknown language 'klingon'" in said(capsys)[0]


def test_language_switch_without_speech_child(mgr, capsys):
    mgr.supervisor.alive = False
    mgr.handle_text("change language to english")
    assert mgr.prefs.get("language") == "en-in"
    assert said(capsys) == ["speech module is not running"]


def test_stray_confirm_is_harmless(mgr, capsys):
    mgr.handle_text("yes")
    assert mgr.adapter.calls == [] and said(capsys) == []


# ── Child events ───────────────────────────────────────────────────────────────

def envelope(kind, module="speech", version=1, **payload):
    """A child message in the shared protocol (docs/ipc-protocol.md)."""
    return {"type": kind, "module": module, "timestamp": 1790000000.0,
            "version": version, "payload": payload}


def test_command_event_runs_the_pipeline(mgr, capsys):
    mgr._dispatch("speech", envelope("command", text=" open chrome "))
    assert mgr.adapter.calls == [("open_app", "chrome-cmd")]
    assert said(capsys)[0] == "heard: 'open chrome'"


def test_dictation_event_runs_the_pipeline(mgr):
    mgr._dispatch("speech", envelope("dictation", text="type hello world"))
    assert mgr.adapter.calls == [("type_text", "hello world")]


def test_mapped_gesture_runs_its_command(mgr, capsys):
    mgr.prefs.set("gesture_actions", {"fist": "pause"})
    mgr._dispatch("gesture", envelope("gesture_fired", "gesture", gesture="fist"))
    assert mgr.adapter.calls == [("media", "play_pause")]


def test_unmapped_gesture_does_nothing(mgr, capsys):
    mgr._dispatch("gesture", envelope("gesture_fired", "gesture", gesture="wave"))
    assert mgr.adapter.calls == [] and said(capsys) == []


@pytest.mark.parametrize("msg, expected", [
    (envelope("ready", status="ready"), "speech module ready"),
    ({"type": "lifecycle", "state": "exited", "returncode": 1}, "speech exited 1"),
    ({"type": "lifecycle", "state": "gave_up"}, "speech gave_up"),
    (envelope("fatal", code="start_failed", message="no mic"), "speech failed: no mic"),
    (envelope("fatal", code="start_failed", message=None), "speech failed: start_failed"),
])
def test_status_events_are_announced(mgr, capsys, msg, expected):
    mgr._dispatch("speech", msg)
    assert said(capsys) == [expected]


@pytest.mark.parametrize("msg", [
    {"type": "lifecycle", "state": "started"}, envelope("partial", text="op"),
    envelope("device", metadata={}), envelope("error", metadata={}),
    envelope("pong"), {"type": "log", "line": "x"}, {},
    envelope("heartbeat", "gesture", status="running"),
    envelope("pointing_started", "gesture", hand="right"),
    envelope("pointing_stopped", "gesture", reason="hand_lost"),
    envelope("drag_started", "gesture"), envelope("drag_stopped", "gesture"),
])
def test_quiet_events(mgr, capsys, caplog, msg):
    mgr._dispatch("speech", msg)
    assert said(capsys) == [] and mgr.adapter.calls == []
    assert "unhandled message type" not in caplog.text


def test_an_unknown_message_type_is_logged_not_acted_on(mgr, capsys, caplog):
    mgr._dispatch("speech", envelope("mystery"))
    assert said(capsys) == [] and mgr.adapter.calls == []
    assert "unhandled message type 'mystery'" in caplog.text


def test_an_unexpected_protocol_version_is_reported(mgr, caplog):
    mgr._dispatch("speech", envelope("command", version=2, text="open chrome"))
    assert "protocol version 2" in caplog.text
    assert mgr.adapter.calls == [("open_app", "chrome-cmd")]     # still read


def test_partials_shown_in_verbose_mode(make_manager, capsys):
    mgr = make_manager(verbose=True)
    mgr._dispatch("speech", envelope("partial", text="open chr"))
    assert "? open chr" in capsys.readouterr().out


def test_event_queue_overflow_drops_instead_of_blocking(mgr, caplog):
    for _ in range(mgr._events.maxsize + 5):
        mgr._enqueue("speech", {"type": "pong"})
    assert mgr._events.full()
    assert "event queue full" in caplog.text


# ── Children & lifecycle ───────────────────────────────────────────────────────

def test_default_children(mgr):
    mgr._build_children()
    speech, gesture = mgr.supervisor.specs
    assert (speech.name, gesture.name) == ("speech", "gesture")
    # Default: the Gesture Studio window, which stays closed if the user
    # closes it, with time to answer an "unsaved mappings?" prompt.
    assert gesture.argv == config.GESTURE_CHILD["argv"]
    assert gesture.restart_on_clean_exit is False
    assert gesture.grace_s == config.GESTURE_CHILD["gui_grace_s"]


def test_no_speech_flag(make_manager):
    mgr = make_manager(enable_speech=False)
    mgr._build_children()
    assert [s.name for s in mgr.supervisor.specs] == ["gesture"]


def test_no_gesture_flag(make_manager):
    mgr = make_manager(enable_gesture=False)
    mgr._build_children()
    assert [s.name for s in mgr.supervisor.specs] == ["speech"]


def test_gesture_section_without_enabled_keeps_the_default(mgr):
    mgr.prefs.set("gesture", {"mode": "gui"})          # hand-edited, "enabled" missing
    mgr._build_children()
    assert [s.name for s in mgr.supervisor.specs] == ["speech", "gesture"]


def test_gesture_disabled_in_preferences(mgr):
    mgr.prefs.set("gesture.enabled", False)
    mgr._build_children()
    assert [s.name for s in mgr.supervisor.specs] == ["speech"]


def test_headless_gesture_uses_the_saved_camera(mgr):
    mgr.prefs.set("gesture.mode", "headless")
    mgr.prefs.set("gesture.camera", 2)
    mgr._build_children()
    gesture = mgr.supervisor.specs[1]
    assert gesture.argv == [*config.GESTURE_CHILD["argv"], "--headless", "--camera", "2"]
    assert gesture.restart_on_clean_exit is True        # no window to close


def test_headless_flag_overrides_the_preference(make_manager):
    mgr = make_manager(gesture_mode="headless")
    mgr._build_children()
    assert "--headless" in mgr.supervisor.specs[1].argv


def test_configured_gesture_app_is_supervised(mgr):
    mgr.prefs.set("gesture", {"enabled": True, "command": ["python", "hand.py"], "cwd": "/g"})
    mgr._build_children()
    gesture = mgr.supervisor.specs[1]
    assert (gesture.name, gesture.argv, gesture.cwd) == ("gesture", ["python", "hand.py"], "/g")


def test_gesture_stub_when_forced(make_manager):
    mgr = make_manager(force_gesture_stub=True)
    mgr._build_children()
    assert mgr.supervisor.specs[1].argv == config.GESTURE_STUB["argv"]


def gesture_msg(kind, **payload):
    return {"type": kind, "module": "gesture", "timestamp": 1790000000.5,
            "version": 1, "payload": payload}


def test_gesture_state_follows_the_gesture_child(mgr, capsys):
    assert mgr.gesture_state["ready"] is False
    mgr._dispatch("gesture", gesture_msg("ready", status="ready", camera_index=1))
    assert said(capsys) == ["gesture module ready (camera 1)"]
    mgr._dispatch("gesture", gesture_msg("pointing_started", hand="right"))
    mgr._dispatch("gesture", gesture_msg("drag_started"))
    state = mgr.gesture_state
    assert (state["ready"], state["camera_index"], state["pointing"], state["dragging"]) == \
        (True, 1, True, True)
    assert state["last_seen"] == 1790000000.5
    mgr._dispatch("gesture", gesture_msg("drag_stopped", reason="gesture"))
    mgr._dispatch("gesture", gesture_msg("pointing_stopped", reason="hand_lost"))
    assert (mgr.gesture_state["pointing"], mgr.gesture_state["dragging"]) == (False, False)


def test_heartbeat_corrects_the_gesture_state(mgr):
    # A missed transition is repaired by the next heartbeat.
    mgr._dispatch("gesture", gesture_msg("heartbeat", status="running",
                                         pointing=True, dragging=False))
    assert mgr.gesture_state["pointing"] is True and mgr.gesture_state["ready"] is True


@pytest.mark.parametrize("msg", [
    gesture_msg("fatal", code="tracking_loop_failed", message="boom"),
    {"type": "lifecycle", "child": "gesture", "state": "exited", "returncode": 0},
])
def test_gesture_state_resets_when_the_child_stops(mgr, msg):
    mgr._dispatch("gesture", gesture_msg("pointing_started"))
    mgr._dispatch("gesture", msg)
    assert mgr.gesture_state["pointing"] is False and mgr.gesture_state["ready"] is False


def test_speech_events_leave_the_gesture_state_alone(mgr):
    mgr._dispatch("speech", {"type": "ready", "module": "speech", "timestamp": 1.0,
                             "version": 1, "payload": {"status": "ready"}})
    assert mgr.gesture_state["ready"] is False


def test_start_restores_saved_language(mgr):
    mgr.prefs.set("language", "en-us")
    mgr.start()
    assert mgr.supervisor.started
    assert mgr.supervisor.sent == [("speech", {"op": "set_language", "lang": "en-us"})]


def test_start_with_default_language_sends_nothing(mgr):
    mgr.start()
    assert mgr.supervisor.sent == []


def test_stop_flushes_learning(mgr, tmp_path):
    mgr.behavior.record_app("chrome")
    mgr.stop()
    assert mgr.supervisor.stopped
    assert (tmp_path / "usage.json").exists()


def test_missing_adapter_drops_to_dry_run(make_manager, monkeypatch, capsys):
    def no_adapter():
        raise OSError("no display")
    monkeypatch.setattr(manager_mod, "get_adapter", no_adapter)
    mgr = manager_mod.CentralManager()
    mgr.handle_text("open chrome")
    assert "[dry-run] open_app" in said(capsys)[0]


def test_simulate_mode_reads_until_quit(mgr, monkeypatch, capsys):
    lines = iter(["open chrome", "", "stats", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    mgr.run_simulated()
    out = capsys.readouterr().out
    assert "aliases learned" in out
    assert mgr.adapter.calls == [("open_app", "chrome-cmd")]


def test_simulate_mode_ends_on_eof(mgr, monkeypatch):
    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    mgr.run_simulated()
