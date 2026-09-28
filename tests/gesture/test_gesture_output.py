"""Gesture output: gesture_fsm.ActionExecutor, and the engine's
ActionDispatcher / MacroDispatcher / cursor selection.

pynput is always the recorder from helpers.fakes (see the root conftest), so
these tests see every press and release without touching the real input
stack. The invariant under test throughout: nothing is ever left held down.
"""

import pytest

import gesture_fsm as g
from gesture_fsm import ActionEvent, ActionExecutor, validate_macro


def ops(rec):
    return rec.events


# ── gesture_fsm.ActionExecutor ─────────────────────────────────────────────────

@pytest.mark.parametrize("action, expected", [
    ("LEFT_CLICK", [("mouse.click", "Button.left", 1)]),
    ("RIGHT_CLICK", [("mouse.click", "Button.right", 1)]),
    ("MIDDLE_CLICK", [("mouse.click", "Button.middle", 1)]),
    ("DOUBLE_CLICK", [("mouse.click", "Button.left", 2)]),
    ("SCROLL_UP", [("mouse.scroll", 0, 1)]),          # a scroll starts slow …
    ("SCROLL_DOWN", [("mouse.scroll", 0, -1)]),
])
def test_mouse_actions(fake_input, action, expected):
    assert ActionExecutor().dispatch(action) is True
    assert ops(fake_input) == expected


def test_held_scroll_accelerates_then_restarts_after_a_pause(fake_input):
    # … and speeds up while the gesture is held: repeats less than 0.5 s
    # apart are one scroll, reaching 10 units per update after 3 s.
    ex = ActionExecutor()
    for i in range(9):
        ex.dispatch("SCROLL_DOWN", now=i * 0.4)
    units = [-e[2] for e in ops(fake_input)]
    assert units[0] == 1 and units[-1] == 10
    assert units == sorted(units)                     # never slows down mid-scroll
    ex.dispatch("SCROLL_DOWN", now=8 * 0.4 + 0.8)     # after a pause: a new scroll
    assert ops(fake_input)[-1] == ("mouse.scroll", 0, -1)


def test_named_macro_presses_then_releases_in_reverse(fake_input):
    assert ActionExecutor().dispatch("SHOW_DESKTOP") is True
    assert ops(fake_input) == [("key.press", "Key.cmd"), ("key.press", "d"),
                               ("key.release", "d"), ("key.release", "Key.cmd")]


def test_keyboard_macro_uses_the_event_chord(fake_input):
    ev = ActionEvent("KEYBOARD_MACRO", keys="Control + Shift + Escape")
    assert ActionExecutor().dispatch(ev) is True
    pressed = [e[1] for e in ops(fake_input) if e[0] == "key.press"]
    assert pressed == ["Key.ctrl", "Key.shift", "Key.esc"]


def test_media_macro(fake_input):
    ActionExecutor().dispatch("VOLUME_UP")
    assert ops(fake_input)[0] == ("key.press", "Key.volume_up")


def test_unknown_key_presses_nothing(fake_input, capsys):
    ev = ActionEvent("KEYBOARD_MACRO", keys="ctrl+banana")
    assert ActionExecutor().dispatch(ev) is False
    assert ops(fake_input) == []
    assert "unknown key 'banana'" in capsys.readouterr().out


def test_failure_mid_chord_releases_what_was_pressed(fake_input, monkeypatch):
    ex = ActionExecutor()
    real_press = ex._keyboard.press

    def press(key):
        if key == "l":
            raise RuntimeError("keyboard hook refused")
        real_press(key)
    monkeypatch.setattr(ex._keyboard, "press", press)
    assert ex.dispatch("LOCK_SCREEN") is False
    assert ops(fake_input) == [("key.press", "Key.cmd"), ("key.release", "Key.cmd")]


def test_drag_is_tracked_and_released(fake_input):
    ex = ActionExecutor()
    ex.dispatch("DRAG_START")
    assert ex._held == {"left"}
    ex.release_all()
    ex.release_all()                                    # idempotent
    assert ops(fake_input) == [("mouse.press", "Button.left"),
                               ("mouse.release", "Button.left")]


def test_context_manager_releases_on_exit(fake_input):
    with ActionExecutor() as ex:
        ex.dispatch("DRAG_START")
    assert ops(fake_input)[-1] == ("mouse.release", "Button.left")


def test_unknown_action_is_refused_even_in_dry_run(fake_input):
    assert ActionExecutor().dispatch("TELEPORT") is False
    assert ActionExecutor(dry_run=True).dispatch("TELEPORT") is False


def test_dry_run_reports_and_touches_nothing(fake_input, capsys):
    ex = ActionExecutor(dry_run=True)
    assert ex.dispatch("COPY") is True and ex.dispatch("LEFT_CLICK") is True
    assert ops(fake_input) == []
    out = capsys.readouterr().out
    assert "[dry-run] COPY (ctrl+c)" in out and "[dry-run] LEFT_CLICK" in out


def test_output_failure_is_reported_once(fake_input, capsys):
    fake_input.fail_on.add("mouse.click")
    ex = ActionExecutor()
    assert ex.dispatch("LEFT_CLICK") is False
    assert ex.dispatch("LEFT_CLICK") is False
    assert capsys.readouterr().out.count("output unavailable") == 1


@pytest.mark.parametrize("spec, ok", [
    ("ctrl+c", True), ("win+shift+s", True), ("Alt + F4", True), ("pgdn", True),
    ("ctrl+banana", False), ("", False), ("+", False),
])
def test_validate_macro(fake_input, spec, ok):
    assert validate_macro(spec) is ok


def test_every_named_macro_resolves(fake_input):
    for chord in g.ACTION_MACROS.values():
        assert validate_macro(chord), chord


# ── Engine: ActionDispatcher ───────────────────────────────────────────────────

class RecordingCursor:
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def _do(self, name, *args, **kwargs):
        if name in self.fail:
            raise OSError(f"{name} failed")
        self.calls.append((name, *args, *kwargs.values()))

    def click(self, button="left", count=1):
        self._do("click", button, count)

    def press(self, button="left"):
        self._do("press", button)

    def release(self, button="left"):
        self._do("release", button)

    def move(self, x, y):
        self._do("move", x, y)


def test_dispatcher_click_and_drag(hc, capsys):
    cursor = RecordingCursor()
    d = hc.ActionDispatcher(cursor)
    d.dispatch("LEFT_CLICK", 1.0)
    d.dispatch("DRAG_START", 2.0)
    d.dispatch("DRAG_START", 2.1)                      # already down: ignored
    d.dispatch("DRAG_STOP", 3.0)
    d.dispatch("DRAG_STOP", 3.1)                       # not down: ignored
    assert cursor.calls == [("click", "left", 1), ("press", "left"), ("release", "left")]
    assert (d.click_count, d.drag_count, d.dragging) == (1, 1, False)
    assert (d.last_action, d.last_action_time) == ("DRAG_STOP", 3.0)


def test_dispatcher_standalone_double_click_sends_two_clicks(hc):
    # A DOUBLE_CLICK with no click before it (a rule bound to DOUBLE_CLICK)
    # is a whole double: two presses, not left to the OS double-click timer.
    cursor = RecordingCursor()
    hc.ActionDispatcher(cursor).dispatch("DOUBLE_CLICK", 0.0)
    assert cursor.calls == [("click", "left", 2)]


def test_dispatcher_promoted_double_click_completes_an_os_double(hc):
    # The FSM emits LEFT_CLICK then, within the window, DOUBLE_CLICK from the
    # same rule in place of the second click — the OS already has one click,
    # so one more completes the double (two would read as a triple).
    cursor = RecordingCursor()
    d = hc.ActionDispatcher(cursor)
    d.dispatch(ActionEvent("LEFT_CLICK", rule_id="r1"), 0.0)
    d.dispatch(ActionEvent("DOUBLE_CLICK", rule_id="r1"), 0.2)
    assert cursor.calls == [("click", "left", 1), ("click", "left", 1)]
    d.dispatch(ActionEvent("DOUBLE_CLICK", rule_id="r1"), 5.0)   # partner used up
    assert cursor.calls[-1] == ("click", "left", 2)


def test_failed_press_does_not_claim_a_drag(hc, capsys):
    d = hc.ActionDispatcher(RecordingCursor(fail={"press"}))
    d.dispatch("DRAG_START", 0.0)
    assert not d.dragging and d.last_action is None
    assert "cursor output unavailable" in capsys.readouterr().out


def test_failed_release_stays_pending_for_the_shutdown_path(hc):
    cursor = RecordingCursor(fail={"release"})
    d = hc.ActionDispatcher(cursor)
    d.dispatch("DRAG_START", 0.0)
    assert d.release() is False and d.dragging          # still owed a release
    cursor.fail.clear()
    assert d.release() is True and not d.dragging


def test_extended_actions_go_through_the_fsm_executor(hc, fake_input):
    d = hc.ActionDispatcher(RecordingCursor())
    d.dispatch("RIGHT_CLICK", 0.0)
    d.dispatch("COPY", 0.1)
    assert fake_input.events[0] == ("mouse.click", "Button.right", 1)
    assert ("key.press", "c") in fake_input.events
    assert d.last_action == "COPY"


def test_unknown_action_is_not_recorded(hc):
    d = hc.ActionDispatcher(RecordingCursor())
    d.dispatch("TELEPORT", 0.0)
    assert d.last_action is None


def test_make_cursor_prefers_pynput(hc, fake_input, monkeypatch):
    monkeypatch.setattr(hc, "CURSOR_BACKEND", "auto")
    cursor = hc.make_cursor(backend=object())
    assert cursor.name == "pynput"
    cursor.move(10.7, 20.2)
    cursor.click("left", 2)
    cursor.press("right")
    cursor.release("right")
    assert fake_input.events == [("mouse.move", (10, 20)), ("mouse.click", "Button.left", 2),
                                 ("mouse.press", "Button.right"),
                                 ("mouse.release", "Button.right")]


def test_make_cursor_falls_back_to_native(hc, monkeypatch, capsys):
    import sys
    monkeypatch.setitem(sys.modules, "pynput.mouse", None)
    monkeypatch.setattr(hc, "CURSOR_BACKEND", "auto")
    backend = RecordingCursor()
    backend.move_cursor = lambda x, y: backend.calls.append(("move", x, y))
    cursor = hc.make_cursor(backend)
    assert cursor.name == "ctypes"
    cursor.move(1.9, 2.1)
    cursor.click("left", 1)
    assert backend.calls == [("move", 1, 2), ("click", "left", 1)]
    assert "falling back to ctypes" in capsys.readouterr().out


def test_forced_pynput_without_pynput_is_an_error(hc, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pynput.mouse", None)
    monkeypatch.setattr(hc, "CURSOR_BACKEND", "pynput")
    with pytest.raises(RuntimeError, match="pynput is unusable"):
        hc.make_cursor(object())


# ── Engine: MacroDispatcher (YOLO semantic labels → OS macros) ─────────────────

@pytest.fixture
def macros(hc, fake_input):
    return hc.MacroDispatcher({"palm": "show_desktop", "fist": "lock_screen",
                               "ok": "no_such_macro"}, cooldown=2.0, min_score=0.6)


def test_macro_is_edge_triggered(macros, fake_input):
    assert macros.update("palm", 0.9, 0.0) is True
    assert macros.update("palm", 0.9, 5.0) is False            # still held
    assert macros.update(None, 0.0, 6.0) is False               # released
    assert macros.update("palm", 0.9, 7.0) is True
    assert macros.fired_count == 2 and macros.last_macro == "show_desktop"
    assert fake_input.events[:4] == [("key.press", "Key.cmd"), ("key.press", "d"),
                                     ("key.release", "d"), ("key.release", "Key.cmd")]


def test_macro_needs_confidence(macros):
    assert macros.update("palm", 0.4, 0.0) is False
    assert macros.update("palm", 0.7, 0.1) is True


def test_macro_cooldown_survives_a_flicker(macros):
    assert macros.update("palm", 0.9, 0.0) is True
    macros.update("no_gesture", 0.9, 0.5)
    assert macros.update("palm", 0.9, 1.0) is False             # within 2 s
    assert macros.update("fist", 0.9, 1.5) is False             # cooldown is global
    assert macros.update("fist", 0.9, 2.5) is True


def test_unknown_macro_name_fires_nothing(macros, fake_input, capsys):
    assert macros.update("ok", 0.9, 0.0) is False
    assert fake_input.events == []
    assert "no such macro" in capsys.readouterr().out


def test_macro_keyboard_failure_releases_pressed_keys(macros, fake_input, monkeypatch):
    real = macros._keyboard.press

    def press(key):
        if key == "l":
            raise RuntimeError("blocked")
        real(key)
    monkeypatch.setattr(macros._keyboard, "press", press)
    assert macros.update("fist", 0.9, 0.0) is False
    assert fake_input.events == [("key.press", "Key.cmd"), ("key.release", "Key.cmd")]


def test_macro_gestures(macros):
    assert set(macros.macro_gestures) == {"palm", "fist", "ok"}


# ── Engine: GestureProbe ───────────────────────────────────────────────────────

def test_probe_is_bounded_and_drains(hc):
    probe = hc.GestureProbe(capacity=3)
    for i in range(5):
        probe.record({"i": i})
    assert probe.dropped == 2
    assert [r["i"] for r in probe.drain()] == [2, 3, 4]
    assert probe.drain() == []
