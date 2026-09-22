"""actions/windows_adapter.py — SendInput event construction (Windows only).

SendInput itself is replaced by a recorder, so these tests inspect exactly
which INPUT structures would have been injected without injecting any.
"""

import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="WindowsAdapter binds user32.dll at import")

if sys.platform == "win32":
    from central_module.actions import windows_adapter as wa
    from central_module.actions.os_adapter import ActionError
    from central_module.nlp.intents import KEY_ALIASES
    from helpers.fakes import RecordedRun

    REAL_SEND = wa._send          # captured before the suite-wide guard


@pytest.fixture
def sent(monkeypatch):
    batches = []
    monkeypatch.setattr(wa, "_send", lambda *inputs: batches.append(inputs))
    monkeypatch.setattr(wa.time, "sleep", lambda s: None)
    return batches


@pytest.fixture
def adapter():
    return wa.WindowsAdapter()


def keys(batch):
    """[(vk, is_up, is_extended), …] for one SendInput batch."""
    out = []
    for inp in batch:
        assert inp.type == wa._INPUT_KEYBOARD
        ki = inp.u.ki
        out.append((ki.wVk, bool(ki.dwFlags & wa._KEYEVENTF_KEYUP),
                    bool(ki.dwFlags & wa._KEYEVENTF_EXTENDEDKEY)))
    return out


def test_every_canonical_key_has_a_virtual_key_code():
    missing = set(KEY_ALIASES.values()) - set(wa._VK)
    assert not missing, f"keys the Windows adapter cannot press: {sorted(missing)}"


def test_key_tap_is_down_then_up(adapter, sent):
    adapter.key_tap("enter")
    assert keys(sent[0]) == [(0x0D, False, False), (0x0D, True, False)]


def test_navigation_keys_carry_the_extended_flag(adapter, sent):
    adapter.key_tap("left")
    assert keys(sent[0]) == [(0x25, False, True), (0x25, True, True)]


def test_hotkey_releases_in_reverse_order(adapter, sent):
    adapter.hotkey("ctrl", "shift", "tab")
    assert keys(sent[0]) == [(0x11, False, False), (0x10, False, False), (0x09, False, False)]
    assert keys(sent[1]) == [(0x09, True, False), (0x10, True, False), (0x11, True, False)]


def test_unknown_key(adapter, sent):
    with pytest.raises(ActionError, match="unknown key 'banana'"):
        adapter.key_tap("banana")
    assert sent == []


def test_type_text_uses_unicode_injection(adapter, sent):
    adapter.type_text("aऄ", interval=0)          # Latin + Devanagari
    for batch, ch in zip(sent, "aऄ"):
        down, up = batch
        assert down.u.ki.wScan == ord(ch) and down.u.ki.dwFlags == wa._KEYEVENTF_UNICODE
        assert up.u.ki.dwFlags == wa._KEYEVENTF_UNICODE | wa._KEYEVENTF_KEYUP


def test_type_text_emoji_becomes_a_surrogate_pair(adapter, sent):
    adapter.type_text("😀", interval=0)
    scans = [inp.u.ki.wScan for inp in sent[0]]
    assert scans == [0xD83D, 0xD83D, 0xDE00, 0xDE00]


def test_type_text_newline_presses_enter(adapter, sent):
    adapter.type_text("\n", interval=0)
    assert keys(sent[0]) == [(0x0D, False, False), (0x0D, True, False)]


WHEEL, HWHEEL = 0x0800, 0x1000                  # MOUSEEVENTF_WHEEL / _HWHEEL


@pytest.mark.parametrize("direction, flag, delta", [
    ("up", WHEEL, 240), ("down", WHEEL, -240),
    ("right", HWHEEL, 240), ("left", HWHEEL, -240),
])
def test_scroll_wheel(adapter, sent, direction, flag, delta):
    adapter.scroll(direction, 2)
    mi = sent[0][0].u.mi
    assert mi.dwFlags == flag
    assert mi.mouseData == delta & 0xFFFFFFFF


def test_scroll_to_top_presses_home(adapter, sent):
    adapter.scroll("top", 3)
    assert keys(sent[0])[0][0] == 0x24


def test_double_click_sends_two_pairs(adapter, sent):
    adapter.click("left", double=True)
    assert len(sent) == 2
    assert [i.u.mi.dwFlags for i in sent[0]] == [0x0002, 0x0004]


def test_right_click_flags(adapter, sent):
    adapter.click("right")
    assert [i.u.mi.dwFlags for i in sent[0]] == [0x0008, 0x0010]


def test_open_app_wraps_shell_errors(adapter, monkeypatch):
    def fail(cmd):
        raise OSError(2, "The system cannot find the file specified")
    monkeypatch.setattr(wa.os, "startfile", fail)
    with pytest.raises(ActionError, match="could not launch 'nosuchapp'"):
        adapter.open_app("nosuchapp")


def test_open_app_uses_shell_execute(adapter, monkeypatch):
    launched = []
    monkeypatch.setattr(wa.os, "startfile", launched.append)
    adapter.open_app("ms-settings:")
    assert launched == ["ms-settings:"]


def test_volume_change_taps_media_keys(adapter, sent):
    adapter.volume_change(3)
    adapter.volume_change(-1)
    vks = [keys(b)[0][0] for b in sent]
    assert vks == [0xAF, 0xAF, 0xAF, 0xAE]


def test_set_volume_zeroes_then_steps_up(adapter, sent):
    adapter.set_volume(40)
    vks = [keys(b)[0][0] for b in sent]
    assert vks == [0xAE] * 50 + [0xAF] * 20


@pytest.mark.parametrize("op, combo", [
    ("minimize", (0x5B, 0x28)), ("maximize", (0x5B, 0x26)), ("close", (0x12, 0x73)),
    ("switch", (0x12, 0x09)), ("show_desktop", (0x5B, ord("D"))),
])
def test_window_combos(adapter, sent, op, combo):
    adapter.window(op)
    assert tuple(vk for vk, _, _ in keys(sent[0])) == combo


def test_lock_failure_is_reported(adapter, monkeypatch):
    monkeypatch.setattr(wa, "user32", type("U", (), {"LockWorkStation": staticmethod(lambda: 0)}))
    with pytest.raises(ActionError, match="LockWorkStation failed"):
        adapter.lock()


@pytest.mark.parametrize("op, argv", [
    ("shutdown", ["shutdown", "/s", "/t", "3"]),
    ("restart", ["shutdown", "/r", "/t", "3"]),
    ("sleep", ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"]),
])
def test_power_commands(adapter, monkeypatch, op, argv):
    run = RecordedRun()
    monkeypatch.setattr(wa.WindowsAdapter, "_run", staticmethod(run))
    adapter.power(op, 3)
    assert run.calls == [(argv, op)]


def test_unknown_power_op(adapter):
    with pytest.raises(ActionError, match="unknown power op"):
        adapter.power("hibernate")


def test_brightness_script_carries_the_delta(adapter, monkeypatch):
    run = RecordedRun()
    monkeypatch.setattr(wa.WindowsAdapter, "_run", staticmethod(run))
    adapter.brightness(-15)
    argv, what = run.calls[0]
    assert argv[:3] == ["powershell", "-NoProfile", "-Command"] and what == "brightness"
    assert "$b+(-15)" in argv[3]


def test_send_detects_partial_injection(monkeypatch):
    fake = type("U", (), {"SendInput": staticmethod(lambda n, arr, size: n - 1)})
    monkeypatch.setattr(wa, "user32", fake)
    with pytest.raises(ActionError, match="injected 1/2"):
        REAL_SEND(wa._key_input(0x41), wa._key_input(0x41, up=True))
