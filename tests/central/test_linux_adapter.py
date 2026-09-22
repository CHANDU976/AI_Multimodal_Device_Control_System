"""actions/linux_adapter.py — command construction for xdotool/pactl/etc.

Runs on every OS: helper commands are recorded, never executed.
"""

import pytest

from central_module.actions import linux_adapter
from central_module.actions.linux_adapter import LinuxAdapter, _KEYSYMS
from central_module.actions.os_adapter import ActionError
from central_module.nlp.intents import KEY_ALIASES
from helpers.fakes import RecordedRun

ALL_TOOLS = {"xdotool", "pactl", "playerctl", "brightnessctl", "loginctl",
             "gnome-screenshot", "google-chrome"}


@pytest.fixture
def tools(monkeypatch):
    """Mutable set of tools that 'exist' on PATH."""
    present = set(ALL_TOOLS)
    monkeypatch.setattr(linux_adapter.shutil, "which",
                        lambda name: f"/usr/bin/{name}" if name in present else None)
    return present


@pytest.fixture
def adapter(monkeypatch, tools):
    run = RecordedRun()
    monkeypatch.setattr(LinuxAdapter, "_run", staticmethod(run))
    a = LinuxAdapter()
    a.recorded = run
    return a


def argvs(adapter):
    return [argv for argv, _ in adapter.recorded.calls]


def test_every_canonical_key_has_a_keysym():
    missing = set(KEY_ALIASES.values()) - set(_KEYSYMS)
    assert not missing, f"keys the Linux adapter cannot press: {sorted(missing)}"


def test_key_tap(adapter):
    adapter.key_tap("enter")
    adapter.key_tap("page down")
    assert argvs(adapter) == [["xdotool", "key", "Return"], ["xdotool", "key", "Next"]]


def test_hotkey_joins_with_plus(adapter):
    adapter.hotkey("ctrl", "shift", "tab")
    assert argvs(adapter) == [["xdotool", "key", "ctrl+shift+Tab"]]


def test_unknown_key(adapter):
    with pytest.raises(ActionError, match="unknown key 'banana'"):
        adapter.key_tap("banana")


def test_missing_xdotool_explains_the_fix(adapter, tools):
    tools.discard("xdotool")
    with pytest.raises(ActionError, match="apt install xdotool"):
        adapter.key_tap("a")


def test_type_text_ends_option_parsing(adapter):
    adapter.type_text("--help me")
    assert argvs(adapter) == [["xdotool", "type", "--delay", "12", "--", "--help me"]]


@pytest.mark.parametrize("direction, steps, argv", [
    ("up", 3, ["xdotool", "click", "--repeat", "3", "4"]),
    ("down", 2, ["xdotool", "click", "--repeat", "2", "5"]),
    ("left", 1, ["xdotool", "click", "--repeat", "1", "6"]),
    ("right", 1, ["xdotool", "click", "--repeat", "1", "7"]),
    ("top", 3, ["xdotool", "key", "Home"]),
    ("bottom", 3, ["xdotool", "key", "End"]),
])
def test_scroll(adapter, direction, steps, argv):
    adapter.scroll(direction, steps)
    assert argvs(adapter) == [argv]


@pytest.mark.parametrize("button, double, argv", [
    ("left", False, ["xdotool", "click", "1"]),
    ("right", False, ["xdotool", "click", "3"]),
    ("middle", False, ["xdotool", "click", "2"]),
    ("left", True, ["xdotool", "click", "--repeat", "2", "--delay", "50", "1"]),
])
def test_click(adapter, button, double, argv):
    adapter.click(button, double)
    assert argvs(adapter) == [argv]


def test_open_url_uses_xdg_open(adapter):
    adapter.open_app("https://www.youtube.com")
    assert argvs(adapter) == [["xdg-open", "https://www.youtube.com"]]


def test_open_app_on_path_is_detached(adapter, monkeypatch):
    launched = []
    monkeypatch.setattr(linux_adapter.subprocess, "Popen",
                        lambda argv, **kw: launched.append((argv, kw)))
    adapter.open_app("google-chrome")
    assert launched[0][0] == ["/usr/bin/google-chrome"]
    assert launched[0][1]["start_new_session"] is True


def test_open_existing_file_uses_xdg_open(adapter, tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("x")
    adapter.open_app(str(doc))
    assert argvs(adapter) == [["xdg-open", str(doc)]]


def test_open_unknown_app(adapter):
    with pytest.raises(ActionError, match="could not find 'nosuchapp'"):
        adapter.open_app("nosuchapp")


@pytest.mark.parametrize("steps, arg", [(2, "+4%"), (-3, "-6%"), (1, "+2%")])
def test_volume_change(adapter, steps, arg):
    adapter.volume_change(steps)
    assert argvs(adapter) == [["pactl", "set-sink-volume", "@DEFAULT_SINK@", arg]]


def test_mute_and_unmute(adapter):
    adapter.volume_mute(True)
    adapter.volume_mute(False)
    assert [a[-1] for a in argvs(adapter)] == ["1", "0"]


@pytest.mark.parametrize("requested, applied", [(40, "40%"), (150, "100%"), (-5, "0%")])
def test_set_volume_clamps(adapter, requested, applied):
    adapter.set_volume(requested)
    assert argvs(adapter)[0][-1] == applied


def test_volume_without_pulseaudio(adapter, tools):
    tools.discard("pactl")
    with pytest.raises(ActionError, match="PulseAudio"):
        adapter.volume_change(1)


@pytest.mark.parametrize("op, cmd", [("play_pause", "play-pause"), ("next", "next"),
                                     ("prev", "previous")])
def test_media(adapter, op, cmd):
    adapter.media(op)
    assert argvs(adapter) == [["playerctl", cmd]]


def test_media_needs_playerctl(adapter, tools):
    tools.discard("playerctl")
    with pytest.raises(ActionError, match="playerctl"):
        adapter.media("next")


@pytest.mark.parametrize("op, argv", [
    ("minimize", ["xdotool", "getactivewindow", "windowminimize"]),
    ("maximize", ["xdotool", "key", "super+Up"]),
    ("close", ["xdotool", "key", "alt+F4"]),
    ("switch", ["xdotool", "key", "alt+Tab"]),
    ("show_desktop", ["xdotool", "key", "super+d"]),
])
def test_window(adapter, op, argv):
    adapter.window(op)
    assert argvs(adapter) == [argv]


def test_lock_prefers_loginctl(adapter, tools):
    adapter.lock()
    tools.discard("loginctl")
    adapter.lock()
    assert argvs(adapter) == [["loginctl", "lock-session"], ["xdg-screensaver", "lock"]]


def test_screenshot_tools_in_preference_order(adapter, tools, tmp_path, monkeypatch):
    monkeypatch.setattr(linux_adapter.os.path, "expanduser", lambda p: str(tmp_path))
    msg = adapter.screenshot()
    assert argvs(adapter)[0][:2] == ["gnome-screenshot", "-f"]
    assert msg.startswith("saved to ") and str(tmp_path) in msg

    tools.discard("gnome-screenshot")
    tools.add("import")
    adapter.screenshot()
    assert argvs(adapter)[1][:3] == ["import", "-window", "root"]

    tools.discard("import")
    with pytest.raises(ActionError, match="no screenshot tool"):
        adapter.screenshot()


@pytest.mark.parametrize("op, argv", [("sleep", ["systemctl", "suspend"]),
                                      ("shutdown", ["systemctl", "poweroff"]),
                                      ("restart", ["systemctl", "reboot"])])
def test_power(adapter, op, argv):
    adapter.power(op)
    assert argvs(adapter) == [argv]


def test_brightness(adapter, tools):
    adapter.brightness(15)
    adapter.brightness(-15)
    assert argvs(adapter) == [["brightnessctl", "set", "+15%"],
                              ["brightnessctl", "set", "15%-"]]
    tools.discard("brightnessctl")
    with pytest.raises(ActionError, match="brightnessctl"):
        adapter.brightness(5)


def test_builtin_app_table_targets_linux_binaries():
    assert LinuxAdapter.APP_COMMANDS["chrome"] == "google-chrome"
    assert all(v for v in LinuxAdapter.APP_COMMANDS.values())
