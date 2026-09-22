"""actions/macos_adapter.py — AppleScript / stock-tool command construction.

Runs on every OS: osascript and friends are recorded, never executed.
"""

import pytest

from central_module.actions import macos_adapter
from central_module.actions.macos_adapter import MacAdapter
from central_module.actions.os_adapter import ActionError
from helpers.fakes import RecordedRun

SE = 'tell application "System Events" to '


@pytest.fixture
def adapter(monkeypatch):
    run = RecordedRun()
    monkeypatch.setattr(MacAdapter, "_run", staticmethod(run))
    monkeypatch.setattr(macos_adapter.shutil, "which", lambda name: None)
    a = MacAdapter()
    a.recorded = run
    return a


def scripts(adapter):
    """AppleScript bodies passed to osascript -e."""
    out = []
    for argv, _ in adapter.recorded.calls:
        assert argv[:2] == ["osascript", "-e"], argv
        out.append(argv[2])
    return out


def test_key_codes(adapter):
    adapter.key_tap("enter")
    adapter.key_tap("left")
    assert scripts(adapter) == [SE + "key code 36", SE + "key code 123"]


def test_single_characters_use_keystroke(adapter):
    adapter.key_tap("a")
    assert scripts(adapter) == [SE + 'keystroke "a"']


def test_hotkey_maps_modifiers(adapter):
    adapter.hotkey("win", "shift", "z")
    adapter.hotkey("ctrl", "tab")
    assert scripts(adapter) == [
        SE + 'keystroke "z" using {command down, shift down}',
        SE + "key code 48 using {control down}",
    ]


def test_modifier_only_hotkey_is_rejected(adapter):
    with pytest.raises(ActionError, match="non-modifier"):
        adapter.hotkey("win", "shift")


@pytest.mark.parametrize("key", ["insert", "print screen", "caps lock", "win"])
def test_documented_key_gaps_fail_cleanly(adapter, key):
    # Keys System Events cannot tap on their own: a readable ActionError,
    # never a crash or a wrong key.
    with pytest.raises(ActionError, match="unknown key"):
        adapter.key_tap(key)


def test_copy_and_tabs_use_command_key(adapter):
    adapter.edit_combo("copy")
    adapter.browser("new_tab")
    adapter.browser("browser_back")
    adapter.browser("refresh")
    assert scripts(adapter) == [
        SE + 'keystroke "c" using {command down}',
        SE + 'keystroke "t" using {command down}',
        SE + 'keystroke "[" using {command down}',
        SE + 'keystroke "r" using {command down}',
    ]


def test_type_text_escapes_applescript_quotes(adapter):
    adapter.type_text('say "hi" \\ bye')
    assert scripts(adapter) == [SE + 'keystroke "say \\"hi\\" \\\\ bye"']


@pytest.mark.parametrize("direction, steps, expected", [
    ("up", 3, [SE + "key code 116"]),
    ("down", 9, [SE + "key code 121"] * 3),
    ("down", 1, [SE + "key code 121"]),            # at least one page
    ("top", 3, [SE + "key code 126 using {command down}"]),
    ("bottom", 3, [SE + "key code 125 using {command down}"]),
])
def test_scroll_maps_to_page_keys(adapter, direction, steps, expected):
    adapter.scroll(direction, steps)
    assert scripts(adapter) == expected


def test_click_requires_cliclick(adapter):
    with pytest.raises(ActionError, match="brew install cliclick"):
        adapter.click()


@pytest.mark.parametrize("button, double, op", [
    ("left", False, "c:."), ("right", False, "rc:."), ("left", True, "dc:."),
])
def test_click_with_cliclick(adapter, monkeypatch, button, double, op):
    monkeypatch.setattr(macos_adapter.shutil, "which", lambda name: "/opt/bin/cliclick")
    adapter.click(button, double)
    assert adapter.recorded.calls == [(["/opt/bin/cliclick", op], "click")]


def test_open_app_by_name_and_url(adapter):
    adapter.open_app("Google Chrome")
    adapter.open_app("https://mail.google.com")
    assert [argv for argv, _ in adapter.recorded.calls] == [
        ["open", "-a", "Google Chrome"], ["open", "https://mail.google.com"]]


def test_volume_change_is_read_modify_write(adapter):
    adapter.recorded.stdout = "50\n"
    adapter.volume_change(2)
    assert scripts(adapter) == ["output volume of (get volume settings)",
                                "set volume output volume 62"]


@pytest.mark.parametrize("requested, applied", [(40, 40), (140, 100), (-3, 0)])
def test_set_volume_clamps(adapter, requested, applied):
    adapter.set_volume(requested)
    assert scripts(adapter) == [f"set volume output volume {applied}"]


def test_mute(adapter):
    adapter.volume_mute(True)
    adapter.volume_mute(False)
    assert scripts(adapter) == ["set volume output muted true",
                                "set volume output muted false"]


def test_media_drives_music_app(adapter):
    for op in ("play_pause", "next", "prev"):
        adapter.media(op)
    assert scripts(adapter) == ['tell application "Music" to playpause',
                                'tell application "Music" to next track',
                                'tell application "Music" to previous track']


def test_window_ops(adapter):
    adapter.window("minimize")
    adapter.window("close")
    adapter.window("show_desktop")
    adapter.window("maximize")
    out = scripts(adapter)
    assert out[:3] == [SE + 'keystroke "m" using {command down}',
                       SE + 'keystroke "w" using {command down}',
                       SE + "key code 103"]
    assert "AXFullScreen" in out[3]


def test_lock_screenshot_power(adapter, tmp_path, monkeypatch):
    monkeypatch.setattr(macos_adapter.os.path, "expanduser", lambda p: str(tmp_path))
    adapter.lock()
    message = adapter.screenshot()
    adapter.power("shutdown")
    calls = [argv for argv, _ in adapter.recorded.calls]
    assert calls[0] == ["pmset", "displaysleepnow"]
    assert calls[1][:2] == ["screencapture", "-x"] and calls[1][2].endswith(".png")
    assert message == f"saved to {calls[1][2]}"
    assert calls[2] == ["osascript", "-e", 'tell application "System Events" to shut down']


def test_brightness_is_honestly_unsupported(adapter):
    with pytest.raises(ActionError, match="not supported on macOS"):
        adapter.brightness(10)
