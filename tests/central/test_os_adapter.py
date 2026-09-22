"""actions/os_adapter.py — the interface every platform implements."""

import subprocess
import sys

import pytest

from central_module.actions import os_adapter
from central_module.actions.os_adapter import ActionError, OSAdapter, get_adapter

# Captured at import, before the suite-wide guard replaces it per test.
REAL_RUN = OSAdapter.__dict__["_run"]


class HotkeyRecorder(OSAdapter):
    os_name = "TestOS"

    def __init__(self):
        self.hotkeys = []

    def hotkey(self, *keys):
        self.hotkeys.append(keys)


@pytest.mark.parametrize("method, args", [
    ("key_tap", ("a",)), ("hotkey", ("ctrl", "c")), ("type_text", ("hi",)),
    ("scroll", ("up", 1)), ("click", ()), ("open_app", ("x",)),
    ("volume_change", (1,)), ("volume_mute", (True,)), ("set_volume", (5,)),
    ("media", ("next",)), ("window", ("minimize",)), ("lock", ()),
    ("screenshot", ()), ("power", ("sleep",)), ("brightness", (5,)),
])
def test_base_primitives_degrade_with_a_clear_message(method, args):
    adapter = OSAdapter()
    adapter.os_name = "Plan9"
    with pytest.raises(ActionError, match="not supported on Plan9"):
        getattr(adapter, method)(*args)


@pytest.mark.parametrize("name, keys", [
    ("copy", ("ctrl", "c")), ("paste", ("ctrl", "v")), ("cut", ("ctrl", "x")),
    ("select_all", ("ctrl", "a")), ("undo", ("ctrl", "z")),
    ("redo", ("ctrl", "y")), ("save", ("ctrl", "s")),
])
def test_edit_combos_use_the_primary_modifier(name, keys):
    adapter = HotkeyRecorder()
    adapter.edit_combo(name)
    assert adapter.hotkeys == [keys]


def test_primary_modifier_is_substituted():
    adapter = HotkeyRecorder()
    adapter.primary_mod = "win"                 # what macOS does for ⌘
    adapter.edit_combo("copy")
    adapter.browser("new_tab")
    adapter.browser("next_tab")                 # not MOD-based: stays ctrl
    assert adapter.hotkeys == [("win", "c"), ("win", "t"), ("ctrl", "tab")]


def test_unknown_combo_raises():
    with pytest.raises(ActionError, match="unknown combo 'teleport'"):
        HotkeyRecorder().browser("teleport")


# ── _run: the helper-command wrapper, exercised with the real interpreter ──────

@pytest.fixture
def real_run(monkeypatch):
    monkeypatch.setattr(OSAdapter, "_run", REAL_RUN)
    return OSAdapter._run


def test_run_returns_stdout(real_run):
    assert real_run([sys.executable, "-c", "print('ok')"], "probe").strip() == "ok"


def test_run_nonzero_exit_reports_stderr(real_run):
    code = "import sys; sys.stderr.write('permission denied'); sys.exit(3)"
    with pytest.raises(ActionError, match="lock failed: permission denied"):
        real_run([sys.executable, "-c", code], "lock")


def test_run_nonzero_exit_without_output_reports_code(real_run):
    with pytest.raises(ActionError, match="exit 4"):
        real_run([sys.executable, "-c", "import sys; sys.exit(4)"], "x")


def test_run_missing_binary(real_run):
    with pytest.raises(ActionError, match="'definitely-not-a-real-tool' not installed"):
        real_run(["definitely-not-a-real-tool"], "brightness")


def test_run_timeout(real_run, monkeypatch):
    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)
    monkeypatch.setattr(os_adapter.subprocess, "run", slow)
    with pytest.raises(ActionError, match="volume timed out"):
        real_run(["x"], "volume")


# ── Platform selection ─────────────────────────────────────────────────────────

def test_get_adapter_linux(monkeypatch):
    monkeypatch.setattr(os_adapter.platform, "system", lambda: "Linux")
    assert type(get_adapter()).__name__ == "LinuxAdapter"


def test_get_adapter_unknown_platform_falls_back_to_linux(monkeypatch):
    monkeypatch.setattr(os_adapter.platform, "system", lambda: "FreeBSD")
    assert type(get_adapter()).__name__ == "LinuxAdapter"


def test_get_adapter_macos(monkeypatch):
    monkeypatch.setattr(os_adapter.platform, "system", lambda: "Darwin")
    assert type(get_adapter()).__name__ == "MacAdapter"


@pytest.mark.skipif(sys.platform != "win32", reason="WindowsAdapter binds user32.dll")
def test_get_adapter_windows(monkeypatch):
    monkeypatch.setattr(os_adapter.platform, "system", lambda: "Windows")
    assert type(get_adapter()).__name__ == "WindowsAdapter"
