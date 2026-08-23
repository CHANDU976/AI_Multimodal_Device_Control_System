"""
os_adapter.py
The OS abstraction: one interface, three implementations.

The executor only ever talks to this interface, so intent handling is fully
OS-independent; porting to a new OS = implementing the primitives below.
Combos that differ only by modifier key (copy/paste, tab handling) are
implemented here once on top of the primitives.

Every method either succeeds or raises ActionError with a human-readable
reason — the executor turns that into a spoken/printed failure message.
Capabilities an OS genuinely lacks call _unsupported() so the system degrades
with a clear message instead of a crash.
"""

from __future__ import annotations

import platform
import subprocess


class ActionError(RuntimeError):
    """An action could not be performed; str(err) explains why."""


class OSAdapter:
    """Base adapter: shared combos + not-supported defaults for primitives."""

    os_name = platform.system() or "unknown"
    primary_mod = "ctrl"                # macOS overrides with "win" (⌘)

    # ── Primitives (override per OS) ───────────────────────────────────────────

    def key_tap(self, key: str) -> None:
        self._unsupported("key press")

    def hotkey(self, *keys: str) -> None:
        self._unsupported("hotkey")

    def type_text(self, text: str) -> None:
        self._unsupported("typing")

    def scroll(self, direction: str, steps: int) -> None:
        self._unsupported("scrolling")

    def click(self, button: str = "left", double: bool = False) -> None:
        self._unsupported("mouse click")

    def open_app(self, command: str) -> None:
        self._unsupported("launching applications")

    def volume_change(self, steps: int) -> None:      # +up / −down
        self._unsupported("volume control")

    def volume_mute(self, on: bool) -> None:
        self._unsupported("mute")

    def set_volume(self, percent: int) -> None:
        self._unsupported("setting volume")

    def media(self, op: str) -> None:                 # play_pause | next | prev
        self._unsupported("media control")

    def window(self, op: str) -> None:  # minimize|maximize|close|switch|show_desktop
        self._unsupported("window management")

    def lock(self) -> None:
        self._unsupported("locking the screen")

    def screenshot(self) -> str:
        """Returns a hint about where the capture went."""
        self._unsupported("screenshots")

    def power(self, op: str, delay_s: int = 3) -> None:  # sleep|shutdown|restart
        self._unsupported("power control")

    def brightness(self, delta: int) -> None:
        self._unsupported("brightness control")

    # ── Shared combos ──────────────────────────────────────────────────────────

    _EDIT = {
        "copy": ("MOD", "c"), "paste": ("MOD", "v"), "cut": ("MOD", "x"),
        "select_all": ("MOD", "a"), "undo": ("MOD", "z"),
        "redo": ("MOD", "y"), "save": ("MOD", "s"),
    }
    _BROWSER = {
        "new_tab": ("MOD", "t"), "close_tab": ("MOD", "w"),
        "next_tab": ("ctrl", "tab"), "prev_tab": ("ctrl", "shift", "tab"),
        "browser_back": ("alt", "left"), "browser_forward": ("alt", "right"),
        "refresh": ("f5",),
    }

    def edit_combo(self, name: str) -> None:
        self._combo(self._EDIT, name)

    def browser(self, op: str) -> None:
        self._combo(self._BROWSER, op)

    def _combo(self, table: dict, name: str) -> None:
        keys = table.get(name)
        if not keys:
            raise ActionError(f"unknown combo '{name}'")
        self.hotkey(*(self.primary_mod if k == "MOD" else k for k in keys))

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _unsupported(self, what: str) -> None:
        raise ActionError(f"{what} is not supported on {self.os_name} yet")

    @staticmethod
    def _run(argv: list[str], what: str, timeout: float = 15.0) -> str:
        """Run a helper command; ActionError on any failure. Returns stdout."""
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout,
            )
        except FileNotFoundError:
            raise ActionError(f"{what} failed: '{argv[0]}' not installed")
        except subprocess.TimeoutExpired:
            raise ActionError(f"{what} timed out")
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[:200]
            raise ActionError(f"{what} failed: {detail or 'exit ' + str(proc.returncode)}")
        return proc.stdout


def get_adapter() -> OSAdapter:
    """Instantiate the adapter for the current platform."""
    system = platform.system()
    if system == "Windows":
        from central_module.actions.windows_adapter import WindowsAdapter
        return WindowsAdapter()
    if system == "Darwin":
        from central_module.actions.macos_adapter import MacAdapter
        return MacAdapter()
    from central_module.actions.linux_adapter import LinuxAdapter
    return LinuxAdapter()
