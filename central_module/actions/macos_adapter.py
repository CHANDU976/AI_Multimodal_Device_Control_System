"""
macos_adapter.py
macOS implementation of OSAdapter, built on stock tools only: osascript
(AppleScript / System Events) for input synthesis, `open` for launching,
screencapture/pmset for system actions.

Requires the Accessibility permission for the terminal/python running the
system (System Settings → Privacy & Security → Accessibility), or every
System Events keystroke fails — the resulting osascript error is surfaced
as an ActionError so the user sees why.

Known gaps, degraded honestly:
  * mouse clicks need `cliclick` (brew) — used when present
  * global media keys aren't scriptable; play/pause falls back to the Music app
  * scroll wheels aren't scriptable; scrolling maps to page up/down keys
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil

from central_module.actions.os_adapter import ActionError, OSAdapter

# Canonical key id → System Events key code.
_KEY_CODES = {
    "enter": 36, "tab": 48, "space": 49, "backspace": 51, "escape": 53,
    "delete": 117, "home": 115, "end": 119, "page up": 116, "page down": 121,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97,
    "f7": 98, "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111,
}
_MODIFIERS = {"ctrl": "control down", "alt": "option down",
              "shift": "shift down", "win": "command down"}


class MacAdapter(OSAdapter):

    os_name = "macOS"
    primary_mod = "win"                 # ⌘ for copy/paste/tabs

    _BROWSER = {
        "new_tab": ("MOD", "t"), "close_tab": ("MOD", "w"),
        "next_tab": ("MOD", "alt", "right"), "prev_tab": ("MOD", "alt", "left"),
        "browser_back": ("MOD", "["), "browser_forward": ("MOD", "]"),
        "refresh": ("MOD", "r"),
    }

    APP_COMMANDS = {
        "chrome": "Google Chrome", "google chrome": "Google Chrome",
        "safari": "Safari", "browser": "Safari",
        "notes": "Notes", "notepad": "TextEdit",
        "calculator": "Calculator", "terminal": "Terminal",
        "finder": "Finder", "files": "Finder", "explorer": "Finder",
        "settings": "System Settings", "spotify": "Spotify",
        "code": "Visual Studio Code", "vs code": "Visual Studio Code",
        "word": "Microsoft Word", "excel": "Microsoft Excel",
        "youtube": "https://www.youtube.com",
        "google": "https://www.google.com",
        "gmail": "https://mail.google.com",
    }

    # ── AppleScript plumbing ───────────────────────────────────────────────────

    def _osa(self, script: str, what: str) -> str:
        return self._run(["osascript", "-e", script], what)

    @staticmethod
    def _tell(inner: str) -> str:
        return f'tell application "System Events" to {inner}'

    # ── Keyboard ───────────────────────────────────────────────────────────────

    def key_tap(self, key: str) -> None:
        self._press(key, ())

    def hotkey(self, *keys: str) -> None:
        mods = tuple(k for k in keys if k in _MODIFIERS)
        rest = [k for k in keys if k not in _MODIFIERS]
        if not rest:                        # e.g. bare modifier — nothing to tap
            raise ActionError(f"hotkey needs a non-modifier key: {keys}")
        self._press(rest[-1], mods)

    def _press(self, key: str, mods: tuple) -> None:
        using = ""
        if mods:
            using = " using {" + ", ".join(_MODIFIERS[m] for m in mods) + "}"
        if key in _KEY_CODES:
            script = self._tell(f"key code {_KEY_CODES[key]}{using}")
        elif len(key) == 1:
            script = self._tell(f'keystroke "{key}"{using}')
        else:
            raise ActionError(f"unknown key '{key}'")
        self._osa(script, f"key '{key}'")

    def type_text(self, text: str) -> None:
        safe = text.replace("\\", "\\\\").replace('"', '\\"')
        self._osa(self._tell(f'keystroke "{safe}"'), "typing")

    # ── Mouse ──────────────────────────────────────────────────────────────────

    def scroll(self, direction: str, steps: int) -> None:
        if direction == "top":
            self.hotkey("win", "up")
            return
        if direction == "bottom":
            self.hotkey("win", "down")
            return
        # No scriptable scroll wheel — page keys are the stock-tool stand-in.
        key = "page up" if direction == "up" else "page down"
        for _ in range(max(1, steps // 3)):
            self.key_tap(key)

    def click(self, button: str = "left", double: bool = False) -> None:
        cliclick = shutil.which("cliclick")
        if not cliclick:
            raise ActionError("mouse clicks need cliclick (brew install cliclick)")
        op = {"left": "c:.", "right": "rc:.", "middle": "c:."}[button]
        if double:
            op = "dc:."
        self._run([cliclick, op], "click")

    # ── Apps ───────────────────────────────────────────────────────────────────

    def open_app(self, command: str) -> None:
        if "://" in command or os.path.exists(command):
            self._run(["open", command], f"open '{command}'")
        else:
            self._run(["open", "-a", command], f"launch '{command}'")

    # ── Volume & media ─────────────────────────────────────────────────────────

    def volume_change(self, steps: int) -> None:
        # System Events has no relative nudge; read-modify-write instead.
        current = int(self._osa(
            "output volume of (get volume settings)", "volume query").strip())
        self.set_volume(current + steps * 6)     # ≈ one key press per step

    def volume_mute(self, on: bool) -> None:
        self._osa(f"set volume output muted {'true' if on else 'false'}", "mute")

    def set_volume(self, percent: int) -> None:
        percent = max(0, min(100, percent))
        self._osa(f"set volume output volume {percent}", "set volume")

    def media(self, op: str) -> None:
        if op == "play_pause":
            self._osa('tell application "Music" to playpause', "play/pause")
        elif op == "next":
            self._osa('tell application "Music" to next track', "next track")
        elif op == "prev":
            self._osa('tell application "Music" to previous track', "previous track")

    # ── Windows ────────────────────────────────────────────────────────────────

    def window(self, op: str) -> None:
        if op == "minimize":
            self.hotkey("win", "m")
        elif op == "maximize":
            self._osa(self._tell(
                "tell (first process whose frontmost is true) to "
                "set value of attribute \"AXFullScreen\" of front window to true"),
                "maximize")
        elif op == "close":
            self.hotkey("win", "w")
        elif op == "switch":
            self.hotkey("win", "tab")
        elif op == "show_desktop":
            self.key_tap("f11")

    # ── System ─────────────────────────────────────────────────────────────────

    def lock(self) -> None:
        self._run(["pmset", "displaysleepnow"], "lock")

    def screenshot(self) -> str:
        path = os.path.join(
            os.path.expanduser("~"), "Desktop",
            f"screenshot-{_dt.datetime.now():%Y%m%d-%H%M%S}.png")
        self._run(["screencapture", "-x", path], "screenshot")
        return f"saved to {path}"

    def power(self, op: str, delay_s: int = 3) -> None:
        scripts = {"sleep": "sleep", "shutdown": "shut down", "restart": "restart"}
        self._osa(f'tell application "System Events" to {scripts[op]}', op)
