"""
windows_adapter.py
Windows implementation of OSAdapter — pure ctypes (SendInput / ShellExecute),
no third-party dependencies.

Keyboard and mouse go through SendInput, which is what the OS itself uses
for injected input: media keys (VK 0xAD-0xB3) give volume/media control
without any audio API, and KEYEVENTF_UNICODE types arbitrary text — including
Hindi dictation — without worrying about keyboard layout.

set_volume() has no absolute API without COM, so it slams the volume to zero
with 50 down-presses and steps up to the target (each press = 2 of 100 units).
Crude, but instant and dependency-free.
"""

from __future__ import annotations

import ctypes
import datetime as _dt
import os
import time
from ctypes import wintypes

from central_module.actions.os_adapter import ActionError, OSAdapter

user32 = ctypes.WinDLL("user32", use_last_error=True)

# ── SendInput plumbing ─────────────────────────────────────────────────────────

_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004
_KEYEVENTF_EXTENDEDKEY = 0x0001

_MOUSE_FLAGS = {
    ("left", False): 0x0002, ("left", True): 0x0004,       # down, up
    ("right", False): 0x0008, ("right", True): 0x0010,
    ("middle", False): 0x0020, ("middle", True): 0x0040,
}
_MOUSEEVENTF_WHEEL = 0x0800
_MOUSEEVENTF_HWHEEL = 0x1000
_WHEEL_DELTA = 120


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t))


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t))


class _INPUTUNION(ctypes.Union):
    _fields_ = (("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT))


class _INPUT(ctypes.Structure):
    _fields_ = (("type", wintypes.DWORD), ("u", _INPUTUNION))


def _send(*inputs: _INPUT) -> None:
    array = (_INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(inputs), array, ctypes.sizeof(_INPUT))
    if sent != len(inputs):
        raise ActionError(f"SendInput injected {sent}/{len(inputs)} events "
                          f"(error {ctypes.get_last_error()})")


def _key_input(vk: int, up: bool = False, extended: bool = False) -> _INPUT:
    flags = (_KEYEVENTF_KEYUP if up else 0) | \
            (_KEYEVENTF_EXTENDEDKEY if extended else 0)
    return _INPUT(type=_INPUT_KEYBOARD,
                  u=_INPUTUNION(ki=_KEYBDINPUT(wVk=vk, wScan=0,
                                               dwFlags=flags, time=0,
                                               dwExtraInfo=0)))


def _unicode_input(code_unit: int, up: bool = False) -> _INPUT:
    flags = _KEYEVENTF_UNICODE | (_KEYEVENTF_KEYUP if up else 0)
    return _INPUT(type=_INPUT_KEYBOARD,
                  u=_INPUTUNION(ki=_KEYBDINPUT(wVk=0, wScan=code_unit,
                                               dwFlags=flags, time=0,
                                               dwExtraInfo=0)))


def _mouse_input(flags: int, data: int = 0) -> _INPUT:
    return _INPUT(type=_INPUT_MOUSE,
                  u=_INPUTUNION(mi=_MOUSEINPUT(dx=0, dy=0,
                                               mouseData=data & 0xFFFFFFFF,
                                               dwFlags=flags, time=0,
                                               dwExtraInfo=0)))


# Canonical key id (see nlp.intents.KEY_ALIASES) → virtual-key code.
_VK = {
    "enter": 0x0D, "escape": 0x1B, "tab": 0x09, "space": 0x20,
    "backspace": 0x08, "delete": 0x2E, "insert": 0x2D,
    "home": 0x24, "end": 0x23, "page up": 0x21, "page down": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "win": 0x5B, "ctrl": 0x11, "alt": 0x12, "shift": 0x10,
    "print screen": 0x2C, "caps lock": 0x14,
    "volume up": 0xAF, "volume down": 0xAE, "volume mute": 0xAD,
    "media play pause": 0xB3, "media next": 0xB0, "media previous": 0xB1,
    **{f"f{i}": 0x6F + i for i in range(1, 13)},
    **{c: ord(c.upper()) for c in "abcdefghijklmnopqrstuvwxyz0123456789"},
}

# Keys whose scancode lives in the extended block — SendInput needs the flag
# or apps see the numpad variant.
_EXTENDED = {"delete", "insert", "home", "end", "page up", "page down",
             "up", "down", "left", "right", "print screen", "win"}


class WindowsAdapter(OSAdapter):

    os_name = "Windows"
    primary_mod = "ctrl"

    APP_COMMANDS = {
        "chrome": "chrome", "google chrome": "chrome",
        "edge": "msedge", "browser": "msedge",
        "notepad": "notepad", "calculator": "calc", "calc": "calc",
        "explorer": "explorer", "file explorer": "explorer", "files": "explorer",
        "terminal": "wt", "command prompt": "cmd", "cmd": "cmd",
        "powershell": "powershell",
        "settings": "ms-settings:", "control panel": "control",
        "task manager": "taskmgr", "paint": "mspaint",
        "word": "winword", "excel": "excel", "powerpoint": "powerpnt",
        "outlook": "outlook", "spotify": "spotify",
        "code": "code", "vs code": "code", "visual studio code": "code",
        "youtube": "https://www.youtube.com",
        "google": "https://www.google.com",
        "gmail": "https://mail.google.com",
        "whatsapp": "https://web.whatsapp.com",
    }

    # ── Keyboard ───────────────────────────────────────────────────────────────

    def _vk(self, key: str) -> tuple[int, bool]:
        vk = _VK.get(key)
        if vk is None:
            raise ActionError(f"unknown key '{key}'")
        return vk, key in _EXTENDED

    def key_tap(self, key: str) -> None:
        vk, ext = self._vk(key)
        _send(_key_input(vk, False, ext), _key_input(vk, True, ext))

    def hotkey(self, *keys: str) -> None:
        vks = [self._vk(k) for k in keys]
        _send(*(_key_input(vk, False, ext) for vk, ext in vks))
        time.sleep(0.03)
        _send(*(_key_input(vk, True, ext) for vk, ext in reversed(vks)))

    def type_text(self, text: str, interval: float = 0.012) -> None:
        for ch in text:
            if ch == "\n":
                self.key_tap("enter")
            else:
                # UTF-16 code units — surrogate pairs cover emoji and any
                # script, independent of the active keyboard layout.
                data = ch.encode("utf-16-le")
                units = [int.from_bytes(data[i:i + 2], "little")
                         for i in range(0, len(data), 2)]
                events = []
                for unit in units:
                    events.append(_unicode_input(unit, up=False))
                    events.append(_unicode_input(unit, up=True))
                _send(*events)
            time.sleep(interval)

    # ── Mouse ──────────────────────────────────────────────────────────────────

    def scroll(self, direction: str, steps: int) -> None:
        if direction in ("top", "bottom"):
            self.key_tap("home" if direction == "top" else "end")
            return
        if direction in ("up", "down"):
            delta = _WHEEL_DELTA * steps * (1 if direction == "up" else -1)
            _send(_mouse_input(_MOUSEEVENTF_WHEEL, delta))
        else:  # left / right
            delta = _WHEEL_DELTA * steps * (1 if direction == "right" else -1)
            _send(_mouse_input(_MOUSEEVENTF_HWHEEL, delta))

    def click(self, button: str = "left", double: bool = False) -> None:
        down = _MOUSE_FLAGS[(button, False)]
        up = _MOUSE_FLAGS[(button, True)]
        for _ in range(2 if double else 1):
            _send(_mouse_input(down), _mouse_input(up))
            if double:
                time.sleep(0.05)

    # ── Apps ───────────────────────────────────────────────────────────────────

    def open_app(self, command: str) -> None:
        # os.startfile = ShellExecute: resolves PATH, App Paths registry
        # ("chrome" → chrome.exe), documents, and URIs like ms-settings:.
        try:
            os.startfile(command)
        except OSError as exc:
            raise ActionError(f"could not launch '{command}': {exc.strerror or exc}")

    # ── Volume & media ─────────────────────────────────────────────────────────

    def volume_change(self, steps: int) -> None:
        key = "volume up" if steps > 0 else "volume down"
        for _ in range(abs(steps)):
            self.key_tap(key)

    def volume_mute(self, on: bool) -> None:
        # The mute key toggles; Windows exposes no keyless query without COM,
        # so both mute and unmute map to the toggle.
        self.key_tap("volume mute")

    def set_volume(self, percent: int) -> None:
        percent = max(0, min(100, percent))
        for _ in range(50):                     # 50 × −2 units → 0
            self.key_tap("volume down")
        for _ in range(round(percent / 2)):
            self.key_tap("volume up")

    def media(self, op: str) -> None:
        keys = {"play_pause": "media play pause", "next": "media next",
                "prev": "media previous"}
        self.key_tap(keys[op])

    # ── Windows ────────────────────────────────────────────────────────────────

    def window(self, op: str) -> None:
        combos = {
            "minimize": ("win", "down"), "maximize": ("win", "up"),
            "close": ("alt", "f4"), "switch": ("alt", "tab"),
            "show_desktop": ("win", "d"),
        }
        self.hotkey(*combos[op])

    # ── System ─────────────────────────────────────────────────────────────────

    def lock(self) -> None:
        if not user32.LockWorkStation():
            raise ActionError("LockWorkStation failed")

    def screenshot(self) -> str:
        self.hotkey("win", "print screen")      # OS saves the capture itself
        shots = os.path.join(os.path.expanduser("~"), "Pictures", "Screenshots")
        return f"saved to {shots}"

    def power(self, op: str, delay_s: int = 3) -> None:
        if op == "sleep":
            self._run(["rundll32.exe", "powrprof.dll,SetSuspendState",
                       "0,1,0"], "sleep")
        elif op == "shutdown":
            self._run(["shutdown", "/s", "/t", str(delay_s)], "shutdown")
        elif op == "restart":
            self._run(["shutdown", "/r", "/t", str(delay_s)], "restart")
        else:
            raise ActionError(f"unknown power op '{op}'")

    def brightness(self, delta: int) -> None:
        # WMI brightness exists on laptop panels; external monitors need DDC,
        # which is out of scope — surface the failure honestly instead.
        script = (
            "$b=(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness"
            " -ErrorAction Stop).CurrentBrightness;"
            "$m=Get-CimInstance -Namespace root/WMI -ClassName"
            " WmiMonitorBrightnessMethods -ErrorAction Stop;"
            f"$t=[math]::Max(0,[math]::Min(100,$b+({delta})));"
            "Invoke-CimMethod -InputObject $m -MethodName WmiSetBrightness"
            " -Arguments @{Timeout=1;Brightness=$t} | Out-Null"
        )
        self._run(["powershell", "-NoProfile", "-Command", script], "brightness")


# ── Smoke-test (harmless actions only) ────────────────────────────────────────

if __name__ == "__main__":
    adapter = WindowsAdapter()
    print("volume up 2 …")
    adapter.volume_change(2)
    time.sleep(0.5)
    print("volume down 2 …")
    adapter.volume_change(-2)
    print("scroll down 2 …")
    adapter.scroll("down", 2)
    print(f"typing test at {_dt.datetime.now():%H:%M:%S} — focus a text field (3 s)")
    time.sleep(3)
    adapter.type_text("multimodal control ✓")
    print("done.")
