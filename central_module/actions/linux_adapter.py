"""
linux_adapter.py
Linux (X11 desktop) implementation of OSAdapter.

System packages it leans on — all best-effort with clear errors when absent:

    xdotool        keyboard, mouse, window management   (sudo apt install xdotool)
    pactl / amixer volume                               (PulseAudio / ALSA, stock)
    playerctl      media keys                           (optional)
    brightnessctl  screen brightness                    (optional)
    xdg-open       apps, files, URLs                    (stock)

Wayland note: xdotool needs X11 (or XWayland focus); on pure Wayland
sessions input synthesis is restricted by design and `ydotool` + udev rules
would be the replacement — out of scope here, the errors say so.
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil
import subprocess

from central_module.actions.os_adapter import ActionError, OSAdapter

# Canonical key id → X keysym for xdotool.
_KEYSYMS = {
    "enter": "Return", "escape": "Escape", "tab": "Tab", "space": "space",
    "backspace": "BackSpace", "delete": "Delete", "insert": "Insert",
    "home": "Home", "end": "End", "page up": "Prior", "page down": "Next",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "win": "super", "ctrl": "ctrl", "alt": "alt", "shift": "shift",
    "print screen": "Print", "caps lock": "Caps_Lock",
    **{f"f{i}": f"F{i}" for i in range(1, 13)},
    **{c: c for c in "abcdefghijklmnopqrstuvwxyz0123456789"},
}


class LinuxAdapter(OSAdapter):

    os_name = "Linux"
    primary_mod = "ctrl"

    APP_COMMANDS = {
        "chrome": "google-chrome", "google chrome": "google-chrome",
        "firefox": "firefox", "browser": "firefox",
        "files": "nautilus", "explorer": "nautilus", "file explorer": "nautilus",
        "terminal": "x-terminal-emulator",
        "calculator": "gnome-calculator",
        "settings": "gnome-control-center",
        "notepad": "gedit", "code": "code", "vs code": "code",
        "spotify": "spotify",
        "youtube": "https://www.youtube.com",
        "google": "https://www.google.com",
        "gmail": "https://mail.google.com",
    }

    def _xdotool(self, *args: str) -> None:
        if not shutil.which("xdotool"):
            raise ActionError("xdotool is not installed (sudo apt install xdotool)")
        self._run(["xdotool", *args], f"xdotool {args[0]}")

    @staticmethod
    def _keysym(key: str) -> str:
        sym = _KEYSYMS.get(key)
        if sym is None:
            raise ActionError(f"unknown key '{key}'")
        return sym

    # ── Keyboard ───────────────────────────────────────────────────────────────

    def key_tap(self, key: str) -> None:
        self._xdotool("key", self._keysym(key))

    def hotkey(self, *keys: str) -> None:
        self._xdotool("key", "+".join(self._keysym(k) for k in keys))

    def type_text(self, text: str) -> None:
        self._xdotool("type", "--delay", "12", "--", text)

    # ── Mouse ──────────────────────────────────────────────────────────────────

    def scroll(self, direction: str, steps: int) -> None:
        if direction in ("top", "bottom"):
            self.key_tap("home" if direction == "top" else "end")
            return
        button = {"up": "4", "down": "5", "left": "6", "right": "7"}[direction]
        self._xdotool("click", "--repeat", str(steps), button)

    def click(self, button: str = "left", double: bool = False) -> None:
        num = {"left": "1", "middle": "2", "right": "3"}[button]
        args = ["click"]
        if double:
            args += ["--repeat", "2", "--delay", "50"]
        self._xdotool(*args, num)

    # ── Apps ───────────────────────────────────────────────────────────────────

    def open_app(self, command: str) -> None:
        if "://" in command:
            self._run(["xdg-open", command], f"open '{command}'")
            return
        exe = shutil.which(command)
        if exe:
            subprocess.Popen([exe], start_new_session=True,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        elif os.path.exists(command):
            self._run(["xdg-open", command], f"open '{command}'")
        else:
            raise ActionError(f"could not find '{command}' on PATH")

    # ── Volume & media ─────────────────────────────────────────────────────────

    def _pactl(self, *args: str) -> None:
        if shutil.which("pactl"):
            self._run(["pactl", *args], "volume")
        else:
            raise ActionError("no pactl found — is PulseAudio/PipeWire running?")

    def volume_change(self, steps: int) -> None:
        sign = "+" if steps > 0 else "-"
        self._pactl("set-sink-volume", "@DEFAULT_SINK@",
                    f"{sign}{abs(steps) * 2}%")

    def volume_mute(self, on: bool) -> None:
        self._pactl("set-sink-mute", "@DEFAULT_SINK@", "1" if on else "0")

    def set_volume(self, percent: int) -> None:
        self._pactl("set-sink-volume", "@DEFAULT_SINK@",
                    f"{max(0, min(100, percent))}%")

    def media(self, op: str) -> None:
        if not shutil.which("playerctl"):
            raise ActionError("media control needs playerctl (sudo apt install playerctl)")
        cmd = {"play_pause": "play-pause", "next": "next", "prev": "previous"}[op]
        self._run(["playerctl", cmd], "media")

    # ── Windows ────────────────────────────────────────────────────────────────

    def window(self, op: str) -> None:
        if op == "minimize":
            self._xdotool("getactivewindow", "windowminimize")
        elif op == "maximize":
            self.hotkey("win", "up")
        elif op == "close":
            self.hotkey("alt", "f4")
        elif op == "switch":
            self.hotkey("alt", "tab")
        elif op == "show_desktop":
            self.hotkey("win", "d")

    # ── System ─────────────────────────────────────────────────────────────────

    def lock(self) -> None:
        if shutil.which("loginctl"):
            self._run(["loginctl", "lock-session"], "lock")
        else:
            self._run(["xdg-screensaver", "lock"], "lock")

    def screenshot(self) -> str:
        path = os.path.join(
            os.path.expanduser("~"), "Pictures",
            f"screenshot-{_dt.datetime.now():%Y%m%d-%H%M%S}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if shutil.which("gnome-screenshot"):
            self._run(["gnome-screenshot", "-f", path], "screenshot")
        elif shutil.which("import"):                     # ImageMagick
            self._run(["import", "-window", "root", path], "screenshot")
        else:
            raise ActionError("no screenshot tool (gnome-screenshot or imagemagick)")
        return f"saved to {path}"

    def power(self, op: str, delay_s: int = 3) -> None:
        cmds = {"sleep": ["systemctl", "suspend"],
                "shutdown": ["systemctl", "poweroff"],
                "restart": ["systemctl", "reboot"]}
        self._run(cmds[op], op)

    def brightness(self, delta: int) -> None:
        if not shutil.which("brightnessctl"):
            raise ActionError("brightness needs brightnessctl (sudo apt install brightnessctl)")
        spec = f"+{delta}%" if delta > 0 else f"{abs(delta)}%-"
        self._run(["brightnessctl", "set", spec], "brightness")
