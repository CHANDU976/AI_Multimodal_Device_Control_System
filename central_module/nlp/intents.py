"""
intents.py
The intent catalog: every action the system can perform, with the spoken
patterns that trigger it.

Pattern syntax
--------------
Literal words plus {slot} placeholders, matched against a *normalized*
utterance (see normalize.py). Slot kinds:

    app        application name — greedy tail ("open {app}")
    text       free payload — greedy tail, matched on the light form so
               dictated words survive ("type {text}", "search for {text}")
    number     integer 0-999 (number words already converted to digits)
    count      like number, used for repeats
    direction  up | down | left | right | top | bottom
    key        a keyboard key name, validated against KEY_ALIASES

Adding a capability = one Intent entry here + one handler in
actions/executor.py. Nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Intent:
    name: str                       # unique id, also used by adaptation
    action: str                     # executor handler name
    patterns: tuple[str, ...]       # spoken templates
    slots: dict = field(default_factory=dict)   # slot name → kind
    confirm: bool = False           # armed + "yes" before executing


# Spoken key name → canonical key id understood by every OS adapter.
KEY_ALIASES = {
    "enter": "enter", "return": "enter",
    "escape": "escape", "esc": "escape",
    "tab": "tab",
    "space": "space", "spacebar": "space",
    "backspace": "backspace",
    "delete": "delete", "del": "delete",
    "insert": "insert",
    "home": "home", "end": "end",
    "page up": "page up", "page down": "page down",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "up arrow": "up", "down arrow": "down",
    "left arrow": "left", "right arrow": "right",
    "windows": "win", "windows key": "win", "win": "win", "super": "win",
    "command": "win", "control": "ctrl", "ctrl": "ctrl",
    "alt": "alt", "option": "alt", "shift": "shift",
    "print screen": "print screen", "caps lock": "caps lock",
    **{f"f{i}": f"f{i}" for i in range(1, 13)},
    **{c: c for c in "abcdefghijklmnopqrstuvwxyz0123456789"},
}

DIRECTIONS = {"up", "down", "left", "right", "top", "bottom"}


CATALOG: tuple[Intent, ...] = (
    # ── Apps & web ─────────────────────────────────────────────────────────────
    Intent("open_app", "open_app",
           ("open {app}", "launch {app}", "start {app}", "run {app}",
            "edit {app}"),      # editing a file = opening it
           {"app": "app"}),
    Intent("search_web", "search_web",
           ("search for {text}", "search {text}", "google {text}",
            "look up {text}"),
           {"text": "text"}),
    Intent("open_url", "open_url",
           ("go to {text}", "open website {text}", "visit {text}"),
           {"text": "text"}),

    # ── Typing & keys ──────────────────────────────────────────────────────────
    Intent("type_text", "type_text",
           ("type {text}", "write {text}"), {"text": "text"}),
    Intent("press_key", "press_key",
           ("press {key}", "hit {key}", "press key {key}"), {"key": "key"}),
    Intent("copy", "edit_combo", ("copy", "copy this", "copy that")),
    Intent("paste", "edit_combo", ("paste", "paste it", "paste here")),
    Intent("cut", "edit_combo", ("cut", "cut this")),
    Intent("select_all", "edit_combo", ("select all", "select everything")),
    Intent("undo", "edit_combo", ("undo", "undo that")),
    Intent("redo", "edit_combo", ("redo", "redo that")),
    Intent("save", "edit_combo", ("save", "save file", "save this file")),
    Intent("delete", "delete_key",
           ("delete", "delete this", "delete that", "delete selection")),

    # ── Cursor ─────────────────────────────────────────────────────────────────
    Intent("scroll", "scroll",
           ("scroll {direction}", "scroll {direction} {count}",
            "scroll {direction} {count} times", "scroll to {direction}"),
           {"direction": "direction", "count": "count"}),
    Intent("click", "click", ("click", "left click", "click here")),
    Intent("double_click", "click", ("double click", "click twice")),
    Intent("right_click", "click", ("right click", "context menu")),

    # ── Volume & media ─────────────────────────────────────────────────────────
    Intent("volume_up", "volume",
           ("volume up", "increase volume", "raise volume", "louder",
            "turn up volume", "turn volume up", "volume up {count}",
            # "by N" is a relative nudge — distinct from "to N" (set_volume)
            "increase volume by {count}", "volume up by {count}",
            "raise volume by {count}", "turn volume up by {count}"),
           {"count": "count"}),
    Intent("volume_down", "volume",
           ("volume down", "decrease volume", "lower volume", "quieter",
            "turn down volume", "turn volume down", "volume down {count}",
            "decrease volume by {count}", "volume down by {count}",
            "lower volume by {count}", "turn volume down by {count}"),
           {"count": "count"}),
    Intent("mute", "volume", ("mute", "mute volume", "silence")),
    Intent("unmute", "volume", ("unmute", "unmute volume", "sound on")),
    Intent("set_volume", "volume",
           ("set volume to {number}", "set volume {number}",
            "volume {number} percent",
            # "increase volume to 67" means set-to-67, not nudge-up — these
            # must exist as exact patterns or the fixed "increase volume"
            # phrase fuzzy-matches and silently drops the level.
            "increase volume to {number}", "raise volume to {number}",
            "decrease volume to {number}", "lower volume to {number}",
            "volume to {number}"),
           {"number": "number"}),
    Intent("play_pause", "media",
           ("play", "pause", "play pause", "resume", "play music",
            "pause music", "stop music")),
    Intent("next_track", "media",
           ("next track", "next song", "skip song", "skip track")),
    Intent("prev_track", "media",
           ("previous track", "previous song", "last song", "go back song")),

    # ── Windows & tabs ─────────────────────────────────────────────────────────
    Intent("minimize", "window", ("minimize", "minimize window", "minimize this")),
    Intent("maximize", "window", ("maximize", "maximize window", "full screen")),
    Intent("close_window", "window",
           ("close window", "close this window", "close current window")),
    Intent("switch_window", "window",
           ("switch window", "switch windows", "next window", "change window")),
    Intent("show_desktop", "window", ("show desktop", "go to desktop")),
    Intent("new_tab", "browser", ("new tab", "open new tab")),
    Intent("close_tab", "browser", ("close tab", "close this tab")),
    Intent("next_tab", "browser", ("next tab", "switch tab")),
    Intent("prev_tab", "browser", ("previous tab", "last tab")),
    Intent("browser_back", "browser", ("go back", "navigate back", "back page")),
    Intent("browser_forward", "browser",
           ("go forward", "navigate forward", "forward page")),
    Intent("refresh", "browser",
           ("refresh", "reload", "refresh page", "reload page")),

    # ── System ─────────────────────────────────────────────────────────────────
    Intent("screenshot", "screenshot",
           ("take screenshot", "take a screenshot", "screenshot",
            "capture screen")),
    Intent("lock", "lock",
           ("lock screen", "lock computer", "lock my computer", "lock pc")),
    Intent("brightness_up", "brightness",
           ("brightness up", "increase brightness", "brighter screen")),
    Intent("brightness_down", "brightness",
           ("brightness down", "decrease brightness", "dim screen",
            "dim the screen")),
    Intent("sleep", "power", ("go to sleep", "sleep computer", "suspend")),
    Intent("shutdown", "power",
           ("shut down", "shutdown", "shut down computer", "power off",
            "turn off computer"),
           confirm=True),
    Intent("restart", "power",
           ("restart", "reboot", "restart computer", "reboot computer"),
           confirm=True),

    # ── Handled by the manager, not the executor ──────────────────────────────
    Intent("set_language", "manager",
           ("switch language to {text}", "change language to {text}",
            "set language to {text}"),
           {"text": "text"}),
    Intent("start_dictation", "manager",
           ("start dictation", "start dictating", "begin dictation",
            "dictation mode")),
    Intent("stop_dictation", "manager",
           ("stop dictation", "end dictation", "stop typing",
            "cancel dictation")),
    Intent("confirm", "manager", ("yes", "confirm", "go ahead", "do it")),
    Intent("cancel", "manager", ("no", "cancel", "never mind", "abort")),
)

BY_NAME: dict[str, Intent] = {i.name: i for i in CATALOG}
