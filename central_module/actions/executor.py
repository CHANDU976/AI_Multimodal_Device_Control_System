"""
executor.py
Executes an IntentMatch through the OS adapter, and closes the adaptation
loop: successful launches teach the app-alias table, every outcome is
reported back to the caller for behaviour recording.

Dry-run mode never touches the adapter — it reports what *would* happen,
which is how the NLP → action path is tested on any machine.
"""

from __future__ import annotations

import logging
import shutil
import webbrowser
from dataclasses import dataclass
from urllib.parse import quote_plus

from central_module.actions.os_adapter import ActionError, OSAdapter
from central_module.config import ACTIONS
from central_module.learning.adaptation import BehaviorModel
from central_module.learning.preferences import PreferenceStore
from central_module.nlp.intent_engine import IntentMatch

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    ok: bool
    message: str
    needs_confirm: bool = False


class ActionExecutor:

    def __init__(
        self,
        adapter: OSAdapter | None,
        prefs: PreferenceStore,
        behavior: BehaviorModel,
        dry_run: bool = False,
    ) -> None:
        self._adapter = adapter
        self._prefs = prefs
        self._behavior = behavior
        self._dry_run = dry_run

    # ── Entry point ────────────────────────────────────────────────────────────

    def execute(self, match: IntentMatch, confirmed: bool = False) -> ActionResult:
        intent = match.intent
        if intent.confirm and not confirmed:
            return ActionResult(
                True, f"say 'yes' to confirm {intent.name.replace('_', ' ')}",
                needs_confirm=True)

        handler = getattr(self, f"_do_{intent.action}", None)
        if handler is None:
            return ActionResult(False, f"no handler for action '{intent.action}'")

        if self._dry_run:
            if intent.name == "open_app":       # keep adaptation stats live
                self._behavior.record_app(match.slots.get("app", ""))
            return ActionResult(
                True, f"[dry-run] {intent.name} {match.slots or ''}".rstrip())

        try:
            message = handler(match) or intent.name.replace("_", " ")
            return ActionResult(True, message)
        except ActionError as exc:
            return ActionResult(False, str(exc))
        except Exception:
            logger.exception("action %s crashed", intent.name)
            return ActionResult(False, f"{intent.name} failed unexpectedly")

    # ── Apps & web ─────────────────────────────────────────────────────────────

    def _resolve_app(self, spoken: str) -> tuple[str, str]:
        """spoken name → (launch command, source). User's aliases win."""
        alias = self._prefs.get_app_alias(spoken)
        if alias:
            return alias, "alias"
        builtin = getattr(self._adapter, "APP_COMMANDS", {}).get(spoken)
        if builtin:
            return builtin, "builtin"
        exe = shutil.which(spoken) or shutil.which(spoken.replace(" ", ""))
        if exe:
            return exe, "which"
        return spoken, "raw"        # last resort: let the OS resolver try

    def _do_open_app(self, match: IntentMatch) -> str:
        spoken = match.slots["app"]
        command, source = self._resolve_app(spoken)
        self._adapter.open_app(command)
        # Discovered resolutions become preferences, so the next "open …"
        # for this name skips discovery entirely.
        if source == "which":
            self._prefs.learn_app_alias(spoken, command)
        self._behavior.record_app(spoken)
        return f"opened {spoken}"

    def _do_search_web(self, match: IntentMatch) -> str:
        query = match.slots["text"]
        webbrowser.open(f"https://www.google.com/search?q={quote_plus(query)}")
        return f"searching for {query}"

    def _do_open_url(self, match: IntentMatch) -> str:
        # Spoken URLs: "github dot com" → github.com; bare names get .com.
        target = match.slots["text"].replace(" dot ", ".").replace(" ", "")
        if "." not in target:
            target += ".com"
        if "://" not in target:
            target = "https://" + target
        webbrowser.open(target)
        return f"opening {target}"

    # ── Typing & keys ──────────────────────────────────────────────────────────

    def _do_type_text(self, match: IntentMatch) -> str:
        text = match.slots["text"]
        self._adapter.type_text(text)
        return f"typed {len(text.split())} words"

    def _do_press_key(self, match: IntentMatch) -> str:
        key = match.slots["key"]
        self._adapter.key_tap(key)
        return f"pressed {key}"

    def _do_edit_combo(self, match: IntentMatch) -> str:
        self._adapter.edit_combo(match.intent.name)
        return match.intent.name.replace("_", " ")

    def _do_delete_key(self, match: IntentMatch) -> str:
        # "delete" = the delete key (removes the selection / next character).
        # Deleting *files* by voice is deliberately not a capability.
        self._adapter.key_tap("delete")
        return "delete"

    # ── Cursor ─────────────────────────────────────────────────────────────────

    def _do_scroll(self, match: IntentMatch) -> str:
        direction = match.slots["direction"]
        steps = match.slots.get("count", ACTIONS["scroll_steps"])
        self._adapter.scroll(direction, steps)
        return f"scrolled {direction}"

    def _do_click(self, match: IntentMatch) -> str:
        name = match.intent.name
        button = "right" if name == "right_click" else "left"
        self._adapter.click(button=button, double=(name == "double_click"))
        return name.replace("_", " ")

    # ── Volume & media ─────────────────────────────────────────────────────────

    def _do_volume(self, match: IntentMatch) -> str:
        name = match.intent.name
        if name == "volume_up":
            self._adapter.volume_change(self._volume_steps(match))
        elif name == "volume_down":
            self._adapter.volume_change(-self._volume_steps(match))
        elif name == "mute":
            self._adapter.volume_mute(True)
        elif name == "unmute":
            self._adapter.volume_mute(False)
        elif name == "set_volume":
            level = match.slots["number"]
            self._adapter.set_volume(level)
            return f"volume set to {level}"
        return name.replace("_", " ")

    @staticmethod
    def _volume_steps(match: IntentMatch) -> int:
        """Spoken counts are percent; one media-key press moves ~2 units."""
        count = match.slots.get("count")
        if count is None:
            return ACTIONS["volume_steps"]
        return max(1, round(count / 2))

    def _do_media(self, match: IntentMatch) -> str:
        op = {"play_pause": "play_pause", "next_track": "next",
              "prev_track": "prev"}[match.intent.name]
        self._adapter.media(op)
        return match.intent.name.replace("_", " ")

    # ── Windows & tabs ─────────────────────────────────────────────────────────

    def _do_window(self, match: IntentMatch) -> str:
        op = {"minimize": "minimize", "maximize": "maximize",
              "close_window": "close", "switch_window": "switch",
              "show_desktop": "show_desktop"}[match.intent.name]
        self._adapter.window(op)
        return match.intent.name.replace("_", " ")

    def _do_browser(self, match: IntentMatch) -> str:
        self._adapter.browser(match.intent.name)
        return match.intent.name.replace("_", " ")

    # ── System ─────────────────────────────────────────────────────────────────

    def _do_screenshot(self, match: IntentMatch) -> str:
        return f"screenshot {self._adapter.screenshot()}"

    def _do_lock(self, match: IntentMatch) -> str:
        self._adapter.lock()
        return "locking"

    def _do_brightness(self, match: IntentMatch) -> str:
        delta = 15 if match.intent.name == "brightness_up" else -15
        self._adapter.brightness(delta)
        return match.intent.name.replace("_", " ")

    def _do_power(self, match: IntentMatch) -> str:
        op = {"sleep": "sleep", "shutdown": "shutdown",
              "restart": "restart"}[match.intent.name]
        self._adapter.power(op, ACTIONS["power_delay_s"])
        return f"{op} in {ACTIONS['power_delay_s']}s" if op != "sleep" else "sleeping"
