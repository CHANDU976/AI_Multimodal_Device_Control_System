"""
manager.py
CentralManager — the decision maker in the middle of the system.

Owns the supervisor (speech + gesture children), consumes their events from
one queue, and routes every final utterance through:

    custom-command expansion → pending-confirmation check → dictation mode
    → IntentEngine → ActionExecutor → BehaviorModel recording

Both children speak one protocol (docs/ipc-protocol.md): every message is
{"type", "module", "timestamp", "version", "payload"}. Gestures reported by
the gesture app ("gesture_fired") are mapped through preferences
"gesture_actions" into the same pipeline, so a fist can mean "pause" without
the gesture app knowing what pause does.

State the manager holds deliberately, not the children:
  * dictation mode   — while ON, utterances are typed, not parsed
  * pending confirm  — armed shutdown/restart waiting for "yes"
Both are conversation state, which belongs with the decision maker.
"""

from __future__ import annotations

import logging
import queue
import time

from central_module import config
from central_module.actions.executor import ActionExecutor, ActionResult
from central_module.actions.os_adapter import ActionError, get_adapter
from central_module.learning.adaptation import BehaviorModel
from central_module.learning.preferences import PreferenceStore
from central_module.nlp.intent_engine import IntentEngine, IntentMatch
from central_module.nlp.normalize import normalize
from central_module.supervisor import ChildSpec, ProcessSupervisor

logger = logging.getLogger(__name__)

# Gesture-module state changes (docs/ipc-protocol.md). Fusion will read them.
GESTURE_STATE_EVENTS = frozenset({
    "pointing_started", "pointing_stopped", "drag_started", "drag_stopped",
})


class CentralManager:

    def __init__(
        self,
        dry_run: bool = False,
        verbose: bool = False,
        enable_speech: bool = True,
        force_gesture_stub: bool = False,
    ) -> None:
        self._dry_run = dry_run
        self._verbose = verbose
        self._enable_speech = enable_speech
        self._force_gesture_stub = force_gesture_stub

        self.prefs = PreferenceStore()
        self.behavior = BehaviorModel()
        self.engine = IntentEngine(alias_provider=self.behavior.get_alias)
        logger.info("intent engine mode: %s", self.engine.mode)

        adapter = None
        if not dry_run:
            try:
                adapter = get_adapter()
            except Exception:
                logger.exception("OS adapter unavailable — dropping to dry-run")
                self._dry_run = dry_run = True
        self.executor = ActionExecutor(adapter, self.prefs, self.behavior,
                                       dry_run=dry_run)
        self._adapter = adapter

        self._events: queue.Queue[tuple[str, dict]] = queue.Queue(maxsize=500)
        self.supervisor = ProcessSupervisor(on_event=self._enqueue)

        self._dictation_on = False
        self._pending: tuple[IntentMatch, float] | None = None
        self._running = False

    # ── Child wiring ───────────────────────────────────────────────────────────

    def _enqueue(self, name: str, msg: dict) -> None:
        try:
            self._events.put_nowait((name, msg))
        except queue.Full:
            logger.warning("event queue full — dropping %s event", name)

    def _build_children(self) -> None:
        if self._enable_speech and config.SPEECH_CHILD["enabled"]:
            self.supervisor.add(ChildSpec(
                name="speech",
                argv=config.SPEECH_CHILD["argv"],
                cwd=config.SPEECH_CHILD["cwd"],
                grace_s=config.SPEECH_CHILD["grace_s"],
            ))

        # Gesture: the user's configured app wins; the bundled stub only runs
        # when explicitly forced (it exists to exercise the supervisor path).
        gesture_cmd = self.prefs.get("gesture.command") or []
        if self.prefs.get("gesture.enabled") and gesture_cmd:
            self.supervisor.add(ChildSpec(
                name="gesture", argv=list(gesture_cmd),
                cwd=self.prefs.get("gesture.cwd"),
                grace_s=config.GESTURE_CHILD["grace_s"],
            ))
        elif self._force_gesture_stub or config.GESTURE_CHILD["enabled"]:
            self.supervisor.add(ChildSpec(
                name="gesture", argv=config.GESTURE_CHILD["argv"],
                cwd=config.GESTURE_CHILD["cwd"],
                grace_s=config.GESTURE_CHILD["grace_s"],
            ))

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._build_children()
        self._running = True
        self.supervisor.start_all()

        # Restore the user's language once the speech child is up; harmless
        # if it matches the default.
        lang = self.prefs.get("language")
        if lang and lang != "en-in":
            self.supervisor.send("speech", {"op": "set_language", "lang": lang})

    def stop(self) -> None:
        self._running = False
        self.supervisor.stop_all()
        self.behavior.save()
        logger.info("central manager stopped.")

    def run_forever(self) -> None:
        """Main loop: pump child events until Ctrl+C."""
        self.start()
        self._say("central module running — speak a command (Ctrl+C to exit)")
        try:
            while True:
                try:
                    name, msg = self._events.get(timeout=0.5)
                except queue.Empty:
                    continue
                self._dispatch(name, msg)
        except KeyboardInterrupt:
            self._say("shutting down …")
        finally:
            self.stop()

    # ── Event dispatch ─────────────────────────────────────────────────────────

    def _dispatch(self, child: str, msg: dict) -> None:
        kind = msg.get("type")
        if not kind:                    # a blank line parses to {}
            return
        # Children wrap their data in the shared envelope; the supervisor's
        # own "lifecycle" and "log" messages are flat and carry none.
        payload = msg.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        version = msg.get("version")
        if version is not None and version != config.IPC_PROTOCOL_VERSION:
            logger.warning("%s speaks protocol version %r; expected %r",
                           child, version, config.IPC_PROTOCOL_VERSION)

        if kind in ("command", "dictation"):
            text = (payload.get("text") or "").strip()
            if text:
                self._say(f"heard: {text!r}")
                self.handle_text(text, origin=child)
        elif kind == "partial":
            if self._verbose:
                print(f"  ? {payload.get('text') or ''}", flush=True)
        elif kind == "gesture_fired":
            self._handle_gesture(payload)
        elif kind in GESTURE_STATE_EVENTS:
            # Pointing and drag state: evidence for speech-gesture fusion,
            # which is not built yet. Accepted, not acted on.
            logger.debug("[%s] %s %s", child, kind, payload)
        elif kind == "ready":
            self._say(f"{child} module ready")
        elif kind == "lifecycle":
            state = msg.get("state")
            if state in ("exited", "restarting", "gave_up"):
                self._say(f"{child} {state} {msg.get('returncode', '')}".rstrip())
        elif kind == "fatal":
            reason = payload.get("message") or payload.get("code")
            self._say(f"{child} failed: {reason}")
        elif kind == "device":
            logger.info("audio device change: %s", payload.get("metadata"))
        elif kind == "error":
            logger.warning("%s error: %s", child, payload.get("metadata"))
        elif kind in ("log", "pong", "devices", "heartbeat"):
            logger.debug("[%s] %s", child, msg)
        else:
            logger.warning("[%s] unhandled message type %r", child, kind)

    def _handle_gesture(self, payload: dict) -> None:
        name = payload.get("gesture") or ""
        mapped = (self.prefs.get("gesture_actions") or {}).get(name)
        if mapped:
            self._say(f"gesture {name!r} → {mapped!r}")
            self.handle_text(mapped, origin="gesture")
        else:
            logger.info("unmapped gesture: %r", name)

    # ── The core text pipeline ─────────────────────────────────────────────────

    def handle_text(self, text: str, origin: str = "speech") -> None:
        """Guard wrapper: one bad utterance must never take the manager down."""
        try:
            self._handle_text(text, origin)
        except Exception:
            logger.exception("failed handling %r (from %s)", text, origin)
            self._say(f"something went wrong handling {text!r} — see log")

    def _handle_text(self, text: str, origin: str) -> None:
        light, cmd = normalize(text)
        if not cmd:
            return

        # 0. User-defined macros expand to their canonical phrase.
        custom = (self.prefs.get("custom_commands") or {}).get(cmd)
        if custom:
            light, cmd = normalize(custom)
            text = custom

        # 1. Armed destructive action: only an explicit yes fires it,
        #    anything else disarms it (then processes normally).
        if self._pending:
            match, armed_at = self._pending
            fresh = time.monotonic() - armed_at <= config.PENDING_TIMEOUT_S
            self._pending = None
            if fresh and cmd in config.CONFIRM_WORDS:
                self._report(cmd, match, self.executor.execute(match, confirmed=True))
                return
            if fresh and cmd in config.CANCEL_WORDS:
                self._say(f"{match.intent.name} cancelled")
                return
            if fresh:
                self._say(f"{match.intent.name} cancelled (no confirmation)")

        # 2. Dictation mode transitions and typing.
        if cmd in config.DICTATION_STOP:
            self._dictation_on = False
            self._say("dictation off")
            return
        if cmd in config.DICTATION_START:
            self._dictation_on = True
            self._say("dictation on — everything you say will be typed")
            return
        if self._dictation_on:
            self._type_dictation(text)
            return

        # 3. Intent parsing.
        match = self.engine.parse(text)
        if match is None:
            self.behavior.record_unknown(cmd)
            self._say(f"didn't understand {cmd!r}")
            return

        # 4. Manager-level intents (need the children or manager state).
        if match.intent.action == "manager":
            self._manager_intent(match, cmd)
            return

        # 5. Everything else goes to the OS.
        result = self.executor.execute(match)
        if result.needs_confirm:
            self._pending = (match, time.monotonic())
            self._say(result.message)
            return
        self.behavior.record_intent(cmd, match.intent.name, match.slots,
                                    match.source, result.ok)
        self._report(cmd, match, result)

    def _manager_intent(self, match: IntentMatch, cmd: str) -> None:
        name = match.intent.name
        if name == "start_dictation":
            self._dictation_on = True
            self._say("dictation on")
        elif name == "stop_dictation":
            self._dictation_on = False
            self._say("dictation off")
        elif name == "set_language":
            spoken = match.slots.get("text", "").strip()
            lang = config.LANGUAGES.get(spoken)
            if not lang:
                self._say(f"unknown language {spoken!r} — try: "
                          + ", ".join(sorted(set(config.LANGUAGES))))
                return
            if self.supervisor.send("speech", {"op": "set_language",
                                               "lang": lang}):
                self.prefs.set("language", lang)
                self._say(f"language switched to {spoken}")
            else:
                self._say("speech module is not running")
        elif name in ("confirm", "cancel"):
            pass        # meaningful only while something is armed (step 1)

    def _type_dictation(self, text: str) -> None:
        if self._dry_run or self._adapter is None:
            self._say(f"[dry-run] would type: {text!r}")
            return
        try:
            self._adapter.type_text(text + " ")
        except ActionError as exc:
            self._say(f"typing failed: {exc}")

    # ── Output ─────────────────────────────────────────────────────────────────

    def _report(self, cmd: str, match: IntentMatch, result: ActionResult) -> None:
        mark = "✓" if result.ok else "✗"
        via = "" if match.source == "exact" else f"  (via {match.source})"
        self._say(f"{mark} {result.message}{via}")

    @staticmethod
    def _say(message: str) -> None:
        """User-facing console line (the module's terminal UI)."""
        print(f"» {message}", flush=True)

    # ── Simulate mode (no children, no mic) ────────────────────────────────────

    def run_simulated(self) -> None:
        """
        Type commands instead of speaking them — exercises the full NLP →
        action → adaptation path with no models or microphone.
        """
        self._say("simulate mode — type commands ('quit' to exit, "
                  "'stats' for adaptation stats)")
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.lower() in ("quit", "exit"):
                break
            if line.lower() == "stats":
                print(self.behavior.summary())
                continue
            self.handle_text(line, origin="keyboard")
        self.behavior.save()
