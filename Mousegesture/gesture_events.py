"""IPC-ready event messages for the gesture module.

WHAT THIS IS.  The single place where the gesture module's protocol
messages are built.  Every message is a plain dict that the standard
`json` module can serialise as-is, wrapped in one envelope:

    {
      "type": "gesture_fired",
      "module": "gesture",
      "timestamp": 1790000000.123,     # Unix seconds, millisecond precision
      "version": 1,
      "payload": { ... }               # flat, compact, primitives only
    }

The envelope is shared with the speech module and defined in
docs/ipc-protocol.md; ready, fatal, heartbeat and pong are shared types
with the same payload keys in every module.

WHAT THIS IS NOT.  A transport.  Nothing here opens a socket, a pipe or
a queue, writes a file, or prints a message.  GestureEventTracker hands
each finished dict to a `sink` callable the integrator supplies -- the
future IPC layer passes something like `queue.put_nowait` -- and with no
sink installed it builds nothing at all.

The tracker sits AFTER the engine's own decisions and never makes one:
it is told "the cursor gate opened", "this ActionEvent was dispatched",
"the button was released", and turns those already-resolved facts into
at most one message per real state change.  It runs no model, reads no
landmarks and holds no frame.

Payload keys are always present.  A value the runtime does not know is
null, never guessed -- confidence in particular is null whenever no
classifier score genuinely belongs to the gesture that fired.

Pure standard library, so it imports in a test without OpenCV,
MediaPipe or pynput.
"""

from __future__ import annotations

import enum
import math
import numbers
import threading
import time

__all__ = [
    "PROTOCOL_VERSION", "MODULE", "EVENT_TYPES",
    "POINTING_STARTED", "POINTING_STOPPED", "GESTURE_FIRED",
    "DRAG_STARTED", "DRAG_STOPPED", "READY", "FATAL", "HEARTBEAT", "PONG",
    "HEARTBEAT_INTERVAL_SEC", "CAMERA_STALL_SEC",
    "build_message", "pointing_started", "pointing_stopped",
    "gesture_fired", "drag_started", "drag_stopped", "ready", "fatal",
    "heartbeat", "pong", "describe_rule", "GestureEventTracker",
]

# ─── Protocol constants ─────────────────────────────────────────────────────

PROTOCOL_VERSION = 1
MODULE = "gesture"              # the envelope's "module"; the supervisor's child name

POINTING_STARTED = "pointing_started"
POINTING_STOPPED = "pointing_stopped"
GESTURE_FIRED = "gesture_fired"
DRAG_STARTED = "drag_started"
DRAG_STOPPED = "drag_stopped"
READY = "ready"
FATAL = "fatal"
HEARTBEAT = "heartbeat"
PONG = "pong"                   # reply to the parent's {"op": "ping"}

EVENT_TYPES = frozenset({
    POINTING_STARTED, POINTING_STOPPED, GESTURE_FIRED, DRAG_STARTED,
    DRAG_STOPPED, READY, FATAL, HEARTBEAT, PONG,
})

# Why pointing or a drag ended.  Stable strings: a consumer may switch on
# them.  Only reasons the engine can actually tell apart are listed.
REASON_HAND_LOST = "hand_lost"              # MediaPipe lost the hand
REASON_GATE_CLOSED = "cursor_gate_closed"   # cursor-control pose / hand
                                            # filter / mode no longer met
REASON_CAMERA_STALLED = "camera_stalled"    # no new frame for a while
REASON_ENGINE_STOPPED = "engine_stopped"    # stop(): disconnect / switch
REASON_CONFIG_RELOADED = "config_reloaded"  # Save Config dropped the drag
REASON_GESTURE = "gesture"                  # a DRAG_STOP mapping fired
REASON_FSM_FAILSAFE = "fsm_failsafe"        # FSM stuck-release / timeout
REASON_FATAL = "fatal"                      # tracking loop crashed
REASON_SHUTDOWN = "shutdown"                # engine closed for good

# Fatal codes.
FATAL_TRACKING_LOOP = "tracking_loop_failed"

# Heartbeats are rate-limited to this interval.
HEARTBEAT_INTERVAL_SEC = 1.0

# A running loop that has seen no new camera frame for this long is
# reported as stalled.  WebcamStream keeps serving its last frame when a
# device stops delivering, so without this an unplugged camera would leave
# pointing reported active indefinitely.  30 fps cameras deliver every
# ~33 ms, so a full second is far outside normal jitter.
CAMERA_STALL_SEC = 1.0

# Fatal messages are cut to this many characters: a reason, not a log.
MAX_TEXT_CHARS = 200

# Payloads are flat by design.  These caps make "compact" enforceable: a
# list of 21 landmarks, a frame, or a nested diagnostic dump is refused
# with a TypeError rather than serialised.
_MAX_DEPTH = 2
_MAX_ITEMS = 16


# ─── Value sanitising ───────────────────────────────────────────────────────

def _plain(value, depth: int = 0):
    """`value` as a JSON primitive, or TypeError.  Never passes objects.

    str subclasses (gesture_fsm.ActionEvent is one) become a plain str,
    enums become their value, numeric scalars of any library become int
    or float, and a non-finite float becomes None so the output stays
    strict JSON.  Containers are allowed only small and shallow.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, enum.Enum):
        return _plain(value.value, depth)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if depth < _MAX_DEPTH:
        if isinstance(value, dict) and len(value) <= _MAX_ITEMS:
            return {str(key): _plain(item, depth + 1)
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)) and len(value) <= _MAX_ITEMS:
            return [_plain(item, depth + 1) for item in value]
    raise TypeError(f"{type(value).__name__} is not allowed in a gesture "
                    f"event payload")


def _hand(value):
    """'left' / 'right' / 'both', or None when the side is unknown."""
    side = str(value or "").strip().lower()
    return side if side in ("left", "right", "both") else None


def _text(value, limit: int = MAX_TEXT_CHARS):
    """One line of at most `limit` characters, or None."""
    if value is None:
        return None
    line = " ".join(str(value).split())
    return line if len(line) <= limit else line[:limit - 1] + "…"


def _confidence(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 3) if math.isfinite(number) else None


# ─── The envelope ───────────────────────────────────────────────────────────

def build_message(event_type: str, payload=None, *, timestamp=None) -> dict:
    """THE envelope.  Every constructor below goes through here."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown gesture event type: {event_type!r}")
    when = time.time() if timestamp is None else float(timestamp)
    return {
        "type": event_type,
        "module": MODULE,
        "timestamp": round(when, 3),
        "version": PROTOCOL_VERSION,
        "payload": _plain(dict(payload or {})),
    }


# ─── Event constructors ─────────────────────────────────────────────────────
# Each returns a finished, JSON-ready dict and sends it nowhere.

def pointing_started(*, hand=None, cursor_control=None,
                     timestamp=None) -> dict:
    return build_message(POINTING_STARTED, {
        "hand": _hand(hand),
        "hand_present": True,
        "cursor_control_active": True,
        "cursor_control": cursor_control,
    }, timestamp=timestamp)


def pointing_stopped(*, reason, hand=None, hand_present=None,
                     cursor_control=None, timestamp=None) -> dict:
    return build_message(POINTING_STOPPED, {
        "reason": reason,
        "hand": _hand(hand),
        "hand_present": hand_present,
        "cursor_control_active": False,
        "cursor_control": cursor_control,
    }, timestamp=timestamp)


def gesture_fired(*, gesture, action, hand=None, confidence=None,
                  from_gesture=None, trigger=None, rule_id=None,
                  rule_name=None, stream=None, executed=None,
                  timestamp=None) -> dict:
    return build_message(GESTURE_FIRED, {
        "gesture": gesture,
        "from_gesture": from_gesture,
        "trigger": trigger,
        "hand": _hand(hand),
        "action": action,
        "confidence": _confidence(confidence),
        "rule_id": rule_id,
        "rule_name": rule_name,
        "stream": stream,
        "executed": executed,
    }, timestamp=timestamp)


def drag_started(*, gesture=None, hand=None, rule_id=None, rule_name=None,
                 stream=None, button="left", timestamp=None) -> dict:
    return build_message(DRAG_STARTED, {
        "gesture": gesture,
        "hand": _hand(hand),
        "action": "DRAG_START",
        "button": button,
        "rule_id": rule_id,
        "rule_name": rule_name,
        "stream": stream,
    }, timestamp=timestamp)


def drag_stopped(*, reason, gesture=None, hand=None, rule_id=None,
                 rule_name=None, duration_sec=None, button="left",
                 timestamp=None) -> dict:
    return build_message(DRAG_STOPPED, {
        "reason": reason,
        "gesture": gesture,
        "hand": _hand(hand),
        "button": button,
        "rule_id": rule_id,
        "rule_name": rule_name,
        "duration_sec": (None if duration_sec is None
                         else round(max(0.0, float(duration_sec)), 3)),
    }, timestamp=timestamp)


def ready(*, camera_index=None, semantic=None, cursor_control=None,
          timestamp=None) -> dict:
    return build_message(READY, {
        "status": "ready",
        "camera_index": camera_index,
        "semantic": semantic,
        "cursor_control": cursor_control,
    }, timestamp=timestamp)


def fatal(*, code, message=None, timestamp=None) -> dict:
    return build_message(FATAL, {
        "code": code,
        "message": _text(message),
    }, timestamp=timestamp)


def heartbeat(*, pointing, dragging, hand_present=None, camera=None,
              semantic=None, fps=None, timestamp=None) -> dict:
    return build_message(HEARTBEAT, {
        "status": "running",
        "pointing": bool(pointing),
        "dragging": bool(dragging),
        "hand_present": hand_present,
        "camera": camera,
        "semantic": semantic,
        "fps": None if fps is None else round(float(fps), 1),
    }, timestamp=timestamp)


def pong(*, timestamp=None) -> dict:
    return build_message(PONG, {}, timestamp=timestamp)


# ─── Rule context ───────────────────────────────────────────────────────────

def describe_rule(machine, rule_id) -> dict:
    """Gesture context for an ActionEvent, read off the rule that fired it.

    Looked up in the machine that produced the event, at the moment it is
    reported, so the names always come from the rule set actually in
    force -- nothing is cached across a config reload.  Only called when
    something fired, never per frame.
    """
    empty = {"rule": None, "gesture": None, "from_gesture": None,
             "trigger": None, "rule_action": None}
    if machine is None or rule_id is None:
        return empty
    try:
        rules = machine.rules
    except Exception:
        return empty
    for rule in rules:
        if getattr(rule, "id", None) != rule_id:
            continue
        pose = getattr(rule, "pose", None)
        if pose is not None:                       # HoldRule
            return {"rule": rule, "gesture": pose, "from_gesture": None,
                    "trigger": "hold", "rule_action": rule.action}
        return {"rule": rule,                      # TransitionRule
                "gesture": getattr(rule, "to_state", None),
                "from_gesture": getattr(rule, "from_state", None),
                "trigger": "transition", "rule_action": rule.action}
    return empty


# ─── Transition tracker ─────────────────────────────────────────────────────

class GestureEventTracker:
    """Turns engine state changes into at most one message each.

    THE STATE IS THE REPORTED STATE, nothing more: `pointing`, `dragging`
    and `ready` say what consumers have been told.  Each setter compares
    against it and emits only on a real change, so a pose held for 100
    frames costs 100 attribute reads and no messages.

    Lifecycle, per engine start():

        begin() -> mark_ready() -> ...transitions / heartbeats... -> end()
                                                          or fatal()
        close() is terminal: it reports any active state as stopped and
        silences everything afterwards.

    Thread-safety.  The tracking thread drives most calls; stop(),
    reload_config() and close() arrive from the GUI thread.  Every state
    change happens under one re-entrant lock, and the sink is called
    inside it so messages leave in the order the changes happened.  The
    hot path reads `pointing`, `ready` and `next_heartbeat` WITHOUT the
    lock -- a stale read only costs a call that re-checks under it.

    The sink must not block: it is called on the camera thread.
    """

    def __init__(self, sink=None, *, heartbeat_interval=HEARTBEAT_INTERVAL_SEC,
                 wall_clock=time.time, monotonic=time.perf_counter):
        self._sink = sink
        self._lock = threading.RLock()
        self._interval = max(0.0, float(heartbeat_interval))
        self._wall = wall_clock
        self._mono = monotonic

        # Reported state.  Read lock-free by the loop; written under lock.
        self.pointing = False
        self.dragging = False
        self.ready = False
        self.next_heartbeat = math.inf

        self._active = False          # inside a begin() .. end() lifecycle
        self._closed = False
        self._drag = None             # context of the reported drag
        self.sink_errors = 0
        self._sink_error_shown = False

    # ── plumbing ────────────────────────────────────────────────────────

    @property
    def sink(self):
        return self._sink

    def set_sink(self, sink) -> None:
        """Install (or clear, with None) the consumer of finished messages."""
        with self._lock:
            self._sink = sink

    @property
    def closed(self) -> bool:
        return self._closed

    def _emit(self, factory, **fields):
        """Build and hand over one message.  Never raises.

        Built only when a sink exists: with nobody listening the state is
        still tracked, so attaching a sink later starts from the truth,
        but no dict is ever allocated for nobody.
        """
        sink = self._sink
        if sink is None:
            return None
        try:
            message = factory(timestamp=self._wall(), **fields)
            sink(message)
            return message
        except Exception as exc:
            self.sink_errors += 1
            if not self._sink_error_shown:
                self._sink_error_shown = True
                print(f"[events] message dropped "
                      f"({exc.__class__.__name__}: {exc})")
            return None

    # ── lifecycle ───────────────────────────────────────────────────────

    def begin(self) -> None:
        """A new engine start(): readiness may be reported once more."""
        with self._lock:
            if self._closed:
                return
            self._active = True
            self.ready = False
            self.next_heartbeat = math.inf

    def mark_ready(self, **fields):
        """Report `ready` once per lifecycle.  Later calls are no-ops."""
        with self._lock:
            if self._closed or not self._active or self.ready:
                return None
            self.ready = True
            self.next_heartbeat = self._mono() + self._interval
            return self._emit(ready, **fields)

    def end(self, reason: str = REASON_ENGINE_STOPPED) -> None:
        """The lifecycle is over: nothing may still be reported pointing.

        The drag is deliberately NOT touched.  Its end is reported from
        the actual button release, and a release the backend refused
        leaves the button genuinely held -- which is what is reported
        until a later release succeeds or close() is called.
        """
        with self._lock:
            self._stop_pointing(reason, None, None, None)
            self._active = False
            self.ready = False
            self.next_heartbeat = math.inf

    def fatal(self, code: str, message=None):
        """The lifecycle ended in an unrecoverable error.  Once only."""
        with self._lock:
            if self._closed or not self._active:
                return None
            self._stop_pointing(REASON_FATAL, None, None, None)
            self._active = False
            self.ready = False
            self.next_heartbeat = math.inf
            return self._emit(fatal, code=code, message=message)

    def close(self, reason: str = REASON_SHUTDOWN) -> None:
        """Terminal.  Reports active state as stopped, then goes silent."""
        with self._lock:
            if self._closed:
                return
            self._stop_pointing(reason, None, None, None)
            if self.dragging:
                self._stop_drag(reason)
            self._active = False
            self.ready = False
            self.next_heartbeat = math.inf
            self._closed = True

    # ── pointing ────────────────────────────────────────────────────────

    def set_pointing(self, active: bool, *, hand=None, hand_present=None,
                     reason=REASON_GATE_CLOSED, cursor_control=None):
        """Report the cursor-control state.  Emits only on a change.

        A start is accepted only inside a ready lifecycle, so a frame that
        straggles in after stop() cannot reopen the reported state.
        """
        active = bool(active)
        if active == self.pointing:
            return None                         # the per-frame fast path
        with self._lock:
            if active:
                if self.pointing or self._closed or not self.ready:
                    return None
                self.pointing = True
                return self._emit(pointing_started, hand=hand,
                                  cursor_control=cursor_control)
            return self._stop_pointing(reason, hand, hand_present,
                                       cursor_control)

    def _stop_pointing(self, reason, hand, hand_present, cursor_control):
        if not self.pointing:
            return None
        self.pointing = False
        return self._emit(pointing_stopped, reason=reason, hand=hand,
                          hand_present=hand_present,
                          cursor_control=cursor_control)

    # ── gestures ────────────────────────────────────────────────────────

    def gesture_fired(self, **fields):
        """One message per FSM firing.  No debounce: two firings, two."""
        with self._lock:
            if self._closed or not self.ready:
                return None
            return self._emit(gesture_fired, **fields)

    # ── drag ────────────────────────────────────────────────────────────

    def drag_started(self, *, gesture=None, hand=None, rule_id=None,
                     rule_name=None, stream=None):
        """The button is down.  Reported once until it comes back up."""
        with self._lock:
            if self._closed or self.dragging:
                return None
            self.dragging = True
            self._drag = {"gesture": gesture, "hand": hand,
                          "rule_id": rule_id, "rule_name": rule_name,
                          "started": self._mono()}
            return self._emit(drag_started, gesture=gesture, hand=hand,
                              rule_id=rule_id, rule_name=rule_name,
                              stream=stream)

    def drag_stopped(self, reason: str):
        """The button is up.  No-op unless a drag was reported active."""
        with self._lock:
            if self._closed or not self.dragging:
                return None
            return self._stop_drag(reason)

    def _stop_drag(self, reason):
        context = self._drag or {}
        self.dragging = False
        self._drag = None
        started = context.get("started")
        duration = None if started is None else self._mono() - started
        return self._emit(drag_stopped, reason=reason,
                          gesture=context.get("gesture"),
                          hand=context.get("hand"),
                          rule_id=context.get("rule_id"),
                          rule_name=context.get("rule_name"),
                          duration_sec=duration)

    # ── heartbeat ───────────────────────────────────────────────────────

    def heartbeat(self, now: float, **status):
        """At most one per interval, only while ready.  `now` is monotonic.

        Callers test `now >= tracker.next_heartbeat` first, so the frames
        in between pay one float compare and build no keyword arguments.
        """
        if now < self.next_heartbeat:
            return None
        with self._lock:
            if self._closed or not self.ready or now < self.next_heartbeat:
                return None
            self.next_heartbeat = now + self._interval
            return self._emit(heartbeat, pointing=self.pointing,
                              dragging=self.dragging, **status)
