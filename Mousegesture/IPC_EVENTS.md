# Gesture protocol events (IPC-ready, no transport)

`gesture_events.py` builds JSON-ready event messages; `hand_cursor_2.py`
reports state changes into it. **No IPC is implemented.** The integration
step left for later is one line:

```python
engine.events.set_sink(ipc_queue.put_nowait)   # or HandTrackerEngine(event_sink=...)
```

With no sink installed (the default, and what `app.py` does today) the
tracker keeps state but builds no messages. The sink is called on the
camera thread and **must not block**. A sink that raises is caught and
counted (`events.sink_errors`); it never stops tracking.

## Envelope

```json
{"type": "gesture_fired", "timestamp": 1790000000.123, "source": "gesture",
 "version": 1, "payload": {}}
```

* `timestamp` — Unix seconds (`time.time()`), rounded to milliseconds.
* Every payload key is always present; unknown values are `null`.
* Payloads accept only primitives. Arrays, frames, landmarks, objects and
  large/deep containers raise `TypeError`. NaN/inf become `null`.

## Events and payloads

| type | payload | emitted from |
|---|---|---|
| `ready` | `status`, `module`, `camera_index`, `semantic`, `cursor_control` | `_run`: first fresh camera frame after each `start()` |
| `pointing_started` | `hand`, `hand_present`, `cursor_control_active`, `cursor_control` | `_run`: the cursor gate (`_cursor_gate_open`) turns true while a hand is tracked |
| `pointing_stopped` | `reason`, `hand`, `hand_present`, `cursor_control_active`, `cursor_control` | gate turns false, hand lost, camera stall, `stop()`, fatal, `close()` |
| `gesture_fired` | `gesture`, `from_gesture`, `trigger`, `hand`, `action`, `confidence`, `rule_id`, `rule_name`, `stream`, `executed` | each `ActionEvent` the geometric or semantic `GestureFSM` fires, at its `actions.dispatch()` call |
| `drag_started` | `gesture`, `hand`, `action`, `button`, `rule_id`, `rule_name`, `stream` | `ActionDispatcher.dragging` goes false→true on a dispatch |
| `drag_stopped` | `reason`, `gesture`, `hand`, `button`, `rule_id`, `rule_name`, `duration_sec` | `ActionDispatcher.dragging` goes true→false (any release path) |
| `fatal` | `code`, `message` (one line, ≤200 chars) | `_run`'s outer `except`: the tracking loop crashed |
| `heartbeat` | `status`, `pointing`, `dragging`, `hand_present`, `camera`, `semantic`, `fps` | `_run`, at most once per second while ready |

Values:

* `semantic`: `ready` / `loading` / `unavailable` / `disabled`.
* `camera`: `live` / `stalled`.
* `stream`: `geometry` / `semantic`.
* `trigger`: `transition` / `hold`.
* `hand`: `left` / `right` / `null`.

Pointing and drag `reason` values:

* `hand_lost`
* `cursor_gate_closed`: the cursor-control pose, hand filter or mode is no longer met.
* `camera_stalled`: no new frame for 1 s.
* `engine_stopped`
* `config_reloaded`
* `gesture`: an explicit DRAG_STOP mapping fired.
* `fsm_failsafe`: the FSM's stuck-release, timeout or reset path.
* `fatal`
* `shutdown`

## Semantics worth knowing

* **Pointing** means a hand is tracked *and* the cursor gate is open. The
  gate is forced open during a drag, as it already was. Landmark frames
  rejected as implausible change nothing.
* **gesture_fired** is one message per FSM firing, with no debounce. Two
  real firings give two messages, and a held pose with no firing gives
  none. A repeating hold (scroll, volume) fires repeatedly, so it reports
  repeatedly.
  * `action` is the action actually fired, e.g. `DOUBLE_CLICK` when a
    click is promoted. It and the gesture names are read from the rule in
    the machine in force at that moment, so reloads never leave stale
    names.
  * `executed` says whether `ActionDispatcher` accepted it.
  * `DRAG_START`/`DRAG_STOP` firings are reported as `drag_*` only, not
    also as `gesture_fired`.
* **confidence** is the YOLO score of the frame that was fed to the
  semantic FSM, but only when that pose is the pose the rule fired on.
  Geometry firings, landmark-synthesised labels, and holds that mature on
  a different frame's label all report `null`.
* **Geometry/semantic overlap.** The engine already feeds the semantic FSM
  an abstention for geometry poses, so one physical gesture fires once and
  reports once. Two `ActionEvent`s would mean two real OS actions, and
  both are reported.
* **Lifecycle.** `start()` calls `begin()`. `ready` follows on the first
  fresh frame, and only then may pointing, gestures and heartbeats be
  reported.
  * `stop()` and a normal loop exit call `end()`: pointing is reported
    stopped and the lifecycle closes.
  * A drag is reported stopped only when the button really came up. A
    release the backend refused stays reported active until a later
    release succeeds or `close()` runs.
  * `close()` is terminal: anything still active is reported stopped
    with reason `shutdown`, and nothing is emitted afterwards.
* **Not fatal:**
  * camera-open failure in `start()`, which the GUI reports and lets you retry;
  * a missing or unloadable YOLO model, which degrades `semantic`;
  * a binding or cursor-output failure, which the loop already catches;
  * hand loss;
  * a camera stall.

## Physical-mouse pointing evidence (for future fusion)

**What is observable here.** Only programmatic output: the engine knows
when it moves the cursor. It calls `cursor.move()` only on frames where
pointing is reported active. The one exception is a single corrective
move on the frame the gate closes (`CursorClutch` snapping back to its
rest point).

**What is missing.** Nothing in this module observes physical mouse
movement. `pynput` is used for output, and the only listener
(`gesture_test.py`) records clicks during the test bench. A position
read cannot separate physical motion from our own `SetCursorPos`.
Telling them apart reliably needs an OS-level hook that sees the
injected-input flag (Windows `LLMHF_INJECTED` in a `WH_MOUSE_LL` hook),
and no such mechanism exists here. Adding one was out of scope.

**Minimal interface for later.** The fusion layer should own a physical
mouse monitor that emits something like `{"type": "cursor_dwell",
"source": "mouse", ...}` in this same envelope. It should treat
`pointing_started`/`pointing_stopped` as the windows during which the
gesture module may be moving the cursor itself. Movement seen outside
those windows cannot have come from this module. Gesture events are
evidence, not a dependency: physical-mouse dwell must work with the
gesture module absent.

## Tests

```
python test_gesture_events.py
```

Covers schema, tracker transitions, and the real `_run` loop driven by a
scripted camera, hand and classifier with a recording cursor. No camera,
model, real mouse or keyboard is used.
