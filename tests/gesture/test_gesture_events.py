"""Tests for the IPC-ready protocol events (gesture_events.py).

WHAT THIS COVERS.  Two layers:

  * gesture_events on its own -- the envelope, the constructors, and the
    GestureEventTracker's transition rules -- with an injected clock.

  * The real HandTrackerEngine._run loop, driven frame by frame by a
    scripted camera, a scripted hand and (optionally) a scripted YOLO
    worker, with a real GestureFSM pair and a real ActionDispatcher over
    a recording cursor.  What is asserted is the exact message stream the
    loop produces, so the hooks are proven at the points they sit, not
    re-described.

NO CAMERA, NO MODEL, NO REAL MOUSE OR KEYBOARD, NO CONFIG FILE.  Extended
actions go to ActionExecutor(dry_run=True); only LEFT_CLICK, DOUBLE_CLICK
and the drag reach the (recording) cursor.

    python test_gesture_events.py
"""
from __future__ import annotations

import ast
import json
import math
import os
import threading
import types
import unittest

import numpy as np

import adaptive_timing
import gesture_events as ge
import hand_cursor_2
from gesture_fsm import (DRAG_START, DRAG_STOP, LEFT_CLICK, ActionEvent,
                         ActionExecutor, GestureFSM)
from hand_cursor_2 import ActionDispatcher, HandTrackerEngine

# The gesture package, not this file's folder: the test lives in tests/gesture.
MODULE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "Mousegesture")


# ─── Shared fakes ───────────────────────────────────────────────────────────

class Clock:
    """A controllable perf_counter.  sleep() advances it."""

    def __init__(self, start=100.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


class RecordingCursor:
    name = "recording"

    def __init__(self, fail_click=False):
        self.calls = []
        self.fail_click = fail_click

    def move(self, x, y):
        self.calls.append(("move", int(x), int(y)))

    def click(self, button="left", count=1):
        if self.fail_click:
            raise RuntimeError("backend refused")
        self.calls.append(("click", button, count))

    def press(self, button="left"):
        self.calls.append(("press", button))

    def release(self, button="left"):
        self.calls.append(("release", button))

    def close(self):
        pass

    def buttons(self):
        return [c for c in self.calls if c[0] != "move"]


def tracker(clock=None, **kwargs):
    clock = clock or Clock()
    sent = []
    t = ge.GestureEventTracker(sent.append, wall_clock=lambda: 1.79e9,
                               monotonic=clock, **kwargs)
    return t, sent, clock


def types_of(messages):
    return [m["type"] for m in messages]


def rule(rid, frm, to, action, **extra):
    out = {"id": rid, "trigger": "transition", "from_state": frm,
           "to_state": to, "action": action, "enabled": True,
           "cooldown_sec": 0.0, "source": "any", "hand": "any",
           "max_time_sec": 5.0}
    out.update(extra)
    return out


def hold(rid, pose, action, **extra):
    out = {"id": rid, "trigger": "hold", "pose": pose, "action": action,
           "enabled": True, "cooldown_sec": 0.0, "source": "any",
           "hand": "any", "hold_sec": 0.2}
    out.update(extra)
    return out


# ─── 1. Envelope and schema ─────────────────────────────────────────────────

ALL_CONSTRUCTED = [
    lambda: ge.pointing_started(hand="right", cursor_control="always"),
    lambda: ge.pointing_stopped(reason=ge.REASON_HAND_LOST),
    lambda: ge.gesture_fired(gesture="grip", action="LEFT_CLICK"),
    lambda: ge.drag_started(gesture="grip", hand="left"),
    lambda: ge.drag_stopped(reason=ge.REASON_GESTURE, duration_sec=1.5),
    lambda: ge.ready(camera_index=0, semantic="loading"),
    lambda: ge.fatal(code=ge.FATAL_TRACKING_LOOP, message="boom"),
    lambda: ge.heartbeat(pointing=False, dragging=False),
]


class EnvelopeTests(unittest.TestCase):

    def test_every_event_is_json_serialisable_with_one_envelope(self):
        seen = set()
        for make in ALL_CONSTRUCTED:
            message = make()
            text = json.dumps(message, allow_nan=False)    # strict JSON
            self.assertEqual(json.loads(text), message)
            self.assertEqual(set(message),
                             {"type", "timestamp", "source", "version",
                              "payload"})
            self.assertEqual(message["source"], "gesture")
            self.assertEqual(message["version"], ge.PROTOCOL_VERSION)
            self.assertIsInstance(message["payload"], dict)
            seen.add(message["type"])
        self.assertEqual(seen, set(ge.EVENT_TYPES), "a type has no builder")

    def test_timestamp_is_unix_seconds_rounded_to_ms(self):
        message = ge.ready(timestamp=1790000000.12345)
        self.assertEqual(message["timestamp"], 1790000000.123)
        self.assertIsInstance(message["timestamp"], float)
        # Default is wall-clock time, not perf_counter.
        self.assertGreater(ge.ready()["timestamp"], 1.6e9)

    def test_required_payload_keys(self):
        fired = ge.gesture_fired(gesture="g", action="A")["payload"]
        for key in ("gesture", "hand", "action", "confidence", "rule_id",
                    "stream"):
            self.assertIn(key, fired)
        self.assertIn("reason", ge.pointing_stopped(reason="x")["payload"])
        self.assertIn("reason", ge.drag_stopped(reason="x")["payload"])
        self.assertEqual(set(ge.fatal(code="c")["payload"]),
                         {"code", "message"})
        self.assertEqual(ge.ready()["payload"]["status"], "ready")
        self.assertEqual(ge.ready()["payload"]["module"], "mousegesture")

    def test_missing_optionals_are_null_not_absent(self):
        payload = ge.gesture_fired(gesture="g", action=None)["payload"]
        self.assertIsNone(payload["action"])
        self.assertIsNone(payload["confidence"])
        self.assertIsNone(payload["hand"])
        self.assertIsNone(ge.drag_stopped(reason="r")["payload"]
                          ["duration_sec"])

    def test_unknown_or_blank_hand_is_null(self):
        for raw in ("", None, "sideways"):
            self.assertIsNone(ge.pointing_started(hand=raw)["payload"]
                              ["hand"])
        self.assertEqual(ge.pointing_started(hand="Left")["payload"]
                         ["hand"], "left")

    def test_custom_values_become_primitives(self):
        import enum

        class Mode(enum.Enum):
            GATED = "gesture-based"

        event = ActionEvent("LEFT_CLICK", rule_id="r1", rule_name="n")
        payload = ge.gesture_fired(
            gesture="g", action=event, confidence=np.float32(0.9123),
            rule_id=np.int64(7))["payload"]
        self.assertIs(type(payload["action"]), str)
        self.assertEqual(payload["action"], "LEFT_CLICK")
        self.assertEqual(payload["confidence"], 0.912)
        self.assertIs(type(payload["rule_id"]), int)
        started = ge.pointing_started(cursor_control=Mode.GATED)
        self.assertEqual(started["payload"]["cursor_control"],
                         "gesture-based")

    def test_non_finite_numbers_become_null(self):
        self.assertIsNone(ge.gesture_fired(
            gesture="g", action="A", confidence=float("nan"))
            ["payload"]["confidence"])
        self.assertIsNone(ge.heartbeat(pointing=False, dragging=False,
                                       fps=float("inf"))["payload"]["fps"])

    def test_arrays_frames_landmarks_and_objects_are_refused(self):
        landmarks = [(0.1, 0.2, 0.0)] * 21
        for bad in (np.zeros((48, 64, 3), np.uint8), np.zeros(3),
                    landmarks, object(), types.SimpleNamespace(x=1)):
            with self.assertRaises(TypeError):
                ge.build_message(ge.HEARTBEAT, {"leak": bad})

    def test_unknown_event_type_is_refused(self):
        with self.assertRaises(ValueError):
            ge.build_message("cursor_moved", {})

    def test_fatal_message_is_one_short_line(self):
        long = "Traceback (most recent call last):\n" + "x" * 5000
        message = ge.fatal(code="c", message=long)["payload"]["message"]
        self.assertNotIn("\n", message)
        self.assertLessEqual(len(message), ge.MAX_TEXT_CHARS)

    def test_describe_rule_reads_the_machine_in_force(self):
        fsm = GestureFSM({"transitions": [rule("t", "a", "b", "LEFT_CLICK")],
                          "holds": [hold("h", "c", "COPY")]},
                         learner=adaptive_timing.TimingLearner())
        self.addCleanup(fsm._timing.stop)
        t = ge.describe_rule(fsm, "t")
        self.assertEqual((t["gesture"], t["from_gesture"], t["trigger"]),
                         ("b", "a", "transition"))
        h = ge.describe_rule(fsm, "h")
        self.assertEqual((h["gesture"], h["from_gesture"], h["trigger"]),
                         ("c", None, "hold"))
        self.assertIsNone(ge.describe_rule(fsm, "gone")["gesture"])
        self.assertIsNone(ge.describe_rule(None, "t")["gesture"])


# ─── 2. Tracker transitions ─────────────────────────────────────────────────

class TrackerTests(unittest.TestCase):

    def ready_tracker(self):
        t, sent, clock = tracker()
        t.begin()
        t.mark_ready(camera_index=0)
        sent.clear()
        return t, sent, clock

    def test_pointing_start_and_stop_are_edge_triggered(self):
        t, sent, _ = self.ready_tracker()
        t.set_pointing(True, hand="right")
        for _ in range(100):
            t.set_pointing(True, hand="right")
        self.assertEqual(types_of(sent), [ge.POINTING_STARTED])
        t.set_pointing(False, reason=ge.REASON_HAND_LOST)
        for _ in range(100):
            t.set_pointing(False, reason=ge.REASON_HAND_LOST)
        self.assertEqual(types_of(sent),
                         [ge.POINTING_STARTED, ge.POINTING_STOPPED])
        self.assertEqual(sent[1]["payload"]["reason"], ge.REASON_HAND_LOST)

    def test_inactive_to_inactive_emits_nothing(self):
        t, sent, _ = self.ready_tracker()
        t.set_pointing(False)
        self.assertEqual(sent, [])

    def test_pointing_cannot_start_before_ready_or_after_end(self):
        t, sent, _ = tracker()
        t.set_pointing(True)                      # no lifecycle
        t.begin()
        t.set_pointing(True)                      # not ready yet
        self.assertEqual(sent, [])
        t.mark_ready()
        t.end()
        t.set_pointing(True)                      # lifecycle over
        self.assertEqual(types_of(sent), [ge.READY])

    def test_ready_once_per_lifecycle(self):
        t, sent, _ = tracker()
        t.begin()
        for _ in range(5):
            t.mark_ready(camera_index=1)
        t.end()
        t.begin()
        t.mark_ready(camera_index=2)
        self.assertEqual(types_of(sent), [ge.READY, ge.READY])
        self.assertEqual([m["payload"]["camera_index"] for m in sent], [1, 2])

    def test_drag_start_stop_once_each(self):
        t, sent, clock = self.ready_tracker()
        t.drag_started(gesture="grip", hand="left", rule_id="d")
        t.drag_started(gesture="grip", hand="left", rule_id="d")
        clock.advance(2.5)
        t.drag_stopped(ge.REASON_GESTURE)
        t.drag_stopped(ge.REASON_GESTURE)
        self.assertEqual(types_of(sent), [ge.DRAG_STARTED, ge.DRAG_STOPPED])
        stop = sent[1]["payload"]
        self.assertEqual((stop["gesture"], stop["hand"], stop["rule_id"]),
                         ("grip", "left", "d"))
        self.assertEqual(stop["duration_sec"], 2.5)

    def test_stopping_an_inactive_drag_is_silent(self):
        t, sent, _ = self.ready_tracker()
        t.drag_stopped(ge.REASON_HAND_LOST)
        self.assertEqual(sent, [])

    def test_repeated_firings_are_never_debounced(self):
        t, sent, _ = self.ready_tracker()
        t.gesture_fired(gesture="grip", action="LEFT_CLICK")
        t.gesture_fired(gesture="grip", action="LEFT_CLICK")
        self.assertEqual(types_of(sent), [ge.GESTURE_FIRED] * 2)

    def test_heartbeat_rate_limited_to_one_per_second(self):
        t, sent, clock = self.ready_tracker()
        start = clock.now
        for i in range(1, 351):                  # 3.5 s at 100 Hz
            t.heartbeat(start + i / 100.0, camera="live")
        self.assertEqual(types_of(sent), [ge.HEARTBEAT] * 3)

    def test_heartbeat_does_not_repeat_transitions(self):
        t, sent, clock = self.ready_tracker()
        t.set_pointing(True)
        t.heartbeat(clock.now + 1.0)
        t.heartbeat(clock.now + 2.0)
        self.assertEqual(types_of(sent), [ge.POINTING_STARTED,
                                          ge.HEARTBEAT, ge.HEARTBEAT])
        self.assertTrue(all(m["payload"]["pointing"] for m in sent[1:]))

    def test_no_heartbeat_before_ready_or_after_end_or_close(self):
        t, sent, clock = tracker()
        t.begin()
        t.heartbeat(clock.now + 10)
        self.assertEqual(sent, [])
        t.mark_ready()
        t.end()
        t.heartbeat(clock.now + 100)
        t.close()
        t.heartbeat(clock.now + 1000)
        self.assertEqual(types_of(sent), [ge.READY])

    def test_end_stops_pointing_but_leaves_a_held_button_reported(self):
        t, sent, _ = self.ready_tracker()
        t.set_pointing(True)
        t.drag_started()
        t.end(ge.REASON_ENGINE_STOPPED)
        self.assertEqual(types_of(sent), [ge.POINTING_STARTED,
                                          ge.DRAG_STARTED,
                                          ge.POINTING_STOPPED])
        self.assertTrue(t.dragging, "end() claimed a release that did "
                                    "not happen")
        t.drag_stopped(ge.REASON_ENGINE_STOPPED)   # the real release
        self.assertFalse(t.dragging)

    def test_fatal_stops_pointing_then_reports_once(self):
        t, sent, _ = self.ready_tracker()
        t.set_pointing(True)
        t.fatal(ge.FATAL_TRACKING_LOOP, "RuntimeError: x")
        t.fatal(ge.FATAL_TRACKING_LOOP, "again")
        self.assertEqual(types_of(sent), [ge.POINTING_STARTED,
                                          ge.POINTING_STOPPED, ge.FATAL])
        self.assertEqual(sent[1]["payload"]["reason"], ge.REASON_FATAL)
        self.assertFalse(t.ready)

    def test_close_reports_everything_stopped_then_goes_silent(self):
        t, sent, clock = self.ready_tracker()
        t.set_pointing(True)
        t.drag_started()
        t.close()
        before = list(sent)
        t.set_pointing(True)
        t.gesture_fired(gesture="g", action="A")
        t.drag_started()
        t.mark_ready()
        t.begin()
        t.mark_ready()
        t.fatal("c")
        t.heartbeat(clock.now + 99)
        t.close()
        self.assertEqual(sent, before, "an event escaped after close()")
        self.assertEqual(types_of(before)[-2:], [ge.POINTING_STOPPED,
                                                 ge.DRAG_STOPPED])
        self.assertTrue(all(m["payload"]["reason"] == ge.REASON_SHUTDOWN
                            for m in before[-2:]))
        self.assertFalse(t.pointing or t.dragging)

    def test_no_sink_builds_nothing_but_tracks_state(self):
        t = ge.GestureEventTracker(None)
        t.begin()
        self.assertIsNone(t.mark_ready())
        self.assertIsNone(t.set_pointing(True))
        self.assertTrue(t.pointing)
        sent = []
        t.set_sink(sent.append)
        t.set_pointing(False, reason="r")
        self.assertEqual(types_of(sent), [ge.POINTING_STOPPED])

    def test_a_failing_sink_never_raises_into_the_caller(self):
        def broken(_message):
            raise OSError("pipe full")
        t = ge.GestureEventTracker(broken)
        t.begin()
        t.mark_ready()
        t.set_pointing(True)
        t.gesture_fired(gesture="g", action="A")
        self.assertEqual(t.sink_errors, 3)
        self.assertTrue(t.pointing)

    def test_state_changes_are_thread_safe(self):
        t, sent, _ = self.ready_tracker()

        def flap():
            for i in range(2000):
                t.set_pointing(i % 2 == 0, reason="r")

        threads = [threading.Thread(target=flap) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        # Whatever the interleaving, starts and stops strictly alternate.
        kinds = types_of(sent)
        for a, b in zip(kinds, kinds[1:]):
            self.assertNotEqual(a, b, "two starts or two stops in a row")
        self.assertEqual(kinds[0], ge.POINTING_STARTED)


# ─── 3. The real loop, scripted ─────────────────────────────────────────────

class Frame:
    """One camera frame of the script."""

    def __init__(self, gesture=None, side="right", yolo=None, score=None,
                 before=None, crash=False):
        self.gesture = gesture          # geometry pose; None = no hand
        self.side = side
        self.yolo = yolo                # semantic label, or None
        self.score = score
        self.before = before            # callable run before the frame
        self.crash = crash              # MediaPipe raises on this frame


def frames(gesture, n, **kw):
    return [Frame(gesture, **kw) for _ in range(n)]


def stall(seconds):
    """A marker: the camera delivers nothing for `seconds`."""
    return ("stall", float(seconds))


class FakeLandmarks:
    def __init__(self):
        self.landmark = [types.SimpleNamespace(x=0.5, y=0.5, z=0.0)
                         for _ in range(21)]


class Script:
    """Camera + MediaPipe + YOLO, all reading one list of frames."""

    def __init__(self, engine, clock, steps):
        self.engine = engine
        self.clock = clock
        self.steps = list(steps)
        self.index = 0
        self.seq = 0
        self.current = None
        self.stall_until = None
        self.image = np.zeros((48, 64, 3), np.uint8)
        self.hand = FakeLandmarks()

    # WebcamStream
    def read(self):
        if self.stall_until is not None:
            if self.clock.now < self.stall_until:
                return True, self.image, self.seq       # same frame again
            self.stall_until = None
        if self.index >= len(self.steps):
            self.engine._stop_event.set()
            return True, self.image, self.seq
        step = self.steps[self.index]
        self.index += 1
        if isinstance(step, tuple) and step[0] == "stall":
            self.stall_until = self.clock.now + step[1]
            return True, self.image, self.seq
        if step.before is not None:
            step.before()
        self.current = step
        self.seq += 1
        self.clock.advance(1.0 / 30.0)
        return True, self.image, self.seq

    # MediaPipe Hands
    def process(self, _rgb):
        if self.current.crash:
            raise RuntimeError("graph exploded")
        seen = [self.hand] if self.current.gesture else None
        return types.SimpleNamespace(multi_hand_landmarks=seen)


class FakeYolo:
    ready = True
    load_error = None

    def __init__(self, script):
        self.script = script

    @property
    def current_state(self):
        step = self.script.current
        return (step.yolo, step.score, 5.0)

    def submit(self, _crop):
        pass

    def poll(self):
        return None

    def request_reset(self):
        pass


class FakeScreen:
    cam_w, cam_h = 64, 48
    center = (960.0, 540.0)
    left, top, width, height = 0, 0, 1920, 1080
    box_left, box_top, box_right, box_bottom = 4, 4, 60, 44
    box_w, box_h = 56, 40
    effective_box = None

    def poll(self, _now):
        return False

    def set_camera_size(self, _w, _h):
        return False

    def to_screen(self, x, y):
        return x * 30.0, y * 22.5


class StubValidator:
    held = 0
    rejected = 0

    def check(self, _hand):
        return True, ""

    def reset(self):
        pass


class LoopCase(unittest.TestCase):
    """Builds an engine around the real _run and replays a script."""

    def setUp(self):
        self.clock = Clock()
        self.script = None
        fake_time = types.SimpleNamespace(
            perf_counter=self.clock,
            sleep=lambda s: self.clock.advance(s),
            time=lambda: 1.79e9)
        patches = {
            "time": fake_time,
            "detect_gesture": lambda hand, w, h: (
                self.script.current.gesture, (False,) * 4, False),
            "handedness_of": lambda results, hand, m: self.script.current.side,
            "palm_is_facing": lambda hand, side, m: True,
            "gun_is_facing": lambda hand, side, m: True,
            "is_hand_pointing_down": lambda hand: False,
            "pick_primary_hand": lambda seen, anchor: seen[0],
            "mp_draw": types.SimpleNamespace(
                draw_landmarks=lambda *a, **k: None),
        }
        for name, value in patches.items():
            self.addCleanup(setattr, hand_cursor_2, name,
                            getattr(hand_cursor_2, name))
            setattr(hand_cursor_2, name, value)

    def build(self, config, *, yolo=False, mode="always", gesture=None,
              hand="any", events=True, cursor=None):
        eng = object.__new__(HandTrackerEngine)
        eng.verbose = False
        eng.logs = []
        eng._log = eng.logs.append
        eng._reported = set()
        eng._enable_yolo = yolo
        eng._cursor_control_mode = mode
        eng._cursor_control_gesture = gesture
        eng._cursor_control_hand = hand
        eng._cursor_enable_gesture = None
        eng._stop_event = threading.Event()
        eng._mp_rebuild = threading.Event()
        eng._filter_reset = threading.Event()
        eng._validator = StubValidator()
        eng.probe = None
        eng.camera_index = 0
        eng.macros = None
        eng.status = {"running": True, "error": None}
        eng.screen = FakeScreen()
        eng.cursor = cursor or RecordingCursor()
        eng.actions = ActionDispatcher(eng.cursor)
        eng.actions._executor = ActionExecutor(dry_run=True)

        geometry_learner = adaptive_timing.TimingLearner()
        semantic_learner = adaptive_timing.TimingLearner()
        self.addCleanup(geometry_learner.stop)
        self.addCleanup(semantic_learner.stop)
        eng.fsm = GestureFSM(config, learner=geometry_learner)
        eng.semantic_fsm = GestureFSM(config, stability_threshold=1,
                                      window_size=1, transition_memory=6,
                                      learner=semantic_learner)
        eng._config_generation = 1

        self.sent = []
        if events:
            eng.events = ge.GestureEventTracker(
                self.sent.append, wall_clock=lambda: 1.79e9,
                monotonic=self.clock)
            eng.events.begin()
        self.eng = eng
        self.yolo = yolo
        return eng

    def run_script(self, steps):
        eng = self.eng
        self.script = Script(eng, self.clock, steps)
        eng.stream = self.script
        eng.hands = self.script
        eng.yolo_worker = FakeYolo(self.script) if self.yolo else None
        eng._stop_event.clear()
        eng._run()
        return self.sent

    def kinds(self, exclude=(ge.HEARTBEAT,)):
        return [m["type"] for m in self.sent if m["type"] not in exclude]

    def of(self, kind):
        return [m for m in self.sent if m["type"] == kind]


GEOMETRY_CLICK = {"transitions": [
    rule("click", "point", "grip", LEFT_CLICK, source="geometry")]}

GEOMETRY_DRAG = {"transitions": [
    rule("drag-on", "open", "grip", DRAG_START, source="geometry"),
    rule("drag-off", "grip", "open", DRAG_STOP, source="geometry")]}


class LoopLifecycleTests(LoopCase):

    def test_ready_is_the_first_message_and_sent_once(self):
        self.build(GEOMETRY_CLICK)
        self.run_script(frames("point", 40))
        self.assertEqual(self.kinds()[0], ge.READY)
        self.assertEqual(len(self.of(ge.READY)), 1)
        payload = self.of(ge.READY)[0]["payload"]
        self.assertEqual(payload["camera_index"], 0)
        self.assertEqual(payload["semantic"], "disabled")

    def test_ready_reports_the_semantic_branch_state(self):
        self.build(GEOMETRY_CLICK, yolo=True)
        self.run_script(frames("point", 3))
        self.assertEqual(self.of(ge.READY)[0]["payload"]["semantic"],
                         "ready")

    def test_a_requested_stop_is_not_fatal(self):
        self.build(GEOMETRY_CLICK)
        self.run_script(frames("point", 10))
        self.assertEqual(self.of(ge.FATAL), [])
        self.assertEqual(self.kinds()[-1], ge.POINTING_STOPPED)
        self.assertEqual(self.of(ge.POINTING_STOPPED)[0]["payload"]
                         ["reason"], ge.REASON_ENGINE_STOPPED)

    def test_a_crashed_loop_is_fatal_and_releases_everything(self):
        self.build(GEOMETRY_DRAG)
        self.run_script(frames("open", 4) + frames("grip", 4)
                        + [Frame("grip", crash=True)])
        kinds = self.kinds()
        self.assertEqual(kinds[-3:], [ge.DRAG_STOPPED, ge.POINTING_STOPPED,
                                      ge.FATAL])
        fatal = self.of(ge.FATAL)[0]["payload"]
        self.assertEqual(fatal["code"], ge.FATAL_TRACKING_LOOP)
        self.assertIn("graph exploded", fatal["message"])
        self.assertEqual(self.of(ge.DRAG_STOPPED)[0]["payload"]["reason"],
                         ge.REASON_FATAL)
        self.assertFalse(self.eng.actions.dragging)

    def test_recoverable_failures_are_not_fatal(self):
        # A backend that refuses clicks, a binding that raises, a hand
        # that comes and goes, and a camera stall: none ends the loop.
        self.build(GEOMETRY_CLICK, cursor=RecordingCursor(fail_click=True))
        original = self.eng.actions.dispatch

        def flaky(action, now):
            if not getattr(flaky, "raised", False):
                flaky.raised = True
                raise ValueError("bad binding")
            return original(action, now)

        self.eng.actions.dispatch = flaky
        self.run_script(frames("point", 3) + frames("grip", 3)
                        + frames("point", 3) + frames("grip", 3)
                        + frames(None, 3) + [stall(2.0)]
                        + frames("point", 3))
        self.assertEqual(self.of(ge.FATAL), [])
        self.assertIsNone(self.eng.status.get("error"))

    def test_no_event_escapes_after_shutdown(self):
        self.build(GEOMETRY_CLICK)
        self.run_script(frames("point", 5))
        self.eng.events.close()
        count = len(self.sent)
        self.eng.events.begin()
        self.run_script(frames("point", 3) + frames("grip", 3))
        self.assertEqual(len(self.sent), count)

    def test_events_after_stop_need_a_new_start(self):
        self.build(GEOMETRY_CLICK)
        self.run_script(frames("point", 5))
        count = len(self.sent)
        # A straggling frame after the lifecycle ended: nothing reported.
        self.eng.events.set_pointing(True)
        self.eng.events.gesture_fired(gesture="grip", action="LEFT_CLICK")
        self.assertEqual(len(self.sent), count)
        # A new start() -> a new ready.
        self.eng.events.begin()
        self.run_script(frames("point", 5))
        self.assertEqual(len(self.of(ge.READY)), 2)

    def test_lifecycle_messages_never_contradict(self):
        self.build(GEOMETRY_DRAG)
        self.run_script(frames("open", 4) + frames("grip", 4)
                        + frames(None, 2) + frames("open", 4)
                        + frames("grip", 4) + frames("open", 4))
        pointing = dragging = False
        for kind in self.kinds():
            if kind == ge.POINTING_STARTED:
                self.assertFalse(pointing)
                pointing = True
            elif kind == ge.POINTING_STOPPED:
                self.assertTrue(pointing)
                pointing = False
            elif kind == ge.DRAG_STARTED:
                self.assertFalse(dragging)
                dragging = True
            elif kind == ge.DRAG_STOPPED:
                self.assertTrue(dragging)
                dragging = False
        self.assertFalse(pointing or dragging)


class LoopPointingTests(LoopCase):

    def test_one_start_for_a_hundred_pointing_frames(self):
        self.build(GEOMETRY_CLICK)
        self.run_script(frames("point", 100))
        self.assertEqual(len(self.of(ge.POINTING_STARTED)), 1)
        start = self.of(ge.POINTING_STARTED)[0]["payload"]
        self.assertEqual(start["hand"], "right")
        self.assertTrue(start["hand_present"])
        self.assertEqual(start["cursor_control"], "always")

    def test_hand_loss_stops_pointing_once(self):
        self.build(GEOMETRY_CLICK)
        self.run_script(frames("point", 5) + frames(None, 30)
                        + frames("point", 5))
        stops = self.of(ge.POINTING_STOPPED)
        self.assertEqual(stops[0]["payload"]["reason"], ge.REASON_HAND_LOST)
        self.assertFalse(stops[0]["payload"]["hand_present"])
        self.assertEqual(len(self.of(ge.POINTING_STARTED)), 2)
        self.assertEqual(len(stops), 2)            # hand lost + stop()

    def test_gate_follows_the_cursor_control_pose(self):
        self.build(GEOMETRY_CLICK, mode="gesture-based", gesture="point")
        self.run_script(frames("open", 5) + frames("point", 5)
                        + frames("open", 5))
        self.assertEqual(self.kinds(), [ge.READY, ge.POINTING_STARTED,
                                        ge.POINTING_STOPPED])
        stop = self.of(ge.POINTING_STOPPED)[0]["payload"]
        self.assertEqual(stop["reason"], ge.REASON_GATE_CLOSED)
        self.assertTrue(stop["hand_present"])

    def test_disabling_cursor_control_ends_pointing(self):
        self.build(GEOMETRY_CLICK)
        gate_on = lambda: self.eng.apply_settings(  # noqa: E731
            cursor_control_mode="gesture-based",
            cursor_control_gesture="peace")
        self.run_script(frames("point", 5)
                        + [Frame("point", before=gate_on)]
                        + frames("point", 5))
        self.assertEqual(self.kinds()[:3], [ge.READY, ge.POINTING_STARTED,
                                            ge.POINTING_STOPPED])
        stop = self.of(ge.POINTING_STOPPED)[0]["payload"]
        self.assertEqual(stop["reason"], ge.REASON_GATE_CLOSED)
        self.assertEqual(stop["cursor_control"], "gesture-based")

    def test_hand_filter_change_is_reported(self):
        self.build(GEOMETRY_CLICK, mode="gesture-based", gesture="point",
                   hand="left")
        to_right = lambda: self.eng.apply_settings(  # noqa: E731
            cursor_control_hand="right")
        self.run_script(frames("point", 5, side="right")
                        + [Frame("point", before=to_right)]
                        + frames("point", 5))
        self.assertEqual(self.kinds(), [ge.READY, ge.POINTING_STARTED,
                                        ge.POINTING_STOPPED])
        self.assertEqual(self.of(ge.POINTING_STOPPED)[0]["payload"]
                         ["reason"], ge.REASON_ENGINE_STOPPED)

    def test_a_stalled_camera_stops_pointing(self):
        self.build(GEOMETRY_CLICK)
        self.run_script(frames("point", 5) + [stall(1.5)]
                        + frames("point", 5))
        stops = self.of(ge.POINTING_STOPPED)
        self.assertEqual(stops[0]["payload"]["reason"],
                         ge.REASON_CAMERA_STALLED)
        self.assertEqual(len(self.of(ge.POINTING_STARTED)), 2)

    def test_a_short_gap_between_frames_is_not_a_stall(self):
        self.build(GEOMETRY_CLICK)
        self.run_script(frames("point", 5) + [stall(0.5)]
                        + frames("point", 5))
        self.assertEqual(len(self.of(ge.POINTING_STARTED)), 1)

    def test_no_per_frame_cursor_messages(self):
        self.build(GEOMETRY_CLICK)
        self.run_script(frames("point", 90))
        moves = [c for c in self.eng.cursor.calls if c[0] == "move"]
        self.assertGreater(len(moves), 50, "the cursor never moved")
        self.assertEqual(self.kinds(), [ge.READY, ge.POINTING_STARTED,
                                        ge.POINTING_STOPPED])
        for message in self.sent:
            for key in ("x", "y", "landmarks", "frame", "position"):
                self.assertNotIn(key, message["payload"])


class LoopGestureTests(LoopCase):

    def test_a_real_firing_produces_one_message(self):
        self.build(GEOMETRY_CLICK)
        self.run_script(frames("point", 4) + frames("grip", 20))
        fired = self.of(ge.GESTURE_FIRED)
        self.assertEqual(len(fired), 1, "held grip repeated the message")
        payload = fired[0]["payload"]
        self.assertEqual(payload["gesture"], "grip")
        self.assertEqual(payload["from_gesture"], "point")
        self.assertEqual(payload["trigger"], "transition")
        self.assertEqual(payload["action"], "LEFT_CLICK")
        self.assertEqual(payload["hand"], "right")
        self.assertEqual(payload["rule_id"], "click")
        self.assertEqual(payload["stream"], "geometry")
        self.assertIsNone(payload["confidence"], "geometry has no score")
        self.assertTrue(payload["executed"])

    def test_predictions_without_a_firing_produce_nothing(self):
        self.build(GEOMETRY_CLICK)
        self.run_script(frames("grip", 10) + frames("open", 10)
                        + frames("grip", 10))
        self.assertEqual(self.of(ge.GESTURE_FIRED), [])

    def test_two_genuine_firings_produce_two_messages(self):
        self.build(GEOMETRY_CLICK)
        self.run_script(frames("point", 4) + frames("grip", 4)
                        + frames("point", 4) + frames("grip", 4))
        self.assertEqual(len(self.of(ge.GESTURE_FIRED)), 2)
        clicks = [c for c in self.eng.cursor.buttons() if c[0] == "click"]
        self.assertEqual(len(clicks), 2)

    def test_a_refused_click_is_reported_as_not_executed(self):
        self.build(GEOMETRY_CLICK, cursor=RecordingCursor(fail_click=True))
        self.run_script(frames("point", 4) + frames("grip", 4))
        self.assertFalse(self.of(ge.GESTURE_FIRED)[0]["payload"]
                         ["executed"])

    def test_semantic_firing_carries_the_classifier_score(self):
        config = {"transitions": [
            rule("sem", "zeta", "quux", "COPY", source="semantic")]}
        self.build(config, yolo=True)
        self.run_script(frames("open", 3, yolo="zeta", score=0.8)
                        + frames("open", 3, yolo="quux", score=0.91))
        fired = self.of(ge.GESTURE_FIRED)
        self.assertEqual(len(fired), 1)
        payload = fired[0]["payload"]
        self.assertEqual((payload["gesture"], payload["action"],
                          payload["stream"], payload["confidence"]),
                         ("quux", "COPY", "semantic", 0.91))

    def test_a_score_for_another_pose_is_not_borrowed(self):
        # A hold fires on its timer; the score on that frame is for a
        # label that only reaches the FSM as an abstention.
        config = {"holds": [hold("h", "quux", "COPY", hold_sec=0.3,
                                 source="semantic")]}
        self.build(config, yolo=True)
        self.run_script(frames("open", 2, yolo="quux", score=0.9)
                        + frames("open", 20, yolo="grip", score=0.7))
        fired = self.of(ge.GESTURE_FIRED)
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0]["payload"]["trigger"], "hold")
        self.assertIsNone(fired[0]["payload"]["confidence"])

    def test_geometry_and_semantic_overlap_fires_once(self):
        # The same pose seen by both streams, under a source-agnostic rule.
        config = {"transitions": [rule("both", "point", "grip",
                                       LEFT_CLICK)]}
        self.build(config, yolo=True)
        self.run_script(frames("point", 4, yolo="point", score=0.9)
                        + frames("grip", 8, yolo="grip", score=0.9))
        self.assertEqual(len(self.of(ge.GESTURE_FIRED)), 1)
        clicks = [c for c in self.eng.cursor.buttons() if c[0] == "click"]
        self.assertEqual(len(clicks), 1)

    def test_a_reloaded_mapping_is_reported_with_its_new_action(self):
        self.build(GEOMETRY_CLICK)
        eng = self.eng

        def remap():
            config = {"transitions": [rule("click", "point", "grip",
                                           "DOUBLE_CLICK",
                                           source="geometry")]}
            eng.fsm = GestureFSM(config, learner=eng.fsm._timing)
            eng.semantic_fsm = GestureFSM(config, stability_threshold=1,
                                          window_size=1,
                                          learner=eng.semantic_fsm._timing)
            eng._config_generation += 1

        self.run_script(frames("point", 4) + frames("grip", 4)
                        + [Frame("open", before=remap)] + frames("open", 3)
                        + frames("point", 4) + frames("grip", 4))
        actions = [m["payload"]["action"] for m in self.of(ge.GESTURE_FIRED)]
        self.assertEqual(actions, ["LEFT_CLICK", "DOUBLE_CLICK"])


class LoopDragTests(LoopCase):

    def test_drag_start_and_normal_release_once_each(self):
        self.build(GEOMETRY_DRAG)
        self.run_script(frames("open", 4) + frames("grip", 30)
                        + frames("open", 4))
        self.assertEqual(len(self.of(ge.DRAG_STARTED)), 1)
        self.assertEqual(len(self.of(ge.DRAG_STOPPED)), 1)
        start = self.of(ge.DRAG_STARTED)[0]["payload"]
        self.assertEqual((start["gesture"], start["rule_id"], start["hand"]),
                         ("grip", "drag-on", "right"))
        stop = self.of(ge.DRAG_STOPPED)[0]["payload"]
        self.assertEqual(stop["reason"], ge.REASON_GESTURE)
        self.assertEqual(stop["rule_id"], "drag-on")
        self.assertGreater(stop["duration_sec"], 0.5)
        self.assertEqual(self.of(ge.GESTURE_FIRED), [],
                         "a drag was also reported as gesture_fired")

    def test_hand_loss_releases_and_reports_the_drag(self):
        self.build(GEOMETRY_DRAG)
        self.run_script(frames("open", 4) + frames("grip", 4)
                        + frames(None, 4))
        stops = self.of(ge.DRAG_STOPPED)
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0]["payload"]["reason"], ge.REASON_HAND_LOST)
        self.assertFalse(self.eng.actions.dragging)
        self.assertFalse(self.eng.events.dragging)

    def test_fsm_failsafe_release_is_labelled(self):
        # Unmapped pose while dragging -> the FSM's stuck-release.
        self.build({"transitions": [rule("drag-on", "open", "grip",
                                         DRAG_START, source="geometry")],
                    "settings": {"stuck_release_sec": 0.2}})
        self.run_script(frames("open", 4) + frames("grip", 4)
                        + frames("peace", 20))
        stops = self.of(ge.DRAG_STOPPED)
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0]["payload"]["reason"],
                         ge.REASON_FSM_FAILSAFE)

    def test_engine_stop_reports_the_drag_released(self):
        self.build(GEOMETRY_DRAG)
        self.run_script(frames("open", 4) + frames("grip", 4))
        stops = self.of(ge.DRAG_STOPPED)
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0]["payload"]["reason"],
                         ge.REASON_ENGINE_STOPPED)
        # stop() after the loop has already let go: nothing more.
        self.eng._thread = None
        self.eng.stream = None
        self.eng.status = {}
        self.eng.stop()
        self.assertEqual(len(self.of(ge.DRAG_STOPPED)), 1)

    def test_reload_mid_drag_reports_config_reloaded(self):
        self.build(GEOMETRY_DRAG)
        self.run_script(frames("open", 4) + frames("grip", 4))
        # Re-press as if mid-drag, then reload from the GUI thread.
        self.eng.actions.dispatch(ActionEvent(DRAG_START), 0.0)
        self.eng.events.drag_started(gesture="grip")
        self.eng._apply_config = lambda: None
        self.eng.fsm = types.SimpleNamespace(describe=lambda: "-")
        self.eng.reload_config()
        self.assertEqual(self.of(ge.DRAG_STOPPED)[-1]["payload"]["reason"],
                         ge.REASON_CONFIG_RELOADED)

    def test_events_do_not_change_what_reaches_the_mouse(self):
        steps = (frames("open", 4) + frames("grip", 10) + frames("open", 4)
                 + frames("point", 4) + frames(None, 3) + frames("open", 4)
                 + frames("grip", 4) + frames(None, 3))

        def replay(with_events):
            self.clock.now = 100.0
            self.build(GEOMETRY_DRAG, events=with_events)
            self.run_script(steps)
            return list(self.eng.cursor.calls)

        silent = replay(False)
        reported = replay(True)
        self.assertEqual(silent, reported)
        self.assertIn(("press", "left"), reported)
        self.assertGreater(len(self.of(ge.DRAG_STARTED)), 0)


class LoopHeartbeatTests(LoopCase):

    def test_one_heartbeat_per_second_of_frames(self):
        self.build(GEOMETRY_CLICK)
        self.run_script(frames("point", 105))          # 3.5 s at 30 fps
        beats = self.of(ge.HEARTBEAT)
        self.assertEqual(len(beats), 3)
        payload = beats[0]["payload"]
        self.assertTrue(payload["pointing"])
        self.assertTrue(payload["hand_present"])
        self.assertEqual(payload["camera"], "live")
        self.assertEqual(self.kinds(), [ge.READY, ge.POINTING_STARTED,
                                        ge.POINTING_STOPPED])

    def test_heartbeats_continue_through_a_stall_and_end_at_stop(self):
        self.build(GEOMETRY_CLICK)
        self.run_script(frames("point", 5) + [stall(3.0)])
        stalled = [m for m in self.of(ge.HEARTBEAT)
                   if m["payload"]["camera"] == "stalled"]
        self.assertGreaterEqual(len(stalled), 1)
        self.assertLessEqual(len(stalled), 3)
        count = len(self.sent)
        self.eng.events.heartbeat(self.clock.now + 60)
        self.assertEqual(len(self.sent), count, "heartbeat after stop")


# ─── 4. Scope ───────────────────────────────────────────────────────────────

class ScopeTests(unittest.TestCase):

    TRANSPORT = {"socket", "multiprocessing", "subprocess", "queue",
                 "asyncio", "http", "urllib", "ssl", "select", "zmq",
                 "mmap", "pickle", "logging"}

    def test_the_builder_has_no_transport_or_io(self):
        path = os.path.join(MODULE_DIR, "gesture_events.py")
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
            elif isinstance(node, ast.Call) and \
                    getattr(node.func, "id", None) == "open":
                self.fail("gesture_events.py opens a file")
        imported.discard("__future__")
        self.assertEqual(imported & self.TRANSPORT, set())
        self.assertLessEqual(imported, {"enum", "math", "numbers",
                                        "threading", "time"})

    def test_the_engine_installs_a_tracker_but_no_sink(self):
        import inspect
        source = inspect.getsource(HandTrackerEngine.__init__)
        self.assertIn("GestureEventTracker(event_sink)", source)
        self.assertIsNone(HandTrackerEngine.events)

    def test_modules_load_from_the_gesture_package(self):
        for module in (ge, hand_cursor_2):
            self.assertEqual(os.path.dirname(os.path.abspath(
                module.__file__)), MODULE_DIR)


if __name__ == "__main__":
    unittest.main()
