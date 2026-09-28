"""Regression tests for the cursor across a gesture-gated transition.

WHAT THIS COVERS.  CursorClutch, the class that keeps the pointer where
it is while cursor control is gated on a pose, driven through the real
CursorStabiliser, OneEuroFilter and GestureFSM in the loop's order, with
the gate opened only by a semantic label that arrives at the network
rate -- which is what makes the gate late on both edges.

THE DEFECT THESE PIN DOWN.  The gate only decided whether cursor.move()
was called, while the filter went on following the hand.  Measured on a
cursor-pose -> origin -> click -> cursor-pose cycle: the click landed
20.3 px from where the cursor had been aimed, and the pointer jumped
21.6 px in one frame when the gate re-opened, ending 43.6 px away.

NO CAMERA, NO MODEL, NO MOUSE, NO CONFIG FILE.

    python test_cursor_clutch.py
"""
from __future__ import annotations

import inspect
import math
import unittest

import hand_cursor_2
from hand_cursor_2 import CursorClutch, CursorStabiliser, OneEuroFilter
from cursor_stability_probe import (FRAME, SEMANTIC, YOLO_PERIOD, Script,
                                    config, rule)
from gesture_fsm import GestureFSM

# Invented labels, as in test_cursor_stability: nothing here may depend
# on a pose name.  CURSOR is the pose the gate is configured to.
CURSOR, AIM, CLOSED = "zeta_wave", "quux_point", "quux_grip"
CLICK = rule(AIM, CLOSED, "LEFT_CLICK", max_time_sec=2.0)
YOLO_LATENCY = 0.055
BOUNDS = (0, 0, 1920, 1080)


def click_cycle(unfold=30.0):
    """Aim, change pose, click, re-open.  The hand does not come back to
    exactly where it was, which is what a real one does."""
    script = Script().still(30, CURSOR).mark("aim")
    script.fold(CURSOR, AIM, frames=8, distance=25.0)
    script.still(10, AIM).fold(AIM, CLOSED).still(8, CLOSED)
    script.unfold(CLOSED, CURSOR, frames=12, distance=unfold)
    script.still(20, CURSOR).mark("travel")
    script.travel(30, 300.0, CURSOR).mark("end")
    return script.still(10, CURSOR)


class GatedReplay:
    """The loop's cursor block, with the gate driven by a late label."""

    def __init__(self, script, clutch=True, gated=True, lost=()):
        self.script = script
        self.use_clutch = clutch
        self.gated = gated
        self.lost = set(lost)
        self.fsm = GestureFSM(config(CLICK), **SEMANTIC)
        self.stab = CursorStabiliser()
        self.clutch = CursorClutch()
        self.oef_x, self.oef_y = self._filter(), self._filter()
        self.cursor = []            # position after each frame
        self.clicks = []            # position when each action fired

    @staticmethod
    def _filter():
        return OneEuroFilter(freq=30.0, min_cutoff=hand_cursor_2.MIN_CUTOFF,
                             beta=hand_cursor_2.BETA,
                             d_cutoff=hand_cursor_2.D_CUTOFF)

    def run(self):
        now = next_yolo = 100.0
        pending, label_seen, geometry, present, pos = [], "", None, False, None
        for index, (x, y, label) in enumerate(self.script.frames):
            now += FRAME
            if index in self.lost:
                present, geometry = False, None
            else:
                if not present:
                    self.oef_x.reset()
                    self.oef_y.reset()
                    self.stab.reset()
                    present = True
                    if self.use_clutch:
                        self.clutch.interrupt()
                if label != geometry:
                    geometry = label
                    self.stab.arm_pose_change((x, y), now)
                armed = self.fsm.armed_action_origin(now, source="semantic")
                hx, hy = self.stab.update(x, y, now, armed)
                sx = self.oef_x(hx, timestamp=now)
                sy = self.oef_y(hy, timestamp=now)
                gate = label_seen == CURSOR if self.gated else True
                if self.use_clutch:
                    if gate and self.clutch.reopens(x, y):
                        self.stab.reset()
                        self.oef_x.reset()
                        self.oef_y.reset()
                        sx = self.oef_x(x, timestamp=now)
                        sy = self.oef_y(y, timestamp=now)
                    out = self.clutch.update(sx, sy, x, y, gate, now, BOUNDS)
                else:
                    out = (sx, sy) if gate else None
                if out is not None:
                    pos = out
            self.cursor.append(pos)

            if now >= next_yolo:
                next_yolo = now + YOLO_PERIOD
                pending.append((now + YOLO_LATENCY, label))
            while pending and pending[0][0] <= now:
                _t, label_seen = pending.pop(0)
                action = self.fsm.update(label_seen, now, source="semantic")
                if action is not None:
                    self.clicks.append(pos)
                    self.stab.note_action(now)
        return self

    def at(self, mark, offset=0):
        return self.cursor[self.script.marks[mark] + offset]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class GatedTransitionTests(unittest.TestCase):

    def test_the_defect_reproduces_without_the_clutch(self):
        replay = GatedReplay(click_cycle(), clutch=False).run()
        aim = replay.at("aim", -1)
        self.assertGreater(dist(replay.clicks[0], aim), 15.0)
        self.assertGreater(dist(replay.at("travel", -1), aim), 40.0)

    def test_the_click_lands_where_the_cursor_was_aimed(self):
        replay = GatedReplay(click_cycle()).run()
        self.assertEqual(len(replay.clicks), 1)
        self.assertLess(dist(replay.clicks[0], replay.at("aim", -1)), 3.0)

    def test_reopening_the_gate_leaves_the_cursor_where_it_was(self):
        replay = GatedReplay(click_cycle()).run()
        self.assertLess(dist(replay.at("travel", -1), replay.at("aim", -1)),
                        3.0)

    def test_no_frame_moves_the_cursor_away_from_the_aim(self):
        # The one sizeable step allowed is the correction back TO the aim
        # as the gate shuts; every position is within 3 px of it after.
        replay = GatedReplay(click_cycle()).run()
        aim = replay.at("aim", -1)
        marks = replay.script.marks
        worst = 0.0
        for i in range(marks["aim"], marks["travel"]):
            step = dist(replay.cursor[i], replay.cursor[i - 1])
            if step > 10.0:
                # A jump, not the fold being followed: it must END at the
                # aim, never leave it.
                self.assertLess(dist(replay.cursor[i], aim), 3.0, i)
            worst = max(worst, dist(replay.cursor[i], aim))
        # Before the gate shuts the cursor may follow the fold a little.
        self.assertLess(worst, 25.0)

    def test_a_skeleton_dropout_during_the_fist_does_not_jump(self):
        replay = GatedReplay(click_cycle(), lost=range(52, 57)).run()
        self.assertLess(dist(replay.at("travel", -1), replay.at("aim", -1)),
                        3.0)

    def test_normal_movement_is_live_after_the_transition(self):
        replay = GatedReplay(click_cycle()).run()
        marks = replay.script.marks
        for i in range(marks["travel"] + 1, marks["end"]):
            self.assertGreater(dist(replay.cursor[i], replay.cursor[i - 1]),
                               5.0, i)
        # The offset is spent during travel, so the cursor ends up back
        # on the hand's absolute mapping.
        hand = replay.script.frames[-1][:2]
        self.assertLess(dist(replay.cursor[-1], hand), 1.0)

    def test_moving_far_while_the_gate_is_shut_snaps_as_before(self):
        script = Script().still(30, CURSOR).mark("aim")
        script.travel(20, 400.0, AIM).still(10, AIM).mark("back")
        script.still(20, CURSOR)
        replay = GatedReplay(script).run()
        hand = script.frames[-1][:2]
        self.assertLess(dist(replay.cursor[-1], hand), 1.0)

    def test_the_offset_never_leaves_the_screen(self):
        clutch = CursorClutch()
        clutch.update(5.0, 5.0, 5.0, 5.0, True, 0.0, BOUNDS)
        clutch.update(5.0, 5.0, 5.0, 5.0, False, 0.1, BOUNDS)
        for step in range(30):
            x, y = clutch.update(40.0 - step, 40.0, 40.0 - step, 40.0, True,
                                 0.2 + step * FRAME, BOUNDS)
            self.assertGreaterEqual(x, BOUNDS[0])


class UngatedIsUnchangedTests(unittest.TestCase):

    def test_always_mode_follows_the_filter_exactly(self):
        script = Script().still(20, CURSOR)
        script.travel(60, 900.0, CURSOR, angle=0.3).still(10, CURSOR)
        before = GatedReplay(script, clutch=False, gated=False).run()
        after = GatedReplay(script, clutch=True, gated=False).run()
        for a, b in zip(before.cursor, after.cursor):
            self.assertAlmostEqual(dist(a, b), 0.0, places=9)


class EngineWiringTests(unittest.TestCase):

    def test_the_loop_moves_the_cursor_through_the_clutch(self):
        body = inspect.getsource(hand_cursor_2.HandTrackerEngine._run)
        self.assertIn("clutch.update(", body)
        self.assertIn("clutch.interrupt()", body)
        self.assertIn("self.cursor.move(int(_out[0]), int(_out[1]))", body)
        self.assertNotIn("self.cursor.move(int(smooth_x)", body)


if __name__ == "__main__":
    unittest.main()
