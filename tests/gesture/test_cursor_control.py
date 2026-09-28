"""Tests for the Cursor Control configuration flow.

Covers the whole path the GUI setting travels:

    GUI values -> config dict -> gesture_config.json -> engine fields
    -> per-frame cursor gate

The gate itself is the part that was broken: the pose a user picks comes
from the gestures/ catalog (the SEMANTIC vocabulary — "stop", "like",
"three_gun"), while the geometry detector only ever emits the four poses
in GEOMETRY_VOCABULARY plus "idle".  A gate that consults one stream
strands every pose the other stream owns.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import gesture_fsm
from gesture_fsm import DRAG_START, DRAG_STOP, load_config

from hand_cursor_2 import (CURSOR_GATE_MODES, GEOMETRY_VOCABULARY,
                           ActionDispatcher, HandTrackerEngine,
                           split_hand_suffix)


class FakeCursor:
    """A cursor backend that records instead of touching the desktop."""

    def __init__(self, fail_release=False):
        self.fail_release = fail_release
        self.moves = []
        self.pressed = 0
        self.released = 0

    def move(self, x, y):
        self.moves.append((x, y))

    def press(self, button):
        self.pressed += 1

    def release(self, button):
        if self.fail_release:
            raise RuntimeError("session revoked")
        self.released += 1

    def click(self, button="left", count=1):
        pass


def gate(mode="gesture-based", gesture="stop", hand="any"):
    """A bare engine carrying just the three cursor-control fields.

    __new__ skips __init__ on purpose: constructing a real engine opens
    MediaPipe and the YOLO worker, none of which the gate touches.  The
    method under test is the real one off the real class.
    """
    engine = HandTrackerEngine.__new__(HandTrackerEngine)
    engine._cursor_control_mode = mode
    engine._cursor_control_gesture = gesture
    engine._cursor_control_hand = hand
    engine._cursor_enable_gesture = None
    return engine


def config(*rules, cursor_control=None):
    cfg = {"settings": {}, "geometry_bindings": list(rules),
           "yolo_bindings": [], "deleted_bindings": []}
    if cursor_control is not None:
        cfg["cursor_control"] = cursor_control
    return cfg


class LabelShapeTests(unittest.TestCase):
    """The premise: "stop_right" is one pose plus one handedness."""

    def test_display_label_splits_into_pose_and_hand(self):
        self.assertEqual(split_hand_suffix("stop_right"), ("stop", "right"))
        self.assertEqual(split_hand_suffix("stop_left"), ("stop", "left"))

    def test_orientation_stays_glued_to_the_pose(self):
        self.assertEqual(split_hand_suffix("three_gun_inverse_right"),
                         ("three_gun_inverse", "right"))

    def test_bare_label_has_no_hand(self):
        self.assertEqual(split_hand_suffix("stop"), ("stop", ""))

    def test_stop_is_not_a_geometry_pose(self):
        """The regression in one line: "stop" can never come from geometry."""
        self.assertNotIn("stop", GEOMETRY_VOCABULARY)


class CursorGateTests(unittest.TestCase):
    """mode=gesture-based, gesture=stop, hand=any — the reported setup.

    The geometry stream reads a flat "stop" hand as "open", so these pass
    the realistic pair: geometry says "open", the caption says "stop_right".
    """

    def test_stop_on_right_hand_enables_cursor(self):
        self.assertTrue(gate()._cursor_gate_open("open", "right",
                                                 "stop_right"))

    def test_stop_on_left_hand_enables_cursor(self):
        self.assertTrue(gate()._cursor_gate_open("open", "left", "stop_left"))

    def test_fist_on_right_hand_disables_cursor(self):
        self.assertFalse(gate()._cursor_gate_open("grip", "right",
                                                  "fist_right"))

    def test_no_hand_disables_cursor(self):
        """Nothing recognised on either stream closes the gate."""
        self.assertFalse(gate()._cursor_gate_open("idle", "", ""))
        self.assertFalse(gate()._cursor_gate_open("", "", None))

    def test_idle_never_opens_the_gate(self):
        self.assertFalse(gate(gesture="idle")._cursor_gate_open(
            "idle", "right", "idle_right"))

    def test_unsuffixed_semantic_label_still_matches(self):
        """Handedness is blank often enough to be the normal case."""
        self.assertTrue(gate()._cursor_gate_open("open", "", "stop"))


class GeometryStreamGateTests(unittest.TestCase):
    """A pose the geometry detector owns must gate just as well."""

    def test_geometry_pose_opens_the_gate(self):
        self.assertTrue(gate(gesture="peace")._cursor_gate_open(
            "peace", "right", ""))

    def test_geometry_pose_mismatch_closes_the_gate(self):
        self.assertFalse(gate(gesture="peace")._cursor_gate_open(
            "open", "right", ""))

    def test_either_stream_may_open_the_gate(self):
        """Geometry silent + semantic matching is still a match."""
        self.assertTrue(gate(gesture="like")._cursor_gate_open(
            "idle", "right", "like_right"))


class HandFilterTests(unittest.TestCase):
    """Hand filtering follows TransitionRule.accepts_hand()."""

    def test_any_accepts_both_hands(self):
        self.assertTrue(gate(hand="any")._cursor_gate_open(
            "open", "right", "stop_right"))
        self.assertTrue(gate(hand="any")._cursor_gate_open(
            "open", "left", "stop_left"))

    def test_right_rejects_the_left_hand(self):
        self.assertFalse(gate(hand="right")._cursor_gate_open(
            "open", "left", "stop_left"))

    def test_right_accepts_the_right_hand(self):
        self.assertTrue(gate(hand="right")._cursor_gate_open(
            "open", "right", "stop_right"))

    def test_left_rejects_the_right_hand(self):
        self.assertFalse(gate(hand="left")._cursor_gate_open(
            "open", "right", "stop_right"))

    def test_unknown_side_passes_a_specific_requirement(self):
        """Mirrors accepts_hand(): blank handedness must not freeze it."""
        self.assertTrue(gate(hand="right")._cursor_gate_open(
            "open", "", "stop"))

    def test_a_pose_typed_with_a_side_carries_its_own_requirement(self):
        engine = gate(gesture="stop_right", hand="any")
        self.assertTrue(engine._cursor_gate_open("open", "right",
                                                 "stop_right"))
        self.assertFalse(engine._cursor_gate_open("open", "left",
                                                  "stop_left"))


class GateModeTests(unittest.TestCase):
    """"always" must never gate; the gating spellings must all gate."""

    def test_always_mode_ignores_the_pose(self):
        engine = gate(mode="always", gesture="stop")
        self.assertTrue(engine._cursor_gate_open("grip", "right",
                                                 "fist_right"))

    def test_every_gating_spelling_gates(self):
        for mode in CURSOR_GATE_MODES:
            engine = gate(mode=mode, gesture="stop")
            self.assertFalse(
                engine._cursor_gate_open("grip", "right", "fist_right"),
                f"mode {mode!r} failed to gate")
            self.assertTrue(
                engine._cursor_gate_open("open", "right", "stop_right"),
                f"mode {mode!r} failed to open")

    def test_gating_with_no_pose_chosen_keeps_the_cursor_alive(self):
        """An unfinished setting must not leave a dead cursor."""
        self.assertTrue(gate(gesture=None)._cursor_gate_open(
            "grip", "right", "fist_right"))
        self.assertTrue(gate(gesture="")._cursor_gate_open(
            "grip", "right", "fist_right"))

    def test_unknown_mode_falls_back_to_ungated(self):
        self.assertTrue(gate(mode="nonsense")._cursor_gate_open(
            "grip", "right", "fist_right"))

    def test_legacy_env_gesture_still_gates_when_mode_is_always(self):
        engine = gate(mode="always", gesture=None)
        engine._cursor_enable_gesture = "stop"
        self.assertTrue(engine._cursor_gate_open("open", "right",
                                                 "stop_right"))
        self.assertFalse(engine._cursor_gate_open("grip", "right",
                                                  "fist_right"))

    def test_gui_config_takes_precedence_over_the_legacy_env_value(self):
        engine = gate(mode="gesture-based", gesture="peace")
        engine._cursor_enable_gesture = "stop"
        self.assertTrue(engine._cursor_gate_open("peace", "right", ""))
        self.assertFalse(engine._cursor_gate_open("open", "right",
                                                  "stop_right"))


class DragOverridesGatingTests(unittest.TestCase):
    """A live drag must move the pointer whatever the gate would say.

    The drag state is not simulated: these drive the real
    ActionDispatcher, because ActionDispatcher.dragging is the only thing
    that knows the press actually reached the OS, and it is what the
    tracking loop hands the gate.
    """

    def setUp(self):
        self.cursor = FakeCursor()
        self.actions = ActionDispatcher(self.cursor)
        # mode=gesture-based, gesture=stop, hand=any — "grip"/"fist_right"
        # is the WRONG pose, so the gate is shut unless a drag opens it.
        self.engine = gate()

    def open_for(self, pose, hand, label):
        return self.engine._cursor_gate_open(
            pose, hand, label, dragging=self.actions.dragging)

    def test_A_wrong_gesture_without_a_drag_stays_shut(self):
        self.assertFalse(self.actions.dragging)
        self.assertFalse(self.open_for("grip", "right", "fist_right"))

    def test_B_correct_gesture_opens_without_a_drag(self):
        self.assertTrue(self.open_for("open", "right", "stop_right"))

    def test_C_drag_start_opens_the_gate_immediately(self):
        self.actions.dispatch(DRAG_START, 1.0)

        self.assertTrue(self.actions.dragging, "DRAG_START did not take")
        self.assertEqual(self.cursor.pressed, 1)
        self.assertTrue(self.open_for("grip", "right", "fist_right"))

    def test_D_gate_stays_open_across_a_moving_hand(self):
        self.actions.dispatch(DRAG_START, 1.0)

        # The pose drifts through poses that would all be refused.
        for pose, label in (("grip", "fist_right"), ("idle", ""),
                            ("point", "one_right")):
            self.assertTrue(self.open_for(pose, "right", label),
                            f"gate shut mid-drag on {pose!r}")

    def test_E_drag_stop_releases_and_restores_gating(self):
        self.actions.dispatch(DRAG_START, 1.0)
        self.actions.dispatch(DRAG_STOP, 2.0)

        self.assertFalse(self.actions.dragging)
        self.assertEqual(self.cursor.released, 1)
        self.assertTrue(self.open_for("open", "right", "stop_right"))

    def test_F_after_drag_stop_the_wrong_pose_shuts_the_gate_again(self):
        self.actions.dispatch(DRAG_START, 1.0)
        self.assertTrue(self.open_for("grip", "right", "fist_right"))

        self.actions.dispatch(DRAG_STOP, 2.0)
        self.assertFalse(self.open_for("grip", "right", "fist_right"))

    def test_G_hand_loss_failsafe_releases_and_restores_gating(self):
        """The loop's hand-lost path calls release() directly."""
        self.actions.dispatch(DRAG_START, 1.0)

        self.assertTrue(self.actions.release())
        self.assertFalse(self.actions.dragging)
        self.assertEqual(self.cursor.released, 1)
        self.assertFalse(self.open_for("grip", "right", "fist_right"))

    def test_H_gating_off_is_unaffected_by_the_drag_state(self):
        """"always" behaves identically either side of a drag."""
        engine = gate(mode="always", gesture=None)
        for dragging in (False, True):
            self.assertTrue(engine._cursor_gate_open(
                "grip", "right", "fist_right", dragging=dragging))

    def test_a_press_the_os_refused_does_not_open_the_gate(self):
        """dragging only goes true once the press actually landed."""
        cursor = FakeCursor()
        cursor.press = lambda button: (_ for _ in ()).throw(
            RuntimeError("no session"))
        actions = ActionDispatcher(cursor)
        actions.dispatch(DRAG_START, 1.0)

        self.assertFalse(actions.dragging)
        self.assertFalse(self.engine._cursor_gate_open(
            "grip", "right", "fist_right", dragging=actions.dragging))

    def test_a_release_the_os_refused_keeps_the_cursor_movable(self):
        """A stuck button must not also pin the pointer."""
        cursor = FakeCursor(fail_release=True)
        actions = ActionDispatcher(cursor)
        actions.dispatch(DRAG_START, 1.0)
        actions.dispatch(DRAG_STOP, 2.0)

        self.assertTrue(actions.dragging, "failed release must stay pending")
        self.assertTrue(self.engine._cursor_gate_open(
            "grip", "right", "fist_right", dragging=actions.dragging))

    def test_the_drag_override_defaults_off(self):
        """Callers that never heard of dragging keep the old behaviour."""
        self.assertFalse(self.engine._cursor_gate_open(
            "grip", "right", "fist_right"))


class ConfigPersistenceTests(unittest.TestCase):
    """Save -> reload -> the values are still what was chosen."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="mg-cursor-test-")
        self.path = os.path.join(self._tmp.name, "gesture_config.json")
        self.addCleanup(self._tmp.cleanup)

    def save(self, cfg):
        gesture_fsm.save_config(cfg, self.path)
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_cursor_control_survives_the_write(self):
        chosen = {"mode": "gesture-based", "gesture": "stop", "hand": "any"}
        on_disk = self.save(config(cursor_control=chosen))
        self.assertEqual(on_disk["cursor_control"], chosen)

    def test_cursor_control_survives_a_reload(self):
        chosen = {"mode": "gesture-based", "gesture": "stop", "hand": "right"}
        self.save(config(cursor_control=chosen))
        self.assertEqual(load_config(self.path)["cursor_control"], chosen)

    def test_every_hand_value_round_trips(self):
        for hand in ("any", "left", "right"):
            chosen = {"mode": "gesture-based", "gesture": "stop",
                      "hand": hand}
            self.save(config(cursor_control=chosen))
            self.assertEqual(
                load_config(self.path)["cursor_control"]["hand"], hand)

    def test_a_config_without_cursor_control_still_loads(self):
        """Backward compatibility: the key is absent in every older file."""
        self.save(config())
        reloaded = load_config(self.path)
        self.assertNotIn("cursor_control", reloaded)
        self.assertEqual(reloaded["geometry_bindings"], [])


class EngineResyncTests(unittest.TestCase):
    """Save/Apply must reach the running engine without a restart."""

    def test_apply_settings_updates_all_three_fields(self):
        engine = gate(mode="always", gesture=None, hand="any")
        HandTrackerEngine.apply_settings(
            engine, cursor_control_mode="gesture-based",
            cursor_control_gesture="stop", cursor_control_hand="right")

        self.assertEqual(engine._cursor_control_mode, "gesture-based")
        self.assertEqual(engine._cursor_control_gesture, "stop")
        self.assertEqual(engine._cursor_control_hand, "right")

    def test_apply_settings_changes_gate_behaviour_immediately(self):
        engine = gate(mode="always", gesture=None)
        self.assertTrue(engine._cursor_gate_open("grip", "right",
                                                 "fist_right"))

        HandTrackerEngine.apply_settings(
            engine, cursor_control_mode="gesture-based",
            cursor_control_gesture="stop", cursor_control_hand="any")

        self.assertFalse(engine._cursor_gate_open("grip", "right",
                                                  "fist_right"))
        self.assertTrue(engine._cursor_gate_open("open", "right",
                                                 "stop_right"))

    def test_apply_settings_leaves_unmentioned_fields_alone(self):
        engine = gate(mode="gesture-based", gesture="stop", hand="right")
        HandTrackerEngine.apply_settings(engine, cursor_speed=1.4)

        self.assertEqual(engine._cursor_control_mode, "gesture-based")
        self.assertEqual(engine._cursor_control_gesture, "stop")
        self.assertEqual(engine._cursor_control_hand, "right")


if __name__ == "__main__":
    unittest.main(verbosity=2)
