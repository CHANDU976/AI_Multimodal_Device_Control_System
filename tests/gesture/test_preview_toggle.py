"""Tests for the Live Preview show/hide toggle.

The toggle is display-only: it must never reach the engine.  These build
a GestureStudio with object.__new__ and stub widgets -- the same pattern
test_config_lifecycle.py uses for the pure-model methods -- so the real
methods run without a window.

NO CAMERA, NO WINDOW, NO ENGINE.
"""
from __future__ import annotations

import unittest

import app


class FakeWidget:
    """Records grid/configure traffic instead of drawing."""

    def __init__(self):
        self.gridded = True
        self.options = {}
        self.image = None
        self.weights = {}

    def grid(self, **kwargs):
        self.gridded = True

    def grid_remove(self):
        self.gridded = False

    def configure(self, **kwargs):
        self.options.update(kwargs)

    def rowconfigure(self, index, weight=None, **kwargs):
        self.weights[index] = weight

    def winfo_width(self):
        return 640

    def winfo_height(self):
        return 480


class FakeEngine:
    """An engine that notices if anyone tries to stop it."""

    def __init__(self):
        self.running = True
        self.stopped = False
        self.status = {"fps": 30.0}
        self.frames_served = 0

    def stop(self):
        self.stopped = True
        self.running = False

    def get_latest_frame(self):
        self.frames_served += 1
        return None


def studio(visible=True):
    """A windowless studio carrying just the preview widgets."""
    obj = object.__new__(app.GestureStudio)
    obj.tk = None
    obj._preview_visible = visible
    obj.video_label = FakeWidget()
    obj.preview_button = FakeWidget()
    obj._preview_stage = FakeWidget()
    obj._video_photo = "a-bitmap"
    obj.preview_meta = FakeWidget()
    obj.engine = None
    obj._video_job = None
    obj._apply_preview_visibility()
    return obj


class ToggleStateTests(unittest.TestCase):

    def test_starts_visible_with_a_hide_label(self):
        s = studio()
        self.assertTrue(s._preview_visible)
        self.assertTrue(s.video_label.gridded)
        self.assertEqual(s.preview_button.options["text"], "Hide Preview")

    def test_hiding_collapses_the_row_and_flips_the_label(self):
        s = studio()
        s._toggle_preview()

        self.assertFalse(s._preview_visible)
        self.assertFalse(s.video_label.gridded)
        self.assertEqual(s.preview_button.options["text"], "Show Preview")
        self.assertEqual(s._preview_stage.weights[1], 0,
                         "row still weighted; the black gap would remain")

    def test_showing_restores_the_row_and_the_label(self):
        s = studio()
        s._toggle_preview()
        s._toggle_preview()

        self.assertTrue(s._preview_visible)
        self.assertTrue(s.video_label.gridded)
        self.assertEqual(s.preview_button.options["text"], "Hide Preview")
        self.assertEqual(s._preview_stage.weights[1], 1)

    def test_hide_show_survives_many_round_trips(self):
        s = studio()
        for _ in range(10):
            s._toggle_preview()
            self.assertFalse(s.video_label.gridded)
            s._toggle_preview()
            self.assertTrue(s.video_label.gridded)
        self.assertTrue(s._preview_visible)

    def test_hiding_drops_the_retained_bitmap(self):
        s = studio()
        s._toggle_preview()

        self.assertIsNone(s._video_photo)
        self.assertIsNone(s.video_label.image)

    def test_applying_before_the_widgets_exist_is_harmless(self):
        """The guard fires during _build, before the widgets are made.

        tk=None for the reason bare_studio documents: tk.Tk.__getattr__
        delegates to self.tk and recurses forever on an uninitialised
        instance, so binding it turns that into the AttributeError the
        hasattr guard already expects.
        """
        obj = object.__new__(app.GestureStudio)
        obj.tk = None
        obj._preview_visible = False
        obj._apply_preview_visibility()      # must not raise


class EngineIsUntouchedTests(unittest.TestCase):
    """The whole point: hiding is not disconnecting."""

    def test_toggling_never_touches_the_engine(self):
        s = studio()
        s.engine = FakeEngine()

        s._toggle_preview()
        s._toggle_preview()

        self.assertFalse(s.engine.stopped)
        self.assertTrue(s.engine.running)

    def test_the_pump_keeps_running_while_hidden(self):
        """A hidden preview must not stop the frame loop."""
        s = studio(visible=False)
        s.engine = FakeEngine()
        scheduled = []
        s._schedule_video = lambda: scheduled.append(True)

        s.update_video_feed()

        self.assertEqual(s.engine.frames_served, 1,
                         "stopped asking the engine for frames")
        self.assertEqual(len(scheduled), 1, "the pump did not reschedule")

    def test_the_pump_still_runs_while_visible(self):
        s = studio(visible=True)
        s.engine = FakeEngine()
        scheduled = []
        s._schedule_video = lambda: scheduled.append(True)

        s.update_video_feed()

        self.assertEqual(s.engine.frames_served, 1)
        self.assertEqual(len(scheduled), 1)

    def test_a_stopped_engine_ends_the_pump_either_way(self):
        """Unchanged behaviour: no engine means no reschedule."""
        for visible in (True, False):
            s = studio(visible=visible)
            s.engine = None
            scheduled = []
            s._schedule_video = lambda: scheduled.append(True)

            s.update_video_feed()

            self.assertEqual(scheduled, [], f"visible={visible}")


class ReconnectTests(unittest.TestCase):
    """Disconnect and reconnect must not disturb the preference."""

    def test_the_preference_outlives_a_disconnect(self):
        s = studio()
        s._toggle_preview()
        self.assertFalse(s._preview_visible)

        # What _disconnect does to these widgets, verbatim.
        s._video_photo = None
        s.video_label.configure(image="", text="Disconnected.")
        s.video_label.image = None
        s.preview_meta.configure(text="not connected")

        self.assertFalse(s._preview_visible)
        self.assertFalse(s.video_label.gridded,
                         "disconnect re-showed a hidden preview")

    def test_reconnecting_while_hidden_leaves_it_hidden(self):
        s = studio()
        s._toggle_preview()

        s.engine = FakeEngine()
        scheduled = []
        s._schedule_video = lambda: scheduled.append(True)
        s.update_video_feed()

        self.assertFalse(s._preview_visible)
        self.assertFalse(s.video_label.gridded)
        self.assertEqual(len(scheduled), 1)

    def test_showing_after_a_reconnect_works(self):
        s = studio()
        s._toggle_preview()
        s.engine = FakeEngine()
        s._toggle_preview()

        self.assertTrue(s._preview_visible)
        self.assertTrue(s.video_label.gridded)
        self.assertFalse(s.engine.stopped)


if __name__ == "__main__":
    unittest.main(verbosity=2)
