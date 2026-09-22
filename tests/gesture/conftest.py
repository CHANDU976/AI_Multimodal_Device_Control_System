"""Gesture-module test setup.

hand_cursor_2.py imports cv2 and mediapipe at module scope. Neither is
needed for the pure logic tested here (geometry, filters, screen mapping,
dispatch), so where they are not installed a minimal stand-in is registered
before the first import. The engine's camera/MediaPipe runtime path is out
of scope for unit tests — it needs a webcam.
"""

import pytest

from helpers.fakes import cv2_stub_module, ensure_importable, mediapipe_stub_module

ensure_importable("cv2", cv2_stub_module)
try:
    import mediapipe
    if not hasattr(mediapipe, "solutions"):         # 0.10.x Tasks-only builds
        raise ImportError
except Exception:
    import sys
    sys.modules["mediapipe"] = mediapipe_stub_module()


@pytest.fixture(scope="session")
def hc():
    """The hand_cursor_2 engine module."""
    import hand_cursor_2
    return hand_cursor_2


@pytest.fixture(scope="session")
def fsm_mod():
    import gesture_fsm
    return gesture_fsm
