"""Synthetic MediaPipe-shaped hands for the gesture-geometry tests.

Builds the 21-landmark skeleton of an upright hand in normalised image
coordinates (y grows DOWNWARD, as in MediaPipe). Every finger can be set
extended or curled and the thumb out or tucked, which is enough to produce
each pose detect_gesture() names. The proportions are those of a real hand
seen at arm's length, so LandmarkValidator accepts the result.
"""

from __future__ import annotations

from types import SimpleNamespace

# Knuckle (MCP) x positions for index, middle, ring, pinky.
_MCP_X = (0.46, 0.50, 0.54, 0.58)
_MCP_Y = (0.60, 0.58, 0.60, 0.62)
_WRIST = (0.50, 0.80)


def _point(x, y):
    return SimpleNamespace(x=float(x), y=float(y), z=0.0)


def make_hand(extended=(False, False, False, False), thumb_out=False,
              dx=0.0, dy=0.0, scale=1.0, flip_x=False, flip_y=False):
    """A hand with the given finger states.

    extended  : (index, middle, ring, pinky) booleans
    thumb_out : thumb tip well away from the index knuckle
    dx, dy    : translation; scale: size about the wrist
    flip_x    : mirror horizontally (swaps which side the thumb is on)
    flip_y    : upside down — an open hand becomes a fingers-down hand
    """
    pts = [None] * 21
    pts[0] = _WRIST

    # Thumb chain 1-4.
    pts[1] = (0.45, 0.75)
    pts[2] = (0.41, 0.70)
    pts[3] = (0.38, 0.66)
    pts[4] = (0.30, 0.62) if thumb_out else (0.47, 0.64)

    for finger, is_ext in enumerate(extended):
        mcp_i = 5 + finger * 4
        x, y = _MCP_X[finger], _MCP_Y[finger]
        pts[mcp_i] = (x, y)
        if is_ext:
            pts[mcp_i + 1] = (x, y - 0.06)      # PIP
            pts[mcp_i + 2] = (x, y - 0.10)      # DIP
            pts[mcp_i + 3] = (x, y - 0.14)      # TIP — far from the wrist
        else:
            pts[mcp_i + 1] = (x, y - 0.05)
            pts[mcp_i + 2] = (x, y - 0.07)
            pts[mcp_i + 3] = (x, y + 0.01)      # folded back toward the palm

    wx, wy = _WRIST
    out = []
    for x, y in pts:
        x = wx + (x - wx) * scale
        y = wy + (y - wy) * scale
        if flip_x:
            x = 1.0 - x
        if flip_y:
            y = 1.0 - y
        out.append(_point(x + dx, y + dy))
    return SimpleNamespace(landmark=out)


POSES = {
    "grip": dict(extended=(False, False, False, False), thumb_out=False),
    "point": dict(extended=(True, False, False, False)),
    "peace": dict(extended=(True, True, False, False)),
    "open": dict(extended=(True, True, True, True), thumb_out=True),
}


def pose(name, **kwargs):
    spec = dict(POSES[name])
    spec.update(kwargs)
    return make_hand(**spec)


def handedness(*labels):
    """MediaPipe's multi_handedness shape for the given labels."""
    return [SimpleNamespace(classification=[SimpleNamespace(label=label, score=0.99)])
            for label in labels]
