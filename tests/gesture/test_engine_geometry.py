"""hand_cursor_2.py — landmark geometry: pose classification, handedness,
palm/gun orientation, the fingers-down pose, landmark plausibility, primary
hand tracking, inference sizing and the key handler."""

import math
from types import SimpleNamespace

import pytest

from helpers.hands import handedness, make_hand, pose


# ── detect_gesture ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name, extended", [
    ("grip", (False, False, False, False)),
    ("point", (True, False, False, False)),
    ("peace", (True, True, False, False)),
    ("open", (True, True, True, True)),
])
def test_named_poses(hc, name, extended):
    state, ext, _ = hc.detect_gesture(pose(name))
    assert state == name and ext == extended


@pytest.mark.parametrize("extended, thumb_out", [
    ((False, True, False, False), False),       # middle finger only
    ((False, False, False, False), True),       # fist with thumb out (a "like")
    ((True, True, True, False), False),         # three fingers
])
def test_in_between_shapes_abstain_as_idle(hc, extended, thumb_out):
    assert hc.detect_gesture(make_hand(extended, thumb_out))[0] == "idle"


def test_thumb_detection(hc):
    assert hc.detect_gesture(make_hand(thumb_out=True))[2] is True
    assert hc.detect_gesture(make_hand(thumb_out=False))[2] is False


@pytest.mark.parametrize("kwargs", [dict(scale=0.5), dict(scale=1.6), dict(dx=0.2, dy=-0.1),
                                    dict(flip_x=True)])
def test_classification_is_invariant_to_size_position_and_side(hc, kwargs):
    for name in ("grip", "point", "peace", "open"):
        assert hc.detect_gesture(pose(name, **kwargs))[0] == name


def test_real_frame_size_is_honoured(hc):
    # Passing the frame size must not change the verdict for an upright hand.
    assert hc.detect_gesture(pose("peace"), 1280, 720)[0] == "peace"


def test_degenerate_hand_refuses_to_guess(hc):
    flat = SimpleNamespace(landmark=[SimpleNamespace(x=0.5, y=0.5, z=0.0)] * 21)
    assert hc.detect_gesture(flat) == ("idle", (False, False, False, False), False)


# ── Handedness & orientation ───────────────────────────────────────────────────

def results(*hands_and_labels):
    hands = [h for h, _ in hands_and_labels]
    return SimpleNamespace(multi_hand_landmarks=hands,
                           multi_handedness=handedness(*[lbl for _, lbl in hands_and_labels]))


def test_handedness_is_taken_for_the_tracked_hand(hc):
    a, b = pose("open"), pose("point")
    res = results((a, "Left"), (b, "Right"))
    assert hc.handedness_of(res, b, mirrored=True) == "right"
    assert hc.handedness_of(res, a, mirrored=True) == "left"


def test_unmirrored_image_swaps_the_label(hc):
    a = pose("open")
    assert hc.handedness_of(results((a, "Left")), a, mirrored=False) == "right"


def test_unknown_handedness(hc):
    a = pose("open")
    assert hc.handedness_of(results((a, "Left")), pose("grip"), True) == ""   # not in frame
    assert hc.handedness_of(SimpleNamespace(multi_hand_landmarks=None,
                                            multi_handedness=None), a, True) == ""
    assert hc.handedness_of(results((a, "Ambidextrous")), a, True) == ""
    assert hc.handedness_of(object(), a, True) == ""


def with_x(hc, idx_a, xa, idx_b, xb):
    hand = pose("open")
    hand.landmark[idx_a].x, hand.landmark[idx_b].x = xa, xb
    return hand


@pytest.mark.parametrize("spread, side, mirrored, facing", [
    (+0.1, "right", False, True), (-0.1, "right", False, False),
    (+0.1, "right", True, False), (+0.1, "left", False, False),
    (+0.1, "left", True, True), (-0.1, "left", True, False),
])
def test_palm_facing_truth_table(hc, spread, side, mirrored, facing):
    hand = with_x(hc, hc.LM_INDEX_MCP, 0.5 + spread, hc.LM_PINKY_MCP, 0.5)
    assert hc.palm_is_facing(hand, side, mirrored) is facing


def test_palm_facing_is_undecidable_edge_on_or_without_side(hc):
    edge_on = with_x(hc, hc.LM_INDEX_MCP, 0.505, hc.LM_PINKY_MCP, 0.5)
    assert hc.palm_is_facing(edge_on, "right", False) is None
    assert hc.palm_is_facing(pose("open"), "", False) is None
    nan = with_x(hc, hc.LM_INDEX_MCP, float("nan"), hc.LM_PINKY_MCP, 0.5)
    assert hc.palm_is_facing(nan, "right", False) is None
    assert hc.palm_is_facing(object(), "right", False) is None


@pytest.mark.parametrize("reach, side, mirrored, facing", [
    (+0.2, "right", False, True), (-0.2, "right", False, False),
    (+0.2, "left", False, False), (+0.2, "right", True, False),
])
def test_gun_facing_truth_table(hc, reach, side, mirrored, facing):
    hand = with_x(hc, hc.LM_INDEX_TIP, 0.5 + reach, hc.LM_WRIST, 0.5)
    assert hc.gun_is_facing(hand, side, mirrored) is facing


def test_gun_deadband_is_wider_than_the_palm_one(hc):
    assert hc.GUN_DEADBAND > hc.PALM_DEADBAND
    hand = with_x(hc, hc.LM_INDEX_TIP, 0.53, hc.LM_WRIST, 0.5)
    assert hc.gun_is_facing(hand, "right", False) is None


def test_orientation_word(hc):
    assert [hc._orientation_word(v) for v in (True, False, None)] == ["palm", "back", "?"]


@pytest.mark.parametrize("label, expected", [
    ("three_gun_inverse_right", ("three_gun_inverse", "right")),
    ("stop_left", ("stop", "left")), ("palm", ("palm", "")), (None, ("", "")),
    ("upright", ("upright", "")),
])
def test_split_hand_suffix(hc, label, expected):
    assert hc.split_hand_suffix(label) == expected


# ── Fist-shaped class correction & the fingers-down pose ───────────────────────

def test_extended_finger_count(hc):
    assert hc.extended_finger_count((True, False, True, True)) == 3
    assert hc.extended_finger_count(None) == 0


@pytest.mark.parametrize("label, fingers, expected", [
    ("dislike", (True, True, True, False), True),     # an open hand is not a thumbs-down
    ("dislike", (False, False, False, False), False),  # a real fist stands
    ("dislike", (True, True, False, False), False),    # 2 fingers: not enough to veto
    ("palm", (True, True, True, True), False),         # never touches other classes
])
def test_contradicts_closed_hand(hc, label, fingers, expected):
    assert hc.contradicts_closed_hand(label, fingers) is expected


def test_pointing_down_needs_an_open_downward_hand(hc):
    assert hc.is_hand_pointing_down(pose("open", flip_y=True)) is True
    assert hc.is_hand_pointing_down(pose("open")) is False            # upright
    assert hc.is_hand_pointing_down(pose("grip", flip_y=True)) is False
    assert hc.is_hand_pointing_down(pose("peace", flip_y=True)) is False
    assert hc.is_hand_pointing_down(object()) is False


# ── LandmarkValidator ──────────────────────────────────────────────────────────

def test_real_hands_are_accepted(hc):
    v = hc.LandmarkValidator()
    for name in ("grip", "point", "peace", "open"):
        assert v.check(pose(name)) == (True, "")
    assert v.rejected == 0


def corrupt(fn):
    hand = pose("open")
    fn(hand.landmark)
    return hand


@pytest.mark.parametrize("hand_fn, reason", [
    (lambda lm: setattr(lm[8], "x", float("nan")), "non-finite"),
    (lambda lm: setattr(lm[8], "y", float("inf")), "non-finite"),
    (lambda lm: lm.pop(), "malformed"),
    (lambda lm: setattr(lm[8], "x", 1.8), "far outside the frame"),
    (lambda lm: setattr(lm[0], "y", 1.25), "impossibly long"),
])
def test_corrupt_frames_are_held(hc, hand_fn, reason):
    ok, why = hc.LandmarkValidator().check(corrupt(hand_fn))
    assert not ok and reason in why


def test_degenerate_hand_is_rejected(hc):
    dot = SimpleNamespace(landmark=[SimpleNamespace(x=0.5, y=0.5)] * 21)
    assert hc.LandmarkValidator().check(dot) == (False, "degenerate hand")


def test_size_cannot_jump_between_frames(hc):
    v = hc.LandmarkValidator()
    assert v.check(pose("open"))[0]
    assert v.check(pose("open", dx=0.01))[0]                          # normal motion
    assert v.check(pose("open", scale=0.3)) == (False, "hand changed size impossibly fast")


def test_wrist_cannot_teleport(hc):
    # The limit is LM_MAX_WRIST_JUMP median bone lengths; a small (distant)
    # hand makes it small enough to exceed while staying inside the frame.
    v = hc.LandmarkValidator()
    assert v.check(pose("open", scale=0.5))[0]
    assert v.check(pose("open", scale=0.5, dx=0.3, dy=-0.2)) == (False, "wrist teleported")


def test_stale_reference_is_dropped_after_repeated_rejects(hc):
    v = hc.LandmarkValidator()
    v.check(pose("open"))
    far = pose("open", scale=0.35)
    for _ in range(hc.LM_MAX_HELD_FRAMES):
        assert not v.check(far)[0]
    assert v.check(far) == (True, "")        # judged on its own merits again
    v.reset()
    assert (v.held, v.rejected) == (0, 0)


# ── Primary-hand tracking ──────────────────────────────────────────────────────

def test_primary_hand_follows_the_previous_anchor(hc):
    left, right = pose("open", dx=-0.3), pose("point", dx=0.3)
    anchor = right.landmark[hc.LM_MIDDLE_MCP]
    assert hc.pick_primary_hand([left, right], (anchor.x, anchor.y)) is right
    assert hc.pick_primary_hand([left, right], None) is left
    assert hc.pick_primary_hand(right, (0, 0)) is right                 # single hand object
    assert hc.pick_primary_hand([], None) is None and hc.pick_primary_hand(None) is None


def test_primary_hand_tracking_can_be_disabled(hc, monkeypatch):
    monkeypatch.setattr(hc, "PRIMARY_HAND_TRACKING", False)
    left, right = pose("open", dx=-0.3), pose("point", dx=0.3)
    assert hc.pick_primary_hand([left, right], (0.8, 0.5)) is left


# ── Sizing, margins, keys ──────────────────────────────────────────────────────

@pytest.mark.parametrize("w, h, expected", [
    (640, 480, (640, 480, 1)),
    (1280, 720, (640, 360, 2)),
    (1920, 1080, (640, 360, 3)),
])
def test_integer_downscale_keeps_aspect(hc, w, h, expected):
    iw, ih, div, interp = hc.pick_infer_size(w, h)
    assert (iw, ih, div) == expected and interp == hc.cv2.INTER_AREA


def test_fractional_downscale_derives_height_from_true_ratio(hc):
    iw, ih, div, interp = hc.pick_infer_size(1366, 768)
    assert iw == 640 and ih == round(640 * 768 / 1366)
    assert interp == hc.cv2.INTER_LINEAR and div == pytest.approx(1366 / 640)


def test_margins_are_sanitised(hc, monkeypatch):
    assert hc.sanitise_margins() == (hc.MARGIN_X, hc.MARGIN_TOP, hc.MARGIN_BOTTOM)
    monkeypatch.setattr(hc, "MARGIN_X", 0.8)
    monkeypatch.setattr(hc, "MARGIN_TOP", 0.6)
    monkeypatch.setattr(hc, "MARGIN_BOTTOM", 0.9)
    mx, mt, mb = hc.sanitise_margins()
    assert mx == 0.49
    assert mt + mb == pytest.approx(0.95) and mt / mb == pytest.approx(0.6 / 0.9)
    monkeypatch.setattr(hc, "MARGIN_X", -1)
    assert hc.sanitise_margins()[0] == 0.0


def test_key_handler(hc, monkeypatch, capsys):
    monkeypatch.setattr(hc, "CURSOR_SENSITIVITY", 1.4)
    monkeypatch.setattr(hc, "IS_MIRRORED", True)
    monkeypatch.setattr(hc, "INVERT_CURSOR_X", False)
    assert hc.handle_key(ord("+")) is False and hc.CURSOR_SENSITIVITY == 1.5
    hc.handle_key(ord("_"))
    assert hc.CURSOR_SENSITIVITY == 1.4                    # rounded, no float drift
    for _ in range(40):
        hc.handle_key(ord("="))
    assert hc.CURSOR_SENSITIVITY == hc.SENS_MAX
    for _ in range(40):
        hc.handle_key(ord("-"))
    assert hc.CURSOR_SENSITIVITY == hc.SENS_MIN
    hc.handle_key(ord("m"))
    hc.handle_key(ord("I"))
    assert hc.IS_MIRRORED is False and hc.INVERT_CURSOR_X is True
    assert hc.handle_key(ord("q")) is True
    assert hc.handle_key(ord("z")) is False


def test_clamp(hc):
    assert (hc.clamp(5, 0, 3), hc.clamp(-1, 0, 3), hc.clamp(2, 0, 3)) == (3, 0, 2)


def test_bone_lengths_of_the_fixture_hand_are_realistic(hc):
    # Guards the fixture itself: if helpers.hands drifts, the validator tests
    # above would silently start testing nonsense.
    lm = pose("open").landmark
    palm = math.dist((lm[0].x, lm[0].y), (lm[9].x, lm[9].y))
    assert 0.15 < palm < 0.3
