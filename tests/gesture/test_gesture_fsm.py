"""gesture_fsm.py — the config-driven state machine that turns a stream of
pose labels into mouse/keyboard actions.

Frames are fed at 30 Hz with explicit timestamps, so every timing rule
(stabilisation, max_time, holds, cooldowns, double-click promotion, drag
failsafes) is tested deterministically.
"""

import json

import pytest

import gesture_fsm as g
from gesture_fsm import GestureFSM, HoldRule, MajorityStabilizer, TransitionRule

DT = 1 / 30


def rule(frm, to, action="LEFT_CLICK", **kw):
    return {"trigger": "transition", "from_state": frm, "to_state": to,
            "action": action, "cooldown_sec": 0.0, **kw}


def hold(pose, action, **kw):
    return {"trigger": "hold", "pose": pose, "action": action, "cooldown_sec": 0.0, **kw}


def machine(*rules, **settings):
    return GestureFSM({"settings": settings, "geometry_bindings": list(rules)})


def run(fsm, labels, t0=0.0, dt=DT, **frame):
    """Feed labels; returns [(frame_index, action)] for every action produced."""
    fired = []
    for i, label in enumerate(labels):
        action = fsm.update(label, t0 + i * dt, **frame)
        if action is not None:
            fired.append((i, str(action)))
    return fired


def actions(fired):
    return [a for _, a in fired]


# ── Stabiliser ─────────────────────────────────────────────────────────────────

def test_majority_needs_threshold_votes():
    s = MajorityStabilizer(3, 2)
    assert s.push("a") is None
    assert s.push("a") == "a"
    assert s.push("b") == "a"                  # one flicker changes nothing
    assert s.push("b") == "b"
    assert s.window == ("a", "b", "b")


def test_inconclusive_window_keeps_the_last_verdict():
    s = MajorityStabilizer(3, 3)
    for label in ("a", "a", "a", "b", "c"):
        stable = s.push(label)
    assert stable == "a"


def test_threshold_is_clamped_to_window_and_validated():
    assert MajorityStabilizer(2, 5).threshold == 2
    for bad in ((0, 1), (3, 0)):
        with pytest.raises(ValueError):
            MajorityStabilizer(*bad)
    s = MajorityStabilizer()
    s.push("x"), s.push("x")
    s.reset()
    assert s.stable is None and s.window == ()


def test_base_stabiliser_is_abstract():
    with pytest.raises(NotImplementedError):
        g.GestureStabilizer().push("x")


# ── Transitions ────────────────────────────────────────────────────────────────

def test_transition_fires_once_when_target_stabilises():
    fsm = machine(rule("point", "grip"))
    fired = run(fsm, ["point"] * 3 + ["grip"] * 4)
    assert fired == [(4, "LEFT_CLICK")]         # 2nd grip frame = stable
    assert fsm.stable_gesture == "grip" and fsm.previous_gesture == "point"
    assert fsm.click_count == 1


def test_chain_survives_a_stray_frame_and_an_unmapped_pose():
    # The gesture_fsm self-check script: point, stray peace, idle curl, grip.
    fsm = machine(rule("point", "grip"))
    fired = run(fsm, ["point", "point", "point", "peace", "idle", "idle", "grip", "grip", "grip"])
    assert actions(fired) == ["LEFT_CLICK"]


def test_one_intermediate_pose_is_tolerated_two_are_not():
    fsm = machine(rule("point", "grip"))
    assert actions(run(fsm, ["point"] * 3 + ["peace"] * 3 + ["grip"] * 3)) == ["LEFT_CLICK"]
    fsm = machine(rule("point", "grip"))
    labels = ["point"] * 3 + ["peace"] * 3 + ["open"] * 3 + ["grip"] * 3
    assert run(fsm, labels) == []


def test_transition_memory_setting_widens_the_window():
    fsm = machine(rule("point", "grip"), transition_memory=3)
    labels = ["point"] * 3 + ["peace"] * 3 + ["open"] * 3 + ["grip"] * 3
    assert actions(run(fsm, labels)) == ["LEFT_CLICK"]


def test_max_time_bounds_the_gap():
    fsm = machine(rule("point", "grip", max_time_sec=0.5))
    fired = run(fsm, ["point"] * 3 + ["peace"] * 30 + ["grip"] * 3)   # 1 s detour
    assert fired == []


def test_zero_max_time_means_unbounded():
    fsm = machine(rule("point", "grip", max_time_sec=0))
    assert actions(run(fsm, ["point"] * 3 + ["peace"] * 60 + ["grip"] * 3)) == ["LEFT_CLICK"]


def test_nearest_origin_wins():
    # point→grip = click, open→grip = drag. open→point→grip is a click only;
    # the farther origin must not ALSO press and hold the button.
    fsm = machine(rule("point", "grip"), rule("open", "grip", "DRAG_START"))
    fired = run(fsm, ["open"] * 3 + ["point"] * 3 + ["grip"] * 3)
    assert actions(fired) == ["LEFT_CLICK"] and not fsm.is_dragging


def test_one_arrival_cannot_be_claimed_twice():
    fsm = machine(rule("point", "grip"))
    fired = run(fsm, ["point"] * 3 + ["grip"] * 3 + ["peace"] * 3 + ["grip"] * 3)
    assert actions(fired) == ["LEFT_CLICK"]


def test_cooldown_blocks_rapid_repeats():
    fsm = machine(rule("point", "grip", cooldown_sec=1.0))
    cycle = ["point"] * 3 + ["grip"] * 3
    assert actions(run(fsm, cycle * 2)) == ["LEFT_CLICK"]
    later = run(fsm, cycle, t0=5.0)
    assert actions(later) == ["LEFT_CLICK"]


def test_ignored_states_never_become_state():
    fsm = machine(rule("point", "grip"))
    run(fsm, ["point"] * 3 + ["idle"] * 10)
    assert fsm.stable_gesture == "point"


def test_missing_label_is_the_hand_lost_state_not_an_abstention():
    # None/"" normalise to "none": losing the hand is a real state, because
    # it is exactly the event that must release a held drag.
    fsm = machine(rule("point", "grip"))
    run(fsm, ["point"] * 3 + [None, "", "  "])
    assert fsm.stable_gesture == g.NO_GESTURE


def test_custom_ignored_states():
    fsm = machine(rule("point", "grip"), ignored_states=["peace"])
    assert actions(run(fsm, ["point"] * 3 + ["peace"] * 5 + ["grip"] * 3)) == ["LEFT_CLICK"]


def test_labels_are_normalised():
    fsm = machine(rule("Point", "GRIP"))
    assert actions(run(fsm, [" POINT "] * 3 + ["grip"] * 3)) == ["LEFT_CLICK"]


# ── Holds ──────────────────────────────────────────────────────────────────────

def test_hold_is_edge_triggered():
    fsm = machine(hold("palm", "SHOW_DESKTOP", hold_sec=0.3))
    fired = run(fsm, ["palm"] * 60)                       # held 2 s
    assert actions(fired) == ["SHOW_DESKTOP"]
    idx = fired[0][0]
    assert (idx - 1) * DT >= 0.3                          # matured after hold_sec
    assert fsm.macro_count == 1


def test_leaving_the_pose_rearms_the_hold():
    fsm = machine(hold("palm", "MUTE", hold_sec=0.1))
    fired = run(fsm, ["palm"] * 10 + ["fist"] * 5 + ["palm"] * 10)
    assert actions(fired) == ["MUTE", "MUTE"]


def test_repeating_hold_fires_at_its_rate():
    fsm = machine(hold("point", "SCROLL_DOWN", hold_sec=0.0, repeat=True, repeat_sec=0.25))
    fired = run(fsm, ["point"] * 31)                      # ~1 s
    assert 4 <= len(fired) <= 5
    assert set(actions(fired)) == {"SCROLL_DOWN"}


def test_hold_ms_is_accepted():
    r = HoldRule({"pose": "fist", "action": "MUTE", "hold_ms": 250}, 0, g.DEFAULT_SETTINGS)
    assert r.hold_sec == pytest.approx(0.25)


def test_rules_on_the_same_trigger_fire_one_action():
    fsm = machine(hold("fist", "MUTE", hold_sec=0.1), hold("fist", "COPY", hold_sec=0.1),
                  hold("fist", "PASTE", hold_sec=0.1, hand="left"))
    fired = run(fsm, ["fist"] * 12, hand="right")
    assert actions(fired) == ["MUTE"]                     # first in config order wins


def test_simultaneous_actions_are_deferred_one_per_update():
    fsm = machine(rule("open", "fist", "PASTE"), hold("fist", "MUTE", hold_sec=0.0))
    fired = run(fsm, ["open"] * 3 + ["fist"] * 3)
    assert fired == [(4, "PASTE"), (5, "MUTE")]           # MUTE queued a frame


def test_poll_returns_everything_at_once():
    fsm = machine(hold("fist", "MUTE", hold_sec=0.0), hold("open", "COPY", hold_sec=0.0),
                  rule("open", "fist", "PASTE"))
    for i, label in enumerate(["open"] * 3):
        fsm.poll(label, i * DT)
    out = []
    for i, label in enumerate(["fist"] * 3, start=3):
        out += [str(a) for a in fsm.poll(label, i * DT)]
    assert out == ["PASTE", "MUTE"]            # both matured on the same frame


# ── Double-click promotion ─────────────────────────────────────────────────────

def cycle(n=1):
    return (["point"] * 3 + ["grip"] * 3) * n


def test_quick_repeat_promotes_to_double():
    fsm = machine(rule("point", "grip", promote_double=True, cooldown_sec=0.1),
                  double_click_sec=0.8)
    assert actions(run(fsm, cycle(2))) == ["LEFT_CLICK", "DOUBLE_CLICK"]
    assert (fsm.click_count, fsm.double_click_count) == (1, 1)


def test_slow_repeat_is_two_singles():
    fsm = machine(rule("point", "grip", promote_double=True), double_click_sec=0.1)
    assert actions(run(fsm, cycle(2))) == ["LEFT_CLICK", "LEFT_CLICK"]


def test_promotion_is_off_by_default():
    fsm = machine(rule("point", "grip"))
    assert actions(run(fsm, cycle(2))) == ["LEFT_CLICK", "LEFT_CLICK"]


def test_different_rules_never_merge_into_a_double():
    fsm = machine(rule("point", "grip", promote_double=True),
                  rule("peace", "grip", promote_double=True))
    labels = ["point"] * 3 + ["grip"] * 3 + ["peace"] * 3 + ["grip"] * 3
    assert actions(run(fsm, labels)) == ["LEFT_CLICK", "LEFT_CLICK"]


def test_blocking_cooldown_is_reported(capsys):
    machine(rule("point", "grip", promote_double=True, cooldown_sec=0.5), double_click_sec=0.8)
    assert "blocks the second click" in capsys.readouterr().out


# ── Drag safety ────────────────────────────────────────────────────────────────

def dragger(**settings):
    return machine(rule("open", "grip", "DRAG_START"), rule("grip", "open", "DRAG_STOP"),
                   **settings)


def test_drag_start_and_explicit_stop():
    fsm = dragger()
    fired = run(fsm, ["open"] * 3 + ["grip"] * 5 + ["open"] * 3)
    assert actions(fired) == ["DRAG_START", "DRAG_STOP"]
    assert not fsm.is_dragging
    assert (fsm.drag_start_count, fsm.drag_stop_count) == (1, 1)


def test_lost_hand_releases_the_drag():
    fsm = dragger(stuck_release_sec=0.3)
    fired = run(fsm, ["open"] * 3 + ["grip"] * 3 + [None] * 20)
    assert actions(fired) == ["DRAG_START", "DRAG_STOP"]
    stop_frame = fired[1][0]
    lost_frame = 6 + 1                              # "none" stabilises on the 2nd frame
    assert (stop_frame - lost_frame) * DT >= 0.3 - 1e-9


def test_unmapped_pose_releases_the_drag():
    fsm = dragger(stuck_release_sec=0.2)
    fired = run(fsm, ["open"] * 3 + ["grip"] * 3 + ["rock"] * 20)
    assert actions(fired)[-1] == "DRAG_STOP"


def test_brief_loss_does_not_release():
    fsm = dragger(stuck_release_sec=0.5)
    fired = run(fsm, ["open"] * 3 + ["grip"] * 3 + [None] * 4 + ["grip"] * 10)
    assert actions(fired) == ["DRAG_START"] and fsm.is_dragging


def test_drag_timeout(capsys):
    fsm = dragger(drag_timeout_sec=1.0)
    fired = run(fsm, ["open"] * 3 + ["grip"] * 60)   # held grip for 2 s
    assert actions(fired) == ["DRAG_START", "DRAG_STOP"]
    assert "time limit" in capsys.readouterr().out


def test_no_double_press_and_no_stray_release():
    fsm = machine(rule("open", "grip", "DRAG_START"), rule("peace", "grip", "DRAG_START"),
                  rule("point", "fist", "DRAG_STOP"))
    fired = run(fsm, ["point"] * 3 + ["fist"] * 3)          # stop while not dragging
    assert fired == []
    fired = run(fsm, ["open"] * 3 + ["grip"] * 3 + ["peace"] * 3 + ["grip"] * 3, t0=10)
    assert actions(fired) == ["DRAG_START"]


def test_reset_releases_a_held_drag():
    fsm = dragger()
    run(fsm, ["open"] * 3 + ["grip"] * 3)
    cleanup = fsm.reset(5.0)
    assert [str(a) for a in cleanup] == ["DRAG_STOP"]
    assert fsm.stable_gesture is None and not fsm.is_dragging


def test_internal_fault_falls_back_to_idle_without_stranding_the_button(capsys):
    fsm = dragger()
    run(fsm, ["open"] * 3 + ["grip"] * 3)

    class Broken(MajorityStabilizer):
        def push(self, label):
            raise RuntimeError("corrupt window")
    fsm._stabilizer = Broken()
    assert fsm.update("grip", 1.0) == "DRAG_STOP"
    assert not fsm.is_dragging
    fsm.update("grip", 1.1)                                  # still faulting
    assert capsys.readouterr().out.count("internal fault") == 1


# ── Hands and sources ──────────────────────────────────────────────────────────

def test_hand_requirement_filters_rules():
    fsm = machine(rule("point", "grip", hand="right"))
    assert run(fsm, cycle(), hand="left") == []
    fsm = machine(rule("point", "grip", hand="right"))
    assert actions(run(fsm, cycle(), hand="right")) == ["LEFT_CLICK"]


def test_unknown_side_satisfies_a_sided_rule_but_fires_once():
    fsm = machine(rule("point", "grip", "LEFT_CLICK", hand="right"),
                  rule("point", "grip", "RIGHT_CLICK", hand="left"))
    assert actions(run(fsm, cycle(), hand="")) == ["LEFT_CLICK"]     # config order wins


def test_both_hands_needs_two_in_frame():
    fsm = machine(hold("timeout", "LOCK_SCREEN", hold_sec=0.0, hand="both"))
    assert run(fsm, ["timeout"] * 5, hand_count=1) == []
    assert actions(run(fsm, ["timeout"] * 5, t0=1, hand_count=2)) == ["LOCK_SCREEN"]


def test_dropping_a_hand_disarms_a_both_hands_hold():
    fsm = machine(hold("timeout", "LOCK_SCREEN", hold_sec=0.0, hand="both"))
    run(fsm, ["timeout"] * 5, hand_count=2)
    run(fsm, ["timeout"] * 3, t0=1, hand_count=1)
    assert actions(run(fsm, ["timeout"] * 3, t0=2, hand_count=2)) == ["LOCK_SCREEN"]


def test_source_pinned_rules():
    fsm = machine(rule("point", "grip", source="semantic"))
    assert run(fsm, cycle(), source="geometry") == []
    fsm = machine(rule("point", "grip", source="semantic"))
    assert actions(run(fsm, cycle(), source=None)) == ["LEFT_CLICK"]


@pytest.mark.parametrize("value, expected", [
    ("R", "right"), ("Right Hand", "right"), ("lefthand", "left"), ("l", "left"),
    ("both", "both"), ("2", "both"), ("two hands", "both"), (None, "any"),
    ("", "any"), ("sideways", "any"),
])
def test_normalise_hand(value, expected):
    assert g.normalise_hand(value) == expected


@pytest.mark.parametrize("a, b, expected", [
    ("right", "left", False), ("left", "right", False), ("any", "left", True),
    ("both", "right", True), ("right", "right", True), ("any", "any", True),
])
def test_hands_can_coincide(a, b, expected):
    assert g.hands_can_coincide(a, b) is expected


# ── Rules & config compilation ─────────────────────────────────────────────────

@pytest.mark.parametrize("raw, message", [
    ({"from_state": "a", "to_state": "b"}, "no action"),
    ({"from_state": "a", "to_state": "a", "action": "COPY"}, "must differ"),
    ({"from_state": "a", "action": "COPY"}, "both from_state and to_state"),
    ({"from_state": "a", "to_state": "b", "action": "KEYBOARD_MACRO"}, "no keys"),
])
def test_invalid_transitions_are_rejected(raw, message):
    with pytest.raises(ValueError, match=message):
        TransitionRule(raw, 0, g.DEFAULT_SETTINGS)


def test_hold_needs_a_pose():
    with pytest.raises(ValueError, match="needs a pose"):
        HoldRule({"action": "MUTE"}, 0, g.DEFAULT_SETTINGS)


def test_bad_rules_cost_only_themselves(capsys):
    fsm = GestureFSM({"geometry_bindings": [
        rule("point", "grip"), {"from_state": "x", "action": ""}, "not a dict",
        {"trigger": "hold", "pose": "fist", "action": "MUTE"},
    ]})
    assert len(fsm.rules) == 2
    assert "skipping transition #1" in capsys.readouterr().out


def test_all_rule_lists_are_read_but_the_recycle_bin_is_not():
    fsm = GestureFSM({
        "transitions": [rule("a", "b")], "holds": [hold("c", "MUTE")],
        "yolo_bindings": [rule("d", "e", "COPY")], "bindings": [hold("f", "PASTE")],
        "mappings": [rule("g", "h"), {"trigger": "hold", "pose": "i", "action": "MUTE"}],
        "deleted_bindings": [rule("x", "y", "LOCK_SCREEN")],
    })
    assert len(fsm.rules) == 6
    assert all(r.action != "LOCK_SCREEN" for r in fsm.rules)


def test_disabled_rules_do_not_compile():
    fsm = GestureFSM({"geometry_bindings": [rule("a", "b", enabled=False)]})
    assert fsm.rules == ()


def test_defaults_fill_missing_timings():
    r = TransitionRule({"from_state": "a", "to_state": "b", "action": "copy"}, 3,
                       g.DEFAULT_SETTINGS)
    assert r.action == "COPY" and r.id == "rule-3"
    assert r.cooldown_sec == g.DEFAULT_SETTINGS["default_cooldown_sec"]
    assert r.max_time_sec == g.DEFAULT_SETTINGS["default_max_time_sec"]
    assert r.name == "a → b"


def test_hand_edited_garbage_is_coerced():
    fsm = GestureFSM({"settings": {"window_size": "five", "stability_threshold": None,
                                   "ignored_states": "idle", "double_click_sec": "nan"},
                      "geometry_bindings": [rule("a", "b", cooldown_sec="soon")]})
    assert fsm.stabilizer.window_size == 3
    assert fsm.rules[0].cooldown_sec == g.DEFAULT_SETTINGS["default_cooldown_sec"]


@pytest.mark.parametrize("value, fallback, expected", [
    ("1.5", 0, 1.5), ("x", 2, 2.0), (float("nan"), 3, 3.0), (None, "bad", 0.0),
])
def test_as_float(value, fallback, expected):
    assert g._as_float(value, fallback) == expected


def test_as_int_and_as_list():
    assert g._as_int("4.9", 1) == 4 and g._as_int("x", "y", minimum=2) == 2
    assert g._as_int(0, 3) == 1
    assert g._as_list("idle") == ["idle"] and g._as_list(None) == []
    assert g._as_list(("a", "b")) == ["a", "b"] and g._as_list(5) == []


def test_rule_round_trips_through_to_dict():
    raw = rule("point", "grip", "KEYBOARD_MACRO", keys="ctrl+z", hand="left",
               source="semantic", promote_double=True, max_time_sec=1.25)
    first = TransitionRule(raw, 0, g.DEFAULT_SETTINGS)
    again = TransitionRule(first.to_dict(), 0, g.DEFAULT_SETTINGS)
    assert again.to_dict() == first.to_dict()
    d = first.to_dict()
    assert d["keys"] == "ctrl+z" and d["hand"] == "left" and d["promote_double"] is True
    h = HoldRule(hold("fist", "SCROLL_UP", repeat=True, repeat_sec=0.01), 0, g.DEFAULT_SETTINGS)
    assert h.to_dict()["repeat_sec"] == 0.05                  # floor


def test_minimal_rule_dict_stays_minimal():
    d = TransitionRule(rule("a", "b"), 0, g.DEFAULT_SETTINGS).to_dict()
    assert "hand" not in d and "source" not in d and "keys" not in d


def test_action_event_is_a_string_with_context():
    ev = g.ActionEvent("SHOW_DESKTOP", rule_id="r1", rule_name="hold palm", at=2)
    assert ev == "SHOW_DESKTOP" and ev in g.MACRO_ACTIONS
    assert ev.keys == "win+d" and ev.rule_id == "r1" and ev.at == 2.0
    assert ev.action == "SHOW_DESKTOP" and "hold palm" in repr(ev)


def test_keyboard_macro_event_carries_its_chord():
    fsm = machine(hold("rock", "KEYBOARD_MACRO", keys="ctrl+shift+esc", hold_sec=0))
    ev = next(a for a in (fsm.update("rock", i * DT) for i in range(3)) if a)
    assert ev.keys == "ctrl+shift+esc"


def test_update_pair_and_describe():
    fsm = machine(rule("point", "grip"), hold("palm", "MUTE"))
    stable, action = fsm.update_pair("point", 0.0)
    assert (stable, action) == (None, None)
    assert "1 transition(s), 1 hold(s), 2/3 window" == fsm.describe()


def test_constructor_accepts_path_and_rejects_junk(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"geometry_bindings": [rule("a", "b")]}), encoding="utf-8")
    assert len(GestureFSM(str(path)).rules) == 1
    assert len(GestureFSM.from_config(str(path)).rules) == 1
    with pytest.raises(TypeError):
        GestureFSM(42)


def test_empty_config_is_a_supported_state(capsys):
    fsm = GestureFSM({})
    assert fsm.rules == () and fsm.update("point", 0.0) is None
    assert "no bindings configured" in capsys.readouterr().out


def test_normalise_gesture():
    assert g.normalise_gesture(None) == g.NO_GESTURE
    assert g.normalise_gesture("  ") == g.NO_GESTURE
    assert g.normalise_gesture(" Three_Gun ") == "three_gun"


def test_action_vocabulary_is_consistent():
    assert set(g.MOUSE_ACTIONS).isdisjoint(g.MACRO_ACTIONS)
    assert set(g.ACTIONS) == set(g.MOUSE_ACTIONS) | set(g.MACRO_ACTIONS)
    assert set(g.ACTION_MACROS) < set(g.MACRO_ACTIONS)
    assert set(g.__all__) <= set(dir(g))


# ── Config I/O ─────────────────────────────────────────────────────────────────

def test_load_missing_file_gives_empty_config(tmp_path, capsys):
    cfg = g.load_config(str(tmp_path / "nope.json"))
    assert cfg == g.empty_config()
    assert "not found" in capsys.readouterr().out


@pytest.mark.parametrize("content", ["{broken", "[1, 2]"])
def test_load_unreadable_file_gives_empty_config(tmp_path, content):
    path = tmp_path / "cfg.json"
    path.write_text(content, encoding="utf-8")
    assert g.load_config(str(path), quiet=True) == g.empty_config()


def test_load_fills_missing_lists(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"yolo_bindings": "oops"}), encoding="utf-8")
    cfg = g.load_config(str(path))
    for key in ("transitions", "holds", "geometry_bindings", "yolo_bindings", "deleted_bindings"):
        assert cfg[key] == []
    assert cfg["settings"] == {}


def test_save_is_atomic_and_round_trips(tmp_path):
    path = tmp_path / "cfg.json"
    cfg = g.empty_config()
    cfg["geometry_bindings"].append(rule("one", "fist"))
    g.save_config(cfg, str(path))
    assert not (tmp_path / "cfg.json.tmp").exists()
    assert g.load_config(str(path)) == {**cfg, "transitions": [], "holds": []}


def test_empty_config_has_exactly_one_mirror_setting_on():
    s = g.empty_config()["settings"]
    assert s["is_mirrored"] != s["invert_cursor_x"]
