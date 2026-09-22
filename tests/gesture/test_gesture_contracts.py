"""Cross-file contracts of the gesture module, and the Gesture Studio helpers.

gesture_config.json is written by the GUI and read by the engine; the
gestures/ artwork names the poses; hand_cursor_2 decides which labels the
classifiers can emit. A merge that breaks the agreement between those three
fails here instead of as a binding that silently never fires.
"""

import json
from pathlib import Path

import pytest

import gesture_fsm as g
from helpers.paths import GESTURE

CONFIG = GESTURE / "gesture_config.json"
ART = GESTURE / "gestures"


@pytest.fixture(scope="module")
def shipped():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def emittable(hc):
    """Every label the running engine can hand the FSM."""
    semantic = {hc.LABEL_ALIASES.get(c, c) for c in hc.YOLO_CLASS_NAMES} - {"no_gesture"}
    oriented = {f"{name}_inverse" for name in hc.ORIENTATION_GESTURES}
    return set(hc.GEOMETRY_VOCABULARY) | semantic | oriented | {"stop_inverse_down"}


def artwork():
    return {p.stem.lower() for p in ART.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")}


# ── The shipped configuration ──────────────────────────────────────────────────

def test_shipped_config_compiles_every_rule(shipped, capsys):
    fsm = g.GestureFSM(shipped)
    out = capsys.readouterr().out
    assert "skipping" not in out and "unusable" not in out
    live = [r for key in ("geometry_bindings", "yolo_bindings", "transitions", "holds")
            for r in shipped.get(key, []) if r.get("enabled", True)]
    assert len(fsm.rules) == len(live)


def test_shipped_rules_use_known_actions_and_unique_ids(shipped):
    rules = [r for key in ("geometry_bindings", "yolo_bindings", "deleted_bindings")
             for r in shipped.get(key, [])]
    ids = [r["id"] for r in rules if "id" in r]
    assert len(ids) == len(set(ids)), "rule ids must be unique across all lists"
    for r in rules:
        assert r["action"].upper() in g.ACTIONS, r


def test_shipped_rules_have_no_conflicting_duplicates(shipped):
    fsm = g.GestureFSM(shipped)
    rules = list(fsm.rules)
    for i, a in enumerate(rules):
        for b in rules[i + 1:]:
            if a.trigger_key() == b.trigger_key():
                assert not g.hands_can_coincide(a.hand, b.hand), \
                    f"{a.name!r} and {b.name!r} can fire on the same frame"


def test_shipped_rules_bind_poses_the_engine_can_emit(shipped, emittable):
    fsm = g.GestureFSM(shipped)
    for r in fsm.rules:
        poses = {r.pose} if isinstance(r, g.HoldRule) else {r.from_state, r.to_state}
        assert poses <= emittable, f"{r.name!r} waits for a pose no classifier emits"


def test_shipped_settings_are_within_gui_bounds(shipped):
    s = shipped["settings"]
    assert g.SENSITIVITY_MIN <= s["cursor_sensitivity"] <= g.SENSITIVITY_MAX
    assert g.AI_CONFIDENCE_MIN <= s["ai_confidence"] <= g.AI_CONFIDENCE_MAX
    assert s["is_mirrored"] != s["invert_cursor_x"], "exactly one mirror setting may be on"


def test_default_settings_are_within_gui_bounds():
    s = g.DEFAULT_SETTINGS
    assert g.SENSITIVITY_MIN <= s["cursor_sensitivity"] <= g.SENSITIVITY_MAX
    assert g.AI_CONFIDENCE_MIN <= s["ai_confidence"] <= g.AI_CONFIDENCE_MAX
    assert s["stability_threshold"] <= s["window_size"]


def test_engine_and_fsm_agree_on_bounds(hc):
    assert (hc.SENS_MIN, hc.SENS_MAX) == (g.SENSITIVITY_MIN, g.SENSITIVITY_MAX)
    assert (hc.AI_CONFIDENCE_MIN, hc.AI_CONFIDENCE_MAX, hc.AI_CONFIDENCE_DEFAULT) == \
        (g.AI_CONFIDENCE_MIN, g.AI_CONFIDENCE_MAX, g.AI_CONFIDENCE_DEFAULT)
    assert hc.CURSOR_SENSITIVITY == g.DEFAULT_SETTINGS["cursor_sensitivity"]


# ── Vocabulary ↔ artwork ───────────────────────────────────────────────────────

def test_label_aliases_land_on_real_artwork(hc):
    """The aliases exist so the model's "_inverted" spelling matches the
    gestures/ filenames; an alias pointing at no PNG is unbindable."""
    for target in hc.LABEL_ALIASES.values():
        assert target in artwork(), target


def test_every_bindable_label_has_artwork(hc, emittable, studio):
    missing = {label for label in emittable
               if not studio.GestureLibrary.is_excluded(label) and label not in artwork()}
    assert not missing, f"poses the GUI cannot offer (no PNG): {sorted(missing)}"


def test_every_artwork_is_a_label_the_engine_emits(emittable):
    orphans = artwork() - emittable
    assert not orphans, f"artwork for poses that can never fire: {sorted(orphans)}"


def test_yolo_class_list_matches_the_trained_model_size(hc):
    assert len(hc.YOLO_CLASS_NAMES) == 34 == len(set(hc.YOLO_CLASS_NAMES))


# ── Gesture Studio (app.py) — the non-GUI helpers ──────────────────────────────

@pytest.fixture(scope="module")
def studio():
    pytest.importorskip("PIL")
    pytest.importorskip("tkinter")
    import app
    return app


def test_pretty(studio):
    assert studio.pretty("little_finger") == "Little Finger"
    assert studio.pretty("ok") == "Ok"


@pytest.mark.parametrize("name, excluded", [
    ("timeout", True), ("Take Picture", True), ("hand-heart", True), ("x_sign", True),
    ("open", True), ("palm", False), ("three_gun", False), ("", False),
])
def test_exclusion_is_spelling_insensitive(studio, name, excluded):
    assert studio.GestureLibrary.is_excluded(name) is excluded


def test_library_scans_the_real_catalog(studio):
    lib = studio.GestureLibrary()
    assert lib.labels == sorted(lib.labels) and "palm" in lib.labels
    assert "open" not in lib.labels                       # byte copy of palm.png
    assert lib.path_for("Palm").endswith("palm.png")
    assert lib.path_for("nope") is None


def test_library_scan_rules(studio, tmp_path):
    for name in ("Point.PNG", "fist.jpg", "notes.txt", "timeout.png", "wave.webp"):
        (tmp_path / name).write_bytes(b"")
    lib = studio.GestureLibrary(str(tmp_path))
    assert lib.labels == ["fist", "point", "wave"]
    assert studio.GestureLibrary(str(tmp_path / "missing")).labels == []


def test_thumbnails_decode_to_square_rgba(studio):
    lib = studio.GestureLibrary()
    image = lib._decode(lib.path_for("palm"), 64)
    assert image.mode == "RGBA" and image.size == (64, 64)


def test_placeholder_and_background_loading(studio, tmp_path):
    from PIL import Image
    Image.new("RGB", (200, 100), "white").save(tmp_path / "ok.png")
    (tmp_path / "broken.png").write_bytes(b"not a png")
    lib = studio.GestureLibrary(str(tmp_path))
    lib._load_all(48)                                     # run the worker inline
    items = dict(item for item in lib.drain() if item[0])
    assert set(items) == {"ok", "broken"}
    assert all(img.size == (48, 48) for img in items.values())   # broken → placeholder
    assert studio.placeholder("little_finger", 32).size == (32, 32)


def test_gui_offers_every_action_except_the_free_text_macro(studio):
    assert set(studio.SELECTABLE_ACTIONS) == set(g.ACTIONS) - {g.KEYBOARD_MACRO}
    assert set(studio.CURSOR_SETTING_KEYS) >= {"cursor_sensitivity", "ai_confidence"}


def test_art_folder_exists():
    assert ART.is_dir() and artwork()
    assert Path(CONFIG).exists()
