"""learning/ — PreferenceStore (what the user chose) and BehaviorModel (what
the system inferred)."""

import json
import os

import pytest

from central_module import config
from central_module.learning.adaptation import BehaviorModel
from central_module.learning.preferences import DEFAULTS, PreferenceStore, _deep_merge


# ── PreferenceStore ────────────────────────────────────────────────────────────

@pytest.fixture
def prefs_path(tmp_path):
    return str(tmp_path / "nested" / "preferences.json")


def test_defaults_on_first_run(prefs_path):
    store = PreferenceStore(prefs_path)
    assert store.get("language") == "en-in"
    assert store.get("gesture.enabled") is False
    assert store.get("gesture.command") == []
    assert not os.path.exists(prefs_path)          # nothing written until a set()


def test_set_persists_atomically_and_reloads(prefs_path):
    store = PreferenceStore(prefs_path)
    store.set("language", "hi-in")
    store.set("gesture.command", ["python", "app.py"])
    assert not os.path.exists(prefs_path + ".tmp")
    again = PreferenceStore(prefs_path)
    assert again.get("language") == "hi-in"
    assert again.get("gesture.command") == ["python", "app.py"]
    assert again.get("gesture.enabled") is False     # sibling default survives


def test_set_without_persist(prefs_path):
    store = PreferenceStore(prefs_path)
    store.set("language", "en-us", persist=False)
    assert store.get("language") == "en-us"
    assert not os.path.exists(prefs_path)


def test_get_missing_and_through_non_dict(prefs_path):
    store = PreferenceStore(prefs_path)
    assert store.get("nope") is None
    assert store.get("nope.deeper", "fallback") == "fallback"
    assert store.get("language.child") is None       # "language" is a str


def test_get_returns_copies(prefs_path):
    store = PreferenceStore(prefs_path)
    store.get("gesture")["enabled"] = True
    assert store.get("gesture.enabled") is False


def test_stored_file_is_merged_over_new_defaults(prefs_path):
    os.makedirs(os.path.dirname(prefs_path))
    with open(prefs_path, "w", encoding="utf-8") as fh:
        json.dump({"gesture": {"enabled": True}, "custom": 1}, fh)
    store = PreferenceStore(prefs_path)
    assert store.get("gesture.enabled") is True
    assert store.get("gesture.command") == []        # filled from DEFAULTS
    assert store.get("custom") == 1


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", ""])
def test_corrupt_file_falls_back_to_defaults(prefs_path, content):
    os.makedirs(os.path.dirname(prefs_path))
    with open(prefs_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    assert PreferenceStore(prefs_path).get("language") == "en-in"


def test_app_alias_learning(prefs_path):
    store = PreferenceStore(prefs_path)
    assert store.get_app_alias("my editor") is None
    store.learn_app_alias("my editor", "code")
    assert PreferenceStore(prefs_path).get_app_alias("my editor") == "code"


def test_learning_an_unchanged_alias_does_not_rewrite(prefs_path, monkeypatch):
    store = PreferenceStore(prefs_path)
    store.learn_app_alias("editor", "code")
    writes = []
    monkeypatch.setattr(store, "save", lambda: writes.append(1))
    store.learn_app_alias("editor", "code")
    assert writes == []


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}}
    merged = _deep_merge(base, {"a": {"c": 2}})
    assert merged == {"a": {"b": 1, "c": 2}} and base == {"a": {"b": 1}}


def test_defaults_are_not_shared_between_stores(tmp_path):
    one = PreferenceStore(str(tmp_path / "1.json"))
    one.set("app_aliases.x", "y", persist=False)
    assert DEFAULTS["app_aliases"] == {}


# ── BehaviorModel ──────────────────────────────────────────────────────────────

@pytest.fixture
def usage(tmp_path):
    return str(tmp_path / "usage.json")


def _hits(model, phrase, n, ok=True, source="fuzzy"):
    for _ in range(n):
        model.record_intent(phrase, "open_app", {"app": "chrome"}, source, ok)


def test_fuzzy_phrase_promoted_after_repeated_success(usage):
    model = BehaviorModel(usage)
    _hits(model, "opun chrome", config.ADAPTATION["alias_promote_hits"] - 1)
    assert model.get_alias("opun chrome") is None
    _hits(model, "opun chrome", 1)
    assert model.get_alias("opun chrome") == {"intent": "open_app", "slots": {"app": "chrome"}}


def test_model_matches_are_promoted_too(usage):
    model = BehaviorModel(usage)
    _hits(model, "fire up chrome", 3, source="model")
    assert model.get_alias("fire up chrome") is not None


def test_exact_matches_are_never_promoted(usage):
    model = BehaviorModel(usage)
    _hits(model, "open chrome", 10, source="exact")
    assert model.get_alias("open chrome") is None


def test_unreliable_phrase_is_not_promoted(usage):
    model = BehaviorModel(usage)
    _hits(model, "opun chrome", 2, ok=False)
    _hits(model, "opun chrome", 3)          # 3 hits vs 2 fails: hits > 2*fails is false
    assert model.get_alias("opun chrome") is None


def test_alias_is_demoted_when_it_starts_failing(usage):
    model = BehaviorModel(usage)
    _hits(model, "opun chrome", 3)
    _hits(model, "opun chrome", 4, ok=False)
    assert model.get_alias("opun chrome") is None


def test_failed_intents_do_not_count_as_usage(usage):
    model = BehaviorModel(usage)
    model.record_intent("lock pc", "lock", {}, "exact", ok=False)
    model.record_intent("lock pc", "lock", {}, "exact", ok=True)
    assert model._data["intent_counts"] == {"lock": 1}
    stats = model._data["phrase_stats"]["lock pc"]
    assert (stats["hits"], stats["fails"]) == (1, 1)


def test_get_alias_returns_a_copy(usage):
    model = BehaviorModel(usage)
    _hits(model, "opun chrome", 3)
    model.get_alias("opun chrome")["intent"] = "hijacked"
    assert model.get_alias("opun chrome")["intent"] == "open_app"


def test_unknown_phrases_are_bounded(usage, monkeypatch):
    monkeypatch.setitem(config.ADAPTATION, "max_unknown", 3)
    model = BehaviorModel(usage)
    for phrase, n in (("a", 5), ("b", 4), ("c", 3), ("d", 1)):
        for _ in range(n):
            model.record_unknown(phrase)
    assert len(model._data["unknown"]) == 3
    assert "a" in model._data["unknown"]
    assert model.suggestions(2) == [("a", 5), ("b", 4)]


def test_saves_are_throttled(usage, monkeypatch):
    monkeypatch.setitem(config.ADAPTATION, "save_every", 3)
    model = BehaviorModel(usage)
    model.record_app("chrome")
    model.record_app("chrome")
    assert not os.path.exists(usage)
    model.record_app("chrome")
    assert os.path.exists(usage)


def test_round_trip_and_summary(usage):
    model = BehaviorModel(usage)
    _hits(model, "opun chrome", 3)
    model.record_app("chrome")
    model.record_unknown("do a barrel roll")
    model.save()
    again = BehaviorModel(usage)
    assert again.get_alias("opun chrome")
    summary = again.summary()
    assert "aliases learned : 1" in summary
    assert "open_app×3" in summary and "chrome×1" in summary
    assert "'do a barrel roll'×1" in summary


def test_empty_summary_uses_dashes(usage):
    assert BehaviorModel(usage).summary().count("—") == 3


@pytest.mark.parametrize("content", ["{bad", '{"aliases": [1, 2]}', "null"])
def test_corrupt_usage_file_starts_fresh(usage, content):
    with open(usage, "w", encoding="utf-8") as fh:
        fh.write(content)
    model = BehaviorModel(usage)
    assert model._data["aliases"] == {}
    model.record_app("x")           # and remains writable
