"""nlp/intents.py — catalog invariants.

"Adding a capability = one Intent entry + one handler." These tests are what
make that sentence true on every merge: a new intent without a handler, a
pattern naming an undeclared slot, or a config set drifting away from the
catalog fails here instead of at runtime in front of a user.
"""

import re

import pytest

from central_module import config
from central_module.actions.executor import ActionExecutor
from central_module.nlp.intent_engine import _SLOT_REGEX
from central_module.nlp.intents import BY_NAME, CATALOG, DIRECTIONS, KEY_ALIASES

_SLOT_RE = re.compile(r"\{(\w+)\}")


def test_intent_names_are_unique():
    names = [i.name for i in CATALOG]
    assert len(names) == len(set(names))
    assert set(BY_NAME) == set(names)


def test_catalog_size_matches_documented_vocabulary():
    # README: "49 voice intents + refusal". Update both together.
    assert len(CATALOG) == 49


@pytest.mark.parametrize("intent", CATALOG, ids=lambda i: i.name)
def test_every_action_has_a_handler(intent):
    if intent.action == "manager":
        return                              # handled by CentralManager
    assert hasattr(ActionExecutor, f"_do_{intent.action}"), \
        f"intent {intent.name!r} routes to missing handler _do_{intent.action}"


@pytest.mark.parametrize("intent", CATALOG, ids=lambda i: i.name)
def test_patterns_only_use_declared_slots(intent):
    assert intent.patterns, f"{intent.name} has no patterns"
    for pattern in intent.patterns:
        for slot in _SLOT_RE.findall(pattern):
            assert slot in intent.slots, \
                f"{intent.name}: pattern {pattern!r} uses undeclared slot {slot!r}"


@pytest.mark.parametrize("intent", CATALOG, ids=lambda i: i.name)
def test_slot_kinds_are_known(intent):
    for kind in intent.slots.values():
        assert kind in _SLOT_REGEX, f"{intent.name}: unknown slot kind {kind!r}"


@pytest.mark.parametrize("intent", CATALOG, ids=lambda i: i.name)
def test_declared_slots_are_used_by_some_pattern(intent):
    used = {s for p in intent.patterns for s in _SLOT_RE.findall(p)}
    assert set(intent.slots) <= used, f"{intent.name}: slot declared but never captured"


@pytest.mark.parametrize("intent", CATALOG, ids=lambda i: i.name)
def test_patterns_are_already_normalised(intent):
    # Patterns are matched against normalised text, so a pattern with capitals,
    # punctuation or doubled spaces could never match anything.
    for pattern in intent.patterns:
        literal = _SLOT_RE.sub("x", pattern)
        assert literal == literal.lower().strip()
        assert "  " not in literal
        assert re.fullmatch(r"[a-z0-9' ]+", literal), pattern


def test_no_pattern_is_shared_between_intents():
    seen = {}
    for intent in CATALOG:
        for pattern in intent.patterns:
            assert pattern not in seen, \
                f"{pattern!r} belongs to both {seen[pattern]} and {intent.name}"
            seen[pattern] = intent.name


def test_confirm_flags_match_config():
    assert {i.name for i in CATALOG if i.confirm} == config.CONFIRM_REQUIRED


def test_confirm_and_cancel_phrases_are_recognised_by_the_manager():
    assert set(BY_NAME["confirm"].patterns) <= config.CONFIRM_WORDS
    assert set(BY_NAME["cancel"].patterns) <= config.CANCEL_WORDS
    assert not (config.CONFIRM_WORDS & config.CANCEL_WORDS)


def test_dictation_intents_agree_with_manager_trigger_sets():
    assert set(BY_NAME["start_dictation"].patterns) <= config.DICTATION_START
    assert set(BY_NAME["stop_dictation"].patterns) <= config.DICTATION_STOP
    assert not (config.DICTATION_START & config.DICTATION_STOP)


def test_key_aliases_map_to_canonical_ids():
    canonical = set(KEY_ALIASES.values())
    # Every canonical id is itself a valid spoken form (idempotent mapping).
    for key in canonical:
        assert KEY_ALIASES[key] == key
    assert {"enter", "escape", "tab", "space", "delete", "win", "ctrl"} <= canonical


def test_directions():
    assert DIRECTIONS == {"up", "down", "left", "right", "top", "bottom"}
