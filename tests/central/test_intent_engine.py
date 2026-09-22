"""nlp/intent_engine.py — the matching ladder: alias → exact → model → fuzzy."""

import re

import pytest

from central_module import config
from central_module.nlp import intent_engine as ie
from central_module.nlp.intent_engine import IntentEngine, IntentMatch, _validate
from central_module.nlp.intents import BY_NAME, CATALOG


@pytest.fixture(scope="module")
def engine():
    # Module-scoped for speed (compiling ~180 patterns per test adds up), so
    # it pins the rule ladder itself rather than relying on the per-test
    # autouse fixture, which runs after module-scoped setup.
    original = config.NLP["engine"]
    config.NLP["engine"] = "rules"
    try:
        yield IntentEngine()
    finally:
        config.NLP["engine"] = original


# ── Reachability: every catalog pattern must be matchable verbatim ─────────────

_SAMPLE = {"app": "notepad", "text": "hello world", "number": "40",
           "count": "3", "direction": "down", "key": "enter"}
_TYPED = {"number": 40, "count": 3}

# Patterns the exact matcher can never hit, because command_normalize()
# rewrites the utterance before matching: "double"/"twice" become the digit 2
# and determiners are dropped. In rules mode (no trained model — a fresh
# clone, or CI) "double click" therefore falls to fuzzy and fires a SINGLE
# click. Strict xfail: fixing the pattern or the normaliser turns these into
# failures that ask for this entry to be removed.
KNOWN_UNREACHABLE = {
    ("double_click", "double click"):
        "command_normalize turns 'double' into '2' before matching",
    ("double_click", "click twice"):
        "command_normalize turns 'twice' into '2' before matching",
    ("press_key", "press the {key} key"):
        "command_normalize drops 'the', so a non-text pattern containing it never matches",
}


def _cases():
    for intent in CATALOG:
        for pattern in intent.patterns:
            marks = []
            reason = KNOWN_UNREACHABLE.get((intent.name, pattern))
            if reason:
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            yield pytest.param(intent, pattern, id=f"{intent.name}:{pattern}", marks=marks)


@pytest.mark.parametrize("intent, pattern", list(_cases()))
def test_every_pattern_is_reachable_by_exact_match(engine, intent, pattern):
    filled = re.sub(r"\{(\w+)\}", lambda m: _SAMPLE[intent.slots[m.group(1)]], pattern)
    match = engine.parse(filled)
    assert match is not None, f"{filled!r} matched nothing"
    assert (match.intent.name, match.source) == (intent.name, "exact"), \
        f"{filled!r} → {match}"
    for name in re.findall(r"\{(\w+)\}", pattern):
        kind = intent.slots[name]
        assert match.slots[name] == _TYPED.get(kind, _SAMPLE[kind])


# ── Exact matching and slots ───────────────────────────────────────────────────

@pytest.mark.parametrize("text, intent, slots", [
    ("open chrome", "open_app", {"app": "chrome"}),
    ("could you open the calculator please", "open_app", {"app": "calculator"}),
    ("Launch VS Code", "open_app", {"app": "vs code"}),
    ("edit chatter.txt", "open_app", {"app": "chatter.txt"}),
    ("set volume to fifty five", "set_volume", {"number": 55}),
    ("increase volume to 67", "set_volume", {"number": 67}),        # set, not nudge
    ("increase volume by five", "volume_up", {"count": 5}),
    ("volume up", "volume_up", {}),
    ("scroll down three times", "scroll", {"direction": "down", "count": 3}),
    ("scroll to top", "scroll", {"direction": "top"}),
    ("press page up", "press_key", {"key": "page up"}),
    ("hit escape", "press_key", {"key": "escape"}),
    ("press key return", "press_key", {"key": "enter"}),            # alias → canonical
    ("go to github dot com", "open_url", {"text": "github dot com"}),
    ("go to desktop", "show_desktop", {}),                          # more literals win
    ("start dictation", "start_dictation", {}),                     # not open_app "dictation"
    ("open new tab", "new_tab", {}),
    ("shut down", "shutdown", {}),
    ("switch language to hindi", "set_language", {"text": "hindi"}),
])
def test_exact_parses(engine, text, intent, slots):
    match = engine.parse(text)
    assert match is not None
    assert match.intent.name == intent
    assert match.slots == slots
    assert match.source == "exact"
    assert match.confidence == 1.0


def test_text_slot_keeps_every_dictated_word(engine):
    match = engine.parse("Type The report is due on Monday.")
    assert match.intent.name == "type_text"
    # Light form: lowercased, punctuation gone, determiners KEPT.
    assert match.slots == {"text": "the report is due on monday"}


def test_match_records_normalised_command_text(engine):
    assert engine.parse("Please open the Calculator").text == "open calculator"


@pytest.mark.parametrize("text", [
    "", "   ", "?!", "what is the meaning of life", "the weather is nice today",
    "set volume to 1000",           # out of the 0-999 slot range
    "scroll sideways",              # not a direction
    "press banana",                 # not a key
])
def test_refuses_rather_than_guesses(engine, text):
    assert engine.parse(text) is None


# ── Fuzzy fallback ─────────────────────────────────────────────────────────────

def test_fuzzy_prefix_keeps_the_slot(engine):
    match = engine.parse("opun chrome")
    assert (match.intent.name, match.slots, match.source) == \
        ("open_app", {"app": "chrome"}, "fuzzy")
    assert 0 < match.confidence < 1.0


def test_fuzzy_fixed_phrase(engine):
    match = engine.parse("volum up")
    assert (match.intent.name, match.source) == ("volume_up", "fuzzy")


def test_fuzzy_score_always_below_exact(engine):
    match = engine.parse("volum up")
    assert match.confidence <= config.NLP["fuzzy_penalty"]


@pytest.mark.parametrize("text", [
    "delete xhatter",               # one wrong word must not pass as "delete that"
    "delete chatter.txt",           # deleting files is not a capability
    "yet",                          # "yet" vs "yes" is 0.67 — below the floor
])
def test_fuzzy_guards_reject_near_misses(engine, text):
    assert engine.parse(text) is None


def test_fuzzy_never_drops_a_payload(engine):
    # "increase volume to sixty sveven" must not collapse onto the fixed
    # phrase "increase volume" with the level silently lost.
    match = engine.parse("increase volume to sixty sveven")
    assert match is None or match.intent.name == "set_volume"


def test_threshold_is_configurable():
    strict = IntentEngine(fuzzy_threshold=1.01)
    assert strict.parse("opun chrome") is None
    assert strict.parse("open chrome").source == "exact"


# ── Learned aliases (adaptation fast path) ─────────────────────────────────────

def test_alias_wins_over_everything():
    seen = []

    def provider(phrase):
        seen.append(phrase)
        return {"intent": "open_app", "slots": {"app": "spotify"}}

    match = IntentEngine(alias_provider=provider).parse("Music Time please")
    assert seen == ["music time"]                      # command form is looked up
    assert (match.intent.name, match.slots, match.source) == \
        ("open_app", {"app": "spotify"}, "alias")


def test_alias_slots_are_copied():
    stored = {"intent": "open_app", "slots": {"app": "spotify"}}
    match = IntentEngine(alias_provider=lambda p: stored).parse("music time")
    match.slots["app"] = "mutated"
    assert stored["slots"]["app"] == "spotify"


def test_alias_naming_an_unknown_intent_is_ignored():
    engine = IntentEngine(alias_provider=lambda p: {"intent": "teleport", "slots": {}})
    assert engine.parse("open chrome").source == "exact"


def test_alias_provider_returning_none_falls_through():
    engine = IntentEngine(alias_provider=lambda p: None)
    assert engine.parse("open chrome").source == "exact"


# ── Slot validation ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind, raw, expected", [
    ("number", "55", 55), ("count", "0", 0), ("number", "999", 999),
    ("number", "1000", None), ("number", "60 sveven", None), ("count", "", None),
    ("direction", "up", "up"), ("direction", "sideways", None),
    ("key", "return", "enter"), ("key", "banana", None),
    ("app", "  chrome ", "chrome"), ("text", "   ", None),
])
def test_validate(kind, raw, expected):
    assert _validate(kind, raw) == expected


def test_intent_match_repr():
    text = repr(IntentMatch(BY_NAME["copy"], {}, 0.5, "fuzzy"))
    assert "copy" in text and "0.50" in text and "fuzzy" in text


# ── Neural rung wiring (with a stand-in model) ─────────────────────────────────

class _FakeNeural:
    trusted = True

    def __init__(self, *a, **k):
        self.calls = []

    def parse(self, text):
        self.calls.append(text)
        if "louder" in text:
            return "volume_up", {}, 0.97
        return None


@pytest.fixture
def neural_engine(monkeypatch):
    from central_module.nlp.nlu import engine as nlu_engine
    monkeypatch.setattr(nlu_engine, "NeuralNLU", _FakeNeural)
    monkeypatch.setitem(config.NLP, "engine", "neural")
    return IntentEngine()


def test_neural_mode_is_reported(neural_engine):
    assert neural_engine.mode == "neural"


def test_model_answers_paraphrases(neural_engine):
    match = neural_engine.parse("make it louder")
    assert (match.intent.name, match.source) == ("volume_up", "model")
    assert match.confidence == pytest.approx(0.97)


def test_exact_patterns_bypass_the_model(neural_engine):
    assert neural_engine.parse("open chrome").source == "exact"
    assert neural_engine._neural.calls == []


@pytest.mark.parametrize("error", [FileNotFoundError("no model"),
                                   ImportError("no transformers"),
                                   RuntimeError("corrupt weights")])
def test_model_load_failure_degrades_to_rules(monkeypatch, error):
    from central_module.nlp.nlu import engine as nlu_engine

    def boom(*a, **k):
        raise error

    monkeypatch.setattr(nlu_engine, "NeuralNLU", boom)
    monkeypatch.setitem(config.NLP, "engine", "neural")
    engine = IntentEngine()
    assert engine.mode == "rules"
    assert engine.parse("open chrome").intent.name == "open_app"


def test_module_exports_slot_regex_for_every_kind():
    assert set(ie._SLOT_REGEX) == {"app", "text", "number", "count", "direction", "key"}
