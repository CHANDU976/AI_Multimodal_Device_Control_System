"""classifier.py, vosk_engine.parse_final and router.py — the decision of
whether an utterance is acted on as a command or escalated to Whisper."""

import pytest

from classifier import Classifier
from config import DICTATION_TRIGGERS, ROUTING
from router import Decision, Router
from vosk_engine import VoskResult, parse_final


# ── Classifier (explicit mode switching) ───────────────────────────────────────

def test_starts_in_command_mode():
    assert Classifier().mode == "command"


def test_command_text_is_lowercased():
    result = Classifier().classify("  Open Chrome ")
    assert result.is_command() and result.text == "open chrome" and not result.mode_changed


def test_trigger_switches_to_dictation_and_strips_itself():
    clf = Classifier()
    result = clf.classify("Start dictation the meeting is at noon")
    assert result.is_dictation_start()
    assert result.text == "the meeting is at noon"
    assert clf.mode == "dictation"


def test_dictation_preserves_capitalisation_until_stopped():
    clf = Classifier()
    clf.classify("start dictation")
    kept = clf.classify("  Hello World.  ")
    assert (kept.event, kept.text) == ("dictation", "Hello World.")
    stop = clf.classify("stop dictation")
    assert stop.is_command() and stop.mode_changed and stop.text == ""
    assert clf.mode == "command"


@pytest.mark.parametrize("text, expected", [
    ("start dictation", True), ("start dictation, please", True),
    ("START DICTATION!", True), ("start dictations", False), ("restart dictation", False),
    ("", False),
])
def test_trigger_matching_is_prefix_on_word_boundaries(text, expected):
    assert Classifier().matches_dictation_trigger(text) is expected


def test_reset_and_force_modes():
    clf = Classifier()
    clf.force_dictation_mode()
    assert clf.mode == "dictation"
    clf.force_command_mode()
    assert clf.mode == "command"
    clf.force_dictation_mode()
    clf.reset()
    assert clf.mode == "command"


@pytest.mark.parametrize("text, expected", [
    ("open chrome", True), ("go to github", True), ("Scroll", True),
    ("opened the door", False), ("hello there", False),
])
def test_looks_like_command(text, expected):
    assert Classifier.looks_like_command(text) is expected


# ── parse_final ────────────────────────────────────────────────────────────────

def word(w, conf=0.95):
    return {"word": w, "conf": conf, "start": 0.0, "end": 0.1}


def test_parse_final_confidences():
    r = parse_final({"text": "open chrome", "result": [word("open", 0.9), word("chrome", 0.7)]},
                    duration_s=1.5)
    assert (r.text, r.is_final, r.event) == ("open chrome", True, "command")
    assert r.mean_confidence == pytest.approx(0.8)
    assert r.min_confidence == pytest.approx(0.7)
    assert (r.word_count, r.duration_s, r.unk_count, r.unk_ratio) == (2, 1.5, 0, 0.0)


def test_parse_final_excludes_unk_from_text_and_confidence():
    r = parse_final({"text": "open [unk] chrome",
                     "result": [word("open"), word("[unk]", 0.1), word("chrome")]})
    assert r.text == "open chrome"
    assert r.min_confidence == pytest.approx(0.95)
    assert r.unk_count == 1 and r.unk_ratio == pytest.approx(1 / 3)
    assert len(r.words) == 3                       # raw words kept for metadata


def test_parse_final_without_word_results_is_permissive():
    r = parse_final({"text": "open chrome"})
    assert (r.mean_confidence, r.min_confidence) == (1.0, 1.0)


def test_parse_final_empty():
    r = parse_final({"text": ""})
    assert (r.text, r.mean_confidence, r.min_confidence, r.word_count) == ("", 0.0, 0.0, 0)


def test_result_repr():
    assert "FINAL" in repr(parse_final({"text": "x"}))
    assert "partial" in repr(VoskResult(event="partial", text="x", is_final=False))


# ── Router ─────────────────────────────────────────────────────────────────────

def result(text, mean=0.95, lo=0.9, dur=1.0, words=None, unk=0):
    words = words if words is not None else [word(w) for w in text.split()]
    return VoskResult(event="command", text=text, is_final=True, words=words,
                      mean_confidence=mean, min_confidence=lo, duration_s=dur,
                      unk_count=unk, unk_ratio=(unk / len(words)) if words else 0.0)


@pytest.fixture
def router():
    return Router()


def test_confident_short_command_stays_on_fast_path(router):
    d = router.route(result("open chrome"))
    assert d.destination == "command" and not d.escalate
    assert d.text == "open chrome"


@pytest.mark.parametrize("case, reason_fragment", [
    (dict(text="open chrome", lo=ROUTING["min_word_confidence"] - 0.01), "weakest word"),
    (dict(text="open chrome", mean=ROUTING["min_mean_confidence"] - 0.01), "mean conf"),
    (dict(text=" ".join(["word"] * (ROUTING["max_command_words"] + 1))), "words >"),
    (dict(text="open chrome", dur=ROUTING["max_command_duration_s"] + 0.1), "s >"),
])
def test_escalation_reasons(router, case, reason_fragment):
    d = router.route(result(**case))
    assert d.escalate and reason_fragment in d.reason


def test_thresholds_are_inclusive(router):
    exact = result("open chrome", mean=ROUTING["min_mean_confidence"],
                   lo=ROUTING["min_word_confidence"],
                   dur=ROUTING["max_command_duration_s"])
    assert not router.route(exact).escalate
    longest = result(" ".join(["w"] * ROUTING["max_command_words"]))
    assert not router.route(longest).escalate


def test_explicit_trigger_forces_whisper_even_when_confident(router):
    trigger = sorted(DICTATION_TRIGGERS)[0]
    d = router.route(result(trigger + " hello"))
    assert d.escalate and d.reason == "explicit dictation trigger"


def test_empty_decode_escalates_by_default(router):
    assert router.route(result("", words=[])).escalate
    quiet = Router(escalate_on_empty=False).route(result("", words=[]))
    assert not quiet.escalate and quiet.text == ""


def test_oov_share_escalates(router):
    # "open <unknown app>": half the words OOV → Whisper.
    d = router.route(result("open", words=[word("open"), word("[unk]")], unk=1))
    assert d.escalate and "OOV" in d.reason


def test_one_stray_unk_is_tolerated(router):
    # A breath decoded as [unk] next to a two-word command: 1/3 < 0.34.
    words = [word("open"), word("chrome"), word("[unk]")]
    assert not router.route(result("open chrome", words=words, unk=1)).escalate


def test_custom_thresholds(router):
    lenient = Router(min_word_confidence=0.1, min_mean_confidence=0.1)
    assert not lenient.route(result("open chrome", mean=0.5, lo=0.4)).escalate


def test_decision_repr():
    assert "WHISPER" in repr(Decision("dictation", "why", "t"))
    assert "COMMAND" in repr(Decision("command", "why", "t"))
