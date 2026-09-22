"""actions/executor.py — IntentMatch → adapter primitive, plus the learning hooks."""

import pytest

from central_module.actions import executor as executor_mod
from central_module.actions.executor import ActionExecutor, ActionResult
from central_module.actions.os_adapter import ActionError
from central_module.learning.adaptation import BehaviorModel
from central_module.learning.preferences import PreferenceStore
from central_module.nlp.intent_engine import IntentMatch
from central_module.nlp.intents import BY_NAME, CATALOG
from helpers.fakes import recording_adapter_class

RecordingAdapter = recording_adapter_class()


@pytest.fixture
def stores(tmp_path):
    return (PreferenceStore(str(tmp_path / "prefs.json")),
            BehaviorModel(str(tmp_path / "usage.json")))


@pytest.fixture
def run(stores):
    """run(name, slots) → (ActionResult, adapter)."""
    prefs, behavior = stores

    def _run(name, slots=None, *, confirmed=False, dry_run=False, adapter=None):
        adapter = adapter or RecordingAdapter()
        ex = ActionExecutor(adapter, prefs, behavior, dry_run=dry_run)
        return ex.execute(IntentMatch(BY_NAME[name], slots or {}), confirmed=confirmed), adapter

    return _run


@pytest.fixture
def browser(monkeypatch):
    opened = []
    monkeypatch.setattr(executor_mod.webbrowser, "open", opened.append)
    return opened


# ── Dispatch table: every executor intent reaches exactly one primitive ────────

DISPATCH_CASES = [
    ("open_app", {"app": "browser"}, [("open_app", "browser-cmd")]),
    ("type_text", {"text": "hello there"}, [("type_text", "hello there")]),
    ("press_key", {"key": "enter"}, [("key_tap", "enter")]),
    ("copy", {}, [("hotkey", ("ctrl", "c"))]),
    ("paste", {}, [("hotkey", ("ctrl", "v"))]),
    ("cut", {}, [("hotkey", ("ctrl", "x"))]),
    ("select_all", {}, [("hotkey", ("ctrl", "a"))]),
    ("undo", {}, [("hotkey", ("ctrl", "z"))]),
    ("redo", {}, [("hotkey", ("ctrl", "y"))]),
    ("save", {}, [("hotkey", ("ctrl", "s"))]),
    ("delete", {}, [("key_tap", "delete")]),
    ("scroll", {"direction": "down"}, [("scroll", "down", 3)]),
    ("scroll", {"direction": "up", "count": 7}, [("scroll", "up", 7)]),
    ("click", {}, [("click", "left", False)]),
    ("double_click", {}, [("click", "left", True)]),
    ("right_click", {}, [("click", "right", False)]),
    ("volume_up", {}, [("volume_change", 2)]),
    ("volume_down", {}, [("volume_change", -2)]),
    ("volume_up", {"count": 10}, [("volume_change", 5)]),    # percent → key presses
    ("volume_down", {"count": 1}, [("volume_change", -1)]),  # never rounds to zero
    ("mute", {}, [("volume_mute", True)]),
    ("unmute", {}, [("volume_mute", False)]),
    ("set_volume", {"number": 40}, [("set_volume", 40)]),
    ("play_pause", {}, [("media", "play_pause")]),
    ("next_track", {}, [("media", "next")]),
    ("prev_track", {}, [("media", "prev")]),
    ("minimize", {}, [("window", "minimize")]),
    ("maximize", {}, [("window", "maximize")]),
    ("close_window", {}, [("window", "close")]),
    ("switch_window", {}, [("window", "switch")]),
    ("show_desktop", {}, [("window", "show_desktop")]),
    ("new_tab", {}, [("hotkey", ("ctrl", "t"))]),
    ("close_tab", {}, [("hotkey", ("ctrl", "w"))]),
    ("next_tab", {}, [("hotkey", ("ctrl", "tab"))]),
    ("prev_tab", {}, [("hotkey", ("ctrl", "shift", "tab"))]),
    ("browser_back", {}, [("hotkey", ("alt", "left"))]),
    ("browser_forward", {}, [("hotkey", ("alt", "right"))]),
    ("refresh", {}, [("hotkey", ("f5",))]),
    ("screenshot", {}, [("screenshot",)]),
    ("lock", {}, [("lock",)]),
    ("brightness_up", {}, [("brightness", 15)]),
    ("brightness_down", {}, [("brightness", -15)]),
    ("sleep", {}, [("power", "sleep", 3)]),
]


@pytest.mark.parametrize("name, slots, expected", DISPATCH_CASES)
def test_dispatch(run, name, slots, expected):
    result, adapter = run(name, slots)
    assert result.ok, result.message
    assert not result.needs_confirm
    assert adapter.calls == expected
    assert result.message


def test_every_executor_intent_is_covered_by_dispatch_table():
    covered = {case[0] for case in DISPATCH_CASES} | \
        {"search_web", "open_url", "shutdown", "restart"}   # tested separately below
    executor_intents = {i.name for i in CATALOG if i.action != "manager"}
    assert executor_intents <= covered, executor_intents - covered


# ── Confirmation gate ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["shutdown", "restart"])
def test_destructive_intents_arm_before_firing(run, name):
    pending, adapter = run(name)
    assert pending.ok and pending.needs_confirm
    assert "yes" in pending.message
    assert adapter.calls == []

    confirmed, adapter = run(name, confirmed=True)
    assert confirmed.ok and not confirmed.needs_confirm
    assert adapter.calls == [("power", name, 3)]
    assert confirmed.message == f"{name} in 3s"


def test_confirmation_gate_applies_even_in_dry_run(run):
    result, adapter = run("shutdown", dry_run=True)
    assert result.needs_confirm and adapter.calls == []


# ── Dry run ────────────────────────────────────────────────────────────────────

def test_dry_run_never_touches_the_adapter(run):
    result, adapter = run("type_text", {"text": "hello"}, dry_run=True)
    assert result.ok and result.message.startswith("[dry-run] type_text")
    assert adapter.calls == []


def test_dry_run_still_counts_app_usage(run, stores):
    run("open_app", {"app": "notepad"}, dry_run=True)
    assert stores[1]._data["app_counts"] == {"notepad": 1}


def test_dry_run_with_no_adapter_at_all(stores):
    ex = ActionExecutor(None, *stores, dry_run=True)
    assert ex.execute(IntentMatch(BY_NAME["copy"], {})).ok


# ── Failure handling ───────────────────────────────────────────────────────────

def test_action_error_becomes_a_readable_failure(run):
    class Broken(RecordingAdapter):
        def lock(self):
            raise ActionError("LockWorkStation failed")

    result, _ = run("lock", adapter=Broken())
    assert result == ActionResult(False, "LockWorkStation failed")


def test_unexpected_exception_is_contained(run):
    class Crashy(RecordingAdapter):
        def lock(self):
            raise ZeroDivisionError

    result, _ = run("lock", adapter=Crashy())
    assert not result.ok
    assert "unexpectedly" in result.message


def test_unknown_action_has_no_handler(stores):
    from central_module.nlp.intents import Intent
    ex = ActionExecutor(RecordingAdapter(), *stores)
    result = ex.execute(IntentMatch(Intent("fly", "teleport", ("fly",)), {}))
    assert not result.ok and "no handler" in result.message


# ── App resolution & learning ──────────────────────────────────────────────────

def test_user_alias_beats_builtin_table(run, stores):
    stores[0].set("app_aliases.browser", "my-browser")
    result, adapter = run("open_app", {"app": "browser"})
    assert adapter.calls == [("open_app", "my-browser")]
    assert result.message == "opened browser"


def test_path_discovery_is_learned_as_alias(run, stores, monkeypatch):
    monkeypatch.setattr(executor_mod.shutil, "which",
                        lambda name: "/usr/bin/gimp" if name == "gimp" else None)
    run("open_app", {"app": "gimp"})
    assert stores[0].get_app_alias("gimp") == "/usr/bin/gimp"
    assert stores[1]._data["app_counts"] == {"gimp": 1}


def test_multi_word_app_tries_joined_name_on_path(run, monkeypatch):
    monkeypatch.setattr(executor_mod.shutil, "which",
                        lambda name: "/opt/obs" if name == "obsstudio" else None)
    _, adapter = run("open_app", {"app": "obs studio"})
    assert adapter.calls == [("open_app", "/opt/obs")]


def test_unresolvable_app_is_passed_raw_to_the_os(run, stores, monkeypatch):
    monkeypatch.setattr(executor_mod.shutil, "which", lambda name: None)
    _, adapter = run("open_app", {"app": "report.docx"})
    assert adapter.calls == [("open_app", "report.docx")]
    assert stores[0].get_app_alias("report.docx") is None     # raw is not learned


# ── Web ────────────────────────────────────────────────────────────────────────

def test_search_web_quotes_the_query(run, browser):
    result, _ = run("search_web", {"text": "weather in bangalore & more"})
    assert browser == ["https://www.google.com/search?q=weather+in+bangalore+%26+more"]
    assert result.message == "searching for weather in bangalore & more"


@pytest.mark.parametrize("spoken, url", [
    ("github dot com", "https://github.com"),
    ("youtube", "https://youtube.com"),
    ("wikipedia dot org", "https://wikipedia.org"),
    ("https://example.org", "https://example.org"),     # already a URL: untouched
    ("college portal", "https://collegeportal.com"),
])
def test_open_url_builds_address_from_speech(run, browser, spoken, url):
    run("open_url", {"text": spoken})
    assert browser == [url]


def test_typing_reports_word_count(run):
    result, _ = run("type_text", {"text": "one two three"})
    assert result.message == "typed 3 words"
