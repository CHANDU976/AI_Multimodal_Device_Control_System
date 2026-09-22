"""Suite-wide setup.

Three guarantees hold for every test:

1. Isolation — user data (preferences.json, usage.json) goes to a throwaway
   directory, never into central_module/data.
2. Determinism — the intent engine runs on the rule ladder, not on whatever
   DistilBERT weights happen to be on this machine. Tests that exercise the
   neural model are marked `ml`/`models` and opt back in.
3. No real side effects — browsers, app launches, OS commands and input
   injection are replaced by guards that fail loudly if a test forgets to
   mock them. A test suite that can shut down the machine it runs on is
   not a test suite.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import webbrowser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Must be set before central_module.config is first imported: DATA_DIR and
# the default store paths are computed at import time.
_SESSION_DATA = tempfile.mkdtemp(prefix="mmctrl-test-data-")
os.environ["MMCTRL_DATA"] = _SESSION_DATA
# This repository tracks some __pycache__/*.pyc files, so a test run that
# wrote bytecode would show up as local modifications. Stop it here and in
# every child process the tests spawn.
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from helpers.fakes import install_fake_pynput  # noqa: E402  (after env setup)


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_SESSION_DATA, ignore_errors=True)


def pytest_report_header(config):
    return [f"project root: {ROOT}", f"test data dir: {_SESSION_DATA}"]


# ── Determinism ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _rule_based_nlp(request, monkeypatch):
    """Rule ladder by default; `ml`/`models` tests keep the configured engine."""
    if request.node.get_closest_marker("ml") or request.node.get_closest_marker("models"):
        return
    from central_module import config
    monkeypatch.setitem(config.NLP, "engine", "rules")


# ── No real side effects ───────────────────────────────────────────────────────

def _forbidden(what):
    def guard(*args, **kwargs):
        raise AssertionError(f"test attempted a real {what}: args={args!r} — mock it")
    return guard


@pytest.fixture(autouse=True)
def _no_real_side_effects(monkeypatch):
    monkeypatch.setattr(webbrowser, "open", _forbidden("browser launch"))
    monkeypatch.setattr(os, "startfile", _forbidden("os.startfile"), raising=False)

    from central_module.actions import os_adapter
    monkeypatch.setattr(os_adapter.OSAdapter, "_run",
                        staticmethod(_forbidden("OS helper command (OSAdapter._run)")))

    if sys.platform == "win32":
        from central_module.actions import windows_adapter
        monkeypatch.setattr(windows_adapter, "_send", _forbidden("SendInput"))
        monkeypatch.setattr(windows_adapter.user32, "LockWorkStation",
                            _forbidden("LockWorkStation"), raising=False)


@pytest.fixture(autouse=True)
def fake_input(monkeypatch):
    """pynput is always the recorder — even where the real package exists."""
    return install_fake_pynput(monkeypatch)


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def project_root() -> Path:
    return ROOT
