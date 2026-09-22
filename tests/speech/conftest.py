"""Speech-module test setup.

speech_module files use flat sibling imports (`from config import ...`), so
the directory is on sys.path via pytest's `pythonpath` setting. vosk is a
compiled dependency imported at module scope by vosk_engine.py; where the
wheel is unavailable a minimal stand-in keeps the pure-Python logic testable
(tests that drive the recogniser always swap in helpers.fakes.FakeVosk).
"""

import pytest

from helpers.fakes import FakeVosk, ensure_importable, vosk_stub_module

ensure_importable("vosk", vosk_stub_module)


@pytest.fixture
def fake_vosk(monkeypatch):
    import vosk_engine
    fake = FakeVosk()
    monkeypatch.setattr(vosk_engine, "vosk", fake)
    return fake
