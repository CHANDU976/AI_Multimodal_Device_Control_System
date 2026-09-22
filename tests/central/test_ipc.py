"""ipc.py — the JSON-lines protocol between the central module and its children."""

import io
import json
import threading

import pytest

from central_module import ipc


def test_send_writes_one_json_line():
    buf = io.StringIO()
    assert ipc.send(buf, {"op": "set_language", "lang": "hi-in"}) is True
    line = buf.getvalue()
    assert line.endswith("\n") and line.count("\n") == 1
    assert json.loads(line) == {"op": "set_language", "lang": "hi-in"}


def test_send_keeps_unicode_readable():
    buf = io.StringIO()
    ipc.send(buf, {"text": "नमस्ते"})
    assert "नमस्ते" in buf.getvalue()


def test_send_stringifies_unserialisable_values():
    buf = io.StringIO()
    ipc.send(buf, {"path": object.__new__(type("P", (), {"__str__": lambda s: "P!"}))})
    assert json.loads(buf.getvalue()) == {"path": "P!"}


def test_send_to_closed_pipe_returns_false():
    buf = io.StringIO()
    buf.close()
    assert ipc.send(buf, {"op": "ping"}) is False


class _BrokenPipe(io.StringIO):
    def write(self, s):
        raise BrokenPipeError


def test_send_to_broken_pipe_returns_false():
    assert ipc.send(_BrokenPipe(), {"op": "ping"}) is False


@pytest.mark.parametrize("line, expected", [
    ('{"type": "ready"}\n', {"type": "ready"}),
    ('  {"type": "command", "text": "open chrome"}  ', {"type": "command", "text": "open chrome"}),
    ("", {}),
    ("   \n", {}),
    ("LOG (VoskAPI) decoding graph", {"type": "log", "line": "LOG (VoskAPI) decoding graph"}),
    ("[1, 2, 3]", {"type": "log", "line": "[1, 2, 3]"}),        # JSON, but not an object
    ('{"type": ', {"type": "log", "line": '{"type":'}),          # truncated JSON
])
def test_parse_line(line, expected):
    assert ipc.parse_line(line) == expected


def _read_all(text, on_message=None):
    got, eof = [], threading.Event()
    reader = ipc.LineReader(io.StringIO(text), on_message or got.append, eof.set, name="t")
    reader.start()
    reader.join(timeout=5)
    assert not reader.is_alive() and eof.is_set()
    return got


def test_line_reader_pumps_messages_then_signals_eof():
    got = _read_all('{"type": "ready"}\n\nnoise\n{"type": "pong"}\n')
    assert got == [{"type": "ready"}, {"type": "log", "line": "noise"}, {"type": "pong"}]


def test_line_reader_survives_a_failing_handler():
    seen = []

    def handler(msg):
        seen.append(msg)
        if msg["type"] == "boom":
            raise RuntimeError("handler bug")

    _read_all('{"type": "boom"}\n{"type": "after"}\n', handler)
    assert [m["type"] for m in seen] == ["boom", "after"]


def test_line_reader_tolerates_stream_teardown():
    class Torn:
        def __iter__(self):
            yield '{"type": "ready"}\n'
            raise ValueError("I/O operation on closed file")

    got, eof = [], threading.Event()
    reader = ipc.LineReader(Torn(), got.append, eof.set)
    reader.start()
    reader.join(timeout=5)
    assert got == [{"type": "ready"}] and eof.is_set()


def test_line_reader_eof_handler_errors_are_contained():
    def bad_eof():
        raise RuntimeError("eof bug")

    reader = ipc.LineReader(io.StringIO(""), lambda m: None, bad_eof)
    reader.start()
    reader.join(timeout=5)
    assert not reader.is_alive()


def test_line_reader_is_a_daemon():
    assert ipc.LineReader(io.StringIO(""), lambda m: None).daemon
