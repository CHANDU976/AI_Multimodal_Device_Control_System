"""
ipc.py
JSON-lines protocol shared by the central module and its child processes.

One JSON object per line, UTF-8, newline-terminated:

    child → parent (stdout):  {"type": "...", ...}      events
    parent → child (stdin):   {"op": "...", ...}        control messages

Children must keep stdout clean — logs go to stderr (speech_runner even
redirects fd 1 so native libraries cannot corrupt the stream). The reader is
tolerant anyway: a line that is not JSON becomes a {"type": "log"} event
instead of an error.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Callable, IO

logger = logging.getLogger(__name__)


def send(stream: IO[str], obj: dict) -> bool:
    """Write one JSON line. Returns False if the pipe is gone."""
    try:
        stream.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        return True
    except (OSError, ValueError):        # closed pipe / closed file
        return False


def parse_line(line: str) -> dict:
    """Decode one protocol line; non-JSON lines become log events."""
    line = line.strip()
    if not line:
        return {}
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return {"type": "log", "line": line}


class LineReader(threading.Thread):
    """
    Daemon thread that pumps JSON lines from a stream into a callback.
    Calls on_eof() (if given) when the stream closes — for a child's stdout
    that means the child exited or closed its end.
    """

    def __init__(
        self,
        stream: IO[str],
        on_message: Callable[[dict], None],
        on_eof: Callable[[], None] | None = None,
        name: str = "LineReader",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._stream = stream
        self._on_message = on_message
        self._on_eof = on_eof

    def run(self) -> None:
        try:
            for line in self._stream:
                msg = parse_line(line)
                if not msg:
                    continue
                try:
                    self._on_message(msg)
                except Exception:
                    logger.exception("IPC message handler raised")
        except (OSError, ValueError):
            pass                         # stream torn down mid-read
        if self._on_eof:
            try:
                self._on_eof()
            except Exception:
                logger.exception("IPC on_eof handler raised")
