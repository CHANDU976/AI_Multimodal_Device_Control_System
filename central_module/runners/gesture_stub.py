"""
gesture_stub.py
Stand-in child for the gesture module: no camera, no model, no GUI. The real
child is gesture_runner.py; this one keeps the supervisor path testable and is
what `python -m central_module --gesture-stub` runs.

It answers pings, echoes injected test gestures, and exits cleanly on
shutdown or parent death. Its messages come from the real gesture_events
builder, so they match the gesture module exactly (docs/ipc-protocol.md).

    stdin:  {"op": "shutdown"} | {"op": "ping"} | {"op": "emit", "name": "..."}
"""

from __future__ import annotations

import json
import os
import sys

# gesture_events is pure standard library; Mousegesture uses flat imports.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "Mousegesture"))
import gesture_events  # noqa: E402


def emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def main() -> int:
    emit(gesture_events.ready(semantic="disabled"))
    for line in sys.stdin:              # EOF = parent died → exit
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        op = msg.get("op")
        if op == "shutdown":
            break
        elif op == "ping":
            emit(gesture_events.pong())
        elif op == "emit":              # test injection: pretend a gesture fired
            emit(gesture_events.gesture_fired(
                gesture=msg.get("name", "unknown"), action=None,
                confidence=float(msg.get("confidence", 1.0)), executed=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
