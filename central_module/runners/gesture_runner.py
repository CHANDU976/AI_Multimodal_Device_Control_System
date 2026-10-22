"""
gesture_runner.py
Stand-in child for the gesture module, which is an independent app with its
own UI and is not part of this repository.

To supervise the real app, set preferences.json:

    "gesture": {
        "enabled": true,
        "command": ["C:/path/to/python.exe", "C:/path/to/gesture_app/main.py"],
        "cwd": "C:/path/to/gesture_app"
    }

The gesture app keeps executing its own mapped actions as it does today; the
central module simply owns its lifecycle (start, restart on crash, shutdown).
If — optionally — it also prints protocol lines on stdout (docs/ipc-protocol.md,
built by Mousegesture/gesture_events.py), the central module routes them, and
gestures named in preferences "gesture_actions" get executed through the
central action pipeline:

    {"type": "gesture_fired", "module": "gesture", "timestamp": ...,
     "version": 1, "payload": {"gesture": "thumbs_up", "confidence": 0.93, ...}}

Until then, this stub keeps the supervisor path testable: it answers pings,
echoes injected test gestures, and exits cleanly on shutdown or parent death.
Its messages come from the real gesture_events builder, so they match the
gesture module exactly.

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
