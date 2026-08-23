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
If — optionally — it also prints JSON lines on stdout, the central module
routes them, and gestures named in preferences "gesture_actions" get executed
through the central action pipeline:

    {"type": "gesture", "name": "thumbs_up", "confidence": 0.93}

Until then, this stub keeps the supervisor path testable: it answers pings,
echoes injected test gestures, and exits cleanly on shutdown or parent death.

    stdin:  {"op": "shutdown"} | {"op": "ping"} | {"op": "emit", "name": "..."}
"""

from __future__ import annotations

import json
import sys


def emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def main() -> int:
    emit({"type": "ready"})
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
            emit({"type": "pong"})
        elif op == "emit":              # test injection: pretend a gesture fired
            emit({"type": "gesture", "name": msg.get("name", "unknown"),
                  "confidence": float(msg.get("confidence", 1.0))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
