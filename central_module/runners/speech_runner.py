"""
speech_runner.py
Child-process wrapper around speech_module.SpeechManager — the "subprocess
protocol" that speech_manager.py's docstrings anticipate.

    stdout (JSON lines, child → central):
        {"type": "ready"}
        {"type": "partial"|"command"|"dictation"|"device"|"error",
         "text": ..., "source": ..., "metadata": {...}}
        {"type": "devices", "devices": [...]}          reply to get_devices
        {"type": "pong"}                               reply to ping
        {"type": "fatal", "error": ...}                unrecoverable, then exit 1

    stdin (JSON lines, central → child):
        {"op": "set_language", "lang": "en-in"|"en-us"|"hi-in"}
        {"op": "set_input_device", "index": <int|null>}
        {"op": "get_devices"}
        {"op": "ping"}
        {"op": "shutdown"}

Protocol hygiene: fd 1 is duplicated for the protocol and then redirected to
stderr, so stray prints — ours or a native library's — land in the log stream
instead of corrupting the event stream. EOF on stdin means the parent died;
the runner shuts down rather than run an orphaned microphone.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading

# ── Protocol channel first, before anything can print ──────────────────────────
_proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8", newline="\n")
os.dup2(2, 1)                  # fd 1 now feeds the parent's log pump
sys.stdout = sys.stderr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("speech_runner")

_lock = threading.Lock()


def emit(obj: dict) -> None:
    with _lock:
        try:
            _proto.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
            _proto.flush()
        except OSError:            # parent gone — nothing left to talk to
            os._exit(0)


def main() -> int:
    # speech_module uses flat sibling imports (path_setup style), so both the
    # project root and the module directory must be importable.
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    speech_dir = os.path.join(project_root, "speech_module")
    for p in (project_root, speech_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from speech_manager import SpeechManager, SpeechEvent  # noqa: E402
    except Exception as exc:
        logger.exception("cannot import speech module")
        emit({"type": "fatal", "error": f"speech import failed: {exc}"})
        return 1

    stop = threading.Event()

    def on_event(event: SpeechEvent) -> None:
        # SpeechEvent.to_ipc() is the fast path — it already returns the
        # exact JSON-serialisable object for the wire protocol. Fall back to
        # the original manual shape for anything that doesn't have it (e.g.
        # a test double that only implements event_type/text/source/
        # metadata), so this boundary can't crash the runner over an
        # interface mismatch, and callers that never asked for the richer
        # fields don't get an unrequested timestamp/model/confidence added
        # to their wire shape.
        to_ipc = getattr(event, "to_ipc", None)
        if callable(to_ipc):
            emit(to_ipc())
            return
        emit({
            "type": getattr(event, "event_type", None),
            "text": getattr(event, "text", ""),
            "source": getattr(event, "source", ""),
            "metadata": getattr(event, "metadata", {}) or {},
        })

    try:
        manager = SpeechManager(callback=on_event)
        manager.start()
    except Exception as exc:
        logger.exception("SpeechManager failed to start")
        emit({"type": "fatal", "error": f"start failed: {exc}"})
        return 1

    emit({"type": "ready"})

    def control_loop() -> None:
        """stdin ops; EOF (parent died or closed us) → shutdown."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                op = msg.get("op")
            except (json.JSONDecodeError, AttributeError):
                logger.warning("bad control line: %r", line)
                continue
            try:
                if op == "shutdown":
                    break
                elif op == "ping":
                    emit({"type": "pong"})
                elif op == "set_language":
                    manager.set_language(msg["lang"])
                elif op == "set_input_device":
                    manager.set_input_device(msg.get("index"))
                elif op == "get_devices":
                    emit({"type": "devices", "devices": manager.input_devices})
                else:
                    logger.warning("unknown op: %r", op)
            except Exception as exc:
                logger.exception("control op %r failed", op)
                emit({"type": "error", "text": "", "source": "runner",
                      "metadata": {"op": op, "error": str(exc)}})
        stop.set()

    threading.Thread(target=control_loop, name="ControlLoop", daemon=True).start()

    try:
        stop.wait()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("shutting down …")
        manager.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())