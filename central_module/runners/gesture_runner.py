"""
gesture_runner.py
Child-process wrapper around the gesture module (Mousegesture/), speaking the
protocol in docs/ipc-protocol.md — the gesture counterpart of speech_runner.

Two ways to run, chosen by the central module from preferences "gesture.mode":

    gui (default)    the Gesture Studio app, exactly as the user knows it:
                     mapping editor, camera picker, live preview. Its engine
                     reports events through the app's `event_queue` hook,
                     which this runner fills in. Tracking starts when the user
                     connects a camera; closing the window ends the child
                     cleanly, and the supervisor leaves it closed.
    --headless       the tracking engine alone, no window, on --camera N
                     (preferences "gesture.camera"). It applies the settings
                     and mappings saved in gesture_config.json.

    stdout (JSON lines, child → central), every message in the shared envelope
    {"type", "module": "gesture", "timestamp", "version", "payload"}:
        the engine's own events (Mousegesture/IPC_EVENTS.md):
            ready, pointing_started/stopped, gesture_fired, drag_started/
            stopped, heartbeat, fatal (tracking_loop_failed)
        built here:
            pong                                reply to ping
            fatal  start_failed                 the module could not load
                   camera_failed                headless: camera would not open

    stdin (JSON lines, central → child):
        {"op": "ping"}
        {"op": "shutdown"}                      also: EOF (the parent died)

Protocol hygiene, as in speech_runner: fd 1 is kept for the protocol and then
pointed at stderr, so the engine's prints and native-library chatter go to the
log stream; control ops are read from a private copy of stdin, and stdin is
pointed at NUL. The engine reports events from its camera thread; they are queued
and written by one writer thread, so a slow parent never stalls tracking.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time

# ── Protocol channel first, before anything can print ──────────────────────────
_proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8", newline="\n")
os.dup2(2, 1)                  # fd 1 now feeds the parent's log pump
sys.stdout = sys.stderr

# ── Control channel: a private copy of stdin, and stdin itself → NUL ───────────
# The control thread blocks reading this pipe. On Windows, while a blocking
# read is pending on a pipe, every other use of that same pipe waits for it —
# including the start-up code of native libraries (numpy's DLLs query the
# standard input handle as they load) and of any child process we start,
# which inherits stdin. The parent rarely writes to us, so those waits never
# ended: the camera scan froze importing numpy, and audio device queries
# timed out. Reading a duplicate and pointing the real stdin at NUL keeps the
# pipe to ourselves. EOF on the duplicate still means the parent died.
_control = os.fdopen(os.dup(0), "r", encoding="utf-8")
_nul = os.open(os.devnull, os.O_RDONLY)
os.dup2(_nul, 0)               # also replaces the Windows standard input handle
os.close(_nul)
sys.stdin = open(os.devnull, encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("gesture_runner")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GESTURE_DIR = os.path.join(PROJECT_ROOT, "Mousegesture")

# Events waiting for the writer. Bounded: if the parent stops reading, the
# oldest event is dropped rather than the camera thread blocking.
OUTBOX_MAX = 1000
_outbox: queue.Queue = queue.Queue(maxsize=OUTBOX_MAX)


def _write(obj: dict) -> None:
    try:
        _proto.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
        _proto.flush()
    except OSError:            # parent gone — nothing left to talk to
        os._exit(0)


def _writer() -> None:
    while True:
        obj = _outbox.get()
        if obj is None:
            return
        _write(obj)


def emit(obj: dict) -> None:
    """Queue one message. Never blocks; safe from any thread."""
    try:
        _outbox.put_nowait(obj)
    except queue.Full:
        try:
            _outbox.get_nowait()
        except queue.Empty:
            pass
        try:
            _outbox.put_nowait(obj)
        except queue.Full:
            pass


def flush_outbox() -> None:
    """Write everything queued, then stop the writer thread."""
    _outbox.put(None)
    _writer_thread.join(timeout=5)


class EventSink:
    """What the Gesture Studio app expects in its `event_queue` slot."""

    def put_nowait(self, message: dict) -> None:
        emit(message)


_writer_thread = threading.Thread(target=_writer, name="ProtocolWriter", daemon=True)


def control_loop(stop: threading.Event, events) -> None:
    """stdin ops. EOF (parent died or closed us) counts as shutdown."""
    for line in _control:
        line = line.strip()
        if not line:
            continue
        try:
            op = json.loads(line).get("op")
        except (json.JSONDecodeError, AttributeError):
            logger.warning("bad control line: %r", line)
            continue
        if op == "shutdown":
            break
        if op == "ping":
            emit(events.pong())
        else:
            logger.warning("unknown op: %r", op)
    stop.set()


def run_headless(camera: int, stop: threading.Event, events) -> int:
    import hand_cursor_2

    engine = hand_cursor_2.HandTrackerEngine(
        enable_monitor=False, verbose=True, event_sink=emit)
    try:
        # Loading the engine takes a while; don't open the camera if the
        # parent asked us to stop in the meantime.
        if stop.is_set():
            return 0
        if not engine.start(camera):
            reason = engine.status.get("error") or f"camera {camera} did not open"
            emit(events.fatal(code="camera_failed", message=reason))
            return 1
        # Run until the parent asks us to stop, or the tracking loop ends on
        # its own (a crash: the engine has already reported `fatal`).
        while not stop.wait(0.25):
            if not engine.running:
                break
        return 1 if engine.status.get("error") else 0
    finally:
        engine.close()                 # releases the camera and any held button


def run_gui(stop: threading.Event) -> int:
    import app

    studio = app.GestureStudio()
    studio.event_queue = EventSink()   # read when the app builds its engine

    def watch_for_stop() -> None:
        # Tk may only be touched from its own thread, so the control thread
        # just sets a flag and this poller acts on it.
        if stop.is_set():
            studio._on_close()         # may ask about unsaved mappings first
            return
        studio.after(200, watch_for_stop)

    studio.after(200, watch_for_stop)
    studio.mainloop()                  # returns when the window closes
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gesture_runner")
    parser.add_argument("--headless", action="store_true",
                        help="run the tracking engine without the Gesture Studio window")
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index for --headless (default 0)")
    args = parser.parse_args(argv)

    # Mousegesture uses flat sibling imports (`import gesture_fsm`).
    if GESTURE_DIR not in sys.path:
        sys.path.insert(0, GESTURE_DIR)
    _writer_thread.start()

    try:
        import gesture_events as events
    except Exception as exc:
        logger.exception("cannot import gesture_events")
        _write({"type": "fatal", "module": "gesture",
                "timestamp": round(time.time(), 3), "version": 1,
                "payload": {"code": "start_failed",
                            "message": f"gesture import failed: {exc}"[:200]}})
        return 1

    stop = threading.Event()
    threading.Thread(target=control_loop, args=(stop, events),
                     name="ControlLoop", daemon=True).start()
    try:
        if args.headless:
            code = run_headless(args.camera, stop, events)
        else:
            code = run_gui(stop)
    except Exception as exc:
        logger.exception("gesture module failed to start")
        emit(events.fatal(code="start_failed", message=f"{exc.__class__.__name__}: {exc}"))
        code = 1
    flush_outbox()
    return code


if __name__ == "__main__":
    sys.exit(main())
