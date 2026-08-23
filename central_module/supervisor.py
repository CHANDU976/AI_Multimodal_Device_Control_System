"""
supervisor.py
Runs the submodules (speech, gesture) as child processes and keeps them alive.

Each child gets:
  * stdin   — JSON-lines control channel (ipc.send)
  * stdout  — JSON-lines event channel, pumped into the supervisor's on_event
  * stderr  — treated as the child's log stream and re-logged here

Crash policy: a child that exits while the supervisor is running is restarted
with exponential backoff (SUPERVISOR config). Staying alive for
healthy_uptime_s resets the crash counter, so a long-running child that
eventually hits a transient error still gets its full retry budget.
Lifecycle transitions are reported as synthetic events on the same channel:

    {"type": "lifecycle", "child": name, "state": "started"|"exited"|"restarting"|"gave_up", ...}
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from central_module import ipc
from central_module.config import SUPERVISOR

logger = logging.getLogger(__name__)

# (child_name, message) — every event the supervisor produces or relays.
EventCallback = Callable[[str, dict], None]


@dataclass
class ChildSpec:
    """Everything needed to launch and babysit one child process."""
    name: str
    argv: list[str]
    cwd: str | None = None
    grace_s: float = 8.0                    # wait after "shutdown" op before terminate()
    restart: bool = True
    max_restarts: int = SUPERVISOR["max_restarts"]
    backoff_base_s: float = SUPERVISOR["backoff_base_s"]
    backoff_cap_s: float = SUPERVISOR["backoff_cap_s"]
    env: dict = field(default_factory=dict)  # overlaid on os.environ


class ChildProcess:
    """One running child: process handle + reader threads + control channel."""

    def __init__(self, spec: ChildSpec, on_event: EventCallback) -> None:
        self.spec = spec
        self._on_event = on_event
        self._proc: subprocess.Popen | None = None
        self.started_at: float = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        import os
        env = {**os.environ, **self.spec.env} if self.spec.env else None
        logger.info("[%s] starting: %s", self.spec.name, " ".join(self.spec.argv))
        self._proc = subprocess.Popen(
            self.spec.argv,
            cwd=self.spec.cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.started_at = time.monotonic()
        ipc.LineReader(
            self._proc.stdout,
            on_message=lambda msg: self._on_event(self.spec.name, msg),
            name=f"{self.spec.name}-stdout",
        ).start()
        threading.Thread(
            target=self._pump_stderr, name=f"{self.spec.name}-stderr", daemon=True
        ).start()

    def _pump_stderr(self) -> None:
        """Child logs (stderr) → our logger, prefixed with the child name."""
        proc = self._proc
        if not proc or not proc.stderr:
            return
        try:
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    logger.info("[%s] %s", self.spec.name, line)
        except (OSError, ValueError):
            pass

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def returncode(self) -> int | None:
        return self._proc.poll() if self._proc else None

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self.started_at if self._proc else 0.0

    def send(self, obj: dict) -> bool:
        """Send a control message; False if the child or its pipe is gone."""
        if not self.alive or not self._proc.stdin:
            return False
        return ipc.send(self._proc.stdin, obj)

    def stop(self) -> None:
        """shutdown op → grace wait → terminate → kill."""
        if not self._proc:
            return
        if self.alive:
            self.send({"op": "shutdown"})
            try:
                self._proc.wait(timeout=self.spec.grace_s)
            except subprocess.TimeoutExpired:
                logger.warning("[%s] no graceful exit, terminating", self.spec.name)
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    logger.warning("[%s] terminate ignored, killing", self.spec.name)
                    self._proc.kill()
                    self._proc.wait()
        logger.info("[%s] stopped (rc=%s)", self.spec.name, self.returncode)


class ProcessSupervisor:
    """
    Owns all children. start_all() launches them and a monitor thread that
    restarts crashed children per their spec; stop_all() shuts everything down.
    """

    def __init__(self, on_event: EventCallback) -> None:
        self._on_event = on_event
        self._specs: dict[str, ChildSpec] = {}
        self._children: dict[str, ChildProcess] = {}
        self._restarts: dict[str, int] = {}
        self._restart_at: dict[str, float] = {}      # monotonic deadline
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._monitor: threading.Thread | None = None

    def add(self, spec: ChildSpec) -> None:
        self._specs[spec.name] = spec

    def send(self, name: str, obj: dict) -> bool:
        child = self._children.get(name)
        return child.send(obj) if child else False

    def child_alive(self, name: str) -> bool:
        child = self._children.get(name)
        return bool(child and child.alive)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start_all(self) -> None:
        self._running.set()
        for spec in self._specs.values():
            self._launch(spec)
        self._monitor = threading.Thread(
            target=self._monitor_loop, name="SupervisorMonitor", daemon=True
        )
        self._monitor.start()

    def stop_all(self) -> None:
        self._running.clear()
        if self._monitor:
            self._monitor.join(timeout=SUPERVISOR["poll_interval_s"] * 3)
        with self._lock:
            children = list(self._children.values())
        # Stop in parallel — each stop can block for its grace period.
        threads = [
            threading.Thread(target=c.stop, name=f"stop-{c.spec.name}")
            for c in children
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _launch(self, spec: ChildSpec) -> None:
        child = ChildProcess(spec, self._on_event)
        try:
            child.start()
        except OSError as exc:
            logger.error("[%s] failed to launch: %s", spec.name, exc)
            self._on_event(spec.name, {"type": "lifecycle", "child": spec.name,
                                       "state": "gave_up", "error": str(exc)})
            return
        with self._lock:
            self._children[spec.name] = child
        self._on_event(spec.name, {"type": "lifecycle", "child": spec.name,
                                   "state": "started"})

    # ── Monitor ────────────────────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        poll = SUPERVISOR["poll_interval_s"]
        while self._running.is_set():
            time.sleep(poll)
            now = time.monotonic()
            with self._lock:
                children = dict(self._children)

            for name, child in children.items():
                spec = child.spec

                # Pending scheduled restart?
                deadline = self._restart_at.get(name)
                if deadline is not None:
                    if now >= deadline and self._running.is_set():
                        del self._restart_at[name]
                        self._on_event(name, {"type": "lifecycle", "child": name,
                                              "state": "restarting",
                                              "attempt": self._restarts[name]})
                        self._launch(spec)
                    continue

                if child.alive:
                    if child.uptime_s > SUPERVISOR["healthy_uptime_s"]:
                        self._restarts[name] = 0
                    continue

                # Child exited.
                rc = child.returncode
                self._on_event(name, {"type": "lifecycle", "child": name,
                                      "state": "exited", "returncode": rc})
                if not (spec.restart and self._running.is_set()):
                    with self._lock:
                        self._children.pop(name, None)
                    continue
                count = self._restarts.get(name, 0) + 1
                self._restarts[name] = count
                if count > spec.max_restarts:
                    logger.error("[%s] crashed %d times — giving up", name, count - 1)
                    self._on_event(name, {"type": "lifecycle", "child": name,
                                          "state": "gave_up", "returncode": rc})
                    with self._lock:
                        self._children.pop(name, None)
                    continue
                delay = min(spec.backoff_base_s * 2 ** (count - 1), spec.backoff_cap_s)
                logger.warning("[%s] exited rc=%s — restart %d/%d in %.1fs",
                               name, rc, count, spec.max_restarts, delay)
                self._restart_at[name] = now + delay


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    def show(name: str, msg: dict) -> None:
        print(f"  [{name}] {msg}")

    sup = ProcessSupervisor(on_event=show)
    # A child that prints one event then dies — watch the backoff kick in.
    sup.add(ChildSpec(
        name="flaky",
        argv=[sys.executable, "-c",
              'import json; print(json.dumps({"type": "ready"}), flush=True)'],
        max_restarts=2, backoff_base_s=1.0,
    ))
    sup.start_all()
    try:
        time.sleep(8)
    finally:
        sup.stop_all()
    print("supervisor smoke-test done.")
