"""supervisor.py — child processes, restart policy, graceful shutdown.

These run real child interpreters (the bundled gesture stub and tiny -c
scripts), with the restart timers shrunk to milliseconds.
"""

import logging
import sys
import threading
import time

import pytest

from central_module import config
from central_module.supervisor import ChildProcess, ChildSpec, ProcessSupervisor

pytestmark = pytest.mark.slow

STUB = [sys.executable, "-u", "-m", "central_module.runners.gesture_stub"]


class Events:
    """Thread-safe event sink with a wait helper."""

    def __init__(self):
        self._items = []
        self._cond = threading.Condition()

    def __call__(self, name, msg):
        with self._cond:
            self._items.append((name, msg))
            self._cond.notify_all()

    @property
    def items(self):
        with self._cond:
            return list(self._items)

    def states(self, name=None):
        return [m.get("state") for n, m in self.items
                if m.get("type") == "lifecycle" and (name is None or n == name)]

    def wait_for(self, predicate, timeout=15.0):
        deadline = time.monotonic() + timeout
        with self._cond:
            while not predicate(list(self._items)):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(f"timed out; events so far: {self._items}")
                self._cond.wait(remaining)


def has(kind, **fields):
    """A message of this type whose fields match, top level or in the payload."""
    def matches(m):
        body = {**m, **(m.get("payload") or {})}
        return m.get("type") == kind and all(body.get(k) == v for k, v in fields.items())

    def pred(items):
        return any(matches(m) for _, m in items)
    return pred


@pytest.fixture
def fast_policy(monkeypatch):
    monkeypatch.setitem(config.SUPERVISOR, "poll_interval_s", 0.05)
    monkeypatch.setitem(config.SUPERVISOR, "healthy_uptime_s", 60.0)


@pytest.fixture
def supervisor(fast_policy):
    events = Events()
    sup = ProcessSupervisor(on_event=events)
    sup.events = events
    yield sup
    sup.stop_all()


def test_child_lifecycle_and_control_channel(supervisor):
    supervisor.add(ChildSpec("gesture", STUB, cwd=str(config.PROJECT_ROOT), grace_s=5))
    supervisor.start_all()
    ev = supervisor.events
    ev.wait_for(has("ready"))
    assert supervisor.child_alive("gesture")

    assert supervisor.send("gesture", {"op": "ping"})
    ev.wait_for(has("pong"))
    supervisor.send("gesture", {"op": "emit", "name": "fist", "confidence": 0.9})
    ev.wait_for(has("gesture_fired", gesture="fist"))

    supervisor.stop_all()
    assert not supervisor.child_alive("gesture")
    assert ev.states("gesture")[0] == "started"
    # A graceful stop is not a crash: no restart, no give-up.
    assert "restarting" not in ev.states() and "gave_up" not in ev.states()


def test_send_to_unknown_or_dead_child_is_false(supervisor):
    assert supervisor.send("nobody", {"op": "ping"}) is False
    assert supervisor.child_alive("nobody") is False


def test_crashing_child_is_restarted_with_backoff_then_abandoned(supervisor):
    supervisor.add(ChildSpec("flaky", [sys.executable, "-c", "import sys; sys.exit(3)"],
                             max_restarts=2, backoff_base_s=0.05, backoff_cap_s=0.1))
    supervisor.start_all()
    ev = supervisor.events
    ev.wait_for(has("lifecycle", state="gave_up"))
    states = ev.states("flaky")
    assert states.count("started") == 3             # first run + 2 restarts
    assert states.count("restarting") == 2
    assert states[-1] == "gave_up"
    exits = [m for _, m in ev.items if m.get("state") == "exited"]
    assert all(m["returncode"] == 3 for m in exits)
    attempts = [m["attempt"] for _, m in ev.items if m.get("state") == "restarting"]
    assert attempts == [1, 2]


def test_clean_exit_can_mean_stay_down(supervisor):
    # A window the user closed exits 0: not a crash, so it is not reopened.
    supervisor.add(ChildSpec("window", [sys.executable, "-c", "pass"],
                             restart_on_clean_exit=False, backoff_base_s=0.05))
    supervisor.start_all()
    supervisor.events.wait_for(has("lifecycle", state="exited"))
    time.sleep(0.3)
    assert "restarting" not in supervisor.events.states()
    assert supervisor.child_alive("window") is False


def test_crash_is_restarted_even_when_clean_exits_stay_down(supervisor):
    supervisor.add(ChildSpec("window", [sys.executable, "-c", "import sys; sys.exit(2)"],
                             restart_on_clean_exit=False, max_restarts=1,
                             backoff_base_s=0.05, backoff_cap_s=0.1))
    supervisor.start_all()
    supervisor.events.wait_for(has("lifecycle", state="gave_up"))
    assert supervisor.events.states("window").count("restarting") == 1


def test_restart_disabled_child_is_just_reported(supervisor):
    supervisor.add(ChildSpec("once", [sys.executable, "-c", "pass"], restart=False))
    supervisor.start_all()
    supervisor.events.wait_for(has("lifecycle", state="exited"))
    time.sleep(0.3)
    assert "restarting" not in supervisor.events.states()


def test_healthy_uptime_resets_the_crash_budget(supervisor, monkeypatch):
    monkeypatch.setitem(config.SUPERVISOR, "healthy_uptime_s", 0.1)
    code = "import time, sys; time.sleep(0.5); sys.exit(1)"
    supervisor.add(ChildSpec("longlived", [sys.executable, "-c", code],
                             max_restarts=1, backoff_base_s=0.05))
    supervisor.start_all()
    supervisor.events.wait_for(lambda items: sum(
        1 for _, m in items if m.get("state") == "restarting") >= 2, timeout=20)
    assert "gave_up" not in supervisor.events.states()


def test_launch_failure_reports_gave_up(supervisor):
    supervisor.add(ChildSpec("ghost", ["definitely-not-an-executable-xyz"]))
    supervisor.start_all()
    msg = next(m for _, m in supervisor.events.items if m.get("state") == "gave_up")
    assert "error" in msg
    assert not supervisor.child_alive("ghost")


def test_child_ignoring_shutdown_is_terminated_after_grace():
    code = "import time\nwhile True: time.sleep(0.1)"   # never reads stdin
    child = ChildProcess(ChildSpec("stubborn", [sys.executable, "-c", code], grace_s=0.3),
                         on_event=lambda n, m: None)
    child.start()
    assert child.alive
    t0 = time.monotonic()
    child.stop()
    assert not child.alive
    assert time.monotonic() - t0 < 10
    assert child.returncode is not None


def test_child_stderr_is_relogged(caplog):
    code = "import sys; sys.stderr.write('model loaded\\n'); sys.stderr.flush()"
    child = ChildProcess(ChildSpec("talker", [sys.executable, "-c", code]),
                         on_event=lambda n, m: None)
    with caplog.at_level(logging.INFO, logger="central_module.supervisor"):
        child.start()
        deadline = time.monotonic() + 10
        while "model loaded" not in caplog.text and time.monotonic() < deadline:
            time.sleep(0.05)
        child.stop()
    assert "[talker] model loaded" in caplog.text


def test_child_env_overlay(tmp_path):
    out = tmp_path / "env.txt"
    code = f"import os; open(r'{out}', 'w').write(os.environ['MMCTRL_PROBE'])"
    child = ChildProcess(ChildSpec("env", [sys.executable, "-c", code],
                                   env={"MMCTRL_PROBE": "42"}),
                         on_event=lambda n, m: None)
    child.start()
    child._proc.wait(timeout=10)
    assert out.read_text() == "42"


def test_unstarted_child_reports_nothing():
    child = ChildProcess(ChildSpec("idle", ["x"]), on_event=lambda n, m: None)
    assert (child.alive, child.returncode, child.uptime_s) == (False, None, 0.0)
    assert child.send({"op": "ping"}) is False
    child.stop()                                        # no-op, no error
