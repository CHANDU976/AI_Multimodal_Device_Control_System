"""
audio_devices.py
Input-device inventory and default-device change detection.

Why a subprocess
----------------
PortAudio enumerates audio devices once, at initialisation, and never again —
a process holding an open stream cannot see a new default microphone, and
re-initialising PortAudio underneath an open stream is not safe. Each poll
therefore runs a tiny child interpreter that initialises a *fresh* PortAudio,
prints the device list as JSON and exits. That also isolates the poll from
the device-enumeration hangs some USB headsets cause (see audio_stream.py):
a hung child is killed by the timeout; the session's audio is untouched.

Devices are identified by (name, hostapi) rather than index — indexes shift
whenever a device appears or disappears.

Person A — Real-Time Engine + Audio Pipeline
"""

from __future__ import annotations
import path_setup  # noqa: F401 — adds project root to sys.path

import json
import logging
import subprocess
import sys
import threading
from typing import Callable

from config import DEVICE_MONITOR

logger = logging.getLogger(__name__)

# Runs in a fresh interpreter: fresh PortAudio, current device world.
_QUERY_SNIPPET = r"""
import json, sys
import sounddevice as sd
apis = sd.query_hostapis()
default_idx = sd.default.device[0]
out = []
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        out.append({
            "index": i,
            "name": d["name"],
            "hostapi": apis[d["hostapi"]]["name"],
            "channels": d["max_input_channels"],
            "default_samplerate": d["default_samplerate"],
            "is_default": i == default_idx,
        })
json.dump(out, sys.stdout)
"""


def identity(device: dict) -> tuple[str, str]:
    """Stable identity of a device across re-enumerations."""
    return (device["name"], device["hostapi"])


def snapshot(timeout_s: float = DEVICE_MONITOR["query_timeout_s"]) -> list[dict] | None:
    """
    Fresh list of input devices (one dict per device, `is_default` marking the
    system default), or None if the query failed or timed out.
    """
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _QUERY_SNIPPET],
            capture_output=True, text=True, timeout=timeout_s, **kwargs,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Device query timed out after %.0fs.", timeout_s)
        return None
    if proc.returncode != 0:
        logger.warning("Device query failed: %s", proc.stderr.strip()[-200:])
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        logger.warning("Device query returned unparsable output.")
        return None


def default_of(devices: list[dict] | None) -> dict | None:
    return next((d for d in devices or [] if d["is_default"]), None)


class DeviceMonitor:
    """
    Background thread that keeps a current snapshot of input devices and
    fires callbacks when the system default input or the device set changes.

    Both callbacks run on the monitor thread:
        on_default_change(new_default: dict, devices: list[dict])
        on_list_change(devices: list[dict])
    """

    def __init__(
        self,
        on_default_change: Callable[[dict, list[dict]], None] | None = None,
        on_list_change: Callable[[list[dict]], None] | None = None,
        poll_interval_s: float = DEVICE_MONITOR["poll_interval_s"],
    ) -> None:
        self.on_default_change = on_default_change
        self.on_list_change = on_list_change
        self.poll_interval_s = poll_interval_s
        self.devices: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def default_device(self) -> dict | None:
        return default_of(self.devices)

    def start(self) -> None:
        self.devices = snapshot() or []
        d = self.default_device
        logger.info(
            "DeviceMonitor: %d input device(s), default: %s",
            len(self.devices), f"{d['name']} [{d['hostapi']}]" if d else "none",
        )
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="DeviceMonitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=DEVICE_MONITOR["query_timeout_s"] + 2)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            current = snapshot()
            if current is None:          # transient failure — keep last state
                continue
            self._diff(self.devices, current)
            self.devices = current
        logger.debug("DeviceMonitor exited.")

    def _diff(self, old: list[dict], new: list[dict]) -> None:
        old_ids = {identity(d) for d in old}
        new_ids = {identity(d) for d in new}
        if old_ids != new_ids and self.on_list_change:
            for name, api in sorted(new_ids - old_ids):
                logger.info("Input device connected: %s [%s]", name, api)
            for name, api in sorted(old_ids - new_ids):
                logger.info("Input device removed: %s [%s]", name, api)
            self._safely(self.on_list_change, new)

        old_default, new_default = default_of(old), default_of(new)
        if new_default is None:
            return
        if old_default is None or identity(old_default) != identity(new_default):
            logger.info("Default input changed: %s → %s [%s]",
                        old_default["name"] if old_default else "none",
                        new_default["name"], new_default["hostapi"])
            if self.on_default_change:
                self._safely(self.on_default_change, new_default, new)

    @staticmethod
    def _safely(fn, *args) -> None:
        try:
            fn(*args)
        except Exception:
            logger.exception("DeviceMonitor callback raised")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    devs = snapshot()
    if devs is None:
        sys.exit("Device query failed.")
    for d in devs:
        mark = " ← default" if d["is_default"] else ""
        print(f"[{d['index']:2d}] {d['name']}  ({d['hostapi']}, "
              f"{d['channels']} ch, {d['default_samplerate']:.0f} Hz){mark}")
