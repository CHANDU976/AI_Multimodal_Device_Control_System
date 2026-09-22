"""monitors.py — display enumeration and target-screen resolution.
Coordinates may be negative (a monitor left of the primary)."""

import sys
import types

import pytest

import monitors as mon
from monitors import ALL_SCREENS, monitor_labels, monitor_union, resolve_monitor_target


def m(x, y, w, h, name=""):
    return {"x": x, "y": y, "width": w, "height": h, "name": name, "primary": x == y == 0}


SIDE_BY_SIDE = [m(0, 0, 1920, 1080), m(1920, 0, 1920, 1080)]
LEFT_OF_PRIMARY = [m(0, 0, 1920, 1080), m(-1920, 0, 1920, 1080)]
STACKED_OFFSET = [m(0, 0, 2560, 1440), m(320, -1080, 1920, 1080)]


@pytest.mark.parametrize("monitors, union", [
    (SIDE_BY_SIDE, (0, 0, 3840, 1080)),
    (LEFT_OF_PRIMARY, (-1920, 0, 3840, 1080)),
    (STACKED_OFFSET, (0, -1080, 2560, 2520)),
    ([m(0, 0, 1366, 768)], (0, 0, 1366, 768)),
])
def test_union_covers_the_whole_arrangement(monitors, union):
    assert monitor_union(monitors) == union


def test_union_of_nothing():
    assert monitor_union([]) is None


def test_labels():
    assert monitor_labels([]) == [ALL_SCREENS]
    assert monitor_labels(SIDE_BY_SIDE) == [ALL_SCREENS, "Screen 1", "Screen 2"]


@pytest.mark.parametrize("label, rect, resolved", [
    (ALL_SCREENS, (-1920, 0, 3840, 1080), ALL_SCREENS),
    ("all", (-1920, 0, 3840, 1080), ALL_SCREENS),
    ("Extended", (-1920, 0, 3840, 1080), ALL_SCREENS),
    ("", (-1920, 0, 3840, 1080), ALL_SCREENS),
    (None, (-1920, 0, 3840, 1080), ALL_SCREENS),
    ("Screen 2", (-1920, 0, 1920, 1080), "Screen 2"),
    ("screen1", (0, 0, 1920, 1080), "Screen 1"),
    ("Screen 3", (0, 0, 1920, 1080), "Screen 1"),     # unplugged → first screen
    ("Screen 0", (0, 0, 1920, 1080), "Screen 1"),
    ("Projector", (0, 0, 1920, 1080), "Screen 1"),
])
def test_resolve_target_falls_back_sensibly(label, rect, resolved):
    assert resolve_monitor_target(label, LEFT_OF_PRIMARY) == (rect, resolved)


def test_resolve_with_no_monitors():
    assert resolve_monitor_target("Screen 1", []) == (None, ALL_SCREENS)


# ── Enumeration fallbacks ──────────────────────────────────────────────────────

@pytest.fixture
def no_sources(monkeypatch):
    for name in ("_from_screeninfo", "_from_win32", "_virtual_desktop"):
        monkeypatch.setattr(mon, name, lambda: None)


def test_first_working_source_wins(monkeypatch):
    monkeypatch.setattr(mon, "_from_screeninfo", lambda: None)
    monkeypatch.setattr(mon, "_from_win32", lambda: SIDE_BY_SIDE)
    monkeypatch.setattr(mon, "_virtual_desktop", lambda: pytest.fail("not reached"))
    assert mon.enumerate_monitors() == SIDE_BY_SIDE


def test_raising_source_is_skipped(monkeypatch):
    def boom():
        raise OSError("driver")
    monkeypatch.setattr(mon, "_from_screeninfo", boom)
    monkeypatch.setattr(mon, "_from_win32", lambda: SIDE_BY_SIDE)
    assert mon.enumerate_monitors() == SIDE_BY_SIDE


def test_backend_is_the_last_resort(no_sources):
    backend = types.SimpleNamespace(read_geometry=lambda: (-1920, 0, 3840, 1080))
    (only,) = mon.enumerate_monitors(backend)
    assert (only["x"], only["width"], only["name"]) == (-1920, 3840, "Virtual desktop")


@pytest.mark.parametrize("geometry", [(0, 0, 0, 0), None])
def test_nothing_usable_gives_empty(no_sources, geometry):
    def read():
        if geometry is None:
            raise OSError("no display")
        return geometry
    assert mon.enumerate_monitors(types.SimpleNamespace(read_geometry=read)) == []
    assert mon.enumerate_monitors() == []


def test_screeninfo_source_filters_bad_entries(monkeypatch):
    fake = types.ModuleType("screeninfo")
    fake.get_monitors = lambda: [
        types.SimpleNamespace(x=0, y=0, width=1920, height=1080, name="DP-1", is_primary=True),
        types.SimpleNamespace(x=1920, y=0, width=0, height=1080, name="ghost"),
        types.SimpleNamespace(x=-1280, y=0, width=1280, height=1024, name=None),
        object(),
    ]
    monkeypatch.setitem(sys.modules, "screeninfo", fake)
    found = mon._from_screeninfo()
    assert [(d["name"], d["x"], d["primary"]) for d in found] == \
        [("DP-1", 0, True), ("Display 3", -1280, False)]


def test_screeninfo_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "screeninfo", None)
    assert mon._from_screeninfo() is None


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows guard")
def test_win32_sources_are_inert_elsewhere():
    assert mon._from_win32() is None and mon._virtual_desktop() is None


@pytest.mark.skipif(sys.platform != "win32", reason="real EnumDisplayMonitors")
def test_win32_enumeration_returns_real_rectangles():
    found = mon._from_win32()
    assert found and all(d["width"] > 0 and d["height"] > 0 for d in found)
