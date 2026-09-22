"""hand_cursor_2.py — camera→desktop mapping (ScreenGeometry), smoothing
(One-Euro / low-pass), frame sizing and camera discovery."""

import statistics

import numpy as np
import pytest

from monitors import ALL_SCREENS


class FakeBackend:
    def __init__(self, geom=(0, 0, 1920, 1080), monitors=1):
        self.geom, self.monitors, self.fail = geom, monitors, False

    def read_geometry(self):
        if self.fail:
            raise OSError("display gone")
        return self.geom

    def monitor_count(self):
        return self.monitors


@pytest.fixture
def plain(hc, monkeypatch):
    """Mapping with sensitivity 1.0, no inversion, default margins."""
    monkeypatch.setattr(hc, "CURSOR_SENSITIVITY", 1.0)
    monkeypatch.setattr(hc, "INVERT_CURSOR_X", False)
    monkeypatch.setattr(hc, "SCREEN_SOURCE", "auto")
    return hc


def geometry(hc, backend=None, cam=(640, 480), **kw):
    return hc.ScreenGeometry(backend or FakeBackend(), *cam, **kw)


def test_box_is_the_frame_inset_by_the_margins(plain):
    sg = geometry(plain)
    mx, mt, mb = plain.sanitise_margins()
    assert (sg.box_left, sg.box_right) == (round(mx * 640), round((1 - mx) * 640))
    assert (sg.box_top, sg.box_bottom) == (round(mt * 480), round((1 - mb) * 480))
    assert sg.ratio == pytest.approx(1920 / 1080)
    assert "1920×1080" in sg.describe()


def test_box_corners_reach_the_desktop_corners(plain):
    sg = geometry(plain)
    assert sg.to_screen(sg.box_left, sg.box_top) == (0, 0)
    assert sg.to_screen(sg.box_right, sg.box_bottom) == (1920, 1080)
    cx, cy = (sg.box_left + sg.box_right) / 2, (sg.box_top + sg.box_bottom) / 2
    assert sg.to_screen(cx, cy) == pytest.approx((960, 540))
    assert sg.center == (960, 540)


def test_outside_the_box_saturates_at_the_edge(plain):
    sg = geometry(plain)
    assert sg.to_screen(0, 0) == (0, 0)
    assert sg.to_screen(640, 480) == (1920, 1080)


def test_negative_origin_desktop_is_fully_reachable(plain):
    sg = geometry(plain, FakeBackend((-1920, 0, 3840, 1080), monitors=2))
    assert sg.to_screen(sg.box_left, sg.box_top) == (-1920, 0)
    assert sg.to_screen(sg.box_right, sg.box_top)[0] == 1920
    assert sg.center == (0, 540) and sg.monitors == 2


def test_sensitivity_scales_about_the_centre_and_clamps(plain, monkeypatch):
    sg = geometry(plain)
    x_quarter = sg.box_left + sg.box_w * 0.625           # maps to 1200 at 1.0×
    assert sg.to_screen(x_quarter, 0)[0] == pytest.approx(1200)
    monkeypatch.setattr(plain, "CURSOR_SENSITIVITY", 2.0)
    assert sg.to_screen(x_quarter, 0)[0] == pytest.approx(1440)
    assert sg.to_screen(sg.box_left + sg.box_w * 0.9, 0)[0] == 1920    # clamped


def test_effective_box_shrinks_with_sensitivity(plain, monkeypatch):
    sg = geometry(plain)
    assert sg.effective_box is None
    monkeypatch.setattr(plain, "CURSOR_SENSITIVITY", 2.0)
    left, top, right, bottom = sg.effective_box
    assert right - left == pytest.approx(sg.box_w / 2, abs=1)
    assert bottom - top == pytest.approx(sg.box_h / 2, abs=1)


def test_invert_reflects_about_the_frame_centre(plain, monkeypatch):
    sg = geometry(plain)
    monkeypatch.setattr(plain, "INVERT_CURSOR_X", True)
    assert sg.to_screen(sg.box_left, sg.box_top)[0] == 1920
    assert sg.to_screen(sg.box_right, sg.box_top)[0] == 0


def test_camera_change_rebuilds_the_box(plain):
    sg = geometry(plain)
    assert sg.set_camera_size(640, 480) is False
    assert sg.set_camera_size(1280, 720) is True
    assert sg.box_right == round((1 - plain.sanitise_margins()[0]) * 1280)


def test_poll_detects_hotplug_on_its_interval(plain):
    backend = FakeBackend()
    sg = geometry(plain, backend, poll_interval=2.0)
    assert sg.poll(0.0) is False                         # unchanged
    backend.geom = (0, 0, 3840, 1080)
    assert sg.poll(1.0) is False                         # too soon
    assert sg.poll(2.0) is True and sg.width == 3840
    backend.geom = (0, 0, 0, 0)                          # driver mid-reconfigure
    assert sg.poll(4.0) is False and sg.width == 3840


def test_degenerate_first_read_falls_back_to_tkinter(plain, monkeypatch, capsys):
    monkeypatch.setattr(plain, "probe_screen_tkinter", lambda: (0, 0, 1366, 768))
    sg = geometry(plain, FakeBackend((0, 0, 0, 0)))
    assert (sg.width, sg.source, sg.monitors) == (1366, "tkinter", 1)
    assert "falling back to tkinter" in capsys.readouterr().out
    assert sg.poll(10.0) is False                        # tkinter is not re-polled


def test_failing_native_backend_falls_back(plain, monkeypatch):
    backend = FakeBackend()
    backend.fail = True
    monkeypatch.setattr(plain, "probe_screen_tkinter", lambda: (0, 0, 1280, 800))
    assert geometry(plain, backend).source == "tkinter"


def test_no_geometry_at_all_is_an_error(plain, monkeypatch):
    backend = FakeBackend()
    backend.fail = True
    monkeypatch.setattr(plain, "probe_screen_tkinter", lambda: None)
    with pytest.raises(RuntimeError, match="could not determine the desktop size"):
        geometry(plain, backend)


def test_forced_native_source_does_not_fall_back(plain, monkeypatch):
    monkeypatch.setattr(plain, "SCREEN_SOURCE", "native")
    backend = FakeBackend()
    backend.fail = True
    monkeypatch.setattr(plain, "probe_screen_tkinter", lambda: pytest.fail("must not probe"))
    with pytest.raises(RuntimeError):
        geometry(plain, backend)


def test_target_screen_confines_the_cursor(plain, monkeypatch, capsys):
    monitors = [{"x": 0, "y": 0, "width": 1920, "height": 1080, "name": "a", "primary": True},
                {"x": -1280, "y": 0, "width": 1280, "height": 1024, "name": "b", "primary": False}]
    monkeypatch.setattr(plain, "enumerate_monitors", lambda backend=None: monitors)
    sg = geometry(plain, FakeBackend((-1280, 0, 3200, 1080)))
    assert sg.target == ALL_SCREENS and sg.left == -1280
    assert sg.set_target("Screen 2") is True
    assert (sg.left, sg.width, sg.target) == (-1280, 1280, "Screen 2")
    assert sg.set_target("Screen 2") is False
    assert sg.set_target("Screen 9") is True             # unplugged → Screen 1
    assert (sg.left, sg.target) == (0, "Screen 1")
    assert "not attached" in capsys.readouterr().out
    assert sg.set_target(ALL_SCREENS) is True and sg.width == 3200


# ── Smoothing ──────────────────────────────────────────────────────────────────

def test_low_pass_filter(hc):
    lp = hc.LowPassFilter(alpha=0.5)
    assert not lp.initialised
    assert lp(10.0) == 10.0 and lp(20.0) == 15.0 and lp.last_value == 15.0
    assert lp(0.0, alpha=1.0) == 0.0
    lp.reset()
    assert lp(7.0) == 7.0


def test_one_euro_adopts_the_first_sample_and_holds_a_constant(hc):
    f = hc.OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.015)
    assert f(100.0, 0.0) == 100.0
    assert all(f(100.0, i / 30) == pytest.approx(100.0) for i in range(1, 30))


def test_one_euro_suppresses_jitter_at_rest(hc):
    rng = np.random.default_rng(0)
    raw = 500 + rng.normal(0, 3, 300)
    f = hc.OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.015)
    out = [f(float(x), i / 30) for i, x in enumerate(raw)]
    # Measured ~2.5× quieter with the shipped tuning; at least halve it.
    assert statistics.pstdev(out[30:]) < statistics.pstdev(raw[30:]) / 2


def test_one_euro_beta_trades_lag_for_speed(hc):
    def lag(beta):
        f = hc.OneEuroFilter(freq=30, min_cutoff=1.0, beta=beta)
        f(0.0, 0.0)
        return 1000.0 - [f(1000.0, i / 30) for i in range(1, 6)][-1]
    assert lag(0.5) < lag(0.0)


def test_one_euro_reset_snaps_to_the_new_position(hc):
    f = hc.OneEuroFilter(freq=30)
    f(0.0, 0.0), f(0.0, 1 / 30)
    f.reset()
    assert f(900.0, 5.0) == 900.0


def test_alpha_formula(hc):
    a = hc.OneEuroFilter._alpha(1.0, dt=1 / 30)
    assert a == pytest.approx(1 / (1 + (1 / (2 * np.pi)) * 30))
    assert hc.OneEuroFilter._alpha(10.0, freq=30) > a
    assert hc.OneEuroFilter._alpha(1.0) == pytest.approx(a)        # 30 Hz fallback


# ── Frame sizing & camera discovery ────────────────────────────────────────────

@pytest.mark.parametrize("size, budget, expected", [
    ((640, 480), (1280, 720), (640, 480)),
    ((1920, 1080), (1280, 720), (1280, 720)),
    ((4000, 3000), (1280, 720), (960, 720)),
    ((1920, 1080), (0, 720), (1920, 1080)),
])
def test_fit_within_keeps_the_aspect_ratio(hc, size, budget, expected):
    assert hc.WebcamStream._fit_within(*size, budget) == expected


class FakeCapture:
    opened = []

    def __init__(self, index, api=None):
        self.index, self.released = index, False
        FakeCapture.opened.append(self)

    def isOpened(self):                                   # noqa: N802 — cv2 API
        return self.index in (0, 1)

    def read(self):
        if self.index == 0:
            return True, np.zeros((720, 1280, 3), np.uint8)
        return False, None                                # opens, never delivers

    def get(self, prop):
        return 640                                        # driver lies about the size

    def release(self):
        self.released = True


def test_scan_requires_a_real_frame(hc, monkeypatch, capsys):
    FakeCapture.opened = []
    monkeypatch.setattr(hc.cv2, "VideoCapture", FakeCapture, raising=False)
    monkeypatch.setattr(hc.time, "sleep", lambda s: None)
    found = hc.scan_cameras(max_index=3)
    assert found == {0: (1280, 720)}                      # the array wins over the claim
    assert all(cap.released for cap in FakeCapture.opened)
    out = capsys.readouterr().out
    assert "driver reported 640×640" in out and "produced no frame" in out


def test_camera_helpers(hc):
    assert hc.pick_best_camera([0, 2, 1]) == 2
    assert hc.describe_api(None) == "auto-negotiated"
    assert hc.describe_api(hc.cv2.CAP_DSHOW) == "DSHOW"
    assert hc.describe_api(123456) == "backend 123456"
