"""Test doubles for every hardware / native boundary in the project.

The rule for the whole suite: tests never touch the real OS. No key is
injected, no click is sent, no microphone or camera is opened and no model
is loaded unless a test is explicitly marked for it. Everything that would
do so is replaced by a recorder defined here.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field

import numpy as np


# ── pynput ─────────────────────────────────────────────────────────────────────

class InputRecorder:
    """Collects every mouse/keyboard call made through the fake pynput."""

    def __init__(self) -> None:
        self.events: list[tuple] = []
        # Operation names ("key.press", "mouse.click", ...) that should raise,
        # to exercise the release-in-finally paths.
        self.fail_on: set[str] = set()
        self.position = (0, 0)

    def record(self, op: str, *args) -> None:
        if op in self.fail_on:
            raise RuntimeError(f"injected failure on {op}")
        self.events.append((op, *args))

    def ops(self) -> list[str]:
        return [e[0] for e in self.events]


_KEY_NAMES = (
    "alt", "backspace", "caps_lock", "cmd", "ctrl", "delete", "down", "end",
    "enter", "esc", "home", "insert", "left", "media_next",
    "media_play_pause", "media_previous", "page_down", "page_up",
    "print_screen", "right", "shift", "space", "tab", "up", "volume_down",
    "volume_mute", "volume_up", *(f"f{i}" for i in range(1, 13)),
)


def install_fake_pynput(monkeypatch) -> InputRecorder:
    """Replace pynput in sys.modules with recorders; returns the recorder."""
    rec = InputRecorder()

    class Button:
        left = "Button.left"
        right = "Button.right"
        middle = "Button.middle"

    Key = type("Key", (), {name: f"Key.{name}" for name in _KEY_NAMES})

    class MouseController:
        @property
        def position(self):
            return rec.position

        @position.setter
        def position(self, value):
            rec.record("mouse.move", tuple(value))
            rec.position = tuple(value)

        def click(self, button, count=1):
            rec.record("mouse.click", button, count)

        def press(self, button):
            rec.record("mouse.press", button)

        def release(self, button):
            rec.record("mouse.release", button)

        def scroll(self, dx, dy):
            rec.record("mouse.scroll", dx, dy)

    class KeyController:
        def press(self, key):
            rec.record("key.press", key)

        def release(self, key):
            rec.record("key.release", key)

    root = types.ModuleType("pynput")
    mouse = types.ModuleType("pynput.mouse")
    keyboard = types.ModuleType("pynput.keyboard")
    mouse.Button, mouse.Controller = Button, MouseController
    keyboard.Key, keyboard.Controller = Key, KeyController
    root.mouse, root.keyboard = mouse, keyboard
    monkeypatch.setitem(sys.modules, "pynput", root)
    monkeypatch.setitem(sys.modules, "pynput.mouse", mouse)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)
    return rec


# ── vosk ───────────────────────────────────────────────────────────────────────

class FakeVosk:
    """Stand-in for the `vosk` module with scriptable recogniser output.

    `final` is the dict FinalResult()/Result() return; `partial` the dict
    PartialResult() returns; `accept` decides AcceptWaveform()'s verdict.
    `reject_grammar` makes grammar-constrained construction raise, the way a
    model without a dynamic graph does. `missing_words` are the words the
    model's lexicon lacks (vosk_model_find_word() → -1).
    """

    def __init__(self) -> None:
        self.final: dict = {"text": ""}
        self.partial: dict = {"partial": ""}
        self.accept = False
        self.reject_grammar = False
        self.missing_words: set[str] = set()
        self.recognizers: list = []
        self.models: list[str] = []
        self.log_level = None
        fake = self

        class Model:
            def __init__(self, path):
                fake.models.append(path)
                self.path = path

            def vosk_model_find_word(self, word):
                return -1 if word in fake.missing_words else 1

        class KaldiRecognizer:
            def __init__(self, model, rate, grammar=None):
                if grammar is not None and fake.reject_grammar:
                    raise RuntimeError("model has no dynamic graph")
                self.model, self.rate = model, rate
                self.grammar = json.loads(grammar) if grammar else None
                self.max_alternatives = None
                self.words = False
                self.audio = b""
                self.resets = 0
                fake.recognizers.append(self)

            def SetMaxAlternatives(self, n):     # noqa: N802 — vosk's API
                self.max_alternatives = n

            def SetWords(self, on):              # noqa: N802
                self.words = on

            def AcceptWaveform(self, data):      # noqa: N802
                self.audio += bytes(data)
                return fake.accept

            def Result(self):                    # noqa: N802
                return json.dumps(fake.final)

            def FinalResult(self):               # noqa: N802
                return json.dumps(fake.final)

            def PartialResult(self):             # noqa: N802
                return json.dumps(fake.partial)

            def Reset(self):                     # noqa: N802
                self.resets += 1
                self.audio = b""

        self.Model = Model
        self.KaldiRecognizer = KaldiRecognizer

    def SetLogLevel(self, level):                # noqa: N802
        self.log_level = level


def vosk_stub_module() -> types.ModuleType:
    """A minimal importable `vosk` for machines without the real wheel."""
    mod = types.ModuleType("vosk")
    fake = FakeVosk()
    mod.Model = fake.Model
    mod.KaldiRecognizer = fake.KaldiRecognizer
    mod.SetLogLevel = fake.SetLogLevel
    mod.__fake__ = True
    return mod


# ── onnxruntime (Silero VAD) ───────────────────────────────────────────────────

class FakeOnnxRuntime(types.ModuleType):
    """`onnxruntime` whose InferenceSession scores speech by frame energy.

    layout "v5" exposes inputs input/state/sr, "v6" exposes input/h/c — the
    two Silero ONNX generations SileroVAD detects.
    """

    def __init__(self, layout: str = "v5", threshold_rms: float = 0.05) -> None:
        super().__init__("onnxruntime")
        self.layout = layout
        self.threshold_rms = threshold_rms
        self.calls = 0
        fake = self

        class SessionOptions:
            inter_op_num_threads = 0
            intra_op_num_threads = 0
            log_severity_level = 0

        class _Input:
            def __init__(self, name):
                self.name = name

        class InferenceSession:
            def __init__(self, path, providers=None, sess_options=None):
                self.path = path

            def get_inputs(self):
                names = {"v5": ("input", "state", "sr"),
                         "v6": ("input", "h", "c")}.get(fake.layout, ("x",))
                return [_Input(n) for n in names]

            def run(self, _outputs, feeds):
                fake.calls += 1
                x = feeds["input"][0, 64:]          # drop the carried context
                rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
                prob = np.array([[0.95 if rms > fake.threshold_rms else 0.02]],
                                dtype=np.float32)
                if "state" in feeds:
                    return [prob, feeds["state"]]
                return [prob, feeds["h"], feeds["c"]]

        self.SessionOptions = SessionOptions
        self.InferenceSession = InferenceSession


# ── sounddevice ────────────────────────────────────────────────────────────────

class FakeSoundDevice:
    """Records RawInputStream construction and PortAudio re-initialisation."""

    def __init__(self, devices: list[dict] | None = None) -> None:
        self.streams: list = []
        self.terminated = 0
        self.initialized = 0
        self._devices = devices or [
            {"name": "Speakers", "max_input_channels": 0},
            {"name": "USB Mic", "max_input_channels": 1},
            {"name": "Array Mic", "max_input_channels": 2},
        ]
        fake = self

        class RawInputStream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.started = self.stopped = self.closed = False
                fake.streams.append(self)

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True

            def close(self):
                self.closed = True

        self.RawInputStream = RawInputStream

    def query_devices(self):
        return list(self._devices)

    def _terminate(self):
        self.terminated += 1

    def _initialize(self):
        self.initialized += 1


# ── cv2 / mediapipe (only when the real packages are absent) ───────────────────

def cv2_stub_module() -> types.ModuleType:
    mod = types.ModuleType("cv2")
    mod.INTER_AREA = 3
    mod.INTER_LINEAR = 1
    for i, name in enumerate(("CAP_PROP_FRAME_WIDTH", "CAP_PROP_FRAME_HEIGHT",
                              "CAP_PROP_FPS", "CAP_PROP_BUFFERSIZE")):
        setattr(mod, name, 3 + i)
    mod.CAP_DSHOW, mod.CAP_MSMF, mod.CAP_V4L2 = 700, 1400, 200
    mod.setNumThreads = lambda n: None
    mod.VideoCapture = None             # tests that scan cameras install one
    mod.resize = lambda frame, size, interpolation=None: frame

    # Just enough of the image API for the engine's real tracking loop
    # (tests/gesture/test_gesture_events.py drives it with scripted frames):
    # numpy versions of the two conversions, and no-op drawing.
    def _into(result, dst):
        if dst is None:
            return result
        dst[...] = result
        return dst

    mod.COLOR_BGR2RGB = 4
    mod.flip = lambda src, code, dst=None: _into(
        src[:, ::-1] if code == 1 else src[::-1] if code == 0 else src[::-1, ::-1], dst)
    mod.cvtColor = lambda src, code, dst=None: _into(src[..., ::-1], dst)
    mod.FILLED, mod.LINE_AA, mod.FONT_HERSHEY_SIMPLEX = -1, 16, 0
    mod.circle = mod.rectangle = mod.putText = lambda *a, **k: None
    mod.__fake__ = True
    return mod


def mediapipe_stub_module() -> types.ModuleType:
    mod = types.ModuleType("mediapipe")
    solutions = types.SimpleNamespace(
        hands=types.SimpleNamespace(Hands=None, HAND_CONNECTIONS=()),
        drawing_utils=types.SimpleNamespace(draw_landmarks=lambda *a, **k: None),
    )
    mod.solutions = solutions
    mod.__fake__ = True
    return mod


def ensure_importable(name: str, factory) -> bool:
    """Import *name*, or register factory() under that name if it is missing.

    Returns True when the real module is in use. The fallback is scoped to
    the native dependency only; the project module under test is always the
    real one.
    """
    try:
        __import__(name)
        return True
    except Exception:
        sys.modules[name] = factory()
        return False


# ── central module: an OSAdapter that only records ─────────────────────────────

def recording_adapter_class():
    """Build a RecordingAdapter subclass of the real OSAdapter interface."""
    from central_module.actions.os_adapter import OSAdapter

    class RecordingAdapter(OSAdapter):
        os_name = "TestOS"
        primary_mod = "ctrl"
        APP_COMMANDS = {"browser": "browser-cmd", "chrome": "chrome-cmd"}

        def __init__(self):
            self.calls: list[tuple] = []

        def _rec(self, *call):
            self.calls.append(call)

        def key_tap(self, key):
            self._rec("key_tap", key)

        def hotkey(self, *keys):
            self._rec("hotkey", keys)

        def type_text(self, text):
            self._rec("type_text", text)

        def scroll(self, direction, steps):
            self._rec("scroll", direction, steps)

        def click(self, button="left", double=False):
            self._rec("click", button, double)

        def open_app(self, command):
            self._rec("open_app", command)

        def volume_change(self, steps):
            self._rec("volume_change", steps)

        def volume_mute(self, on):
            self._rec("volume_mute", on)

        def set_volume(self, percent):
            self._rec("set_volume", percent)

        def media(self, op):
            self._rec("media", op)

        def window(self, op):
            self._rec("window", op)

        def lock(self):
            self._rec("lock")

        def screenshot(self):
            self._rec("screenshot")
            return "saved to /tmp/shot.png"

        def power(self, op, delay_s=3):
            self._rec("power", op, delay_s)

        def brightness(self, delta):
            self._rec("brightness", delta)

    return RecordingAdapter


@dataclass
class RecordedRun:
    """Captures OSAdapter._run invocations for command-building tests."""
    calls: list = field(default_factory=list)
    stdout: str = ""

    def __call__(self, argv, what, timeout=15.0):
        self.calls.append((list(argv), what))
        return self.stdout
