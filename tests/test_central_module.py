import tempfile
import unittest
from pathlib import Path

from central_module import config
from central_module.actions.executor import ActionExecutor
from central_module.actions.os_adapter import OSAdapter
from central_module.learning.adaptation import BehaviorModel
from central_module.learning.preferences import PreferenceStore
from central_module.nlp.intent_engine import IntentMatch, IntentEngine
from central_module.nlp.intents import BY_NAME


class RecordingAdapter(OSAdapter):
    APP_COMMANDS = {"browser": "browser-command"}

    def __init__(self):
        self.calls = []

    def open_app(self, command):
        self.calls.append(("open_app", command))

    def hotkey(self, *keys):
        self.calls.append(("hotkey", keys))

    def type_text(self, text):
        self.calls.append(("type_text", text))

    def power(self, op, delay_s=3):
        self.calls.append(("power", op, delay_s))


class CentralModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._nlp_engine = config.NLP["engine"]
        config.NLP["engine"] = "rules"

    @classmethod
    def tearDownClass(cls):
        config.NLP["engine"] = cls._nlp_engine

    def make_executor(self):
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        prefs = PreferenceStore(str(root / "preferences.json"))
        behavior = BehaviorModel(str(root / "usage.json"))
        adapter = RecordingAdapter()
        executor = ActionExecutor(adapter, prefs, behavior)
        self.addCleanup(temp_dir.cleanup)
        return executor, adapter

    @staticmethod
    def match(name, slots=None):
        return IntentMatch(BY_NAME[name], slots or {})

    def test_exact_intent_recognition_and_typed_slot(self):
        engine = IntentEngine(fuzzy_threshold=1.1)

        match = engine.parse("increase volume by five")

        self.assertIsNotNone(match)
        self.assertEqual(match.intent.name, "volume_up")
        self.assertEqual(match.slots, {"count": 5})

    def test_executor_dispatches_to_adapter(self):
        executor, adapter = self.make_executor()

        result = executor.execute(self.match("open_app", {"app": "browser"}))

        self.assertTrue(result.ok)
        self.assertEqual(adapter.calls, [("open_app", "browser-command")])

    def test_destructive_intents_require_confirmation(self):
        executor, adapter = self.make_executor()
        match = self.match("shutdown")

        pending = executor.execute(match)
        confirmed = executor.execute(match, confirmed=True)

        self.assertTrue(pending.ok)
        self.assertTrue(pending.needs_confirm)
        self.assertTrue(confirmed.ok)
        self.assertEqual(adapter.calls, [("power", "shutdown", 3)])

    def test_shared_edit_combo_uses_platform_modifier(self):
        executor, adapter = self.make_executor()

        result = executor.execute(self.match("copy"))

        self.assertTrue(result.ok)
        self.assertEqual(adapter.calls, [("hotkey", ("ctrl", "c"))])

    def test_dry_run_does_not_call_adapter(self):
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        adapter = RecordingAdapter()
        executor = ActionExecutor(
            adapter,
            PreferenceStore(str(root / "preferences.json")),
            BehaviorModel(str(root / "usage.json")),
            dry_run=True,
        )
        self.addCleanup(temp_dir.cleanup)

        result = executor.execute(self.match("type_text", {"text": "hello"}))

        self.assertTrue(result.ok)
        self.assertIn("dry-run", result.message)
        self.assertEqual(adapter.calls, [])


if __name__ == "__main__":
    unittest.main()