# Central Module

The orchestrator of the Context-Aware Multimodal Device Control System. It
runs the submodules as supervised child processes, turns recognised speech
into intents, executes them on the PC through a per-OS adapter, and adapts to
the user over time.

```
                       ┌─────────────────────────────────────────────┐
  speech child ──────▶ │              CentralManager                 │
  (speech_runner wraps │                                             │
   SpeechManager)      │  normalize → custom cmds → confirm check    │
                       │  → dictation mode → IntentEngine            │
  gesture child ─────▶ │  → ActionExecutor → OSAdapter               │
  (your gesture app,   │        │                    win / mac / linux│
   or the test stub)   │        └──▶ BehaviorModel + PreferenceStore │
                       └─────────────────────────────────────────────┘
```

Everything is stdlib-only — the central module adds **zero** pip
dependencies on Windows/macOS (Linux wants `xdotool`, see below).

## Running

```bash
# full system: speech child + actions (venv python, project root)
.venv\Scripts\python.exe -m central_module

# develop without a microphone: type commands, see what would happen
python -m central_module --simulate --dry-run

# one utterance through the whole pipeline
python -m central_module --once "open chrome" --dry-run

# -v shows live speech partials and debug logs
```

## Process supervision

`supervisor.py` launches each child with pipes:

| channel | direction        | protocol                                        |
| ------- | ---------------- | ----------------------------------------------- |
| stdout  | child → central | JSON lines: events (`{"type": ...}`)          |
| stdin   | central → child | JSON lines: control ops (`{"op": ...}`)       |
| stderr  | child → central | free-form logs, re-logged with the child's name |

Crashed children restart with exponential backoff (1 s → 30 s cap, 5
attempts; a minute of healthy uptime resets the count). `{"op": "shutdown"}`
asks for a graceful exit before terminate/kill.

### Speech child

`runners/speech_runner.py` wraps `speech_module.SpeechManager` and speaks the
protocol its docstrings anticipate. Control ops: `set_language`,
`set_input_device`, `get_devices`, `ping`, `shutdown`. It redirects fd 1 to
stderr after duplicating it, so a chatty native library can never corrupt the
event stream.

### Gesture child

The gesture app is independent and already executes its own mapped actions —
the central module just owns its lifecycle. Point preferences at it:

```json
"gesture": {
  "enabled": true,
  "command": ["C:/path/to/python.exe", "C:/path/to/gesture_app/main.py"]
}
```

Optionally, if it prints `{"type": "gesture", "name": "fist"}` lines on
stdout, map gestures to spoken commands and they run through the central
pipeline:

```json
"gesture_actions": { "fist": "pause", "thumbs_up": "volume up" }
```

## NLP

`nlp/` is a ladder, cheapest first:

1. **alias** — exact lookup of phrases the adaptation layer has learned.
2. **exact** — compiled regex patterns with typed slots
   (`open {app}`, `set volume to {number}`). Free precision: a verbatim
   catalog phrase never risks a model error.
3. **model** — a fine-tuned **DistilBERT joint intent + slot model**
   (`nlp/nlu/`, JointBERT architecture): one forward pass classifies the
   intent (~50 classes) and BIO-tags the argument words. Trained on tens of
   thousands of synthetic utterances generated from the catalog plus natural
   paraphrases ("fire up chrome", "make it louder"), augmented with
   politeness, determiners, number-word forms, and realistic typos — and a
   **none class** (prose, questions, near-miss commands like
   "delete chatter.txt") so it *refuses* instead of guessing. Inference is
   int8-quantized, ~15-40 ms on CPU, fully offline from `Models/NLU/`.
4. **fuzzy** — difflib similarity as a last-resort safety net, seeing only
   what both the patterns and the model rejected. Wordpiece tokenization is
   brittle on misspelled verbs ("opun chrome"), where character-level
   similarity is the right tool; length and token-agreement guards keep
   genuine non-commands out.

**Measured** (2 epochs, 34k train / 1.8k dev, ~36 min CPU): **98.9% dev
intent accuracy, 99.7% slot accuracy**, ~15 ms per parse end-to-end.

No match → the phrase is *recorded*, never guessed at.

**Training the model** (one-time, downloads `distilbert-base-uncased` once):

```bash
.venv\Scripts\python.exe -m central_module.nlp.nlu.train           # ~45 min CPU
.venv\Scripts\python.exe -m central_module.nlp.nlu.train --quick   # 2-min pipeline check
```

**Adding a command** = one `Intent` entry in `nlp/intents.py` + one
`_do_<action>` handler in `actions/executor.py` + retrain so the model
learns the new class (patterns and paraphrases feed the generator).

## OS independence

`actions/os_adapter.py` defines the interface; the executor never touches an
OS API directly.

|                | Windows                                | macOS                       | Linux (X11)                                                                |
| -------------- | -------------------------------------- | --------------------------- | -------------------------------------------------------------------------- |
| implementation | ctypes`SendInput` / `ShellExecute` | `osascript` + stock tools | `xdotool`, `pactl`, `xdg-open`                                       |
| extra install  | none                                   | none (grant Accessibility)  | `sudo apt install xdotool` (+ `playerctl`, `brightnessctl` optional) |

Volume/media use media keys, typing is Unicode-injection (any script incl.
Hindi), apps resolve via user aliases → built-in table → `PATH` → OS shell.
Unsupported capabilities raise a clear message instead of crashing.

Destructive intents (shutdown, restart) arm and wait for a spoken "yes";
anything else disarms them.

## Preferences & adaptation

Two JSON files in `central_module/data/` (override with `MMCTRL_DATA`):

* **preferences.json** — what the user chose: language, gesture app command,
  `app_aliases`, `gesture_actions`, `custom_commands` (macros like
  `"battle stations": "open code"`).
* **usage.json** — what the system learned: intent/app usage counts, phrase
  statistics, promoted aliases, unknown phrases.

Adaptation loop: a phrase that only fuzzy-matched but then executed
successfully 3 times is **promoted to an alias** — from then on it hits the
exact fast path. Aliases that start failing are demoted. Spoken app names
that resolved through discovery are saved to `app_aliases`. Unrecognised
phrases are counted; `stats` in simulate mode shows the top ones — ready to
become custom commands.

## Smoke tests

Every non-trivial file runs standalone:
The safe central execution tests mock OS side effects while exercising intent
recognition, slot extraction, adapter dispatch, confirmation, and dry-run
isolation:

```bash
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

```bash
python -m central_module.nlp.normalize
python -m central_module.nlp.intent_engine
python -m central_module.learning.preferences
python -m central_module.learning.adaptation
python -m central_module.supervisor          # spawns a tiny flaky child
python -m central_module.actions.windows_adapter   # touches volume/typing!
```
