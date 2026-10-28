# PR #14 — Vv2

> ⚠️ **Mergeable — needs reviewer attention** · 24 files · +1336 / −217 · 4 commit(s)

- Calibrated constants changed

| | |
| --- | --- |
| Pull request | https://github.com/CHANDU976/AI_Multimodal_Device_Control_System/pull/14 |
| Author | CHANDU976 |
| Branch | `vv2` → `main` |
| Base / head | `daf618b4cd` / `436af27faa` |
| Workflow run | https://github.com/CHANDU976/AI_Multimodal_Device_Control_System/actions/runs/40626956628 |
| Generated | 2026-10-28T03:31:22Z (push) |

## Checks

| Check | Result | Details |
| --- | --- | --- |
| Lint (blocking rules) | ✅ success | 0 blocking finding(s) · advisory: 2 in changed files, 0 on added lines |
| Repository hygiene | ✅ success | 0 error(s), 0 warning(s) |
| Tests (all platforms) | ✅ success | `tests-macos-latest`: 1972 passed, 0 failed, 36 skipped, 4 xfail · 23.9 s<br>`tests-ubuntu-latest`: 1972 passed, 0 failed, 36 skipped, 4 xfail · 27.5 s<br>`tests-windows-latest`: 2005 passed, 0 failed, 3 skipped, 4 xfail · 49.1 s |
| NLU model tests (torch) | ✅ success | `ml`: 13 passed, 0 failed, 13 skipped, 0 xfail · 9.3 s |
| Dependency resolution | ⏭️ skipped | no requirement files changed |
| Coverage of added lines | ✅ | 89.3% of 243 executable added line(s) (target 60%) |

## Needs attention

- ⚠️ **Calibrated constants changed** — `Mousegesture/gesture_config.json`, `central_module/config.py`, `speech_module/config.py`  
  `speech_module/config.py` (speech routing / VAD / denoiser thresholds); `central_module/config.py` (central NLP, supervisor and adaptation constants); `Mousegesture/gesture_config.json` (gesture bindings and cursor settings). The project rule is that every constant carries the measurement that justifies it — check the comments were updated with the new numbers.
- ℹ️ **Needs a hands-on check CI cannot do** — `speech_module/speech_manager.py`  
  CI fakes the hardware boundary for: `speech_manager.py` → the real-time speech loop with a microphone.
- ℹ️ **Large change set (1553 lines)**  
  Consider splitting it: smaller PRs review faster and bisect cleanly.

## Changes by module

| Module | Files | + | − | Components |
| --- | ---: | ---: | ---: | --- |
| Central module (orchestrator + NLU) | 9 | 501 | 97 | `core` ×5, `runners` ×3, `learning` |
| Speech module | 2 | 44 | 23 | `config`, `speech_manager` |
| Gesture module | 2 | 17 | 10 | `config`, `gesture_events` |
| Tests | 8 | 633 | 69 | `central` ×5, `gesture`, `helpers`, `speech` |
| Documentation | 3 | 141 | 18 | `gesture`, `README.md`, `docs` |

## Coverage

### Lines added by this change set

**89.3%** of 243 executable added line(s) are exercised by the tests.

| File | Covered | Uncovered added lines |
| --- | ---: | --- |
| `Mousegesture/gesture_events.py` | 4/4 (100.0%) | — |
| `central_module/__main__.py` | 2/2 (100.0%) | — |
| `central_module/config.py` | 2/2 (100.0%) | — |
| `central_module/manager.py` | 70/71 (98.6%) | 144 |
| `central_module/runners/gesture_runner.py` | 87/111 (78.4%) | 90-91, 106-114, 141-142, 149, 170-171, 211-213, 217, 227-230 |
| `central_module/runners/gesture_stub.py` | 26/26 (100.0%) | — |
| `central_module/runners/speech_runner.py` | 22/23 (95.7%) | 94 |
| `central_module/supervisor.py` | 3/3 (100.0%) | — |
| `speech_module/speech_manager.py` | 1/1 (100.0%) | — |

### Whole modules

| Module | Files | Statements | Covered |
| --- | ---: | ---: | ---: |
| Central module (orchestrator + NLU) | 29 | 2065 | 93.3% |
| Gesture module | 13 | 7207 | 45.8% |
| Speech module | 13 | 1506 | 91.3% |

## Lint findings in changed files

| Location | Rule | Message | New |
| --- | --- | --- | :---: |
| `speech_module/config.py:8` | E501 | Line too long (105 > 100) |  |
| `tests/gesture/test_gesture_events.py:25` | F401 | `math` imported but unused |  |

## Known issues pinned by tests (xfail)

Documented defects the suite tracks; each flips to a failure once fixed, so the marker gets removed.

- `tests.central.test_intent_engine::test_every_pattern_is_reachable_by_exact_match[press_key:press the {key} key]` — command_normalize drops 'the', so a non-text pattern containing it never matches
- `tests.central.test_intent_engine::test_every_pattern_is_reachable_by_exact_match[double_click:double click]` — command_normalize turns 'double' into '2' before matching
- `tests.central.test_intent_engine::test_every_pattern_is_reachable_by_exact_match[double_click:click twice]` — command_normalize turns 'twice' into '2' before matching
- `tests.repo.test_fetch_deepfilternet_model::test_extract_blocks_sibling_prefix_traversal` — zip-slip: the guard compares string prefixes, so '../DFN3evil/x' resolves to a SIBLING directory whose path still starts with '.../DFN3'. Compare with Path.is_relative_to(dest.resolve()) instead.

<details><summary>Skipped tests by reason</summary>

| Reason | Count |
| --- | ---: |
| WindowsAdapter binds user32.dll at import | 64 |
| Models/NLU/joint-distilbert/joint_nlu.pt is a Git-LFS pointer (run `git lfs pull` to fetch it) | 13 |
| collection skipped | 3 |
| could not import 'onnxruntime' | 3 |
| WindowsAdapter binds user32.dll | 2 |
| real EnumDisplayMonitors | 2 |
| checks the non-Windows guard | 1 |

</details>

## Commits

| Commit | Author | Date | Subject |
| --- | --- | --- | --- |
| `09dd16c` | CHANDAN B L | 2026-10-22 | Gesture config: cursor control always on |
| `8c404bb` | CHANDAN B L | 2026-10-22 | Run the gesture module as a supervised child of the central module |
| `1f6b01b` | CHANDAN B L | 2026-10-22 | Give the speech and gesture modules one IPC protocol |
| `a569647` | CHANDAN B L | 2026-10-22 | Move the gesture event tests into tests/gesture so CI runs them |

<details><summary>All 24 changed project files</summary>

| Status | File | Module | + | − |
| :---: | --- | --- | ---: | ---: |
| M | `Mousegesture/IPC_EVENTS.md` | docs/gesture | 21 | 15 |
| M | `Mousegesture/gesture_config.json` | gesture/config | 1 | 1 |
| M | `Mousegesture/gesture_events.py` | gesture/gesture_events | 16 | 9 |
| M | `README.md` | docs/README.md | 6 | 3 |
| M | `central_module/README.md` | central/core | 19 | 6 |
| M | `central_module/__main__.py` | central/core | 16 | 4 |
| M | `central_module/config.py` | central/core | 16 | 6 |
| M | `central_module/learning/preferences.py` | central/learning | 4 | 2 |
| M | `central_module/manager.py` | central/core | 116 | 30 |
| M | `central_module/runners/gesture_runner.py` | central/runners | 206 | 30 |
| A | `central_module/runners/gesture_stub.py` | central/runners | 53 | 0 |
| M | `central_module/runners/speech_runner.py` | central/runners | 66 | 18 |
| M | `central_module/supervisor.py` | central/core | 5 | 1 |
| A | `docs/ipc-protocol.md` | docs/docs | 114 | 0 |
| M | `speech_module/config.py` | speech/config | 3 | 0 |
| M | `speech_module/speech_manager.py` | speech/speech_manager | 41 | 23 |
| A | `tests/central/test_ipc_protocol.py` | tests/central | 194 | 0 |
| M | `tests/central/test_learning.py` | tests/central | 7 | 6 |
| M | `tests/central/test_manager.py` | tests/central | 120 | 14 |
| M | `tests/central/test_runners_and_cli.py` | tests/central | 234 | 24 |
| M | `tests/central/test_supervisor.py` | tests/central | 28 | 4 |
| R | `Mousegesture/test_gesture_events.py` → `tests/gesture/test_gesture_events.py` | tests/gesture | 10 | 7 |
| M | `tests/helpers/fakes.py` | tests/helpers | 16 | 0 |
| M | `tests/speech/test_speech_manager.py` | tests/speech | 24 | 14 |

</details>

<details><summary>git diff --stat</summary>

```text
 Mousegesture/IPC_EVENTS.md                             |  36 ++++----
 Mousegesture/gesture_config.json                       |   2 +-
 Mousegesture/gesture_events.py                         |  25 ++++--
 README.md                                              |   9 +-
 central_module/README.md                               |  25 ++++--
 central_module/__main__.py                             |  20 ++++-
 central_module/config.py                               |  22 +++--
 central_module/learning/preferences.py                 |   6 +-
 central_module/manager.py                              | 146 +++++++++++++++++++++++++-------
 central_module/runners/gesture_runner.py               | 236 +++++++++++++++++++++++++++++++++++++++++++++-------
 central_module/runners/gesture_stub.py                 |  53 ++++++++++++
 central_module/runners/speech_runner.py                |  84 +++++++++++++++----
 central_module/supervisor.py                           |   6 +-
 docs/ipc-protocol.md                                   | 114 +++++++++++++++++++++++++
 speech_module/config.py                                |   3 +
 speech_module/speech_manager.py                        |  64 +++++++++-----
 tests/central/test_ipc_protocol.py                     | 194 +++++++++++++++++++++++++++++++++++++++++++
 tests/central/test_learning.py                         |  13 +--
 tests/central/test_manager.py                          | 134 ++++++++++++++++++++++++++----
 tests/central/test_runners_and_cli.py                  | 258 +++++++++++++++++++++++++++++++++++++++++++++++++++------
 tests/central/test_supervisor.py                       |  32 ++++++-
 {Mousegesture => tests/gesture}/test_gesture_events.py |  17 ++--
 tests/helpers/fakes.py                                 |  16 ++++
 tests/speech/test_speech_manager.py                    |  38 +++++----
 24 files changed, 1336 insertions(+), 217 deletions(-)
```

</details>

---
_Generated by `scripts/ci/pr_report.py` — see docs/ci-cd.md._
