# PR #9 — Implement speech IPC message generation

> ⚠️ **Mergeable — needs reviewer attention** · 5 files · +158 / −22 · 2 commit(s)

- Calibrated constants changed
- Source changed without test changes
- 5 new advisory lint finding(s)

| | |
| --- | --- |
| Pull request | https://github.com/CHANDU976/AI_Multimodal_Device_Control_System/pull/9 |
| Author | CHANDU976 |
| Branch | `feature/speech-ipc-messages` → `main` |
| Base / head | `ec8f061f90` / `b756262794` |
| Workflow run | https://github.com/CHANDU976/AI_Multimodal_Device_Control_System/actions/runs/36297784242 |
| Generated | 2026-09-27T05:40:09Z (push) |

## Checks

| Check | Result | Details |
| --- | --- | --- |
| Lint (blocking rules) | ✅ success | 0 blocking finding(s) · advisory: 11 in changed files, 5 on added lines |
| Repository hygiene | ✅ success | 0 error(s), 0 warning(s) |
| Tests (all platforms) | ✅ success | `tests-macos-latest`: 1525 passed, 0 failed, 35 skipped, 4 xfail · 18.5 s<br>`tests-ubuntu-latest`: 1525 passed, 0 failed, 35 skipped, 4 xfail · 20.2 s<br>`tests-windows-latest`: 1557 passed, 0 failed, 3 skipped, 4 xfail · 32.3 s |
| NLU model tests (torch) | ✅ success | `ml`: 13 passed, 0 failed, 13 skipped, 0 xfail · 9.8 s |
| Dependency resolution | ⏭️ skipped | no requirement files changed |
| Coverage of added lines | ✅ | 64.6% of 48 executable added line(s) (target 60%) |

## Needs attention

- ⚠️ **Calibrated constants changed** — `speech_module/config.py`  
  `speech_module/config.py` (speech routing / VAD / denoiser thresholds). The project rule is that every constant carries the measurement that justifies it — check the comments were updated with the new numbers.
- ⚠️ **Source changed without test changes** — `central_module/runners/speech_runner.py`, `speech_module/config.py`, `speech_module/speech_manager.py`, `speech_module/vosk_engine.py`, `speech_module/whisper_engine.py`  
  No file under tests/ was touched. Add or update tests, or say in the PR why none are needed.
- ℹ️ **Needs a hands-on check CI cannot do** — `speech_module/speech_manager.py`  
  CI fakes the hardware boundary for: `speech_manager.py` → the real-time speech loop with a microphone.

## Changes by module

| Module | Files | + | − | Components |
| --- | ---: | ---: | ---: | --- |
| Central module (orchestrator + NLU) | 1 | 17 | 5 | `runners` |
| Speech module | 4 | 141 | 17 | `config`, `speech_manager`, `vosk_engine`, `whisper_engine` |

## Coverage

### Lines added by this change set

**64.6%** of 48 executable added line(s) are exercised by the tests.

| File | Covered | Uncovered added lines |
| --- | ---: | --- |
| `central_module/runners/speech_runner.py` | 2/4 (50.0%) | 88-89 |
| `speech_module/config.py` | 1/1 (100.0%) | — |
| `speech_module/speech_manager.py` | 13/28 (46.4%) | 148, 158-160, 162-166, 168-169, 287-288, 449, 452 |
| `speech_module/vosk_engine.py` | 7/7 (100.0%) | — |
| `speech_module/whisper_engine.py` | 8/8 (100.0%) | — |

### Whole modules

| Module | Files | Statements | Covered |
| --- | ---: | ---: | ---: |
| Central module (orchestrator + NLU) | 28 | 1866 | 94.0% |
| Gesture module | 6 | 5083 | 38.0% |
| Speech module | 13 | 1434 | 89.5% |

## Lint findings in changed files

| Location | Rule | Message | New |
| --- | --- | --- | :---: |
| `central_module/runners/speech_runner.py:151` | W292 | No newline at end of file | 🆕 |
| `speech_module/config.py:6` | E501 | Line too long (105 > 100) |  |
| `speech_module/config.py:195` | B033 | Sets should not contain duplicate item `"word"` |  |
| `speech_module/config.py:319` | W292 | No newline at end of file | 🆕 |
| `speech_module/speech_manager.py:678` | W292 | No newline at end of file | 🆕 |
| `speech_module/vosk_engine.py:407` | W292 | No newline at end of file | 🆕 |
| `speech_module/whisper_engine.py:329` | B904 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |  |
| `speech_module/whisper_engine.py:379` | B904 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |  |
| `speech_module/whisper_engine.py:471` | W293 | Blank line contains whitespace |  |
| `speech_module/whisper_engine.py:483` | W293 | Blank line contains whitespace |  |
| `speech_module/whisper_engine.py:493` | W292 | No newline at end of file | 🆕 |

## Known issues pinned by tests (xfail)

Documented defects the suite tracks; each flips to a failure once fixed, so the marker gets removed.

- `tests.central.test_intent_engine::test_every_pattern_is_reachable_by_exact_match[press_key:press the {key} key]` — command_normalize drops 'the', so a non-text pattern containing it never matches
- `tests.central.test_intent_engine::test_every_pattern_is_reachable_by_exact_match[double_click:double click]` — command_normalize turns 'double' into '2' before matching
- `tests.central.test_intent_engine::test_every_pattern_is_reachable_by_exact_match[double_click:click twice]` — command_normalize turns 'twice' into '2' before matching
- `tests.repo.test_fetch_deepfilternet_model::test_extract_blocks_sibling_prefix_traversal` — zip-slip: the guard compares string prefixes, so '../DFN3evil/x' resolves to a SIBLING directory whose path still starts with '.../DFN3'. Compare with Path.is_relative_to(dest.resolve()) instead.

<details><summary>Skipped tests by reason</summary>

| Reason | Count |
| --- | ---: |
| WindowsAdapter binds user32.dll at import | 62 |
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
| `22a8244` | sampadasshetty | 2026-09-26 | Fix CI: missing numpy import in vosk_engine.py, defensive to_ipc() fallback in speech_runner.py |
| `76e8782` | sampadasshetty | 2026-09-23 | Implement speech IPC message generation |

<details><summary>All 5 changed project files</summary>

| Status | File | Module | + | − |
| :---: | --- | --- | ---: | ---: |
| M | `central_module/runners/speech_runner.py` | central/runners | 17 | 5 |
| M | `speech_module/config.py` | speech/config | 17 | 0 |
| M | `speech_module/speech_manager.py` | speech/speech_manager | 88 | 7 |
| M | `speech_module/vosk_engine.py` | speech/vosk_engine | 13 | 8 |
| M | `speech_module/whisper_engine.py` | speech/whisper_engine | 23 | 2 |

</details>

<details><summary>git diff --stat</summary>

```text
 central_module/runners/speech_runner.py | 22 +++++++++++++----
 speech_module/config.py                 | 17 +++++++++++++
 speech_module/speech_manager.py         | 95 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++------
 speech_module/vosk_engine.py            | 21 +++++++++-------
 speech_module/whisper_engine.py         | 25 +++++++++++++++++--
 5 files changed, 158 insertions(+), 22 deletions(-)
```

</details>

---
_Generated by `scripts/ci/pr_report.py` — see docs/ci-cd.md._
