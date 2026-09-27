# PR #10 — Vv2

> ⚠️ **Mergeable — needs reviewer attention** · 30171 files · +833 / −5839095 · 3 commit(s)

- Calibrated constants changed
- Virtualenv / bytecode files in the diff

| | |
| --- | --- |
| Pull request | https://github.com/CHANDU976/AI_Multimodal_Device_Control_System/pull/10 |
| Author | CHANDU976 |
| Branch | `vv2` → `main` |
| Base / head | `57a369e903` / `9d659eb023` |
| Workflow run | https://github.com/CHANDU976/AI_Multimodal_Device_Control_System/actions/runs/36332153678 |
| Generated | 2026-09-27T16:10:48Z (push) |

## Checks

| Check | Result | Details |
| --- | --- | --- |
| Lint (blocking rules) | ✅ success | 0 blocking finding(s) · advisory: 5 in changed files, 0 on added lines |
| Repository hygiene | ✅ success | 0 error(s), 0 warning(s) |
| Tests (all platforms) | ✅ success | `tests-macos-latest`: 1558 passed, 0 failed, 36 skipped, 4 xfail · 14.9 s<br>`tests-ubuntu-latest`: 1558 passed, 0 failed, 36 skipped, 4 xfail · 19.1 s<br>`tests-windows-latest`: 1591 passed, 0 failed, 3 skipped, 4 xfail · 37.5 s |
| NLU model tests (torch) | ✅ success | `ml`: 13 passed, 0 failed, 13 skipped, 0 xfail · 7.3 s |
| Dependency resolution | ⏭️ skipped | no requirement files changed |
| Coverage of added lines | ✅ | 98.0% of 99 executable added line(s) (target 60%) |

## Needs attention

- ⚠️ **Calibrated constants changed** — `speech_module/config.py`  
  `speech_module/config.py` (speech routing / VAD / denoiser thresholds). The project rule is that every constant carries the measurement that justifies it — check the comments were updated with the new numbers.
- ⚠️ **Virtualenv / bytecode files in the diff** — `.venv/Lib/site-packages/DeepFilterLib-0.5.6.dist-info/DELVEWHEEL`, `.venv/Lib/site-packages/DeepFilterLib-0.5.6.dist-info/INSTALLER`, `.venv/Lib/site-packages/DeepFilterLib-0.5.6.dist-info/METADATA`, `.venv/Lib/site-packages/DeepFilterLib-0.5.6.dist-info/RECORD`, `.venv/Lib/site-packages/DeepFilterLib-0.5.6.dist-info/WHEEL` (+30149 more)  
  30154 file(s) under .venv/ or __pycache__/. They are machine-specific and bloat history; drop them from the PR (`git restore --staged .venv`).
- ℹ️ **Needs a hands-on check CI cannot do** — `speech_module/speech_manager.py`  
  CI fakes the hardware boundary for: `speech_manager.py` → the real-time speech loop with a microphone.
- ℹ️ **CI/tooling changed** — `.gitattributes`, `.github/workflows/ci.yml`, `.gitignore`  
  The pipeline validating this change is itself modified — review it as code.

## Changes by module

| Module | Files | + | − | Components |
| --- | ---: | ---: | ---: | --- |
| Central module (orchestrator + NLU) | 1 | 6 | 20 | `runners` |
| Speech module | 5 | 427 | 88 | `config`, `router`, `speech_manager`, `vosk_engine`, `whisper_engine` |
| Tests | 7 | 387 | 27 | `speech` ×5, `central`, `helpers` |
| CI/CD & tooling | 3 | 9 | 5 | `.gitattributes`, `workflows`, `.gitignore` |
| Documentation | 1 | 4 | 4 | `docs` |
| Committed virtualenv (.venv) | 30129 | 0 | 5838951 | `torch` ×14077, `sympy` ×3093, `scipy` ×2498, `sklearn` ×1708, `numba` ×1526, `numpy` ×1414, `networkx` ×1175, `pip` ×1012, … (+121) |
| Generated files (__pycache__, *.pyc) | 25 | 0 | 0 | `speech_module` ×24, `__pycache__` |

## Coverage

### Lines added by this change set

**98.0%** of 99 executable added line(s) are exercised by the tests.

| File | Covered | Uncovered added lines |
| --- | ---: | --- |
| `central_module/runners/speech_runner.py` | 1/1 (100.0%) | — |
| `speech_module/config.py` | 28/28 (100.0%) | — |
| `speech_module/router.py` | 28/28 (100.0%) | — |
| `speech_module/speech_manager.py` | 15/16 (93.8%) | 289 |
| `speech_module/vosk_engine.py` | 24/25 (96.0%) | 283 |
| `speech_module/whisper_engine.py` | 1/1 (100.0%) | — |

### Whole modules

| Module | Files | Statements | Covered |
| --- | ---: | ---: | ---: |
| Central module (orchestrator + NLU) | 28 | 1862 | 94.1% |
| Gesture module | 6 | 5083 | 38.0% |
| Speech module | 13 | 1512 | 91.3% |

## Lint findings in changed files

| Location | Rule | Message | New |
| --- | --- | --- | :---: |
| `speech_module/config.py:8` | E501 | Line too long (105 > 100) |  |
| `speech_module/whisper_engine.py:331` | B904 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |  |
| `speech_module/whisper_engine.py:381` | B904 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |  |
| `speech_module/whisper_engine.py:473` | W293 | Blank line contains whitespace |  |
| `speech_module/whisper_engine.py:485` | W293 | Blank line contains whitespace |  |

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
| `8756cece` | CHANDAN B L | 2026-09-27 | Stop tracking the virtualenv and bytecode caches |
| `5c9c0bc1` | CHANDAN B L | 2026-09-27 | Rebuild the VOSK command grammar as whole command phrases |
| `86a2baf3` | CHANDAN B L | 2026-09-27 | Address Copilot review of the speech IPC messages (PR #9) |

<details><summary>All 17 changed project files (+30154 summarised below)</summary>

| Status | File | Module | + | − |
| :---: | --- | --- | ---: | ---: |
| ∑ | **30129 files** — Committed virtualenv (.venv) (torch ×14077, sympy ×3093, scipy ×2498, sklearn ×1708, numba ×1526, numpy ×1414) | environment | 0 | 5838951 |
| ∑ | **25 files** — Generated files (__pycache__, *.pyc) (speech_module ×24, __pycache__ ×1) | generated | 0 | 0 |
| M | `.gitattributes` | ci/.gitattributes | 0 | 2 |
| M | `.github/workflows/ci.yml` | ci/workflows | 1 | 1 |
| M | `.gitignore` | ci/.gitignore | 8 | 2 |
| M | `central_module/runners/speech_runner.py` | central/runners | 6 | 20 |
| M | `docs/ci-cd.md` | docs/docs | 4 | 4 |
| M | `speech_module/config.py` | speech/config | 275 | 46 |
| M | `speech_module/router.py` | speech/router | 59 | 17 |
| M | `speech_module/speech_manager.py` | speech/speech_manager | 35 | 13 |
| M | `speech_module/vosk_engine.py` | speech/vosk_engine | 54 | 10 |
| M | `speech_module/whisper_engine.py` | speech/whisper_engine | 4 | 2 |
| M | `tests/central/test_runners_and_cli.py` | tests/central | 7 | 1 |
| M | `tests/helpers/fakes.py` | tests/helpers | 6 | 1 |
| M | `tests/speech/test_routing.py` | tests/speech | 58 | 10 |
| M | `tests/speech/test_speech_config.py` | tests/speech | 122 | 12 |
| M | `tests/speech/test_speech_manager.py` | tests/speech | 159 | 0 |
| M | `tests/speech/test_vosk_engine.py` | tests/speech | 27 | 3 |
| M | `tests/speech/test_whisper_engine.py` | tests/speech | 8 | 0 |

</details>

<details><summary>git diff --stat</summary>

```text
 .gitattributes                          |   2 -
 .github/workflows/ci.yml                |   2 +-
 .gitignore                              |  10 ++-
 __pycache__/classifier.cpython-313.pyc  | Bin 7759 -> 0 bytes
 central_module/runners/speech_runner.py |  26 ++----
 docs/ci-cd.md                           |   8 +-
 speech_module/config.py                 | 321 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-----------
 speech_module/router.py                 |  76 +++++++++++++----
 speech_module/speech_manager.py         |  48 ++++++++---
 speech_module/vosk_engine.py            |  64 ++++++++++++---
 speech_module/whisper_engine.py         |   6 +-
 tests/central/test_runners_and_cli.py   |   8 +-
 tests/helpers/fakes.py                  |   7 +-
 tests/speech/test_routing.py            |  68 +++++++++++++---
 tests/speech/test_speech_config.py      | 134 +++++++++++++++++++++++++++---
 tests/speech/test_speech_manager.py     | 159 ++++++++++++++++++++++++++++++++++++
 tests/speech/test_vosk_engine.py        |  30 ++++++-
 tests/speech/test_whisper_engine.py     |   8 ++
 18 files changed, 833 insertions(+), 144 deletions(-)
```

</details>

---
_Generated by `scripts/ci/pr_report.py` — see docs/ci-cd.md._
