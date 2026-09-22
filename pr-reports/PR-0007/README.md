# PR #7 — Add CI/CD pipeline, test suite and PR reports

> ⚠️ **Mergeable — needs reviewer attention** · 67 files · +10782 / −42 · 4 commit(s)

- Intent catalog changed without a retrained NLU model
- 2 new advisory lint finding(s)

| | |
| --- | --- |
| Pull request | https://github.com/CHANDU976/AI_Multimodal_Device_Control_System/pull/7 |
| Author | CHANDU976 |
| Branch | `vv2` → `main` |
| Base / head | `852d94487f` / `f2698a14d0` |
| Workflow run | https://github.com/CHANDU976/AI_Multimodal_Device_Control_System/actions/runs/35738690806 |
| Generated | 2026-09-22T14:14:56Z (push) |

## Checks

| Check | Result | Details |
| --- | --- | --- |
| Lint (blocking rules) | ✅ success | 0 blocking finding(s) · advisory: 3 in changed files, 2 on added lines |
| Repository hygiene | ✅ success | 0 error(s), 0 warning(s) |
| Tests (all platforms) | ✅ success | `tests-macos-latest`: 1525 passed, 0 failed, 35 skipped, 4 xfail · 16.5 s<br>`tests-ubuntu-latest`: 1525 passed, 0 failed, 35 skipped, 4 xfail · 19.8 s<br>`tests-windows-latest`: 1557 passed, 0 failed, 3 skipped, 4 xfail · 29.0 s |
| NLU model tests (torch) | ✅ success | `ml`: 13 passed, 0 failed, 13 skipped, 0 xfail · 9.7 s |
| Dependency resolution | ✅ success |  |
| Coverage of added lines | ✅ | 100.0% of 2 executable added line(s) (target 60%) |

## Needs attention

- ⚠️ **Intent catalog changed without a retrained NLU model** — `central_module/nlp/intents.py`  
  Catalog patterns feed the synthetic training set. Added/removed/renamed intents fail the label-space contract test; pattern-only changes do not — retrain with `python -m central_module.nlp.nlu.train` if paraphrase coverage matters.
- ℹ️ **Needs a hands-on check CI cannot do** — `speech_module/audio_stream.py`  
  CI fakes the hardware boundary for: `audio_stream.py` → live microphone capture.
- ℹ️ **CI/tooling changed** — `.github/CODEOWNERS`, `.github/dependabot.yml`, `.github/labeler.yml`, `.github/pull_request_template.md`, `.github/workflows/ci.yml` (+10 more)  
  The pipeline validating this change is itself modified — review it as code.
- ℹ️ **Large change set (10800 lines)**  
  Consider splitting it: smaller PRs review faster and bisect cleanly.

## Changes by module

| Module | Files | + | − | Components |
| --- | ---: | ---: | ---: | --- |
| Central module (orchestrator + NLU) | 3 | 40 | 20 | `nlp` ×2, `core` |
| Speech module | 2 | 8 | 2 | `audio_stream`, `vosk_engine` |
| Gesture module | 1 | 0 | 17 | `.claude` |
| Tests | 42 | 8014 | 0 | `central` ×15, `speech` ×10, `gesture` ×7, `helpers` ×5, `repo` ×3, `root` ×2 |
| CI/CD & tooling | 15 | 2391 | 0 | `scripts` ×5, `.github` ×4, `workflows` ×3, `.gitignore`, `pyproject.toml`, `requirements-test.txt` |
| Documentation | 2 | 305 | 3 | `README.md`, `docs` |
| PR reports | 2 | 24 | 0 |  |

## Coverage

### Lines added by this change set

**100.0%** of 2 executable added line(s) are exercised by the tests.

| File | Covered | Uncovered added lines |
| --- | ---: | --- |
| `speech_module/audio_stream.py` | 1/1 (100.0%) | — |
| `speech_module/vosk_engine.py` | 1/1 (100.0%) | — |

### Whole modules

| Module | Files | Statements | Covered |
| --- | ---: | ---: | ---: |
| Central module (orchestrator + NLU) | 28 | 1862 | 94.1% |
| Gesture module | 6 | 5083 | 38.0% |
| Speech module | 13 | 1395 | 90.3% |

## Lint findings in changed files

| Location | Rule | Message | New |
| --- | --- | --- | :---: |
| `speech_module/audio_stream.py:16` | F401 | `numpy` imported but unused |  |
| `tests/repo/test_ci_scripts.py:285` | E501 | Line too long (119 > 100) | 🆕 |
| `tests/test_central_module.py:114` | W292 | No newline at end of file | 🆕 |

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
| `83b3dd9` | CHANDAN B L | 2026-09-22 | Address Copilot review: token scope, report verdict, test command |
| `0afc4e4` | CHANDAN B L | 2026-09-22 | Fix CI failures from the first PR run |
| `34d2fa4` | CHANDAN B L | 2026-09-22 | Ignore .claude/ settings in every directory |
| `a15ffb0` | CHANDAN B L | 2026-09-22 | Add CI, tests, and intent updates |

<details><summary>All 67 changed project files</summary>

| Status | File | Module | + | − |
| :---: | --- | --- | ---: | ---: |
| A | `.github/CODEOWNERS` | ci/.github | 16 | 0 |
| A | `.github/dependabot.yml` | ci/.github | 12 | 0 |
| A | `.github/labeler.yml` | ci/.github | 45 | 0 |
| A | `.github/pull_request_template.md` | ci/.github | 29 | 0 |
| A | `.github/workflows/ci.yml` | ci/workflows | 463 | 0 |
| A | `.github/workflows/labeler.yml` | ci/workflows | 23 | 0 |
| A | `.github/workflows/release.yml` | ci/workflows | 48 | 0 |
| A | `.gitignore` | ci/.gitignore | 26 | 0 |
| D | `Mousegesture/.claude/settings.json` | gesture/.claude | 0 | 17 |
| M | `README.md` | docs/README.md | 20 | 3 |
| M | `central_module/README.md` | central/core | 28 | 12 |
| M | `central_module/nlp/intent_engine.py` | central/nlp | 5 | 6 |
| M | `central_module/nlp/intents.py` | central/nlp | 7 | 2 |
| A | `docs/ci-cd.md` | docs/docs | 285 | 0 |
| A | `pr-reports/README.md` | reports/ | 20 | 0 |
| A | `pr-reports/index.json` | reports/ | 4 | 0 |
| A | `pyproject.toml` | ci/pyproject.toml | 70 | 0 |
| A | `requirements-test.txt` | ci/requirements-test.txt | 28 | 0 |
| A | `scripts/ci/archive_report.sh` | ci/scripts | 47 | 0 |
| A | `scripts/ci/ci_common.py` | ci/scripts | 341 | 0 |
| A | `scripts/ci/pr_report.py` | ci/scripts | 902 | 0 |
| A | `scripts/ci/release.py` | ci/scripts | 168 | 0 |
| A | `scripts/ci/repo_checks.py` | ci/scripts | 173 | 0 |
| M | `speech_module/audio_stream.py` | speech/audio_stream | 4 | 1 |
| M | `speech_module/vosk_engine.py` | speech/vosk_engine | 4 | 1 |
| A | `tests/central/test_executor.py` | tests/central | 227 | 0 |
| A | `tests/central/test_intent_catalog.py` | tests/central | 106 | 0 |
| A | `tests/central/test_intent_engine.py` | tests/central | 269 | 0 |
| A | `tests/central/test_ipc.py` | tests/central | 110 | 0 |
| A | `tests/central/test_learning.py` | tests/central | 213 | 0 |
| A | `tests/central/test_linux_adapter.py` | tests/central | 214 | 0 |
| A | `tests/central/test_macos_adapter.py` | tests/central | 173 | 0 |
| A | `tests/central/test_manager.py` | tests/central | 373 | 0 |
| A | `tests/central/test_nlu.py` | tests/central | 265 | 0 |
| A | `tests/central/test_nlu_ml.py` | tests/central | 224 | 0 |
| A | `tests/central/test_normalize.py` | tests/central | 90 | 0 |
| A | `tests/central/test_os_adapter.py` | tests/central | 120 | 0 |
| A | `tests/central/test_runners_and_cli.py` | tests/central | 189 | 0 |
| A | `tests/central/test_supervisor.py` | tests/central | 183 | 0 |
| A | `tests/central/test_windows_adapter.py` | tests/central | 196 | 0 |
| A | `tests/conftest.py` | tests/root | 95 | 0 |
| A | `tests/gesture/conftest.py` | tests/gesture | 34 | 0 |
| A | `tests/gesture/test_engine_geometry.py` | tests/gesture | 298 | 0 |
| A | `tests/gesture/test_engine_screen.py` | tests/gesture | 256 | 0 |
| A | `tests/gesture/test_gesture_contracts.py` | tests/gesture | 185 | 0 |
| A | `tests/gesture/test_gesture_fsm.py` | tests/gesture | 556 | 0 |
| A | `tests/gesture/test_gesture_output.py` | tests/gesture | 295 | 0 |
| A | `tests/gesture/test_monitors.py` | tests/gesture | 127 | 0 |
| A | `tests/helpers/__init__.py` | tests/helpers | 2 | 0 |
| A | `tests/helpers/audio.py` | tests/helpers | 41 | 0 |
| A | `tests/helpers/fakes.py` | tests/helpers | 388 | 0 |
| A | `tests/helpers/hands.py` | tests/helpers | 86 | 0 |
| A | `tests/helpers/paths.py` | tests/helpers | 39 | 0 |
| A | `tests/repo/test_ci_scripts.py` | tests/repo | 513 | 0 |
| A | `tests/repo/test_fetch_deepfilternet_model.py` | tests/repo | 109 | 0 |
| A | `tests/repo/test_release_and_workflows.py` | tests/repo | 268 | 0 |
| A | `tests/speech/conftest.py` | tests/speech | 22 | 0 |
| A | `tests/speech/test_audio.py` | tests/speech | 256 | 0 |
| A | `tests/speech/test_calibrate.py` | tests/speech | 87 | 0 |
| A | `tests/speech/test_denoiser.py` | tests/speech | 194 | 0 |
| A | `tests/speech/test_routing.py` | tests/speech | 179 | 0 |
| A | `tests/speech/test_speech_config.py` | tests/speech | 86 | 0 |
| A | `tests/speech/test_speech_manager.py` | tests/speech | 275 | 0 |
| A | `tests/speech/test_vad.py` | tests/speech | 194 | 0 |
| A | `tests/speech/test_vosk_engine.py` | tests/speech | 145 | 0 |
| A | `tests/speech/test_whisper_engine.py` | tests/speech | 218 | 0 |
| A | `tests/test_central_module.py` | tests/root | 114 | 0 |

</details>

<details><summary>git diff --stat</summary>

```text
 .github/CODEOWNERS                           |  16 ++
 .github/dependabot.yml                       |  12 +
 .github/labeler.yml                          |  45 ++++
 .github/pull_request_template.md             |  29 +++
 .github/workflows/ci.yml                     | 463 ++++++++++++++++++++++++++++++++++
 .github/workflows/labeler.yml                |  23 ++
 .github/workflows/release.yml                |  48 ++++
 .gitignore                                   |  26 ++
 Mousegesture/.claude/settings.json           |  17 --
 README.md                                    |  23 +-
 central_module/README.md                     |  40 ++-
 central_module/nlp/intent_engine.py          |  11 +-
 central_module/nlp/intents.py                |   9 +-
 docs/ci-cd.md                                | 285 +++++++++++++++++++++
 pr-reports/README.md                         |  20 ++
 pr-reports/index.json                        |   4 +
 pyproject.toml                               |  70 ++++++
 requirements-test.txt                        |  28 +++
 scripts/ci/archive_report.sh                 |  47 ++++
 scripts/ci/ci_common.py                      | 341 +++++++++++++++++++++++++
 scripts/ci/pr_report.py                      | 902 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 scripts/ci/release.py                        | 168 +++++++++++++
 scripts/ci/repo_checks.py                    | 173 +++++++++++++
 speech_module/audio_stream.py                |   5 +-
 speech_module/vosk_engine.py                 |   5 +-
 tests/central/test_executor.py               | 227 +++++++++++++++++
 tests/central/test_intent_catalog.py         | 106 ++++++++
 tests/central/test_intent_engine.py          | 269 ++++++++++++++++++++
 tests/central/test_ipc.py                    | 110 +++++++++
 tests/central/test_learning.py               | 213 ++++++++++++++++
 tests/central/test_linux_adapter.py          | 214 ++++++++++++++++
 tests/central/test_macos_adapter.py          | 173 +++++++++++++
 tests/central/test_manager.py                | 373 ++++++++++++++++++++++++++++
 tests/central/test_nlu.py                    | 265 ++++++++++++++++++++
 tests/central/test_nlu_ml.py                 | 224 +++++++++++++++++
 tests/central/test_normalize.py              |  90 +++++++
 tests/central/test_os_adapter.py             | 120 +++++++++
 tests/central/test_runners_and_cli.py        | 189 ++++++++++++++
 tests/central/test_supervisor.py             | 183 ++++++++++++++
 tests/central/test_windows_adapter.py        | 196 +++++++++++++++
 tests/conftest.py                            |  95 +++++++
 tests/gesture/conftest.py                    |  34 +++
 tests/gesture/test_engine_geometry.py        | 298 ++++++++++++++++++++++
 tests/gesture/test_engine_screen.py          | 256 +++++++++++++++++++
 tests/gesture/test_gesture_contracts.py      | 185 ++++++++++++++
 tests/gesture/test_gesture_fsm.py            | 556 +++++++++++++++++++++++++++++++++++++++++
 tests/gesture/test_gesture_output.py         | 295 ++++++++++++++++++++++
 tests/gesture/test_monitors.py               | 127 ++++++++++
 tests/helpers/__init__.py                    |   2 +
 tests/helpers/audio.py                       |  41 ++++
 tests/helpers/fakes.py                       | 388 +++++++++++++++++++++++++++++
 tests/helpers/hands.py                       |  86 +++++++
 tests/helpers/paths.py                       |  39 +++
 tests/repo/test_ci_scripts.py                | 513 ++++++++++++++++++++++++++++++++++++++
 tests/repo/test_fetch_deepfilternet_model.py | 109 ++++++++
 tests/repo/test_release_and_workflows.py     | 268 ++++++++++++++++++++
 tests/speech/conftest.py                     |  22 ++
 tests/speech/test_audio.py                   | 256 +++++++++++++++++++
 tests/speech/test_calibrate.py               |  87 +++++++
 tests/speech/test_denoiser.py                | 194 +++++++++++++++
 tests/speech/test_routing.py                 | 179 ++++++++++++++
 tests/speech/test_speech_config.py           |  86 +++++++
 tests/speech/test_speech_manager.py          | 275 +++++++++++++++++++++
 tests/speech/test_vad.py                     | 194 +++++++++++++++
 tests/speech/test_vosk_engine.py             | 145 +++++++++++
 tests/speech/test_whisper_engine.py          | 218 ++++++++++++++++
 tests/test_central_module.py                 | 114 +++++++++
 67 files changed, 10782 insertions(+), 42 deletions(-)
```

</details>

---
_Generated by `scripts/ci/pr_report.py` — see docs/ci-cd.md._
