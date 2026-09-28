# PR #11 — Gesture dev

> ⚠️ **Mergeable — needs reviewer attention** · 27 files · +9235 / −164 · 9 commit(s)

- Dependency pins changed
- Calibrated constants changed
- changed-line coverage 39.6% < 60%
- 6 new advisory lint finding(s)

| | |
| --- | --- |
| Pull request | https://github.com/CHANDU976/AI_Multimodal_Device_Control_System/pull/11 |
| Author | BangerAtifAhmed |
| Branch | `gesture_dev` → `main` |
| Base / head | `fa83515f05` / `8c2774ac63` |
| Workflow run | https://github.com/CHANDU976/AI_Multimodal_Device_Control_System/actions/runs/36424617885 |
| Generated | 2026-09-28T12:54:30Z (push) |

## Checks

| Check | Result | Details |
| --- | --- | --- |
| Lint (blocking rules) | ✅ success | 0 blocking finding(s) · advisory: 23 in changed files, 6 on added lines |
| Repository hygiene | ✅ success | 0 error(s), 0 warning(s) |
| Tests (all platforms) | ✅ success | `tests-macos-latest`: 1843 passed, 0 failed, 36 skipped, 4 xfail · 19.4 s<br>`tests-ubuntu-latest`: 1843 passed, 0 failed, 36 skipped, 4 xfail · 22.9 s<br>`tests-windows-latest`: 1876 passed, 0 failed, 3 skipped, 4 xfail · 49.7 s |
| NLU model tests (torch) | ✅ success | `ml`: 13 passed, 0 failed, 13 skipped, 0 xfail · 9.7 s |
| Dependency resolution | ✅ success |  |
| Coverage of added lines | ⚠️ | 39.6% of 1888 executable added line(s) (target 60%) |

## Needs attention

- ⚠️ **Dependency pins changed** — `Mousegesture/requirements.txt`, `requirements.txt`  
  The `deps` job resolves them on Linux and Windows. Mind the documented constraints: Python ≤ 3.11 and numpy < 2 (DeepFilterNet), CPU torch builds.
- ⚠️ **Calibrated constants changed** — `Mousegesture/gesture_config.json`  
  `Mousegesture/gesture_config.json` (gesture bindings and cursor settings). The project rule is that every constant carries the measurement that justifies it — check the comments were updated with the new numbers.
- ℹ️ **Needs a hands-on check CI cannot do** — `Mousegesture/app.py`, `Mousegesture/hand_cursor_2.py`  
  CI fakes the hardware boundary for: `app.py` → the Gesture Studio GUI; `hand_cursor_2.py` → camera + MediaPipe runtime with a webcam.
- ℹ️ **Large change set (9399 lines)**  
  Consider splitting it: smaller PRs review faster and bisect cleanly.

## Changes by module

| Module | Files | + | − | Components |
| --- | ---: | ---: | ---: | --- |
| Gesture module | 14 | 4598 | 136 | `model` ×3, `adaptive_timing`, `studio-gui`, `calibration_ui`, `cameras`, `config_lifecycle_probe`, `cursor_stability_probe`, `config`, … (+4) |
| Tests | 10 | 4507 | 7 | `gesture` ×10 |
| Dependency pins | 2 | 78 | 0 | `Mousegesture/requirements.txt`, `requirements.txt` |
| Documentation | 1 | 52 | 21 | `gesture` |

## Coverage

### Lines added by this change set

**39.6%** of 1888 executable added line(s) are exercised by the tests.

| File | Covered | Uncovered added lines |
| --- | ---: | --- |
| `Mousegesture/adaptive_timing.py` | 164/238 (68.9%) | 112, 192-193, 198, 200, 238, 259, 331, 346, 366-370, 407, 410, 453, 467, 479, 501-506, 508-509, 514, 518, 522, 525-527, 531, 535-542, 548-554, 556-560, 562-563, 565-566, 568, 570-576, 578-580, 582-585, 588 |
| `Mousegesture/app.py` | 71/424 (16.7%) | 181-182, 210-213, 594, 612-615, 620-623, 628, 645, 648, 894-896, 899-900, 903-904, 906, 913-914, 917-918, 922-923, 926-927, 1070-1072, 1074-1075, 1077-1078, 1080-1082, 1086-1087, 1399-1400, 1402, 1411, 1414, 1418-1419, 1545-1549, 1559-1563, 1569-1570, 1572, 1683, 1691-1692, 1694, 1696-1697, 1746-1749, 1752-1754, 1758-1760, 1762-1763, 1801, 1806-1811, 1813, 1815-1816, 1838-1843, 1846, 1848-1852, 1854-1861, 1863-1864, 1870-1872, 1874-1879, 1882-1883, 2033, 2080-2085, 2110-2111, 2113, 2184-2185, 2189-2190, 2192-2195, 2198-2201, 2203-2213, 2272, 2426, 2428, 2430, 2432, 2450, 2521, 2523-2527, 2531, 2545-2549, 2552, 2554, 2557-2560, 2562-2564, 2599, 2807, 2809, 2822-2823, 2835, 2842-2843, 2846-2847, 2850-2852, 2854-2855, 2857, 2864, 2866-2867, 2870, 2929, 2932-2937, 2947-2954, 2959-2961, 2963-2970, 2972-2973, 2976-2988, 2991-2995, 2997-3000, 3003-3004, 3006-3007, 3009-3011, 3014, 3017-3022, 3031-3037, 3040-3044, 3054-3063, 3066, 3071-3072, 3075-3076, 3080-3083, 3087, 3095-3098, 3102-3108, 3112-3116, 3130-3132, 3137-3138, 3147-3148, 3178, 3191, 3204-3205, 3207-3217, 3222, 3229, 3317-3319, 3322-3326, 3627, 3777 |
| `Mousegesture/calibration_ui.py` | 0/278 (0.0%) | 31, 33-34, 36-37, 41, 43-50, 55, 58, 60, 65-66, 69, 76-82, 85, 90, 92, 97-101, 103, 108-113, 116, 121-122, 124-125, 130, 132, 134, 137-138, 141, 143, 149, 151-152, 157, 166-167, 170-172, 175-176, 178, 180, 182, 186, 192, 194, 198, 200, 202, 204, 207, 209, 216-222, 226, 228, 230-233, 235, 238, 240-243, 245-250, 252, 254-255, 259-264, 266, 268-277, 279, 281-288, 293-294, 296, 299-304, 306, 308-316, 318, 320-324, 328, 336-341, 344, 346, 348-351, 353-357, 359, 361-366, 368-375, 377, 379-389, 391, 398-401, 405, 410-423, 425, 428-433, 435-440, 445-446, 448-451, 456-458, 462-463, 466-467, 471-474, 476-481, 483-484, 486, 488, 490-492, 494, 497-509 |
| `Mousegesture/cameras.py` | 0/61 (0.0%) | 21, 23-24, 26, 31, 35, 39, 42, 44-49, 51-52, 54-57, 59, 65, 72-75, 81-83, 86, 92-100, 106-107, 109-113, 116, 124-132, 137-141 |
| `Mousegesture/config_lifecycle_probe.py` | 0/143 (0.0%) | 13, 15-19, 21-22, 26, 28, 30-31, 34-37, 39, 42, 44, 55-56, 59, 66-67, 72-76, 79, 81, 84-88, 90-92, 95-96, 102-103, 105-106, 108-110, 112, 115, 117-120, 122, 124, 130-132, 135-137, 140-141, 144-145, 148, 152-153, 156-157, 160, 163-164, 170-173, 178-183, 185-186, 189-197, 199, 201, 206-210, 212, 214, 217-220, 224, 226, 230, 232, 234, 236, 238, 243, 249-251, 254-257, 260-263, 268, 270, 274-275, 279-281, 283-284, 286, 288-293 |
| `Mousegesture/cursor_stability_probe.py` | 130/178 (73.0%) | 190, 195, 205, 235-242, 246-248, 250-253, 258-263, 269-270, 277, 281, 283, 286-290, 294-296, 299, 303-307, 312-315, 317 |
| `Mousegesture/gesture_fsm.py` | 136/147 (92.5%) | 1412, 1414, 1421, 1811, 1837-1838, 1842, 1849, 1877, 1934-1935 |
| `Mousegesture/hand_cursor_2.py` | 246/342 (71.9%) | 3877, 4148-4150, 4153, 4182-4183, 4226, 4340-4343, 4345, 4349-4353, 4356, 4358-4359, 4362-4372, 4379-4380, 4382-4384, 4386, 4739, 4936, 4983, 5001-5002, 5013, 5055-5058, 5170-5174, 5177-5178, 5185-5188, 5193-5196, 5227-5228, 5233, 5261-5262, 5281-5284, 5288-5289, 5291-5292, 5295-5296, 5325, 5334-5340, 5350-5351, 5369, 5425, 5686, 5688, 5691-5692, 5708-5709 |
| `Mousegesture/transition_probe.py` | 0/77 (0.0%) | 23, 25, 27, 29, 32, 36, 39, 46, 53-54, 57, 60, 62, 64-67, 69-71, 74-77, 80, 83-89, 91-92, 94, 97, 100, 146, 148-150, 152, 160, 162-167, 170, 173-176, 178, 180-189, 193, 195-197, 199-204, 207-208 |

### Whole modules

| Module | Files | Statements | Covered |
| --- | ---: | ---: | ---: |
| Central module (orchestrator + NLU) | 28 | 1862 | 94.1% |
| Gesture module | 12 | 6906 | 40.2% |
| Speech module | 13 | 1512 | 91.3% |

## Lint findings in changed files

| Location | Rule | Message | New |
| --- | --- | --- | :---: |
| `Mousegesture/app.py:55` | F401 | `gesture_fsm.DOUBLE_CLICK` imported but unused |  |
| `Mousegesture/app.py:57` | F401 | `gesture_fsm.MOUSE_ACTIONS` imported but unused |  |
| `Mousegesture/app.py:60` | F401 | `gesture_fsm.SENSITIVITY_STEP` imported but unused |  |
| `Mousegesture/app.py:61` | F401 | `gesture_fsm.save_config` imported but unused |  |
| `Mousegesture/app.py:1714` | F841 | Local variable `poses` is assigned to but never used |  |
| `Mousegesture/app.py:2633` | F841 | Local variable `poses` is assigned to but never used |  |
| `Mousegesture/hand_cursor_2.py:100` | F401 | `monitors.monitor_labels` imported but unused |  |
| `Mousegesture/hand_cursor_2.py:101` | F401 | `monitors.monitor_union` imported but unused |  |
| `Mousegesture/hand_cursor_2.py:103` | F401 | `gesture_fsm.HAND_LEFT` imported but unused | 🆕 |
| `Mousegesture/hand_cursor_2.py:103` | F401 | `gesture_fsm.HAND_RIGHT` imported but unused | 🆕 |
| `Mousegesture/hand_cursor_2.py:1985` | F541 | f-string without any placeholders |  |
| `Mousegesture/hand_cursor_2.py:1986` | F541 | f-string without any placeholders |  |
| `Mousegesture/hand_cursor_2.py:2007` | B904 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |  |
| `Mousegesture/hand_cursor_2.py:5361` | F841 | Local variable `prev_x` is assigned to but never used |  |
| `Mousegesture/hand_cursor_2.py:5362` | F841 | Local variable `prev_y` is assigned to but never used |  |
| `Mousegesture/hand_cursor_2.py:5424` | F841 | Local variable `thumb_out` is assigned to but never used |  |
| `Mousegesture/model/convert.py:42` | W292 | No newline at end of file |  |
| `Mousegesture/model/onxxmodel.py:64` | W292 | No newline at end of file |  |
| `Mousegesture/model/ptmodel.py:73` | W292 | No newline at end of file |  |
| `tests/gesture/test_action_dispatch.py:627` | F841 | Local variable `high_speed_count` is assigned to but never used | 🆕 |
| `tests/gesture/test_preview_toggle.py:180` | B023 | Function definition does not bind loop variable `scheduled` | 🆕 |
| `tests/gesture/test_swipe_actions.py:24` | F401 | `gesture_fsm.save_config` imported but unused | 🆕 |
| `tests/gesture/test_swipe_actions.py:163` | F841 | Local variable `first` is assigned to but never used | 🆕 |

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
| `306170d` | CHANDAN B L | 2026-09-28 | Fix the flaky rule-id uniqueness test |
| `e3b7012` | CHANDAN B L | 2026-09-28 | Pin the gesture requirements to one set with the speech stack |
| `d6158b0` | CHANDAN B L | 2026-09-28 | Update gesture output tests for accelerating scroll and double-click |
| `e11ae36` | CHANDAN B L | 2026-09-28 | Classify the swipe actions as keyboard macros only |
| `6d96686` | CHANDAN B L | 2026-09-28 | Move the Mousegesture tests into tests/gesture so CI runs them |
| `d42e401` | CHANDAN B L | 2026-09-28 | Restore the model files deleted again in "version 2" |
| `a8213e6` | BangerAtifAhmed | 2026-09-28 | version 2 |
| `94b6452` | BangerAtifAhmed | 2026-09-24 | Restore model and .venv files accidentally deleted in version 1 |
| `c2b5608` | BangerAtifAhmed | 2026-09-24 | version 1 |

<details><summary>All 27 changed project files</summary>

| Status | File | Module | + | − |
| :---: | --- | --- | ---: | ---: |
| M | `Mousegesture/README.md` | docs/gesture | 52 | 21 |
| A | `Mousegesture/adaptive_timing.py` | gesture/adaptive_timing | 596 | 0 |
| M | `Mousegesture/app.py` | gesture/studio-gui | 849 | 64 |
| A | `Mousegesture/calibration_ui.py` | gesture/calibration_ui | 509 | 0 |
| A | `Mousegesture/cameras.py` | gesture/cameras | 141 | 0 |
| A | `Mousegesture/config_lifecycle_probe.py` | gesture/config_lifecycle_probe | 297 | 0 |
| A | `Mousegesture/cursor_stability_probe.py` | gesture/cursor_stability_probe | 321 | 0 |
| M | `Mousegesture/gesture_config.json` | gesture/config | 165 | 17 |
| M | `Mousegesture/gesture_fsm.py` | gesture/fsm | 431 | 10 |
| M | `Mousegesture/hand_cursor_2.py` | gesture/engine | 1042 | 36 |
| M | `Mousegesture/model/convert.py` | gesture/model | 5 | 1 |
| M | `Mousegesture/model/onxxmodel.py` | gesture/model | 5 | 1 |
| M | `Mousegesture/model/ptmodel.py` | gesture/model | 10 | 2 |
| A | `Mousegesture/requirements.txt` | dependencies/Mousegesture/requirements.txt | 73 | 0 |
| M | `Mousegesture/test.py` | gesture/test | 15 | 5 |
| A | `Mousegesture/transition_probe.py` | gesture/transition_probe | 212 | 0 |
| M | `requirements.txt` | dependencies/requirements.txt | 5 | 0 |
| A | `tests/gesture/test_action_dispatch.py` | tests/gesture | 688 | 0 |
| A | `tests/gesture/test_config_lifecycle.py` | tests/gesture | 967 | 0 |
| A | `tests/gesture/test_cursor_clutch.py` | tests/gesture | 219 | 0 |
| A | `tests/gesture/test_cursor_control.py` | tests/gesture | 402 | 0 |
| A | `tests/gesture/test_cursor_stability.py` | tests/gesture | 274 | 0 |
| A | `tests/gesture/test_gesture_fsm_transitions.py` | tests/gesture | 862 | 0 |
| M | `tests/gesture/test_gesture_output.py` | tests/gesture | 32 | 7 |
| A | `tests/gesture/test_preview_toggle.py` | tests/gesture | 230 | 0 |
| A | `tests/gesture/test_static_hold.py` | tests/gesture | 568 | 0 |
| A | `tests/gesture/test_swipe_actions.py` | tests/gesture | 265 | 0 |

</details>

<details><summary>git diff --stat</summary>

```text
 Mousegesture/README.md                        |   73 +++--
 Mousegesture/adaptive_timing.py               |  596 ++++++++++++++++++++++++++++++++++++
 Mousegesture/app.py                           |  913 +++++++++++++++++++++++++++++++++++++++++++++++++++----
 Mousegesture/calibration_ui.py                |  509 +++++++++++++++++++++++++++++++
 Mousegesture/cameras.py                       |  141 +++++++++
 Mousegesture/config_lifecycle_probe.py        |  297 ++++++++++++++++++
 Mousegesture/cursor_stability_probe.py        |  321 ++++++++++++++++++++
 Mousegesture/gesture_config.json              |  182 +++++++++--
 Mousegesture/gesture_fsm.py                   |  441 ++++++++++++++++++++++++++-
 Mousegesture/hand_cursor_2.py                 | 1078 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++---
 Mousegesture/model/convert.py                 |    6 +-
 Mousegesture/model/onxxmodel.py               |    6 +-
 Mousegesture/model/ptmodel.py                 |   12 +-
 Mousegesture/requirements.txt                 |   73 +++++
 Mousegesture/test.py                          |   20 +-
 Mousegesture/transition_probe.py              |  212 +++++++++++++
 requirements.txt                              |    5 +
 tests/gesture/test_action_dispatch.py         |  688 +++++++++++++++++++++++++++++++++++++++++
 tests/gesture/test_config_lifecycle.py        |  967 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/gesture/test_cursor_clutch.py           |  219 ++++++++++++++
 tests/gesture/test_cursor_control.py          |  402 ++++++++++++++++++++++++
 tests/gesture/test_cursor_stability.py        |  274 +++++++++++++++++
 tests/gesture/test_gesture_fsm_transitions.py |  862 ++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/gesture/test_gesture_output.py          |   39 ++-
 tests/gesture/test_preview_toggle.py          |  230 ++++++++++++++
 tests/gesture/test_static_hold.py             |  568 ++++++++++++++++++++++++++++++++++
 tests/gesture/test_swipe_actions.py           |  265 ++++++++++++++++
 27 files changed, 9235 insertions(+), 164 deletions(-)
```

</details>

---
_Generated by `scripts/ci/pr_report.py` — see docs/ci-cd.md._
