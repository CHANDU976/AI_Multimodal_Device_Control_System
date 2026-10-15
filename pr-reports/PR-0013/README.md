# PR #13 — feat: add gesture event reporting and queue sink integration

> ⚠️ **Mergeable — needs reviewer attention** · 5 files · +1965 / −12 · 1 commit(s)

- Source changed without test changes
- changed-line coverage 7.9% < 60%
- 1 new advisory lint finding(s)

| | |
| --- | --- |
| Pull request | https://github.com/CHANDU976/AI_Multimodal_Device_Control_System/pull/13 |
| Author | BangerAtifAhmed |
| Branch | `gesture_dev` → `main` |
| Base / head | `9ee0caf4b0` / `6f60a3da91` |
| Workflow run | https://github.com/CHANDU976/AI_Multimodal_Device_Control_System/actions/runs/38700405351 |
| Generated | 2026-10-15T07:43:16Z (push) |

## Checks

| Check | Result | Details |
| --- | --- | --- |
| Lint (blocking rules) | ✅ success | 0 blocking finding(s) · advisory: 17 in changed files, 1 on added lines |
| Repository hygiene | ✅ success | 0 error(s), 0 warning(s) |
| Tests (all platforms) | ✅ success | `tests-macos-latest`: 1843 passed, 0 failed, 36 skipped, 4 xfail · 18.5 s<br>`tests-ubuntu-latest`: 1843 passed, 0 failed, 36 skipped, 4 xfail · 22.9 s<br>`tests-windows-latest`: 1876 passed, 0 failed, 3 skipped, 4 xfail · 43.3 s |
| NLU model tests (torch) | ✅ success | `ml`: 13 passed, 0 failed, 13 skipped, 0 xfail · 10.5 s |
| Dependency resolution | ⏭️ skipped | no requirement files changed |
| Coverage of added lines | ⚠️ | 7.9% of 1044 executable added line(s) (target 60%) |

## Needs attention

- ⚠️ **Source changed without test changes** — `Mousegesture/app.py`, `Mousegesture/gesture_events.py`, `Mousegesture/hand_cursor_2.py`, `Mousegesture/test_gesture_events.py`  
  No file under tests/ was touched. Add or update tests, or say in the PR why none are needed.
- ℹ️ **Needs a hands-on check CI cannot do** — `Mousegesture/app.py`, `Mousegesture/hand_cursor_2.py`  
  CI fakes the hardware boundary for: `app.py` → the Gesture Studio GUI; `hand_cursor_2.py` → camera + MediaPipe runtime with a webcam.
- ℹ️ **Large change set (1977 lines)**  
  Consider splitting it: smaller PRs review faster and bisect cleanly.

## Changes by module

| Module | Files | + | − | Components |
| --- | ---: | ---: | ---: | --- |
| Gesture module | 4 | 1830 | 12 | `studio-gui`, `gesture_events`, `engine`, `test_gesture_events` |
| Documentation | 1 | 135 | 0 | `gesture` |

## Coverage

### Lines added by this change set

**7.9%** of 1044 executable added line(s) are exercised by the tests.

| File | Covered | Uncovered added lines |
| --- | ---: | --- |
| `Mousegesture/app.py` | 1/3 (33.3%) | 633, 3162 |
| `Mousegesture/gesture_events.py` | 68/232 (29.3%) | 119-132, 134-136, 142-143, 148-151, 155-161, 168-171, 185, 195, 208, 224, 238, 252, 262, 270, 291, 293-304, 306, 310, 342-346, 349-352, 354-358, 364, 368-369, 373, 382-393, 395, 401-406, 410-415, 425-429, 433-440, 444-453, 464-472, 474, 478-481, 489-492, 499-503, 506, 512-515, 518-523, 538-544 |
| `Mousegesture/hand_cursor_2.py` | 13/74 (17.6%) | 4152, 4500-4501, 4524, 4576-4577, 4993, 5004-5010, 5012, 5025-5028, 5030-5032, 5036, 5040, 5044, 5049, 5052, 5064-5065, 5067-5071, 5097-5099, 5152, 5156-5158, 5160, 5164-5165, 5176, 5182-5183, 5395, 5472, 5474, 5536-5537, 5544, 5863, 5960-5961, 5972, 5980-5982, 5984 |
| `Mousegesture/test_gesture_events.py` | 0/735 (0.0%) | 21, 23-29, 31, 33-36, 38, 40, 45, 48-49, 51-52, 54-55, 58-59, 61-63, 65-66, 68-71, 73-74, 76-77, 79-80, 82-83, 86-89, 91, 94-95, 98-99, 103-104, 107-108, 111-112, 117, 129, 131-137, 140-144, 146-149, 151, 153-155, 157-160, 162-163, 165-170, 173-175, 177, 180-181, 183-184, 186-187, 190-195, 198-199, 202, 205-207, 209-210, 212-214, 216-220, 222-223, 226-228, 230-231, 233-234, 239, 241-246, 248-257, 259, 261-264, 266-275, 277-286, 288-297, 299, 301-304, 306-310, 312-317, 319-324, 326, 328-338, 340-345, 348, 350-351, 353-358, 360-361, 363-379, 381, 383, 385-394, 396-405, 407-408, 410-412, 414-418, 420-423, 428, 431, 433-438, 441-442, 445, 447, 450-452, 456, 459-468, 471-489, 492-496, 499-501, 503-504, 506-509, 511-512, 514-515, 517-518, 521-527, 529-530, 532-533, 535-536, 539-541, 543-544, 546-547, 550, 553-556, 560, 572-573, 575, 577, 579-600, 602-607, 610, 612-614, 617-620, 622-630, 632-633, 635-636, 639, 642, 647, 649-656, 658-661, 664-669, 672-674, 676-677, 679-682, 684, 686, 689-690, 692-696, 698-699, 703-704, 706-713, 715-718, 720-722, 724-726, 728-730, 733-747, 750, 752-759, 761-763, 765-769, 771-773, 775, 777-779, 781-783, 786, 789, 791-793, 795-796, 798, 800, 803, 805, 808-810, 812-813, 815, 817-819, 821, 823-828, 830-832, 835, 837-851, 853-855, 857, 859-861, 863-865, 867-870, 873-874, 876-877, 879-882, 886, 889, 891-892, 894-897, 899, 901, 903-904, 906-908, 910-912, 914-915, 918-919, 922, 924, 927-928, 931, 933-935, 937-940, 942-946, 949-951, 953-957, 959, 961, 964, 966-968, 971-976, 979-983, 985-987, 989-994, 997-998, 1002-1006, 1008-1012, 1015, 1017-1026, 1029-1032, 1034-1038, 1043, 1045, 1049-1059, 1061-1064, 1067-1071, 1073-1075 |

### Whole modules

| Module | Files | Statements | Covered |
| --- | ---: | ---: | ---: |
| Central module (orchestrator + NLU) | 28 | 1862 | 94.1% |
| Gesture module | 14 | 7940 | 36.0% |
| Speech module | 13 | 1512 | 91.3% |

## Lint findings in changed files

| Location | Rule | Message | New |
| --- | --- | --- | :---: |
| `Mousegesture/app.py:55` | F401 | `gesture_fsm.DOUBLE_CLICK` imported but unused |  |
| `Mousegesture/app.py:57` | F401 | `gesture_fsm.MOUSE_ACTIONS` imported but unused |  |
| `Mousegesture/app.py:60` | F401 | `gesture_fsm.SENSITIVITY_STEP` imported but unused |  |
| `Mousegesture/app.py:61` | F401 | `gesture_fsm.save_config` imported but unused |  |
| `Mousegesture/app.py:1723` | F841 | Local variable `poses` is assigned to but never used |  |
| `Mousegesture/app.py:2642` | F841 | Local variable `poses` is assigned to but never used |  |
| `Mousegesture/hand_cursor_2.py:100` | F401 | `monitors.monitor_labels` imported but unused |  |
| `Mousegesture/hand_cursor_2.py:101` | F401 | `monitors.monitor_union` imported but unused |  |
| `Mousegesture/hand_cursor_2.py:104` | F401 | `gesture_fsm.HAND_LEFT` imported but unused |  |
| `Mousegesture/hand_cursor_2.py:104` | F401 | `gesture_fsm.HAND_RIGHT` imported but unused |  |
| `Mousegesture/hand_cursor_2.py:1986` | F541 | f-string without any placeholders |  |
| `Mousegesture/hand_cursor_2.py:1987` | F541 | f-string without any placeholders |  |
| `Mousegesture/hand_cursor_2.py:2008` | B904 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |  |
| `Mousegesture/hand_cursor_2.py:5512` | F841 | Local variable `prev_x` is assigned to but never used |  |
| `Mousegesture/hand_cursor_2.py:5513` | F841 | Local variable `prev_y` is assigned to but never used |  |
| `Mousegesture/hand_cursor_2.py:5580` | F841 | Local variable `thumb_out` is assigned to but never used |  |
| `Mousegesture/test_gesture_events.py:25` | F401 | `math` imported but unused | 🆕 |

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
| `1afc8a5` | BangerAtifAhmed | 2026-10-15 | feat: add gesture event reporting and queue sink integration |

<details><summary>All 5 changed project files</summary>

| Status | File | Module | + | − |
| :---: | --- | --- | ---: | ---: |
| A | `Mousegesture/IPC_EVENTS.md` | docs/gesture | 135 | 0 |
| M | `Mousegesture/app.py` | gesture/studio-gui | 16 | 1 |
| A | `Mousegesture/gesture_events.py` | gesture/gesture_events | 545 | 0 |
| M | `Mousegesture/hand_cursor_2.py` | gesture/engine | 189 | 11 |
| A | `Mousegesture/test_gesture_events.py` | gesture/test_gesture_events | 1080 | 0 |

</details>

<details><summary>git diff --stat</summary>

```text
 Mousegesture/IPC_EVENTS.md          |  135 ++++++++++
 Mousegesture/app.py                 |   17 +-
 Mousegesture/gesture_events.py      |  545 ++++++++++++++++++++++++++++++++++++++
 Mousegesture/hand_cursor_2.py       |  200 +++++++++++++-
 Mousegesture/test_gesture_events.py | 1080 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 5 files changed, 1965 insertions(+), 12 deletions(-)
```

</details>

---
_Generated by `scripts/ci/pr_report.py` — see docs/ci-cd.md._
