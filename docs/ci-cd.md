# CI/CD, tests and PR reports

Every pull request is linted, hygiene-checked and tested on Linux and Windows
before it can merge. CI then writes a structured report of everything the PR
changes. The report is posted on the PR, and once the PR lands on `main` it is
archived under [`pr-reports/`](../pr-reports/README.md). A version tag
publishes a release.

- [The pipeline](#the-pipeline)
- [Protecting `main`](#protecting-main) (one-time setup, repository admin)
- [PR reports](#pr-reports)
- [The test suite](#the-test-suite)
- [Known defects the suite pins](#known-defects-the-suite-pins)
- [When CI fails](#when-ci-fails)
- [Releases](#releases)

## The pipeline

`.github/workflows/ci.yml` runs on every pull request and on every push to `main`:

```
changes ──┬── lint ───────────┐
          ├── hygiene ────────┤
          ├── tests (per OS) ─┼──► CI gate   ← the one required status check
          ├── ml-tests ───────┤
          └── deps (*) ───────┴──► PR report ← comment on the PR, artifact,
                                               archived to pr-reports/ on main
(*) only when a requirements*.txt file changed
```

| Job | What it checks | Blocks merge? |
| --- | --- | --- |
| **Change detection** | Resolves the base and head commits and whether any requirement files changed. | yes |
| **Lint** | `ruff` with the rules that mean *broken*, not *untidy*: syntax errors, undefined names, invalid comparisons, redefinitions (`E9, F63, F7, F82, F811`). Also runs `actionlint` on the workflow files. A broader advisory set (`E, W, F, B`) goes into the PR report. | yes (blocking set only) |
| **Repository hygiene** | Scans what the PR *adds* for conflict markers, broken JSON or Python, LFS bypasses, non-LFS files over 10 MB and credentials. It only warns about `.venv/`/`__pycache__` changes, 2–10 MB files and a branch that is behind `main`. | errors only |
| **Tests (ubuntu, windows)** | The whole pytest suite except the torch tests. Pushes to `main` also run it on macOS. | yes |
| **NLU model tests** | The torch/transformers tests (`-m ml`), on CPU wheels. | yes |
| **Dependencies resolve** | `pip install --dry-run` of every pinned requirement file on each platform it supports. | yes, when it runs |
| **CI gate** | Fails unless every job above passed. A skipped `deps` job counts as passing. | **this is the required check** |
| **PR report** | Builds the report, comments on the PR, uploads the `pr-report` artifact, and archives it on `main`. | no |

Other automation:

- **`labeler.yml`** labels each PR by area: `area: central`, `area: speech`,
  `area: gesture`, `area: models`, `dependencies`, and so on.
- **`dependabot.yml`** keeps the GitHub Actions versions current.
- **`pull_request_template.md`** gives every PR a verification checklist.
- **`CODEOWNERS`** is a ready template: fill in the GitHub handles to get
  automatic review requests per module.

Test jobs check out the repository sparsely, without the ~30k-file committed
`.venv`. They also skip Git LFS: the ~1 GB of model weights arrive as pointer
files, and the few tests that need real weights skip themselves (see
`models` below).

## Protecting `main`

The pipeline informs merges. Branch protection is what enforces them.
Settings → Branches (or Rules → Rulesets) → add a rule for `main`:

1. **Require a pull request before merging.** Add 1 approval if the team
   wants reviews.
2. **Require status checks to pass**, and select **`CI gate`** only. It
   already aggregates every other job, so new jobs never need re-adding here.
3. Optionally, **require branches to be up to date before merging**. The
   report flags a branch that is behind anyway.
4. If `CODEOWNERS` is filled in, **require review from Code Owners**.

Archiving reports needs one more decision. When a PR merges, the `report`
job commits `pr-reports/PR-NNNN/` to `main` as `github-actions[bot]`. If a
rule blocks direct pushes to `main`, that commit is rejected. The job then
warns but does not fail, and the report stays on the PR comment and as a
30-day artifact. To keep the permanent archive, let GitHub Actions bypass
the rule (Rulesets → *Bypass list*) or exempt `pr-reports/**`.

## PR reports

### Where they appear

| When | Where |
| --- | --- |
| Every PR push | A comment on the PR, updated in place: verdict, checks, what needs attention, and changes by module. The workflow run's **summary page** shows the same. |
| Every CI run | The **`pr-report` artifact**: the full report folder, kept 30 days. |
| After merge to `main` | **`pr-reports/PR-NNNN/`** in the repository, plus a regenerated [`pr-reports/README.md`](../pr-reports/README.md) index of every merged PR. |
| On your machine | `python scripts/ci/pr_report.py build` writes `pr-reports/local/<branch>/`, which is git-ignored. |

### Folder structure

```
pr-reports/
├── README.md            index: every merged PR, its verdict, size, tests and modules
├── index.json           the same, machine-readable
└── PR-0012/
    ├── README.md        the full human-readable report
    ├── summary.json     verdict and headline numbers (the index is built from these)
    ├── changes.json     every changed file (status, module, component, +/−),
    │                    per-module totals, commits
    ├── quality.json     tests per platform, coverage, lint, hygiene, CI job results
    └── diffstat.txt     git diff --stat of the change set
```

### Reading a report

The verdict is one of four values:

- ✅ **Ready to merge**: every check passed and nothing needs a reviewer's
  judgement.
- ⚠️ **Mergeable, needs reviewer attention**: checks passed, but something is
  flagged below.
- ❌ **Blocked**: a required check failed.
- ℹ️ **Checks not run**: a local report without CI results.

The **Needs attention** section lists what no automated check can decide:

- dependency pins changed
- calibrated constants changed in `speech_module/config.py`,
  `central_module/config.py` or `gesture_config.json`. The project rule is
  that each constant carries its measurement.
- the intent catalog changed without a retrained NLU model
- model files changed
- source changed with no test changes
- `.venv` or bytecode in the diff
- code whose real behaviour needs hardware CI doesn't have, such as a
  microphone, a webcam or a specific OS
- the branch is behind `main`

After that, the report covers:

- **coverage of the lines this PR added**, per file, with the uncovered line
  ranges
- test failures, with their output
- lint findings in the changed files, with those on added lines marked 🆕
- hygiene findings
- the known defects the suite pins (xfails)
- commits and every changed file

The committed `.venv` is summarised rather than listed, so a
virtualenv update doesn't bury the report.

## The test suite

```
tests/
├── conftest.py          isolation + safety nets for the whole suite
├── helpers/             fakes (pynput, vosk, onnxruntime, sounddevice, cv2,
│                        mediapipe), synthetic hands and audio, repo paths
├── central/             normalisation, intent catalog contracts, matching ladder,
│                        executor, all three OS adapters, IPC, supervisor (real
│                        child processes), manager, runners + CLI end to end,
│                        learning, NLU labels/dataset/post-processing, NLU model (ml)
├── speech/              config contracts, classifier, router, VOSK engine, VAD
│                        (energy + Silero), resampling, audio stream/devices,
│                        denoiser, Whisper worker, speech manager, calibration
├── gesture/             FSM, output executors/dispatchers, monitors, landmark
│                        geometry, screen mapping + filters, config/artwork contracts,
│                        Gesture Studio helpers
└── repo/                model fetch script, CI tooling, release script, workflow files
```

### Running it

The test environment is separate from the project `.venv`. It needs Python
3.11 and a few small packages: no torch, no microphone, no webcam.

```powershell
# Windows
py -3.11 -m venv .venv-test
.venv-test\Scripts\python -m pip install -r requirements-test.txt
.venv-test\Scripts\python -m pytest
```

```bash
# Linux / macOS
python3.11 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-test.txt
.venv-test/bin/python -m pytest
```

Useful selections:

```bash
python -m pytest tests/speech                 # one module
python -m pytest -m "not slow"                # skip the tests that spawn processes
python -m pytest -m ml                        # torch tests — run with the project .venv,
                                              #   which has torch + transformers
python -m pytest -m models                    # tests against the real weights (needs git lfs pull)
python -m pytest --cov --cov-report=html      # coverage report in htmlcov/
```

`.venv-test/` is git-ignored.

### Markers

| Marker | Meaning | In CI |
| --- | --- | --- |
| `slow` | Spawns real child processes or waits on real timers | runs |
| `ml` | Needs torch + transformers | the `ml-tests` job |
| `models` | Needs the real weights, not LFS pointers | skipped (the weights aren't fetched), so run them locally |

### Guarantees every test gets (tests/conftest.py)

- **No real side effects.** Guards fail the test loudly if code under test
  tries to open a browser, launch an app, run an OS command such as
  `shutdown`, or inject input. pynput is always a recorder, even where the
  real package is installed.
- **Isolated user data.** `MMCTRL_DATA` points at a throwaway directory, so
  `central_module/data/` is never touched.
- **Deterministic NLU.** The intent engine runs on the rule ladder, whatever
  weights are on disk. The `ml` and `models` tests opt back in.
- **No bytecode.** This repository tracks some `.pyc` files, so a test run
  that rewrote them would show up as local changes.

### Contract tests

Some tests check that separate files still agree, because that's where
merges break things silently:

- Every catalog intent has an executor handler, and every pattern is reachable.
- The shipped model's `nlu_config.json` label space equals the intent catalog.
  Changing intents without retraining fails here.
- `central_module` language names map to real VOSK models, and both modules
  agree on dictation stop words.
- Every OS adapter can press every canonical key.
- The shipped `gesture_config.json` compiles cleanly, binds only poses a
  classifier can emit, and has no rules that fire on the same frame.
  Label aliases and the `gestures/` artwork agree.
- The workflow files keep their expressions intact, their actions pinned, and
  their permissions minimal.

## Known defects the suite pins

These tests found real bugs. Each is marked `xfail(strict=True)`: CI stays
green, and once the bug is fixed the test turns into a failure that asks for
the marker to be removed. Every report lists them.

| Where | Defect |
| --- | --- |
| `central_module/nlp/intents.py` + `normalize.py` | `command_normalize` rewrites "double" and "twice" to `2` before pattern matching. On the rule ladder (no trained model), "double click" and "click twice" therefore fuzzy-match `click` and perform a **single** click. The pattern `press the {key} key` can never match, because `the` is stripped first. |
| `fetch_deepfilternet_model.py` | **Zip-slip.** The path-traversal guard compares string prefixes, so an archive entry `../DFN3evil/x` escapes into a sibling directory. Use `Path.is_relative_to(dest.resolve())` instead. |
| Shipped NLU weights (`models` tests, local only) | Multi-word app names are truncated to one word: "launch google chrome" gives `google`, and "bring up file explorer" gives `file`. In fp32, "google chrome" is tagged `B-app B-app`, because both words are standalone apps in the dataset's `APPS` bank. int8 quantisation additionally drops `I-app` on "file explorer". The literal phrasings still work, because exact patterns run first. |

## When CI fails

| Failure | Usual cause and fix |
| --- | --- |
| `test_shipped_model_label_space_matches_the_catalog` | Intents were added, removed or renamed. Retrain (`python -m central_module.nlp.nlu.train`) and commit `Models/NLU/joint-distilbert/` with the catalog change. |
| `test_every_action_has_a_handler` | A new intent without an `ActionExecutor._do_<action>` method. |
| `test_shipped_config_*` (gesture) | `gesture_config.json` has an invalid rule, an unknown action, a pose nothing emits, or two rules that fire together. |
| hygiene `lfs-bypass` | A weight file was committed without Git LFS. Run `git lfs install`, `git rm --cached <file>`, then `git add <file>`, and commit again. |
| hygiene `secret` | Revoke the credential first, then remove it from history. |
| hygiene `large-file` | Track the file with Git LFS (add a rule to `.gitattributes`). |
| Lint `F821` | An undefined name. Annotation-only names need an `if TYPE_CHECKING:` import. |
| `deps` job | A pin doesn't resolve on that platform. The constraints are in `requirements.txt`: Python ≤ 3.11 and numpy < 2 for DeepFilterNet. |
| An `XPASS(strict)` failure | You fixed a pinned defect. Remove the `xfail` marker from that test. |

## Releases

Push a version tag from `main`:

```bash
git tag v1.0.0 && git push origin v1.0.0
```

`.github/workflows/release.yml` runs the full CI pipeline on the tagged
commit, on all three operating systems. Only if that passes does it publish
a GitHub Release with:

- `multimodal-device-control-v1.0.0-src.zip`: the source at the tag, without
  the committed `.venv`, bytecode, archived reports or model weights. A
  `MODELS.md` inside lists every weight file with its size and LFS object id,
  and explains how to fetch them.
- a `.sha256` checksum
- release notes: the PRs merged since the previous tag, built from their
  archived reports, falling back to the commit list.

Build the same artifacts locally with
`python scripts/ci/release.py --tag v1.0.0 --out dist`.
