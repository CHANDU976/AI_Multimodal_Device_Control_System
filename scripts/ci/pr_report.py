"""Structured report of a change set — what changed, where, and whether it
is safe to merge.

    # a report for the current branch (writes pr-reports/local/<branch>/)
    python scripts/ci/pr_report.py build

    # CI: feed it every job's artifacts and results
    python scripts/ci/pr_report.py build --base <sha> --head <sha> \
        --artifacts ci-artifacts --out pr-reports/PR-0012 --comment-out comment.md

    # rebuild pr-reports/README.md + index.json from every archived report
    python scripts/ci/pr_report.py index

Report folder layout (one folder per PR, see pr-reports/README.md):

    README.md      the human-readable report
    summary.json   verdict + headline numbers (what the index is built from)
    changes.json   every changed file, per-module totals, commits
    quality.json   tests, coverage, lint, hygiene and CI job results
    diffstat.txt   `git diff --stat` for the change set

Inputs picked up from --artifacts (all optional — missing ones are shown as
"not run", never guessed): junit-*.xml, coverage-*.json, lint-blocking.json,
lint-advisory.json, hygiene.json. CI job results come from --jobs-json (the
workflow's `toJSON(needs)`), PR metadata from REPORT_* environment variables.

Standard library only.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ci_common import (  # noqa: E402
    AREAS, ROOT, SOURCE_AREAS, added_lines, behind_count, changed_files, classify, commits,
    git, github_env, ranges, repo_root, resolve_range,
)

COMMENT_MARKER = "<!-- ci-pr-report -->"
DIFF_COVERAGE_TARGET = 60.0          # % of changed executable lines; below → attention
LARGE_CHANGE_LINES = 1500

ICON = {"ready": "✅", "attention": "⚠️", "blocked": "❌", "unverified": "ℹ️"}
LABEL = {"ready": "Ready to merge", "attention": "Mergeable — needs reviewer attention",
         "blocked": "Blocked — required checks failed", "unverified": "Checks not run"}
OPTIONAL_JOBS = {"deps"}              # skipped when no requirements file changed
JOB_NAMES = {
    "changes": "Change detection", "lint": "Lint (blocking rules)",
    "hygiene": "Repository hygiene", "tests": "Tests (all platforms)",
    "ml-tests": "NLU model tests (torch)", "deps": "Dependency resolution",
}


# ── Metadata ───────────────────────────────────────────────────────────────────

def collect_meta(args) -> dict:
    number = args.pr_number or github_env("REPORT_PR_NUMBER")
    branch = ""
    try:
        branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    except Exception:
        pass
    return {
        "number": int(number) if str(number).isdigit() else None,
        "title": args.title or github_env("REPORT_PR_TITLE") or _last_subject(),
        "author": github_env("REPORT_PR_AUTHOR") or _git_author(),
        "url": github_env("REPORT_PR_URL"),
        "base_ref": github_env("REPORT_BASE_REF", "main"),
        "head_ref": github_env("REPORT_HEAD_REF", branch),
        "event": github_env("REPORT_EVENT", "local"),
        "run_url": github_env("REPORT_RUN_URL"),
        "repository": github_env("GITHUB_REPOSITORY"),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _last_subject() -> str:
    try:
        return git("log", "-1", "--format=%s").strip()
    except Exception:
        return ""


def _git_author() -> str:
    try:
        return git("config", "user.name").strip()
    except Exception:
        return ""


# ── Artifact parsing ───────────────────────────────────────────────────────────

def parse_junit(path: Path, label: str | None = None) -> dict:
    """pytest --junitxml → counts, failures, skip reasons and xfails."""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    stem = path.stem[len("junit-"):] if path.stem.startswith("junit-") else path.stem
    out = {"label": label or stem, "tests": 0, "passed": 0,
           "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0, "time": 0.0,
           "failures": [], "skip_reasons": {}, "xfails": []}
    reasons: Counter = Counter()
    for suite in suites:
        out["time"] += float(suite.get("time") or 0)
        for case in suite.iter("testcase"):
            out["tests"] += 1
            test_id = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")
            if failure is not None or error is not None:
                node = failure if failure is not None else error
                out["failed" if failure is not None else "errors"] += 1
                out["failures"].append({
                    "test": test_id,
                    "kind": "failure" if failure is not None else "error",
                    "message": (node.get("message") or "").strip()[:500],
                    "details": (node.text or "").strip()[-3000:],
                })
            elif skipped is not None:
                message = (skipped.get("message") or "").strip()
                if skipped.get("type") == "pytest.xfail" or message.startswith("xfail"):
                    out["xfailed"] += 1
                    out["xfails"].append({"test": test_id, "reason": message})
                else:
                    out["skipped"] += 1
                    reasons[_skip_reason(message)] += 1
            else:
                out["passed"] += 1
    out["skip_reasons"] = dict(reasons.most_common())
    out["time"] = round(out["time"], 1)
    return out


def _skip_reason(message: str) -> str:
    # "tests/x.py:12: could not import 'torch': ..." → "could not import 'torch'"
    message = re.sub(r"^\S+:\d+: ", "", message)
    return message.split(": No module named")[0][:160] or "(no reason given)"


def load_coverage(paths: list[Path]) -> dict:
    """Union of coverage.py JSON reports: {file: {"executed": set, "statements": set}}."""
    merged: dict[str, dict] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, info in data.get("files", {}).items():
            key = name.replace("\\", "/")
            entry = merged.setdefault(key, {"executed": set(), "statements": set()})
            executed = set(info.get("executed_lines", []))
            entry["executed"] |= executed
            entry["statements"] |= executed | set(info.get("missing_lines", []))
    return merged


def coverage_by_area(cov: dict) -> dict:
    table: dict[str, dict] = {}
    for path, entry in cov.items():
        area, _ = classify(path)
        row = table.setdefault(area, {"statements": 0, "covered": 0, "files": 0})
        row["statements"] += len(entry["statements"])
        row["covered"] += len(entry["executed"] & entry["statements"])
        row["files"] += 1
    for row in table.values():
        row["percent"] = _pct(row["covered"], row["statements"])
    return dict(sorted(table.items()))


def diff_coverage(added: dict[str, set[int]], cov: dict) -> dict:
    """Coverage of the executable lines this change set added."""
    files, total, hit = [], 0, 0
    for path in sorted(added):
        entry = cov.get(path)
        if entry is None:
            continue
        executable = added[path] & entry["statements"]
        if not executable:
            continue
        covered = executable & entry["executed"]
        total += len(executable)
        hit += len(covered)
        files.append({"path": path, "executable": len(executable), "covered": len(covered),
                      "percent": _pct(len(covered), len(executable)),
                      "uncovered": ranges(executable - covered)})
    return {"executable": total, "covered": hit,
            "percent": _pct(hit, total) if total else None, "files": files}


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 100.0


def normalize_lint(items: list, roots: list[str]) -> list[dict]:
    """ruff JSON → [{path, line, code, message}] with repo-relative paths."""
    out = []
    for item in items:
        out.append({
            "path": _relpath(item.get("filename", ""), roots),
            "line": (item.get("location") or {}).get("row", 0),
            "code": item.get("code") or "syntax",
            "message": item.get("message", ""),
        })
    return sorted(out, key=lambda f: (f["path"], f["line"], f["code"]))


_TOP = ("central_module/", "speech_module/", "Mousegesture/", "tests/", "scripts/",
        "Models/", "fetch_deepfilternet_model.py")


def _relpath(filename: str, roots: list[str]) -> str:
    name = filename.replace("\\", "/")
    for root in roots:
        root = root.replace("\\", "/").rstrip("/") + "/"
        if root != "/" and name.lower().startswith(root.lower()):
            return name[len(root):]
    for top in _TOP:
        idx = name.rfind("/" + top)
        if idx >= 0:
            return name[idx + 1:]
    return name


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ── Analysis ───────────────────────────────────────────────────────────────────

def area_summary(changes) -> list[dict]:
    rows: dict[str, dict] = {}
    for c in changes:
        row = rows.setdefault(c.area, {"area": c.area, "title": AREAS.get(c.area, c.area),
                                       "files": 0, "additions": 0, "deletions": 0,
                                       "binary": 0, "components": Counter()})
        row["files"] += 1
        row["additions"] += c.additions
        row["deletions"] += c.deletions
        row["binary"] += int(c.binary)
        if c.component:
            row["components"][c.component] += 1
    order = list(AREAS)
    out = []
    for area in sorted(rows, key=lambda a: order.index(a) if a in order else 99):
        row = rows[area]
        row["components"] = dict(row["components"].most_common())
        out.append(row)
    return out


# Areas summarised rather than listed file by file: a virtualenv update can be
# tens of thousands of files, which would bury the report (and bloat the archive).
COLLAPSED = ("environment", "generated")


def collapsed_summary(changes) -> dict:
    out = {}
    for area in COLLAPSED:
        items = [c for c in changes if c.area == area]
        if items:
            out[area] = {"files": len(items),
                         "additions": sum(c.additions for c in items),
                         "deletions": sum(c.deletions for c in items),
                         "top_components": dict(Counter(c.component for c in items)
                                                .most_common(10)),
                         "sample": [c.path for c in items[:25]]}
    return out


CALIBRATED = {
    "speech_module/config.py": "speech routing / VAD / denoiser thresholds",
    "central_module/config.py": "central NLP, supervisor and adaptation constants",
    "Mousegesture/gesture_config.json": "gesture bindings and cursor settings",
}
MANUAL_CHECKS = {
    "central_module/actions/windows_adapter.py": "Windows input injection — try it on Windows",
    "central_module/actions/linux_adapter.py": "Linux/X11 tools — try it on an X11 desktop",
    "central_module/actions/macos_adapter.py": "macOS AppleScript — try it on a Mac "
                                               "(Accessibility permission required)",
    "speech_module/audio_stream.py": "live microphone capture",
    "speech_module/speech_manager.py": "the real-time speech loop with a microphone",
    "speech_module/audio_devices.py": "device hot-swap with a real headset",
    "Mousegesture/hand_cursor_2.py": "camera + MediaPipe runtime with a webcam",
    "Mousegesture/app.py": "the Gesture Studio GUI",
}


def attention(changes, behind: int) -> list[dict]:
    """Things a reviewer should look at that no automated check can decide."""
    flags = []
    paths = {c.path for c in changes}
    live = {c.path for c in changes if c.status != "D"}
    areas = Counter(c.area for c in changes)

    def flag(level, title, detail, files=()):
        flags.append({"level": level, "title": title, "detail": detail,
                      "files": sorted(files)[:20], "file_count": len(files)})

    deps = [c.path for c in changes if c.area == "dependencies"]
    if deps:
        flag("warn", "Dependency pins changed",
             "The `deps` job resolves them on Linux and Windows. Mind the documented "
             "constraints: Python ≤ 3.11 and numpy < 2 (DeepFilterNet), CPU torch builds.",
             deps)
    if "central_module/nlp/intents.py" in paths and \
            "Models/NLU/joint-distilbert/joint_nlu.pt" not in paths:
        flag("warn", "Intent catalog changed without a retrained NLU model",
             "Catalog patterns feed the synthetic training set. Added/removed/renamed intents "
             "fail the label-space contract test; pattern-only changes do not — retrain with "
             "`python -m central_module.nlp.nlu.train` if paraphrase coverage matters.",
             ["central_module/nlp/intents.py"])
    nlu_code = [p for p in paths if p.startswith("central_module/nlp/nlu/")
                and p.endswith(("dataset.py", "train.py", "labels.py", "model.py"))]
    if nlu_code and "Models/NLU/joint-distilbert/joint_nlu.pt" not in paths:
        flag("info", "NLU training code changed",
             "The shipped weights were trained by the previous version; retrain before release.",
             nlu_code)
    calibrated = [(p, what) for p, what in CALIBRATED.items() if p in live]
    if calibrated:
        flag("warn", "Calibrated constants changed",
             "; ".join(f"`{p}` ({what})" for p, what in calibrated) + ". The project rule is "
             "that every constant carries the measurement that justifies it — check the "
             "comments were updated with the new numbers.", [p for p, _ in calibrated])
    models = [c.path for c in changes if c.area == "models"]
    if models:
        flag("warn", "Model files changed",
             "Hygiene verifies they are Git LFS pointers. Update the model table in README.md "
             "and re-run the `models`-marked tests locally (`pytest -m models`).", models)
    src = [c.path for c in changes
           if c.area in SOURCE_AREAS and c.status != "D" and c.path.endswith(".py")]
    if src and not areas["tests"]:
        flag("warn", "Source changed without test changes",
             "No file under tests/ was touched. Add or update tests, or say in the PR why "
             "none are needed.", src)
    env = [c.path for c in changes if c.area in COLLAPSED]
    if env:
        flag("warn", "Virtualenv / bytecode files in the diff",
             f"{len(env)} file(s) under .venv/ or __pycache__/. They are machine-specific and "
             "bloat history; drop them from the PR (`git restore --staged .venv`).", env)
    manual = [(p, MANUAL_CHECKS[p]) for p in sorted(live) if p in MANUAL_CHECKS]
    if manual:
        flag("info", "Needs a hands-on check CI cannot do",
             "CI fakes the hardware boundary for: " +
             "; ".join(f"`{Path(p).name}` → {what}" for p, what in manual) + ".",
             [p for p, _ in manual])
    if areas["ci"]:
        flag("info", "CI/tooling changed",
             "The pipeline validating this change is itself modified — review it as code.",
             [c.path for c in changes if c.area == "ci"])
    size = sum(c.additions + c.deletions for c in changes
               if c.area not in COLLAPSED + ("models", "reports"))
    if size > LARGE_CHANGE_LINES:
        flag("info", f"Large change set ({size} lines)",
             "Consider splitting it: smaller PRs review faster and bisect cleanly.")
    if behind:
        flag("info", f"Branch is {behind} commit(s) behind its base",
             "CI tested the merge result, but update the branch before merging if the base "
             "moved in the same modules.")
    return flags


def job_rows(jobs: dict | None) -> list[dict]:
    rows = []
    for name, info in (jobs or {}).items():
        if name == "report":
            continue
        rows.append({"job": name, "title": JOB_NAMES.get(name, name),
                     "result": (info or {}).get("result", "unknown")})
    return rows


def decide(jobs, tests, lint_blocking, hygiene, flags, diffcov,
           advisory_new) -> tuple[str, list[str]]:
    """(verdict, reasons)."""
    reasons = []
    for row in job_rows(jobs):
        # Same rule as the CI gate: anything but success blocks, except a
        # skipped job that is allowed to skip. Otherwise a skipped test job
        # would read "Ready to merge" while the gate is red.
        if row["result"] != "success" and not (row["job"] in OPTIONAL_JOBS
                                               and row["result"] == "skipped"):
            reasons.append(f"CI job `{row['job']}` {row['result']}")
    failed = sum(t["failed"] + t["errors"] for t in tests)
    if failed:
        reasons.append(f"{failed} test(s) failed")
    if lint_blocking:
        reasons.append(f"{len(lint_blocking)} blocking lint finding(s)")
    if hygiene and hygiene.get("errors"):
        reasons.append(f"{len(hygiene['errors'])} hygiene error(s)")
    if reasons:
        return "blocked", reasons

    if not jobs and not tests and hygiene is None:
        return "unverified", ["no CI results were supplied — run in CI, or pass --artifacts"]

    notes = [f["title"] for f in flags if f["level"] == "warn"]
    if hygiene and hygiene.get("warnings"):
        notes.append(f"{len(hygiene['warnings'])} hygiene warning(s)")
    if diffcov and diffcov["percent"] is not None and diffcov["executable"] >= 10 \
            and diffcov["percent"] < DIFF_COVERAGE_TARGET:
        notes.append(f"changed-line coverage {diffcov['percent']}% < {DIFF_COVERAGE_TARGET:g}%")
    if advisory_new:
        notes.append(f"{len(advisory_new)} new advisory lint finding(s)")
    return ("attention", notes) if notes else ("ready", [])


# ── Build ──────────────────────────────────────────────────────────────────────

def build(args) -> dict:
    meta = collect_meta(args)
    base, head = resolve_range(args.base, args.head)
    changes = changed_files(base, head)
    log = commits(base, head)
    behind = behind_count(base, head)
    # Added lines of project Python only — the committed virtualenv can hold
    # thousands of .py files and says nothing about this change's coverage.
    has_py = any(c.path.endswith(".py") and c.area not in ("environment", "generated")
                 and c.status != "D" for c in changes)
    added = added_lines(base, head, ["*.py", ":(exclude).venv"]) if has_py else {}

    art = Path(args.artifacts) if args.artifacts else None
    files = sorted(art.rglob("*")) if art and art.exists() else []
    tests = [parse_junit(p) for p in files if p.name.startswith("junit-") and p.suffix == ".xml"]
    cov_paths = [p for p in files if p.name.startswith("coverage-") and p.suffix == ".json"]
    cov = load_coverage(cov_paths) if cov_paths else {}
    roots = [str(repo_root()), str(ROOT), github_env("GITHUB_WORKSPACE")]
    lint_b = normalize_lint(_first_json(files, "lint-blocking.json") or [], roots)
    lint_a = normalize_lint(_first_json(files, "lint-advisory.json") or [], roots)
    hygiene = _first_json(files, "hygiene.json")
    jobs = json.loads(args.jobs_json) if args.jobs_json else \
        (json.loads(os.environ["JOBS_JSON"]) if os.environ.get("JOBS_JSON") else None)

    changed = {c.path for c in changes}
    advisory = [f for f in lint_a if f["path"] in changed]
    advisory_new = [f for f in advisory if f["line"] in added.get(f["path"], set())]
    diffcov = diff_coverage(added, cov) if cov else None
    flags = attention(changes, behind)
    verdict, reasons = decide(jobs, tests, lint_b, hygiene, flags, diffcov, advisory_new)

    totals = {
        "files": len(changes),
        "additions": sum(c.additions for c in changes),
        "deletions": sum(c.deletions for c in changes),
        "commits": len(log),
    }
    report = {
        "schema": 1,
        "meta": meta,
        "range": {"base": base, "head": head, "behind": behind},
        "verdict": verdict, "verdict_reasons": reasons,
        "totals": totals,
        "areas": area_summary(changes),
        "attention": flags,
        "changes": [c.to_dict() for c in changes if c.area not in COLLAPSED],
        "collapsed": collapsed_summary(changes),
        "commits": log,
        "quality": {
            "jobs": job_rows(jobs),
            "tests": tests,
            "coverage": {"by_area": coverage_by_area(cov) if cov else {},
                         "changed_lines": diffcov},
            "lint": {"blocking": lint_b, "advisory_in_changed_files": advisory,
                     "advisory_new": advisory_new, "advisory_total": len(lint_a),
                     "supplied": any(p.name.startswith("lint-") for p in files)},
            "hygiene": hygiene,
        },
    }
    report["diffstat"] = git("diff", "--stat=120", f"{base}...{head}", "--", ".",
                             ":(exclude).venv", ":(exclude)**/__pycache__/**")
    return report


def _first_json(files: list[Path], name: str):
    for p in files:
        if p.name == name:
            return load_json(p)
    return None


def default_out(meta: dict) -> Path:
    if meta["number"]:
        return repo_root() / "pr-reports" / f"PR-{meta['number']:04d}"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", meta["head_ref"] or "HEAD").strip("-") or "HEAD"
    return repo_root() / "pr-reports" / "local" / slug


def write_report(report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    q = report["quality"]
    summary = {
        "schema": report["schema"], "meta": report["meta"], "range": report["range"],
        "verdict": report["verdict"], "verdict_reasons": report["verdict_reasons"],
        "totals": report["totals"],
        "areas": {a["area"]: {k: a[k] for k in ("files", "additions", "deletions")}
                  for a in report["areas"]},
        "tests": _test_totals(q["tests"]),
        "changed_line_coverage": (q["coverage"]["changed_lines"] or {}).get("percent"),
        "attention": [f["title"] for f in report["attention"]],
    }
    _dump(out / "summary.json", summary)
    _dump(out / "changes.json", {"areas": report["areas"], "files": report["changes"],
                                 "collapsed": report["collapsed"],
                                 "commits": report["commits"]})
    _dump(out / "quality.json", q)
    (out / "diffstat.txt").write_text(report["diffstat"], encoding="utf-8")
    (out / "README.md").write_text(render_readme(report), encoding="utf-8")


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
                    encoding="utf-8")


def _test_totals(tests: list[dict]) -> dict:
    keys = ("tests", "passed", "failed", "errors", "skipped", "xfailed")
    totals = {k: sum(t[k] for t in tests) for k in keys}
    totals["suites"] = len(tests)
    return totals


# ── Rendering ──────────────────────────────────────────────────────────────────

RESULT_ICON = {"success": "✅", "failure": "❌", "cancelled": "🚫", "skipped": "⏭️"}


def _title(report) -> str:
    m = report["meta"]
    head = f"PR #{m['number']}" if m["number"] else f"Change set `{m['head_ref'] or 'HEAD'}`"
    return f"{head} — {m['title']}" if m["title"] else head


def _headline(report) -> str:
    t = report["totals"]
    v = report["verdict"]
    return (f"> {ICON[v]} **{LABEL[v]}** · {t['files']} files · +{t['additions']} / "
            f"−{t['deletions']} · {t['commits']} commit(s)")


def _suite_detail(t: dict) -> str:
    return (f"`{t['label']}`: {t['passed']} passed, {t['failed'] + t['errors']} failed, "
            f"{t['skipped']} skipped, {t['xfailed']} xfail · {t['time']} s")


def _suites_for(job: str, tests: list[dict]) -> list[dict]:
    if job == "ml-tests":
        return [t for t in tests if t["label"].startswith("ml")]
    if job == "tests":
        return [t for t in tests if not t["label"].startswith("ml")]
    return []


def _checks_table(report) -> list[str]:
    """One row per CI check: the job's result, with detail from its artifacts."""
    q = report["quality"]
    lint, hy, dc = q["lint"], q["hygiene"], q["coverage"]["changed_lines"]
    detail = {}
    if lint["supplied"]:
        detail["lint"] = (f"{len(lint['blocking'])} blocking finding(s) · advisory: "
                          f"{len(lint['advisory_in_changed_files'])} in changed files, "
                          f"{len(lint['advisory_new'])} on added lines")
    if hy is not None:
        detail["hygiene"] = (f"{len(hy.get('errors', []))} error(s), "
                             f"{len(hy.get('warnings', []))} warning(s)")
    for job in ("tests", "ml-tests"):
        suites = _suites_for(job, q["tests"])
        if suites:
            detail[job] = "<br>".join(_suite_detail(t) for t in suites)

    rows = ["| Check | Result | Details |", "| --- | --- | --- |"]
    shown = set()
    for job in q["jobs"]:
        if job["job"] == "changes":
            continue
        text = detail.get(job["job"], "")
        if job["job"] == "deps" and job["result"] == "skipped":
            text = "no requirement files changed"
        rows.append(f"| {job['title']} | {RESULT_ICON.get(job['result'], '❔')} "
                    f"{job['result']} | {text} |")
        shown.add(job["job"])
    # Artifacts without a job result (local runs): derive the verdict from them.
    for t in q["tests"]:
        job = "ml-tests" if t["label"].startswith("ml") else "tests"
        if job not in shown:
            ok = "✅" if not (t["failed"] or t["errors"]) else "❌"
            rows.append(f"| pytest | {ok} | {_suite_detail(t)} |")
    if "lint" not in shown and lint["supplied"]:
        rows.append(f"| {JOB_NAMES['lint']} | {'❌' if lint['blocking'] else '✅'} "
                    f"| {detail['lint']} |")
    if "hygiene" not in shown and hy is not None:
        icon = "❌" if hy.get("errors") else ("⚠️" if hy.get("warnings") else "✅")
        rows.append(f"| {JOB_NAMES['hygiene']} | {icon} | {detail['hygiene']} |")
    if dc and dc["percent"] is not None:
        icon = "✅" if dc["percent"] >= DIFF_COVERAGE_TARGET or dc["executable"] < 10 else "⚠️"
        rows.append(f"| Coverage of added lines | {icon} | {dc['percent']}% of "
                    f"{dc['executable']} executable added line(s) "
                    f"(target {DIFF_COVERAGE_TARGET:g}%) |")
    return rows if len(rows) > 2 else ["_No CI results were supplied to this report._"]


def _areas_table(report) -> list[str]:
    rows = ["| Module | Files | + | − | Components |", "| --- | ---: | ---: | ---: | --- |"]
    for a in report["areas"]:
        comps = ", ".join(f"`{k}`" + (f" ×{v}" if v > 1 else "")
                          for k, v in list(a["components"].items())[:8])
        if len(a["components"]) > 8:
            comps += f", … (+{len(a['components']) - 8})"
        rows.append(f"| {a['title']} | {a['files']} | {a['additions']} | {a['deletions']} "
                    f"| {comps} |")
    return rows


def _attention_list(report) -> list[str]:
    out = []
    for f in report["attention"]:
        icon = "⚠️" if f["level"] == "warn" else "ℹ️"
        files = ""
        if f["files"]:
            total = f.get("file_count", len(f["files"]))
            shown = ", ".join(f"`{p}`" for p in f["files"][:5])
            more = f" (+{total - 5} more)" if total > 5 else ""
            files = f" — {shown}{more}"
        out.append(f"- {icon} **{f['title']}**{files}  \n  {f['detail']}")
    return out or ["- Nothing flagged."]


def render_readme(report: dict) -> str:
    m, q, r = report["meta"], report["quality"], report["range"]
    lines = [f"# {_title(report)}", "", _headline(report), ""]
    if report["verdict_reasons"]:
        lines += [f"- {reason}" for reason in report["verdict_reasons"]] + [""]
    lines += ["| | |", "| --- | --- |"]
    if m["url"]:
        lines.append(f"| Pull request | {m['url']} |")
    if m["author"]:
        lines.append(f"| Author | {m['author']} |")
    lines.append(f"| Branch | `{m['head_ref']}` → `{m['base_ref']}` |")
    lines.append(f"| Base / head | `{r['base'][:10]}` / `{r['head'][:10]}` |")
    if m["run_url"]:
        lines.append(f"| Workflow run | {m['run_url']} |")
    lines.append(f"| Generated | {m['generated_at']} ({m['event']}) |")

    lines += ["", "## Checks", "", *_checks_table(report)]
    lines += ["", "## Needs attention", "", *_attention_list(report)]
    lines += ["", "## Changes by module", "", *_areas_table(report)]

    dc = q["coverage"]["changed_lines"]
    by_area = q["coverage"]["by_area"]
    if dc or by_area:
        lines += ["", "## Coverage"]
        if dc:
            lines += ["", "### Lines added by this change set", ""]
            if dc["percent"] is None:
                lines.append("No executable Python lines were added.")
            else:
                lines.append(f"**{dc['percent']}%** of {dc['executable']} executable added "
                             f"line(s) are exercised by the tests.")
                if dc["files"]:
                    lines += ["", "| File | Covered | Uncovered added lines |",
                              "| --- | ---: | --- |"]
                    for f in dc["files"]:
                        lines.append(f"| `{f['path']}` | {f['covered']}/{f['executable']} "
                                     f"({f['percent']}%) | {f['uncovered'] or '—'} |")
        if by_area:
            lines += ["", "### Whole modules", "", "| Module | Files | Statements | Covered |",
                      "| --- | ---: | ---: | ---: |"]
            for area, row in by_area.items():
                lines.append(f"| {AREAS.get(area, area)} | {row['files']} | "
                             f"{row['statements']} | {row['percent']}% |")

    failures = [(t["label"], f) for t in q["tests"] for f in t["failures"]]
    if failures:
        lines += ["", "## Test failures", ""]
        for label, f in failures[:30]:
            lines += [f"### `{f['test']}` ({label})", "", "```text",
                      (f["message"] + "\n\n" + f["details"]).strip()[:2500], "```", ""]

    lint = q["lint"]
    if lint["blocking"] or lint["advisory_in_changed_files"]:
        lines += ["", "## Lint findings in changed files", "",
                  "| Location | Rule | Message | New |", "| --- | --- | --- | :---: |"]
        new = {(f["path"], f["line"], f["code"]) for f in lint["advisory_new"]}
        for f in lint["blocking"]:
            lines.append(f"| `{f['path']}:{f['line']}` | **{f['code']}** (blocking) | "
                         f"{_md(f['message'])} | |")
        for f in lint["advisory_in_changed_files"][:100]:
            mark = "🆕" if (f["path"], f["line"], f["code"]) in new else ""
            lines.append(f"| `{f['path']}:{f['line']}` | {f['code']} | {_md(f['message'])} "
                         f"| {mark} |")

    hy = q["hygiene"]
    if hy and (hy.get("errors") or hy.get("warnings")):
        lines += ["", "## Repository hygiene", ""]
        for level, items in (("❌", hy.get("errors", [])), ("⚠️", hy.get("warnings", []))):
            for f in items:
                loc = f" `{f['path']}{':' + str(f['line']) if f.get('line') else ''}`" \
                    if f.get("path") else ""
                lines.append(f"- {level} **{f['check']}**{loc} — {f['message']}")

    xfails = [(t["label"], x) for t in q["tests"] for x in t["xfails"]]
    if xfails:
        seen, lines_x = set(), []
        for _label, x in xfails:
            if x["test"] in seen:
                continue
            seen.add(x["test"])
            lines_x.append(f"- `{x['test']}` — {x['reason'] or 'no reason given'}")
        lines += ["", "## Known issues pinned by tests (xfail)", "",
                  "Documented defects the suite tracks; each flips to a failure once fixed, "
                  "so the marker gets removed.", "", *lines_x]

    skips = Counter()
    for t in q["tests"]:
        skips.update(t["skip_reasons"])
    if skips:
        lines += ["", "<details><summary>Skipped tests by reason</summary>", "",
                  "| Reason | Count |", "| --- | ---: |"]
        lines += [f"| {_md(reason)} | {n} |" for reason, n in skips.most_common()]
        lines += ["", "</details>"]

    if report["commits"]:
        lines += ["", "## Commits", "", "| Commit | Author | Date | Subject |",
                  "| --- | --- | --- | --- |"]
        for c in report["commits"][:100]:
            lines.append(f"| `{c['short']}` | {c['author']} | {c['date']} | {_md(c['subject'])} |")

    collapsed = report.get("collapsed", {})
    listed = len(report["changes"])
    note = (f" (+{sum(v['files'] for v in collapsed.values())} summarised below)"
            if collapsed else "")
    lines += ["", f"<details><summary>All {listed} changed project files{note}</summary>", "",
              "| Status | File | Module | + | − |", "| :---: | --- | --- | ---: | ---: |"]
    for area, info in collapsed.items():
        top = ", ".join(f"{k} ×{v}" for k, v in list(info["top_components"].items())[:6])
        lines.append(f"| ∑ | **{info['files']} files** — {AREAS[area]} ({top}) | {area} | "
                     f"{info['additions']} | {info['deletions']} |")
    for c in report["changes"][:2000]:
        path = f"`{c['old_path']}` → `{c['path']}`" if c["old_path"] else f"`{c['path']}`"
        adds = "bin" if c["binary"] else c["additions"]
        dels = "bin" if c["binary"] else c["deletions"]
        lines.append(f"| {c['status']} | {path} | {c['area']}/{c['component']} | {adds} "
                     f"| {dels} |")
    if len(report["changes"]) > 2000:
        lines.append(f"| … | {len(report['changes']) - 2000} more (see changes.json) | | | |")
    lines += ["", "</details>", "", "<details><summary>git diff --stat</summary>", "",
              "```text", report["diffstat"].rstrip()[-20000:], "```", "", "</details>", "",
              "---", "_Generated by `scripts/ci/pr_report.py` — see docs/ci-cd.md._", ""]
    return "\n".join(lines)


def render_comment(report: dict, artifact_hint: str = "") -> str:
    lines = [COMMENT_MARKER, f"## {ICON[report['verdict']]} CI report — {_title(report)}", "",
             _headline(report), ""]
    if report["verdict_reasons"]:
        lines += [f"- {r}" for r in report["verdict_reasons"]] + [""]
    lines += ["### Checks", "", *_checks_table(report), "",
              "### Needs attention", "", *_attention_list(report), "",
              "### Changes by module", "", *_areas_table(report), ""]
    if artifact_hint:
        lines += [artifact_hint, ""]
    text = "\n".join(lines)
    return text if len(text) < 60000 else text[:60000] + "\n\n_(truncated)_\n"


def _md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


# ── Index of archived reports ──────────────────────────────────────────────────

def build_index(root: Path) -> dict:
    entries = []
    for summary_path in sorted(root.glob("*/summary.json")):
        data = load_json(summary_path)
        if not data:
            continue
        meta = data.get("meta", {})
        entries.append({
            "folder": summary_path.parent.name,
            "number": meta.get("number"),
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "generated_at": meta.get("generated_at", ""),
            "verdict": data.get("verdict", "unverified"),
            "files": data.get("totals", {}).get("files", 0),
            "additions": data.get("totals", {}).get("additions", 0),
            "deletions": data.get("totals", {}).get("deletions", 0),
            "tests": data.get("tests", {}),
            "areas": sorted(data.get("areas", {})),
            "head": data.get("range", {}).get("head", ""),
        })
    entries.sort(key=lambda e: (e["number"] or 0, e["generated_at"]), reverse=True)
    return {"generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reports": entries}


INDEX_HEADER = """# PR reports

One folder per merged pull request, written by CI when the PR lands on
`main` (see [docs/ci-cd.md](../docs/ci-cd.md)). Each folder holds:

| File | Contents |
| --- | --- |
| `README.md` | The human-readable report: verdict, checks, what needs attention, \
changes by module, coverage of the added lines, lint, hygiene, known issues, commits, every file |
| `summary.json` | Verdict and headline numbers — what this index is built from |
| `changes.json` | Every changed file (status, module, component, +/−), per-module totals, commits |
| `quality.json` | Test results per platform, coverage, lint, hygiene and CI job results |
| `diffstat.txt` | `git diff --stat` of the change set |

Reports for open PRs are posted as a PR comment and attached to the workflow
run as the `pr-report` artifact. `local/` holds reports generated on a
developer machine and is git-ignored.
"""


def render_index(index: dict) -> str:
    lines = [INDEX_HEADER, "## Reports", ""]
    if not index["reports"]:
        return "\n".join(lines + ["_No reports archived yet._", ""])
    lines += ["| PR | Title | Author | Verdict | Files | +/− | Tests | Modules | Date |",
              "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |"]
    for e in index["reports"]:
        pr = f"#{e['number']}" if e["number"] else e["folder"]
        t = e["tests"] or {}
        tests = f"{t.get('passed', 0)}✓ {t.get('failed', 0) + t.get('errors', 0)}✗" if t else "—"
        modules = ", ".join(a for a in e["areas"] if a not in ("environment", "generated"))
        lines.append(f"| [{pr}]({e['folder']}/README.md) | {_md(e['title'])} | {e['author']} | "
                     f"{ICON.get(e['verdict'], '')} {e['verdict']} | {e['files']} | "
                     f"+{e['additions']}/−{e['deletions']} | {tests} | {modules} | "
                     f"{e['generated_at'][:10]} |")
    return "\n".join(lines) + "\n"


def write_index(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    index = build_index(root)
    _dump(root / "index.json", index)
    (root / "README.md").write_text(render_index(index), encoding="utf-8")
    return index


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build", help="generate a report for a change set")
    b.add_argument("--base", help="base ref (default: merge-base with origin/main)")
    b.add_argument("--head", default="HEAD")
    b.add_argument("--artifacts", help="directory holding downloaded CI artifacts")
    b.add_argument("--jobs-json", help="JSON of CI job results (toJSON(needs))")
    b.add_argument("--out", type=Path, help="report folder (default: pr-reports/PR-NNNN "
                                             "or pr-reports/local/<branch>)")
    b.add_argument("--comment-out", type=Path, help="also write a PR-comment version here")
    b.add_argument("--artifact-hint", default="", help="line appended to the comment")
    b.add_argument("--pr-number")
    b.add_argument("--title")
    b.add_argument("--fail-on-blocked", action="store_true",
                   help="exit 1 when the verdict is 'blocked'")
    i = sub.add_parser("index", help="rebuild the pr-reports index")
    i.add_argument("--root", type=Path, help="default: <repo>/pr-reports")
    args = ap.parse_args(argv)

    if args.cmd == "index":
        root = args.root or repo_root() / "pr-reports"
        index = write_index(root)
        print(f"indexed {len(index['reports'])} report(s) in {root}")
        return 0
    if args.cmd != "build":
        ap.print_help()
        return 2

    report = build(args)
    out = args.out or default_out(report["meta"])
    write_report(report, out)
    comment = render_comment(report, args.artifact_hint)
    if args.comment_out:
        args.comment_out.parent.mkdir(parents=True, exist_ok=True)
        args.comment_out.write_text(comment, encoding="utf-8")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write(comment.replace(COMMENT_MARKER, "") + "\n")
    print(f"{ICON[report['verdict']]} {LABEL[report['verdict']]} — report written to {out}")
    for reason in report["verdict_reasons"]:
        print(f"   - {reason}")
    return 1 if args.fail_on_blocked and report["verdict"] == "blocked" else 0


if __name__ == "__main__":
    sys.exit(main())
