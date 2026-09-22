"""scripts/ci — the pipeline's own tooling (change classification, the
hygiene gate and the PR report), tested against throwaway git repositories.

Global/system git config is isolated so a developer's Git LFS filter driver
cannot rewrite the fixtures: files matching `filter=lfs` are stored exactly
as written, which is what lets these tests commit both proper pointers and
LFS "bypasses".
"""

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

import ci_common
import pr_report
import repo_checks

POINTER = ("version https://git-lfs.github.com/spec/v1\n"
           "oid sha256:" + "a" * 64 + "\nsize 123\n")


class Repo:
    def __init__(self, root: Path):
        self.root = root

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True,
                              capture_output=True, text=True).stdout

    def write(self, path, content):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8", newline="\n")

    def commit(self, message="change", files=None):
        """commit("msg", {path: content}) or commit({path: content})."""
        if isinstance(message, dict):
            message, files = "change", message
        for path, content in (files or {}).items():
            self.write(path, content)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD").strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    empty = tmp_path / "empty.gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "repo"
    root.mkdir()
    r = Repo(root)
    r.git("init", "-q", "-b", "main")
    for key, value in (("user.email", "ci@example.com"), ("user.name", "CI Test"),
                       ("commit.gpgsign", "false"), ("core.autocrlf", "false")):
        r.git("config", key, value)
    r.write(".gitattributes", "*.pt filter=lfs diff=lfs merge=lfs -text\n")
    r.write("central_module/config.py", "A = 1\nB = 2\nC = 3\n")
    r.write("speech_module/router.py", "def route():\n    return 1\n")
    r.write("README.md", "# project\n")
    r.write("old_name.txt", "hello\n")
    r.commit("base")
    r.git("checkout", "-q", "-b", "feature")
    monkeypatch.setenv("CI_REPO_ROOT", str(root))
    return r


# ── ci_common ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path, area, component", [
    ("central_module/manager.py", "central", "core"),
    ("central_module/actions/linux_adapter.py", "central", "actions"),
    ("central_module/nlp/nlu/train.py", "central", "nlp/nlu"),
    ("speech_module/vad.py", "speech", "vad"),
    ("fetch_deepfilternet_model.py", "speech", "model-fetch"),
    ("Mousegesture/hand_cursor_2.py", "gesture", "engine"),
    ("Mousegesture/gesture_config.json", "gesture", "config"),
    ("Mousegesture/gestures/palm.png", "gesture", "gestures"),
    ("Mousegesture/README.md", "docs", "gesture"),
    ("Mousegesture/others/requirements.txt", "dependencies",
     "Mousegesture/others/requirements.txt"),
    ("requirements.txt", "dependencies", "requirements.txt"),
    ("requirements-test.txt", "ci", "requirements-test.txt"),
    ("Models/NLU/joint-distilbert/joint_nlu.pt", "models", "NLU"),
    ("tests/central/test_ipc.py", "tests", "central"),
    (".github/workflows/ci.yml", "ci", "workflows"),
    ("scripts/ci/pr_report.py", "ci", "scripts"),
    ("pyproject.toml", "ci", "pyproject.toml"),
    ("pr-reports/PR-0001/README.md", "reports", "PR-0001"),
    (".venv/Lib/site-packages/numpy/__init__.py", "environment", "numpy"),
    ("speech_module/__pycache__/vad.cpython-311.pyc", "generated", "speech_module"),
    ("README.md", "docs", "README.md"),
    ("Architecture_diagram/HLD_v7.png", "docs", "Architecture_diagram"),
    ("Tools/.gitkeep", "other", "Tools"),
])
def test_classify(path, area, component):
    assert ci_common.classify(path) == (area, component)


def test_every_area_has_a_title():
    assert set(ci_common.SOURCE_AREAS) <= set(ci_common.AREAS)


@pytest.mark.parametrize("numbers, text", [
    ([], ""), ([5], "5"), ([1, 2, 3, 7, 9, 10], "1-3, 7, 9-10"), ({4, 2, 3}, "2-4"),
])
def test_ranges(numbers, text):
    assert ci_common.ranges(numbers) == text


def test_parse_added_lines():
    diff = textwrap.dedent("""\
        diff --git a/x.py b/x.py
        --- a/x.py
        +++ b/x.py
        @@ -3,0 +4,2 @@ def f():
        +    a = 1
        +    b = 2
        @@ -10 +12 @@
        -old
        +new
        diff --git a/gone.py b/gone.py
        --- a/gone.py
        +++ /dev/null
        @@ -1,3 +0,0 @@
        -x
        """)
    assert ci_common.parse_added_lines(diff) == {"x.py": {4, 5, 12}}


def test_changed_files_statuses_and_counts(repo):
    repo.git("mv", "old_name.txt", "new_name.txt")
    repo.commit({"central_module/config.py": "A = 1\nB = 20\nC = 3\nD = 4\n",
                 "speech_module/vad.py": "x = 1\n",
                 "logo.png": b"\x89PNG\x00\x01\x02"})
    repo.git("rm", "-q", "speech_module/router.py")
    repo.commit("delete")
    changes = {c.path: c for c in ci_common.changed_files("main", "HEAD")}
    assert changes["central_module/config.py"].status == "M"
    assert (changes["central_module/config.py"].additions,
            changes["central_module/config.py"].deletions) == (2, 1)
    assert changes["speech_module/vad.py"].status == "A"
    assert changes["speech_module/router.py"].status == "D"
    renamed = changes["new_name.txt"]
    assert renamed.status == "R" and renamed.old_path == "old_name.txt"
    assert changes["logo.png"].binary
    assert changes["speech_module/vad.py"].area == "speech"


def test_added_lines_and_text(repo):
    repo.commit({"central_module/config.py": "A = 1\nB = 20\nC = 3\nD = 4\n"})
    assert ci_common.added_lines("main", "HEAD") == {"central_module/config.py": {2, 4}}
    assert ci_common.added_text("main", "HEAD", "central_module/config.py") == \
        [(2, "B = 20"), (4, "D = 4")]


def test_lfs_attribute_lookup(repo):
    assert ci_common.lfs_tracked(["model.pt", "a/b/weights.pt", "x.py"]) == \
        {"model.pt", "a/b/weights.pt"}
    assert ci_common.lfs_tracked([]) == set()


def test_range_resolution_commits_and_behind(repo):
    feature = repo.commit("feat: add thing", {"speech_module/vad.py": "x = 1\n"})
    main_tip = repo.git("rev-parse", "main").strip()
    assert ci_common.resolve_range(None, "HEAD") == (main_tip, feature)
    assert ci_common.resolve_range(ci_common.ZERO_SHA, "HEAD")[0] == main_tip
    log = ci_common.commits("main", "HEAD")
    assert [c["subject"] for c in log] == ["feat: add thing"]
    assert log[0]["author"] == "CI Test"
    repo.git("checkout", "-q", "main")
    repo.commit("main moves on", {"README.md": "# project v2\n"})
    repo.git("checkout", "-q", "feature")
    assert ci_common.behind_count("main", "HEAD") == 1


def test_blob_helpers(repo):
    repo.commit({"big.bin": b"\x00" * 3000})
    assert ci_common.blob_size("HEAD", "big.bin") == 3000
    assert ci_common.blob_size("HEAD", "missing") is None
    assert ci_common.blob_head("HEAD", "README.md", 3) == b"# p"
    assert ci_common.blob_head("HEAD", "missing") == b""


# ── repo_checks ────────────────────────────────────────────────────────────────

def checks(**kwargs):
    base, head = ci_common.resolve_range("main", "HEAD")
    return repo_checks.run_checks(base, head, **kwargs)


def kinds(items):
    return sorted({f["check"] for f in items})


def test_clean_change_passes(repo):
    repo.commit({"speech_module/vad.py": "def feed():\n    return None\n",
                 "Models/NLU/model.pt": POINTER})
    result = checks()
    assert result["errors"] == [] and result["warnings"] == []
    assert result["files_checked"] == 2


def test_every_error_class_is_caught(repo):
    token = "ghp_" + "A1b2C3d4" * 4 + "Zz9Y"
    repo.commit({
        "README.md": "# project\n<<<<<<< HEAD\nmine\n=======\ntheirs\n>>>>>>> branch\n",
        "central_module/broken.py": "def f(:\n    pass\n",
        "Mousegesture/gesture_config.json": '{"settings": {,}}',
        "speech_module/secrets.py": f'TOKEN = "{token}"\n',
        "Models/NLU/joint_nlu.pt": b"\x80\x02real torch pickle bytes",   # not a pointer
    })
    result = checks()
    assert kinds(result["errors"]) == ["conflict-markers", "invalid-json", "lfs-bypass",
                                       "python-syntax", "secret"]
    conflict = [f for f in result["errors"] if f["check"] == "conflict-markers"]
    assert [f["line"] for f in conflict] == [2, 6]
    secret = next(f for f in result["errors"] if f["check"] == "secret")
    assert (secret["path"], secret["line"]) == ("speech_module/secrets.py", 1)


def test_pre_existing_problems_do_not_block(repo):
    # A secret already on main is not this PR's doing; only ADDED lines count.
    repo.git("checkout", "-q", "main")
    repo.commit({"legacy.py": 'KEY = "AKIAABCDEFGHIJKLMNOP"\n'})
    repo.git("checkout", "-q", "feature")
    repo.git("merge", "-q", "main")
    repo.commit({"legacy.py": 'KEY = "AKIAABCDEFGHIJKLMNOP"\nOTHER = 1\n'})
    assert checks()["errors"] == []


def test_size_limits(repo):
    repo.commit({"data.csv": "x," * 60_000})                        # ~120 KB
    assert kinds(checks(max_file_mb=0.1)["errors"]) == ["large-file"]
    result = checks(max_file_mb=1, warn_file_mb=0.05)
    assert result["errors"] == [] and kinds(result["warnings"]) == ["big-file"]


def test_environment_and_generated_files_warn(repo):
    repo.commit({".venv/Lib/site-packages/pkg/__init__.py": "x = 1\n",
                   "speech_module/__pycache__/vad.cpython-311.pyc": b"\x00\x01"})
    result = checks()
    assert result["errors"] == []
    assert kinds(result["warnings"]) == ["environment", "generated"]


def test_behind_base_warns(repo):
    repo.commit({"speech_module/vad.py": "x = 1\n"})
    repo.git("checkout", "-q", "main")
    repo.commit({"README.md": "# v2\n"})
    repo.git("checkout", "-q", "feature")
    assert kinds(checks()["warnings"]) == ["behind-base"]


def test_cli_exit_codes_and_json(repo, tmp_path, capsys):
    repo.commit({"broken.json": "{"})
    out = tmp_path / "h.json"
    assert repo_checks.main(["--base", "main", "--json", str(out), "--github"]) == 1
    assert json.loads(out.read_text())["errors"][0]["check"] == "invalid-json"
    printed = capsys.readouterr().out
    assert "::error title=hygiene/invalid-json,file=broken.json" in printed
    assert repo_checks.main(["--base", "main", "--no-fail"]) == 0


# ── pr_report: parsing ─────────────────────────────────────────────────────────

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="1" skipped="2" tests="6" time="3.21">
  <testcase classname="tests.a" name="test_ok" time="0.1"/>
  <testcase classname="tests.a" name="test_ok2" time="0.1"/>
  <testcase classname="tests.a" name="test_bad" time="0.1">
    <failure message="assert 1 == 2">E   assert 1 == 2</failure></testcase>
  <testcase classname="tests.b" name="test_setup" time="0.1">
    <error message="fixture exploded">trace</error></testcase>
  <testcase classname="tests.b" name="test_torch" time="0">
    <skipped type="pytest.skip" message="tests/b.py:25: could not import 'torch': No module named 'torch'"/></testcase>
  <testcase classname="tests.c" name="test_known" time="0">
    <skipped type="pytest.xfail" message="zip-slip bug"/></testcase>
</testsuite></testsuites>"""


def test_parse_junit(tmp_path):
    path = tmp_path / "junit-tests-ubuntu-latest.xml"
    path.write_text(JUNIT, encoding="utf-8")
    t = pr_report.parse_junit(path)
    assert t["label"] == "tests-ubuntu-latest"
    assert (t["tests"], t["passed"], t["failed"], t["errors"], t["skipped"], t["xfailed"]) == \
        (6, 2, 1, 1, 1, 1)
    assert t["time"] == 3.2
    assert t["failures"][0] == {"test": "tests.a::test_bad", "kind": "failure",
                                "message": "assert 1 == 2", "details": "E   assert 1 == 2"}
    assert t["skip_reasons"] == {"could not import 'torch'": 1}
    assert t["xfails"] == [{"test": "tests.c::test_known", "reason": "zip-slip bug"}]


def cov_json(tmp_path, name, files):
    path = tmp_path / name
    path.write_text(json.dumps({"files": {
        k: {"executed_lines": e, "missing_lines": m} for k, (e, m) in files.items()}}))
    return path


def test_coverage_union_and_changed_lines(tmp_path):
    linux = cov_json(tmp_path, "coverage-linux.json",
                     {"central_module/x.py": ([1, 2], [3, 4, 5])})
    windows = cov_json(tmp_path, "coverage-windows.json",
                       {"central_module\\x.py": ([1, 3], [2, 4, 5]),
                        "speech_module/y.py": ([1], [])})
    cov = pr_report.load_coverage([linux, windows])
    assert cov["central_module/x.py"]["executed"] == {1, 2, 3}
    assert cov["central_module/x.py"]["statements"] == {1, 2, 3, 4, 5}
    by_area = pr_report.coverage_by_area(cov)
    assert by_area["central"] == {"statements": 5, "covered": 3, "files": 1, "percent": 60.0}
    dc = pr_report.diff_coverage({"central_module/x.py": {3, 4, 5, 99},
                                  "docs.md": {1}}, cov)
    assert (dc["executable"], dc["covered"], dc["percent"]) == (3, 1, 33.3)
    assert dc["files"][0]["uncovered"] == "4-5"
    assert pr_report.diff_coverage({}, cov)["percent"] is None


@pytest.mark.parametrize("filename, expected", [
    ("/home/runner/work/p/p/central_module/ipc.py", "central_module/ipc.py"),
    ("D:\\a\\p\\p\\speech_module\\vad.py", "speech_module/vad.py"),
    ("/elsewhere/checkout/Mousegesture/app.py", "Mousegesture/app.py"),
    ("relative/unknown.py", "relative/unknown.py"),
])
def test_lint_paths_become_repo_relative(filename, expected):
    items = [{"filename": filename, "code": "F401", "message": "unused",
              "location": {"row": 3, "column": 1}}]
    (finding,) = pr_report.normalize_lint(items, ["/home/runner/work/p/p"])
    assert finding == {"path": expected, "line": 3, "code": "F401", "message": "unused"}


# ── pr_report: analysis ────────────────────────────────────────────────────────

def fc(path, status="M", add=1, dele=0):
    c = ci_common.FileChange(path, status, add, dele)
    c.area, c.component = ci_common.classify(path)
    return c


def titles(flags):
    return [f["title"] for f in flags]


def test_attention_flags():
    flags = pr_report.attention([
        fc("requirements.txt"), fc("central_module/nlp/intents.py"),
        fc("speech_module/config.py"), fc("Models/VOSK/x/final.mdl"),
        fc(".venv/Lib/site-packages/a.py"), fc("speech_module/audio_stream.py"),
        fc(".github/workflows/ci.yml"),
    ], behind=2)
    got = titles(flags)
    for expected in ("Dependency pins changed",
                     "Intent catalog changed without a retrained NLU model",
                     "Calibrated constants changed", "Model files changed",
                     "Source changed without test changes",
                     "Virtualenv / bytecode files in the diff",
                     "Needs a hands-on check CI cannot do", "CI/tooling changed",
                     "Branch is 2 commit(s) behind its base"):
        assert expected in got


def test_tests_and_retrained_model_silence_their_flags():
    got = titles(pr_report.attention([
        fc("central_module/nlp/intents.py"), fc("Models/NLU/joint-distilbert/joint_nlu.pt"),
        fc("tests/central/test_nlu.py")], behind=0))
    assert "Source changed without test changes" not in got
    assert "Intent catalog changed without a retrained NLU model" not in got


def test_large_change_flag():
    got = titles(pr_report.attention([fc("central_module/manager.py", add=2000)], 0))
    assert "Large change set (2000 lines)" in got


def test_deleted_source_needs_no_tests():
    got = titles(pr_report.attention([fc("central_module/old.py", status="D")], 0))
    assert "Source changed without test changes" not in got


PASS_TESTS = [{"failed": 0, "errors": 0}]


@pytest.mark.parametrize("kwargs, verdict", [
    (dict(jobs={"lint": {"result": "success"}}, tests=PASS_TESTS), "ready"),
    (dict(jobs={"tests": {"result": "failure"}}), "blocked"),
    (dict(jobs={"lint": {"result": "cancelled"}}), "blocked"),
    (dict(tests=[{"failed": 1, "errors": 0}]), "blocked"),
    (dict(tests=PASS_TESTS, lint_blocking=[{"code": "F821"}]), "blocked"),
    (dict(tests=PASS_TESTS, hygiene={"errors": [{}], "warnings": []}), "blocked"),
    (dict(tests=PASS_TESTS, hygiene={"errors": [], "warnings": [{}]}), "attention"),
    (dict(tests=PASS_TESTS, flags=[{"level": "warn", "title": "x"}]), "attention"),
    (dict(tests=PASS_TESTS, flags=[{"level": "info", "title": "x"}]), "ready"),
    (dict(tests=PASS_TESTS, diffcov={"percent": 40.0, "executable": 50}), "attention"),
    (dict(tests=PASS_TESTS, diffcov={"percent": 40.0, "executable": 5}), "ready"),
    (dict(tests=PASS_TESTS, advisory_new=[{}]), "attention"),
    (dict(), "unverified"),
    (dict(jobs={"deps": {"result": "skipped"}}, tests=PASS_TESTS), "ready"),
])
def test_verdict(kwargs, verdict):
    args = dict(jobs=None, tests=[], lint_blocking=[], hygiene=None, flags=[], diffcov=None,
                advisory_new=[])
    args.update(kwargs)
    assert pr_report.decide(**args)[0] == verdict


# ── pr_report: end to end ──────────────────────────────────────────────────────

def test_build_writes_a_structured_report(repo, tmp_path, monkeypatch):
    repo.commit("feat(speech): tune router", {"speech_module/router.py": (
        "def route():\n    return 2\n\n\ndef extra():\n    return 3\n"),
        "tests/speech/test_router.py": "def test_x():\n    pass\n"})
    art = tmp_path / "artifacts"
    (art / "test-results-ubuntu").mkdir(parents=True)
    (art / "test-results-ubuntu" / "junit-tests-ubuntu-latest.xml").write_text(JUNIT)
    cov_json(art / "test-results-ubuntu", "coverage-ubuntu-latest.json",
             {"speech_module/router.py": ([1, 2, 5], [6])})
    (art / "hygiene.json").write_text(json.dumps({"errors": [], "warnings": [],
                                                  "files_checked": 2}))
    (art / "lint-blocking.json").write_text("[]")
    (art / "lint-advisory.json").write_text(json.dumps([{
        "filename": str(repo.root / "speech_module" / "router.py"), "code": "E302",
        "message": "expected 2 blank lines", "location": {"row": 5, "column": 1}}]))
    for key, value in {"REPORT_PR_NUMBER": "12", "REPORT_PR_TITLE": "Tune | router",
                       "REPORT_PR_AUTHOR": "octocat", "REPORT_EVENT": "pull_request",
                       "REPORT_RUN_URL": "https://example.test/run/1",
                       "REPORT_HEAD_REF": "feature"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    comment = tmp_path / "comment.md"
    jobs = json.dumps({"lint": {"result": "success"}, "tests": {"result": "success"},
                       "deps": {"result": "skipped"}})

    code = pr_report.main(["build", "--base", "main", "--artifacts", str(art),
                           "--jobs-json", jobs, "--comment-out", str(comment),
                           "--fail-on-blocked"])
    assert code == 1                                   # the junit fixture has failures
    out = repo.root / "pr-reports" / "PR-0012"
    assert sorted(p.name for p in out.iterdir()) == [
        "README.md", "changes.json", "diffstat.txt", "quality.json", "summary.json"]

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "blocked"
    assert summary["meta"]["number"] == 12 and summary["meta"]["author"] == "octocat"
    assert summary["tests"]["failed"] == 1
    assert summary["changed_line_coverage"] == 66.7        # lines 2,5 covered; 6 not
    assert set(summary["areas"]) == {"speech", "tests"}

    changes = json.loads((out / "changes.json").read_text(encoding="utf-8"))
    assert {f["path"] for f in changes["files"]} == {"speech_module/router.py",
                                                     "tests/speech/test_router.py"}
    quality = json.loads((out / "quality.json").read_text(encoding="utf-8"))
    assert quality["lint"]["advisory_new"][0]["code"] == "E302"

    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# PR #12 — Tune | router")
    for heading in ("## Checks", "## Needs attention", "## Changes by module", "## Coverage",
                    "## Test failures", "## Lint findings in changed files",
                    "## Known issues pinned by tests (xfail)", "## Commits"):
        assert heading in readme
    assert "| feat(speech): tune router |" in readme        # commit table
    text = comment.read_text(encoding="utf-8")
    assert text.startswith(pr_report.COMMENT_MARKER) and "❌" in text


def test_local_build_without_artifacts(repo, monkeypatch):
    for key in ("REPORT_PR_NUMBER", "JOBS_JSON", "GITHUB_STEP_SUMMARY", "REPORT_HEAD_REF"):
        monkeypatch.delenv(key, raising=False)
    repo.commit({"central_module/manager.py": "x = 1\n"})
    assert pr_report.main(["build", "--base", "main"]) == 0
    summary = json.loads((repo.root / "pr-reports" / "local" / "feature" / "summary.json")
                         .read_text(encoding="utf-8"))
    assert summary["verdict"] == "unverified"
    assert "Source changed without test changes" in summary["attention"]


def test_index_lists_reports_newest_first(tmp_path):
    root = tmp_path / "pr-reports"
    for number, verdict in ((3, "ready"), (11, "attention")):
        folder = root / f"PR-{number:04d}"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps({
            "meta": {"number": number, "title": f"PR {number}", "author": "a",
                     "generated_at": "2026-09-01T00:00:00Z"},
            "verdict": verdict, "totals": {"files": 2, "additions": 5, "deletions": 1},
            "tests": {"passed": 10, "failed": 0, "errors": 0},
            "areas": {"central": {}, "environment": {}}, "range": {"head": "abc"}}))
    (root / "local").mkdir()                            # never indexed
    assert pr_report.main(["index", "--root", str(root)]) == 0
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    assert [e["number"] for e in index["reports"]] == [11, 3]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert readme.index("[#11](PR-0011/README.md)") < readme.index("[#3](PR-0003/README.md)")
    assert "10✓ 0✗" in readme and "| central |" in readme


def test_empty_index(tmp_path):
    pr_report.write_index(tmp_path / "pr-reports")
    assert "_No reports archived yet._" in (tmp_path / "pr-reports" / "README.md").read_text(
        encoding="utf-8")
