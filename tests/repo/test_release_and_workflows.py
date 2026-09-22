"""The CD half (scripts/ci/release.py) and the workflow definitions.

Workflow files are code that only runs on GitHub; a typo there fails after
merge. These tests catch the classes of mistake that parse as valid YAML but
break the pipeline: truncated expressions (a " #" in a plain scalar starts a
YAML comment), unpinned actions, jobs the gate forgets, scripts that do not
exist, permissions wider than needed.
"""

import json
import re
import zipfile

import pytest

import release
from helpers.paths import ROOT

yaml = pytest.importorskip("yaml")

WORKFLOWS = ROOT / ".github" / "workflows"


def load(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def strings(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from strings(v)
    elif isinstance(node, str):
        yield node


# ── Workflow definitions ───────────────────────────────────────────────────────

@pytest.fixture(scope="module", params=sorted(p.name for p in WORKFLOWS.glob("*.yml")))
def workflow(request):
    return request.param, load(request.param)


def test_expressions_survive_yaml_parsing(workflow):
    name, data = workflow
    for s in strings(data):
        assert s.count("${{") == s.count("}}"), f"{name}: truncated expression in {s!r}"


def test_actions_are_pinned_to_a_version(workflow):
    name, data = workflow
    uses = [step["uses"] for job in data["jobs"].values()
            for step in job.get("steps", []) if "uses" in step]
    uses += [job["uses"] for job in data["jobs"].values() if "uses" in job]
    for ref in uses:
        if ref.startswith("./"):
            assert (ROOT / ref).exists(), f"{name}: {ref} does not exist"
        else:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@v\d+(\.\d+)*", ref), f"{name}: unpinned {ref}"


def test_default_token_permissions_are_read_only(workflow):
    name, data = workflow
    top = data.get("permissions", {})
    if name == "labeler.yml":
        assert top == {"contents": "read", "pull-requests": "write"}
    else:
        assert top == {"contents": "read"}, f"{name}: widen per job, not per workflow"


def test_every_job_has_a_timeout(workflow):
    name, data = workflow
    for job_name, job in data["jobs"].items():
        if "uses" in job:
            continue                                  # reusable workflow call
        assert "timeout-minutes" in job, f"{name}: job {job_name} can hang for 6 h"


def test_referenced_scripts_exist(workflow):
    name, data = workflow
    for s in strings(data):
        for path in re.findall(r"scripts/ci/[\w.]+\.(?:py|sh)", s):
            assert (ROOT / path).exists(), f"{name}: {path} is referenced but missing"


def test_needs_refer_to_real_jobs(workflow):
    name, data = workflow
    jobs = data["jobs"]
    for job_name, job in jobs.items():
        needs = job.get("needs", [])
        for dep in [needs] if isinstance(needs, str) else needs:
            assert dep in jobs, f"{name}: {job_name} needs unknown job {dep}"


def test_ci_gate_covers_every_check():
    jobs = load("ci.yml")["jobs"]
    checks = set(jobs) - {"gate", "report"}
    assert set(jobs["gate"]["needs"]) == checks
    assert jobs["gate"]["if"] == "always()"
    assert set(jobs["report"]["needs"]) == checks


def test_only_the_report_job_can_write():
    jobs = load("ci.yml")["jobs"]
    writers = {name for name, job in jobs.items()
               if "write" in json.dumps(job.get("permissions", {}))}
    assert writers == {"report"}


def test_ci_triggers():
    data = load("ci.yml")
    on = data.get(True) or data.get("on")                 # YAML 1.1 reads `on` as True
    assert set(on) == {"pull_request", "push", "workflow_dispatch", "workflow_call"}
    assert on["push"]["branches"] == ["main"]


def test_untrusted_pr_fields_never_reach_a_shell_directly():
    # Titles and branch names are attacker-controlled; they may only flow
    # through env: blocks, never be interpolated into `run:` scripts.
    for path in WORKFLOWS.glob("*.yml"):
        data = load(path.name)
        for job in data["jobs"].values():
            for step in job.get("steps", []):
                run = step.get("run", "")
                untrusted = (r"\$\{\{\s*github\."
                             r"(event\.pull_request\.(title|body|head\.ref)|head_ref)")
                assert not re.search(untrusted, run), f"{path.name}: {step.get('name')}"


def test_release_needs_green_ci_and_only_runs_on_tags():
    data = load("release.yml")
    on = data.get(True) or data.get("on")
    assert on == {"push": {"tags": ["v*"]}}
    assert data["jobs"]["ci"]["uses"] == "./.github/workflows/ci.yml"
    assert data["jobs"]["publish"]["needs"] == "ci"


def test_labeler_rules_cover_every_source_area():
    rules = yaml.safe_load((ROOT / ".github" / "labeler.yml").read_text(encoding="utf-8"))
    for label in ("area: central", "area: speech", "area: gesture", "area: models"):
        assert label in rules


def test_ruff_pin_is_shared_by_ci_and_local():
    pins = (ROOT / "requirements-test.txt").read_text(encoding="utf-8")
    assert re.search(r"^ruff==\d+\.\d+\.\d+", pins, re.M)


# ── release.py ────────────────────────────────────────────────────────────────

def summary(number, head, **extra):
    return {"meta": {"number": number, "title": f"Change {number}", "author": "dev"},
            "range": {"head": head}, "areas": {"speech": {}, "environment": {}},
            "tests": {"passed": 10, "failed": 0, "errors": 0},
            "attention": ["Calibrated constants changed"], **extra}


def test_release_notes_from_pr_reports():
    notes = release.render_notes("v1.1.0", "v1.0.0",
                                 [summary(4, "a"), summary(7, "b")], [])
    assert notes.startswith("# v1.1.0\n\nChanges since v1.0.0.")
    assert "| #4 | Change 4 | dev | Speech module | 10 passed, 0 failed |" in notes
    assert "Committed virtualenv" not in notes                 # noise areas hidden
    assert "- Calibrated constants changed" in notes
    assert "src.zip" in notes


def test_release_notes_fall_back_to_commits():
    notes = release.render_notes("v0.1.0", None, [], ["abc1234 first commit"])
    assert "(first release)" in notes and "- abc1234 first commit" in notes


def test_models_manifest_rendering():
    text = release.render_models("v1", [{"path": "Models/x.pt", "oid": "f" * 64,
                                         "size": 2_500_000}])
    assert "git lfs pull" in text and "| `Models/x.pt` | 2.5 MB |" in text


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    import subprocess
    empty = tmp_path / "empty.gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                              text=True).stdout

    git("init", "-q", "-b", "main")
    git("config", "user.email", "ci@example.com")
    git("config", "user.name", "CI")
    files = {
        ".gitattributes": "*.pt filter=lfs diff=lfs merge=lfs -text\n",
        "central_module/manager.py": "x = 1\n",
        "central_module/__pycache__/manager.cpython-311.pyc": "junk",
        ".venv/Lib/site-packages/pkg.py": "y = 2\n",
        "Models/NLU/joint_nlu.pt": ("version https://git-lfs.github.com/spec/v1\n"
                                    "oid sha256:" + "b" * 64 + "\nsize 265685515\n"),
        "pr-reports/README.md": "# PR reports\n",
    }
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    git("add", "-A")
    git("commit", "-q", "-m", "first")
    git("tag", "v0.1.0")
    monkeypatch.setenv("CI_REPO_ROOT", str(root))
    return root, git


def test_bundle_excludes_environment_and_weights(git_repo, tmp_path):
    root, _ = git_repo
    out = tmp_path / "dist"
    assert release.main(["--tag", "v0.1.0", "--out", str(out)]) == 0
    bundle = out / f"{release.PROJECT}-v0.1.0-src.zip"
    names = zipfile.ZipFile(bundle).namelist()
    prefix = f"{release.PROJECT}-v0.1.0/"
    assert prefix + "central_module/manager.py" in names
    assert not any(".venv" in n or "__pycache__" in n or n.endswith(".pt") or "pr-reports" in n
                   for n in names)
    models = zipfile.ZipFile(bundle).read(prefix + "MODELS.md").decode()
    assert "Models/NLU/joint_nlu.pt" in models and "265.7 MB" in models
    digest, fname = (out / f"{bundle.name}.sha256").read_text().split()
    assert fname == bundle.name and len(digest) == 64
    notes = (out / "RELEASE_NOTES.md").read_text(encoding="utf-8")
    assert "(first release)" in notes and "first" in notes


def test_reports_are_selected_by_tag_range(git_repo):
    root, git = git_repo
    first = git("rev-parse", "HEAD").strip()
    (root / "a.txt").write_text("a")
    git("add", "-A")
    git("commit", "-q", "-m", "second")
    second = git("rev-parse", "HEAD").strip()
    git("tag", "v0.2.0")
    reports = root / "pr-reports"
    for n, head in ((1, first), (2, second)):
        folder = reports / f"PR-{n:04d}"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps(summary(n, head)))
    picked = release.reports_between("v0.1.0", "v0.2.0", reports)
    assert [r["meta"]["number"] for r in picked] == [2]
    assert release.previous_tag("v0.2.0") == "v0.1.0"
    assert release.previous_tag("v0.1.0") is None
