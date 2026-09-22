"""Shared helpers for the CI scripts: git plumbing, diff parsing and the map
from repository paths to project areas.

Standard library only — these run before (and independently of) any project
dependency install, locally and in CI alike.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
ZERO_SHA = "0" * 40


def repo_root() -> Path:
    """The repository the scripts operate on (CI_REPO_ROOT overrides, for tests)."""
    return Path(os.environ.get("CI_REPO_ROOT") or ROOT)


# ── git ────────────────────────────────────────────────────────────────────────

class GitError(RuntimeError):
    pass


def git(*args: str, cwd: Path | None = None, check: bool = True, binary: bool = False,
        stdin: str | None = None):
    """Run git and return stdout (str, or bytes when binary=True)."""
    proc = subprocess.run(["git", *args], cwd=cwd or repo_root(), capture_output=True,
                          input=stdin.encode("utf-8") if (stdin is not None and binary) else stdin,
                          text=not binary, encoding=None if binary else "utf-8",
                          errors=None if binary else "replace")
    if check and proc.returncode != 0:
        err = proc.stderr if not binary else proc.stderr.decode("utf-8", "replace")
        raise GitError(f"git {' '.join(args)} failed: {err.strip()}")
    return proc.stdout


def rev_parse(ref: str, cwd: Path | None = None) -> str:
    return git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd=cwd).strip()


def resolve_range(base: str | None, head: str | None, cwd: Path | None = None) -> tuple[str, str]:
    """(base_sha, head_sha) for a diff. A missing/zero base falls back to the
    merge-base with origin/main, then main, then the head's parent."""
    head_sha = rev_parse(head or "HEAD", cwd)
    if base and base != ZERO_SHA:
        return rev_parse(base, cwd), head_sha
    for candidate in ("origin/main", "main"):
        try:
            mb = git("merge-base", candidate, head_sha, cwd=cwd).strip()
            if mb and mb != head_sha:
                return mb, head_sha
        except GitError:
            continue
    try:
        return rev_parse(f"{head_sha}~1", cwd), head_sha
    except GitError:                                  # root commit: diff against empty tree
        return git("hash-object", "-t", "tree", os.devnull, cwd=cwd).strip(), head_sha


# ── Changed files ──────────────────────────────────────────────────────────────

@dataclass
class FileChange:
    path: str
    status: str                  # A M D R C T
    additions: int = 0
    deletions: int = 0
    binary: bool = False
    old_path: str | None = None
    area: str = "other"
    component: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def changed_files(base: str, head: str, cwd: Path | None = None) -> list[FileChange]:
    """Files changed on head since its merge-base with base (a PR's own diff)."""
    rng = f"{base}...{head}"
    status_raw = git("diff", "--name-status", "-z", "-M", rng, cwd=cwd)
    numstat_raw = git("diff", "--numstat", "-z", "-M", rng, cwd=cwd)

    changes: dict[str, FileChange] = {}
    tokens = status_raw.split("\0")
    i = 0
    while i < len(tokens) and tokens[i]:
        code = tokens[i]
        kind = code[0]
        if kind in "RC":
            old, new = tokens[i + 1], tokens[i + 2]
            changes[new] = FileChange(new, kind, old_path=old)
            i += 3
        else:
            path = tokens[i + 1]
            changes[path] = FileChange(path, kind)
            i += 2

    # numstat -z: "add\tdel\tpath\0" or, for renames, "add\tdel\t\0old\0new\0".
    tokens = numstat_raw.split("\0")
    i = 0
    while i < len(tokens) and tokens[i]:
        add, dele, path = tokens[i].split("\t", 2)
        if path == "":
            path = tokens[i + 2]
            i += 3
        else:
            i += 1
        fc = changes.get(path)
        if fc is None:
            continue
        if add == "-" and dele == "-":
            fc.binary = True
        else:
            fc.additions, fc.deletions = int(add), int(dele)

    for fc in changes.values():
        fc.area, fc.component = classify(fc.path)
    return sorted(changes.values(), key=lambda c: c.path)


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def parse_added_lines(unified_diff: str) -> dict[str, set[int]]:
    """{path: {new-side line numbers added}} from `git diff -U0` output."""
    added: dict[str, set[int]] = {}
    current = None
    for line in unified_diff.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            if target == "/dev/null":
                current = None
            else:
                current = target[2:] if target.startswith("b/") else target
            if current is not None:
                added.setdefault(current, set())
        elif line.startswith("@@") and current is not None:
            m = _HUNK.match(line)
            if m:
                start, count = int(m.group(1)), int(m.group(2) or 1)
                added[current].update(range(start, start + count))
    return added


def added_lines(base: str, head: str, paths: list[str] | None = None,
                cwd: Path | None = None) -> dict[str, set[int]]:
    """*paths* may be file paths or git pathspecs (":(exclude).venv")."""
    args = ["diff", "-U0", "--no-color", "--no-ext-diff", "-M", f"{base}...{head}"]
    if not paths:
        return parse_added_lines(git(*args, cwd=cwd))
    added: dict[str, set[int]] = {}
    for i in range(0, len(paths), 200):          # stay under OS command-line limits
        for path, lines in parse_added_lines(git(*args, "--", *paths[i:i + 200],
                                                 cwd=cwd)).items():
            added.setdefault(path, set()).update(lines)
    return added


def added_text(base: str, head: str, path: str, cwd: Path | None = None) -> list[tuple[int, str]]:
    """[(line_no, text)] of lines added to *path* on head."""
    out, current_line = [], 0
    for line in git("diff", "-U0", "--no-color", f"{base}...{head}", "--", path,
                    cwd=cwd).splitlines():
        if line.startswith("@@"):
            m = _HUNK.match(line)
            current_line = int(m.group(1)) if m else 0
        elif line.startswith("+") and not line.startswith("+++"):
            out.append((current_line, line[1:]))
            current_line += 1
    return out


def blob_size(rev: str, path: str, cwd: Path | None = None) -> int | None:
    try:
        return int(git("cat-file", "-s", f"{rev}:{path}", cwd=cwd).strip())
    except (GitError, ValueError):
        return None


def blob_head(rev: str, path: str, n: int = 256, cwd: Path | None = None) -> bytes:
    try:
        return git("cat-file", "blob", f"{rev}:{path}", cwd=cwd, binary=True)[:n]
    except GitError:
        return b""


def lfs_tracked(paths: list[str], rev: str | None = None,
                cwd: Path | None = None) -> set[str]:
    """Subset of *paths* whose gitattributes route them through Git LFS.

    Attributes are read from *rev* (git >= 2.40) or else the index, never the
    working tree: CI uses sparse checkouts, and nested .gitattributes files
    (Mousegesture/.gitattributes holds the *.pt/*.onnx rules) may not be on disk.
    """
    if not paths:
        return set()
    # --stdin: a PR touching thousands of files would overflow the command line.
    stdin = "\x00".join(paths) + "\x00"
    try:
        source = [f"--source={rev}"] if rev else ["--cached"]
        out = git("check-attr", *source, "--stdin", "-z", "filter", cwd=cwd, stdin=stdin)
    except GitError:
        out = git("check-attr", "--cached", "--stdin", "-z", "filter", cwd=cwd, stdin=stdin)
    parts = out.split("\0")
    tracked = set()
    for i in range(0, len(parts) - 2, 3):
        path, _attr, value = parts[i], parts[i + 1], parts[i + 2]
        if value == "lfs":
            tracked.add(path)
    return tracked


def commits(base: str, head: str, cwd: Path | None = None) -> list[dict]:
    fmt = "%H%x1f%h%x1f%an%x1f%ad%x1f%s"
    out = git("log", "--no-merges", f"--format={fmt}", "--date=short",
              f"{base}..{head}", cwd=cwd)
    rows = []
    for line in out.splitlines():
        sha, short, author, date, subject = (line.split("\x1f") + [""] * 5)[:5]
        rows.append({"sha": sha, "short": short, "author": author, "date": date,
                     "subject": subject})
    return rows


def behind_count(base: str, head: str, cwd: Path | None = None) -> int:
    """Commits on base that head does not have yet."""
    try:
        return int(git("rev-list", "--count", f"{head}..{base}", cwd=cwd).strip())
    except (GitError, ValueError):
        return 0


# ── Project areas ──────────────────────────────────────────────────────────────

AREAS = {
    "central": "Central module (orchestrator + NLU)",
    "speech": "Speech module",
    "gesture": "Gesture module",
    "models": "Model weights & configs (Git LFS)",
    "tests": "Tests",
    "ci": "CI/CD & tooling",
    "dependencies": "Dependency pins",
    "docs": "Documentation",
    "reports": "PR reports",
    "environment": "Committed virtualenv (.venv)",
    "generated": "Generated files (__pycache__, *.pyc)",
    "other": "Other",
}

SOURCE_AREAS = ("central", "speech", "gesture")

_CI_FILES = {"pyproject.toml", "requirements-test.txt", ".gitignore", ".gitattributes"}


def classify(path: str) -> tuple[str, str]:
    """(area, component) for a repository path."""
    p = path.replace("\\", "/")
    parts = p.split("/")
    name = parts[-1]
    if p.startswith(".venv/"):
        return "environment", parts[3] if len(parts) > 3 and parts[1] == "Lib" else parts[1]
    if "__pycache__" in parts or name.endswith((".pyc", ".pyo")):
        return "generated", parts[0]
    if re.fullmatch(r"requirements[^/]*\.txt", name) and name != "requirements-test.txt":
        return "dependencies", p
    if p.startswith(("pr-reports/",)):
        return "reports", parts[1] if len(parts) > 2 else ""
    if p.startswith((".github/", "scripts/")) or p in _CI_FILES:
        return "ci", parts[1] if p.startswith(".github/") and len(parts) > 2 else parts[0]
    if p.startswith("tests/"):
        return "tests", parts[1] if len(parts) > 2 else "root"
    if p.startswith("Models/"):
        return "models", parts[1] if len(parts) > 2 else ""
    if p.startswith("central_module/"):
        if p.startswith("central_module/nlp/nlu/"):
            return "central", "nlp/nlu"
        return "central", parts[1] if len(parts) > 2 else "core"
    if p.startswith("speech_module/"):
        return "speech", name.rsplit(".", 1)[0] if len(parts) == 2 else parts[1]
    if p.startswith("Mousegesture/"):
        comp = {"hand_cursor_2.py": "engine", "gesture_fsm.py": "fsm", "app.py": "studio-gui",
                "monitors.py": "monitors", "gesture_config.json": "config",
                "system_monitor.py": "system-monitor", "gesture_test.py": "bench"}
        if len(parts) == 2:
            if name.endswith(".md"):
                return "docs", "gesture"
            return "gesture", comp.get(name, name.rsplit(".", 1)[0])
        return "gesture", parts[1]
    if name.endswith((".md", ".rst")) or p.startswith(("docs/", "Architecture_diagram/")):
        return "docs", parts[0] if len(parts) > 1 else name
    if p == "fetch_deepfilternet_model.py":
        return "speech", "model-fetch"
    return "other", parts[0]


def is_text_path(path: str) -> bool:
    return Path(path).suffix.lower() in {
        ".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg",
        ".conf", ".sh", ".ps1", ".bat", ".html", ".css", ".js", ".drawio", ".csv",
    } or Path(path).name in {"README", "CODEOWNERS", ".gitignore", ".gitattributes"}


def github_env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "")
    return value if value else default


def ranges(numbers) -> str:
    """{1,2,3,7,9,10} → '1-3, 7, 9-10'."""
    nums = sorted(set(numbers))
    if not nums:
        return ""
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = n
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(out)


@dataclass
class Finding:
    check: str
    message: str
    path: str = ""
    line: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
