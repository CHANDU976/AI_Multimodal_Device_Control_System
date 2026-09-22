"""Repository hygiene gate for a change set (a PR, or a push to main).

    python scripts/ci/repo_checks.py --base origin/main --head HEAD --json hygiene.json

Errors fail the build; warnings are reported (and surface in the PR report)
but never block. Everything is judged on what the change set ADDS, so the
state the repository is already in never turns a PR red on its own.

Errors
  conflict-markers   unresolved <<<<<<< / >>>>>>> left in a text file
  invalid-json       a changed .json file does not parse
  python-syntax      a changed .py file does not compile
  lfs-bypass         a file matching an LFS rule in .gitattributes was
                     committed as a real blob instead of an LFS pointer
  large-file         a non-LFS blob over --max-file-mb (default 10 MB)
  secret             something shaped like a credential was added

Warnings
  environment        changes under .venv/ (the committed virtualenv)
  generated          __pycache__ / *.pyc changes
  big-file           a non-LFS blob over --warn-file-mb (default 2 MB)
  behind-base        the branch is behind its base (rebase before merging)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ci_common import (  # noqa: E402
    LFS_HEADER, Finding, behind_count, blob_head, blob_size, changed_files,
    added_text, git, is_text_path, lfs_tracked, resolve_range,
)

SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b"),
    "GitHub fine-grained token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "OpenAI/Anthropic-style key": re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{32,}\b"),
}

_CONFLICT = re.compile(r"^(<{7}|>{7})(?: |$)")


def run_checks(base: str, head: str, cwd: Path | None = None, *, max_file_mb: float = 10.0,
               warn_file_mb: float = 2.0) -> dict:
    changes = changed_files(base, head, cwd=cwd)
    present = [c for c in changes if c.status != "D"]
    errors: list[Finding] = []
    warnings: list[Finding] = []

    env = [c.path for c in present if c.area == "environment"]
    gen = [c.path for c in present if c.area == "generated"]
    if env:
        warnings.append(Finding("environment", f"{len(env)} file(s) under .venv/ changed — the "
                                "virtualenv should not travel with a PR", extra={"count": len(env),
                                                                               "sample": env[:10]}))
    if gen:
        warnings.append(Finding("generated", f"{len(gen)} bytecode file(s) changed "
                                "(__pycache__/*.pyc)",
                                extra={"count": len(gen), "sample": gen[:10]}))

    # Everything below looks at project files; the virtualenv is reported above.
    project = [c for c in present if c.area not in ("environment", "generated")]
    lfs = lfs_tracked([c.path for c in project], rev=head, cwd=cwd)

    for c in project:
        size = blob_size(head, c.path, cwd=cwd) or 0
        if c.path in lfs:
            if not blob_head(head, c.path, len(LFS_HEADER), cwd=cwd).startswith(LFS_HEADER):
                errors.append(Finding("lfs-bypass", "matches an LFS rule in .gitattributes but "
                                      "was committed as a regular blob — run `git lfs install`, "
                                      "then re-add the file", c.path, extra={"bytes": size}))
            continue
        if size > max_file_mb * 1024 * 1024:
            errors.append(Finding("large-file", f"{size / 1048576:.1f} MB non-LFS blob "
                                  f"(limit {max_file_mb:g} MB) — track it with Git LFS",
                                  c.path, extra={"bytes": size}))
        elif size > warn_file_mb * 1024 * 1024:
            warnings.append(Finding("big-file", f"{size / 1048576:.1f} MB non-LFS blob",
                                    c.path, extra={"bytes": size}))

        if c.binary or not is_text_path(c.path) or size > 5 * 1024 * 1024:
            continue
        content = blob_head(head, c.path, size + 1, cwd=cwd).decode("utf-8", "replace")

        if c.path.endswith(".py"):
            try:
                ast.parse(content, filename=c.path)
            except SyntaxError as exc:
                errors.append(Finding("python-syntax", f"{exc.msg}", c.path, exc.lineno or 0))
        elif c.path.endswith(".json"):
            try:
                json.loads(content)
            except ValueError as exc:
                errors.append(Finding("invalid-json", str(exc), c.path,
                                      getattr(exc, "lineno", 0)))

        for line_no, text in added_text(base, head, c.path, cwd=cwd):
            if _CONFLICT.match(text):
                errors.append(Finding("conflict-markers", "unresolved merge conflict marker",
                                      c.path, line_no))
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    errors.append(Finding("secret", f"looks like a {label} — revoke it and "
                                          "remove it from history", c.path, line_no))

    behind = behind_count(base, head, cwd=cwd)
    if behind:
        warnings.append(Finding("behind-base", f"branch is {behind} commit(s) behind its base — "
                                "rebase or merge the base before merging", extra={"count": behind}))

    return {
        "base": base, "head": head,
        "files_checked": len(project),
        "errors": [f.to_dict() for f in errors],
        "warnings": [f.to_dict() for f in warnings],
    }


def render(result: dict) -> str:
    lines = [f"Repository hygiene: {len(result['errors'])} error(s), "
             f"{len(result['warnings'])} warning(s) across {result['files_checked']} file(s)"]
    for kind, items in (("ERROR", result["errors"]), ("warning", result["warnings"])):
        for f in items:
            where = f"{f['path']}:{f['line']}" if f.get("line") else f.get("path", "")
            lines.append(f"  {kind:7s} [{f['check']}] {where}  {f['message']}".rstrip())
    return "\n".join(lines)


def annotate(result: dict) -> None:
    """GitHub workflow-command annotations, shown inline on the PR diff."""
    for level, items in (("error", result["errors"]), ("warning", result["warnings"])):
        for f in items:
            loc = f",file={f['path']}" if f.get("path") else ""
            loc += f",line={f['line']}" if f.get("line") else ""
            print(f"::{level} title=hygiene/{f['check']}{loc}::{f['message']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="base ref/sha (default: merge-base with origin/main)")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--json", type=Path, help="write the result here")
    ap.add_argument("--max-file-mb", type=float, default=10.0)
    ap.add_argument("--warn-file-mb", type=float, default=2.0)
    ap.add_argument("--github", action="store_true", help="emit ::error/::warning annotations")
    ap.add_argument("--no-fail", action="store_true", help="exit 0 even with errors")
    args = ap.parse_args(argv)

    git("--version")
    base, head = resolve_range(args.base, args.head)
    result = run_checks(base, head, max_file_mb=args.max_file_mb, warn_file_mb=args.warn_file_mb)
    print(render(result))
    if args.github:
        annotate(result)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 1 if result["errors"] and not args.no_fail else 0


if __name__ == "__main__":
    sys.exit(main())
