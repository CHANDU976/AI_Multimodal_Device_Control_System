"""Release packaging for a tag: a clean source bundle + release notes.

    python scripts/ci/release.py --tag v1.0.0 --out dist

Produces in --out:

    <project>-<tag>-src.zip     source at the tag — without the committed .venv,
                                bytecode, archived PR reports or model weights
    <project>-<tag>-src.zip.sha256
    RELEASE_NOTES.md            the pull requests merged since the previous tag
                                (from pr-reports/*/summary.json), else commits

Model weights are Git LFS objects (~1 GB) and are not bundled: the zip holds
MODELS.md instead, listing every weight file with its LFS oid and size and
how to fetch them. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ci_common import (  # noqa: E402
    AREAS, GitError, LFS_HEADER, blob_head, git, lfs_tracked, repo_root,
)

PROJECT = "multimodal-device-control"

# Pathspecs left out of the source bundle.
EXCLUDE = (".venv", "**/__pycache__/**", "*.pyc", "pr-reports", "Models", "Tools",
           "*.pt", "*.onnx", "*.ckpt.best")


def previous_tag(tag: str) -> str | None:
    try:
        return git("describe", "--tags", "--abbrev=0", f"{tag}^").strip() or None
    except GitError:
        return None


def is_ancestor(commit: str, of: str) -> bool:
    try:
        git("merge-base", "--is-ancestor", commit, of)
        return True
    except GitError:
        return False


def reports_between(prev: str | None, tag: str, root: Path | None = None) -> list[dict]:
    """Archived PR reports whose head landed in (prev, tag]."""
    root = root or repo_root() / "pr-reports"
    found = []
    for path in sorted(root.glob("*/summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        head = data.get("range", {}).get("head")
        if not head or not is_ancestor(head, tag):
            continue
        if prev and is_ancestor(head, prev):
            continue
        found.append(data)
    found.sort(key=lambda d: d.get("meta", {}).get("number") or 0)
    return found


def lfs_manifest(tag: str) -> list[dict]:
    """[{path, oid, size}] for every LFS-tracked file at *tag*."""
    files = [line.split("\t", 1)[1] for line in git("ls-tree", "-r", tag).splitlines()
             if "\t" in line and not line.split("\t", 1)[1].startswith(".venv/")]
    out = []
    for path in sorted(lfs_tracked(files, rev=tag)):
        text = blob_head(tag, path, 512).decode("utf-8", "replace")
        if not text.startswith(LFS_HEADER.decode()):
            continue
        oid = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.M)
        size = re.search(r"^size (\d+)$", text, re.M)
        out.append({"path": path, "oid": oid.group(1) if oid else "",
                    "size": int(size.group(1)) if size else 0})
    return out


def render_models(tag: str, manifest: list[dict]) -> str:
    total = sum(m["size"] for m in manifest)
    lines = [f"# Model weights for {tag}", "",
             f"This bundle leaves out the {len(manifest)} Git LFS weight files "
             f"({total / 1e9:.2f} GB). Fetch them with Git LFS from the repository:", "",
             "```bash", "git lfs install", "git clone <repository-url>",
             f"git checkout {tag} && git lfs pull", "```", "",
             "Then copy `Models/` and `Mousegesture/model/` next to this source tree.", "",
             "| File | Size | SHA-256 (LFS oid) |", "| --- | ---: | --- |"]
    lines += [f"| `{m['path']}` | {m['size'] / 1e6:.1f} MB | `{m['oid']}` |" for m in manifest]
    return "\n".join(lines) + "\n"


def render_notes(tag: str, prev: str | None, reports: list[dict], commits: list[str]) -> str:
    since = f"since {prev}" if prev else "(first release)"
    lines = [f"# {tag}", "", f"Changes {since}.", ""]
    if reports:
        lines += ["## Pull requests", "", "| PR | Title | Author | Modules | Tests at merge |",
                  "| --- | --- | --- | --- | --- |"]
        for r in reports:
            m, t = r.get("meta", {}), r.get("tests", {}) or {}
            modules = ", ".join(AREAS.get(a, a) for a in r.get("areas", {})
                                if a not in ("environment", "generated", "reports"))
            tests = (f"{t.get('passed', 0)} passed, {t.get('failed', 0) + t.get('errors', 0)}"
                     f" failed") if t else "—"
            number = f"#{m['number']}" if m.get("number") else "—"
            lines.append(f"| {number} | {m.get('title', '')} | {m.get('author', '')} | "
                         f"{modules} | {tests} |")
        attention = sorted({a for r in reports for a in r.get("attention", [])})
        if attention:
            lines += ["", "### Flagged for review during these merges", ""]
            lines += [f"- {a}" for a in attention]
    if commits:
        lines += ["", "## Commits", ""] + [f"- {c}" for c in commits]
    lines += ["", "## Downloads", "",
              f"- `{PROJECT}-{tag}-src.zip` — source without the committed virtualenv or "
              "model weights (see `MODELS.md` inside)", "- `*.sha256` — checksum", ""]
    return "\n".join(lines)


def build_bundle(tag: str, out: Path) -> Path:
    name = f"{PROJECT}-{tag}"
    raw = git("archive", "--format=zip", f"--prefix={name}/", tag, "--", ".",
              *(f":(exclude){p}" for p in EXCLUDE), binary=True)
    buf = io.BytesIO(raw)
    with zipfile.ZipFile(buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}/MODELS.md", render_models(tag, lfs_manifest(tag)))
    path = out / f"{name}-src.zip"
    path.write_bytes(buf.getvalue())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / f"{path.name}.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=Path("dist"))
    args = ap.parse_args(argv)

    git("rev-parse", "--verify", f"{args.tag}^{{commit}}")
    args.out.mkdir(parents=True, exist_ok=True)
    prev = previous_tag(args.tag)
    reports = reports_between(prev, args.tag)
    rng = f"{prev}..{args.tag}" if prev else args.tag
    commits = [] if reports else git("log", "--no-merges", "--format=%h %s", rng).splitlines()
    (args.out / "RELEASE_NOTES.md").write_text(
        render_notes(args.tag, prev, reports, commits), encoding="utf-8")
    bundle = build_bundle(args.tag, args.out)
    print(f"built {bundle} ({bundle.stat().st_size / 1e6:.1f} MB) and RELEASE_NOTES.md "
          f"({len(reports)} PR report(s) since {prev or 'the beginning'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
