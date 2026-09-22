#!/usr/bin/env bash
# Commit a generated PR report into pr-reports/ on a branch (normally main)
# and refresh the pr-reports index. Called by the CI `report` job after a PR
# merges; safe to re-run (re-archiving the same PR replaces its folder).
#
#   scripts/ci/archive_report.sh <report-dir> <label> [branch]
#
# Retries on top of the latest branch tip, so two merges landing at once
# both get archived instead of one push rejecting the other.
set -euo pipefail

report_dir="${1:?report directory}"
label="${2:?label, e.g. 'PR #12'}"
branch="${3:-main}"
name="$(basename "$report_dir")"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

# Keep the generated report outside the work tree while branches move.
stash="$(mktemp -d)"
cp -R "$report_dir" "$stash/$name"
rm -rf "$report_dir"

for attempt in 1 2 3 4 5; do
  git fetch --quiet origin "$branch"
  git checkout --quiet --force -B ci-report-archive "origin/$branch"
  mkdir -p pr-reports
  rm -rf "pr-reports/$name"
  cp -R "$stash/$name" "pr-reports/$name"
  python scripts/ci/pr_report.py index --root pr-reports
  git add pr-reports
  if git diff --cached --quiet; then
    echo "pr-reports/$name is already up to date"
    exit 0
  fi
  git commit --quiet -m "ci: archive report for $label [skip ci]"
  if git push --quiet origin "HEAD:$branch"; then
    echo "archived pr-reports/$name to $branch"
    exit 0
  fi
  echo "push rejected (attempt $attempt) — retrying on the new $branch tip"
  sleep $((attempt * 3))
done

echo "::warning::could not archive pr-reports/$name to $branch (branch protection?)"
exit 1
