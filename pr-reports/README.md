# PR reports

One folder per merged pull request, written by CI when the PR lands on
`main` (see [docs/ci-cd.md](../docs/ci-cd.md)). Each folder holds:

| File | Contents |
| --- | --- |
| `README.md` | The human-readable report: verdict, checks, what needs attention, changes by module, coverage of the added lines, lint, hygiene, known issues, commits, every file |
| `summary.json` | Verdict and headline numbers — what this index is built from |
| `changes.json` | Every changed file (status, module, component, +/−), per-module totals, commits |
| `quality.json` | Test results per platform, coverage, lint, hygiene and CI job results |
| `diffstat.txt` | `git diff --stat` of the change set |

Reports for open PRs are posted as a PR comment and attached to the workflow
run as the `pr-report` artifact. `local/` holds reports generated on a
developer machine and is git-ignored.

## Reports

| PR | Title | Author | Verdict | Files | +/− | Tests | Modules | Date |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| [#9](PR-0009/README.md) | Implement speech IPC message generation | CHANDU976 | ⚠️ attention | 5 | +158/−22 | 4620✓ 0✗ | central, speech | 2026-09-27 |
| [#8](PR-0008/README.md) | Bump the actions group with 6 updates | dependabot[bot] | ✅ ready | 3 | +29/−29 | 4620✓ 0✗ | ci | 2026-09-22 |
| [#7](PR-0007/README.md) | Add CI/CD pipeline, test suite and PR reports | CHANDU976 | ⚠️ attention | 67 | +10782/−42 | 4620✓ 0✗ | central, ci, docs, gesture, reports, speech, tests | 2026-09-22 |
