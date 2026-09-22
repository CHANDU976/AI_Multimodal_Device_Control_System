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

_No reports archived yet._
