# Incremental Changed-File Format Gate Design

**Date:** 2026-07-25

## Context

The changed-file quality command now passes Ruff lint and codespell, but Ruff format still reports 30 tracked Python
files that were already unformatted at `HEAD`. Reformatting all of them in the active Mixture contract batch would
produce unrelated churn, while dropping format verification would allow new debt. The quality command needs a usable
default for incremental development and an explicit full-enforcement mode for cleanup branches and CI migrations.

## Decision

The default discovered-file workflow will classify each changed Python file against a Git baseline (`HEAD`, or the
revision supplied by `--base`). New files and files whose baseline content already passes Ruff format remain in the
format-check command. Files whose baseline content is already unformatted are skipped from formatting only, remain in
Ruff lint and codespell, and are printed as baseline debt. Explicit file arguments remain strict because they express
an intentional scope. `--strict-format` disables the exemption and checks every selected Python file.

Baseline content is read with `git show` and checked through Ruff's stdin interface, so the worktree is never mutated.
Ruff exit code 0 means formatted, 1 means baseline debt, and other exit codes fail the quality command as an
infrastructure error. Missing paths at the baseline are treated as new files. The existing subprocess argument model
continues to preserve paths containing spaces without shell interpolation.

## Verification

- Unit-test new, preformatted, and legacy-unformatted classifications.
- Unit-test a narrowed formatting command while lint still covers every Python file.
- Unit-test Ruff infrastructure failures and the `--strict-format` CLI flag.
- Run the command against the current worktree: default mode must pass and report the baseline-debt set.
- Run strict mode: it must fail on the remaining unformatted files without modifying them.
- Re-run Ruff lint, codespell, `git diff --check`, and the focused quality test file.

## Execution Record

- Seven focused unit tests pass for filtering, command construction, baseline classification, error handling, explicit
  file precedence, and strict-mode parsing.
- The default command passes on the active worktree, checks 21 eligible Python files for formatting, and reports 35
  baseline-debt files without failing lint or spelling checks.
- The baseline gate detected two newly introduced formatting regressions in `mot/router.py` and
  `vpeft/placement_plan.py`; formatting only those files made the default command pass.
- Strict mode remains effective and reports 28 currently unformatted files with a non-zero exit status.
