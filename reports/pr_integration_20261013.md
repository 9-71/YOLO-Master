# GitHub integration checkpoint — 2026-10-13

## Submitted local work

Five separate commits are on upstream main: `8c95d93` (MoE regression), `bb43970` (matched MoT benchmark), `25afe71` (MPS preflight/A800 catalog), `9f6262f` (checkpoint routing diagnostics), `0ac9bea` (research findings and scoped PEFT claims).

## Confirmed merged PRs

GitHub REST API confirmed `merged: true` for all four:

- #283: `6c0f235c`, routing observability; 113 targeted tests passed.
- #285: `9287cf84`, precision/dynamic runtime; 169 targeted tests passed, including ONNX Runtime.
- #290: `432b96bf`, DINOv3 cache; initial suite 426 passed/56 skipped/12 failed. After integration fixes, affected 109 passed; training suite 77 passed/4 skipped. Engine/default-config regression subsequently 42 passed.
- #233: `bb13695f`, MoT boundary/hybrid work; 148 targeted tests passed. Retained main telemetry, reference-grid caching, export behavior and existing regression tests.

These results are local macOS results, not a claim that remote CI or A800 training has passed.

## Held integrations

- #299: conflicts resolved and saved as `39b5a2b7` on `codex/rhino-pr-integration`, **not main**. API/UI subset 174 passed/1 skipped; frontend build and ESLint passed; scoped Python Ruff passed. npm audit found zero vulnerabilities after lockfile repair. Full backend/API/UI suite: 419 passed, 1 skipped, 7 failed and 1 teardown error. Four failures plus teardown concern macOS process-tree cleanup; three export failures pass when run separately (4 export tests passed), so suite interaction remains unresolved. Do not merge until lifecycle and isolation failures are addressed. Retained main AGENTS and canonical pytest configuration.
- #292: alternative F1 platform introduces root `api.py`, overlapping #299's `api/` package and a distinct persistence/task runtime. Needs an explicit compatible entry point and integration validation; not merged.
- #295: advanced STAL uses `stal_area_threshold=0.01` as an image-area ratio, while main uses `16.0` as transformed pixel area. PR default YAML also contains both definitions. Must separate keys and preserve main assignment behavior before merge.
- #280: older STAL uses a different assigner signature and small-object parameter names. Must reconcile with existing main STAL and #295 rather than overwrite TAL/loss.
- #303: 3,024 changed files, approximately 877k inserted lines and 37k deletions, with about 156 textual conflicts in initial assessment. Includes broad unrelated changes. Requires scope reduction and independent review of deletions; not safe to merge as an A2-only change.

No force pushes, contributor-branch rewrites, or artificial ours-only merges were used. Remaining PRs stay open. Original local work was backed up before submission. The primary checkout is synchronized to upstream main; the managed integration worktree retains the F1 candidate.
