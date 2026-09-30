# Studio Core Task Engine

This layer supplies the JobRequest schema, credential sanitization,
path validation, dispatcher FSM, registry and synchronous train / val / predict /
export / diagnose handlers. Install from the checkout with `pip install -e ".[studio]"`.
The handlers use Ultralytics engines and the repository pytest/doctest settings.

Import `studio.dispatcher.JobDispatcherStateMachine` or `studio.handlers` before concurrent
execution; importing handlers registers all five classes. Register extensions before
dispatch threads start. Direct handlers have two steps: `validate_params(params,
constraints)`, then `execute(job_id, params, output_dir)`. The existing result is a
dict containing success, artifacts, metadata and error. The dispatcher returns the
updated JobRequest; its existing FSM and failure codes are retained.

## Trusted caller and safety boundary

Studio Core is an in-process library for trusted callers. The caller owns roots/regex
rules and must validate output_dir and the final job directory before execution.
Handler validation checks model/data paths against those trusted rules; direct
execute calls assume validated input. Do not accept untrusted roots or regex patterns.
allow_shell=false and path_whitelisted=true are required by the dispatcher.
Containment resolves symlinks, rejects broken symlinks and sibling-prefix escapes,
and rejects malformed-only regex rules. Empty roots/patterns reject input paths.

Returned artifact paths are discovery candidates. They do not authorize downloads;
a consuming runtime or service must normalize an authorized manifest and recheck
containment on every read.
append_log and dispatcher error/traceback handling sanitize credentials. Direct
handler results remain internal engine facts and require sanitization before a
public response or persistence boundary.

The legacy direct dispatcher can report cancellation/timeout while a detached daemon
thread is still computing. It does not provide OS process-tree cleanup. The existing
managed=True path executes synchronously and expects external supervision. Core does
not implement a public lifecycle owner, queue, workers, recovery, API or shutdown.

## Upstream compatibility

TrainHandler forwards explicitly supplied imgsz after the same integer normalization
used by val/export. Omitting imgsz preserves the upstream default. The optimizer
metadata summarizes trainer.optimizer_group_audit, recorded by upstream after final
optimizer construction; it does not perform a second audit. Missing audit snapshots
report audited=false. LoRA/router name detection alone is not coverage proof.

## Verification

Offline unit/fake-engine tests prohibit implicit model/data downloads:

```powershell
python -m pytest tests/studio -q
python -m pytest --doctest-modules core studio -q
ruff check core studio tests/studio
ruff format --check core studio tests/studio
codespell core studio tests/studio
```

Real engine tests require both explicit flags and existing local detection weights:

```powershell
$env:STUDIO_TEST_MODEL="D:\path\to\local-model.pt"
$env:YOLO_AUTOINSTALL="false"
python -m pytest tests/studio/test_core_integration.py --slow --studio-integration -q
```

These tests use generated images/data, CPU, one training epoch at imgsz=64, and a
TorchScript export. They never download weights/data. Ordinary upstream --slow does
not opt into them. They verify Core calls, not service lifecycle or checkpoint resume.
Directories containing symlink tests require symlink privileges; skipped cases are
unverified and must be rerun on a capable Windows environment and on Linux.

## Runtime integration responsibilities

A consuming runtime must own admission/output roots, public lifecycle decisions,
process supervision, sanitized snapshots, persistence recovery and manifest authorization.
Core does not provide checkpoint-aware shutdown, an HTTP service or a user interface.
