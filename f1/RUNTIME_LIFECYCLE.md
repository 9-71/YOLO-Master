# Studio runtime lifecycle

This round preserves the existing job ID, trusted path, artifact manifest and
redaction implementation. It does not change Agent code, Ultralytics code or
Gradio components.

## Execution

All four task types follow the same entry point:

`API/UI → JobsManager → CPU/GPU waiting queue → fixed supervisor slot → managed process → F1 dispatcher → handler → YOLO Python API`

On Windows the managed process owns computation and is assigned to a kill-on-close
Job Object before the parent permits execution. Child and grandchild processes
remain in that object. On Linux the managed root is a subreaper guardian; its
computation child runs the same dispatcher and handler. The guardian retains
orphaned dataloader/DDP descendants, including torchrun ranks that create a new
session. No fork of the server's GPU context is used: workers use `spawn`.

Studio calls the dispatcher with `managed=True`, executing the handler on the
computation process's main thread. The old direct-dispatcher thread mode remains
for compatibility outside Studio; it does not provide forced termination.

## Capacity and waiting

- `F1_CPU_CONCURRENCY`: positive integer, default **2**.
- `F1_GPU_CONCURRENCY`: positive integer, default **1** for all accelerator jobs.
- `F1_STOP_GRACE_SECONDS`: 0–30 seconds, default **2**.
- Constructor settings override these server environment settings.
- Explicit `cpu` uses CPU slots; absent device follows existing handler default
  `cpu`. Auto/empty/None, CUDA device IDs/lists and MPS use GPU slots.
- Waiting jobs remain publicly `pending` and do not get a process or a dedicated
  thread. Each slot is retained through execution and cleanup.
- Limits apply to one JobsManager in one service process. Run the Studio API with
  one ASGI worker. Multiple service processes or separate UI/API managers must not
  share a state file; cross-process scheduling/storage coordination is not provided.

The waiting/history records remain in memory and JSON persistence; this is not a
distributed or durable execution queue. There is no priority preemption, resource
prediction, per-GPU scheduling or automatic recovery of unfinished computation.

## Cancellation, timeout and cleanup

`cancel` acknowledges a stop request with HTTP 202. A running job remains active
while cleanup is in progress. Cancellation of a waiting job requires no process
launch and finishes immediately.

Timeout starts when a slot launches the worker and includes interpreter startup
and validation; waiting time does not consume the execution timeout. Cancellation
and timeout use the same stop routine:

1. Send a stop request through a private one-way pipe.
2. On POSIX also send SIGTERM to the process group and known descendants.
3. Wait the configured grace period.
4. Force termination if needed: Windows `TerminateJobObject`; Linux freeze/kill
   descendants, then SIGKILL the owned process group.
5. Join the root, verify no owned computation remains, close IPC/OS handles.
6. Release the slot and persist the final result under the manager lock.

Successful handlers also pass through tree cleanup before `completed` becomes
visible. Results arriving during an accepted cancellation cannot overwrite it.
An OS cleanup failure retains the worker handle and slot, exposes
`WORKER_STOP_FAILED` on the still-running job and retries cleanup.

The stop channel deliberately avoids multiprocessing Event/Queue locks: abruptly
killing a process can leave such locks permanently held. There is no thread-kill
hack, `shell=True`, shell interpolation or client-selected worker executable.

## History and errors

The public status set is `pending | running | completed | failed | cancelled`.

- Cancel: `cancelled` + `USER_CANCELLED`, after tree exit.
- Deadline: `failed` + `TIMEOUT`, after tree exit.
- Worker crash/missing result: `failed` + `WORKER_LOST`.
- Startup/unhandled execution exception: `failed` + `EXECUTION_FAILED`.
- Graceful service shutdown: `failed` + `SERVICE_SHUTDOWN`.
- Startup finds historical pending/running: `failed` + `SERVICE_RESTARTED`, with
  an error timestamp and sanitized log entry, immediately persisted.

FastAPI lifespan loads/reconciles history before serving and shuts the manager
down on exit. Windows kill-on-close containment and the Linux guardian's parent
watch handle abnormal parent termination. Restart never claims that old work was
resumed, and persisted PIDs are not used to signal potentially unrelated processes.

## Lightweight verification

`tests/f1/test_worker_lifecycle.py` uses sleep workers, real children/grandchildren,
separate POSIX sessions and fake YOLO classes behind the actual four handlers.
It checks cancellation, timeout, normal completion, worker/startup failures,
parent exit, shutdown, restart/API errors, limits, and CPU/GPU slot reuse after
each terminal outcome. GPU tests select a GPU resource class but execute only CPU
sleep fixtures. No torch, real training, model/data download or large export is
needed for this suite.

Example command (use a fresh temporary directory per run):

```text
python -m pytest tests/f1/test_worker_lifecycle.py tests/api/test_api_v1.py tests/api/test_security_boundaries.py -o addopts= -p no:cacheprovider --confcutdir=tests/f1 --basetemp=runs/runtime-validation -q
```

The implementation follows Python's [spawn/process lifecycle documentation](https://docs.python.org/3/library/multiprocessing.html)
and Microsoft's [Job Object containment documentation](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).

## Verification record: 2026-09-10

- The supplied official baseline `acce839c7e895d6b179de7f7093fa879e237cc7b` is
  present locally and is an ancestor of the current checkout. Existing working
  tree changes from the first round were retained.
- Windows, Python 3.12: 35 lifecycle tests plus 28 API/security tests passed.
  Two existing symlink tests skipped because Windows denied symlink creation.
- Ubuntu under WSL, Python 3.14: all 35 lifecycle tests and all 30 API/security
  tests passed. The lifecycle fixtures create child and grandchild processes in
  separate POSIX sessions to represent torchrun behavior.
  After the final descendant-verification adjustment, the five directly affected
  Linux completion/cancel/timeout/shutdown/crash tests passed again.
- Changed Python files passed Ruff and formatting checks; changed runtime/test
  files and this document passed codespell. `git diff --check` passed.
- The required broad Ruff check of `ultralytics/ tests/ scripts/ agent/` reported
  2,682 issues outside this round's changes. Broad formatting passed for 672 files.
  Broad spelling checks also report existing taxonomy/vendor/generated-file
  findings; those unrelated files were not edited.

Remaining limits: concurrency is per service process, pending/history records are
not memory-capped, and no real GPU kernels or training were exercised. Process
checks establish absence of live computation; POSIX zombie reaping ultimately
depends on the guardian/OS init. A platform-level refusal to terminate remains
visible as `WORKER_STOP_FAILED` with the slot retained. Only the Windows/Linux
paths have been exercised; other POSIX systems do not have the Linux subreaper
guarantee. Run task execution through JobsManager to obtain these guarantees.
