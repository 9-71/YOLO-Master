# Studio runtime lifecycle

This document describes the current F1 runtime and lifecycle boundaries.

F1 owns task orchestration only. It calls the existing Ultralytics Python API and
does not route production jobs through the Agent runtime.

## Execution

All five task types follow the same backend-owned execution path:

`Gradio UI → StudioJobsApiClient → FastAPI task API → JobsManager → CPU/GPU waiting queue → fixed supervisor slot → managed process → F1 dispatcher → handler → YOLO Python API`

Other API clients join at the FastAPI task API. The Gradio path never constructs
its own `JobsManager`.

The zero-build console at `/`, the optional built React console at `/console`,
and Gradio all call the same REST API. FastAPI always serves the zero-build
console. It serves `/console` only when `web/dist` exists. `start_studio.py` runs
Gradio in its launcher process and starts the API as a child process when needed;
Gradio is not mounted inside the engine.

On Windows the managed process owns computation and is assigned to a kill-on-close
Job Object before the parent permits execution. Child and grandchild processes
remain in that object. On Linux the managed root is a subreaper guardian; its
computation child runs the same dispatcher and handler. The guardian retains
orphaned dataloader/DDP descendants, including torchrun ranks that create a new
session. No fork of the server's GPU context is used: workers use `spawn`.

The API-owned `JobsManager` calls the dispatcher with `managed=True`, executing
the handler on the computation process's main thread. The old direct-dispatcher
thread mode remains for compatibility outside Studio; it does not provide forced
termination.

The dispatcher state machine owns `PENDING -> RUNNING -> COMPLETED/FAILED` inside
the worker. `JobsManager` owns admission and the public lifecycle, including
`CANCELLED`, timeout/shutdown failures, terminal arbitration, and persistence.
The manager only accepts a worker result while the public job is non-terminal.

## Capacity and waiting

- `F1_CPU_CONCURRENCY`: positive integer, default **2**.
- `F1_GPU_CONCURRENCY`: positive integer, default **1** for all accelerator jobs.
- `F1_STOP_GRACE_SECONDS`: 0–30 seconds, default **2**.
- Constructor settings override these server environment settings.
- Explicit `cpu` uses CPU slots; absent device follows existing handler default
  `cpu`. Auto/empty/None, CUDA device IDs/lists and MPS use GPU slots.
- Waiting jobs remain publicly `pending` and do not get a process or a dedicated
  thread. Each slot is retained through execution and cleanup.
- Limits apply to the sole API-owned JobsManager in one service process. Run the
  Studio API with one ASGI worker. Multiple service processes must not share a
  state file; cross-process scheduling/storage coordination is not provided.

The waiting/history records remain in memory and JSON persistence; this is not a
distributed or durable execution queue. There is no priority preemption, resource
prediction, per-GPU scheduling or automatic recovery of unfinished computation.
Persistence uses local temporary-file replacement and is best-effort; an I/O
failure is not promoted to a transactional storage guarantee.

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

## Logs, artifacts and security

The manager publishes ordered structured log events and exposes them through the
HTTP offset/limit cursor API. `JobRequest.append_log()`, manager errors, and
persisted structured fields pass through the credential sanitizer. Arbitrary
third-party output written directly to process stdout/stderr is outside that
structured logging guarantee.

Terminal status, result, and error snapshots are immutable. Historical logs stay
readable, and log events already in flight can be retained without changing the
terminal snapshot.

Handlers collect files below `output_dir/job_id`; the manager normalizes the
result to safe relative artifact IDs. Download requires an exact manifest match.
Client-supplied artifacts are cleared at admission.

At the same boundary the manager forces shell execution off and path whitelisting
on, discards client roots and regex patterns, and independently checks model,
data, and output paths against server-owned roots. Network data sources require
an allowlisted host. Process containment supports stop and cleanup; it is not a
per-job security sandbox, and the API has no authentication or RBAC.

## Lightweight verification

`tests/f1/test_worker_lifecycle.py` uses sleep workers, real children/grandchildren,
separate POSIX sessions and fake YOLO classes behind the four YOLO-backed handlers
(`train`, `val`, `predict`, and `export`). The non-YOLO `diagnose` handler is covered
by the handler inventory and focused handler suites.
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

## Verification record: 2026-09-15

- The official baseline `acce839c7e895d6b179de7f7093fa879e237cc7b` is an
  ancestor of the reviewed commit `0e7a8f83b53c97abc8eb3caedc532f5779fbb086`.
- Windows/Python 3.12 in the repository `.venv`: `412 passed, 11 skipped,
  3 warnings`; combined `f1` + `core` coverage was 78%. The Windows skips require
  symlink privileges. One warning was host-specific pytest cache permission noise.
- Ubuntu/Python 3.10 source-branch CI at the same HEAD: `422 passed, 1 skipped,
  3 warnings`, 78% coverage, smoke 5/5, Ruff lint and format passed.
- CI run: <https://github.com/9-71/YOLO-Master/actions/runs/34793607227>.
  This is source-branch push evidence, not a Tencent required PR check.
- The CI workflow exercises Python F1/API tests and smoke. It does not run Node
  lint, TypeScript checks, or a Vite production build.

Remaining limits: concurrency is per service process, pending/history records are
not memory-capped, and no distributed coordination is provided. Process checks
establish absence of live computation; POSIX zombie reaping ultimately depends on
the guardian/OS init. A platform-level refusal to terminate remains visible as
`WORKER_STOP_FAILED` with the slot retained. Only Windows and Linux have been
exercised; other POSIX systems do not have the Linux subreaper guarantee. Run
task execution through `JobsManager` to obtain these lifecycle guarantees.

---

**Document version**: 1.2.0

**Last synchronized**: 2026-09-15
