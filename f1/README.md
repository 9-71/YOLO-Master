# YOLO-Master F1 Task Platform

This document describes the current F1 implementation and runtime boundaries.

F1 is an orchestration layer around the existing Ultralytics YOLO Python API. It
does not replace the training, validation, prediction, or export engines.

## 1. Runtime architecture

```text
Gradio UI -- StudioJobsApiClient --+
Zero-build console (/) ------------+--> FastAPI task API
React/Vite console (/console) -----+          |
Other REST clients ----------------+          v
                                     JobsManager
                                      |       |
                                  CPU queue  GPU queue
                                      |       |
                                  fixed supervisor slots
                                              |
                                              v
                                  ManagedWorker (spawn process)
                                              |
                                              v
                                  JobDispatcherStateMachine
                                              |
                                              v
                                    TaskHandlerRegistry
                                              |
                                              v
                         train / val / predict / export / diagnose
                                              |
                                              v
                                  Ultralytics YOLO Python API
```

The singleton `JobsManager` owned by the FastAPI service is the sole production
lifecycle owner. It owns admission, queues, workers, cancellation, timeout,
terminal publication, history, logs, artifacts, and the state file. Gradio uses
`StudioJobsApiClient`; it does not create a second production manager or open the
state file. The `JobsManager` export in `f1/ui/jobs_tab.py` is a lazy compatibility
shim for legacy imports.

Run the API with one ASGI worker. F1 does not coordinate multiple API processes
or multiple hosts. Task execution itself is not single-process: each running job
uses a spawned managed process.

## 2. Contract and task routing

`core/schema.py` is the canonical contract. Public task types are:

- `predict` -> `f1/handlers/predict.py`
- `train` -> `f1/handlers/train.py`
- `val` -> `f1/handlers/val.py`
- `export` -> `f1/handlers/export.py`
- `diagnose` -> `f1/handlers/diagnose.py`

Importing `f1.handlers` registers all five handlers. The dispatcher resolves the
concrete class through `TaskHandlerRegistry`, validates parameters and security
constraints, injects cooperative cancellation tracking, invokes the handler, and
normalizes the returned result.

The public statuses are:

```text
pending | running | completed | failed | cancelled
```

The worker-side dispatcher state machine permits:

```text
PENDING -> RUNNING -> COMPLETED
    |          |
    +----------+----> FAILED
```

`CANCELLED` is published by `JobsManager`, not by the dispatcher state machine.
For a running job the manager first stops and joins the owned process tree, then
publishes `CANCELLED / USER_CANCELLED`. Terminal status, result, and error
snapshots cannot be overwritten by a late worker result.

## 3. Lifecycle and capacity

- CPU and accelerator jobs use separate queues.
- Defaults are two CPU slots, one accelerator slot, and 100 pending jobs.
- Explicit `device=cpu` uses a CPU slot. Empty/auto values, CUDA device strings,
  multi-GPU strings, and MPS use the accelerator queue.
- A waiting job remains `pending` and owns no process or dedicated thread.
- A full pending queue returns HTTP `429`.
- Timeout starts when a slot launches a worker; queue waiting time is excluded.
- A fresh cancellation request returns HTTP `202`. The caller polls until the
  job reaches a terminal state.
- Graceful shutdown stops pending/running work before the service exits.
- On startup, persisted `pending` or `running` jobs become
  `FAILED / SERVICE_RESTARTED`; execution is not resumed.

Windows uses a kill-on-close Job Object. Linux uses a guardian, process groups,
and subreaper behavior to contain and reap descendants. These mechanisms provide
lifecycle cleanup, not an OS security sandbox. Other POSIX platforms do not have
the same verified subreaper guarantee. See `f1/RUNTIME_LIFECYCLE.md`.

## 4. State, logs, and artifacts

The API manager persists history to `F1_JOBS_STATE_PATH`, defaulting to
`runs/jobs_state.json`. Writes use a temporary file and replace, but persistence
is local and best-effort; it is not a transactional database or durable queue.

Logs are exposed with HTTP offset/limit cursors. `JobRequest.append_log()` and
manager-owned structured log/error paths sanitize credentials before storage and
API delivery. This guarantee applies to controlled job logs, errors, and persisted
fields; arbitrary third-party output written directly to process stdout/stderr is
not captured or guaranteed to pass through the sanitizer.

Terminal status/result/error snapshots are immutable. Historical log entries
remain readable, and log events already in flight may still be retained without
changing the terminal snapshot.

Handlers collect existing files below their job-specific output directory into a
sorted artifact manifest. The API returns safe relative artifact IDs and serves a
file only when that exact ID exists in the stored manifest. It does not expose an
arbitrary directory browser or accept client-preloaded artifact entries.

## 5. Security boundary

At API admission, `JobsManager` resets client-controlled lifecycle state and
enforces the server policy:

- `allow_shell = false`
- `path_whitelisted = true`
- client `allowed_paths` and `allowed_path_patterns` are discarded
- model, data, and output paths are validated against independent server roots
- network data sources require an explicitly allowlisted hostname
- client-provided artifact entries are cleared
- `job_id` accepts only safe alphanumeric, underscore, and hyphen identifiers

Path validation rejects traversal, sibling-prefix escapes, existing symlink
components, and broken symlinks. A missing ordinary output leaf below the trusted
output root is allowed so the handler can create it.

The API has no authentication, RBAC, tenant isolation, or per-job filesystem
sandbox. It is intended for a trusted host/network and controlled service account.

## 6. Frontends and startup

Use the combined launcher:

```bash
python start_studio.py
```

It health-checks or starts `main_engine.py` as a child process when needed, then
runs Gradio in the launcher process. Gradio and the API remain separate services.
The default endpoints are:

- Gradio: `http://127.0.0.1:7860`
- FastAPI: `http://127.0.0.1:8000`
- Health: `http://127.0.0.1:8000/health`
- OpenAPI: `http://127.0.0.1:8000/docs`
- Zero-build console: `http://127.0.0.1:8000/`

For API-only operation:

```bash
python main_engine.py
```

`frontend/` is always mounted at `/`. The React/TypeScript/Vite source is under
`web/`; `/console` is mounted only when a local `web/dist` build exists. Gradio is
not mounted by FastAPI and communicates through REST.

## 7. Configuration

- `F1_JOBS_STATE_PATH`: local manager state file; default `runs/jobs_state.json`
- `F1_ENGINE_HOST` / `F1_ENGINE_PORT`: API bind address; defaults `127.0.0.1:8000`
- `F1_STUDIO_API_URL`: API origin used by Gradio; default `http://127.0.0.1:8000`
- `F1_MODEL_ROOTS`: trusted model roots, separated by `os.pathsep`
- `F1_DATA_ROOTS`: trusted data roots, separated by `os.pathsep`
- `F1_OUTPUT_ROOT`: one trusted output root; default `<cwd>/runs`
- `F1_NETWORK_INPUT_HOSTS`: comma-separated network input host allowlist
- `F1_MAX_PENDING_JOBS`: pending capacity; default `100`
- `F1_CPU_CONCURRENCY`: CPU slots; default `2`
- `F1_GPU_CONCURRENCY`: accelerator slots; default `1`
- `F1_STOP_GRACE_SECONDS`: worker stop grace, 0-30 seconds; default `2`
- `F1_CORS_ORIGINS`: comma-separated CORS allowlist replacing dev defaults

## 8. Verification record

The following results were recorded on commit
`0e7a8f83b53c97abc8eb3caedc532f5779fbb086`, reviewed against the official
baseline `acce839c7e895d6b179de7f7093fa879e237cc7b` (2026-08-21 23:59:59 UTC+8).

Windows verification used the repository `.venv` with Python 3.12 and a fresh
repository-local pytest temp directory:

```text
pytest tests/f1/ tests/api/ --cov=f1 --cov=core
412 passed, 11 skipped, 3 warnings
TOTAL: 2369 statements, 518 missed, 78% coverage
```

The 11 Windows skips require symlink privileges. Two warnings are ONNX exporter
deprecations; the third was a host-specific pytest cache permission warning.

The Ubuntu/Python 3.10 source-branch CI run at that commit completed successfully:

```text
422 passed, 1 skipped, 3 warnings
TOTAL: 2369 statements, 516 missed, 78% coverage
F1 smoke: 5/5 passed
Ruff lint and format: passed
```

CI evidence:
<https://github.com/9-71/YOLO-Master/actions/runs/34793607227>

This is source-branch push CI evidence, not a Tencent repository required PR
check. The F1 workflow covers Python Ruff, smoke, unit tests, and coverage. It
does not currently run Node lint, TypeScript checking, or a Vite production build.

Local checks at the same commit additionally passed `npm run lint` and TypeScript
`--noEmit` checks for the React app and Node configuration. No claim is made here
that a production React bundle is exercised by CI.

## 9. Agent/F1 boundary

The F1 production execution path is `f1/jobs_manager.py` ->
`f1/worker_runtime.py` -> `f1/dispatcher.py` -> `f1/handlers/`. It does not route
jobs through `agent/runtime/cli/async_jobs.py` or an Agent worker.

`agent/` remains a separate skill/CLI integration layer. Agent documentation and
validation belong to `agent/SKILL.md` and
`agent/scripts/validate_yolo_master_skill.py`; those checks do not replace F1 API,
lifecycle, handler, or security tests.

## 10. Known limitations

- One API service process and one local lifecycle owner only
- Local best-effort JSON state; no database-backed durable queue
- No distributed scheduling, priority execution, preemption, or automatic retry
- Static CPU/GPU slot counts; no memory-aware or per-GPU placement
- HTTP cursor polling only; no WebSocket/SSE stream
- No authentication, RBAC, tenant isolation, or per-job OS sandbox
- React build artifacts are deployment-provided and are not covered by F1 CI
- Windows/Linux process containment is verified; other POSIX behavior is weaker
- Some OBB/classification metric displays remain incomplete
- Export tests currently emit legacy ONNX exporter deprecation warnings
- Dependency versions are not fully upper-bounded or locked

## References

- User manual: `docs/f1_user_manual.md`
- Handler framework: `f1/handlers/README.md`
- Handler extension guide: `f1/handlers/USAGE.md`
- Runtime lifecycle: `f1/RUNTIME_LIFECYCLE.md`
- Agent skill boundary: `agent/SKILL.md`
- Ultralytics documentation: <https://docs.ultralytics.com/>

---

**Document version**: 1.5.0

**Last synchronized**: 2026-09-15
