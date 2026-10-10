# Studio FastAPI Service (PR3)

The Service exposes the existing JobsManager over REST. It imports no frontend,
Gradio, Launcher or Agent code. JobsManager remains the only public lifecycle
owner; handlers and the Service never publish their own job terminal state.

Use Python 3.10+ and install the optional Service dependencies:

```bash
pip install -e ".[studio]"
python main_engine.py
```

This entry point binds to `127.0.0.1:8000`, uses one server process, disables
proxy-header trust, and reports handled lifespan startup/shutdown failure with a
nonzero exit. Invalid environment configuration exits 1 with the fixed message
`Invalid Studio Service configuration`, without echoing input or a traceback.
Module import rejects invalid settings with the same fixed ValueError and does
not construct a manager; HTTP dependency resolution also creates no manager.
FastAPI lifespan creates the manager once, reconciles persisted history before
serving requests, and waits for Runtime shutdown before dropping the reference.
On cleanup failure the reference is retained and the lifespan fails explicitly.
An unrecoverable live-worker cleanup failure does not guarantee bounded process
exit: Runtime atexit retries and multiprocessing joins may continue waiting.

Running raw `uvicorn main_engine:app` embeds the same lifespan, but its CLI exit
code and signal replay behavior are not the supported failure-propagation entry
point. Multiple ASGI workers, reload, public/LAN deployment and token auth are
outside this PR's supported configuration.

## REST contract

- `POST /api/v1/jobs[/]`: `JobRequest` submission, 201 after durable admission;
  duplicate ID 409, pending capacity 429, invalid request/path 422,
  closing owner or admission persistence failure 503.
- `GET /api/v1/jobs[/]?limit=10&offset=0`: existing recent-jobs window and total
  captured together under the owner lock. No new pagination capability is added.
- `GET /api/v1/jobs/{job_id}`: one detached lifecycle snapshot, metadata,
  execution duration, structured error and final artifact count; unknown ID 404.
- `POST /api/v1/jobs/{job_id}/cancel`: typed owner decision; accepted request 202
  with `status=cancel_requested`, terminal replay 200 with the existing state,
  not cancellable 409, unknown ID 404. A running job stays active during cleanup.
- `GET /api/v1/jobs/{job_id}/logs?offset=0&limit=500`: sanitized flattened-line
  windows. `next_offset=null` means the current tail; later worker logs may extend
  it. Terminal consumers must drain remaining pages.
- `GET /api/v1/jobs/{job_id}/artifacts`: exact final manifest entries, relative
  nested IDs, image IDs and `/static/artifacts/...` URLs, without server paths.
- `GET /static/artifacts/{job_id}/{artifact_id:path}`: exact membership plus
  fresh containment validation, then opened-file identity revalidation before
  streaming. Unlisted, deleted, escaped, broken-symlink and path-like IDs return
  404. No directory scanner grants download authorization.
- `GET /health`: readiness, persistence warning and configured shutdown budgets.
  The legacy `service="YOLO-Master F1 Task Engine"` identifier is preserved.
  Admission disk failure rolls back acceptance; later persistence failure retains
  actual in-memory execution facts and is observable through the health snapshot.

HTTP errors have structured codes. Validation errors omit raw `input` and `ctx`;
response errors and unexpected server tracebacks are sanitized. The supported
Service entry point disables Uvicorn's default access log because it echoes
untrusted query strings without credential sanitization. API/download
responses use `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`.

## Local security boundary

Only a trusted local machine is supported. The entry point permits bind hosts
`127.0.0.1`, `localhost` and `::1`. Request peers must be loopback, Host must be a
literal supported local host, and any Origin must be explicitly allowlisted.
Remote, wildcard and opaque `null` origins are rejected before job admission.
Forwarded headers cannot turn a remote peer into a trusted local request.

`STUDIO_CORS_ORIGINS` is a comma-separated list of explicit local HTTP(S) origins;
the default is `http://localhost:8000,http://127.0.0.1:8000`. Credentials are not
enabled in CORS. This boundary is not authentication or a multi-user isolation
promise. Server-owned model/data/output roots and network-input host policy remain
Runtime admission rules; a request cannot widen them or enable shell execution.

## Shutdown budgets and signals

`STUDIO_ENGINE_HOST` and `STUDIO_ENGINE_PORT` configure the supported bind.
`STUDIO_JOBS_STATE_PATH` defaults to `runs/jobs_state.json`. Existing Runtime
`STUDIO_*` root/concurrency/shutdown configuration remains authoritative.
`STUDIO_HTTP_DRAIN_SECONDS` defaults to 5 (range 0..30).

Uvicorn stops accepting connections and drains HTTP tasks first. Its HTTP drain
budget is separate from ASGI lifespan: it does not bound manager checkpoint
shutdown. Lifespan then waits for `JobsManager.shutdown()` without abandoning the
cleanup thread. Runtime's stated caller budget is
`shutdown_grace_seconds + stop_grace_seconds + 16`, excluding persistent cleanup
failure retries. Cooperation uses Runtime's original absolute deadline; result,
checkpoint or EOF alone does not authorize terminal publication or slot release.

The supported server captures SIGINT/SIGTERM and Windows CTRL_BREAK, restores
prior handlers after ASGI shutdown, and checks lifespan failure before exiting.
Windows managed workers detach their console before executing task code so a
console broadcast cannot abort native training libraries before Runtime stop IPC;
Windows Job Object containment and IPC handles remain in effect. The detached
Worker routes Python stdout/stderr and logging handlers already bound to those
streams through the existing sanitized job-log IPC channel, preserving Unicode
and real exception tracebacks. Explicit file logging sinks remain unchanged.
Complete text chunks retain known multiline credential context before sanitization;
stdout and stderr keep separate buffers with a shared log-sequence lock, so a
newline in one stream cannot truncate the other stream's credential context. A
partial-line flush keeps its buffer until publication is safe. The final tail
is sanitized before the result or exception traceback, including a known
cross-line credential prefix interrupted by an exception.
Native writes to OS file descriptors are outside this Python text-stream path;
forced termination can lose an incomplete buffered line.

Credential sanitization does not hide every local absolute path in diagnostics.
Such a path does not authorize a download: exact manifest membership and fresh
containment checks still apply. POSIX guardian, process-group and subreaper
behavior is unchanged. Force-kill, repeated emergency signals and host termination
cannot promise a cooperative checkpoint.

PR4 must give the production Launcher its own measured budget for HTTP drain,
Runtime cooperation/cleanup and service exit. This PR does not create a Launcher.

## Verification

```bash
python -m pytest tests/studio/test_service_api.py tests/studio/test_service_process.py -q
# Windows console-backed Worker output probes (hidden console, real CTRL_BREAK)
python -m pytest tests/studio/test_worker_console.py -q
STUDIO_TEST_MODEL=/absolute/local/yolov8n.pt python -m pytest \
  tests/studio/test_service_process.py::test_real_model_http_signal_checkpoint_and_resume \
  tests/studio/test_worker_console.py \
  --slow --studio-integration -q
```

REST unit/IPC tests use a named deterministic executor with real Runtime workers.
Process tests use real loopback HTTP and external OS service signals, capture
PID/create-time identities including zombies, and check identity absence after
Service exit plus manager enter/return facts. They do not measure the exact kernel
reap instant or first public terminal visibility. Cleanup-before-terminal is
supported separately by owner code, Runtime regressions and the API cancellation
stop barrier.

The cleanup-failure harness exercises the actual manager wait budget and retained
owner/active state, then restores cleanup in its finalizer before exit 1. This
proves failure propagation and restored-cleanup exit, not bounded production exit
with an unrecoverable live worker. The separately opted-in model test requests
workers=1 with a local model, tiny generated dataset, CPU and amp=False. The saved
trainer log reports `Using 0 dataloader workers`, so effective workers=0; it proves
checkpoint epoch 0 load, next-epoch resume and Service restart from history, not
real dataloader children, GPU, AMP-enabled, DDP or PR4 Launcher acceptance.
