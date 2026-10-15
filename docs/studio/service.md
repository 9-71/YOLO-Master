# Existing FastAPI service with Studio Runtime (PR3 integration)

PR3 connects the mainline `main_engine.py` and `api/v1/jobs.py` to
`studio.JobsManager`. It reuses the existing REST service and clients.
FastAPI lifespan creates one owner, reconciles history before admission, and
waits for Runtime cleanup before releasing ownership. Concurrent lifespan
startup is rejected. Requests never construct an owner. The old `f1/` package,
its HTTP client, and the existing frontend and Launcher remain available.

Install the optional service dependencies with Python 3.10+:

```bash
pip install -e ".[studio]"
python main_engine.py
```

The supported script entry binds to `127.0.0.1:8000`, runs one process, disables
proxy-header trust and access-query logging, and returns a nonzero exit on
handled startup or shutdown failure. Invalid service configuration produces a
fixed error without echoing the value. Raw `uvicorn main_engine:app` uses the
same application lifespan, but does not provide this script's failure-exit and
signal-replay guarantees. Multiple server processes, reload, public/LAN service,
authentication and multi-user isolation remain outside the supported scope.

## Compatibility

The existing routes, request and response fields, OpenAPI operation IDs,
collection slash spellings and execution-duration semantics are preserved:

- `POST /api/v1/jobs[/]`: 201 admission, 409 duplicate, 429 capacity, 422 invalid
  payload or path. Duplicate and invalid-path `detail` remain strings; queue
  errors retain `detail.code` and `detail.message`.
- `GET /api/v1/jobs[/]?limit=10&offset=0`: the existing page and total, now taken
  together under the owner lock.
- `GET /api/v1/jobs/{job_id}`: coherent status, error, timestamps, metadata,
  artifact count and numeric duration. Queue time is excluded; missing start
  stays null and zero duration stays `0.0`.
- `POST /api/v1/jobs/{job_id}/cancel`: 202 acknowledgement, 200 terminal replay,
  409 non-cancellable and 404 unknown. Runtime's typed decision determines the
  response; running cancellation remains active until actual cleanup.
- `GET /api/v1/jobs/{job_id}/logs`: sanitized flattened-line offset/limit pages.
  A null cursor denotes the current tail; consumers must drain terminal pages.
- `GET /api/v1/jobs/{job_id}/artifacts` and
  `GET /static/artifacts/{job_id}/{artifact_id:path}`: existing manifest fields
  and nested download IDs. Unknown-job and missing-artifact details remain
  strings. Downloads recheck containment and opened-file identity before streaming.
- `GET /health`: exactly `status=ok` and
  `service="YOLO-Master F1 Task Engine"`, preserving the existing liveness response.

New 503 structured errors represent unavailable/closing service and admission
that could not be persisted. Such an admission is rolled back and starts no
worker. Validation error lists omit raw input/context to avoid credential echo;
unexpected failures, log lines and persistence snapshots are sanitized.
Cancellation messages are presentation text; clients must use status/code.

Existing `create_app()` and `app.dependency_overrides[get_jobs_manager]` remain
usable. An override must return the same Studio-compatible manager throughout
the lifespan. `manager_factory` is an additional trusted test/embedding seam.
Calling `get_jobs_manager` to lazily construct a manager outside ASGI lifespan
is no longer supported. HTTP clients need no Runtime imports or state-file access.

## Trusted configuration and existing UI entry points

`F1_*` remains an alias for matching `STUDIO_*` settings: `ENGINE_HOST`,
`ENGINE_PORT`, `CORS_ORIGINS`, `JOBS_STATE_PATH`, `MODEL_ROOTS`, `DATA_ROOTS`,
`OUTPUT_ROOT`, `NETWORK_INPUT_HOSTS`, `CPU_CONCURRENCY`, `GPU_CONCURRENCY`,
`MAX_PENDING_JOBS`, `STOP_GRACE_SECONDS`, `SHUTDOWN_GRACE_SECONDS` and
`HTTP_DRAIN_SECONDS`. If both names contain different trimmed values, startup
fails with a fixed error rather than silently changing trusted roots or budgets.
Empty legacy CORS/root/state-path values retain their mainline default behavior;
explicit empty `STUDIO_CORS_ORIGINS` disables the allowlist. Root lists use the
platform path separator. State defaults to `runs/jobs_state.json`.
The existing client/Launcher `F1_STUDIO_API_URL` setting is unchanged.

The mainline CORS defaults remain local origins on ports 8000, 5173, 3000 and
8080 for localhost/127.0.0.1, plus `null` for the existing trusted file console.
Retaining `null` is an explicit compatibility choice; it cannot identify the
source of an opaque origin and is not an authentication boundary. Request peers
must be loopback and Host must be a literal local host. Unsupported remote bind
hosts, remote configured origins, wildcard origins and credential-bearing
origins are rejected. Disallowed preflight origins retain the existing CORS 400;
disallowed actual requests receive 403. Proxy headers cannot grant local trust.

When present, the existing `frontend/` is mounted at `/`, its favicon is served,
and a built `web/dist` is mounted at `/console`. API routes take precedence.
Range/If-Range, full/stat headers and multipart ranges use FileResponse semantics
through the already-open authorized handle (Starlette 0.49.3+). These assets are optional: a pure installed service starts without UI resources
or Gradio. A wheel includes main_engine/api/studio and the existing F1
compatibility packages; source UI assets are not bundled. No frontend or
Launcher behavior is redesigned in this change.

## History and lifecycle

Version-1 F1 snapshots use the existing shared core schema. Terminal records,
timestamps, errors and stored manifests are retained. Active history is
reconciled to `failed/SERVICE_RESTARTED`, never re-executed or used to signal an
old PID. Safe absolute historical manifest entries obtain contained relative
download IDs at read time without rewriting the stored entry. Unlisted files,
outside paths, traversal and symlink escapes cannot grant download authorization.

Studio validates primary and backup snapshots, atomically replaces state and
fails startup if neither existing copy is valid. Back up and validate real
history before operational migration; two services must never share the state
store. After admission, a later persistence failure preserves execution truth
in memory and sets the owner's `persistence_error`. The unchanged public health
response does not expose this internal warning; an additive client-facing
durability indicator needs separate API agreement. Persistent write failure
cannot promise that restart will see the latest in-memory terminal state.

Shutdown drains HTTP requests first (default 5 seconds), then awaits Runtime's
checkpoint cooperation and cleanup. Runtime's caller budget is
`shutdown_grace_seconds + stop_grace_seconds + 16` (default 78 seconds).
Checkpoint/ACK/result alone cannot publish terminal or release a slot.
Cleanup drains IPC without a disk snapshot for each log; the final ordered tail
is persisted at checkpoint confirmation or terminal publication. This prevents
console log bursts from postponing cancellation or process cleanup. Active jobs stopped by successful
service shutdown become `failed/SERVICE_SHUTDOWN`, distinct from cancellation and
completion. Cleanup failure retains owner/slot and fails lifespan. Permanent
cleanup failure may continue through atexit retries or multiprocessing joins;
there is no unconditional bounded-exit promise.

Windows workers detach from the service console before task execution and route
Python stdout/stderr and prebound console logging through sanitized IPC. Job
Object containment remains active. Native descriptor writes and interrupted
partial lines are outside this text-stream guarantee. POSIX guardian/subreaper
behavior is unchanged. Production `start_studio.py` still uses terminate/5s/kill;
it cannot provide Studio checkpoint shutdown. Launcher repair and acceptance
remain PR4 Gates and this change does not recommend that shutdown path.

## Verification

```bash
python -m pytest tests/studio tests/api/test_api_v1.py -q
# Explicit local-model, production entry and existing HTTP-client acceptance
STUDIO_TEST_MODEL=/absolute/local/yolov8n.pt python -m pytest \
  tests/studio/test_mainline_integration.py --slow --studio-integration -q
```

The production integration uses real loopback HTTP, the default production
factory, five real task handlers, existing HTTP client, real running cancellation,
external service signal, checkpoint load/resume and restart. It uses CPU and a
tiny generated dataset; effective loader workers are zero and CPU AMP is disabled
even if the configured AMP option is true. It cannot close GPU/loader/AMP-scaler,
DDP, Linux/macOS, Launcher or browser-human Gates. Process harness tests additionally
cover cooperation, fallback and retained ownership after injected cleanup failure.
Their finalizer restores cleanup; they do not prove permanent-failure exit bounds.
Identity absence after cleanup/exit does not measure the exact kernel reap instant.
