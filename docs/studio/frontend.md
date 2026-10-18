# Studio clients and launcher

Studio keeps three HTTP clients with different entry points. React/Vite in `web/`
provides the full task console; `frontend/` is the zero-build fallback served at
`/`; Gradio's Jobs tab lives beside the existing synchronous Inference Studio.
All three use the same `/api/v1/jobs` Service and Studio Runtime. The synchronous
Inference Studio retains its existing direct inference path. A UI never creates a
JobsManager or determines task completion from a submission or cancellation ack.

Run `python main_engine.py` to serve the API and zero-build console at
`http://127.0.0.1:8000/`. React development uses `npm ci` and `npm run dev` in `web/`.
Run `npm run lint` and `npm run build` before delivering React changes; the build
includes TypeScript checks. A locally built `web/dist/` is served at `/console/`
when present. Keep the lockfile; do not commit `node_modules/` or `dist/`.

Run `python start_studio.py` for the Service plus Gradio. It reuses a compatible
local Service if both its existing `/health` identity and jobs endpoint respond.
Otherwise it starts the existing `main_engine.py` entry point and owns that child
until Gradio closes or the launcher receives Ctrl+C, SIGTERM, or Windows
CTRL_BREAK. Gradio's printed local URL is the browser entry point; the launcher
keeps browser opening disabled. Gradio is closed on normal exit, launch failure,
interruption, and unexpected death of the owned API. An API reused from another
process is never stopped by this launcher.

## Trusted local configuration

`STUDIO_API_URL` selects the local HTTP API origin. `F1_STUDIO_API_URL` remains a
compatibility alias; contradictory dual values fail before startup. The default
is `http://127.0.0.1:8000`. Only `http` with `127.0.0.1`, `localhost`, or `::1` is
accepted by the combined launcher; credentials, query strings, fragments, and
non-root paths are rejected. The launcher health probe ignores ambient proxy
settings and follows no redirects. During combined startup the launcher temporarily
adds loopback hosts to both `NO_PROXY` and `no_proxy`, preserving existing exclusions
and proxy settings so Gradio can finish its own local startup probes. Prior values
are restored when the launcher returns. `Studio launcher ready` is printed only
after Gradio launch completes, rather than when an early config endpoint appears.

If `STUDIO_ENGINE_HOST` / `STUDIO_ENGINE_PORT` or their `F1_` aliases are explicitly
set, they must agree with each other and the API URL. The child receives both
matching bind aliases. Runtime roots, the durable state path, concurrency, CORS,
and admission remain configured by the existing trusted Service environment;
client form fields do not widen those permissions. Use explicit local model and
data paths within the Service roots. The launcher provides no public hosting,
authentication, or multi-user isolation.

The outer shutdown wait snapshots the Service and Runtime configuration at child
creation: HTTP drain (default 5 seconds) + training cooperation (60) + stop grace
(2) + the Runtime's current bounded cleanup allowance (16) + service exit (5).
`STUDIO_HTTP_DRAIN_SECONDS`, `STUDIO_SHUTDOWN_GRACE_SECONDS`, and
`STUDIO_STOP_GRACE_SECONDS` accept the existing `F1_` aliases, reject conflicts,
and require finite values within the Service/Runtime ranges (0–30, 0–3600, and
0–30 seconds respectively). Later environment changes do not shorten the captured
wait for an already owned child.

Windows children are created in a new process group and receive CTRL_BREAK for
cooperative Service shutdown; POSIX children receive SIGTERM in their own session.
The existing Service lifespan owns Runtime shutdown, including checkpoint
cooperation and process-tree cleanup. The launcher does not replace that lifecycle.
A nonzero service exit or forced kill is an error and produces a nonzero launcher
exit. Forced kill after the complete budget is a failure fallback, not checkpoint
acceptance. Closing a browser tab alone does not stop Gradio or the service;
close the launcher with Ctrl+C when finished.

## Client behavior and acceptance

Read public task states from the API. HTTP 201 accepts a submission; HTTP 202
accepts a running cancellation request and does not prove it has finished.
Terminal cancellation, errors, and output readiness come from later status polls.
Live controls stop log polling only. Clients drain cursor pages through the tail
and discard obsolete responses after selecting another job or API address.
Downloads come from the exact manifest and preserve nested artifact IDs. Gradio
may cache manifest-authorized HTTP downloads for local presentation; it never
scans output directories to invent downloads.

Run launcher unit and real production helper checks with:

```bash
python -m pytest tests/studio/test_launcher.py tests/studio/test_launcher_process.py -q
```

The explicit production model integration (including the combined launcher entry)
uses an existing local model, CPU, and a generated tiny dataset:

```bash
STUDIO_TEST_MODEL=/absolute/local/yolov8n.pt python -m pytest \
  tests/studio/test_mainline_integration.py --slow --studio-integration -q
```

These automated checks do not replace the human PR4 Gate. Before publication,
exercise React, zero-build, and Gradio with actual submit → running → logs →
artifacts → cancel flows. Check a large terminal log tail, rapid job/API switching,
unavailable backend, nested artifact downloads, original synchronous inference,
and combined launcher shutdown/restart. Review actual checkpoint load/resume,
terminal state, and absence of owned descendants. Windows CPU evidence does not
establish Linux/macOS, GPU/AMP/dataloader, or DDP support; retain their separate
platform and training Gates until verified.
