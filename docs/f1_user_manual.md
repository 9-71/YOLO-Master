# F1 Task Platform — User Manual

> **Audience**: end-users and evaluation reviewers.
> **Scope**: submitting and monitoring detection jobs through the Gradio Web UI and the standalone REST engine (FastAPI) — not framework development.

**Version**: 1.2 · **Last synchronized**: 2026-09-15

---

## 1. Introduction

The **F1 Task Platform** is the unified, browser-based job console of YOLO-Master. It wraps the
training / validation / inference / export engines behind a single submission form and an incremental
monitoring dashboard. You never touch training or inference internals — the platform acts as an
orchestration layer that accepts a job, enforces security rules, forwards the work to the underlying
Ultralytics engine, and collects the results back for you.

The platform provides four core capabilities:

| Capability | What it does |
|---|---|
| **Unified submission** | One form dispatches every task type — `predict`, `train`, `val`, `export`, and `diagnose`. |
| **Task dispatcher** | The manager routes each admitted job through a worker, dispatcher, and registered handler. |
| **Incremental logging** | The consoles poll sanitized structured logs with an HTTP cursor while the job runs. |
| **Artifact management** | Generated files (weights, metrics, plots, exported models) are listed, previewed, and exposed for download. |

### 1.1 Supported task types

| Task type | Purpose |
|---|---|
| `predict` | Run object detection on images or videos (single file, directory, or batch). |
| `train` | Fine-tune a model on a dataset defined by a `data.yaml` file. |
| `val` | Validate a checkpoint and compute metrics (mAP50, mAP50‑95, precision, recall). |
| `export` | Convert a checkpoint to a deployment format (ONNX, TorchScript, etc.). |
| `diagnose` | Collect a system environment report (Python, PyTorch, CUDA, GPU). |

### 1.2 Job lifecycle at a glance

Every job follows the public lifecycle below. `JobsManager` owns admission, queueing,
cancellation, timeout, and terminal publication. Inside the worker, the dispatcher enforces
`PENDING → RUNNING → COMPLETED/FAILED`; `CANCELLED` is published by the manager after it has
stopped the owned process tree. Terminal status/result/error snapshots cannot be restarted or reset.

```
PENDING ──► RUNNING ──► COMPLETED   (success)
   │            │
   ├────────────┴──► FAILED         (failure or timeout)
   └───────────────► CANCELLED      (user cancellation)
```

---

## 2. Quick Start & UI Launch

### 2.1 Prerequisites

The platform runs in the same environment as YOLO-Master. The recorded verification used
Windows/Python 3.12 and by the Ubuntu/Python 3.10 F1 CI job; these are verification environments,
not an additional package compatibility guarantee beyond the project metadata.

| Component | Requirement |
|---|---|
| Python | 3.10 – 3.12 (verified on 3.12.10) |
| PyTorch | 2.5.1+ (CUDA 12.1 build recommended) |
| Ultralytics | 8.4.x |
| CUDA (optional) | 12.1 with a compatible driver — used for GPU acceleration |
| Python packages | `gradio`, `fastapi`, `uvicorn[standard]`, `opencv-python`, `numpy`, `pandas`, `pydantic` |

> **Tip** — Verify your environment before launching:
>
> ```bash
> python --version
> python -c "import torch; print(torch.cuda.is_available())"   # True = GPU available
> python -c "from ultralytics import YOLO; print(YOLO.__version__)"
> ```

Install the project in editable mode if you have not already:

```bash
pip install -e .
```

### 2.2 Launching the Web UI

From the **project root directory**, start the server:

```bash
python start_studio.py
```

On startup the app:

1. Starts and health-checks the FastAPI Studio Job API.
2. Creates a `ckpts/` checkpoint directory if it does not exist.
3. Scans `ckpts/` (recursively) for `.pt` weights and categorizes them by task.
4. Serves the Gradio interface and opens it in your default browser.

For manual startup, run `python main_engine.py` first, then run `python app.py` in another terminal. The latter starts
Gradio only and connects to `F1_STUDIO_API_URL` (default: `http://127.0.0.1:8000`).

> **Note** — `main_engine.py` is the **standalone FastAPI engine**, a production entry point
> that runs independently of Gradio. See §7 for its full HTTP API, interactive docs, and
> environment configuration.

> **Tip** — The interface is served on the default Gradio address
> `http://127.0.0.1:7860` and opens automatically (`inbrowser=True`). If the
> browser does not open, navigate to that URL manually.

### 2.3 The interface layout

The UI is organized into two top-level tabs:

- **🖼️ Inference Studio** — the interactive, synchronous inference playground.
- **📋 Jobs** — the asynchronous task console described in this manual.

A top-level **Language** selector (English / 中文) switches the entire interface in one place.

```text
+-------------------------------------------------------------------------------+
|  YOLO-Master Web Console                      [ 🌐 Language: English / 中文 ] |
+-------------------------------------------------------------------------------+
|  [ 🖼️ Inference Studio ]  |  [ 📋 Jobs (Active) ]                            |
+-----------------------------------+-------------------------------------------+
|  LEFT PANEL: Job Submission Form  |  RIGHT PANEL: Lifecycle & Diagnostics     |
|                                   |  [ 📊 Status ] [ 📜 Logs ] [ 📁 Artifacts ]|
|  - Task Type: [predict/train/...] |  ---------------------------------------- |
|  - Model Path: ckpts/yolov8n.pt   |  Job ID: 20260908-120000-xxxx            |
|  - Data Source: assets/bus.jpg    |  Status: RUNNING                          |
|  - Output Dir: runs/predict       |  ---------------------------------------- |
|  - [⚙️ Hyperparameters]           |  Console Stream:                          |
|  - [🔒 Security Constraints]      |  [INFO] Model loaded successfully...      |
|                                   |  [INFO] Processing batch 1/1...           |
|  [ 🔥 Submit Job ]                |  [ 🚫 Cancel Job ]                        |
+-----------------------------------+-------------------------------------------+
```

---

## 3. Job Submission Workflow

All asynchronous work is submitted through the **Jobs** tab. The submission panel is on the left;
the monitoring panels are on the right.

### 3.1 Step-by-step submission

1. Open the **📋 Jobs** tab.
2. Select a **Task Type** (`predict`, `train`, `val`, `export`, or `diagnose`).
   Selecting a task auto-fills the form with sensible defaults for that task.
3. Fill in **Model Path**, **Data Source**, and **Output Directory** (see the parameter tables below).
4. (Optional) Open **⚙️ Hyperparameters** and adjust **Confidence Threshold** and **Device**.
5. (Optional) Open **🔒 Security Constraints** to inspect the compatibility path list. The
   FastAPI service does not trust or extend access from this client-controlled value.
6. Click **🔥 Submit Job**. The generated **Job ID** appears in the *Status Monitor*.

### 3.2 Form field reference

| Field | Meaning | Default |
|---|---|---|
| **Task Type** | Which handler will run the job. | `predict` |
| **Model Path** | Path to model weights (`.pt`) or, for `train`, an architecture YAML. | `./ckpts/yolov8n.pt` |
| **Data Source** | Input data: an image/video path, a directory of media, or a dataset YAML for `train`/`val`. | `ultralytics/assets/bus.jpg` |
| **Output Directory** | Base directory where per-job results are written. | `runs/predict` |
| **Confidence Threshold** | Detection confidence cutoff, in `(0.0, 1.0]`. | `0.25` |
| **Device** | Compute device: `0` (GPU), `cpu`, or `mps`. | `0` |
| **Allowed Paths** | Compatibility field serialized by the client. The server discards it and applies its own trusted roots. | `., ultralytics/assets, runs, ckpts` |

> **Security boundary** — editing **Allowed Paths** cannot grant filesystem access. The server
> independently validates model, data, and output locations against `F1_MODEL_ROOTS`,
> `F1_DATA_ROOTS`, and `F1_OUTPUT_ROOT`.

The **🔄 Reset** buttons next to *Model Path* and *Data Source* restore the active task’s default
value without changing the selected task type.

### 3.3 Per-task presets and requirements

The form repopulates itself when you change the task type:

| Task type | Model Path preset | Data Source preset | Output Dir preset | Additional notes |
|---|---|---|---|---|
| `predict` | `./ckpts/yolov8n.pt` | `ultralytics/assets/bus.jpg` | `runs/predict` | Accepts image/video/directory/list. |
| `train` | `./ckpts/yolov8n.pt` | `coco8.yaml` | `runs/train` | `data_source` **must** be a `.yaml`/`.yml`. Defaults: 1 epoch, `imgsz=640`. |
| `val` | `./ckpts/yolov8n.pt` | `coco8.yaml` | `runs/val` | `data_source` **must** be a `.yaml`/`.yml`. Default `imgsz=640`. |
| `export` | `./ckpts/yolov8n.pt` | *(empty)* | `runs/export` | Exports to ONNX by default. |
| `diagnose` | *(empty)* | *(empty)* | `runs/diagnose` | No inputs required; reads system state only. |

> **Warning** — `train` and `val` reject any non-YAML data source at validation time. Submitting
> an image path (e.g. `bus.jpg`) for these tasks fails immediately with
> `PARAM_VALIDATION_FAILED` rather than hanging mid-run.

### 3.4 Data Source formats for `predict`

The `predict` handler normalizes the *Data Source* into a deterministic file list:

- A **single file** path → one input.
- A **directory** → all supported media files inside it, sorted (recursive-free).
- A **list** of paths → expanded entry-by-entry, order preserved.
- An explicitly authorized `http`, `https`, `rtmp`, `rtsp`, or `tcp` URL → one
  network input. The hostname must appear in the server's `F1_NETWORK_INPUT_HOSTS`;
  network inputs are rejected by default.

Supported media extensions: `.jpg .jpeg .png .bmp .webp .tif .tiff .mp4 .avi .mov .mkv .ts`.
Inputs are split into batches (`batch_size` defaults to 8), and each batch is written to its own
`batch_NNN` sub-folder under the job directory.

---

## 4. Job Monitoring & Lifecycle Management

### 4.1 Navigating the Jobs tab

The right-hand panel has four monitoring sub-tabs:

| Sub-tab | Purpose |
|---|---|
| **📊 Status Monitor** | Current job ID, structured status, error diagnostics, and the **Cancel** button. |
| **📜 Live Logs** | Streaming console output for the selected job. |
| **📁 Artifacts** | Generated-file list with image preview and download access. |
| **🕒 Recent Jobs** | Newest-first history of every submitted job. |

### 4.2 Job states

The *Status Monitor* shows a JSON snapshot containing the canonical job state:

Public job statuses: `pending | running | completed | failed | cancelled`.

| State | Meaning | Terminal? |
|---|---|---|
| `PENDING` | Job accepted but not yet started. | No |
| `RUNNING` | Handler is executing on the engine. | No |
| `COMPLETED` | Execution finished successfully; artifacts captured. | Yes |
| `FAILED` | Execution failed — see the error code for the cause. | Yes |
| `CANCELLED` | Execution stopped after a user cancellation request; see `USER_CANCELLED`. | Yes |
| `NOT_FOUND` | The selected Job ID does not exist. | Yes |

While a job is `PENDING` or `RUNNING`, the console refreshes **every second** and stops
automatically once the job reaches a terminal state. A slower **30-second** background sync keeps
the *Recent Jobs* table and final artifacts fresh at all other times.

### 4.3 Job cancellation

1. Select the job (its Job ID must be shown in the *Status Monitor*).
2. Click **🚫 Cancel Job**.

Cancellation is manager-owned. For a waiting job, `JobsManager` can publish cancellation without
starting a process. For a running job, it records the request, asks the managed worker to stop,
escalates through the platform process-tree cleanup path when needed, joins and closes the worker,
and only then publishes `CANCELLED / USER_CANCELLED`. Dispatcher/handler checkpoints provide an
additional cooperative path between engine calls, but do not replace manager-owned termination.

> **Warning** — constraints on cancellation:
> - You can only cancel a job that is `PENDING` or `RUNNING`. A job already in `COMPLETED` or
>   `FAILED` or `CANCELLED` is terminal and cannot be cancelled.
> - Cancellation marks the job as `CANCELLED` with error code `USER_CANCELLED`; it is *not* resumable.
>   Re-submit the job to run it again.

### 4.4 Timeouts

Every job carries a **default execution limit of 300 seconds** (`timeout_seconds = 300`).

- The deadline starts when a queue slot launches the managed worker. Time spent waiting in
  `PENDING` does not consume the timeout.
- `JobsManager` supervises the managed process against this deadline, stops and joins the owned
  process tree, and then publishes `FAILED / TIMEOUT`.
- A timeout never produces a partial `COMPLETED` result.

> **Tip** — Long-running `train` jobs with many epochs can exceed the 300-second limit. Keep early
> experiments short (the default is 1 epoch), and watch the elapsed-time column in the *Status
> Monitor*.

---

## 5. Live Diagnostics & Log Inspection

### 5.1 Viewing incremental output

Open the **📜 Live Logs** sub-tab while a job is active. The console polls the HTTP log endpoint
and displays the available structured entries, including:

- Submission and state-machine transitions (`transitioned to: RUNNING/COMPLETED/FAILED/CANCELLED`).
- Handler progress (e.g. the number of artifacts captured).
- A sanitized environment audit line.
- Sanitized tracebacks captured by the dispatcher for failed execution.

The log panel keeps up to the most recent 100 lines and refreshes automatically.

The sanitizer covers controlled job log entries, structured errors, and persisted fields. Output
written directly to stdout/stderr by Ultralytics or another third-party component is not guaranteed
to be captured by this API log stream or processed by the sanitizer. A terminal job's state,
result, and error are immutable; historical or already-in-flight log entries may remain readable
without changing that terminal snapshot.

### 5.2 Error codes and troubleshooting

The *Status Monitor* shows the `error_code` and `error_message`. Common causes:

| Error code | What it means | Recommended fix |
|---|---|---|
| `SEC_ERR_001` | Security policy violation: a path fell outside the whitelist, shell execution was requested, or whitelisting was disabled. | Ensure every path in *Model Path*, *Data Source*, and *Output Directory* lives under a server-configured trusted root (see §7.10 and Appendix — Security guarantees). |
| `PARAM_VALIDATION_FAILED` | A parameter was missing, out of range, or of the wrong type. | Check required fields (e.g. `model_path`, `data_source`) and value bounds (e.g. confidence in `(0.0, 1.0]`). |
| `TASK_TYPE_UNKNOWN` | The submitted task type is not registered. | Use one of `predict`, `train`, `val`, `export`, `diagnose`. |
| `USER_CANCELLED` | The job was cancelled by the user. | Re-submit the job if you need the result. |
| `TIMEOUT` | Execution exceeded the 300-second deadline. | Reduce the workload (fewer epochs / smaller inputs) or re-run. |
| `EXEC_ERR_500` | An unhandled exception occurred in the handler. | Inspect the live logs for the captured traceback. |
| `HANDLER_EXEC_FAILED` | The handler reported a controlled failure (e.g. no supported media files, model load error). | Read the specific message in the *Error Diagnostics* box. |

### 5.3 Common scenarios

| Symptom | Likely cause | Guidance |
|---|---|---|
| **Out-of-memory (OOM) on GPU** | Model + batch too large for VRAM. | Reduce `batch_size`, use `device=cpu`, or free the GPU by restarting the app. |
| **"no supported media files found"** | `predict` pointed at a directory with no supported files. | Point *Data Source* at a valid image/video file or directory. |
| **"Not a YAML file" / validation reject** | `train`/`val` received a non-YAML data source. | Supply a dataset `.yaml` (e.g. `coco8.yaml`). |
| **Job stuck in `PENDING`/`RUNNING`** | A previous process died mid-execution. | Orphaned active jobs are healed to `FAILED` on the next app restart. |
| **"Output directory does not exist"** | You tried to open the output folder before any artifact was written. | Wait for `COMPLETED`, then reopen. |

---

## 6. Artifact Retrieval & Verification

Handlers write outputs under `<output_dir>/<job_id>/` and collect existing files into the job's
artifact manifest. The manager normalizes those files to safe relative artifact IDs before the API
exposes them. Failed or cancelled jobs may leave partial files on disk, but only manifest entries
are downloadable through the API.

### 6.1 Locating artifacts

Open the **📁 Artifacts** sub-tab. The *Generated Artifacts* table lists each file with:

- **Filename** — the file’s base name.
- **Path** — in the same-host Gradio client, a resolved local source path when it can be safely
  reconstructed; otherwise an API download URL.

Alternatively, click **📂 Open Folder** to open the job’s specific output directory in your
operating system’s file manager (Windows Explorer / macOS Finder / Linux file manager).

### 6.2 Previewing image artifacts

The *Artifact Preview* pane renders the first image file produced by the job. When a job produces
multiple images, **◀ Prev** / **Next ▶** buttons and a selector dropdown become available to cycle
through them. Clicking an image row in the artifacts table also loads it into the preview.

> **Tip** — Selecting a non-image row (e.g. a `.pt` or `.csv` file) does not change the preview; a
> short notice explains that the file cannot be rendered.

### 6.3 What each task produces

| Task type | Representative artifacts |
|---|---|
| `predict` | Annotated images/labels saved per batch under `.../job_id/batch_NNN/`. |
| `train` | `best.pt`, `last.pt`, `results.csv`, `results.png`, confusion matrix and curve plots, sample batches (`train_batch*.jpg`, `val_batch*_labels.jpg`). |
| `val` | `results.csv`, `confusion_matrix.png`, `F1_curve.png`, `PR_curve.png`, `P_curve.png`, `R_curve.png`, sample validation batches. |
| `export` | The exported model file (e.g. `yolov8n.onnx`) plus the job-local `.pt` copy. |
| `diagnose` | `system_diagnostics.json` and `system_diagnostics.txt`. |

> **Note** — Production handlers capture artifacts with a sorted **full-tree scan** of the job
> directory. The API filters the result to existing files below that job root and serves only exact
> manifest matches; it does not expose a general filesystem browser.

### 6.4 Verifying exported models

For `export` jobs, the exported model (e.g. `yolov8n.onnx`) is written *inside* the job’s own
directory, next to a job-local copy of the source weights. This isolation guarantees that concurrent
exports never overwrite each other and that the original model directory is never modified. Locate
the `.onnx` file in the artifacts table and verify it exists on disk at the listed path.

---

## 7. Standalone REST Engine (FastAPI)

The **standalone REST engine** (`main_engine.py`) exposes the same F1 task dispatcher as a plain HTTP
API with native OpenAPI documentation. It has no Gradio dependency: it serves jobs, logs, cancellation
and artifact delivery directly, so a reviewer can verify the platform end-to-end with only a web
browser and command-line tools such as `curl`.

### 7.1 Starting the engine

Any of the following starts the engine:

| Command | What it does |
|---|---|
| `python start_studio.py` | One-command launcher: starts the FastAPI engine (auto-started if `/health` is not ready), then launches the Gradio WebUI. |
| `python main_engine.py` | Runs only the standalone FastAPI engine (Gradio-free). |
| `uvicorn main_engine:app --host 127.0.0.1 --port 8000` | Direct ASGI launch for embedded/development setups. |

Verify liveness at any time:

```bash
curl http://127.0.0.1:8000/health
# → {"status":"ok","service":"YOLO-Master F1 Task Engine"}
```

`/health` is a static liveness probe and does not depend on job state.

### 7.2 Interactive API docs

The engine exposes its own OpenAPI/Swagger surface, no extra tooling required:

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8000/docs` | Swagger UI — interactive, form-driven endpoint explorer. |
| `http://127.0.0.1:8000/openapi.json` | Raw OpenAPI schema consumed by clients/generators. |
| `http://127.0.0.1:8000/redoc` | Redoc reference view of the same schema. |
| `docs/api/job_schema.json` | Repository file: the kernel schema registry for `JobRequest` and related types. |

### 7.3 Endpoint reference

| Method & path | Summary |
|---|---|
| `POST /api/v1/jobs` | Submit a new job (`201 Created`). Accepts a `JobRequest` body; rejects malformed bodies with `422`, duplicates with `409`, and a full pending queue with `429`. |
| `GET /api/v1/jobs` | List recent jobs, newest first, with `limit`/`offset` pagination. |
| `GET /api/v1/jobs/{job_id}` | Current status and metadata of one job (`404` when unknown). |
| `POST /api/v1/jobs/{job_id}/cancel` | Request cooperative cancellation (`202` new, `200` idempotent replay, `404` unknown, `409` not cancellable). |
| `GET /api/v1/jobs/{job_id}/logs` | Sanitized execution logs with `offset`/`limit` windowing. |
| `GET /api/v1/jobs/{job_id}/artifacts` | Artifact manifest and image artifact IDs (`/static/artifacts/...` references). |
| `GET /static/artifacts/{job_id}/{artifact_id}` | Download one artifact file, manifest-gated (fail-closed). |
| `GET /health` | Liveness probe. |

The job collection routes also accept a trailing slash (`/api/v1/jobs/`) for clients that disagree
with the canonical form.

### 7.4 Job submission example

Submit a CPU `predict` job with repository-relative paths (no machine-specific absolute paths):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "predict_demo_001",
    "task_type": "predict",
    "params": {
      "model_path": "ckpts/yolov8n.pt",
      "data_source": "ultralytics/assets/bus.jpg",
      "conf": 0.25,
      "device": "cpu"
    },
    "output": {
      "output_dir": "runs/predict"
    }
  }'
```

A successful submission returns `201 Created` with the server-normalized `JobRequest`.

> **Important** — client-supplied security fields do **not** determine the final server whitelist.
> On every submission the engine forces `allow_shell=False` and `path_whitelisted=True`, discards the
> caller's `security_constraints.allowed_paths` / `allowed_path_patterns`, and replaces them with the
> server-configured trusted roots (`F1_MODEL_ROOTS` + `F1_DATA_ROOTS`). `model_path`, `data_source`
> and `output_dir` are then resolved independently and must each land inside their trusted root
> (`F1_OUTPUT_ROOT` for output). See §7.10.

### 7.5 Status query

```bash
curl http://127.0.0.1:8000/api/v1/jobs/predict_demo_001
```

The response includes the canonical lifecycle status plus metadata:

```json
{
  "job_id": "predict_demo_001",
  "task_type": "predict",
  "status": "completed",
  "created_at": "2026-09-12T03:00:45.000000+00:00",
  "started_at": "2026-09-12T03:00:46.000000+00:00",
  "completed_at": "2026-09-12T03:00:52.000000+00:00",
  "duration": 6.0,
  "error_code": null,
  "error_message": null,
  "artifact_count": 2,
  "metadata": { "created_by": "anonymous", "priority": "normal", "tags": [] }
}
```

`status` is one of `pending | running | completed | failed | cancelled`.

### 7.6 Live Logs

```bash
curl "http://127.0.0.1:8000/api/v1/jobs/predict_demo_001/logs?offset=0&limit=100"
```

- `offset` — number of lines to skip from the start (default `0`, `>= 0`).
- `limit` — maximum lines in this window (default `null` = the full remainder, capped at `10000`).
- `next_offset` — the cursor for the next page, or `null` once the tail is reached.

Every structured line returned by this endpoint is sanitized: credentials (Bearer tokens, `sk-`
keys, `KEY=value` secrets) are redacted as `***REDACTED***` before the line reaches the response or
persisted state. This does not claim capture or sanitization of arbitrary child-process
stdout/stderr outside the structured log path.

### 7.7 Cancellation

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs/predict_demo_001/cancel
```

Cancellation is asynchronous and manager-owned: the engine sets the cancellation flag, stops and
joins the owned process tree, then publishes `CANCELLED` with error code `USER_CANCELLED`.
Dispatcher/handler checkpoints supplement this process-level stop between engine calls.

- `202 Accepted` — a fresh cancellation request was acknowledged.
- `200 OK` — an idempotent replay against a job already in a terminal state, returning its existing state.
- `409 Conflict` — the job is not cancellable.
- `404 Not Found` — the job is unknown.

Terminal jobs (`completed` / `failed` / `cancelled`) cannot be re-cancelled; replaying the request is
safe and does not produce a second lifecycle transition.

### 7.8 Artifacts / download

```bash
curl http://127.0.0.1:8000/api/v1/jobs/predict_demo_001/artifacts
```

The response lists the handler-produced, manager-normalized artifact manifest (`artifacts`, with per-file
`download_url`) and safe image artifact IDs (`image_artifacts`). Server filesystem paths are never
returned. Download one file via its `download_url`:

```bash
curl -O "http://127.0.0.1:8000/static/artifacts/predict_demo_001/results.csv"
```

Artifact delivery is **manifest-gated and fail-closed**:

- Only files recorded by the dispatcher after a successful execution are served; unlisted files and
  directory-traversal attempts return `404`.
- Each job's artifacts are isolated under its own `job_id` sub-directory, and the route performs an
  exact manifest lookup rather than joining client input onto a filesystem path, so cross-job access
  is blocked.

### 7.9 Console entry points

The platform provides three client entry points with different hosting relationships:

| Entry point | Path | Notes |
|---|---|---|
| Zero-build console | `/` | Always mounted by FastAPI (`frontend/index.html` + `app.js`, plain ES6). |
| React console | `/console` | Served only when `web/dist` is present on disk. |
| Gradio WebUI | `http://127.0.0.1:7860` | Runs in the `start_studio.py` launcher process, separate from the API child service; it calls FastAPI through `StudioJobsApiClient`. |

The repository contains React/TypeScript/Vite source under `web/`, but does not commit a guaranteed
`web/dist` deployment artifact. The current F1 CI workflow does not run Node lint, TypeScript
checking, or a Vite production build.

### 7.10 Environment configuration

The following `F1_*` variables are read by the engine/manager source. Variables whose separator is
`os.pathsep` accept multiple entries (`;` on Windows, `:` on Unix); comma-separated variables accept
a comma-delimited list.

| Variable | Meaning | Default |
|---|---|---|
| `F1_JOBS_STATE_PATH` | API/backend JobsManager persistence file; Gradio observes it through the Studio Job API. | `runs/jobs_state.json` |
| `F1_ENGINE_HOST` / `F1_ENGINE_PORT` | Bind address for `python main_engine.py`. | `127.0.0.1` / `8000` |
| `F1_CORS_ORIGINS` | Comma-separated CORS allowlist; when non-empty it fully replaces the dev defaults. | dev defaults (§7.11) |
| `F1_MODEL_ROOTS` | Server-owned trusted model roots (`os.pathsep`-separated). | project root (cwd) |
| `F1_DATA_ROOTS` | Server-owned trusted data roots (`os.pathsep`-separated). | project root (cwd) |
| `F1_OUTPUT_ROOT` | Server-owned trusted output root (single). | `<cwd>/runs` |
| `F1_NETWORK_INPUT_HOSTS` | Comma-separated hostnames allowed as network `data_source`. | *(empty — network inputs rejected)* |
| `F1_MAX_PENDING_JOBS` | Pending-job capacity before `429`. | `100` |
| `F1_CPU_CONCURRENCY` | CPU execution slots. | `2` |
| `F1_GPU_CONCURRENCY` | Accelerator execution slots. | `1` |
| `F1_STOP_GRACE_SECONDS` | Worker stop grace (0–30). | `2` |
| `F1_STUDIO_API_URL` | Studio API origin used by the Gradio client. | `http://127.0.0.1:8000` |

### 7.11 CORS behavior

By default the engine adds `CORSMiddleware` with `allow_methods=["*"]` and `allow_headers=["*"]`, and
an origin allowlist covering local development: `http://localhost:8000`, `http://127.0.0.1:8000`,
the `null` origin (for `file://` pages), and the local Vite/CRA/port `5173` / `3000` / `8080`
spellings. Setting `F1_CORS_ORIGINS` to a non-empty comma-separated list **replaces** these defaults
rather than appending to them.

### 7.12 Relationship to the Gradio Jobs tab

The FastAPI REST engine is the sole lifecycle owner. It owns the process-wide
`f1.jobs_manager.JobsManager`, workers, lifecycle state, and the `F1_JOBS_STATE_PATH` persistence file.
The Gradio **📋 Jobs** tab uses `StudioJobsApiClient` to submit, poll, cancel, and download through the
task API; it neither opens that state file nor creates another manager. The `JobsManager` name exported
from `f1/ui/jobs_tab.py` is only a lazy compatibility shim for legacy imports.

### 7.13 Relationship to Agent Skills

The F1 production runtime ends at `f1/handlers/` and does not dispatch through
`agent/runtime/cli/async_jobs.py` or an Agent worker. `agent/` is a separate skill/CLI integration
layer documented by `agent/SKILL.md`. Agent validation does not replace F1 API, lifecycle, handler,
or security tests. Shared task names are defined through the explicit contracts under `core/`.

---

## 8. Recorded verification and limits

The recorded review range is baseline
`acce839c7e895d6b179de7f7093fa879e237cc7b` to reviewed commit
`0e7a8f83b53c97abc8eb3caedc532f5779fbb086`.

Windows/Python 3.12 verification in the repository `.venv` produced:

```text
412 passed, 11 skipped, 3 warnings
78% combined f1/core coverage
F1 smoke: 5/5 passed
```

The 11 skips require Windows symlink privileges. The Ubuntu/Python 3.10 source-branch CI at the
reviewed commit produced `422 passed, 1 skipped, 3 warnings`, 78% coverage, smoke 5/5, and passing
Ruff lint/format:
<https://github.com/9-71/YOLO-Master/actions/runs/34793607227>. This is source-branch push evidence,
not a Tencent repository required PR check.

Current operational limits:

- one API process / one lifecycle owner; task execution uses spawned child processes
- local best-effort JSON persistence; no distributed durable queue, priority, or automatic retry
- static CPU/GPU slots; no memory-aware or per-GPU placement
- HTTP cursor polling; no WebSocket/SSE
- no authentication, RBAC, tenant isolation, or per-job OS sandbox
- Windows/Linux process containment verified; other POSIX platforms have weaker guarantees
- React deployment build and Node checks are not covered by the current F1 CI
- some OBB/classification metric displays remain incomplete

---

## Appendix — Quick reference

### Server-owned trusted roots

The submission form may serialize `allowed_paths` for compatibility, but client-supplied
`allowed_paths` and `allowed_path_patterns` never grant filesystem access. The FastAPI backend
discards both fields and replaces them with its trusted `F1_MODEL_ROOTS` and `F1_DATA_ROOTS`;
`output_dir` is checked independently against `F1_OUTPUT_ROOT` (see §7.10).

### Security guarantees

- **Shell execution** is permanently disabled (`allow_shell = False`).
- **Path whitelisting** is always enforced (`path_whitelisted = True`) using server-owned roots;
  the handler layer fails closed when its trusted whitelist is empty.
- **Directory traversal** (`../`) is neutralized by resolving paths to absolute form before the
  containment check.
- **Structured log sanitization** redacts credentials (API keys, `KEY=value` secrets) before
  controlled job entries reach the API console or persisted state. Arbitrary child stdout/stderr
  is outside this guarantee.
- **Job isolation** uses a unique `job_id` sub-directory per job to prevent output collisions.

### State persistence

The FastAPI service makes a best-effort local persistence of job history to
`runs/jobs_state.json`, using temporary-file replacement. Jobs left in `PENDING`/`RUNNING` when the
service restarts are marked `FAILED / SERVICE_RESTARTED`; execution is not resumed. This is not a
transactional database or durable distributed queue.
