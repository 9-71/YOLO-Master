/**
 * YOLO-Master F1 Verification Console (P2 Step 3).
 *
 * Zero-install single-page client for the standalone FastAPI engine. Plain
 * ES6 + Tailwind Play CDN; no build toolchain, no npm install. Runs served
 * by the engine itself (http://127.0.0.1:8000/) or opened directly from disk
 * via file:// — the engine's CORS allowlist includes the "null" origin that
 * browsers send for file:// pages (see DEV_CORS_ORIGINS in main_engine.py).
 *
 * Endpoints consumed:
 *   GET  /health                                connection probe
 *   GET  /api/v1/jobs                           recent-jobs listing
 *   POST /api/v1/jobs                           job dispatch (core/schema.py)
 *   GET  /api/v1/jobs/{id}                      status + metadata
 *   POST /api/v1/jobs/{id}/cancel               cooperative cancellation
 *   GET  /api/v1/jobs/{id}/logs?offset=&limit=  cursor-paginated logs
 *   GET  /api/v1/jobs/{id}/artifacts            manifest + image paths
 *   GET  /static/artifacts/{id}/{file}          fail-closed artifact delivery
 *
 * HTTP 400 / 404 / 409 / 422 responses and network failures surface as
 * dismissible banners with the FastAPI `detail` payload extracted.
 */

"use strict";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** localStorage key persisting the engine base URL across reloads. */
const BASE_URL_STORAGE_KEY = "f1.console.baseUrl";
/** Default engine address (matches `python main_engine.py`). */
const DEFAULT_BASE_URL = "http://localhost:8000";
/**
 * Canonical job-collection path (no trailing slash). The engine serves both
 * `/api/v1/jobs` and `/api/v1/jobs/`, so keeping every call site on this
 * constant stays clean regardless of slash formatting.
 */
const JOBS_PATH = "/api/v1/jobs";

/** Polling intervals in milliseconds. */
const POLL_MS = { jobs: 2000, logs: 1000, health: 5000 };
/** Maximum log lines kept in the terminal DOM (mirrors the Gradio console cap). */
const MAX_TERMINAL_LINES = 1000;
/** Maximum log lines requested per cursor window. */
const LOG_WINDOW_LIMIT = 500;
/** Job listing page size. */
const JOBS_LIST_LIMIT = 50;
/** Seconds a banner stays visible before auto-dismissing. */
const BANNER_AUTO_HIDE_MS = 8000;

/**
 * Per-task form presets (model_path / data_source / output_dir), mirroring
 * f1/ui/jobs_tab.py TASK_FORM_PRESETS so handlers receive engine-compatible
 * inputs: predict takes an image source, train/val a dataset YAML.
 */
const TASK_PRESETS = {
  predict: { modelPath: "./ckpts/yolov8n.pt", dataSource: "ultralytics/assets/bus.jpg", outputDir: "runs/predict" },
  train: { modelPath: "./ckpts/yolov8n.pt", dataSource: "coco8.yaml", outputDir: "runs/train" },
  val: { modelPath: "./ckpts/yolov8n.pt", dataSource: "coco8.yaml", outputDir: "runs/val" },
  export: { modelPath: "./ckpts/yolov8n.pt", dataSource: "", outputDir: "runs/export" },
  diagnose: { modelPath: "", dataSource: "", outputDir: "runs/diagnose" },
};

/** Lifecycle states that are not terminal (mirrors f1.jobs_manager.ACTIVE_STATUSES). */
const ACTIVE_STATUSES = new Set(["pending", "running"]);

/** Tailwind badge classes per backend status value. */
const STATUS_BADGE = {
  pending: "border-amber-500/40 bg-amber-500/15 text-amber-300",
  running: "border-sky-500/40 bg-sky-500/15 text-sky-300",
  completed: "border-emerald-500/40 bg-emerald-500/15 text-emerald-300",
  failed: "border-rose-500/40 bg-rose-500/15 text-rose-300",
  cancelled: "border-slate-500/40 bg-slate-500/15 text-slate-300",
};

/** Friendly prefixes for the error banners requested by the P2 console spec. */
const HTTP_FRIENDLY = { 400: "Bad request", 404: "Not found", 409: "Conflict", 422: "Validation failed" };

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  baseUrl: (localStorage.getItem(BASE_URL_STORAGE_KEY) || DEFAULT_BASE_URL).replace(/\/+$/, ""),
  baseRev: 0,
  watchSeq: 0,
  jobsRequest: 0,
  jobsSettledRequest: 0,
  detailRequest: 0,
  artifactsRequest: 0,
  logFlight: null,
  tickFlight: null,
  cancellation: null,
  cancelTimer: null,
  jobs: [],
  selectedJobId: null,
  detail: null, // latest GET /api/v1/jobs/{id} payload
  logOffset: 0, // next log cursor passed as ?offset=
  jobsTimer: null,
  logsTimer: null,
};

/** Shorthand for document.getElementById. */
const $ = (id) => document.getElementById(id);

function engineIdentity() {
  return { baseUrl: state.baseUrl, rev: state.baseRev };
}

function engineCurrent(identity) {
  return identity.rev === state.baseRev && identity.baseUrl === state.baseUrl;
}

function watchIdentity() {
  return { ...engineIdentity(), jobId: state.selectedJobId, watchSeq: state.watchSeq };
}

function watchCurrent(identity) {
  return engineCurrent(identity) && identity.jobId === state.selectedJobId && identity.watchSeq === state.watchSeq;
}

function jobPath(jobId) {
  return `${JOBS_PATH}/${encodeURIComponent(jobId)}`;
}

// ---------------------------------------------------------------------------
// HTTP helpers (strict 400 / 404 / 409 / 422 handling)
// ---------------------------------------------------------------------------

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = "ApiError";
    this.status = status; // 0 = network failure (engine unreachable)
  }
}

/**
 * Extract a human-readable message from a FastAPI error body.
 *
 * `detail` is a plain string for raised HTTPException (400/404/409) and an
 * array of Pydantic violation objects for 422 validation errors.
 *
 * @param {object|null} body - Parsed JSON error body (may be null).
 * @returns {string} Human-readable failure description.
 */
function extractDetail(body) {
  const detail = body && body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const loc = (item.loc || []).filter((part) => part !== "body").join(".");
        const msg = item.msg || "invalid value";
        return loc ? `${loc}: ${msg}` : msg;
      })
      .join("; ");
  }
  if (detail && typeof detail === "object" && typeof detail.message === "string") {
    return `${detail.code ? `${detail.code}: ` : ""}${detail.message}`;
  }
  return JSON.stringify(body || {});
}

/**
 * Fetch a JSON endpoint on the engine, throwing {@link ApiError} on failure.
 *
 * @param {string} path - API path starting with "/".
 * @param {object} [options] - fetch options (method, body...).
 * @returns {Promise<object>} Parsed JSON response body.
 */
async function api(path, options = {}, baseUrl = state.baseUrl) {
  const url = baseUrl + path;
  let response;
  try {
    response = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
  } catch (error) {
    throw new ApiError(0, `Cannot reach engine at ${baseUrl} — ${error.message}`);
  }
  if (!response.ok) {
    let body = null;
    try {
      body = await response.json();
    } catch {
      body = null; // non-JSON error body (e.g. middleware rejection text)
    }
    const label = HTTP_FRIENDLY[response.status] || "Request failed";
    throw new ApiError(response.status, `${label} (HTTP ${response.status}): ${extractDetail(body)}`);
  }
  return response.json();
}

/**
 * Resolve a download reference (absolute URL or engine-relative path)
 * against the configured engine base URL.
 *
 * @param {string} path - download_url from the artifacts manifest.
 * @returns {string} Absolute URL usable in src/href attributes.
 */
function absUrl(path) {
  if (/^https?:\/\//.test(path)) return path;
  return state.baseUrl + (path.startsWith("/") ? path : `/${path}`);
}

// ---------------------------------------------------------------------------
// Connection header
// ---------------------------------------------------------------------------

/** Map of connection states to pill (border/text/dot) classes. */
const CONN_STYLES = {
  checking: {
    pill: "border-slate-700 bg-slate-800/60 text-slate-300",
    dot: "bg-slate-500",
  },
  online: {
    pill: "border-emerald-500/40 bg-emerald-500/15 text-emerald-300",
    dot: "bg-emerald-400 animate-pulse",
  },
  offline: {
    pill: "border-rose-500/40 bg-rose-500/15 text-rose-300",
    dot: "bg-rose-500",
  },
};

/**
 * Update the connection status pill. Polling paths call this silently;
 * only explicit user actions raise banners on top of it.
 *
 * @param {"checking"|"online"|"offline"} mode - State to display.
 */
function setConnState(mode) {
  const style = CONN_STYLES[mode] || CONN_STYLES.checking;
  $("conn-pill").className = `inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${style.pill}`;
  $("conn-dot").className = `h-2 w-2 rounded-full ${style.dot}`;
  $("conn-text").textContent =
    mode === "online" ? "Connected" : mode === "offline" ? "Offline" : "Checking…";
}

/** Ping GET /health and update the connection pill. Never raises banners. */
async function checkHealth() {
  const identity = engineIdentity();
  try {
    await api("/health", {}, identity.baseUrl);
    if (engineCurrent(identity)) setConnState("online");
  } catch {
    if (engineCurrent(identity)) setConnState("offline");
  }
}

/**
 * Apply the base URL input: persist it, reset console state and re-probe.
 */
function applyBaseUrl() {
  state.baseRev += 1;
  state.watchSeq += 1;
  window.clearInterval(state.cancelTimer);
  state.cancelTimer = null;
  state.cancellation = null;
  state.baseUrl = $("base-url").value.trim().replace(/\/+$/, "") || DEFAULT_BASE_URL;
  $("base-url").value = state.baseUrl;
  localStorage.setItem(BASE_URL_STORAGE_KEY, state.baseUrl);

  // A new engine may not know the previous selection; reset derived state.
  state.selectedJobId = null;
  state.detail = null;
  state.logOffset = 0;
  state.jobs = [];
  closeLightbox();
  $("dispatch-submit").disabled = false;
  $("dispatch-submit").textContent = "Dispatch Job";
  $("dispatch-feedback").textContent = "";
  stopLogsTimer();
  $("terminal-body").textContent = "";
  $("terminal-job").textContent = "no job selected";
  $("terminal-status").className = "hidden";
  $("job-detail").classList.add("hidden");
  renderJobs();
  renderArtifacts({ artifacts: [], image_artifacts: [] });
  setConnState("checking");
  checkHealth();
  refreshJobs();
}

// ---------------------------------------------------------------------------
// Banners (global, dismissible, auto-hiding)
// ---------------------------------------------------------------------------

let bannerTimer = null;

/**
 * Show (or restyle) the global banner and arm its auto-dismiss timer.
 *
 * @param {"error"|"success"} kind - Banner variant.
 * @param {string} message - Text to display (multiline preserved).
 */
function showBanner(kind, message) {
  const banner = $("banner");
  const classes =
    kind === "success"
      ? "border-emerald-500/40 bg-emerald-950/95 text-emerald-200"
      : "border-rose-500/40 bg-rose-950/95 text-rose-200";
  banner.className = `fixed top-0 inset-x-0 z-50 border-b px-4 py-3 text-sm shadow-lg ${classes}`;
  $("banner-text").textContent = message;
  banner.classList.remove("hidden");
  window.clearTimeout(bannerTimer);
  bannerTimer = window.setTimeout(() => banner.classList.add("hidden"), BANNER_AUTO_HIDE_MS);
}

function hideBanner() {
  window.clearTimeout(bannerTimer);
  $("banner").classList.add("hidden");
}

// ---------------------------------------------------------------------------
// Job dispatch form
// ---------------------------------------------------------------------------

/**
 * Build a unique job identifier: ``{task_type}_{YYYYMMDD_HHMMSS}_{uuid8}``,
 * the same shape f1.jobs_manager.JobsManager.submit_job generates.
 *
 * @param {string} taskType - Selected task type value.
 * @returns {string} Unique job id.
 */
function buildJobId(taskType) {
  // crypto.randomUUID requires a secure context; http://localhost and file://
  // both qualify, which covers the console's two supported launch modes.
  const stamp = new Date()
    .toISOString()
    .replace(/[-:]/g, "")
    .replace(/\.\d+Z$/, "")
    .replace("T", "_");
  return `${taskType}_${stamp}_${crypto.randomUUID().replace(/-/g, "").slice(0, 8)}`;
}

/** Split a comma-separated whitelist string into trimmed non-empty entries. */
function parseList(raw) {
  return raw.split(",").map((item) => item.trim()).filter(Boolean);
}

/** Repopulate model/data/output fields from the preset of the selected task. */
function applyTaskPreset() {
  const preset = TASK_PRESETS[$("task-type").value];
  $("model-path").value = preset.modelPath;
  $("data-source").value = preset.dataSource;
  $("output-dir").value = preset.outputDir;
}

/**
 * Handle the dispatch form submit: validate client-side, POST
 * /api/v1/jobs, then auto-select the created job in the console.
 *
 * @param {Event} event - submit event (prevented).
 */
async function onDispatchSubmit(event) {
  event.preventDefault();
  const submitBtn = $("dispatch-submit");
  const feedback = $("dispatch-feedback");
  feedback.className = "hidden";

  const taskType = $("task-type").value;
  const modelPath = $("model-path").value.trim();
  const dataSource = $("data-source").value.trim();
  const device = $("device").value.trim() || "cpu";
  const outputDir = $("output-dir").value.trim();

  // Optional extra params must be a JSON object (client-side 400-style check).
  let extraParams = {};
  const extraRaw = $("extra-params").value.trim();
  if (extraRaw) {
    try {
      extraParams = JSON.parse(extraRaw);
      if (extraParams === null || typeof extraParams !== "object" || Array.isArray(extraParams)) {
        throw new Error("top-level value must be a JSON object, e.g. {\"epochs\": 5}");
      }
    } catch (error) {
      showBanner("error", `Extra params are not valid JSON: ${error.message}`);
      return;
    }
  }

  // Params shape mirrors JobsManager.submit_job: predict/train/val carry
  // model_path + data_source + conf + device; export carries model_path +
  // format; diagnose takes none. Extra JSON keys merge in first and the
  // form's explicit fields win.
  const params = taskType === "diagnose" ? {} : { ...extraParams };
  if (["predict", "train", "val"].includes(taskType)) {
    params.model_path = modelPath;
    params.data_source = dataSource;
    if (taskType !== "train") params.conf = parseFloat($("conf").value);
    if (taskType === "train" && params.epochs === undefined) params.epochs = 10;
    params.device = device;
  } else if (taskType === "export") {
    params.model_path = modelPath;
    params.format = params.format || "onnx";
  }

  const payload = {
    job_id: buildJobId(taskType),
    task_type: taskType,
    params,
    output: { output_dir: outputDir },
    runtime_tracking: { timeout_seconds: 300, cancellable: true },
    security_constraints: {
      allow_shell: false, // server forces these fail-closed anyway
      path_whitelisted: true,
      allowed_paths: parseList($("allowed-paths").value),
    },
  };

  const identity = engineIdentity();
  submitBtn.disabled = true;
  submitBtn.textContent = "Dispatching…";
  try {
    const job = await api(JOBS_PATH, { method: "POST", body: JSON.stringify(payload) }, identity.baseUrl);
    if (!engineCurrent(identity)) return;
    feedback.textContent = `✅ Job ${job.job_id} submitted (201) — selected in console`;
    feedback.className = "text-xs font-semibold text-emerald-400";
    showBanner("success", `Job ${job.job_id} dispatched successfully — monitoring now.`);
    await refreshJobs();
    if (engineCurrent(identity)) await selectJob(job.job_id);
  } catch (error) {
    if (!engineCurrent(identity)) return;
    feedback.textContent = `❌ ${error.message}`;
    feedback.className = "text-xs font-semibold text-rose-400";
    showBanner("error", error.message);
  } finally {
    if (engineCurrent(identity)) {
      submitBtn.disabled = false;
      submitBtn.textContent = "Dispatch Job";
    }
  }
}

// ---------------------------------------------------------------------------
// Job supervision & history table
// ---------------------------------------------------------------------------

/**
 * Compute seconds elapsed since an ISO 8601 timestamp (live for active jobs).
 *
 * @param {string} iso - creation timestamp.
 * @returns {number|null} Elapsed seconds, or null when unparsable.
 */
function elapsedSeconds(iso) {
  const created = Date.parse(iso);
  return Number.isNaN(created) ? null : Math.max(0, (Date.now() - created) / 1000);
}

/**
 * Render execution time from the Runtime contract, excluding queue wait.
 * Active execution ticks from started_at; terminal durations come from the API.
 *
 * @param {{job_id: string, status: string, created_at: string}} job - Summary row.
 * @returns {string} Duration string like "12.3s", or "N/A".
 */
function jobDuration(job) {
  if (ACTIVE_STATUSES.has(job.status) && job.started_at) {
    const seconds = elapsedSeconds(job.started_at);
    return seconds === null ? "N/A" : `${seconds.toFixed(2)}s`;
  }
  return typeof job.duration === "number" ? `${job.duration.toFixed(2)}s` : "N/A";
}

/** Format an ISO timestamp as local HH:MM:SS. */
function fmtClock(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleTimeString();
}

/** Create a <td> with the given text and optional className. */
function td(text, className) {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) cell.className = className;
  return cell;
}

/** Create a status badge <span>. */
function badge(status) {
  const span = document.createElement("span");
  span.textContent = status;
  span.className = `inline-block rounded-full border px-2.5 py-0.5 text-[11px] font-bold uppercase ${STATUS_BADGE[status] || "border-slate-600 bg-slate-800 text-slate-300"}`;
  return span;
}

/** Render the jobs table from the latest GET /api/v1/jobs payload. */
function renderJobs() {
  const tbody = $("jobs-body");
  tbody.textContent = "";
  $("jobs-empty").classList.toggle("hidden", state.jobs.length > 0);

  for (const job of state.jobs) {
    const active = ACTIVE_STATUSES.has(job.status);
    const selected = job.job_id === state.selectedJobId;
    const tr = document.createElement("tr");
    tr.className = `cursor-pointer border-b border-slate-800/60 transition hover:bg-slate-800/50 ${selected ? "bg-slate-800" : ""}`;
    tr.title = "Select this job";

    const jobIdCell = td(job.job_id, "px-4 py-2 font-mono text-xs text-cyan-300 max-w-0 truncate");
    jobIdCell.style.maxWidth = "16rem";
    const typeCell = td(job.task_type.toUpperCase(), "px-4 py-2 text-xs font-bold uppercase text-slate-300");
    const statusCell = document.createElement("td");
    statusCell.className = "px-4 py-2";
    statusCell.appendChild(badge(job.status));
    tr.append(jobIdCell, typeCell, statusCell, td(jobDuration(job), "px-4 py-2 font-mono text-xs"), td(fmtClock(job.created_at), "px-4 py-2 text-xs text-slate-400"));

    const actionsCell = document.createElement("td");
    actionsCell.className = "px-4 py-2 text-right whitespace-nowrap";
    const viewBtn = document.createElement("button");
    viewBtn.type = "button";
    viewBtn.textContent = "View Details";
    viewBtn.className = "rounded-md border border-slate-700 px-2 py-1 text-[11px] font-semibold hover:bg-slate-700";
    viewBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      selectJob(job.job_id).catch((error) => showBanner("error", error.message));
    });
    const cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.textContent = state.cancellation?.jobId === job.job_id ? "Cancelling…" : "Cancel Job";
    cancelBtn.className = active
      ? "ml-1.5 rounded-md border border-rose-500/40 bg-rose-500/10 px-2 py-1 text-[11px] font-semibold text-rose-300 hover:bg-rose-500/20"
      : "ml-1.5 rounded-md border border-slate-800 px-2 py-1 text-[11px] font-semibold text-slate-600 cursor-not-allowed";
    cancelBtn.disabled = !active || state.cancellation !== null;
    cancelBtn.title = active ? `POST /api/v1/jobs/${job.job_id}/cancel` : "Only PENDING/RUNNING jobs can be cancelled";
    if (active) {
      cancelBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        cancelJob(job.job_id);
      });
    }
    actionsCell.append(viewBtn, cancelBtn);
    tr.appendChild(actionsCell);

    tr.addEventListener("click", () => selectJob(job.job_id).catch((error) => showBanner("error", error.message)));
    tbody.appendChild(tr);
  }
}

/** Fetch the recent-jobs listing and re-render the table. */
async function refreshJobs() {
  const identity = engineIdentity();
  const request = ++state.jobsRequest;
  try {
    const body = await api(`${JOBS_PATH}?limit=${JOBS_LIST_LIMIT}`, {}, identity.baseUrl);
    // A pending poll must not invalidate a slow response from this engine.
    if (!engineCurrent(identity) || request < state.jobsSettledRequest) return;
    state.jobsSettledRequest = request;
    state.jobs = body.jobs;
    renderJobs();
    setConnState("online");
  } catch {
    if (!engineCurrent(identity) || request < state.jobsSettledRequest) return;
    state.jobsSettledRequest = request;
    setConnState("offline");
  }
}

/** Observe cancellation separately from the selected job and Live toggles. */
async function observeCancellation(operation) {
  if (state.cancellation !== operation || !engineCurrent(operation) || operation.inFlight) return;
  operation.inFlight = true;
  try {
    const detail = await api(jobPath(operation.jobId), {}, operation.baseUrl);
    if (state.cancellation !== operation || !engineCurrent(operation)) return;
    if (ACTIVE_STATUSES.has(detail.status)) return;
    window.clearInterval(state.cancelTimer);
    state.cancelTimer = null;
    state.cancellation = null;
    const cancelled = detail.status === "cancelled" && detail.error_code === "USER_CANCELLED";
    showBanner(cancelled ? "success" : "error", cancelled
      ? `Job ${operation.jobId} reached cancelled / USER_CANCELLED. Cancellation completed.`
      : `Cancellation race for ${operation.jobId}: job reached ${detail.status}${detail.error_code ? ` / ${detail.error_code}` : ""}.`);
    renderJobs();
    await refreshJobs();
    if (engineCurrent(operation) && state.selectedJobId === operation.jobId) await logsTick();
  } catch (error) {
    if (state.cancellation === operation && engineCurrent(operation)) showBanner("error", error.message);
  } finally {
    operation.inFlight = false;
  }
}

/** A 202 is an acknowledgement; only observed terminal state completes cancellation. */
async function cancelJob(jobId) {
  if (state.cancellation) return;
  if (!window.confirm(`Request cooperative cancellation of job ${jobId}?`)) return;
  const operation = { ...engineIdentity(), jobId, inFlight: false };
  state.cancellation = operation;
  renderJobs();
  try {
    const body = await api(`${jobPath(jobId)}/cancel`, { method: "POST" }, operation.baseUrl);
    if (state.cancellation !== operation || !engineCurrent(operation)) return;
    showBanner("success", `${body.message || "Cancellation acknowledged"} Waiting for terminal cancellation state…`);
    state.cancelTimer = window.setInterval(() => void observeCancellation(operation), POLL_MS.logs);
    await observeCancellation(operation);
  } catch (error) {
    if (state.cancellation !== operation || !engineCurrent(operation)) return;
    state.cancellation = null;
    renderJobs();
    showBanner("error", error.message);
  }
}

// ---------------------------------------------------------------------------
// Selected job: detail, terminal, artifacts
// ---------------------------------------------------------------------------

/**
 * Select a job: load detail + artifacts, reset the log cursor and restart
 * cursor-based log tailing. Selecting the same job re-syncs without reset.
 *
 * @param {string} jobId - Job to monitor.
 */
async function selectJob(jobId) {
  const sameJob = jobId === state.selectedJobId;
  if (!sameJob) {
    state.watchSeq += 1;
    state.logOffset = 0;
    state.detail = null;
    $("terminal-body").textContent = "";
    renderArtifacts({ artifacts: [], image_artifacts: [] });
    closeLightbox();
  }
  state.selectedJobId = jobId;
  const identity = watchIdentity();
  stopLogsTimer();
  $("terminal-job").textContent = jobId;
  $("terminal-status").className = "hidden";
  renderJobs();
  $("job-detail").classList.remove("hidden");
  $("job-detail-body").textContent = "Loading…";
  try {
    const detail = await refreshDetail();
    if (!watchCurrent(identity) || !detail) return;
    const terminal = !ACTIVE_STATUSES.has(detail.status);
    await pollLogs(terminal);
    if (!watchCurrent(identity)) return;
    const artifactsReady = await loadArtifacts();
    if (watchCurrent(identity) && (!terminal || !artifactsReady)) startLogsTimer();
  } catch (error) {
    if (!watchCurrent(identity)) return;
    showBanner("error", error.message);
    startLogsTimer(); // retry status / final cursor after a transient failure
  }
}

/** GET /api/v1/jobs/{id}; discard superseded selections and requests. */
async function refreshDetail() {
  const identity = watchIdentity();
  if (!identity.jobId) return null;
  const request = ++state.detailRequest;
  const detail = await api(jobPath(identity.jobId), {}, identity.baseUrl);
  if (!watchCurrent(identity) || request !== state.detailRequest) return null;
  state.detail = detail;
  renderDetail(detail);
  return detail;
}

/** Render the selected job's status card (and the terminal status chip). */
function renderDetail(detail) {
  const body = $("job-detail-body");
  body.textContent = "";

  const header = document.createElement("div");
  header.className = "flex flex-wrap items-center gap-3";
  const idSpan = document.createElement("span");
  idSpan.textContent = detail.job_id;
  idSpan.className = "font-mono text-sm font-bold text-cyan-300";
  header.append(idSpan, badge(detail.status));
  body.appendChild(header);

  const facts = [
    ["Task Type", detail.task_type.toUpperCase()],
    ["Duration", jobDuration(detail)],
    ["Created At", detail.created_at ? new Date(detail.created_at).toLocaleString() : "—"],
    ["Artifacts", String(detail.artifact_count)],
    ["Created By", (detail.metadata && detail.metadata.created_by) || "—"],
    ["Priority", (detail.metadata && detail.metadata.priority) || "—"],
  ];
  const grid = document.createElement("dl");
  grid.className = "mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3";
  for (const [label, value] of facts) {
    const term = document.createElement("dt");
    term.textContent = label;
    term.className = "text-[10px] font-bold uppercase tracking-wider text-slate-500";
    const desc = document.createElement("dd");
    desc.textContent = value;
    desc.className = "font-mono text-xs text-slate-300";
    const cell = document.createElement("div");
    cell.append(term, desc);
    grid.appendChild(cell);
  }
  body.appendChild(grid);

  if (detail.error_message) {
    const errorBox = document.createElement("div");
    errorBox.className = "mt-3 rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2";
    const code = document.createElement("div");
    code.textContent = `Error: ${detail.error_code || "UNKNOWN"}`;
    code.className = "text-xs font-bold text-rose-300";
    const message = document.createElement("div");
    message.textContent = detail.error_message;
    message.className = "mt-0.5 break-words font-mono text-[11px] text-rose-200/80";
    errorBox.append(code, message);
    body.appendChild(errorBox);
  }

  const chip = $("terminal-status");
  chip.textContent = detail.status.toUpperCase();
  chip.className = `inline-block rounded-full border px-2 py-0.5 text-[10px] font-bold uppercase ${STATUS_BADGE[detail.status] || ""}`;
}

/** GET /api/v1/jobs/{id}/logs and advance the cursor. */
async function pollLogs(drain = false) {
  const identity = watchIdentity();
  if (!identity.jobId) return null;
  const prior = state.logFlight;
  if (prior && watchCurrent(prior.identity)) {
    await prior.promise;
    return drain && watchCurrent(identity) ? pollLogs(true) : null;
  }
  const flight = { identity, promise: null };
  const pull = async () => {
    let body;
    do {
      const offset = state.logOffset;
      body = await api(`${jobPath(identity.jobId)}/logs?offset=${offset}&limit=${LOG_WINDOW_LIMIT}`, {}, identity.baseUrl);
      if (!watchCurrent(identity)) return null;
      const next = body.next_offset ?? body.offset + body.logs.length;
      if (body.next_offset !== null && next <= offset) throw new Error("Log cursor did not advance");
      appendLogLines(body.logs);
      state.logOffset = next;
    } while (drain && body.next_offset !== null);
    return body;
  };
  state.logFlight = flight;
  flight.promise = pull();
  try {
    return await flight.promise;
  } finally {
    if (state.logFlight === flight) state.logFlight = null;
  }
}

/**
 * Append log lines to the dark console window, capping the DOM at
 * MAX_TERMINAL_LINES and honoring the auto-scroll toggle.
 *
 * @param {string[]} lines - One flattened log line per entry.
 */
function appendLogLines(lines) {
  const pre = $("terminal-body");
  for (const line of lines) {
    const div = document.createElement("div");
    div.textContent = line; // textContent: sanitized display, never HTML
    pre.appendChild(div);
  }
  while (pre.children.length > MAX_TERMINAL_LINES) pre.removeChild(pre.firstChild);
  if ($("auto-scroll").checked) pre.scrollTop = pre.scrollHeight;
}

/**
 * GET /api/v1/jobs/{id}/artifacts and render the gallery + file list.
 */
async function loadArtifacts() {
  const identity = watchIdentity();
  if (!identity.jobId) return false;
  const request = ++state.artifactsRequest;
  try {
    const body = await api(`${jobPath(identity.jobId)}/artifacts`, {}, identity.baseUrl);
    if (!watchCurrent(identity) || request !== state.artifactsRequest) return false;
    renderArtifacts(body);
    return true;
  } catch (error) {
    if (watchCurrent(identity) && request === state.artifactsRequest) throw error;
    return false;
  }
}

/**
 * Render manifest-authorized artifacts. Nested IDs retain every path segment
 * when constructing preview/download URLs for image_artifacts.
 *
 * @param {{artifacts: Array<{filename: string, is_image: boolean, download_url: string}>, image_artifacts: string[]}} body - Artifacts payload.
 */
function renderArtifacts(body) {
  const gallery = $("artifact-gallery");
  const files = $("artifact-files");
  gallery.textContent = "";
  files.textContent = "";

  const images = [];
  for (const entry of body.artifacts || []) {
    if (entry.is_image) images.push({ filename: entry.filename, url: absUrl(entry.download_url) });
  }
  for (const artifactId of body.image_artifacts || []) {
    const filename = artifactId.split("/").pop();
    const encodedId = artifactId.split("/").map(encodeURIComponent).join("/");
    if (filename) {
      images.push({ filename, url: absUrl(`/static/artifacts/${encodeURIComponent(state.selectedJobId)}/${encodedId}`) });
    }
  }

  const seenUrls = new Set();
  for (const image of images) {
    if (seenUrls.has(image.url)) continue;
    seenUrls.add(image.url);

    const figure = document.createElement("figure");
    figure.className = "group overflow-hidden rounded-md border border-slate-800 bg-slate-950";
    const img = document.createElement("img");
    img.src = image.url;
    img.alt = image.filename;
    img.loading = "lazy";
    img.className = "h-28 w-full cursor-zoom-in object-cover transition group-hover:opacity-80";
    img.title = `${image.filename} — click to enlarge`;
    img.addEventListener("click", () => openLightbox(image.url, image.filename));
    img.addEventListener("error", () => img.classList.add("opacity-30"));
    const caption = document.createElement("figcaption");
    caption.className = "flex items-center justify-between gap-2 border-t border-slate-800 px-2 py-1";
    const name = document.createElement("span");
    name.textContent = image.filename;
    name.title = image.filename;
    name.className = "truncate font-mono text-[10px] text-slate-400";
    // Always-visible direct link: also the access path when the preview 404s.
    const open = document.createElement("a");
    open.href = image.url;
    open.target = "_blank";
    open.rel = "noopener";
    open.textContent = "Open";
    open.className = "shrink-0 text-[10px] font-semibold text-cyan-400 hover:underline";
    caption.append(name, open);
    figure.append(img, caption);
    gallery.appendChild(figure);
  }

  for (const entry of body.artifacts || []) {
    if (entry.is_image) continue;
    const row = document.createElement("li");
    row.className = "flex items-center gap-2 rounded-md border border-slate-800 bg-slate-950 px-3 py-2";
    const name = document.createElement("span");
    name.textContent = entry.filename;
    name.title = entry.filename;
    name.className = "flex-1 truncate font-mono text-xs text-slate-300";
    const url = absUrl(entry.download_url);
    const view = document.createElement("a");
    view.href = url;
    view.target = "_blank";
    view.rel = "noopener";
    view.textContent = "View";
    view.className = "text-xs font-semibold text-cyan-400 hover:underline";
    const download = document.createElement("a");
    download.href = url;
    download.download = entry.filename;
    download.textContent = "Download";
    download.className = "text-xs font-semibold text-cyan-400 hover:underline";
    row.append(name, view, download);
    files.appendChild(row);
  }

  const total = seenUrls.size + (body.artifacts || []).filter((entry) => !entry.is_image).length;
  $("artifact-count").textContent = total ? `${total} file(s)` : "";
  $("artifacts-empty").classList.toggle("hidden", total > 0);
}

/** Open the full-size lightbox for an artifact preview. */
function openLightbox(url, filename) {
  $("lightbox-img").src = url;
  $("lightbox-img").alt = filename || "Artifact preview";
  $("lightbox").classList.remove("hidden");
  $("lightbox").classList.add("flex");
}

function closeLightbox() {
  $("lightbox").classList.add("hidden");
  $("lightbox").classList.remove("flex");
  $("lightbox-img").src = "";
}

// ---------------------------------------------------------------------------
// Polling timers
// ---------------------------------------------------------------------------

function startJobsTimer() {
  stopJobsTimer();
  state.jobsTimer = window.setInterval(refreshJobs, POLL_MS.jobs);
}

function stopJobsTimer() {
  if (state.jobsTimer !== null) {
    window.clearInterval(state.jobsTimer);
    state.jobsTimer = null;
  }
}

function startLogsTimer() {
  stopLogsTimer();
  state.logsTimer = window.setInterval(logsTick, POLL_MS.logs);
}

function stopLogsTimer() {
  if (state.logsTimer !== null) {
    window.clearInterval(state.logsTimer);
    state.logsTimer = null;
  }
}

/** Status monitoring continues with Live off; terminal logs drain to the tail. */
async function logsTick() {
  const identity = watchIdentity();
  if (!identity.jobId || (state.tickFlight && watchCurrent(state.tickFlight))) return;
  state.tickFlight = identity;
  try {
    const detail = await refreshDetail();
    if (!detail || !watchCurrent(identity)) return;
    const terminal = !ACTIVE_STATUSES.has(detail.status);
    if (terminal || $("live-logs").checked) await pollLogs(terminal);
    if (!watchCurrent(identity)) return;
    if (terminal) {
      const artifactsReady = await loadArtifacts();
      if (watchCurrent(identity) && artifactsReady) stopLogsTimer();
    }
  } catch {
    if (watchCurrent(identity)) setConnState("offline"); // retain the timer for retry
  } finally {
    if (state.tickFlight === identity) state.tickFlight = null;
  }
}

// ---------------------------------------------------------------------------
// Event bindings + init
// ---------------------------------------------------------------------------

function bindEvents() {
  $("base-url-apply").addEventListener("click", applyBaseUrl);
  $("base-url").addEventListener("keydown", (event) => {
    if (event.key === "Enter") applyBaseUrl();
  });
  $("banner-dismiss").addEventListener("click", hideBanner);

  $("task-type").addEventListener("change", applyTaskPreset);
  $("dispatch-form").addEventListener("submit", onDispatchSubmit);

  $("jobs-refresh").addEventListener("click", () => refreshJobs());
  $("artifacts-refresh").addEventListener("click", () =>
    loadArtifacts().catch((error) => showBanner("error", error.message)));

  $("terminal-clear").addEventListener("click", () => {
    $("terminal-body").textContent = "";
  });

  $("live-jobs").addEventListener("change", () => {
    if ($("live-jobs").checked) startJobsTimer();
    else stopJobsTimer();
  });
  $("live-logs").addEventListener("change", () => {
    if (state.selectedJobId) startLogsTimer();
  });

  $("lightbox").addEventListener("click", closeLightbox);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeLightbox();
  });
}

function init() {
  $("base-url").value = state.baseUrl;
  bindEvents();
  applyTaskPreset();
  checkHealth();
  refreshJobs();
  startJobsTimer();
  window.setInterval(checkHealth, POLL_MS.health);
}

document.addEventListener("DOMContentLoaded", init);
