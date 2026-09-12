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
};

/** Friendly prefixes for the error banners requested by the P2 console spec. */
const HTTP_FRIENDLY = { 400: "Bad request", 404: "Not found", 409: "Conflict", 422: "Validation failed" };

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  baseUrl: (localStorage.getItem(BASE_URL_STORAGE_KEY) || DEFAULT_BASE_URL).replace(/\/+$/, ""),
  jobs: [],
  selectedJobId: null,
  detail: null, // latest GET /api/v1/jobs/{id} payload
  logOffset: 0, // next log cursor passed as ?offset=
  wasTerminal: false, // selected job observed in a terminal state
  terminalPolls: 0, // consecutive terminal polls before log tailing stops
  frozenDurations: new Map(), // job_id -> seconds elapsed when it left the active states
  jobsTimer: null,
  logsTimer: null,
};

/** Shorthand for document.getElementById. */
const $ = (id) => document.getElementById(id);

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
  return JSON.stringify(body || {});
}

/**
 * Fetch a JSON endpoint on the engine, throwing {@link ApiError} on failure.
 *
 * @param {string} path - API path starting with "/".
 * @param {object} [options] - fetch options (method, body...).
 * @returns {Promise<object>} Parsed JSON response body.
 */
async function api(path, options = {}) {
  const url = state.baseUrl + path;
  let response;
  try {
    response = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
  } catch (error) {
    setConnState("offline");
    throw new ApiError(0, `Cannot reach engine at ${state.baseUrl} — ${error.message}`);
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
  setConnState("checking");
  try {
    await api("/health");
    setConnState("online");
  } catch {
    setConnState("offline");
  }
}

/**
 * Apply the base URL input: persist it, reset console state and re-probe.
 */
function applyBaseUrl() {
  state.baseUrl = $("base-url").value.trim().replace(/\/+$/, "") || DEFAULT_BASE_URL;
  $("base-url").value = state.baseUrl;
  localStorage.setItem(BASE_URL_STORAGE_KEY, state.baseUrl);

  // A new engine may not know the previous selection; reset derived state.
  state.selectedJobId = null;
  state.detail = null;
  state.logOffset = 0;
  state.wasTerminal = false;
  state.terminalPolls = 0;
  state.jobs = [];
  state.frozenDurations.clear();
  stopLogsTimer();
  $("terminal-body").textContent = "";
  $("terminal-job").textContent = "no job selected";
  $("terminal-status").className = "hidden";
  $("job-detail").classList.add("hidden");
  renderJobs();
  renderArtifacts({ artifacts: [], image_artifacts: [] });
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
  const params = { ...extraParams };
  if (["predict", "train", "val"].includes(taskType)) {
    params.model_path = modelPath;
    params.data_source = dataSource;
    params.conf = parseFloat($("conf").value);
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
    security_constraints: {
      allow_shell: false, // server forces these fail-closed anyway
      path_whitelisted: true,
      allowed_paths: parseList($("allowed-paths").value),
    },
  };

  submitBtn.disabled = true;
  submitBtn.textContent = "Dispatching…";
  try {
    const job = await api(JOBS_PATH, { method: "POST", body: JSON.stringify(payload) });
    feedback.textContent = `✅ Job ${job.job_id} submitted (201) — selected in console`;
    feedback.className = "text-xs font-semibold text-emerald-400";
    showBanner("success", `Job ${job.job_id} dispatched successfully — monitoring now.`);
    await refreshJobs();
    await selectJob(job.job_id);
  } catch (error) {
    feedback.textContent = `❌ ${error.message}`;
    feedback.className = "text-xs font-semibold text-rose-400";
    showBanner("error", error.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Dispatch Job";
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
 * Render the Duration cell: live ticking for PENDING/RUNNING rows, frozen at
 * the first observation of a terminal state (matches the backend's freeze
 * semantics in JobsManager.get_job_status).
 *
 * @param {{job_id: string, status: string, created_at: string}} job - Summary row.
 * @returns {string} Duration string like "12.3s", or "N/A".
 */
function jobDuration(job) {
  let seconds;
  if (ACTIVE_STATUSES.has(job.status)) {
    seconds = elapsedSeconds(job.created_at);
  } else {
    seconds = state.frozenDurations.get(job.job_id);
    if (seconds === undefined) {
      seconds = elapsedSeconds(job.created_at);
      if (seconds !== null) state.frozenDurations.set(job.job_id, seconds);
    }
  }
  return seconds === null ? "N/A" : `${seconds.toFixed(1)}s`;
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
    cancelBtn.textContent = "Cancel Job";
    cancelBtn.className = active
      ? "ml-1.5 rounded-md border border-rose-500/40 bg-rose-500/10 px-2 py-1 text-[11px] font-semibold text-rose-300 hover:bg-rose-500/20"
      : "ml-1.5 rounded-md border border-slate-800 px-2 py-1 text-[11px] font-semibold text-slate-600 cursor-not-allowed";
    cancelBtn.disabled = !active;
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
  try {
    const body = await api(`${JOBS_PATH}?limit=${JOBS_LIST_LIMIT}`);
    state.jobs = body.jobs;
    renderJobs();
    setConnState("online");
  } catch {
    setConnState("offline"); // silent: the pill is the live-feedback surface here
  }
}

/** POST /api/v1/jobs/{id}/cancel for an active job. */
async function cancelJob(jobId) {
  if (!window.confirm(`Request cooperative cancellation of job ${jobId}?`)) return;
  try {
    const body = await api(`/api/v1/jobs/${jobId}/cancel`, { method: "POST" });
    showBanner("success", body.message || `Cancellation requested for ${jobId} (202).`);
  } catch (error) {
    showBanner("error", error.message);
  } finally {
    refreshJobs();
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
  state.selectedJobId = jobId;
  $("terminal-job").textContent = jobId;
  if (!sameJob) {
    state.logOffset = 0;
    state.wasTerminal = false;
    state.terminalPolls = 0;
    $("terminal-body").textContent = "";
  }
  renderJobs();
  $("job-detail").classList.remove("hidden");

  try {
    const detail = await refreshDetail();
    await pollLogs(); // one-time snapshot on selection, independent of the tailing toggle
    await loadArtifacts();
    if (detail && !ACTIVE_STATUSES.has(detail.status)) stopLogsTimer();
    else if ($("live-logs").checked) startLogsTimer();
  } catch (error) {
    showBanner("error", error.message);
    stopLogsTimer();
  }
}

/** GET /api/v1/jobs/{id} and render the detail card. */
async function refreshDetail() {
  const jobId = state.selectedJobId;
  if (!jobId) return null;
  try {
    const detail = await api(`/api/v1/jobs/${jobId}`);
    state.detail = detail;
    renderDetail(detail);
    return detail;
  } catch {
    state.detail = null;
    return null;
  }
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
    ["Duration", detail.duration || "N/A"],
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
async function pollLogs() {
  const jobId = state.selectedJobId;
  if (!jobId) return null;
  const body = await api(`/api/v1/jobs/${jobId}/logs?offset=${state.logOffset}&limit=${LOG_WINDOW_LIMIT}`);
  appendLogLines(body.logs);
  // next_offset is the cursor for the following page; null means the tail was
  // reached, so the next poll resumes at the last known line count and picks
  // up any lines appended since.
  state.logOffset = body.next_offset !== null && body.next_offset !== undefined
    ? body.next_offset
    : body.offset + body.logs.length;
  return body;
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
  const jobId = state.selectedJobId;
  if (!jobId) return;
  try {
    const body = await api(`/api/v1/jobs/${jobId}/artifacts`);
    renderArtifacts(body);
  } catch {
    renderArtifacts({ artifacts: [], image_artifacts: [] });
  }
}

/**
 * Render the artifact inspector: an image preview gallery for entries marked
 * is_image plus the validated image_artifacts scan (served by basename
 * through the fail-closed /static/artifacts route), and view/download links
 * for non-image files.
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
  for (const path of body.image_artifacts || []) {
    const filename = path.split(/[\\/]/).pop();
    if (filename) {
      images.push({ filename, url: absUrl(`/static/artifacts/${state.selectedJobId}/${encodeURIComponent(filename)}`) });
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

  const total = images.length + (body.artifacts || []).filter((entry) => !entry.is_image).length;
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

/**
 * One log-tailing tick: pull the next cursor window, then refresh the job
 * detail. Terminal jobs get one extra tick (so the final log lines land)
 * before tailing stops, and the artifact manifest is re-fetched exactly once
 * on the transition into a terminal state.
 */
async function logsTick() {
  if (!$("live-logs").checked) {
    stopLogsTimer(); // toggle flipped off between ticks: cease tailing
    return;
  }
  try {
    await pollLogs();
    const detail = await refreshDetail();
    if (!detail) return;
    const terminal = !ACTIVE_STATUSES.has(detail.status);
    if (terminal) {
      if (!state.wasTerminal) loadArtifacts();
      state.terminalPolls += 1;
      if (state.terminalPolls >= 2) stopLogsTimer();
    } else {
      state.terminalPolls = 0;
    }
    state.wasTerminal = terminal;
  } catch {
    stopLogsTimer(); // engine unreachable / job vanished: stop silently
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
    if ($("live-logs").checked && state.selectedJobId) startLogsTimer();
    else stopLogsTimer();
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
