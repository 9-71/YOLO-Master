import { createContext, useCallback, useContext, useEffect, useMemo, useReducer, useRef, type ReactNode } from "react";
import { api } from "../api/client";
import type { ArtifactsResponse, BannerState, ConnectionState, JobDetail, JobRequest, JobsResponse, JobSummary, LogsResponse } from "../types";
import { ACTIVE_STATUSES, EMPTY_ARTIFACTS } from "../types";

const DEFAULT_BASE_URL = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const STORAGE_KEY = "f1.console.baseUrl";
const JOBS_PATH = "/api/v1/jobs";

interface State {
  epoch: number; baseUrl: string; jobs: JobSummary[]; selectedJobId: string | null; detail: JobDetail | null;
  logs: string[]; logOffset: number; artifacts: ArtifactsResponse; connection: ConnectionState;
  banner: BannerState | null; liveJobs: boolean; liveLogs: boolean; autoScroll: boolean; cancellationPending: string | null;
}

type Action =
  | { type: "base"; value: string } | { type: "jobs"; value: JobSummary[]; epoch: number }
  | { type: "select"; value: string; epoch: number } | { type: "detail"; value: JobDetail; epoch: number; jobId: string }
  | { type: "logs"; lines: string[]; offset: number; expectedOffset: number; epoch: number; jobId: string }
  | { type: "clearLogs" } | { type: "artifacts"; value: ArtifactsResponse; epoch: number; jobId: string }
  | { type: "connection"; value: ConnectionState; epoch: number } | { type: "banner"; value: BannerState | null; epoch?: number }
  | { type: "toggle"; key: "liveJobs" | "liveLogs" | "autoScroll"; value: boolean }
  | { type: "cancelStart"; jobId: string } | { type: "cancelClear"; jobId: string }
  | { type: "cancelOutcome"; value: JobDetail; jobId: string; epoch: number };

const initialState: State = {
  epoch: 0, baseUrl: (localStorage.getItem(STORAGE_KEY) || DEFAULT_BASE_URL).replace(/\/+$/, ""), jobs: [], selectedJobId: null,
  detail: null, logs: [], logOffset: 0, artifacts: EMPTY_ARTIFACTS, connection: "checking", banner: null,
  liveJobs: true, liveLogs: true, autoScroll: true, cancellationPending: null,
};

function isCurrent(state: State, action: { epoch: number; jobId?: string }) {
  return action.epoch === state.epoch && (action.jobId === undefined || action.jobId === state.selectedJobId);
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "base": return { ...initialState, epoch: state.epoch + 1, baseUrl: action.value, liveJobs: state.liveJobs, liveLogs: state.liveLogs, autoScroll: state.autoScroll };
    case "jobs": return isCurrent(state, action) ? { ...state, jobs: [...action.value].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)) } : state;
    case "select": return action.epoch === state.epoch && action.value !== state.selectedJobId ? { ...state, epoch: state.epoch + 1, selectedJobId: action.value, detail: null, logs: [], logOffset: 0, artifacts: EMPTY_ARTIFACTS } : state;
    case "detail": {
      if (!isCurrent(state, action)) return state;
      return { ...state, detail: action.value };
    }
    case "logs": return isCurrent(state, action) && state.logOffset === action.expectedOffset ? { ...state, logs: [...state.logs, ...action.lines].slice(-1000), logOffset: action.offset } : state;
    case "clearLogs": return { ...state, logs: [] };
    case "artifacts": return isCurrent(state, action) ? { ...state, artifacts: action.value } : state;
    case "connection": return isCurrent(state, action) ? { ...state, connection: action.value } : state;
    case "banner": return action.epoch === undefined || action.epoch === state.epoch ? { ...state, banner: action.value } : state;
    case "toggle": return { ...state, [action.key]: action.value };
    case "cancelStart": return state.cancellationPending ? state : { ...state, cancellationPending: action.jobId };
    case "cancelClear": return state.cancellationPending === action.jobId ? { ...state, cancellationPending: null } : state;
    case "cancelOutcome": {
      if (action.epoch !== state.epoch || state.cancellationPending !== action.jobId || ACTIVE_STATUSES.has(action.value.status)) return state;
      const cancelled = action.value.status === "cancelled" && action.value.error_code === "USER_CANCELLED";
      return {
        ...state,
        cancellationPending: null,
        banner: cancelled
          ? { kind: "success", message: `Job ${action.jobId} reached cancelled / USER_CANCELLED. Cancellation completed.` }
          : { kind: "error", message: `Cancellation race for ${action.jobId}: job reached ${action.value.status}${action.value.error_code ? ` / ${action.value.error_code}` : ""} before USER_CANCELLED was observed.` },
      };
    }
  }
}

interface JobContextValue extends State {
  applyBaseUrl(value: string): void; refreshJobs(showError?: boolean): Promise<JobSummary[]>; selectJob(jobId: string): void;
  submitJob(payload: JobRequest): Promise<void>; cancelJob(jobId: string): Promise<void>; refreshArtifacts(showError?: boolean): Promise<void>;
  clearLogs(): void; dismissBanner(): void; setToggle(key: "liveJobs" | "liveLogs" | "autoScroll", value: boolean): void;
}
const JobContext = createContext<JobContextValue | null>(null);

export function JobProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const logController = useRef<AbortController | null>(null);
  const cancellationInFlight = useRef<string | null>(null);
  const showError = useCallback((error: unknown, epoch?: number) => {
    const message = error instanceof Error ? error.message : String(error);
    dispatch({ type: "banner", epoch, value: { kind: message.includes("SEC_ERR_001") ? "security" : "error", message } });
  }, []);

  const refreshJobs = useCallback(async (loud = false) => {
    const epoch = state.epoch;
    try {
      const body = await api<JobsResponse>(state.baseUrl, `${JOBS_PATH}?limit=50`);
      dispatch({ type: "jobs", value: body.jobs, epoch }); dispatch({ type: "connection", value: "online", epoch }); return body.jobs;
    } catch (error) {
      dispatch({ type: "connection", value: "offline", epoch }); if (loud) showError(error, epoch); return [];
    }
  }, [showError, state.baseUrl, state.epoch]);

  const fetchDetail = useCallback(async (jobId: string, epoch: number) => {
    const detail = await api<JobDetail>(state.baseUrl, `${JOBS_PATH}/${encodeURIComponent(jobId)}`);
    dispatch({ type: "detail", value: detail, epoch, jobId });
    if (detail.error_code === "SEC_ERR_001") dispatch({ type: "banner", epoch, value: { kind: "security", message: `Security violation: ${detail.error_message || detail.error_code}` } });
    return detail;
  }, [state.baseUrl]);

  const fetchArtifacts = useCallback(async (jobId: string, epoch: number, loud = false) => {
    try {
      const body = await api<ArtifactsResponse>(state.baseUrl, `${JOBS_PATH}/${encodeURIComponent(jobId)}/artifacts`);
      dispatch({ type: "artifacts", value: body, epoch, jobId });
    } catch (error) {
      dispatch({ type: "artifacts", value: { ...EMPTY_ARTIFACTS, job_id: jobId }, epoch, jobId }); if (loud) showError(error, epoch);
    }
  }, [showError, state.baseUrl]);

  const pullLogs = useCallback(async (jobId: string, epoch: number, startOffset: number, drain: boolean) => {
    if (!drain && logController.current) return;
    if (drain) logController.current?.abort();
    const controller = new AbortController(); logController.current = controller;
    const lines: string[] = []; let cursor = startOffset;
    try {
      do {
        const body = await api<LogsResponse>(state.baseUrl, `${JOBS_PATH}/${encodeURIComponent(jobId)}/logs?offset=${cursor}&limit=500`, { signal: controller.signal });
        lines.push(...body.logs); cursor = body.next_offset ?? body.offset + body.logs.length;
        if (body.next_offset === null) break;
      } while (drain);
      dispatch({ type: "logs", lines, offset: cursor, expectedOffset: startOffset, epoch, jobId });
    } catch (error) {
      if (!controller.signal.aborted) throw error;
    } finally { if (logController.current === controller) logController.current = null; }
  }, [state.baseUrl]);

  useEffect(() => {
    const epoch = state.epoch; dispatch({ type: "connection", value: "checking", epoch });
    const check = async () => { try { await api(state.baseUrl, "/health"); dispatch({ type: "connection", value: "online", epoch }); } catch { dispatch({ type: "connection", value: "offline", epoch }); } };
    void check(); const timer = window.setInterval(check, 5000); return () => window.clearInterval(timer);
  }, [state.baseUrl, state.epoch]);

  useEffect(() => { void refreshJobs(); }, [refreshJobs]);
  useEffect(() => {
    if (!state.liveJobs) return;
    if (state.jobs.length > 0 && !state.jobs.some((job) => ACTIVE_STATUSES.has(job.status)) && !state.cancellationPending) return;
    const timer = window.setInterval(() => void refreshJobs(), 2000); return () => window.clearInterval(timer);
  }, [refreshJobs, state.cancellationPending, state.jobs, state.liveJobs]);

  // Cancellation observation follows the accepted job even when the user selects a different row.
  // It uses a dedicated action so its response can never replace the selected job's detail card.
  useEffect(() => {
    if (!state.cancellationPending) return;
    const jobId = state.cancellationPending;
    const epoch = state.epoch;
    const observe = async () => {
      try {
        const detail = await api<JobDetail>(state.baseUrl, `${JOBS_PATH}/${encodeURIComponent(jobId)}`);
        dispatch({ type: "cancelOutcome", value: detail, jobId, epoch });
      } catch (error) { showError(error, epoch); }
    };
    void observe();
    const timer = window.setInterval(observe, 1000);
    return () => window.clearInterval(timer);
  }, [showError, state.baseUrl, state.cancellationPending, state.epoch]);

  useEffect(() => {
    if (!state.selectedJobId) return;
    const { selectedJobId: jobId, epoch, logOffset } = state;
    void fetchDetail(jobId, epoch).catch((error) => showError(error, epoch));
    void pullLogs(jobId, epoch, logOffset, false).catch((error) => showError(error, epoch));
    void fetchArtifacts(jobId, epoch);
    return () => { logController.current?.abort(); logController.current = null; };
    // Polling is handled below; this effect is a selection/base identity snapshot.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.epoch, state.selectedJobId]);

  // Status monitoring remains active when Tail Logs is disabled, including after cancel returns 202.
  useEffect(() => {
    if (!state.selectedJobId || !state.detail || !ACTIVE_STATUSES.has(state.detail.status)) return;
    const jobId = state.selectedJobId;
    const epoch = state.epoch;
    const timer = window.setInterval(() => void fetchDetail(jobId, epoch).catch(() => undefined), 1000);
    return () => window.clearInterval(timer);
  }, [fetchDetail, state.detail, state.epoch, state.selectedJobId]);

  useEffect(() => {
    if (!state.liveLogs || !state.selectedJobId || !state.detail || !ACTIVE_STATUSES.has(state.detail.status)) return;
    const jobId = state.selectedJobId;
    const epoch = state.epoch;
    const logOffset = state.logOffset;
    const timer = window.setInterval(() => void pullLogs(jobId, epoch, logOffset, false).catch(() => undefined), 1000);
    return () => window.clearInterval(timer);
  }, [pullLogs, state.detail, state.epoch, state.liveLogs, state.logOffset, state.selectedJobId]);

  // A terminal transition drains all remaining pages through next_offset=null, even if live tailing is off.
  useEffect(() => {
    if (!state.selectedJobId || !state.detail || ACTIVE_STATUSES.has(state.detail.status)) return;
    const { selectedJobId: jobId, epoch, logOffset } = state;
    void pullLogs(jobId, epoch, logOffset, true).catch((error) => showError(error, epoch)); void fetchArtifacts(jobId, epoch);
    // logOffset changes after draining and must not retrigger the terminal drain.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.detail?.status, state.epoch, state.selectedJobId]);

  useEffect(() => { if (!state.banner) return; const timer = window.setTimeout(() => dispatch({ type: "banner", value: null }), 8000); return () => window.clearTimeout(timer); }, [state.banner]);
  useEffect(() => { cancellationInFlight.current = state.cancellationPending; }, [state.cancellationPending]);

  const value = useMemo<JobContextValue>(() => ({
    ...state,
    applyBaseUrl(value) { logController.current?.abort(); cancellationInFlight.current = null; const normalized = (value.trim() || DEFAULT_BASE_URL).replace(/\/+$/, ""); localStorage.setItem(STORAGE_KEY, normalized); dispatch({ type: "base", value: normalized }); },
    refreshJobs,
    selectJob(jobId) { logController.current?.abort(); dispatch({ type: "select", value: jobId, epoch: state.epoch }); },
    async submitJob(payload) {
      const epoch = state.epoch;
      try {
        const result = await api<JobRequest>(state.baseUrl, JOBS_PATH, { method: "POST", body: JSON.stringify(payload) });
        dispatch({ type: "banner", epoch, value: { kind: "success", message: `Job ${result.job_id} dispatched successfully — monitoring now.` } });
        dispatch({ type: "select", value: result.job_id, epoch }); await refreshJobs();
      } catch (error) { showError(error, epoch); throw error; }
    },
    async cancelJob(jobId) {
      const epoch = state.epoch;
      const pendingJobId = cancellationInFlight.current || state.cancellationPending;
      if (pendingJobId) {
        dispatch({ type: "banner", value: { kind: "error", message: `Wait for cancellation of ${pendingJobId} to reach a terminal state before cancelling another job.` } });
        return;
      }
      cancellationInFlight.current = jobId;
      dispatch({ type: "cancelStart", jobId });
      try {
        const result = await api<{ message: string }>(state.baseUrl, `${JOBS_PATH}/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
        dispatch({ type: "banner", value: { kind: "success", message: `${result.message || "Cancellation accepted"} Waiting for cancelled / USER_CANCELLED terminal state…` } });
        dispatch({ type: "select", value: jobId, epoch }); await refreshJobs();
      } catch (error) {
        cancellationInFlight.current = null;
        dispatch({ type: "cancelClear", jobId });
        showError(error);
      }
    },
    async refreshArtifacts(loud = false) { if (state.selectedJobId) await fetchArtifacts(state.selectedJobId, state.epoch, loud); },
    clearLogs() { dispatch({ type: "clearLogs" }); }, dismissBanner() { dispatch({ type: "banner", value: null }); },
    setToggle(key, value) { dispatch({ type: "toggle", key, value }); },
  }), [fetchArtifacts, refreshJobs, showError, state]);

  return <JobContext.Provider value={value}>{children}</JobContext.Provider>;
}

// The provider and its hook intentionally share this module as one public context API.
// eslint-disable-next-line react-refresh/only-export-components
export function useJobs(): JobContextValue {
  const value = useContext(JobContext); if (!value) throw new Error("useJobs must be used inside JobProvider"); return value;
}
