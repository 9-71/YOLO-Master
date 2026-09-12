import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from "react";
import { api, ApiError } from "../api/client";
import type {
  ArtifactsResponse,
  BannerState,
  CancelAck,
  ConnectionState,
  JobDetail,
  JobRequest,
  JobsResponse,
  JobSummary,
  LogsResponse,
} from "../types";
import { ACTIVE_STATUSES, EMPTY_ARTIFACTS } from "../types";

const DEFAULT_BASE_URL = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const STORAGE_KEY = "f1.console.baseUrl";
const JOBS_PATH = "/api/v1/jobs";

/** Adaptive list cadence: fast while anything needs watching, slow otherwise. */
const JOBS_FAST_INTERVAL_MS = 2000;
const JOBS_SLOW_INTERVAL_MS = 30000;
const HEALTH_INTERVAL_MS = 5000;
const DETAIL_INTERVAL_MS = 1000;

interface State {
  baseRev: number;
  baseUrl: string;
  connection: ConnectionState;
  banner: BannerState | null;
  jobs: JobSummary[];
  selectedJobId: string | null;
  watchSeq: number;
  detail: JobDetail | null;
  detailMissing: boolean;
  logs: string[];
  logOffset: number;
  artifacts: ArtifactsResponse;
  liveJobs: boolean;
  liveLogs: boolean;
  autoScroll: boolean;
  cancellationPending: string | null;
  cancelOpId: number | null;
}

type Action =
  | { type: "base"; value: string }
  | { type: "jobs"; value: JobSummary[]; rev: number }
  | { type: "connection"; value: ConnectionState; rev: number }
  | { type: "banner"; value: BannerState | null; rev?: number }
  | { type: "select"; jobId: string | null }
  | { type: "detail"; value: JobDetail; rev: number; watchSeq: number; jobId: string }
  | { type: "detailMissing"; rev: number; watchSeq: number; jobId: string }
  | { type: "logs"; lines: string[]; offset: number; expectedOffset: number; rev: number; watchSeq: number; jobId: string }
  | { type: "artifacts"; value: ArtifactsResponse; rev: number; watchSeq: number; jobId: string }
  | { type: "toggle"; key: "liveJobs" | "liveLogs" | "autoScroll"; value: boolean }
  | { type: "clearLogs" }
  | { type: "cancelStart"; jobId: string; opId: number }
  | { type: "cancelOutcome"; value: JobDetail; jobId: string; opId: number; rev: number };

const initialState: State = {
  baseRev: 0,
  baseUrl: (localStorage.getItem(STORAGE_KEY) || DEFAULT_BASE_URL).replace(/\/+$/, ""),
  connection: "checking",
  banner: null,
  jobs: [],
  selectedJobId: null,
  watchSeq: 0,
  detail: null,
  detailMissing: false,
  logs: [],
  logOffset: 0,
  artifacts: EMPTY_ARTIFACTS,
  liveJobs: true,
  liveLogs: true,
  autoScroll: true,
  cancellationPending: null,
  cancelOpId: null,
};

function watchCurrent(state: State, guard: { rev: number; watchSeq: number; jobId?: string }) {
  return (
    guard.rev === state.baseRev &&
    guard.watchSeq === state.watchSeq &&
    (guard.jobId === undefined || guard.jobId === state.selectedJobId)
  );
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "base":
      return {
        ...initialState,
        baseRev: state.baseRev + 1,
        baseUrl: action.value,
        liveJobs: state.liveJobs,
        liveLogs: state.liveLogs,
        autoScroll: state.autoScroll,
      };
    case "jobs":
      if (action.rev !== state.baseRev) return state;
      return {
        ...state,
        jobs: [...action.value].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)),
      };
    case "connection":
      return action.rev === state.baseRev ? { ...state, connection: action.value } : state;
    case "banner":
      return action.rev === undefined || action.rev === state.baseRev ? { ...state, banner: action.value } : state;
    case "select":
      if (action.jobId === state.selectedJobId) return state;
      return {
        ...state,
        watchSeq: state.watchSeq + 1,
        selectedJobId: action.jobId,
        detail: null,
        detailMissing: false,
        logs: [],
        logOffset: 0,
        artifacts: EMPTY_ARTIFACTS,
      };
    case "detail":
      return watchCurrent(state, action) ? { ...state, detail: action.value, detailMissing: false } : state;
    case "detailMissing":
      return watchCurrent(state, action) ? { ...state, detail: null, detailMissing: true } : state;
    case "logs":
      return watchCurrent(state, action) && state.logOffset === action.expectedOffset
        ? { ...state, logs: [...state.logs, ...action.lines].slice(-1000), logOffset: action.offset }
        : state;
    case "artifacts":
      return watchCurrent(state, action) ? { ...state, artifacts: action.value } : state;
    case "toggle":
      return { ...state, [action.key]: action.value };
    case "clearLogs":
      return { ...state, logs: [] };
    case "cancelStart":
      return state.cancellationPending
        ? state
        : { ...state, cancellationPending: action.jobId, cancelOpId: action.opId };
    case "cancelOutcome": {
      if (
        action.rev !== state.baseRev ||
        state.cancelOpId !== action.opId ||
        state.cancellationPending !== action.jobId ||
        ACTIVE_STATUSES.has(action.value.status)
      ) {
        return state;
      }
      const cancelled = action.value.status === "cancelled" && action.value.error_code === "USER_CANCELLED";
      return {
        ...state,
        cancellationPending: null,
        cancelOpId: null,
        banner: cancelled
          ? {
              kind: "success",
              message: `Job ${action.jobId} reached cancelled / USER_CANCELLED. Cancellation completed.`,
            }
          : {
              kind: "error",
              message: `Cancellation race for ${action.jobId}: job reached ${action.value.status}${
                action.value.error_code ? ` / ${action.value.error_code}` : ""
              } before USER_CANCELLED was observed.`,
            },
      };
    }
  }
}

interface JobContextValue extends State {
  applyBaseUrl(value: string): void;
  refreshJobs(showError?: boolean): Promise<JobSummary[]>;
  selectJob(jobId: string | null): void;
  submitJob(payload: JobRequest): Promise<string>;
  cancelJob(jobId: string): Promise<void>;
  refreshArtifacts(showError?: boolean): Promise<void>;
  clearLogs(): void;
  dismissBanner(): void;
  setToggle(key: "liveJobs" | "liveLogs" | "autoScroll", value: boolean): void;
}

const JobContext = createContext<JobContextValue | null>(null);

export function JobProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const stateRef = useRef(state);
  stateRef.current = state;

  const logController = useRef<AbortController | null>(null);
  const jobsInFlight = useRef<{ seq: number; rev: number; controller: AbortController } | null>(null);
  const jobsSeq = useRef(0);
  const cancelOpSeq = useRef(0);
  const cancelLock = useRef<{ rev: number; jobId: string; opId: number } | null>(null);

  const releaseCancelLock = useCallback((rev: number, jobId: string, opId: number) => {
    const current = cancelLock.current;
    if (current?.rev === rev && current.jobId === jobId && current.opId === opId) {
      cancelLock.current = null;
    }
  }, []);

  const showError = useCallback((error: unknown, rev?: number) => {
    const message = error instanceof Error ? error.message : String(error);
    dispatch({
      type: "banner",
      rev,
      value: { kind: message.includes("SEC_ERR_001") ? "security" : "error", message },
    });
  }, []);

  // ---- Jobs list: single adaptive sync loop with single-flight + stale guards ----
  const syncJobs = useCallback(
    async (loud = false): Promise<JobSummary[]> => {
      const rev = stateRef.current.baseRev;
      const baseUrl = stateRef.current.baseUrl;

      // Single-flight within an epoch; a newer epoch supersedes (aborts) any
      // in-flight request so a fresh refresh is never blocked across epochs.
      const prev = jobsInFlight.current;
      if (prev && prev.rev === rev) return [];
      prev?.controller.abort();

      const seq = ++jobsSeq.current;
      const controller = new AbortController();
      jobsInFlight.current = { seq, rev, controller };
      try {
        const body = await api<JobsResponse>(baseUrl, `${JOBS_PATH}?limit=50`, {
          signal: controller.signal,
        });
        // Stale response: a newer request or an engine switch superseded this one.
        if (rev !== stateRef.current.baseRev || seq !== jobsSeq.current) return body.jobs;
        dispatch({ type: "jobs", value: body.jobs, rev });
        dispatch({ type: "connection", value: "online", rev });
        return body.jobs;
      } catch (error) {
        // Superseded by a newer request/epoch: silence the abort.
        if (controller.signal.aborted) return [];
        if (rev === stateRef.current.baseRev && seq === jobsSeq.current) {
          dispatch({ type: "connection", value: "offline", rev });
          if (loud) showError(error, rev);
        }
        return [];
      } finally {
        if (jobsInFlight.current?.seq === seq) jobsInFlight.current = null;
      }
    },
    [showError],
  );

  const refreshJobs = syncJobs;

  // Stable identity: route-driven pages call this in effects/cleanups.
  const selectJob = useCallback((jobId: string | null) => {
    logController.current?.abort();
    dispatch({ type: "select", jobId });
  }, []);

  const fetchDetail = useCallback(
    async (jobId: string, rev: number, watchSeq: number): Promise<JobDetail> => {
      const detail = await api<JobDetail>(stateRef.current.baseUrl, `${JOBS_PATH}/${encodeURIComponent(jobId)}`);
      dispatch({ type: "detail", value: detail, rev, watchSeq, jobId });
      if (detail.error_code === "SEC_ERR_001") {
        dispatch({
          type: "banner",
          rev,
          value: { kind: "security", message: `Security violation: ${detail.error_message || detail.error_code}` },
        });
      }
      return detail;
    },
    [],
  );

  const fetchArtifacts = useCallback(
    async (jobId: string, rev: number, watchSeq: number, loud = false) => {
      try {
        const body = await api<ArtifactsResponse>(
          stateRef.current.baseUrl,
          `${JOBS_PATH}/${encodeURIComponent(jobId)}/artifacts`,
        );
        dispatch({ type: "artifacts", value: body, rev, watchSeq, jobId });
      } catch (error) {
        dispatch({
          type: "artifacts",
          value: { ...EMPTY_ARTIFACTS, job_id: jobId },
          rev,
          watchSeq,
          jobId,
        });
        if (loud) showError(error, rev);
      }
    },
    [showError],
  );

  const pullLogs = useCallback(
    async (jobId: string, rev: number, watchSeq: number, startOffset: number, drain: boolean) => {
      if (!drain && logController.current) return;
      if (drain) logController.current?.abort();
      const controller = new AbortController();
      logController.current = controller;
      const lines: string[] = [];
      let cursor = startOffset;
      try {
        do {
          const body = await api<LogsResponse>(
            stateRef.current.baseUrl,
            `${JOBS_PATH}/${encodeURIComponent(jobId)}/logs?offset=${cursor}&limit=500`,
            { signal: controller.signal },
          );
          lines.push(...body.logs);
          cursor = body.next_offset ?? body.offset + body.logs.length;
          if (body.next_offset === null) break;
        } while (drain);
        dispatch({ type: "logs", lines, offset: cursor, expectedOffset: startOffset, rev, watchSeq, jobId });
      } catch (error) {
        if (!controller.signal.aborted) throw error;
      } finally {
        if (logController.current === controller) logController.current = null;
      }
    },
    [],
  );

  // ---- Engine health probe ----
  useEffect(() => {
    const rev = state.baseRev;
    dispatch({ type: "connection", value: "checking", rev });
    const check = async () => {
      try {
        await api(state.baseUrl, "/health");
        if (rev === stateRef.current.baseRev) dispatch({ type: "connection", value: "online", rev });
      } catch {
        if (rev === stateRef.current.baseRev) dispatch({ type: "connection", value: "offline", rev });
      }
    };
    void check();
    const timer = window.setInterval(check, HEALTH_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [state.baseRev, state.baseUrl]);

  // ---- Adaptive list synchronization ----
  // 2s while any job is active or a cancellation is settling locally; 30s
  // otherwise. The loop re-arms only after the previous request settles
  // (single-flight), and every response is validated against its epoch/seq.
  const hasActiveJobs = state.jobs.some((job) => ACTIVE_STATUSES.has(job.status));
  const syncFast = hasActiveJobs || state.cancellationPending !== null;
  useEffect(() => {
    // liveJobs=false disables the adaptive auto-sync loop entirely; manual
    // refresh and the detail/log lifecycles are unaffected.
    if (!state.liveJobs) return;
    let timer = 0;
    let alive = true;
    const intervalMs = syncFast ? JOBS_FAST_INTERVAL_MS : JOBS_SLOW_INTERVAL_MS;
    const loop = async () => {
      await syncJobs(false);
      if (alive) timer = window.setTimeout(loop, intervalMs);
    };
    void loop();
    // StrictMode/unmount cleanup: do not chain a new timer; stale responses are
    // additionally dropped by the seq guard inside syncJobs.
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [syncJobs, syncFast, state.liveJobs, state.baseRev]);

  // ---- Cancellation observation (follows the job regardless of selection) ----
  useEffect(() => {
    const jobId = state.cancellationPending;
    const opId = state.cancelOpId;
    if (!jobId || opId === null) return;
    const rev = state.baseRev;
    let timer = 0;
    let alive = true;
    const observe = async () => {
      try {
        const detail = await api<JobDetail>(
          stateRef.current.baseUrl,
          `${JOBS_PATH}/${encodeURIComponent(jobId)}`,
        );
        if (alive && rev === stateRef.current.baseRev) {
          dispatch({ type: "cancelOutcome", value: detail, jobId, opId, rev });
          if (!ACTIVE_STATUSES.has(detail.status)) releaseCancelLock(rev, jobId, opId);
        }
      } catch (error) {
        if (alive && rev === stateRef.current.baseRev) showError(error, rev);
      }
    };
    void observe();
    timer = window.setInterval(observe, 1000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [state.cancellationPending, state.cancelOpId, state.baseRev, releaseCancelLock, showError]);

  // ---- Selected-job identity snapshot: detail + first log window + artifacts ----
  useEffect(() => {
    const jobId = state.selectedJobId;
    if (!jobId) return;
    const { baseRev: rev, watchSeq, logOffset } = state;
    void fetchDetail(jobId, rev, watchSeq).catch((error) => {
      if (error instanceof ApiError && error.status === 404) {
        dispatch({ type: "detailMissing", rev, watchSeq, jobId });
      } else {
        showError(error, rev);
      }
    });
    void pullLogs(jobId, rev, watchSeq, logOffset, false).catch((error) => showError(error, rev));
    void fetchArtifacts(jobId, rev, watchSeq);
    return () => {
      logController.current?.abort();
      logController.current = null;
    };
    // Polling lives in the effects below; this snapshot binds identity only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.baseRev, state.watchSeq, state.selectedJobId]);

  // ---- Detail status monitoring while active (also after a 202 cancel) ----
  useEffect(() => {
    if (!state.selectedJobId || !state.detail || !ACTIVE_STATUSES.has(state.detail.status)) return;
    const jobId = state.selectedJobId;
    const rev = state.baseRev;
    const watchSeq = state.watchSeq;
    const timer = window.setInterval(
      () => void fetchDetail(jobId, rev, watchSeq).catch(() => undefined),
      DETAIL_INTERVAL_MS,
    );
    return () => window.clearInterval(timer);
  }, [fetchDetail, state.detail, state.baseRev, state.watchSeq, state.selectedJobId]);

  // ---- Live log tailing while active ----
  useEffect(() => {
    if (!state.liveLogs || !state.selectedJobId || !state.detail || !ACTIVE_STATUSES.has(state.detail.status)) {
      return;
    }
    const jobId = state.selectedJobId;
    const rev = state.baseRev;
    const watchSeq = state.watchSeq;
    const logOffset = state.logOffset;
    const timer = window.setInterval(
      () => void pullLogs(jobId, rev, watchSeq, logOffset, false).catch(() => undefined),
      DETAIL_INTERVAL_MS,
    );
    return () => window.clearInterval(timer);
  }, [pullLogs, state.detail, state.baseRev, state.watchSeq, state.liveLogs, state.logOffset, state.selectedJobId]);

  // ---- Terminal transition: drain every remaining log page + refresh artifacts ----
  useEffect(() => {
    if (!state.selectedJobId || !state.detail || ACTIVE_STATUSES.has(state.detail.status)) return;
    const { selectedJobId: jobId, baseRev: rev, watchSeq, logOffset } = state;
    void pullLogs(jobId, rev, watchSeq, logOffset, true).catch((error) => showError(error, rev));
    void fetchArtifacts(jobId, rev, watchSeq);
    // logOffset changes after draining and must not retrigger the terminal drain.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.detail?.status, state.baseRev, state.watchSeq, state.selectedJobId]);

  // Banner auto-dismiss.
  useEffect(() => {
    if (!state.banner) return;
    const timer = window.setTimeout(() => dispatch({ type: "banner", value: null }), 8000);
    return () => window.clearTimeout(timer);
  }, [state.banner]);

  const value = useMemo<JobContextValue>(
    () => ({
      ...state,
      applyBaseUrl(value) {
        logController.current?.abort();
        const lock = cancelLock.current;
        if (lock) releaseCancelLock(lock.rev, lock.jobId, lock.opId);
        const normalized = (value.trim() || DEFAULT_BASE_URL).replace(/\/+$/, "");
        localStorage.setItem(STORAGE_KEY, normalized);
        dispatch({ type: "base", value: normalized });
      },
      refreshJobs: (loud = false) => refreshJobs(loud),
      selectJob,
      async submitJob(payload) {
        const rev = stateRef.current.baseRev;
        try {
          const result = await api<JobRequest>(stateRef.current.baseUrl, JOBS_PATH, {
            method: "POST",
            body: JSON.stringify(payload),
          });
          dispatch({
            type: "banner",
            rev,
            value: { kind: "success", message: `Job ${result.job_id} dispatched successfully — monitoring now.` },
          });
          await syncJobs(false);
          return result.job_id;
        } catch (error) {
          showError(error, rev);
          throw error;
        }
      },
      async cancelJob(jobId) {
        const rev = stateRef.current.baseRev;

        // Terminal guard first: never POST cancel against an already-terminal job.
        const known: JobSummary | JobDetail | null =
          stateRef.current.jobs.find((job) => job.job_id === jobId) ??
          (stateRef.current.selectedJobId === jobId ? stateRef.current.detail : null);
        if (known && !ACTIVE_STATUSES.has(known.status)) {
          dispatch({
            type: "banner",
            value: { kind: "error", message: `Job ${jobId} is already in terminal state '${known.status}'.` },
          });
          return;
        }

        // The synchronous ref closes the same-tick gap; cancellationPending
        // keeps serialization enforced until the accepted operation settles.
        if (cancelLock.current || stateRef.current.cancellationPending) {
          const pendingJobId = cancelLock.current?.jobId ?? stateRef.current.cancellationPending;
          dispatch({
            type: "banner",
            value: {
              kind: "error",
              message: `Wait for cancellation of ${pendingJobId} to reach a terminal state before cancelling another job.`,
            },
          });
          return;
        }

        const opId = ++cancelOpSeq.current;
        const lock = { rev, jobId, opId };
        cancelLock.current = lock;

        let ack: CancelAck;
        try {
          ack = await api<CancelAck>(
            stateRef.current.baseUrl,
            `${JOBS_PATH}/${encodeURIComponent(jobId)}/cancel`,
            { method: "POST" },
          );
        } catch (error) {
          releaseCancelLock(rev, jobId, opId);
          if (rev === stateRef.current.baseRev) showError(error, rev);
          return;
        }

        // Identity check before any side effect: still the same engine epoch and
        // the same operation (the observer may have settled, or the epoch flipped).
        if (cancelLock.current !== lock || rev !== stateRef.current.baseRev) return;

        const accepted = ack.status === "cancel_requested";

        if (!accepted) {
          // 200 idempotent replay: the job was already terminal. Settle directly
          // (no observer) and refresh the watched detail so its terminal drain runs.
          dispatch({
            type: "banner",
            rev,
            value: {
              kind: "success",
              message: `${ack.message || "Cancellation acknowledged"} (terminal state: ${ack.status})`,
            },
          });
          try {
            const detail = await api<JobDetail>(
              stateRef.current.baseUrl,
              `${JOBS_PATH}/${encodeURIComponent(jobId)}`,
            );
            if (cancelLock.current !== lock || rev !== stateRef.current.baseRev) return;
            const current = stateRef.current;
            if (current.selectedJobId === jobId) {
              dispatch({ type: "detail", value: detail, rev, watchSeq: current.watchSeq, jobId });
            }
          } catch (error) {
            if (rev === stateRef.current.baseRev) showError(error, rev);
          } finally {
            releaseCancelLock(rev, jobId, opId);
          }
          await syncJobs(false);
          return;
        }

        // 202 accepted: only now start the observer (cancellationPending gates it).
        dispatch({ type: "cancelStart", jobId, opId });
        dispatch({
          type: "banner",
          rev,
          value: {
            kind: "success",
            message: `${ack.message || "Cancellation accepted"} Waiting for cancelled / USER_CANCELLED terminal state…`,
          },
        });

        await syncJobs(false);

        // Post-POST compensating drain: use the latest log offset, and never
        // write to a selection that has since switched away from this job.
        const current = stateRef.current;
        if (
          current.baseRev === rev &&
          current.selectedJobId === jobId &&
          current.detail &&
          !ACTIVE_STATUSES.has(current.detail.status)
        ) {
          void pullLogs(jobId, rev, current.watchSeq, current.logOffset, true).catch(() => undefined);
        }
      },
      async refreshArtifacts(loud = false) {
        const { selectedJobId, baseRev, watchSeq } = stateRef.current;
        if (selectedJobId) await fetchArtifacts(selectedJobId, baseRev, watchSeq, loud);
      },
      clearLogs() {
        dispatch({ type: "clearLogs" });
      },
      dismissBanner() {
        dispatch({ type: "banner", value: null });
      },
      setToggle(key, value) {
        dispatch({ type: "toggle", key, value });
      },
    }),
    [fetchArtifacts, pullLogs, refreshJobs, releaseCancelLock, selectJob, showError, state, syncJobs],
  );

  return <JobContext.Provider value={value}>{children}</JobContext.Provider>;
}

// The provider and its hook intentionally share this module as one public context API.
// eslint-disable-next-line react-refresh/only-export-components
export function useJobs(): JobContextValue {
  const value = useContext(JobContext);
  if (!value) throw new Error("useJobs must be used inside JobProvider");
  return value;
}
