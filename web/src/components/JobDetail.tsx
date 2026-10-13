import { useEffect, useState, type ReactElement } from "react";
import { useJobs } from "../state/JobContext";
import { ACTIVE_STATUSES, formatCreatedAt, formatDuration } from "../types";
import { ArtifactInspector } from "./ArtifactInspector";
import { StatusBadge } from "./StatusBadge";
import { Terminal } from "./Terminal";

type DetailTab = "logs" | "artifacts";

function formatElapsed(startedAt: string, now: number): string {
  const started = Date.parse(startedAt);
  if (!Number.isFinite(started)) return "—";
  const totalSeconds = Math.max(0, Math.floor((now - started) / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [hours, minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":");
}

export function JobDetailPage({ jobId, onBack }: { jobId: string; onBack: () => void }) {
  const {
    detail,
    detailMissing,
    selectedJobId,
    cancellationPending,
    liveLogs,
    setToggle,
    selectJob,
    cancelJob,
  } = useJobs();
  const [tab, setTab] = useState<DetailTab>("logs");
  const [now, setNow] = useState(() => Date.now());

  // The page selection drives the watched job; cleanup stops polling when leaving the page.
  useEffect(() => {
    selectJob(jobId);
    return () => selectJob(null);
  }, [jobId, selectJob]);

  const cancelling = cancellationPending === jobId;
  const active = detail !== null && ACTIVE_STATUSES.has(detail.status);

  useEffect(() => {
    if (!active || !detail?.started_at) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [active, detail?.started_at]);

  let body: ReactElement;
  if (detailMissing) {
    body = (
      <div className="rounded-lg border border-slate-800 bg-slate-900 px-4 py-10 text-center">
        <p className="text-sm font-semibold text-slate-300">Job not found</p>
        <p className="mt-1 font-mono text-xs text-slate-500">{jobId}</p>
        <p className="mt-3 text-xs text-slate-500">The engine has no record of this job id.</p>
      </div>
    );
  } else if (!detail || selectedJobId !== jobId) {
    body = (
      <div className="rounded-lg border border-slate-800 bg-slate-900 px-4 py-10 text-center text-xs text-slate-500">
        Loading job <span className="font-mono text-slate-400">{jobId}</span>…
      </div>
    );
  } else {
    const duration =
      active && detail.started_at ? `Running · ${formatElapsed(detail.started_at, now)}` : formatDuration(detail.duration);
    const facts: Array<[string, string]> = [
      ["Task Type", detail.task_type.toUpperCase()],
      ["Duration", duration],
      ["Created At", formatCreatedAt(detail.created_at)],
      ["Artifacts", String(detail.artifact_count)],
      ["Started At", formatCreatedAt(detail.started_at)],
      ["Completed At", formatCreatedAt(detail.completed_at)],
      ["Created By", String(detail.metadata.created_by ?? "—")],
      ["Priority", String(detail.metadata.priority ?? "—")],
    ];
    const security = detail.error_code === "SEC_ERR_001";
    body = (
      <>
        <div className="rounded-lg border border-slate-800 bg-slate-900">
          <div className="border-b border-slate-800 px-4 py-3">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300">Overview</h2>
          </div>
          <div className="px-4 py-4 text-sm">
            <div className="flex flex-wrap items-center gap-3">
              <span className="font-mono text-sm font-bold text-cyan-300">{detail.job_id}</span>
              <StatusBadge status={detail.status} />
              {cancelling && (
                <span className="text-xs font-semibold text-amber-300">
                  Cancellation accepted; awaiting terminal cancellation state…
                </span>
              )}
              <button
                type="button"
                disabled={!active || cancellationPending !== null}
                onClick={() => {
                  if (window.confirm(`Request cooperative cancellation of job ${detail.job_id}?`)) {
                    void cancelJob(detail.job_id);
                  }
                }}
                className="ml-auto rounded-md border border-rose-500/40 bg-rose-500/10 px-2.5 py-1 text-[11px] font-semibold text-rose-300 hover:bg-rose-500/20 disabled:border-slate-800 disabled:text-slate-600"
              >
                {cancelling ? "Cancelling…" : cancellationPending ? "Cancel pending" : "Cancel job"}
              </button>
            </div>
            <dl className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {facts.map(([label, value]) => (
                <div key={label}>
                  <dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">{label}</dt>
                  <dd className="break-words font-mono text-xs text-slate-300">{value}</dd>
                </div>
              ))}
            </dl>
            {detail.error_message && (
              <div
                className={`mt-3 rounded-md border px-3 py-2 ${
                  security
                    ? "border-amber-300 bg-amber-500/15 ring-2 ring-amber-400/40"
                    : "border-rose-500/40 bg-rose-500/10"
                }`}
              >
                <div className={`text-xs font-black ${security ? "text-amber-200" : "text-rose-300"}`}>
                  {security ? "SECURITY ALERT" : "Error"}: {detail.error_code || "UNKNOWN"}
                </div>
                <div
                  className={`mt-0.5 break-words font-mono text-[11px] ${
                    security ? "text-amber-100" : "text-rose-200/80"
                  }`}
                >
                  {detail.error_message}
                </div>
              </div>
            )}
          </div>
        </div>

        <div>
          <div className="flex items-center gap-2 px-1 pb-2 pt-4">
            {(["logs", "artifacts"] as DetailTab[]).map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => setTab(item)}
                className={`rounded-md border px-3 py-1.5 text-xs font-bold uppercase tracking-wider transition-colors ${
                  tab === item
                    ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-300"
                    : "border-slate-800 text-slate-400 hover:bg-slate-800/60"
                }`}
                aria-pressed={tab === item}
              >
                {item === "logs" ? "Logs" : "Artifacts"}
              </button>
            ))}
            {tab === "logs" && (
              <label className="ml-auto flex cursor-pointer items-center gap-2 text-xs text-slate-400">
                <input
                  type="checkbox"
                  checked={liveLogs}
                  onChange={(event) => setToggle("liveLogs", event.target.checked)}
                  className="accent-cyan-500"
                />
                Tail logs while active
              </label>
            )}
          </div>
          {/* Both panels stay mounted so log lines and artifact state survive tab switches. */}
          <div className={tab === "logs" ? "block" : "hidden"}>
            <Terminal />
          </div>
          <div className={tab === "artifacts" ? "block" : "hidden"}>
            <ArtifactInspector />
          </div>
        </div>
      </>
    );
  }

  return (
    <div className="space-y-2">
      <div className="px-1">
        <button type="button" onClick={onBack} className="text-xs font-semibold text-cyan-400 hover:underline">
          ← Back to Jobs
        </button>
      </div>
      {body}
    </div>
  );
}
