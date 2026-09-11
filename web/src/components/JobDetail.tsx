import { useJobs } from "../state/JobContext";
import { formatDuration } from "../types";
import { StatusBadge } from "./StatusBadge";

export function JobDetail() {
  const { detail, cancellationPending } = useJobs(); if (!detail) return null;
  const facts = [
    ["Task Type", detail.task_type.toUpperCase()], ["Duration", formatDuration(detail.duration)],
    ["Created At", detail.created_at ? new Date(detail.created_at).toLocaleString() : "—"], ["Artifacts", String(detail.artifact_count)],
    ["Created By", String(detail.metadata.created_by ?? "—")], ["Priority", String(detail.metadata.priority ?? "—")],
  ];
  const security = detail.error_code === "SEC_ERR_001";
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900">
      <div className="border-b border-slate-800 px-4 py-3"><h2 className="text-sm font-bold uppercase tracking-wider text-slate-300">Job Details</h2></div>
      <div className="px-4 py-4 text-sm"><div className="flex flex-wrap items-center gap-3"><span className="font-mono text-sm font-bold text-cyan-300">{detail.job_id}</span><StatusBadge status={detail.status} />{cancellationPending === detail.job_id && <span className="text-xs font-semibold text-amber-300">Cancellation accepted; awaiting USER_CANCELLED…</span>}</div>
        <dl className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">{facts.map(([label, value]) => <div key={label}><dt className="text-[10px] font-bold uppercase tracking-wider text-slate-500">{label}</dt><dd className="font-mono text-xs text-slate-300">{value}</dd></div>)}</dl>
        {detail.error_message && <div className={`mt-3 rounded-md border px-3 py-2 ${security ? "border-amber-300 bg-amber-500/15 ring-2 ring-amber-400/40" : "border-rose-500/40 bg-rose-500/10"}`}><div className={`text-xs font-black ${security ? "text-amber-200" : "text-rose-300"}`}>{security ? "SECURITY ALERT" : "Error"}: {detail.error_code || "UNKNOWN"}</div><div className={`mt-0.5 break-words font-mono text-[11px] ${security ? "text-amber-100" : "text-rose-200/80"}`}>{detail.error_message}</div></div>}
      </div>
    </div>
  );
}
