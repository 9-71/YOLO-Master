import type { JobStatus } from "../types";

const STYLES: Record<JobStatus, string> = {
  pending: "border-amber-500/40 bg-amber-500/15 text-amber-300",
  running: "border-sky-500/40 bg-sky-500/15 text-sky-300",
  completed: "border-emerald-500/40 bg-emerald-500/15 text-emerald-300",
  failed: "border-rose-500/40 bg-rose-500/15 text-rose-300",
  cancelled: "border-slate-500/40 bg-slate-500/15 text-slate-300",
};

export function StatusBadge({ status, small = false }: { status: JobStatus; small?: boolean }) {
  return <span className={`inline-block rounded-full border px-2.5 py-0.5 font-bold uppercase ${small ? "text-[10px]" : "text-[11px]"} ${STYLES[status]}`}>{status}</span>;
}
