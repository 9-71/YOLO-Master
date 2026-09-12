import { useJobs } from "../state/JobContext";

const STYLES = {
  checking: ["border-slate-700 bg-slate-800/60 text-slate-300", "bg-slate-500", "Checking…"],
  online: ["border-emerald-500/40 bg-emerald-500/15 text-emerald-300", "bg-emerald-400 animate-pulse", "Connected"],
  offline: ["border-rose-500/40 bg-rose-500/15 text-rose-300", "bg-rose-500", "Offline"],
} as const;

export function ConnectionPill() {
  const { connection } = useJobs();
  const [pill, dot, label] = STYLES[connection];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${pill}`}
      title="Engine connectivity (GET /health)"
    >
      <span className={`h-2 w-2 rounded-full ${dot}`} />
      {label}
    </span>
  );
}
