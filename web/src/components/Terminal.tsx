import { useLogStream } from "../hooks/useLogStream";
import { useJobs } from "../state/JobContext";
import { StatusBadge } from "./StatusBadge";

export function Terminal() {
  const { selectedJobId, detail, logs, autoScroll, clearLogs, setToggle } = useJobs();
  const terminalRef = useLogStream(logs, autoScroll);
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900">
      <div className="flex flex-wrap items-center gap-3 border-b border-slate-800 px-4 py-3"><h2 className="text-sm font-bold uppercase tracking-wider text-slate-300">Execution Terminal</h2><span className="max-w-72 truncate rounded bg-slate-800 px-2 py-0.5 font-mono text-[11px] text-cyan-300">{selectedJobId || "no job selected"}</span>{detail && <StatusBadge status={detail.status} small />}<div className="ml-auto flex items-center gap-4"><label className="flex cursor-pointer items-center gap-2 text-xs text-slate-400"><input type="checkbox" checked={autoScroll} onChange={(event) => setToggle("autoScroll", event.target.checked)} className="accent-cyan-500" />Auto-scroll</label><button type="button" onClick={clearLogs} className="rounded-md border border-slate-700 px-2.5 py-1 text-xs font-semibold hover:bg-slate-800">Clear Screen</button></div></div>
      <div className="p-3"><pre ref={terminalRef} className="h-80 overflow-y-auto whitespace-pre-wrap rounded-md bg-black p-3 font-mono text-[11px] leading-relaxed text-emerald-400">{logs.join("\n")}</pre></div>
    </section>
  );
}
