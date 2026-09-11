import { useEffect, useState } from "react";
import { useJobs } from "../state/JobContext";

const STYLES = {
  checking: ["border-slate-700 bg-slate-800/60 text-slate-300", "bg-slate-500", "Checking…"],
  online: ["border-emerald-500/40 bg-emerald-500/15 text-emerald-300", "bg-emerald-400 animate-pulse", "Connected"],
  offline: ["border-rose-500/40 bg-rose-500/15 text-rose-300", "bg-rose-500", "Offline"],
} as const;

export function ConnectionPill() {
  const { baseUrl, connection, applyBaseUrl, liveJobs, liveLogs, setToggle } = useJobs();
  const [draft, setDraft] = useState(baseUrl);
  useEffect(() => setDraft(baseUrl), [baseUrl]);
  const [pill, dot, label] = STYLES[connection];
  return (
    <header className="sticky top-0 z-40 border-b border-slate-800 bg-slate-900/90 px-4 py-3 backdrop-blur">
      <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-6 gap-y-3">
        <div className="mr-2 flex items-baseline gap-2"><h1 className="text-lg font-bold tracking-tight text-cyan-400">YOLO-Master</h1><span className="text-sm text-slate-400">React Verification Console</span></div>
        <div className="flex items-center gap-2">
          <label htmlFor="base-url" className="text-xs uppercase tracking-wider text-slate-500">Engine</label>
          <input id="base-url" value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") applyBaseUrl(draft); }} inputMode="url" spellCheck={false} className="w-64 rounded-md border border-slate-700 bg-slate-950 px-3 py-1.5 font-mono text-xs focus:border-cyan-500 focus:outline-none" />
          <button type="button" onClick={() => applyBaseUrl(draft)} className="rounded-md border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-semibold hover:bg-slate-700">Apply</button>
        </div>
        <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${pill}`}><span className={`h-2 w-2 rounded-full ${dot}`} />{label}</span>
        <div className="ml-auto flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-400">
          <label className="flex cursor-pointer items-center gap-2"><input type="checkbox" checked={liveJobs} onChange={(event) => setToggle("liveJobs", event.target.checked)} className="accent-cyan-500" />Live refresh (jobs)</label>
          <label className="flex cursor-pointer items-center gap-2"><input type="checkbox" checked={liveLogs} onChange={(event) => setToggle("liveLogs", event.target.checked)} className="accent-cyan-500" />Tail logs</label>
        </div>
      </div>
    </header>
  );
}
