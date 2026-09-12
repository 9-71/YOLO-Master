import { useEffect, useState } from "react";
import { ConnectionPill } from "./components/ConnectionPill";
import { DispatchForm } from "./components/DispatchForm";
import { ErrorBanner } from "./components/ErrorBanner";
import { JobDetailPage } from "./components/JobDetail";
import { JobsPage } from "./components/JobsTable";
import { useJobs } from "./state/JobContext";

type Page = { name: "new" } | { name: "jobs" } | { name: "detail"; jobId: string };

export default function App() {
  const { baseUrl, applyBaseUrl } = useJobs();
  const [page, setPage] = useState<Page>({ name: "jobs" });
  const [engineDraft, setEngineDraft] = useState(baseUrl);

  useEffect(() => setEngineDraft(baseUrl), [baseUrl]);

  function applyEngineUrl() {
    applyBaseUrl(engineDraft);
    setPage({ name: "jobs" });
  }

  function openJob(jobId: string) {
    setPage({ name: "detail", jobId });
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <ErrorBanner />
      <header className="sticky top-0 z-40 border-b border-slate-800 bg-slate-900/90 px-4 py-3 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-6 gap-y-3">
          <h1 className="text-lg font-bold tracking-tight text-cyan-400">YOLO-Master Studio</h1>
          <nav className="flex items-center gap-2" aria-label="Primary navigation">
            <button
              type="button"
              onClick={() => setPage({ name: "new" })}
              className={`rounded-md px-3 py-1.5 text-sm font-semibold ${
                page.name === "new" ? "bg-cyan-500/15 text-cyan-300" : "text-slate-400 hover:bg-slate-800"
              }`}
            >
              New Job
            </button>
            <button
              type="button"
              onClick={() => setPage({ name: "jobs" })}
              className={`rounded-md px-3 py-1.5 text-sm font-semibold ${
                page.name === "jobs" ? "bg-cyan-500/15 text-cyan-300" : "text-slate-400 hover:bg-slate-800"
              }`}
            >
              Jobs
            </button>
          </nav>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <label htmlFor="engine-url" className="text-xs uppercase tracking-wider text-slate-500">
              Engine
            </label>
            <input
              id="engine-url"
              value={engineDraft}
              onChange={(event) => setEngineDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") applyEngineUrl();
              }}
              inputMode="url"
              spellCheck={false}
              className="w-64 rounded-md border border-slate-700 bg-slate-950 px-3 py-1.5 font-mono text-xs focus:border-cyan-500 focus:outline-none"
            />
            <button
              type="button"
              onClick={applyEngineUrl}
              className="rounded-md border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-semibold hover:bg-slate-700"
            >
              Apply
            </button>
            <ConnectionPill />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] p-4">
        {page.name === "new" && <DispatchForm onJobCreated={openJob} />}
        {page.name === "jobs" && <JobsPage onOpenJob={openJob} />}
        {page.name === "detail" && <JobDetailPage jobId={page.jobId} onBack={() => setPage({ name: "jobs" })} />}
      </main>
    </div>
  );
}
