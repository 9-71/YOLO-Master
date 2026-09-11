import { ArtifactInspector } from "./components/ArtifactInspector";
import { ConnectionPill } from "./components/ConnectionPill";
import { DispatchForm } from "./components/DispatchForm";
import { ErrorBanner } from "./components/ErrorBanner";
import { JobDetail } from "./components/JobDetail";
import { JobsTable } from "./components/JobsTable";
import { Terminal } from "./components/Terminal";

export default function App() {
  return <><ErrorBanner /><ConnectionPill /><main className="mx-auto grid max-w-[1600px] grid-cols-1 gap-4 p-4 lg:grid-cols-3"><DispatchForm /><section className="space-y-4 lg:col-span-2"><JobsTable /><JobDetail /></section></main><div className="mx-auto grid max-w-[1600px] grid-cols-1 gap-4 px-4 pb-6 lg:grid-cols-2"><Terminal /><ArtifactInspector /></div><footer className="mx-auto max-w-[1600px] px-4 pb-4 text-center text-[11px] text-slate-600">Vite + React + TypeScript console — API calls remain cross-origin by design.</footer></>;
}
