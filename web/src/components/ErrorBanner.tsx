import { useJobs } from "../state/JobContext";

export function ErrorBanner() {
  const { banner, dismissBanner } = useJobs();
  if (!banner) return null;
  const style = banner.kind === "success"
    ? "border-emerald-500/40 bg-emerald-950/95 text-emerald-200"
    : banner.kind === "security"
      ? "border-amber-300 bg-amber-950/95 text-amber-100 ring-2 ring-inset ring-amber-400"
      : "border-rose-500/40 bg-rose-950/95 text-rose-200";
  return (
    <div className={`fixed inset-x-0 top-0 z-50 border-b px-4 py-3 text-sm shadow-lg ${style}`} role="alert">
      <div className="mx-auto flex max-w-[1600px] items-start gap-3">
        {banner.kind === "security" && <span className="font-black" aria-hidden="true">SECURITY</span>}
        <span className="flex-1 whitespace-pre-wrap break-words">{banner.message}</span>
        <button type="button" className="font-bold opacity-70 hover:opacity-100" onClick={dismissBanner} aria-label="Dismiss banner">×</button>
      </div>
    </div>
  );
}
