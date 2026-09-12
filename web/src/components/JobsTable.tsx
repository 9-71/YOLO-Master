import { useJobs } from "../state/JobContext";
import { ACTIVE_STATUSES, formatCreatedAt, formatDuration } from "../types";
import { StatusBadge } from "./StatusBadge";

export function JobsTable({ onOpenJob }: { onOpenJob: (jobId: string) => void }) {
  const { jobs, cancellationPending, cancelJob } = useJobs();
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-xs uppercase tracking-wider text-slate-500">
          <tr className="border-b border-slate-800">
            <th className="px-4 py-2">Job ID</th>
            <th className="px-4 py-2">Task Type</th>
            <th className="px-4 py-2">Status</th>
            <th className="px-4 py-2">Duration</th>
            <th className="px-4 py-2">Created At</th>
            <th className="px-4 py-2 text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => {
            const active = ACTIVE_STATUSES.has(job.status);
            const cancelling = cancellationPending === job.job_id;
            const open = () => onOpenJob(job.job_id);
            return (
              <tr
                key={job.job_id}
                onClick={open}
                className="cursor-pointer border-b border-slate-800/60 hover:bg-slate-800/50"
              >
                <td className="max-w-64 truncate px-4 py-2 font-mono text-xs text-cyan-300" title={job.job_id}>
                  {job.job_id}
                </td>
                <td className="px-4 py-2 text-xs font-bold uppercase text-slate-300">{job.task_type}</td>
                <td className="px-4 py-2">
                  <StatusBadge status={job.status} />
                </td>
                <td className="px-4 py-2 font-mono text-xs">{formatDuration(job.duration)}</td>
                <td className="whitespace-nowrap px-4 py-2 text-xs text-slate-400">
                  {formatCreatedAt(job.created_at)}
                </td>
                <td className="whitespace-nowrap px-4 py-2 text-right">
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      open();
                    }}
                    className="rounded-md border border-slate-700 px-2 py-1 text-[11px] font-semibold hover:bg-slate-700"
                  >
                    View
                  </button>
                  <button
                    type="button"
                    disabled={!active || cancellationPending !== null}
                    onClick={(event) => {
                      event.stopPropagation();
                      if (window.confirm(`Request cooperative cancellation of job ${job.job_id}?`)) {
                        void cancelJob(job.job_id);
                      }
                    }}
                    className="ml-1.5 rounded-md border border-rose-500/40 bg-rose-500/10 px-2 py-1 text-[11px] font-semibold text-rose-300 hover:bg-rose-500/20 disabled:border-slate-800 disabled:text-slate-600"
                  >
                    {cancelling ? "Cancelling…" : cancellationPending ? "Cancel pending" : "Cancel"}
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {jobs.length === 0 && (
        <p className="px-4 py-6 text-center text-xs text-slate-500">No jobs yet — create one from New Job.</p>
      )}
    </div>
  );
}

export function JobsPage({ onOpenJob }: { onOpenJob: (jobId: string) => void }) {
  const { jobs, cancellationPending, liveJobs, setToggle, refreshJobs } = useJobs();
  const syncFast =
    jobs.some((job) => ACTIVE_STATUSES.has(job.status)) || cancellationPending !== null;
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-4 py-3">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300">Jobs</h2>
          <p className="mt-0.5 text-[11px] text-slate-500">
            Auto-sync: {liveJobs ? (syncFast ? "2s (active / cancelling)" : "30s (idle)") : "off"}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-400">
            <input
              type="checkbox"
              checked={liveJobs}
              onChange={(event) => setToggle("liveJobs", event.target.checked)}
              className="accent-cyan-500"
            />
            Live refresh
          </label>
          <button
            type="button"
            onClick={() => void refreshJobs(true)}
            className="rounded-md border border-slate-700 px-2.5 py-1 text-xs font-semibold hover:bg-slate-800"
          >
            Refresh now
          </button>
        </div>
      </div>
      <JobsTable onOpenJob={onOpenJob} />
    </div>
  );
}
