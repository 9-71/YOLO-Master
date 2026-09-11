export type TaskType = "predict" | "train" | "val" | "export" | "diagnose";
export type JobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface JobRequest {
  job_id: string;
  task_type: TaskType;
  params: Record<string, unknown>;
  output: { output_dir: string };
  security_constraints: { allowed_paths: string[] };
  runtime_tracking: { timeout_seconds: number; cancellable: boolean };
}

export interface JobSummary {
  job_id: string;
  task_type: TaskType;
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  duration: number | null;
}

export interface JobsResponse { jobs: JobSummary[]; total: number; limit: number; offset: number }

export interface JobDetail extends Omit<JobSummary, "created_at"> {
  created_at: string | null;
  error_code: string | null;
  error_message: string | null;
  artifact_count: number;
  metadata: Record<string, unknown>;
}

export interface LogsResponse {
  job_id: string;
  total: number;
  offset: number;
  limit: number | null;
  logs: string[];
  next_offset: number | null;
}

export interface ArtifactEntry { filename: string; artifact_id: string; is_image: boolean; download_url: string }
export interface ArtifactsResponse { job_id: string; artifacts: ArtifactEntry[]; image_artifacts: string[] }

export type ConnectionState = "checking" | "online" | "offline";
export interface BannerState { kind: "error" | "success" | "security"; message: string }

export const ACTIVE_STATUSES = new Set<JobStatus>(["pending", "running"]);
export const EMPTY_ARTIFACTS: ArtifactsResponse = { job_id: "", artifacts: [], image_artifacts: [] };

export function formatDuration(duration: number | null): string {
  return duration === null ? "—" : `${duration.toFixed(2)}s`;
}
