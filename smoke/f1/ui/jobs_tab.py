"""Gradio Jobs Tab for YOLO-Master Studio.

This module provides a dedicated Jobs Tab UI component for submitting tasks, monitoring
real-time lifecycle states, viewing live logs, and downloading generated artifacts.

Architecture:
    UI Components → Job Submission → JobDispatcherStateMachine.execute()
                 ↓
            State Monitor → Real-time status polling
                 ↓
            Log Console → stdout/stderr streaming
                 ↓
            Artifacts → File explorer with download buttons

Security:
    - Fail-closed path whitelisting (auto-fill allowed_paths from inputs/outputs)
    - Shell execution permanently disabled (allow_shell=False)
    - Path traversal protection via security constraints validation
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

from smoke.f1.dispatcher import JobDispatcherStateMachine
from smoke.f1.test_f1_smoke import JobRequest, JobStatus, SecurityConstraints, TaskType


class JobsManager:
    """Thread-safe job management with real-time state tracking."""

    def __init__(self) -> None:
        """Initialize job manager with in-memory storage."""
        self.jobs: dict[str, JobRequest] = {}
        self.job_logs: dict[str, list[str]] = {}
        self.lock = threading.Lock()
        self.dispatcher = JobDispatcherStateMachine()

    def submit_job(
        self,
        task_type: str,
        model_path: str,
        data_source: str,
        output_dir: str,
        conf: float,
        device: str,
        allowed_paths: list[str],
    ) -> tuple[str, str]:
        """Submit a new job for execution.

        Args:
            task_type: Task type (predict, train, export, diagnose)
            model_path: Path to model weights (.pt file)
            data_source: Path to input data (image/video/directory)
            output_dir: Base output directory for results
            conf: Confidence threshold (0.0, 1.0]
            device: Device specification ("0", "cpu", "mps")
            allowed_paths: Whitelist of allowed directory roots

        Returns:
            tuple[str, str]: (job_id, status_message)
        """
        # Generate unique job ID
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_id = f"{task_type}_{timestamp}_{uuid.uuid4().hex[:8]}"

        # Construct dynamic whitelist: include all input/output paths
        dynamic_whitelist = list(set(allowed_paths + [output_dir]))

        # Build params dict based on task type
        params = {}
        if task_type in ["predict", "train"]:
            params["model_path"] = model_path
            params["data_source"] = data_source
            params["conf"] = conf
            params["device"] = device
        elif task_type == "export":
            params["model_path"] = model_path
            params["format"] = "onnx"  # Default export format

        # Create JobRequest with fail-closed security
        job_request = JobRequest(
            job_id=job_id,
            task_type=TaskType(task_type),
            params=params,
            output={"output_dir": output_dir},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,  # Permanently disabled
                allowed_paths=dynamic_whitelist,
            ),
            runtime_tracking={
                "stream_logs": True,
                "timeout_seconds": 300,
                "cancellable": True,
                "cancel_requested": False,
            },
        )

        with self.lock:
            self.jobs[job_id] = job_request
            self.job_logs[job_id] = [f"[{datetime.now(timezone.utc).isoformat()}] Job {job_id} submitted"]

        # Execute job in background thread
        thread = threading.Thread(target=self._execute_job, args=(job_id,), daemon=True)
        thread.start()

        return job_id, f"✅ Job {job_id} submitted successfully"

    def _execute_job(self, job_id: str) -> None:
        """Execute job in background thread with log capture.

        Args:
            job_id: Job identifier
        """
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return

        self._append_log(job_id, f"[{datetime.now(timezone.utc).isoformat()}] Starting execution...")

        try:
            # Execute via dispatcher
            result = self.dispatcher.execute(job)

            with self.lock:
                self.jobs[job_id] = result

            if result.status == JobStatus.COMPLETED:
                self._append_log(
                    job_id,
                    f"[{datetime.now(timezone.utc).isoformat()}] ✅ Completed. Artifacts: {len(result.output.artifacts)}",
                )
            elif result.status == JobStatus.FAILED:
                error_msg = result.error.message if result.error else "Unknown error"
                self._append_log(job_id, f"[{datetime.now(timezone.utc).isoformat()}] ❌ Failed: {error_msg}")

        except Exception as e:  # noqa: BLE001
            self._append_log(
                job_id, f"[{datetime.now(timezone.utc).isoformat()}] ❌ Exception: {type(e).__name__}: {e}"
            )
            with self.lock:
                if job_id in self.jobs:
                    self.jobs[job_id].status = JobStatus.FAILED

    def _append_log(self, job_id: str, message: str) -> None:
        """Append log message to job log buffer."""
        with self.lock:
            if job_id not in self.job_logs:
                self.job_logs[job_id] = []
            self.job_logs[job_id].append(message)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get current job status and metadata.

        Args:
            job_id: Job identifier

        Returns:
            dict containing status, duration, error info, and artifact count
        """
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return {"status": "NOT_FOUND", "message": "Job not found"}

            duration = "N/A"
            if hasattr(job.metadata, "created_at"):
                created = datetime.fromisoformat(job.metadata.created_at)
                now = datetime.now(timezone.utc)
                duration = f"{(now - created).total_seconds():.1f}s"

            return {
                "status": job.status.value.upper(),
                "duration": duration,
                "error_code": job.error.code if job.error else None,
                "error_message": job.error.message if job.error else None,
                "artifact_count": len(job.output.artifacts) if hasattr(job.output, "artifacts") else 0,
            }

    def get_job_logs(self, job_id: str) -> str:
        """Get job logs as formatted string.

        Args:
            job_id: Job identifier

        Returns:
            Formatted log string
        """
        with self.lock:
            logs = self.job_logs.get(job_id, [])
            return "\n".join(logs) if logs else "No logs available"

    def get_job_artifacts(self, job_id: str) -> list[tuple[str, str]]:
        """Get job artifacts as (filename, absolute_path) tuples.

        Args:
            job_id: Job identifier

        Returns:
            List of (filename, path) tuples for artifact downloads
        """
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or not hasattr(job.output, "artifacts"):
                return []

            artifacts = []
            for artifact_path in job.output.artifacts:
                path = Path(artifact_path)
                if path.exists():
                    artifacts.append((path.name, str(path.absolute())))
            return artifacts

    def cancel_job(self, job_id: str) -> str:
        """Request job cancellation.

        Args:
            job_id: Job identifier

        Returns:
            Status message
        """
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return "❌ Job not found"

            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                return f"⚠️ Job already in terminal state: {job.status.value}"

            job.runtime_tracking.cancel_requested = True

        # Append log AFTER releasing the lock: _append_log acquires self.lock
        # internally and threading.Lock is not reentrant (self-deadlock).
        self._append_log(job_id, f"[{datetime.now(timezone.utc).isoformat()}] 🚫 Cancellation requested")
        return f"✅ Cancellation requested for {job_id}"

    def list_recent_jobs(self, limit: int = 10) -> list[dict[str, str]]:
        """List recent jobs with summary info.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of job summary dicts
        """
        with self.lock:
            jobs_list = []
            for job_id, job in sorted(self.jobs.items(), key=lambda x: x[1].metadata.created_at, reverse=True)[:limit]:
                jobs_list.append(
                    {
                        "job_id": job_id,
                        "task_type": job.task_type.value,
                        "status": job.status.value.upper(),
                        "created_at": job.metadata.created_at,
                    }
                )
            return jobs_list


def create_jobs_tab(jobs_manager: JobsManager) -> gr.Blocks:
    """Create Gradio Jobs Tab UI component.

    Args:
        jobs_manager: JobsManager instance for job orchestration

    Returns:
        gr.Blocks: Gradio tab component
    """
    with gr.Blocks() as jobs_tab:
        gr.Markdown("# 📋 Jobs Management")

        with gr.Row(equal_height=False):
            # ==================== Left Panel: Job Submission ====================
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 🚀 Submit Job")

                # Task type selector
                task_type_radio = gr.Radio(
                    choices=["predict", "train", "export", "diagnose"],
                    value="predict",
                    label="Task Type",
                )

                # Dynamic input parameters form
                with gr.Group():
                    model_path_txt = gr.Textbox(
                        value="yolov8n.pt",
                        label="Model Path",
                        placeholder="yolov8n.pt or ./ckpts/model.pt",
                    )
                    data_source_txt = gr.Textbox(
                        value="ultralytics/assets/bus.jpg",
                        label="Data Source",
                        placeholder="Image/video/directory path",
                    )
                    output_dir_txt = gr.Textbox(
                        value="runs/predict",
                        label="Output Directory",
                        placeholder="runs/predict",
                    )

                # Hyperparameters
                with gr.Accordion("⚙️ Hyperparameters", open=True):
                    conf_slider = gr.Slider(0.0, 1.0, 0.25, step=0.01, label="Confidence Threshold")
                    device_txt = gr.Textbox("0", label="Device (0 for GPU, cpu for CPU)")

                # Security constraints
                with gr.Accordion("🔒 Security Constraints", open=False):
                    allowed_paths_txt = gr.Textbox(
                        value="., ultralytics/assets, runs, ckpts",
                        label="Allowed Paths (comma-separated)",
                        info="Whitelist of allowed directory roots",
                    )
                    gr.Markdown(
                        "**Security Policy**: Shell execution is **permanently disabled**. "
                        "All paths are validated against whitelist."
                    )

                submit_btn = gr.Button("🔥 Submit Job", variant="primary", size="lg")

            # ==================== Right Panel: Monitoring ====================
            with gr.Column(scale=2), gr.Tabs():
                # Tab 1: State & Progress Monitor
                with gr.TabItem("📊 Status Monitor"):
                    job_id_display = gr.Textbox(label="Current Job ID", interactive=False)
                    status_display = gr.JSON(label="Job Status")

                    with gr.Row():
                        refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")
                        cancel_job_btn = gr.Button("🚫 Cancel Job", size="sm", variant="stop")

                    error_box = gr.Textbox(label="Error Diagnostics", interactive=False, lines=3)

                # Tab 2: Live Logs & Output Console
                with gr.TabItem("📜 Live Logs"):
                    logs_console = gr.Textbox(
                        label="Execution Logs",
                        lines=20,
                        interactive=False,
                        max_lines=100,
                    )
                    refresh_logs_btn = gr.Button("🔄 Refresh Logs", size="sm")

                # Tab 3: Artifacts Section
                with gr.TabItem("📁 Artifacts"):
                    artifacts_list = gr.Dataframe(
                        headers=["Filename", "Path"],
                        label="Generated Artifacts",
                        interactive=False,
                    )
                    refresh_artifacts_btn = gr.Button("🔄 Refresh Artifacts", size="sm")
                    gr.Markdown("**Download**: Click on artifact path to copy, then use file explorer")

                # Tab 4: Recent Jobs
                with gr.TabItem("🕒 Recent Jobs"):
                    recent_jobs_table = gr.Dataframe(
                        headers=["Job ID", "Task Type", "Status", "Created At"],
                        label="Recent Jobs",
                        interactive=False,
                    )
                    refresh_recent_btn = gr.Button("🔄 Refresh List", size="sm")

        # ==================== Event Handlers ====================

        def submit_job_handler(
            task_type: str,
            model_path: str,
            data_source: str,
            output_dir: str,
            conf: float,
            device: str,
            allowed_paths_str: str,
        ) -> tuple[str, dict, str]:
            """Handle job submission."""
            # Parse allowed_paths from comma-separated string
            allowed_paths = [p.strip() for p in allowed_paths_str.split(",") if p.strip()]

            job_id, message = jobs_manager.submit_job(
                task_type=task_type,
                model_path=model_path,
                data_source=data_source,
                output_dir=output_dir,
                conf=conf,
                device=device,
                allowed_paths=allowed_paths,
            )

            # Return job_id, initial status, and message
            status = jobs_manager.get_job_status(job_id)
            return job_id, status, message

        def refresh_status_handler(job_id: str) -> tuple[dict, str]:
            """Refresh job status."""
            if not job_id:
                return {}, "⚠️ No job selected"

            status = jobs_manager.get_job_status(job_id)
            error_msg = ""
            if status.get("error_message"):
                error_msg = f"[{status.get('error_code', 'ERROR')}] {status['error_message']}"

            return status, error_msg

        def refresh_logs_handler(job_id: str) -> str:
            """Refresh job logs."""
            if not job_id:
                return "⚠️ No job selected"
            return jobs_manager.get_job_logs(job_id)

        def refresh_artifacts_handler(job_id: str) -> list:
            """Refresh artifacts list."""
            if not job_id:
                return []
            artifacts = jobs_manager.get_job_artifacts(job_id)
            return [[name, path] for name, path in artifacts]

        def cancel_job_handler(job_id: str) -> str:
            """Handle job cancellation."""
            if not job_id:
                return "⚠️ No job selected"
            return jobs_manager.cancel_job(job_id)

        def refresh_recent_jobs_handler() -> list:
            """Refresh recent jobs list."""
            jobs_list = jobs_manager.list_recent_jobs(limit=20)
            return [[j["job_id"], j["task_type"], j["status"], j["created_at"]] for j in jobs_list]

        # Bind events
        submit_btn.click(
            fn=submit_job_handler,
            inputs=[
                task_type_radio,
                model_path_txt,
                data_source_txt,
                output_dir_txt,
                conf_slider,
                device_txt,
                allowed_paths_txt,
            ],
            outputs=[job_id_display, status_display, gr.Textbox(visible=False)],
        ).then(fn=lambda x: x, inputs=job_id_display, outputs=job_id_display)

        refresh_status_btn.click(
            fn=refresh_status_handler,
            inputs=job_id_display,
            outputs=[status_display, error_box],
        )

        refresh_logs_btn.click(
            fn=refresh_logs_handler,
            inputs=job_id_display,
            outputs=logs_console,
        )

        refresh_artifacts_btn.click(
            fn=refresh_artifacts_handler,
            inputs=job_id_display,
            outputs=artifacts_list,
        )

        cancel_job_btn.click(
            fn=cancel_job_handler,
            inputs=job_id_display,
            outputs=error_box,
        )

        refresh_recent_btn.click(
            fn=refresh_recent_jobs_handler,
            inputs=None,
            outputs=recent_jobs_table,
        )

    return jobs_tab
