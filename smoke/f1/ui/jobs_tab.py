"""Gradio Jobs Tab for YOLO-Master Studio.

This module provides a dedicated Jobs Tab UI component for submitting tasks, monitoring
real-time lifecycle states, viewing live logs, and downloading generated artifacts.

Architecture:
    UI Components → Job Submission → JobDispatcherStateMachine.execute()
                 ↓
            Adaptive Polling → 1s lifecycle timer while PENDING/RUNNING
                 ↓            → 30s background sync (recent jobs, artifacts)
            Log Console → stdout/stderr streaming
                 ↓
            Artifacts → File explorer with download buttons

Polling:
    The fast lifecycle timer (1s) ticks while the selected job is active. The tick that
    observes a terminal state performs the final refresh (including artifacts) and
    deactivates itself via ``gr.update(active=False)``. A slow always-on timer (30s)
    keeps the recent-jobs table and final artifacts fresh while the fast timer is idle.

Security:
    - Fail-closed path whitelisting (auto-fill allowed_paths from inputs/outputs)
    - Shell execution permanently disabled (allow_shell=False)
    - Path traversal protection via security constraints validation
    - SEC_ERR_001 / PARAM_VALIDATION_FAILED map to one-shot gr.Warning toasts and a
      persistent localized status banner
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

from smoke.f1.dispatcher import JobDispatcherStateMachine
from smoke.f1.test_f1_smoke import JobRequest, JobStatus, SecurityConstraints, TaskType
from smoke.f1.ui.i18n import DEFAULT_LANGUAGE, get_columns, get_text

#: Job states that still require high-frequency lifecycle polling.
ACTIVE_STATUSES = frozenset({"PENDING", "RUNNING"})
#: Backend error codes that map to visual security/validation user alerts.
SECURITY_ALERT_CODES = frozenset({"SEC_ERR_001", "PARAM_VALIDATION_FAILED"})
#: Fast lifecycle polling interval (seconds) while a job is active.
POLL_FAST_SECONDS = 1.0
#: Slow background sync interval (seconds) for recent jobs and final artifacts.
POLL_SLOW_SECONDS = 30.0


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


def is_terminal_status(status: str) -> bool:
    """Return True when a job status string is terminal (not PENDING/RUNNING).

    COMPLETED, FAILED and NOT_FOUND are terminal from the poller's perspective;
    a cancelled job also surfaces as FAILED (USER_CANCELLED) via the state machine,
    and CANCELLED is treated as terminal for forward compatibility.

    Args:
        status: Uppercase backend status string.

    Returns:
        bool: True when high-frequency polling should stop.

    Example:
        >>> is_terminal_status("RUNNING")
        False
        >>> is_terminal_status("COMPLETED")
        True
        >>> is_terminal_status("CANCELLED")
        True
    """
    return status not in ACTIVE_STATUSES


def alert_banner(lang: str, code: str, message: str | None) -> str:
    """Build a localized Markdown alert banner for a failed job.

    Security/validation error codes get dedicated localized titles and bodies;
    other failure codes fall back to the generic failure title with the raw message.

    Args:
        lang: ISO language code.
        code: Backend error code (e.g. "SEC_ERR_001").
        message: Raw backend error message.

    Returns:
        str: Markdown blockquote banner.

    Example:
        >>> banner = alert_banner("en", "SEC_ERR_001", "path not in whitelist")
        >>> "Security Policy Violation" in banner
        True
        >>> banner = alert_banner("zh", "EXEC_ERR_500", "boom")
        >>> "作业失败" in banner
        True
    """
    if code in SECURITY_ALERT_CODES:
        title = get_text(lang, f"alert.{code}.title")
        body = get_text(lang, f"alert.{code}.body")
        detail = f"{code}: {message}" if message else code
        return f"> **{title}**\n> {body}\n> `{detail}`"
    title = get_text(lang, "alert.generic.title")
    detail = f"{code}: {message}" if message else code
    return f"> **{title}**\n> `{detail}`"


def security_alert_toast(lang: str, code: str) -> str:
    """Compose the one-shot gr.Warning toast text for a security/validation error.

    Args:
        lang: ISO language code.
        code: Backend error code from SECURITY_ALERT_CODES.

    Returns:
        str: Localized toast text.

    Example:
        >>> toast = security_alert_toast("zh", "SEC_ERR_001")
        >>> "安全策略违规" in toast
        True
    """
    return f"{get_text(lang, f'alert.{code}.title')} — {get_text(lang, f'alert.{code}.body')}"


@dataclass(frozen=True)
class PollState:
    """Snapshot of all job-monitoring panels produced by one polling cycle.

    Attributes:
        status: Localized status dict rendered by the Status Monitor JSON panel.
        error_text: Diagnostics text for the error box (empty when healthy).
        banner: Markdown alert banner (empty when the job has no failure).
        logs: Formatted execution logs.
        artifacts: Rows for the artifacts dataframe (filename, path).
        recent: Rows for the recent-jobs dataframe.
        keep_polling: True while the job is active; False once terminal (deactivates the fast timer).
    """

    status: dict[str, Any]
    error_text: str = ""
    banner: str = ""
    logs: str = ""
    artifacts: list[list[str]] = field(default_factory=list)
    recent: list[list[str]] = field(default_factory=list)
    keep_polling: bool = False


def compute_poll_state(jobs_manager: JobsManager, job_id: str, lang: str = DEFAULT_LANGUAGE) -> PollState:
    """Compute the complete polling snapshot for one job in the given language.

    Pure presentation logic over the JobsManager backend API: the backend responses
    are never modified, so backend-level test assertions remain valid. Raw status
    fields are preserved alongside localized ``status_label`` entries.

    Args:
        jobs_manager: JobsManager instance (or duck-typed equivalent for tests).
        job_id: Selected job identifier (may be empty).
        lang: ISO language code passed to i18n lookups.

    Returns:
        PollState: Snapshot with localized status, banner, logs, artifacts, recent
        jobs, and the keep_polling flag used to deactivate high-frequency polling.

    Example:
        >>> manager = JobsManager()
        >>> state = compute_poll_state(manager, "", "en")
        >>> state.keep_polling
        False
        >>> state.status["status_label"] == "⚠️ No job selected"
        True
    """
    if not job_id:
        return PollState(
            status={"status": "NO_SELECTION", "status_label": get_text(lang, "msg.no_job_selected")},
        )

    raw = jobs_manager.get_job_status(job_id)
    status_str = raw.get("status", "NOT_FOUND")

    status = {
        "job_id": job_id,
        "status": status_str,
        "status_label": get_text(lang, f"status.{status_str}"),
        "duration": raw.get("duration"),
        "error_code": raw.get("error_code"),
        "error_message": raw.get("error_message"),
        "artifact_count": raw.get("artifact_count"),
    }

    error_text = ""
    banner = ""
    if status_str == "FAILED":
        code = raw.get("error_code") or "UNKNOWN"
        error_text = f"[{code}] {raw.get('error_message') or ''}"
        banner = alert_banner(lang, code, raw.get("error_message"))

    logs = jobs_manager.get_job_logs(job_id)
    artifacts = [[name, path] for name, path in jobs_manager.get_job_artifacts(job_id)]
    recent = [
        [j["job_id"], j["task_type"], j["status"], j["created_at"]] for j in jobs_manager.list_recent_jobs(limit=20)
    ]

    return PollState(
        status=status,
        error_text=error_text,
        banner=banner,
        logs=logs,
        artifacts=artifacts,
        recent=recent,
        keep_polling=not is_terminal_status(status_str),
    )


def create_jobs_tab(jobs_manager: JobsManager, lang: str = DEFAULT_LANGUAGE) -> gr.Blocks:
    """Create the Jobs Tab UI with adaptive polling, security alerts and i18n.

    Args:
        jobs_manager: Application-level JobsManager singleton shared across tabs.
        lang: Initial UI language ("en" or "zh"); switchable at runtime via the
            language selector inside the tab.

    Returns:
        gr.Blocks: The Jobs Tab as a Gradio Blocks.

    Mounting contract:
        Mount the returned Blocks by calling it bare inside the parent tab context::

            with gr.TabItem("📋 Jobs"):
                create_jobs_tab(jobs_manager)

        Gradio auto-embeds a child Blocks on context exit, so do NOT additionally
        call ``.render()`` — that would mount every Jobs component a second time
        (duplicated tabs in the DOM).

    Polling design:
        - Fast lifecycle timer (1s): ticks while the selected job is PENDING/RUNNING and
          refreshes status, logs, artifacts and recent jobs on every tick. The tick that
          observes a terminal state performs the final refresh (including artifacts) and
          deactivates the timer via ``gr.update(active=False)``.
        - Slow sync timer (30s): always-on low-frequency refresh of the same panels so
          recent jobs and final artifacts stay fresh while the fast timer is idle.

    Security alerts:
        - The first tick observing SEC_ERR_001 / PARAM_VALIDATION_FAILED raises a
          one-shot gr.Warning toast; a persistent localized banner stays visible in the
          Status Monitor tab.
    """
    # Job IDs whose security alert has already been toasted (one-shot warning guard).
    _security_warned: set[str] = set()

    with gr.Blocks() as jobs_tab:
        lang_state = gr.State(lang)
        title_md = gr.Markdown(f"# {get_text(lang, 'tab.title')}")

        with gr.Row(equal_height=False):
            # ==================== Left Panel: Job Submission ====================
            with gr.Column(scale=1, variant="panel"):
                submit_md = gr.Markdown(f"### {get_text(lang, 'panel.submit')}")
                lang_radio = gr.Radio(
                    choices=[("English", "en"), ("中文", "zh")],
                    value=lang,
                    label=get_text(lang, "lang.label"),
                )

                # Task type selector
                task_type_radio = gr.Radio(
                    choices=["predict", "train", "export", "diagnose"],
                    value="predict",
                    label=get_text(lang, "field.task_type"),
                )

                # Dynamic input parameters form
                with gr.Group():
                    model_path_txt = gr.Textbox(
                        value="yolov8n.pt",
                        label=get_text(lang, "field.model_path"),
                        placeholder=get_text(lang, "field.model_path.placeholder"),
                    )
                    data_source_txt = gr.Textbox(
                        value="ultralytics/assets/bus.jpg",
                        label=get_text(lang, "field.data_source"),
                        placeholder=get_text(lang, "field.data_source.placeholder"),
                    )
                    output_dir_txt = gr.Textbox(
                        value="runs/predict",
                        label=get_text(lang, "field.output_dir"),
                        placeholder=get_text(lang, "field.output_dir.placeholder"),
                    )

                # Hyperparameters
                with gr.Accordion(get_text(lang, "accordion.hyperparams"), open=True) as hyperparams_accordion:
                    conf_slider = gr.Slider(0.0, 1.0, 0.25, step=0.01, label=get_text(lang, "field.conf"))
                    device_txt = gr.Textbox("0", label=get_text(lang, "field.device"))

                # Security constraints
                with gr.Accordion(get_text(lang, "accordion.security"), open=False) as security_accordion:
                    allowed_paths_txt = gr.Textbox(
                        value="., ultralytics/assets, runs, ckpts",
                        label=get_text(lang, "field.allowed_paths"),
                        info=get_text(lang, "field.allowed_paths.info"),
                    )
                    security_md = gr.Markdown(get_text(lang, "security.policy"))

                submit_btn = gr.Button(get_text(lang, "button.submit"), variant="primary", size="lg")
                submit_msg = gr.Markdown()

            # ==================== Right Panel: Monitoring ====================
            with gr.Column(scale=2), gr.Tabs():
                # Tab 1: State & Progress Monitor
                with gr.TabItem(get_text(lang, "subtab.status")) as status_tab:
                    job_id_display = gr.Textbox(label=get_text(lang, "field.job_id"), interactive=False)
                    status_display = gr.JSON(label=get_text(lang, "field.status"))
                    cancel_job_btn = gr.Button(get_text(lang, "button.cancel"), size="sm", variant="stop")
                    banner_md = gr.Markdown()
                    error_box = gr.Textbox(label=get_text(lang, "field.error"), interactive=False, lines=3)

                # Tab 2: Live Logs & Output Console
                with gr.TabItem(get_text(lang, "subtab.logs")) as logs_tab:
                    logs_console = gr.Textbox(
                        label=get_text(lang, "field.logs"),
                        lines=20,
                        interactive=False,
                        max_lines=100,
                    )

                # Tab 3: Artifacts Section
                with gr.TabItem(get_text(lang, "subtab.artifacts")) as artifacts_tab:
                    artifacts_list = gr.Dataframe(
                        headers=get_columns(lang, "artifacts"),
                        label=get_text(lang, "df.artifacts"),
                        interactive=False,
                    )
                    artifacts_hint_md = gr.Markdown(get_text(lang, "hint.artifacts"))

                # Tab 4: Recent Jobs
                with gr.TabItem(get_text(lang, "subtab.recent")) as recent_tab:
                    recent_jobs_table = gr.Dataframe(
                        headers=get_columns(lang, "recent"),
                        label=get_text(lang, "df.recent"),
                        interactive=False,
                    )
                    poll_note_md = gr.Markdown(get_text(lang, "poll.note"))

        # Adaptive timers: fast lifecycle poll (activated on submit, self-deactivates
        # on terminal state) and slow always-on background sync.
        poll_timer = gr.Timer(POLL_FAST_SECONDS, active=False)
        sync_timer = gr.Timer(POLL_SLOW_SECONDS)

        # ==================== Event Handlers ====================

        def poll_snapshot(job_id: str, lang_value: str) -> PollState:
            """Compute one snapshot, raising a one-shot warning for new security alerts."""
            if job_id:
                raw = jobs_manager.get_job_status(job_id)
                code = raw.get("error_code")
                if code in SECURITY_ALERT_CODES and job_id not in _security_warned:
                    _security_warned.add(job_id)
                    raise gr.Warning(security_alert_toast(lang_value, code))
            return compute_poll_state(jobs_manager, job_id, lang_value)

        def poll_handler(job_id: str, lang_value: str) -> tuple:
            """Fast lifecycle poll: refresh every panel and deactivate on terminal state."""
            state = poll_snapshot(job_id, lang_value)
            return (
                state.status,
                state.error_text,
                state.banner,
                state.logs,
                state.artifacts,
                state.recent,
                gr.update(active=state.keep_polling),
            )

        def sync_handler(job_id: str, lang_value: str) -> tuple:
            """Slow background sync: refresh panels without touching the fast timer."""
            state = poll_snapshot(job_id, lang_value)
            return state.status, state.error_text, state.banner, state.logs, state.artifacts, state.recent

        def submit_job_handler(
            task_type: str,
            model_path: str,
            data_source: str,
            output_dir: str,
            conf: float,
            device: str,
            allowed_paths_str: str,
            lang_value: str,
        ) -> tuple[str, str, Any]:
            """Submit a job and (re)activate the fast lifecycle timer."""
            # Parse allowed_paths from comma-separated string
            allowed_paths = [p.strip() for p in allowed_paths_str.split(",") if p.strip()]

            job_id, _message = jobs_manager.submit_job(
                task_type=task_type,
                model_path=model_path,
                data_source=data_source,
                output_dir=output_dir,
                conf=conf,
                device=device,
                allowed_paths=allowed_paths,
            )

            # A fresh submission may reuse the security-warning guard.
            _security_warned.discard(job_id)

            return (
                job_id,
                get_text(lang_value, "msg.job_submitted").format(job_id=job_id),
                gr.update(active=True),
            )

        def cancel_job_handler(job_id: str, lang_value: str) -> tuple[str, Any]:
            """Request cancellation, surfacing localized warnings for invalid states."""
            if not job_id:
                raise gr.Warning(get_text(lang_value, "msg.no_job_selected"))
            raw = jobs_manager.get_job_status(job_id)
            if raw.get("status") == "NOT_FOUND":
                raise gr.Warning(get_text(lang_value, "msg.job_not_found"))
            if is_terminal_status(raw.get("status", "")):
                raise gr.Warning(get_text(lang_value, "msg.terminal_state").format(status=raw["status"]))

            jobs_manager.cancel_job(job_id)
            return get_text(lang_value, "msg.cancel_requested").format(job_id=job_id), gr.update(active=True)

        def apply_language(lang_value: str) -> tuple:
            """Relabel every localizable component when the language changes."""
            return (
                lang_value,  # lang_state
                gr.update(label=get_text(lang_value, "lang.label")),  # lang_radio
                gr.update(value=f"# {get_text(lang_value, 'tab.title')}"),  # title_md
                gr.update(value=f"### {get_text(lang_value, 'panel.submit')}"),  # submit_md
                gr.update(label=get_text(lang_value, "field.task_type")),  # task_type_radio
                gr.update(
                    label=get_text(lang_value, "field.model_path"),
                    placeholder=get_text(lang_value, "field.model_path.placeholder"),
                ),  # model_path_txt
                gr.update(
                    label=get_text(lang_value, "field.data_source"),
                    placeholder=get_text(lang_value, "field.data_source.placeholder"),
                ),  # data_source_txt
                gr.update(
                    label=get_text(lang_value, "field.output_dir"),
                    placeholder=get_text(lang_value, "field.output_dir.placeholder"),
                ),  # output_dir_txt
                gr.update(label=get_text(lang_value, "accordion.hyperparams")),  # hyperparams_accordion
                gr.update(label=get_text(lang_value, "field.conf")),  # conf_slider
                gr.update(label=get_text(lang_value, "field.device")),  # device_txt
                gr.update(label=get_text(lang_value, "accordion.security")),  # security_accordion
                gr.update(
                    label=get_text(lang_value, "field.allowed_paths"),
                    info=get_text(lang_value, "field.allowed_paths.info"),
                ),  # allowed_paths_txt
                gr.update(value=get_text(lang_value, "security.policy")),  # security_md
                gr.update(value=get_text(lang_value, "button.submit")),  # submit_btn
                gr.update(value=get_text(lang_value, "button.cancel")),  # cancel_job_btn
                gr.update(label=get_text(lang_value, "subtab.status")),  # status_tab
                gr.update(label=get_text(lang_value, "field.job_id")),  # job_id_display
                gr.update(label=get_text(lang_value, "field.status")),  # status_display
                gr.update(label=get_text(lang_value, "field.error")),  # error_box
                gr.update(label=get_text(lang_value, "subtab.logs")),  # logs_tab
                gr.update(label=get_text(lang_value, "field.logs")),  # logs_console
                gr.update(label=get_text(lang_value, "subtab.artifacts")),  # artifacts_tab
                gr.update(
                    headers=get_columns(lang_value, "artifacts"),
                    label=get_text(lang_value, "df.artifacts"),
                ),  # artifacts_list
                gr.update(value=get_text(lang_value, "hint.artifacts")),  # artifacts_hint_md
                gr.update(label=get_text(lang_value, "subtab.recent")),  # recent_tab
                gr.update(
                    headers=get_columns(lang_value, "recent"),
                    label=get_text(lang_value, "df.recent"),
                ),  # recent_jobs_table
                gr.update(value=get_text(lang_value, "poll.note")),  # poll_note_md
            )

        # ==================== Event Bindings ====================

        lang_radio.change(
            fn=apply_language,
            inputs=lang_radio,
            outputs=[
                lang_state,
                lang_radio,
                title_md,
                submit_md,
                task_type_radio,
                model_path_txt,
                data_source_txt,
                output_dir_txt,
                hyperparams_accordion,
                conf_slider,
                device_txt,
                security_accordion,
                allowed_paths_txt,
                security_md,
                submit_btn,
                cancel_job_btn,
                status_tab,
                job_id_display,
                status_display,
                error_box,
                logs_tab,
                logs_console,
                artifacts_tab,
                artifacts_list,
                artifacts_hint_md,
                recent_tab,
                recent_jobs_table,
                poll_note_md,
            ],
        )

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
                lang_state,
            ],
            outputs=[job_id_display, submit_msg, poll_timer],
        ).then(
            fn=poll_handler,
            inputs=[job_id_display, lang_state],
            outputs=[
                status_display,
                error_box,
                banner_md,
                logs_console,
                artifacts_list,
                recent_jobs_table,
                poll_timer,
            ],
        )

        poll_timer.tick(
            fn=poll_handler,
            inputs=[job_id_display, lang_state],
            outputs=[
                status_display,
                error_box,
                banner_md,
                logs_console,
                artifacts_list,
                recent_jobs_table,
                poll_timer,
            ],
        )

        sync_timer.tick(
            fn=sync_handler,
            inputs=[job_id_display, lang_state],
            outputs=[
                status_display,
                error_box,
                banner_md,
                logs_console,
                artifacts_list,
                recent_jobs_table,
            ],
        )

        cancel_job_btn.click(
            fn=cancel_job_handler,
            inputs=[job_id_display, lang_state],
            outputs=[submit_msg, poll_timer],
        ).then(
            fn=poll_handler,
            inputs=[job_id_display, lang_state],
            outputs=[
                status_display,
                error_box,
                banner_md,
                logs_console,
                artifacts_list,
                recent_jobs_table,
                poll_timer,
            ],
        )

    return jobs_tab
