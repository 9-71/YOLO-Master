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

import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

from core.schema import JobRequest, JobStatus, SecurityConstraints, TaskType
from smoke.f1.dispatcher import JobDispatcherStateMachine
from smoke.f1.ui.i18n import DEFAULT_LANGUAGE, get_columns, get_text

#: Job states that still require high-frequency lifecycle polling.
ACTIVE_STATUSES = frozenset({"PENDING", "RUNNING"})
#: Backend error codes that map to visual security/validation user alerts.
SECURITY_ALERT_CODES = frozenset({"SEC_ERR_001", "PARAM_VALIDATION_FAILED"})
#: Fast lifecycle polling interval (seconds) while a job is active.
POLL_FAST_SECONDS = 1.0
#: Slow background sync interval (seconds) for recent jobs and final artifacts.
POLL_SLOW_SECONDS = 30.0
#: Image file extensions recognized by the artifact preview gallery.
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
#: Default form values per task type. Selecting a task_type repopulates model_path,
#: data_source and output_dir together (mirroring Inference Studio's dynamic weight
#: switching) so downstream handlers always receive engine-compatible inputs:
#: predict takes an image source while train/val require a dataset YAML. The per-field
#: 🔄 reset buttons restore the same preset values on demand without changing the task.
TASK_FORM_PRESETS: dict[str, dict[str, str]] = {
    "predict": {
        "model_path": "./ckpts/yolov8n.pt",
        "data_source": "ultralytics/assets/bus.jpg",
        "output_dir": "runs/predict",
    },
    "train": {
        "model_path": "./ckpts/yolov8n.pt",
        "data_source": "coco8.yaml",
        "output_dir": "runs/train",
    },
    "val": {
        "model_path": "./ckpts/yolov8n.pt",
        "data_source": "coco8.yaml",
        "output_dir": "runs/val",
    },
    "export": {
        "model_path": "./ckpts/yolov8n.pt",
        "data_source": "",
        "output_dir": "runs/export",
    },
    "diagnose": {
        "model_path": "",
        "data_source": "",
        "output_dir": "runs/diagnose",
    },
}


def _format_elapsed(created_at: str | None) -> str | None:
    """Return ``now - created_at`` as a ``"{seconds:.1f}s"`` string (live read).

    Args:
        created_at: Raw ISO 8601 creation timestamp (may be ``None``/malformed).

    Returns:
        str | None: The elapsed duration, or ``None`` when it cannot be parsed.
    """
    if not created_at:
        return None
    try:
        created = datetime.fromisoformat(created_at)
        return f"{(datetime.now(timezone.utc) - created).total_seconds():.1f}s"
    except (ValueError, TypeError):
        return None


def _compute_duration(status_str: str, created_at: str | None, completed_at: str | None = None) -> str | None:
    """Return a stable duration string, or ``None`` when it cannot be resolved.

    RUNNING keeps a live read (``now - created_at``). Non-RUNNING states are
    frozen to the exact ``completed_at - created_at`` total; a missing
    completion timestamp returns ``None`` so the caller can retain a previous
    valid value instead of falling back to ``"N/A"``.

    Args:
        status_str: Uppercase lifecycle status string.
        created_at: Raw ISO 8601 creation timestamp (may be ``None``/malformed).
        completed_at: Raw ISO 8601 completion timestamp (may be ``None``).

    Returns:
        str | None: A ``"{seconds:.1f}s"`` duration, or ``None`` when the exact
        total cannot be computed deterministically.
    """
    if status_str == "RUNNING":
        return _format_elapsed(created_at)
    if not created_at or not completed_at:
        return None
    try:
        created = datetime.fromisoformat(created_at)
        completed = datetime.fromisoformat(completed_at)
        return f"{(completed - created).total_seconds():.1f}s"
    except (ValueError, TypeError):
        return None


def _resolve_completion_time(job: Any) -> str | None:
    """Return the first available completion timestamp for a job.

    Completion time may live under different metadata fields depending on the
    backend that produced the job. Metadata fields are checked in order, then
    the failure error timestamp (``error.timestamp``) as a final fallback.

    Args:
        job: A ``JobRequest`` (or duck-typed equivalent).

    Returns:
        str | None: The completion timestamp, or ``None`` when absent.
    """
    metadata = getattr(job, "metadata", None)
    for meta_field in ("completed_at", "finished_at", "updated_at"):
        value = getattr(metadata, meta_field, None)
        if value:
            return value
    error = getattr(job, "error", None)
    timestamp = getattr(error, "timestamp", None)
    return timestamp if timestamp else None


#: Localized artifact-preview label resolved by the language broadcast. Kept
#: local (rather than in ``i18n.py``) so the shared i18n module stays untouched.
_ARTIFACT_PREVIEW_BASE: dict[str, str] = {"en": "Artifact Preview", "zh": "产物预览"}
#: Localized label for the multi-image selector dropdown.
_ARTIFACT_SELECTOR_LABEL: dict[str, str] = {"en": "Select Image Artifact", "zh": "选择预览图片"}
#: Localized labels for the prev/next paging buttons.
_ARTIFACT_PREV_LABEL: dict[str, str] = {"en": "◀ Prev", "zh": "◀ 上一张"}
_ARTIFACT_NEXT_LABEL: dict[str, str] = {"en": "Next ▶", "zh": "下一张 ▶"}
#: Localized label for the open-output-folder button.
_OPEN_FOLDER_LABEL: dict[str, str] = {"en": "📂 Open Folder", "zh": "📂 打开输出目录"}
#: Local CSS injected into the Jobs zone to hide the Gradio Dataframe
#: column-header options button (the three-dot "Open cell menu" trigger). The
#: tab's Dataframes are read-only monitors, so the built-in column sort/filter
#: menu — whose labels are not localized by this app's i18n layer — is hidden
#: entirely. ``.cell-menu-button`` is the stable semantic class Gradio applies
#: to that trigger (``aria-label="Open cell menu"``).
_DATAFRAME_HEADER_MENU_CSS: str = "<style>.cell-menu-button { display: none !important; }</style>"


def _artifact_preview_label(lang: str | None, filename: str | None) -> str:
    """Return the localized preview label, appending the current filename.

    Args:
        lang: ISO language code ("en" or "zh").
        filename: Base name of the currently displayed image (may be ``None``).

    Returns:
        str: ``"Artifact Preview"`` (or ``"产物预览"``), with ``": {filename}"``
        appended when a file is being shown.
    """
    base = _ARTIFACT_PREVIEW_BASE.get(lang or DEFAULT_LANGUAGE, _ARTIFACT_PREVIEW_BASE[DEFAULT_LANGUAGE])
    return f"{base}: {filename}" if filename else base


def _image_preview_update(image_artifacts: list[str], lang: str | None) -> Any:
    """Build the ``gr.Image`` update for the first image artifact.

    Args:
        image_artifacts: Absolute paths of image files for the job.
        lang: ISO language code used to localize the label.

    Returns:
        Any: A ``gr.update`` carrying the selected path and localized label.
    """
    path = image_artifacts[0] if image_artifacts else None
    filename = Path(path).name if path else None
    return gr.update(value=path, label=_artifact_preview_label(lang, filename))


def _artifact_selector_update(image_artifacts: list[str], lang: str | None) -> Any:
    """Build the ``gr.Dropdown`` update listing switchable image filenames.

    Args:
        image_artifacts: Absolute paths of image files for the job.
        lang: ISO language code used to localize the label.

    Returns:
        Any: A ``gr.update`` with the filename choices, selected value and
        visibility (shown only when more than one image exists).
    """
    names = [Path(p).name for p in image_artifacts]
    value = names[0] if names else None
    label = _ARTIFACT_SELECTOR_LABEL.get(lang or DEFAULT_LANGUAGE, _ARTIFACT_SELECTOR_LABEL[DEFAULT_LANGUAGE])
    return gr.update(choices=names, value=value, label=label, visible=len(names) > 1)


def _cycle_artifact(paths: list[str], selected: str | None, step: int) -> tuple[str | None, str | None]:
    """Return ``(filename, path)`` of the item ``step`` positions away (cyclic).

    Args:
        paths: Absolute image paths for the job.
        selected: Currently selected filename (may be ``None``).
        step: Offset to advance (+1 next, -1 prev); wraps around the list.

    Returns:
        tuple[str | None, str | None]: The selected filename and full path, or
        ``(None, None)`` when ``paths`` is empty.
    """
    if not paths:
        return None, None
    names = [Path(p).name for p in paths]
    idx = names.index(selected) if selected and selected in names else 0
    new_idx = (idx + step) % len(names)
    return names[new_idx], paths[new_idx]


class JobsManager:
    """Thread-safe job management with real-time state tracking."""

    def __init__(self, storage_path: str | None = None) -> None:
        """Initialize job manager; in-memory only unless storage_path is given.

        Args:
            storage_path: Optional JSON file path. When provided, job state is
                persisted on every mutation and restored on startup. When None
                (default), behavior is purely in-memory.
        """
        self.jobs: dict[str, JobRequest] = {}
        self.job_logs: dict[str, list[str]] = {}
        self._durations: dict[str, str] = {}
        self.lock = threading.Lock()
        self.dispatcher = JobDispatcherStateMachine()
        self._storage_path = Path(storage_path) if storage_path else None
        if self._storage_path is not None:
            self._load()

    def _save(self) -> None:
        """Persist jobs and logs to JSON atomically; no-op in memory-only mode.

        Writes to a temporary sibling file and atomically replaces the target
        via ``os.replace`` so a crash mid-write never leaves a corrupt state
        file. Persistence failures are swallowed and never block job flow.

        Callers MUST hold ``self.lock``; this method does not acquire it
        (``threading.Lock`` is not reentrant).
        """
        if self._storage_path is None:
            return
        payload = {
            "version": 1,
            "jobs": {jid: job.model_dump(mode="json") for jid, job in self.jobs.items()},
            "job_logs": self.job_logs,
        }
        tmp_path = self._storage_path.with_suffix(self._storage_path.suffix + ".tmp")
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, self._storage_path)
        except OSError:
            pass

    def _load(self) -> None:
        """Restore persisted state from the storage file, if present.

        Jobs left in PENDING/RUNNING are orphaned (their execution threads died
        with the previous process) and are healed to FAILED so the UI never
        shows them as eternally active. A missing or corrupt file silently
        degrades to an empty in-memory state.
        """
        if self._storage_path is None or not self._storage_path.is_file():
            return
        try:
            payload = json.loads(self._storage_path.read_text(encoding="utf-8"))
            for jid, raw in payload.get("jobs", {}).items():
                job = JobRequest.model_validate(raw)
                if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                    job.status = JobStatus.FAILED
                self.jobs[jid] = job
            self.job_logs.update(payload.get("job_logs", {}))
        except (OSError, ValueError):
            pass

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
            task_type: Task type (predict, train, val, export, diagnose)
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
        if task_type in ["predict", "train", "val"]:
            params["model_path"] = model_path
            params["data_source"] = data_source
            params["conf"] = conf
            params["device"] = device
        elif task_type == "export":
            params["model_path"] = model_path
            params["format"] = "onnx"  # Default export format

        # Default train parameters: epochs and imgsz are required by TrainHandler
        if task_type == "train":
            params.setdefault("epochs", 1)
            params.setdefault("imgsz", 640)
        # Default val parameter: imgsz for ValHandler
        if task_type == "val":
            params.setdefault("imgsz", 640)

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

        # The backend Metadata default is deep-copied from a class-definition-time
        # instance, so every job would otherwise share one created_at. Stamp each
        # submission with its own actual creation time (used by the Recent Jobs
        # timestamp column and its newest-first ordering).
        job_request.metadata.created_at = datetime.now(timezone.utc).isoformat()

        with self.lock:
            self.jobs[job_id] = job_request
            self.job_logs[job_id] = [f"[{datetime.now(timezone.utc).isoformat()}] Job {job_id} submitted"]
            self._save()

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
                self._save()

            if result.status == JobStatus.COMPLETED:
                self._append_log(
                    job_id,
                    f"[{datetime.now(timezone.utc).isoformat()}] ✅ Completed. "
                    f"Artifacts: {len(result.output.artifacts)}",
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
                    self._save()

    def _append_log(self, job_id: str, message: str) -> None:
        """Append log message to job log buffer."""
        with self.lock:
            if job_id not in self.job_logs:
                self.job_logs[job_id] = []
            self.job_logs[job_id].append(message)
            self._save()

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

            status = job.status.value.upper()
            created_at = job.metadata.created_at if hasattr(job.metadata, "created_at") else None
            completed_at = _resolve_completion_time(job)
            output_duration = getattr(job.output, "duration", None)

            # Prefer an explicitly recorded duration, then an exact computed total.
            duration = output_duration or _compute_duration(status, created_at, completed_at)

            if status == "RUNNING":
                # Cache the live reading so a later terminal transition can freeze on it.
                if duration:
                    self._durations[job_id] = duration
            elif status not in ACTIVE_STATUSES:
                # Terminal states must freeze instead of re-ticking across polls.
                if not duration:
                    duration = self._durations.get(job_id)
                if not duration:
                    duration = _format_elapsed(created_at)
                if duration:
                    self._durations[job_id] = duration

            return {
                "status": status,
                "duration": duration if duration else "N/A",
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

    def get_job_image_artifacts(self, job_id: str) -> list[str]:
        """Get image artifact paths for a completed job.

        Args:
            job_id: Job identifier

        Returns:
            List of absolute image file paths under the job's output_dir.
            Returns an empty list when the job is not found or not completed.
        """
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return []
            return get_job_image_artifacts(job)

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
            self._save()

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


def _is_valid_image_under(candidate: Path, root_dir: Path) -> str | None:
    """Resolve ``candidate`` and return its absolute path if it is a real image under ``root_dir``.

    Args:
        candidate: The path to validate (may be relative or absolute).
        root_dir: The directory the resolved file must live under.

    Returns:
        str | None: The absolute path string when the file exists, has an
        image extension, and is a descendant of ``root_dir``; ``None``
        otherwise.
    """
    if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    try:
        resolved = candidate.resolve()
        if not resolved.is_file():
            return None
        resolved.relative_to(root_dir)  # raises ValueError if outside root
        return str(resolved)
    except (OSError, ValueError):
        return None


def _is_historical_task_dir(name: str, task_type: str, job_id: str) -> bool:
    """Return True when ``name`` is a dated run directory of another job.

    Dated run directories follow the ``{task_type}_YYYY...`` shape (e.g.
    ``predict_20260824_...`` or ``train_202...``). When such a directory does not
    reference the current ``job_id`` it belongs to a historical job and must be
    skipped so its artifacts never leak into this job's preview gallery.

    Args:
        name: Basename of a candidate first-level subdirectory.
        task_type: The current job's task type value (e.g. "predict").
        job_id: The current job's identifier.

    Returns:
        bool: True when ``name`` is a dated ``{task_type}_YYYY...`` directory not
        belonging to the current job.
    """
    if job_id and job_id in name:
        return False
    prefix = f"{task_type}_"
    if not name.startswith(prefix):
        return False
    remainder = name[len(prefix) :]
    return len(remainder) >= 4 and remainder[:4].isdigit()


def get_job_image_artifacts(job: JobRequest) -> list[str]:
    """Return image artifacts for a completed job, scoped strictly to that job.

    Three-tier lookup ordered from the most precise source to a conservative
    fallback that still avoids pulling historical images out of a shared
    top-level output directory (e.g. ``runs/predict``):

    1. **Artifacts-first** — image entries in ``job.output.artifacts`` are
       resolved against ``output_dir`` and validated. If any survive, return
       them sorted by filename.
    2. **Job-subdirectory** — a subdirectory of ``output_dir`` whose name
       contains ``job.job_id`` is scanned recursively and returned.
    3. **Root / one-level fallback** — first-level image files plus images from
       ``batch_*`` or task-related subdirectories are collected, excluding
       dated historical job directories (``predict_202...``, ``train_202...``).

    Args:
        job: The job whose image artifacts should be collected.

    Returns:
        list[str]: Absolute paths of image files belonging to this job, sorted
        by filename in ascending order.  Returns an empty list when the job is
        not COMPLETED, ``output_dir`` is invalid, or no matching images exist.

    Example:
        >>> from core.schema import JobRequest, JobStatus, TaskType
        >>> job = JobRequest(job_id="t", task_type=TaskType.PREDICT, status=JobStatus.PENDING)
        >>> get_job_image_artifacts(job)
        []
    """
    if job.status != JobStatus.COMPLETED:
        return []

    output_dir_str = job.output.output_dir if hasattr(job.output, "output_dir") else ""
    if not output_dir_str:
        return []

    try:
        output_dir = Path(output_dir_str).resolve()
    except (OSError, ValueError):
        return []

    if not output_dir.is_dir():
        return []

    task_type = job.task_type.value

    # ---- Tier 1: use the job's own artifact manifest (precise, no history bleed) ----
    artifacts = getattr(job.output, "artifacts", None)
    if artifacts:
        images: list[str] = []
        for entry in artifacts:
            candidate = Path(entry)
            # If the entry is relative, resolve it against output_dir.
            if not candidate.is_absolute():
                candidate = output_dir / candidate
            result = _is_valid_image_under(candidate, output_dir)
            if result:
                images.append(result)
        if images:
            images.sort()
            return images

    # ---- Tier 2: scan only the job's own subdirectory (name contains job_id) ----
    try:
        for entry in output_dir.iterdir():
            if entry.is_dir() and job.job_id in entry.name:
                images = []
                for path in entry.rglob("*"):
                    if not path.is_file():
                        continue
                    result = _is_valid_image_under(path, output_dir)
                    if result:
                        images.append(result)
                images.sort()
                return images
    except OSError:
        return []

    # ---- Tier 3: root files + conservative one-level subdirectory fallback ----
    # First-level image files and images under ``batch_*`` / task-related
    # subdirectories are collected, but dated historical job directories of the
    # same task type are excluded so another job's artifacts never leak in.
    try:
        entries = list(output_dir.iterdir())
    except OSError:
        return []

    seen: set[str] = set()
    for entry in entries:
        if entry.is_file():
            result = _is_valid_image_under(entry, output_dir)
            if result:
                seen.add(result)
        elif entry.is_dir():
            name = entry.name
            if _is_historical_task_dir(name, task_type, job.job_id):
                continue
            if name.startswith(("batch_", f"{task_type}_")) or job.job_id in name:
                try:
                    for path in entry.rglob("*"):
                        if not path.is_file():
                            continue
                        result = _is_valid_image_under(path, output_dir)
                        if result:
                            seen.add(result)
                except OSError:
                    continue

    return sorted(seen)


def format_created_at(created_at: str | None) -> str:
    """Format an ISO 8601 UTC ``created_at`` as local time for the Recent Jobs table.

    Backend timestamps remain raw ISO UTC strings under the hood; this helper only
    formats them for dataframe display.

    Args:
        created_at: Raw ISO 8601 UTC timestamp from ``JobRequest.metadata.created_at``.

    Returns:
        str: Local time as ``YYYY-MM-DD HH:MM:SS``; ``"-"`` when the value is empty
        or missing, and the raw string when it cannot be parsed.

    Example:
        >>> format_created_at("")
        '-'
        >>> format_created_at(None)
        '-'
        >>> format_created_at("not-a-timestamp")
        'not-a-timestamp'
        >>> from datetime import datetime
        >>> now_local = datetime.now().astimezone()
        >>> format_created_at(now_local.isoformat()) == now_local.strftime("%Y-%m-%d %H:%M:%S")
        True
    """
    if not created_at:
        return "-"
    try:
        return datetime.fromisoformat(created_at).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(created_at)


def recent_jobs_rows(jobs_manager: JobsManager, limit: int = 20) -> list[list[str]]:
    """Format the recent-jobs listing into Recent Jobs dataframe rows.

    Each row is ``[job_id, task_type, status, local_time]``, newest first, with
    raw ISO UTC timestamps localized via :func:`format_created_at`. Extracted
    from :func:`compute_poll_state` so the Recent Jobs table can be pre-populated
    at construction time (its initial ``value``) and stay byte-for-byte
    consistent with every polling refresh.

    Args:
        jobs_manager: JobsManager instance (or duck-typed equivalent).
        limit: Maximum number of recent jobs to include.

    Returns:
        list[list[str]]: Formatted rows for the Recent Jobs ``gr.Dataframe``.

    Example:
        >>> manager = JobsManager()
        >>> recent_jobs_rows(manager)
        []
    """
    return [
        [j["job_id"], j["task_type"], j["status"], format_created_at(j["created_at"])]
        for j in jobs_manager.list_recent_jobs(limit=limit)
    ]


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
        >>> "任务失败" in banner
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
        status: Canonical status dict (backend keys/values only) rendered by the Status Monitor JSON panel.
        error_text: Diagnostics text for the error box (empty when healthy).
        banner: Markdown alert banner (empty when the job has no failure).
        logs: Formatted execution logs.
        artifacts: Rows for the artifacts dataframe (filename, path).
        image_artifacts: Absolute paths of image files under output_dir for the preview image.
        recent: Rows for the recent-jobs dataframe (timestamps formatted as local time).
        keep_polling: True while the job is active; False once terminal (deactivates the fast timer).
    """

    status: dict[str, Any]
    error_text: str = ""
    banner: str = ""
    logs: str = ""
    artifacts: list[list[str]] = field(default_factory=list)
    image_artifacts: list[str] = field(default_factory=list)
    recent: list[list[str]] = field(default_factory=list)
    keep_polling: bool = False


def compute_poll_state(jobs_manager: JobsManager, job_id: str, lang: str = DEFAULT_LANGUAGE) -> PollState:
    """Compute the complete polling snapshot for one job in the given language.

    Pure presentation logic over the JobsManager backend API: the backend responses
    are never modified, so backend-level test assertions remain valid. The Status
    Monitor JSON payload carries canonical backend keys/values only — localized
    display text is confined to banners, toasts and column headers. Recent-jobs
    timestamps are converted to local time for display while the backend keeps
    raw ISO UTC strings. In every state — including the idle "no selection"
    branch — the returned ``recent`` field is populated via
    :func:`recent_jobs_rows`, so polling never clears the Recent Jobs table.

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
        >>> state.status
        {'status': 'NO_SELECTION'}
    """
    if not job_id:
        # Idle state: there is no selected job to monitor, but the Recent Jobs
        # table must still reflect the persisted history. Populate ``recent``
        # explicitly so the always-on slow sync timer (and any poll tick) never
        # overwrites the table with an empty list and blanks out the rows that
        # were pre-populated on first render.
        return PollState(
            status={"status": "NO_SELECTION"},
            recent=recent_jobs_rows(jobs_manager, limit=20),
        )

    raw = jobs_manager.get_job_status(job_id)
    status_str = raw.get("status", "NOT_FOUND")

    status = {
        "job_id": job_id,
        "status": status_str,
        "duration": raw.get("duration"),
        "error_code": raw.get("error_code"),
        "error_message": raw.get("error_message"),
        "artifact_count": raw.get("artifact_count"),
    }

    # Live read-seconds apply strictly to RUNNING jobs. Terminal statuses must
    # freeze to a previously computed value when no completion timestamp is
    # available. NOT_FOUND has no job record and is left untouched so its empty
    # duration payload remains ``None``.
    if status_str != "NOT_FOUND" and not status["duration"]:
        created_at: str | None = None
        completed_at: str | None = None
        jobs = getattr(jobs_manager, "jobs", None)
        job = jobs.get(job_id) if isinstance(jobs, dict) else None
        if job is not None:
            created_at = getattr(job.metadata, "created_at", None)
            completed_at = _resolve_completion_time(job)
        resolved = _compute_duration(status_str, created_at, completed_at)
        if resolved:
            status["duration"] = resolved

    error_text = ""
    banner = ""
    if status_str == "FAILED":
        code = raw.get("error_code") or "UNKNOWN"
        error_text = f"[{code}] {raw.get('error_message') or ''}"
        banner = alert_banner(lang, code, raw.get("error_message"))

    logs = jobs_manager.get_job_logs(job_id)
    artifacts = [[name, path] for name, path in jobs_manager.get_job_artifacts(job_id)]
    # Safe reflection: test Mocks may not implement get_job_image_artifacts.
    getter = getattr(jobs_manager, "get_job_image_artifacts", None)
    image_artifacts = getter(job_id) if callable(getter) else []
    recent = recent_jobs_rows(jobs_manager, limit=20)

    return PollState(
        status=status,
        error_text=error_text,
        banner=banner,
        logs=logs,
        artifacts=artifacts,
        image_artifacts=image_artifacts,
        recent=recent,
        keep_polling=not is_terminal_status(status_str),
    )


def jobs_tab_language_updates(lang_value: str) -> tuple[Any, ...]:
    """Build the localized relabel payload covering every Jobs Tab component.

    Pure presentation helper used by the top-level language broadcast wired in
    app.py (the Jobs zone no longer has its own language listener). The payload
    mirrors the Jobs zone language outputs 1-to-1: element at position ``i``
    updates the component at position ``i`` of the zone output tuple (also
    exposed as ``create_jobs_tab(...)._language_outputs``).

    Args:
        lang_value: ISO language code ("en" or "zh"); unknown codes fall back
            to English via the i18n layer.

    Returns:
        tuple[Any, ...]: 33-element payload: the raw language value first (for
            the shared language State), then one ``gr.update`` per localizable
            component.
    """
    return (
        lang_value,  # lang_state
        gr.update(value=f"# {get_text(lang_value, 'tab.title')}"),  # title_md
        gr.update(value=f"### {get_text(lang_value, 'panel.submit')}"),  # submit_md
        gr.update(label=get_text(lang_value, "field.task_type")),  # task_type_radio
        gr.update(
            label=get_text(lang_value, "field.model_path"),
            placeholder=get_text(lang_value, "field.model_path.placeholder"),
        ),  # model_path_txt
        gr.update(value=get_text(lang_value, "button.reset_model")),  # model_path_reset_btn
        gr.update(
            label=get_text(lang_value, "field.data_source"),
            placeholder=get_text(lang_value, "field.data_source.placeholder"),
        ),  # data_source_txt
        gr.update(value=get_text(lang_value, "button.reset_data")),  # data_source_reset_btn
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
        gr.update(value=_OPEN_FOLDER_LABEL.get(lang_value, _OPEN_FOLDER_LABEL[DEFAULT_LANGUAGE])),  # open_folder_btn
        gr.update(
            headers=get_columns(lang_value, "artifacts"),
            label=get_text(lang_value, "df.artifacts"),
        ),  # artifacts_list
        gr.update(value=_ARTIFACT_PREV_LABEL.get(lang_value, _ARTIFACT_PREV_LABEL[DEFAULT_LANGUAGE])),  # prev_btn
        gr.update(
            label=_ARTIFACT_SELECTOR_LABEL.get(lang_value, _ARTIFACT_SELECTOR_LABEL[DEFAULT_LANGUAGE])
        ),  # artifact_selector
        gr.update(value=_ARTIFACT_NEXT_LABEL.get(lang_value, _ARTIFACT_NEXT_LABEL[DEFAULT_LANGUAGE])),  # next_btn
        gr.update(
            label=_ARTIFACT_PREVIEW_BASE.get(lang_value, _ARTIFACT_PREVIEW_BASE[DEFAULT_LANGUAGE])
        ),  # artifacts_image
        gr.update(label=get_text(lang_value, "subtab.recent")),  # recent_tab
        gr.update(
            headers=get_columns(lang_value, "recent"),
            label=get_text(lang_value, "df.recent"),
        ),  # recent_jobs_table
        gr.update(value=get_text(lang_value, "poll.note")),  # poll_note_md
    )


def create_jobs_tab(jobs_manager: JobsManager, lang: str = DEFAULT_LANGUAGE) -> gr.Blocks:
    """Create the Jobs Tab UI with adaptive polling, security alerts and i18n.

    Args:
        jobs_manager: Application-level JobsManager singleton shared across tabs.
        lang: Initial UI language ("en" or "zh"); the host app (app.py) owns the
            language selector and relabels this zone via the language broadcast.

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
        - The first tick observing SEC_ERR_001 / PARAM_VALIDATION_FAILED emits a
          one-shot gr.Warning toast; a persistent localized banner stays visible in the
          Status Monitor tab.

    Language-state lifting (unidirectional):
        Language selection is owned exclusively by the host app (app.py), whose
        top-level selector is the single source of truth. The tab itself never
        listens for language changes. Exposed broadcast handles:

        - ``jobs_tab._language_state``: the tab-local language State consumed by
          the polling handlers to localize status panels.
        - ``jobs_tab._language_outputs``: 33-element tuple of every localizable
          component, position-aligned with :func:`jobs_tab_language_updates`.

        The host relabels this zone with one explicit output list built from
        ``jobs_tab_language_updates(lang)`` — never through chained events.
    """
    # Job IDs whose security alert has already been toasted (one-shot warning guard).
    _security_warned: set[str] = set()

    with gr.Blocks() as jobs_tab:
        lang_state = gr.State(lang)
        title_md = gr.Markdown(f"# {get_text(lang, 'tab.title')}")
        # Hide the Dataframe column-header options (three-dot) menu button across
        # the read-only monitoring tables in this zone (see module constant).
        gr.HTML(_DATAFRAME_HEADER_MENU_CSS)

        with gr.Row(equal_height=False):
            # ==================== Left Panel: Job Submission ====================
            with gr.Column(scale=1, variant="panel"):
                submit_md = gr.Markdown(f"### {get_text(lang, 'panel.submit')}")

                # Task type selector
                task_type_radio = gr.Radio(
                    choices=["predict", "train", "val", "export", "diagnose"],
                    value="predict",
                    label=get_text(lang, "field.task_type"),
                )

                # Dynamic input parameters form
                with gr.Group():
                    model_path_txt = gr.Textbox(
                        value="./ckpts/yolov8n.pt",
                        label=get_text(lang, "field.model_path"),
                        placeholder=get_text(lang, "field.model_path.placeholder"),
                    )
                    model_path_reset_btn = gr.Button(
                        get_text(lang, "button.reset_model"), size="sm", variant="secondary"
                    )
                    data_source_txt = gr.Textbox(
                        value="ultralytics/assets/bus.jpg",
                        label=get_text(lang, "field.data_source"),
                        placeholder=get_text(lang, "field.data_source.placeholder"),
                    )
                    data_source_reset_btn = gr.Button(
                        get_text(lang, "button.reset_data"), size="sm", variant="secondary"
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
                    open_folder_btn = gr.Button(_OPEN_FOLDER_LABEL["en"], size="sm")
                    artifacts_list = gr.Dataframe(
                        headers=get_columns(lang, "artifacts"),
                        label=get_text(lang, "df.artifacts"),
                        interactive=False,
                    )
                    artifacts_image = gr.Image(
                        label=_ARTIFACT_PREVIEW_BASE["en"],
                        interactive=False,
                        type="filepath",
                        height=420,
                        visible=True,
                    )
                    with gr.Row(equal_height=True):
                        prev_btn = gr.Button(
                            _ARTIFACT_PREV_LABEL["en"],
                            size="sm",
                            scale=1,
                            min_width=80,
                            visible=False,
                        )
                        artifact_selector = gr.Dropdown(
                            label=_ARTIFACT_SELECTOR_LABEL["en"],
                            choices=[],
                            value=None,
                            interactive=True,
                            show_label=False,
                            container=False,
                            scale=6,
                            visible=False,
                        )
                        next_btn = gr.Button(
                            _ARTIFACT_NEXT_LABEL["en"],
                            size="sm",
                            scale=1,
                            min_width=80,
                            visible=False,
                        )

                # Tab 4: Recent Jobs
                with gr.TabItem(get_text(lang, "subtab.recent")) as recent_tab:
                    recent_jobs_table = gr.Dataframe(
                        headers=get_columns(lang, "recent"),
                        label=get_text(lang, "df.recent"),
                        value=recent_jobs_rows(jobs_manager, limit=20),
                        interactive=False,
                    )
                    poll_note_md = gr.Markdown(get_text(lang, "poll.note"))

        # Adaptive timers: fast lifecycle poll (activated on submit, self-deactivates
        # on terminal state) and slow always-on background sync.
        poll_timer = gr.Timer(POLL_FAST_SECONDS, active=False)
        sync_timer = gr.Timer(POLL_SLOW_SECONDS)

        # Localizable outputs relabeled by the host language broadcast: exactly 33
        # distinct components, position-aligned with the jobs_tab_language_updates()
        # payload. Exposed on the returned Blocks as ``_language_outputs`` so the host
        # app can broadcast a language change into this zone with one explicit output
        # list.
        language_outputs: tuple[gr.Component, ...] = (
            lang_state,
            title_md,
            submit_md,
            task_type_radio,
            model_path_txt,
            model_path_reset_btn,
            data_source_txt,
            data_source_reset_btn,
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
            open_folder_btn,
            artifacts_list,
            prev_btn,
            artifact_selector,
            next_btn,
            artifacts_image,
            recent_tab,
            recent_jobs_table,
            poll_note_md,
        )

        # ==================== Event Handlers ====================

        def poll_snapshot(job_id: str, lang_value: str) -> PollState:
            """Compute one snapshot, emitting a one-shot warning toast for new security alerts.

            The warning is queued as a toast rather than raised: raising terminates the
            event, whereas the snapshot below must still reach the monitoring panels.
            """
            if job_id:
                raw = jobs_manager.get_job_status(job_id)
                code = raw.get("error_code")
                if code in SECURITY_ALERT_CODES and job_id not in _security_warned:
                    _security_warned.add(job_id)
                    gr.Warning(security_alert_toast(lang_value, code))
            return compute_poll_state(jobs_manager, job_id, lang_value)

        def poll_handler(job_id: str, lang_value: str) -> tuple:
            """Fast lifecycle poll: refresh every panel and deactivate on terminal state."""
            state = poll_snapshot(job_id, lang_value)
            nav_visible = len(state.image_artifacts) > 1
            return (
                state.status,
                state.error_text,
                state.banner,
                state.logs,
                state.artifacts,
                gr.update(visible=nav_visible),
                _artifact_selector_update(state.image_artifacts, lang_value),
                gr.update(visible=nav_visible),
                _image_preview_update(state.image_artifacts, lang_value),
                state.recent,
                gr.update(active=state.keep_polling),
            )

        def sync_handler(job_id: str, lang_value: str) -> tuple:
            """Slow background sync: refresh panels without touching the fast timer."""
            state = poll_snapshot(job_id, lang_value)
            nav_visible = len(state.image_artifacts) > 1
            return (
                state.status,
                state.error_text,
                state.banner,
                state.logs,
                state.artifacts,
                gr.update(visible=nav_visible),
                _artifact_selector_update(state.image_artifacts, lang_value),
                gr.update(visible=nav_visible),
                _image_preview_update(state.image_artifacts, lang_value),
                state.recent,
            )

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
            """Request cancellation, surfacing localized warnings for invalid states.

            Invalid requests (no selection, unknown job, terminal state) queue a
            one-shot gr.Warning toast and return a fallback update: the localized
            warning in the message panel and a deactivated fast poll timer.
            """
            if not job_id:
                message = get_text(lang_value, "msg.no_job_selected")
                gr.Warning(message)
                return message, gr.update(active=False)
            raw = jobs_manager.get_job_status(job_id)
            if raw.get("status") == "NOT_FOUND":
                message = get_text(lang_value, "msg.job_not_found")
                gr.Warning(message)
                return message, gr.update(active=False)
            if is_terminal_status(raw.get("status", "")):
                message = get_text(lang_value, "msg.terminal_state").format(status=raw.get("status", ""))
                gr.Warning(message)
                return message, gr.update(active=False)

            jobs_manager.cancel_job(job_id)
            return get_text(lang_value, "msg.cancel_requested").format(job_id=job_id), gr.update(active=True)

        # ==================== Event Bindings ====================

        # NOTE: this zone deliberately has no language listener. Language changes
        # are broadcast by the host app (app.py) through
        # jobs_tab_language_updates() -> _language_outputs; binding a change
        # listener here would let the host's programmatic writes re-trigger this
        # zone and form a bidirectional event loop.

        def _on_task_type_change(task_type: str) -> tuple[Any, Any, Any]:
            """Repopulate output_dir, data_source and model_path for the selected task type.

            train/val need a dataset YAML (coco8.yaml) while predict needs an image
            source; switching tasks updates the whole form at once so stale values
            from the previous task (e.g. data_source=bus.jpg) never reach the engine.
            """
            preset = TASK_FORM_PRESETS.get(task_type, TASK_FORM_PRESETS["predict"])
            return (
                gr.update(value=preset["output_dir"]),
                gr.update(value=preset["data_source"]),
                gr.update(value=preset["model_path"]),
            )

        task_type_radio.change(
            fn=_on_task_type_change,
            inputs=task_type_radio,
            outputs=[output_dir_txt, data_source_txt, model_path_txt],
        )

        def _make_form_reset(field_key: str):
            """Build a click handler restoring one form field from the active task preset.

            The handler reads the current task_type and writes the preset value for
            ``field_key`` back into its own textbox. It only touches the form fields —
            never language state or language outputs — so it cannot disturb the
            unidirectional language broadcast owned by the host app (app.py).

            Args:
                field_key: Key into TASK_FORM_PRESETS ("model_path" or "data_source").

            Returns:
                Callable (task_type: str) -> gr.update restoring the preset value.
            """

            def _reset_to_preset(task_type: str) -> Any:
                preset = TASK_FORM_PRESETS.get(task_type, TASK_FORM_PRESETS["predict"])
                return gr.update(value=preset[field_key])

            return _reset_to_preset

        model_path_reset_btn.click(
            fn=_make_form_reset("model_path"),
            inputs=task_type_radio,
            outputs=model_path_txt,
        )
        data_source_reset_btn.click(
            fn=_make_form_reset("data_source"),
            inputs=task_type_radio,
            outputs=data_source_txt,
        )

        def on_artifact_select(selected: str, job_id: str, lang_value: str) -> Any:
            """Switch the preview image when the user picks a different artifact.

            Args:
                selected: Selected image filename (or ``None`` when cleared).
                job_id: Selected job identifier.
                lang_value: ISO language code used to localize the label.

            Returns:
                Any: A ``gr.update`` with the matched path and localized label.
            """
            getter = getattr(jobs_manager, "get_job_image_artifacts", None)
            paths = getter(job_id) if callable(getter) else []
            match = next((p for p in paths if selected and Path(p).name == selected), None)
            filename = Path(match).name if match else None
            return gr.update(value=match, label=_artifact_preview_label(lang_value, filename))

        artifact_selector.change(
            fn=on_artifact_select,
            inputs=[artifact_selector, job_id_display, lang_state],
            outputs=[artifacts_image],
        )

        def _make_artifact_step(step: int):
            """Return a handler advancing the preview by ``step`` (cyclic).

            Args:
                step: Offset to advance (+1 next, -1 prev).
            """

            def _step(selected: str | None, job_id: str, lang_value: str) -> tuple[Any, Any]:
                getter = getattr(jobs_manager, "get_job_image_artifacts", None)
                paths = getter(job_id) if callable(getter) else []
                filename, path = _cycle_artifact(paths, selected, step)
                return (
                    gr.update(value=filename),
                    gr.update(value=path, label=_artifact_preview_label(lang_value, filename)),
                )

            return _step

        prev_btn.click(
            fn=_make_artifact_step(-1),
            inputs=[artifact_selector, job_id_display, lang_state],
            outputs=[artifact_selector, artifacts_image],
        )
        next_btn.click(
            fn=_make_artifact_step(1),
            inputs=[artifact_selector, job_id_display, lang_state],
            outputs=[artifact_selector, artifacts_image],
        )

        def on_artifact_table_select(evt: gr.SelectData, job_id: str, lang_value: str) -> tuple[Any, Any]:
            """Preview an image row selected in the artifacts table.

            Non-image rows (e.g. ``.pt``/``.csv``) are ignored so the current
            preview stays untouched. The selected image also syncs the selector.

            Args:
                evt: Gradio selection event carrying the clicked row index.
                job_id: Selected job identifier.
                lang_value: ISO language code used to localize the label.

            Returns:
                tuple[Any, Any]: Updates for the selector value and the preview
                image (path + localized label); empty updates when ignored.
            """
            row = evt.index[0] if evt.index else -1
            getter = getattr(jobs_manager, "get_job_artifacts", None)
            artifacts = getter(job_id) if callable(getter) else []
            if row < 0 or row >= len(artifacts):
                return gr.update(), gr.update()
            filename, path = artifacts[row]
            if Path(path).suffix.lower() not in IMAGE_EXTENSIONS:
                if lang_value == "zh":
                    gr.Info(f"'{filename}' 不是图片文件，无法预览。")
                else:
                    gr.Info(f"'{filename}' is not an image file and cannot be previewed.")
                return gr.update(), gr.update()
            return (
                gr.update(value=filename),
                gr.update(value=path, label=_artifact_preview_label(lang_value, filename)),
            )

        artifacts_list.select(
            fn=on_artifact_table_select,
            inputs=[job_id_display, lang_state],
            outputs=[artifact_selector, artifacts_image],
        )

        def open_output_folder(job_id: str, lang_value: str) -> None:
            """Open the current job's specific output folder in the OS file manager.

            Prefers the parent directory of the first recorded artifact, otherwise
            falls back to the ``output_dir`` subdirectory whose name contains the
            ``job_id``. Windows uses ``os.startfile``, macOS uses ``open`` and
            Linux uses ``xdg-open``; a missing folder surfaces a localized warning.
            """

            def _warn() -> None:
                if lang_value == "zh":
                    gr.Warning("输出目录尚不存在。")
                else:
                    gr.Warning("Output directory does not exist.")

            jobs = getattr(jobs_manager, "jobs", None)
            job = jobs.get(job_id) if isinstance(jobs, dict) and job_id else None
            if job is None:
                _warn()
                return

            output_dir_str = getattr(job.output, "output_dir", None)
            output_dir = Path(output_dir_str).resolve() if output_dir_str else None

            # Walk every artifact's parent chain to find the directory whose
            # name contains this job's id (the job-specific root folder).
            target: Path | None = None
            artifacts = getattr(job.output, "artifacts", None) or []
            for artifact in artifacts:
                if not artifact:
                    continue
                first = Path(artifact)
                if not first.is_absolute() and output_dir is not None:
                    first = output_dir / first
                for parent in first.parents:
                    if job.job_id in parent.name:
                        target = parent
                        break
                if target is not None:
                    break

            # No artifact manifest (or no job-id ancestor found): fall back to a
            # first-level ``output_dir`` subdirectory named after the job.
            if target is None and output_dir is not None:
                try:
                    for entry in output_dir.iterdir():
                        if entry.is_dir() and job.job_id in entry.name:
                            target = entry
                            break
                except OSError:
                    target = None

            if target is None or not target.is_dir():
                _warn()
                return

            try:
                if os.name == "nt":
                    os.startfile(str(target))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(target)])
                else:
                    subprocess.Popen(["xdg-open", str(target)])
            except OSError:
                _warn()

        open_folder_btn.click(
            fn=open_output_folder,
            inputs=[job_id_display, lang_state],
            outputs=[],
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
                prev_btn,
                artifact_selector,
                next_btn,
                artifacts_image,
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
                prev_btn,
                artifact_selector,
                next_btn,
                artifacts_image,
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
                prev_btn,
                artifact_selector,
                next_btn,
                artifacts_image,
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
                prev_btn,
                artifact_selector,
                next_btn,
                artifacts_image,
                recent_jobs_table,
                poll_timer,
            ],
        )

    # Language-state lifting handles for the host app (app.py): the top level owns
    # the language choice and broadcasts relabels into this zone. The language state
    # is consumed by polling handlers to localize status panels.
    jobs_tab._language_state = lang_state
    jobs_tab._language_outputs = language_outputs

    return jobs_tab
