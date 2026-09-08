"""Headless job-management core for the YOLO-Master F1 task platform.

This module owns the ``JobsManager`` backend plus the thread-safe job, log and
artifact helpers it depends on. It is intentionally free of any Gradio (or
other UI-framework) import: P2 (Standalone FastAPI Engine & Decoupled
Architecture) requires the job engine to be importable and executable
headlessly, so the FastAPI layer (``api.v1.jobs``) and the Gradio layer
(``f1.ui.jobs_tab``) both consume this module instead of each other.

History: the code below originally lived in ``f1.ui.jobs_tab`` (P0/P1). P2
relocated it here verbatim (plus the API-facing ``submit_job_request``,
``get_job`` and ``get_job_log_lines`` additions) so that importing the job
engine never requires Gradio UI state; ``f1.ui.jobs_tab`` re-exports the public
names for backward compatibility.

Security invariants (unchanged from the UI path):

    - ``allow_shell=False`` and ``path_whitelisted=True`` are forced on every
      submission, regardless of the caller's payload (fail-closed).
    - ``output.artifacts`` is reset server-side at submission; only the
      dispatcher may populate it after a successful execution.
    - All log lines are appended via ``JobRequest.append_log``, which routes
      every line through :func:`core.security.sanitize_log_text`.

Example:
    >>> from core.schema import JobRequest, TaskType
    >>> manager = JobsManager()
    >>> manager.get_job("missing-job") is None
    True
    >>> manager.list_recent_jobs()
    []
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.schema import JobRequest, JobStatus, SecurityConstraints, TaskType
from f1.dispatcher import JobDispatcherStateMachine

#: Job states that still require high-frequency lifecycle polling.
ACTIVE_STATUSES = frozenset({"PENDING", "RUNNING"})
#: Image file extensions recognized by the artifact preview gallery.
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

__all__ = [
    "ACTIVE_STATUSES",
    "IMAGE_EXTENSIONS",
    "JobsManager",
    "get_job_image_artifacts",
    "is_terminal_status",
]


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

        self.submit_job_request(job_request)

        return job_id, f"✅ Job {job_id} submitted successfully"

    def submit_job_request(self, request: JobRequest) -> JobRequest:
        """Submit a fully-constructed ``JobRequest`` for background execution.

        This is the canonical headless submission API consumed by the FastAPI
        engine (``POST /api/v1/jobs``). The UI path :meth:`submit_job` builds
        its request and delegates here, so both surfaces share one execution
        pipeline.

        Server-side normalization (fail-closed, applied before registration):

        - ``status`` is reset to ``PENDING``, ``error`` cleared, ``logs``
          emptied and ``runtime_tracking.cancel_requested`` reset to ``False``:
          a fresh submission always starts a fresh lifecycle.
        - ``security_constraints.allow_shell`` is forced to ``False`` and
          ``path_whitelisted`` forced to ``True``, regardless of the caller's
          payload.
        - ``security_constraints.allowed_paths`` gains the job's ``output_dir``
          (dynamic whitelist), mirroring the UI submission path.
        - ``output.artifacts`` is reset to ``[]``: only the dispatcher may
          populate the manifest after a successful execution, so clients can
          never pre-load arbitrary files into the artifact delivery routes.
        - Engine-required defaults are injected for ``train`` (``epochs=1``,
          ``imgsz=640``) and ``val`` (``imgsz=640``) exactly like the UI form.
        - ``metadata.created_at`` is stamped with the server's current UTC time
          so Recent Jobs ordering reflects real creation times.

        Args:
            request: The job to submit. The instance is normalized and
                registered; callers should use the returned instance.

        Returns:
            JobRequest: The registered job, with ``status=PENDING`` and the
            submission log line appended.

        Raises:
            ValueError: If ``request.job_id`` is already registered.

        Example:
            >>> from core.schema import JobRequest, JobStatus, TaskType
            >>> manager = JobsManager()
            >>> def _noop_execute(job):  # dispatcher stub: no engine work, no stdout
            ...     job.status = JobStatus.COMPLETED
            ...     return job
            >>> manager.dispatcher.execute = _noop_execute
            >>> request = JobRequest(job_id="doc-001", task_type=TaskType.PREDICT)
            >>> submitted = manager.submit_job_request(request)
            >>> submitted.job_id
            'doc-001'
            >>> manager.submit_job_request(request)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            ValueError: Duplicate job_id ...
        """
        # ---- Server-side normalization (fail-closed) ----
        request.status = JobStatus.PENDING
        request.error = None
        request.logs = []
        request.runtime_tracking.cancel_requested = False
        request.security_constraints.allow_shell = False
        request.security_constraints.path_whitelisted = True
        request.security_constraints.allowed_paths = list(
            set(request.security_constraints.allowed_paths + [request.output.output_dir])
        )
        request.output.artifacts = []

        # Engine-required defaults (mirroring the UI form path)
        if request.task_type == TaskType.TRAIN:
            request.params.setdefault("epochs", 1)
            request.params.setdefault("imgsz", 640)
        elif request.task_type == TaskType.VAL:
            request.params.setdefault("imgsz", 640)

        # The backend Metadata default is deep-copied from a class-definition-time
        # instance, so every job would otherwise share one created_at. Stamp each
        # submission with its own actual creation time (used by the Recent Jobs
        # timestamp column and its newest-first ordering).
        request.metadata.created_at = datetime.now(timezone.utc).isoformat()

        with self.lock:
            if request.job_id in self.jobs:
                raise ValueError(f"Duplicate job_id '{request.job_id}': a job with this identifier already exists")
            self.jobs[request.job_id] = request
            self.job_logs[request.job_id] = [
                f"[{datetime.now(timezone.utc).isoformat()}] Job {request.job_id} submitted"
            ]
            self._save()

        # Execute job in background thread
        thread = threading.Thread(target=self._execute_job, args=(request.job_id,), daemon=True)
        thread.start()

        return request

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

    def get_job(self, job_id: str) -> JobRequest | None:
        """Return the registered job for ``job_id``, or ``None`` when unknown.

        Thread-safe read accessor for callers (e.g. the FastAPI engine) that
        need the full ``JobRequest`` record rather than the status summary.

        Args:
            job_id: Job identifier.

        Returns:
            JobRequest | None: The registered job, or ``None`` when no job with
            this identifier exists.

        Example:
            >>> JobsManager().get_job("missing-job") is None
            True
        """
        with self.lock:
            return self.jobs.get(job_id)

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

    def get_job_log_lines(self, job_id: str) -> list[str]:
        """Return a copy of the raw log entries recorded for a job.

        Entries are appended exclusively through :meth:`_append_log` and the
        dispatcher's sanitized ``JobRequest.append_log`` interface, so the
        returned entries never carry plaintext credentials. Unlike
        :meth:`get_job_logs` this accessor returns structured entries (each may
        embed newlines, e.g. sanitized tracebacks) so REST consumers can
        paginate line-by-line.

        Args:
            job_id: Job identifier.

        Returns:
            list[str]: Copy of the job's log entries; empty when the job is
            unknown or has no logs yet.

        Example:
            >>> JobsManager().get_job_log_lines("missing-job")
            []
        """
        with self.lock:
            return list(self.job_logs.get(job_id, []))

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
