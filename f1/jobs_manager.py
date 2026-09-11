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
    - An unhandled exception in the execution worker fails the job with
      ``error_code="EXECUTION_FAILED"`` and a sanitized traceback appended to
      its logs, so a failed job never reports ``null`` error fields.

Example:
    >>> from core.schema import JobRequest, TaskType
    >>> manager = JobsManager()
    >>> manager.get_job("missing-job") is None
    True
    >>> manager.list_recent_jobs()
    []
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from core.schema import ErrorInfo, JobRequest, JobStatus, SecurityConstraints, TERMINAL_STATUSES, TaskType
from core.security import sanitize_for_persistence, sanitize_log_text
from f1.dispatcher import JobDispatcherStateMachine
from f1.worker_runtime import ManagedWorker, execute_job

#: Job states that still require high-frequency lifecycle polling.
ACTIVE_STATUSES = frozenset({"PENDING", "RUNNING"})
#: Image file extensions recognized by the artifact preview gallery.
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
MODEL_ROOTS_ENV = "F1_MODEL_ROOTS"
DATA_ROOTS_ENV = "F1_DATA_ROOTS"
OUTPUT_ROOT_ENV = "F1_OUTPUT_ROOT"
NETWORK_INPUT_HOSTS_ENV = "F1_NETWORK_INPUT_HOSTS"
MAX_PENDING_JOBS_ENV = "F1_MAX_PENDING_JOBS"
#: Default number of jobs allowed to wait for an execution slot.
MAX_PENDING_JOBS = 100
NETWORK_INPUT_SCHEMES: frozenset[str] = frozenset({"http", "https", "rtmp", "rtsp", "tcp"})

__all__ = [
    "ACTIVE_STATUSES",
    "IMAGE_EXTENSIONS",
    "MAX_PENDING_JOBS",
    "MAX_PENDING_JOBS_ENV",
    "JobsManager",
    "QueueFullError",
    "get_job_image_artifacts",
    "is_terminal_status",
]


def _compute_duration(
    started_at: str | None, completed_at: str | None, legacy_completed_at: str | None = None
) -> float | None:
    """Return exact execution seconds derived from lifecycle timestamps."""
    # ``f1.ui.jobs_tab`` still imports this private helper with its historical
    # ``(status, created_at, completed_at)`` signature. It has no started_at in
    # that compatibility path, so returning None is the only valid duration.
    if legacy_completed_at is not None or started_at in ACTIVE_STATUSES:
        return None
    if not started_at or not completed_at:
        return None
    try:
        return (datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)).total_seconds()
    except (ValueError, TypeError):
        return None


def _resolve_completion_time(job: Any) -> str | None:
    """Return the canonical completion timestamp for legacy UI callers."""
    return getattr(getattr(job, "metadata", None), "completed_at", None)


def _configured_roots(env_name: str, defaults: list[Path]) -> list[Path]:
    """Resolve a server-owned root list; request payloads never participate."""
    raw = os.environ.get(env_name)
    entries = raw.split(os.pathsep) if raw else [str(path) for path in defaults]
    return [Path(entry).resolve() for entry in entries if entry.strip()]


def _resolve_contained(path_value: str, roots: list[Path], label: str) -> Path:
    """Resolve a local path (including symlinks) and require containment in a trusted root."""
    if not path_value or any(ord(char) < 32 for char in path_value):
        raise ValueError(f"{label} is empty or contains control characters")
    try:
        resolved = Path(path_value).resolve()
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is not a valid local path") from exc
    for root in roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ValueError(f"{label} is outside the server-configured trusted roots")


def _is_network_input(source: str) -> bool:
    """Return whether the source is an explicitly supported non-file URL."""
    parsed = urlsplit(source)
    return parsed.scheme.lower() in NETWORK_INPUT_SCHEMES and bool(parsed.netloc)


class QueueFullError(RuntimeError):
    """Raised when the configured pending-job capacity is exhausted."""

    code = "QUEUE_FULL"


class JobsManager:
    """Bounded process execution with parent-owned lifecycle and persisted history."""

    def __init__(
        self,
        storage_path: str | None = None,
        *,
        model_roots: list[str | Path] | None = None,
        data_roots: list[str | Path] | None = None,
        output_root: str | Path | None = None,
        network_input_hosts: list[str] | None = None,
        cpu_concurrency: int | None = None,
        gpu_concurrency: int | None = None,
        max_pending_jobs: int | None = None,
        stop_grace_seconds: float | None = None,
    ) -> None:
        """Initialize job manager; in-memory only unless storage_path is given.

        Args:
            storage_path: Optional JSON file path. When provided, job state is
                persisted on every mutation and restored on startup. When None
                (default), behavior is purely in-memory.
        """
        self.jobs: dict[str, JobRequest] = {}
        self.job_logs: dict[str, list[str]] = {}
        self.lock = threading.Lock()
        self.dispatcher = JobDispatcherStateMachine()
        self._limits = {
            "cpu": int(os.environ.get("F1_CPU_CONCURRENCY", "2")) if cpu_concurrency is None else cpu_concurrency,
            "gpu": int(os.environ.get("F1_GPU_CONCURRENCY", "1")) if gpu_concurrency is None else gpu_concurrency,
        }
        if any(type(value) is not int or value < 1 for value in self._limits.values()):
            raise ValueError("CPU/GPU concurrency must be positive integers")
        self.max_pending_jobs = (
            int(os.environ.get(MAX_PENDING_JOBS_ENV, str(MAX_PENDING_JOBS)))
            if max_pending_jobs is None
            else max_pending_jobs
        )
        if type(self.max_pending_jobs) is not int or self.max_pending_jobs < 1:
            raise ValueError("Maximum pending jobs must be a positive integer")
        self._stop_grace = (
            float(os.environ.get("F1_STOP_GRACE_SECONDS", "2")) if stop_grace_seconds is None else stop_grace_seconds
        )
        if not 0 <= self._stop_grace <= 30:
            raise ValueError("Stop grace must be between 0 and 30 seconds")
        self._queues = {resource: queue.Queue() for resource in self._limits}
        self._supervisors: list[threading.Thread] = []
        self._workers: dict[str, ManagedWorker] = {}
        self._closing = False
        self._worker_executor = execute_job  # Server-only injection seam for lightweight process tests.
        cwd = Path.cwd().resolve()
        self._model_roots = (
            [Path(path).resolve() for path in model_roots]
            if model_roots is not None
            else _configured_roots(MODEL_ROOTS_ENV, [cwd])
        )
        self._data_roots = (
            [Path(path).resolve() for path in data_roots]
            if data_roots is not None
            else _configured_roots(DATA_ROOTS_ENV, [cwd])
        )
        self._output_root = (
            Path(output_root).resolve()
            if output_root is not None
            else _configured_roots(OUTPUT_ROOT_ENV, [cwd / "runs"])[0]
        )
        configured_hosts = (
            network_input_hosts
            if network_input_hosts is not None
            else os.environ.get(NETWORK_INPUT_HOSTS_ENV, "").split(",")
        )
        self._network_input_hosts = {host.strip().casefold() for host in configured_hosts if host.strip()}
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
        payload = sanitize_for_persistence(
            {
                "version": 1,
                "jobs": {jid: job.model_dump(mode="json") for jid, job in self.jobs.items()},
                "job_logs": self.job_logs,
            }
        )
        tmp_path = self._storage_path.with_suffix(self._storage_path.suffix + ".tmp")
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, self._storage_path)
        except OSError:
            pass

    def _load(self) -> None:
        """Restore persisted state from the storage file, if present.

        Jobs left in PENDING/RUNNING have no owned worker in this manager and
        are healed to FAILED with an explicit restart reason so the UI never
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
                    job.metadata.completed_at = datetime.now(timezone.utc).isoformat()
                    job.error = ErrorInfo(
                        code="SERVICE_RESTARTED",
                        message="Service restarted; the previous job has no owned worker and will not be resumed",
                    )
                    job.append_log(f"[SERVICE_RESTARTED] {job.error.message}")
                self.jobs[jid] = job
            self.job_logs.update(payload.get("job_logs", {}))
            for jid, job in self.jobs.items():
                if job.error and job.error.code == "SERVICE_RESTARTED":
                    self.job_logs.setdefault(jid, []).append(sanitize_log_text(job.error.message))
            self._save()
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
                allowed_paths=allowed_paths,
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
        - Client path lists and regex patterns are discarded. Only trusted,
          server-configured model/data roots are passed to handlers.
        - Model, data and output paths are independently resolved and checked;
          output must remain beneath the trusted output root.
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
            JobRequest: The registered job. It starts PENDING and may already be
            RUNNING when the caller reads it after a capacity slot is assigned.

        Raises:
            ValueError: If ``request.job_id`` is already registered.
            QueueFullError: If the pending-job capacity is exhausted.

        Execution uses a fresh Python interpreter. Parent-side handler/dispatcher
        monkeypatches are not inherited; lifecycle tests supply an importable
        server-owned executor through ``_worker_executor`` instead.
        """
        # ---- Server-side normalization (fail-closed) ----
        request.status = JobStatus.PENDING
        request.error = None
        request.logs = []
        request.metadata.started_at = None
        request.metadata.completed_at = None
        request.runtime_tracking.cancel_requested = False
        request.security_constraints.allow_shell = False
        request.security_constraints.path_whitelisted = True
        request.security_constraints.allowed_paths = sorted(
            {str(path) for path in (*self._model_roots, *self._data_roots)}
        )
        request.security_constraints.allowed_path_patterns = []
        _resolve_contained(request.output.output_dir, [self._output_root], "output_dir")
        model_path = request.params.get("model_path")
        if model_path:
            _resolve_contained(str(model_path), self._model_roots, "model_path")
        data_source = request.params.get("data_source")
        if data_source is not None:
            sources = data_source if isinstance(data_source, (list, tuple)) else [data_source]
            for source in sources:
                source_value = str(source)
                if _is_network_input(source_value):
                    hostname = urlsplit(source_value).hostname
                    if hostname is None or hostname.casefold() not in self._network_input_hosts:
                        raise ValueError("data_source network host is not server-authorized")
                else:
                    _resolve_contained(source_value, self._data_roots, "data_source")
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
            if self._closing:
                raise ValueError("Job manager is shutting down")
            if request.job_id in self.jobs:
                raise ValueError(f"Duplicate job_id '{request.job_id}': a job with this identifier already exists")
            pending_count = sum(job.status == JobStatus.PENDING for job in self.jobs.values())
            if pending_count >= self.max_pending_jobs:
                raise QueueFullError(f"Pending job capacity exhausted ({pending_count}/{self.max_pending_jobs})")
            self.jobs[request.job_id] = request
            self.job_logs[request.job_id] = [
                f"[{datetime.now(timezone.utc).isoformat()}] Job {request.job_id} submitted"
            ]
            self._save()
            self._start_supervisors()
            self._queues[self._resource_class(request)].put(request.job_id)

        return request

    @staticmethod
    def _resource_class(job: JobRequest) -> str:
        """Explicit CPU uses CPU slots; auto, CUDA, multi-GPU and MPS share GPU slots."""
        device = job.params.get("device", "cpu")
        return "cpu" if str(device).strip().lower() == "cpu" else "gpu"

    def _start_supervisors(self) -> None:
        """Start a fixed number of supervisors once, while holding the manager lock."""
        if self._supervisors:
            return
        for resource, capacity in self._limits.items():
            for index in range(capacity):
                thread = threading.Thread(
                    target=self._consume_queue, args=(resource,), name=f"studio-{resource}-{index}", daemon=True
                )
                self._supervisors.append(thread)
                thread.start()
        atexit.register(self.shutdown)

    def _consume_queue(self, resource: str) -> None:
        """Own one capacity slot; no thread or process is allocated to waiting jobs."""
        pending = self._queues[resource]
        while True:
            job_id = pending.get()
            try:
                if job_id is None:
                    return
                try:
                    self._execute_job(job_id)
                except Exception as exc:
                    with self.lock:
                        if job_id in self._workers:
                            raise  # Never discard ownership of an unexpectedly live worker.
                        self._fail_job(self.jobs[job_id], "EXECUTION_FAILED", str(exc))
            finally:
                pending.task_done()

    def _fail_job(self, job: JobRequest, code: str, message: str) -> None:
        """Set a terminal result only when there is no live owned computation; lock held."""
        job.status = JobStatus.FAILED
        job.metadata.completed_at = datetime.now(timezone.utc).isoformat()
        job.error = ErrorInfo(code=code, message=sanitize_log_text(message))
        job.append_log(f"[{code}] {message}")
        self.job_logs.setdefault(job.job_id, []).append(sanitize_log_text(f"[{code}] {message}"))
        self._save()

    def _cancel_job(self, job: JobRequest, message: str) -> None:
        """Publish confirmed cancellation after no owned computation remains; lock held."""
        job.status = JobStatus.CANCELLED
        job.metadata.completed_at = datetime.now(timezone.utc).isoformat()
        job.error = ErrorInfo(code="USER_CANCELLED", message=sanitize_log_text(message))
        job.append_log(f"[USER_CANCELLED] {message}")
        self.job_logs.setdefault(job.job_id, []).append(sanitize_log_text(f"[USER_CANCELLED] {message}"))
        self._save()

    def _execute_job(self, job_id: str) -> None:
        """Supervise one process through execution, tree cleanup and final persistence."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                return
            if self._closing or job.runtime_tracking.cancel_requested:
                if self._closing:
                    self._fail_job(job, "SERVICE_SHUTDOWN", "Job stopped before worker launch")
                else:
                    self._cancel_job(job, "Job cancelled before worker launch")
                return
            job.metadata.started_at = datetime.now(timezone.utc).isoformat()
            worker = ManagedWorker(job, self._worker_executor)
            self._workers[job_id] = worker
            # The child's request remains PENDING for the existing dispatcher FSM.
            job.status = JobStatus.RUNNING
            self._save()
        result = None
        code, message = None, None
        deadline = time.monotonic() + max(float(job.runtime_tracking.timeout_seconds), 0)
        try:
            worker.start()
            # Run our cleanup before multiprocessing's interpreter-exit join.
            atexit.unregister(self.shutdown)
            atexit.register(self.shutdown)
            while True:
                with self.lock:
                    if self._closing:
                        code, message = "SERVICE_SHUTDOWN", "Service is shutting down"
                    elif job.runtime_tracking.cancel_requested:
                        code, message = "USER_CANCELLED", "Job execution cancelled by user request"
                    elif time.monotonic() >= deadline:
                        code, message = "TIMEOUT", "Job execution exceeded its configured timeout"
                if code:
                    break
                kind, payload = worker.receive()
                if kind == "lost":
                    code, message = "WORKER_LOST", "Computation process exited without reporting a result"
                    break
                if kind == "result":
                    result = JobRequest.model_validate(payload)
                    if result.status not in TERMINAL_STATUSES:
                        code, message = "WORKER_LOST", "Worker returned without a terminal result"
                    break
                if not worker.process.is_alive():
                    code, message = "WORKER_LOST", "Worker exited without reporting a result"
                    break
        except (EOFError, BrokenPipeError):
            code, message = "WORKER_LOST", "Worker connection closed without a result"
        except Exception as exc:  # noqa: BLE001 - parent must always clean up the tree
            code, message = "EXECUTION_FAILED", str(exc)
            self._append_log(job_id, traceback.format_exc())
        finally:
            # Never detach a worker or release its resource slot on cleanup failure.
            # Keep RUNNING and a structured reason while retaining/retrying ownership.
            while True:
                try:
                    worker.stop(self._stop_grace if code else 0)
                    break
                except Exception as exc:  # noqa: BLE001 - retain ownership on OS cleanup failure
                    with self.lock:
                        job.error = ErrorInfo(code="WORKER_STOP_FAILED", message=sanitize_log_text(str(exc)))
                        self._save()
                    time.sleep(0.2)
            worker.close()
            with self.lock:
                self._workers.pop(job_id, None)
                # Publish only from RUNNING. A natural terminal result already
                # published under this lock wins over a later cancellation.
                current = self.jobs[job_id]
                if current.status not in TERMINAL_STATUSES:
                    if current.runtime_tracking.cancel_requested:
                        code, message = "USER_CANCELLED", "Job execution cancelled by user request"
                    elif self._closing:
                        code, message = "SERVICE_SHUTDOWN", "Service is shutting down"
                    if code == "USER_CANCELLED":
                        self._cancel_job(current, message)
                    elif code:
                        self._fail_job(current, code, message)
                    else:
                        self._normalize_job_artifacts(result)
                        result.metadata.created_at = current.metadata.created_at
                        result.metadata.started_at = current.metadata.started_at
                        result.metadata.completed_at = datetime.now(timezone.utc).isoformat()
                        self.jobs[job_id] = result
                        self.job_logs[job_id].extend(sanitize_log_text(line) for line in result.logs)
                        self._save()

    def shutdown(self) -> None:
        """Reject submissions, end pending jobs and join all owned execution slots."""
        with self.lock:
            if not self._closing:
                self._closing = True
                for job_id, job in self.jobs.items():
                    if job.status == JobStatus.PENDING and job_id not in self._workers:
                        self._fail_job(job, "SERVICE_SHUTDOWN", "Service stopped before worker launch")
                for resource, capacity in self._limits.items():
                    for _ in range(capacity):
                        self._queues[resource].put(None)
        deadline = time.monotonic() + self._stop_grace + 15
        for thread in self._supervisors:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in self._supervisors):
            raise RuntimeError("Worker cleanup is still pending; ownership has been retained")
        atexit.unregister(self.shutdown)

    def _append_log(self, job_id: str, message: str) -> None:
        """Append log message to job log buffer."""
        with self.lock:
            if job_id not in self.job_logs:
                self.job_logs[job_id] = []
            self.job_logs[job_id].append(sanitize_log_text(message))
            self._save()

    def _job_artifact_root(self, job: JobRequest) -> Path | None:
        """Return the resolved per-job output directory when it is trusted."""
        try:
            output_dir = _resolve_contained(job.output.output_dir, [self._output_root], "output_dir")
            root = (output_dir / job.job_id).resolve()
            root.relative_to(self._output_root)
            return root
        except (OSError, ValueError):
            return None

    def _normalize_job_artifacts(self, job: JobRequest) -> None:
        """Replace handler paths with collision-free relative artifact IDs."""
        root = self._job_artifact_root(job)
        if root is None:
            job.output.artifacts = []
            return
        artifact_ids: list[str] = []
        seen: set[str] = set()
        for entry in job.output.artifacts:
            candidate = Path(entry)
            if not candidate.is_absolute():
                candidate = root / candidate
            try:
                resolved = candidate.resolve()
                artifact_id = resolved.relative_to(root).as_posix()
            except (OSError, ValueError):
                continue
            if not resolved.is_file() or artifact_id in seen:
                continue
            seen.add(artifact_id)
            artifact_ids.append(artifact_id)
        job.output.artifacts = sorted(artifact_ids)

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
            created_at = job.metadata.created_at
            started_at = job.metadata.started_at
            completed_at = job.metadata.completed_at
            duration = _compute_duration(started_at, completed_at)

            return {
                "status": status,
                "created_at": created_at,
                "started_at": started_at,
                "completed_at": completed_at,
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
        """Get validated artifacts as (safe relative ID, internal absolute path) tuples.

        Args:
            job_id: Job identifier

        Returns:
            List of (filename, path) tuples for artifact downloads
        """
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or not hasattr(job.output, "artifacts"):
                return []

            root = self._job_artifact_root(job)
            if root is None:
                return []
            artifacts = []
            seen: set[str] = set()
            for artifact_path in job.output.artifacts:
                path = root / artifact_path
                try:
                    resolved = path.resolve()
                    artifact_id = resolved.relative_to(root).as_posix()
                except (OSError, ValueError):
                    continue
                if resolved.is_file() and artifact_id not in seen:
                    seen.add(artifact_id)
                    artifacts.append((artifact_id, str(resolved)))
            return artifacts

    def get_job_image_artifacts(self, job_id: str) -> list[str]:
        """Get safe relative image artifact identifiers for a completed job.

        Args:
            job_id: Job identifier

        Returns:
            List of relative artifact IDs under the job's output directory.
            Returns an empty list when the job is not found or not completed.
        """
        return [
            artifact_id
            for artifact_id, path in self.get_job_artifacts(job_id)
            if Path(path).suffix.lower() in IMAGE_EXTENSIONS
        ]

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

            if job.status in TERMINAL_STATUSES:
                return f"ℹ️ Job already in terminal state: {job.status.value}"

            if not job.runtime_tracking.cancellable:
                return "⚠️ Job is not cancellable"

            job.runtime_tracking.cancel_requested = True
            if job_id not in self._workers:
                self._cancel_job(job, "Job cancelled before worker launch")
            else:
                self._save()

        # Append log AFTER releasing the lock: _append_log acquires self.lock
        # internally and threading.Lock is not reentrant (self-deadlock).
        self._append_log(job_id, f"[{datetime.now(timezone.utc).isoformat()}] 🚫 Cancellation requested")
        return f"✅ Cancellation requested for {job_id}"

    def list_recent_jobs(self, limit: int = 10) -> list[dict[str, Any]]:
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
                        "started_at": job.metadata.started_at,
                        "completed_at": job.metadata.completed_at,
                        "duration": _compute_duration(job.metadata.started_at, job.metadata.completed_at),
                    }
                )
            return jobs_list


def is_terminal_status(status: str) -> bool:
    """Return True when a job status string is terminal (not PENDING/RUNNING).

    COMPLETED, FAILED, CANCELLED and NOT_FOUND are terminal from the poller's
    perspective.

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
