"""FastAPI REST endpoints for F1 job submission, monitoring and artifacts.

P2 (Standalone FastAPI Engine & Decoupled Architecture): this router exposes the
core task dispatcher endpoints (``/api/v1/jobs/*``) backed by the headless
:class:`f1.jobs_manager.JobsManager`, with no Gradio dependency. Request bodies
are validated with Pydantic against ``core/schema.py`` (``JobRequest``,
``JobStatus``, ``TaskType``), and every response log line / error message is
routed through :func:`core.security.sanitize_log_text`.

Endpoints (the collection routes accept both ``/api/v1/jobs`` and
``/api/v1/jobs/`` — browsers and CLI clients disagree on the canonical form):
    POST   /api/v1/jobs[/]                  Submit a new job (201 Created)
    GET    /api/v1/jobs[/]                  List recent jobs (limit/offset pagination)
    GET    /api/v1/jobs/{job_id}            Current status and metadata (404 when unknown)
    POST   /api/v1/jobs/{job_id}/cancel     Cooperative cancellation (202/200/409/404)
    GET    /api/v1/jobs/{job_id}/logs       Sanitized logs (offset/limit windows)
    GET    /api/v1/jobs/{job_id}/artifacts  Artifact manifest and image paths

The process-wide manager is exposed through the :func:`get_jobs_manager`
dependency so tests and embeddings can replace it via
``app.dependency_overrides``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel

from core.schema import ErrorInfo, JobRequest, JobStatus, TaskType
from core.security import sanitize_log_text
from f1.jobs_manager import IMAGE_EXTENSIONS, JobsManager, QueueFullError

__all__ = ["get_jobs_manager", "router"]

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

#: Environment variable naming the JobsManager persistence file.
F1_JOBS_STATE_PATH_ENV = "F1_JOBS_STATE_PATH"
#: Default persistence file, shared with the Gradio WebUI (``app.py``).
DEFAULT_JOBS_STATE_PATH = "runs/jobs_state.json"

_manager: JobsManager | None = None
_manager_lock = threading.Lock()


def shutdown_jobs_manager() -> None:
    """Close the singleton on service shutdown without creating a new manager."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.shutdown()
            _manager = None


def get_jobs_manager() -> JobsManager:
    """Return the process-wide :class:`JobsManager` singleton, built lazily.

    The storage path comes from ``F1_JOBS_STATE_PATH`` or defaults to
    ``runs/jobs_state.json`` (the same state file the Gradio WebUI persists
    to), so the REST engine and the WebUI can observe each other's job
    history. Construction is guarded by a lock; FastAPI routes consume this
    function through ``Depends`` and tests can replace it via
    ``app.dependency_overrides``.

    Returns:
        JobsManager: The shared job manager instance.

    Example:
        >>> manager = get_jobs_manager()
        >>> isinstance(manager, JobsManager)
        True
    """
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                storage_path = os.environ.get(F1_JOBS_STATE_PATH_ENV) or DEFAULT_JOBS_STATE_PATH
                _manager = JobsManager(storage_path=storage_path)
    return _manager


#: Shared FastAPI dependency resolving the process-wide JobsManager singleton.
#: Defined once at module level (rather than calling ``Depends`` in each
#: endpoint default) per FastAPI's shareable-dependency pattern.
MANAGER_DEPENDENCY = Depends(get_jobs_manager)


class JobSummary(BaseModel):
    """Summary row for the recent-jobs listing."""

    job_id: str
    task_type: TaskType
    status: JobStatus
    created_at: str
    started_at: str | None
    completed_at: str | None
    duration: float | None


class JobListResponse(BaseModel):
    """Paginated recent-jobs listing, newest first."""

    jobs: list[JobSummary]
    total: int
    limit: int
    offset: int


class JobStatusResponse(BaseModel):
    """Current lifecycle status and metadata of one job."""

    job_id: str
    task_type: TaskType
    status: JobStatus
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    duration: float | None
    error_code: str | None
    error_message: str | None
    artifact_count: int
    metadata: dict[str, Any]


class CancelResponse(BaseModel):
    """Cooperative-cancellation request outcome."""

    job_id: str
    status: str
    message: str


class LogsResponse(BaseModel):
    """One paginated window of sanitized execution logs."""

    job_id: str
    total: int
    offset: int
    limit: int | None
    logs: list[str]
    next_offset: int | None


class ArtifactEntry(BaseModel):
    """One generated artifact file with its download reference."""

    filename: str
    artifact_id: str
    is_image: bool
    download_url: str


class ArtifactsResponse(BaseModel):
    """Manifest of generated artifacts plus image artifact paths."""

    job_id: str
    artifacts: list[ArtifactEntry]
    image_artifacts: list[str]


# Collection routes are bound under both slash spellings so the frontend and
# ad-hoc clients (which mix "/api/v1/jobs" and "/api/v1/jobs/") never fall
# through to the root static mount's 404 for the unmatched one.
@router.post(
    "",
    response_model=JobRequest,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a new job",
)
@router.post(
    "/",
    response_model=JobRequest,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a new job",
)
def create_job(payload: JobRequest, manager: JobsManager = MANAGER_DEPENDENCY) -> JobRequest:
    """Submit a ``JobRequest`` for background execution and return it (201).

    The body is validated by Pydantic against ``core/schema.py``; unknown task
    types and malformed fields are rejected with 422. The manager normalizes
    the payload fail-closed (``allow_shell=False``, ``path_whitelisted=True``,
    fresh PENDING lifecycle, server-stamped ``created_at``) before registering
    it, so the returned request reflects the enforced server-side state.

    Raises:
        HTTPException: 409 when ``payload.job_id`` is already registered.
            429 when the pending-job capacity is exhausted.
    """
    try:
        return manager.submit_job_request(payload)
    except QueueFullError as exc:
        error = ErrorInfo(code=exc.code, message=sanitize_log_text(str(exc)))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=error.model_dump(mode="json"),
        ) from exc
    except ValueError as exc:
        is_duplicate = str(exc).startswith("Duplicate job_id")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT if is_duplicate else status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=sanitize_log_text(str(exc)),
        ) from exc


@router.get(
    "",
    response_model=JobListResponse,
    summary="List recent jobs",
)
@router.get(
    "/",
    response_model=JobListResponse,
    summary="List recent jobs",
)
def list_jobs(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    manager: JobsManager = MANAGER_DEPENDENCY,
) -> JobListResponse:
    """Return recent jobs, newest first, with ``limit``/``offset`` pagination.

    Args:
        limit: Maximum number of rows in this page (1..100).
        offset: Number of newest rows to skip.
    """
    rows = manager.list_recent_jobs(limit=offset + limit)[offset:]
    return JobListResponse(
        jobs=[
            JobSummary(
                job_id=row["job_id"],
                task_type=TaskType(row["task_type"]),
                status=JobStatus(row["status"].lower()),
                created_at=row["created_at"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                duration=row["duration"],
            )
            for row in rows
        ],
        # CPython dict length is atomic for this metadata counter; the rows
        # themselves are already lock-protected by the manager.
        total=len(manager.jobs),
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status and metadata",
)
def read_job(job_id: str, manager: JobsManager = MANAGER_DEPENDENCY) -> JobStatusResponse:
    """Return the current lifecycle status and metadata of one job.

    Raises:
        HTTPException: 404 when no job with ``job_id`` is registered.
    """
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=sanitize_log_text(f"Job '{job_id}' not found"),
        )
    info = manager.get_job_status(job_id)
    error_message = info.get("error_message")
    return JobStatusResponse(
        job_id=job_id,
        task_type=job.task_type,
        status=job.status,
        created_at=job.metadata.created_at,
        started_at=job.metadata.started_at,
        completed_at=job.metadata.completed_at,
        duration=info.get("duration"),
        error_code=info.get("error_code"),
        # Defense in depth: dispatcher-attached messages are already sanitized,
        # but re-sanitizing at the response boundary is idempotent.
        error_message=sanitize_log_text(error_message) if error_message else None,
        artifact_count=info.get("artifact_count", 0),
        metadata=job.metadata.model_dump(mode="json"),
    )


@router.post(
    "/{job_id}/cancel",
    response_model=CancelResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request worker cancellation",
)
def cancel_job(
    job_id: str, response: Response, manager: JobsManager = MANAGER_DEPENDENCY
) -> CancelResponse:
    """Request cooperative cancellation of a PENDING/RUNNING job (202).

    The manager stops the owned process tree and only then persists CANCELLED
    with error code ``USER_CANCELLED``. A 202 response acknowledges a new request;
    an idempotent replay against a terminal job returns 200 and its existing state.

    Raises:
        HTTPException: 404 when the job is unknown, 409 when it is not cancellable.
    """
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=sanitize_log_text(f"Job '{job_id}' not found"),
        )
    message = manager.cancel_job(job_id)
    if "not cancellable" in message:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=sanitize_log_text(f"Job '{job_id}' is not cancellable"),
        )
    if "already in terminal state" in message:
        # Idempotent replay: manager made this decision while holding its lock,
        # so a stale active-state read above cannot produce a false 202.
        current = manager.get_job(job_id)
        response.status_code = status.HTTP_200_OK
        return CancelResponse(
            job_id=job_id,
            status=current.status.value,
            message=sanitize_log_text(message),
        )
    return CancelResponse(
        job_id=job_id,
        status="cancel_requested",
        message=sanitize_log_text(message),
    )


@router.get(
    "/{job_id}/logs",
    response_model=LogsResponse,
    summary="Get sanitized execution logs",
)
def get_logs(
    job_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int | None = Query(default=None, ge=1, le=10000),
    manager: JobsManager = MANAGER_DEPENDENCY,
) -> LogsResponse:
    """Return one window of the job's sanitized execution logs.

    Log entries may embed newlines (e.g. sanitized tracebacks), so the window
    is computed over flattened lines. ``next_offset`` is the cursor for the
    following page, or ``None`` when the tail was reached.

    Args:
        offset: Number of lines to skip from the beginning.
        limit: Maximum lines in this window (``None`` returns the remainder).

    Raises:
        HTTPException: 404 when no job with ``job_id`` is registered.
    """
    if manager.get_job(job_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=sanitize_log_text(f"Job '{job_id}' not found"),
        )
    lines = [line for entry in manager.get_job_log_lines(job_id) for line in sanitize_log_text(entry).splitlines()]
    total = len(lines)
    window = lines[offset:] if limit is None else lines[offset : offset + limit]
    next_offset = offset + len(window) if offset + len(window) < total else None
    return LogsResponse(
        job_id=job_id,
        total=total,
        offset=offset,
        limit=limit,
        logs=window,
        next_offset=next_offset,
    )


@router.get(
    "/{job_id}/artifacts",
    response_model=ArtifactsResponse,
    summary="List generated artifacts",
)
def get_artifacts(job_id: str, manager: JobsManager = MANAGER_DEPENDENCY) -> ArtifactsResponse:
    """Return the artifact manifest and image artifact paths of one job.

    ``artifacts`` lists the dispatcher-captured files (existing on disk only)
    with their ``/static/artifacts/...`` download references;
    ``image_artifacts`` carries safe relative IDs scoped to the job's output
    directory; server filesystem paths are never returned.

    Raises:
        HTTPException: 404 when no job with ``job_id`` is registered.
    """
    if manager.get_job(job_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=sanitize_log_text(f"Job '{job_id}' not found"),
        )
    artifacts = [
        ArtifactEntry(
            filename=Path(name).name,
            artifact_id=name,
            is_image=Path(path).suffix.lower() in IMAGE_EXTENSIONS,
            download_url=f"/static/artifacts/{job_id}/{quote(name, safe='/')}",
        )
        for name, path in manager.get_job_artifacts(job_id)
    ]
    return ArtifactsResponse(
        job_id=job_id,
        artifacts=artifacts,
        image_artifacts=manager.get_job_image_artifacts(job_id),
    )
