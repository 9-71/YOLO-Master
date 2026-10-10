"""Jobs REST adapters consuming detached snapshots and typed owner decisions."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from core.schema import JobRequest, JobStatus, TaskType
from studio.artifacts import IMAGE_EXTENSIONS
from studio.job_logs import sanitize_log_text
from studio.jobs_manager import (
    CancelCode,
    DuplicateJobError,
    JobsManager,
    ManagerClosingError,
    QueueFullError,
    StatePersistenceError,
    compute_duration,
)

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])
download_router = APIRouter(tags=["artifacts"])


def error(status_code, code, message):
    """Create a stable, sanitized transport error without inspecting text semantics."""
    return HTTPException(status_code, detail={"code": code, "message": sanitize_log_text(message)})


def get_jobs_manager(request: Request) -> JobsManager:
    """Resolve the lifespan-owned manager; never create an owner inside a request."""
    manager = getattr(request.app.state, "jobs_manager", None)
    if manager is None or not request.app.state.ready:
        raise error(503, "SERVICE_UNAVAILABLE", "Service lifespan is not ready")
    return manager


MANAGER = Depends(get_jobs_manager)


class JobSummary(BaseModel):
    """Detached summary with execution time excluding the queue wait."""

    job_id: str
    task_type: TaskType
    status: JobStatus
    created_at: str
    started_at: str | None
    completed_at: str | None
    duration: float | None


class JobListResponse(BaseModel):
    """Existing recent-jobs window and atomic total."""

    jobs: list[JobSummary]
    total: int
    limit: int
    offset: int


class JobStatusResponse(JobSummary):
    """One coherent owner snapshot, including any execution failure."""

    error_code: str | None
    error_message: str | None
    artifact_count: int
    metadata: dict[str, Any]


class CancelResponse(BaseModel):
    """Request acknowledgement, distinct from the eventual cleanup-confirmed state."""

    job_id: str
    status: str
    message: str


class LogsResponse(BaseModel):
    """Flattened-line cursor window; null cursor means only the current tail."""

    job_id: str
    total: int
    offset: int
    limit: int | None
    logs: list[str]
    next_offset: int | None


class ArtifactEntry(BaseModel):
    """An exact relative manifest ID and a download URL, never a filesystem path."""

    filename: str
    artifact_id: str
    is_image: bool
    download_url: str


class ArtifactsResponse(BaseModel):
    """Only manifest-authorized files that still pass containment checks."""

    job_id: str
    artifacts: list[ArtifactEntry]
    image_artifacts: list[str]


def require_job(manager, job_id):
    """Read one detached record or return a transport 404."""
    job = manager.get_job(job_id)
    if job is None:
        raise error(404, "JOB_NOT_FOUND", "Job not found")
    return job


@router.post("", response_model=JobRequest, status_code=201)
@router.post("/", response_model=JobRequest, status_code=201, include_in_schema=False)
def create_job(payload: JobRequest, manager: JobsManager = MANAGER):
    """Admit through the single owner; durable acceptance precedes scheduling."""
    try:
        return manager.submit_job_request(payload)
    except DuplicateJobError as exc:
        raise error(409, exc.code, str(exc)) from None
    except ManagerClosingError as exc:
        raise error(503, exc.code, str(exc)) from None
    except QueueFullError as exc:
        raise error(429, exc.code, str(exc)) from None
    except StatePersistenceError as exc:
        raise error(503, exc.code, str(exc)) from None
    except ValueError as exc:
        raise error(422, "INVALID_JOB", str(exc)) from None


@router.get("", response_model=JobListResponse)
@router.get("/", response_model=JobListResponse, include_in_schema=False)
def list_jobs(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    manager: JobsManager = MANAGER,
):
    """Return an atomic recent-jobs snapshot with the existing limit/offset contract."""
    return manager.list_jobs_snapshot(limit, offset)


@router.get("/{job_id}", response_model=JobStatusResponse)
def read_job(job_id: str, manager: JobsManager = MANAGER):
    """Build every status field from the same detached owner record."""
    job = require_job(manager, job_id)
    return JobStatusResponse(
        job_id=job.job_id,
        task_type=job.task_type,
        status=job.status,
        created_at=job.metadata.created_at,
        started_at=job.metadata.started_at,
        completed_at=job.metadata.completed_at,
        duration=compute_duration(job.metadata.started_at, job.metadata.completed_at),
        error_code=job.error.code if job.error else None,
        error_message=job.error.message if job.error else None,
        artifact_count=len(job.output.artifacts),
        metadata=job.metadata.model_dump(mode="json"),
    )


@router.post("/{job_id}/cancel", response_model=CancelResponse, status_code=202)
def cancel_job(job_id: str, response: Response, manager: JobsManager = MANAGER):
    """Map the owner's typed decision; accepted running cancel may remain active."""
    decision = manager.request_cancel(job_id)
    if decision.code == CancelCode.NOT_FOUND:
        raise error(404, "JOB_NOT_FOUND", decision.message)
    if decision.code == CancelCode.NOT_CANCELLABLE:
        raise error(409, "NOT_CANCELLABLE", decision.message)
    if decision.code == CancelCode.ALREADY_TERMINAL:
        response.status_code = 200
        return CancelResponse(job_id=job_id, status=decision.job.status.value, message=decision.message)
    return CancelResponse(job_id=job_id, status="cancel_requested", message=decision.message)


@router.get("/{job_id}/logs", response_model=LogsResponse)
def get_logs(
    job_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int | None = Query(default=None, ge=1, le=10000),
    manager: JobsManager = MANAGER,
):
    """Page by flattened lines from a detached, ordered and sanitized Runtime buffer."""
    require_job(manager, job_id)
    lines = [line for entry in manager.get_job_log_lines(job_id) for line in sanitize_log_text(entry).splitlines()]
    window = lines[offset:] if limit is None else lines[offset : offset + limit]
    return LogsResponse(
        job_id=job_id,
        total=len(lines),
        offset=offset,
        limit=limit,
        logs=window,
        next_offset=offset + len(window) if offset + len(window) < len(lines) else None,
    )


@router.get("/{job_id}/artifacts", response_model=ArtifactsResponse)
def get_artifacts(job_id: str, manager: JobsManager = MANAGER):
    """List exact final-manifest IDs, preserving nested paths and rechecking containment."""
    require_job(manager, job_id)
    artifacts = [
        ArtifactEntry(
            filename=Path(identifier).name,
            artifact_id=identifier,
            is_image=Path(identifier).suffix.lower() in IMAGE_EXTENSIONS,
            download_url=f"/static/artifacts/{quote(job_id, safe='')}/{quote(identifier, safe='/')}",
        )
        for identifier, _ in manager.get_job_artifacts(job_id)
    ]
    return ArtifactsResponse(
        job_id=job_id, artifacts=artifacts, image_artifacts=[item.artifact_id for item in artifacts if item.is_image]
    )


@download_router.get("/static/artifacts/{job_id}/{artifact_id:path}")
def download_artifact(job_id: str, artifact_id: str, manager: JobsManager = MANAGER):
    """Open an authorized file and revalidate the opened identity before streaming."""
    path = manager.resolve_artifact(job_id, artifact_id)
    if path is None:
        raise error(404, "ARTIFACT_NOT_FOUND", "Artifact not found")
    handle = None
    try:
        handle = path.open("rb")
        checked = manager.resolve_artifact(job_id, artifact_id)
        opened = os.fstat(handle.fileno())
        current = checked.stat() if checked is not None else None
        if checked != path or current is None or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise OSError("Artifact identity changed")
    except OSError:
        if handle is not None:
            handle.close()
        raise error(404, "ARTIFACT_NOT_FOUND", "Artifact not found") from None

    def chunks():
        try:
            while chunk := handle.read(64 * 1024):
                yield chunk
        finally:
            handle.close()

    return StreamingResponse(
        chunks(),
        media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(path.name, safe='')}"},
        background=BackgroundTask(handle.close),
    )
