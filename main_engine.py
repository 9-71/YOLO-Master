"""Standalone FastAPI engine for the YOLO-Master F1 task platform.

P2 (Standalone FastAPI Engine & Decoupled Architecture): a Gradio-independent
REST engine exposing the core task dispatcher endpoints (``/api/v1/jobs/*``)
with native OpenAPI/Swagger documentation backed by ``core/schema.py``.

Run it with::

    uvicorn main_engine:app --host 127.0.0.1 --port 8000

or simply::

    python main_engine.py

Artifact delivery is fail-closed: files are served at
``/static/artifacts/{job_id}/{filename}`` only when they appear in the job's
dispatcher-produced artifact manifest (or the validated image-artifact scan).
The route performs an exact manifest lookup and never joins client input onto
filesystem paths, so unlisted files and traversal attempts always 404.

Configuration environment variables:
    - ``F1_JOBS_STATE_PATH``: JobsManager persistence file (default
      ``runs/jobs_state.json``, shared with the Gradio WebUI).
    - ``F1_CORS_ORIGINS``: comma-separated CORS allowlist overriding the local
      frontend development defaults.
    - ``F1_ENGINE_HOST`` / ``F1_ENGINE_PORT``: bind address for
      ``python main_engine.py`` (default ``127.0.0.1:8000``).
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.v1.jobs import get_jobs_manager
from api.v1.jobs import router as jobs_router
from core.security import sanitize_log_text
from f1.jobs_manager import JobsManager

APP_TITLE = "YOLO-Master F1 Task Engine"
APP_VERSION = "0.1.0"

#: Local frontend development origins allowed by the CORS middleware by default.
DEV_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",  # Vite
    "http://127.0.0.1:5173",
    "http://localhost:3000",  # Create React App / Next.js
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
)

#: Environment variable overriding the CORS allowlist (comma-separated origins).
CORS_ORIGINS_ENV = "F1_CORS_ORIGINS"
#: Environment variables controlling the ``python main_engine.py`` bind address.
ENGINE_HOST_ENV = "F1_ENGINE_HOST"
ENGINE_PORT_ENV = "F1_ENGINE_PORT"

#: Shared FastAPI dependency resolving the process-wide JobsManager singleton
#: (defined at module level rather than calling ``Depends`` in the endpoint
#: default, per FastAPI's shareable-dependency pattern).
MANAGER_DEPENDENCY = Depends(get_jobs_manager)


def build_cors_origins() -> list[str]:
    """Resolve the CORS origin allowlist from ``F1_CORS_ORIGINS`` or the dev defaults.

    Returns:
        list[str]: The effective allowlist; the environment variable (when set
        and non-empty) completely replaces the local development defaults.
    """
    raw = os.environ.get(CORS_ORIGINS_ENV)
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return list(DEV_CORS_ORIGINS)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    The returned app registers CORS for local frontend development origins,
    includes the ``/api/v1/jobs`` router, and exposes the fail-closed artifact
    delivery route at ``/static/artifacts``.

    Returns:
        FastAPI: The configured YOLO-Master F1 task engine application.

    Example:
        >>> from main_engine import create_app
        >>> app = create_app()
        >>> app.title
        'YOLO-Master F1 Task Engine'
    """
    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        description=(
            "Standalone REST engine for YOLO-Master F1 job submission, "
            "lifecycle monitoring, sanitized log streaming, cancellation and "
            "artifact delivery."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=build_cors_origins(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(jobs_router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        """Return engine identity and documentation pointers."""
        return {
            "service": APP_TITLE,
            "version": APP_VERSION,
            "docs": "/docs",
            "openapi": "/openapi.json",
        }

    @app.get("/health", tags=["system"], summary="Liveness probe")
    def health() -> dict[str, str]:
        """Return a static liveness response (no dependency on job state)."""
        return {"status": "ok", "service": APP_TITLE}

    @app.get(
        "/static/artifacts/{job_id}/{filename}",
        response_class=FileResponse,
        tags=["artifacts"],
        summary="Download one artifact file",
    )
    def serve_artifact(job_id: str, filename: str, manager: JobsManager = MANAGER_DEPENDENCY) -> FileResponse:
        """Serve one artifact file, restricted to the job's manifest (fail-closed).

        The requested ``filename`` (a single path segment, so slashes can never
        reach this handler) must match — by exact basename — a file listed in
        the job's artifact manifest or validated image-artifact scan. The
        manifest is populated exclusively by the dispatcher after a successful
        execution, so arbitrary filesystem paths can never be downloaded
        through this route.

        Args:
            job_id: Job identifier owning the artifact.
            filename: Basename of the artifact to download.

        Raises:
            HTTPException: 404 when the job is unknown or the file is not
                listed in the job's artifact manifest.

        Returns:
            FileResponse: The artifact file with a guessed media type.
        """
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=sanitize_log_text(f"Job '{job_id}' not found"),
            )

        allowed = {Path(name).name: path for name, path in manager.get_job_artifacts(job_id)}
        allowed.update({Path(path).name: path for path in manager.get_job_image_artifacts(job_id)})
        target = allowed.get(filename)
        if target is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=sanitize_log_text(f"Artifact '{filename}' not found for job '{job_id}'"),
            )

        media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return FileResponse(target, media_type=media_type, filename=filename)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main_engine:app",
        host=os.environ.get(ENGINE_HOST_ENV, "127.0.0.1"),
        port=int(os.environ.get(ENGINE_PORT_ENV, "8000")),
    )
