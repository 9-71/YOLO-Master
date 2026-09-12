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

The web root serves the zero-build verification console (``frontend/``):
visiting ``http://127.0.0.1:8000/`` loads the single-page dispatch/monitoring
UI with no separate Node.js server. The console can also be opened directly
from disk via ``file://`` — the default CORS allowlist includes the ``null``
origin browsers send for such pages, plus both engine origin spellings
(``localhost``/``127.0.0.1`` on port 8000) because the console's default API
base URL is ``http://localhost:8000``, making calls from the other spelling
cross-origin.

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
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.v1.jobs import get_jobs_manager, shutdown_jobs_manager
from api.v1.jobs import router as jobs_router
from core.security import sanitize_log_text
from f1.jobs_manager import JobsManager

APP_TITLE = "YOLO-Master F1 Task Engine"
APP_VERSION = "0.1.0"

#: Local frontend development origins allowed by the CORS middleware by default.
DEV_CORS_ORIGINS: tuple[str, ...] = (
    # The engine's own bind origins: the console (served at
    # http://127.0.0.1:8000/) defaults its API base URL to
    # http://localhost:8000, so browsers preflight calls between the two
    # host spellings as cross-origin and need both spellings allowlisted.
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    # The "null" origin covers the zero-build verification console opened
    # directly from disk (file://), where browsers send Origin: null on
    # cross-origin calls against the local engine.
    "null",
    "http://localhost:5173",  # Vite
    "http://127.0.0.1:5173",
    "http://localhost:3000",  # Create React App / Next.js
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
)

#: Environment variable overriding the CORS allowlist (comma-separated origins).
CORS_ORIGINS_ENV = "F1_CORS_ORIGINS"
#: Directory of the zero-build verification console served at the web root
#: (``frontend/index.html`` + ``app.js``, plain ES6 + Tailwind CDN).
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
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

    The returned app registers CORS for local frontend development origins
    (including the ``null`` origin used by ``file://`` pages), includes the
    ``/api/v1/jobs`` router, exposes the fail-closed artifact delivery route
    at ``/static/artifacts`` and mounts the zero-build verification console
    (``frontend/``) at the web root.

    Returns:
        FastAPI: The configured YOLO-Master F1 task engine application.

    Example:
        >>> from main_engine import create_app
        >>> app = create_app()
        >>> app.title
        'YOLO-Master F1 Task Engine'
    """

    @asynccontextmanager
    async def lifespan(app):
        # Load/reconcile persisted jobs at service startup, before serving requests.
        # Respect the existing dependency seam used by embedded apps and API tests.
        provider = app.dependency_overrides.get(get_jobs_manager, get_jobs_manager)
        manager = provider()
        try:
            yield
        finally:
            if provider is get_jobs_manager:
                shutdown_jobs_manager()
            else:
                manager.shutdown()

    app = FastAPI(
        lifespan=lifespan,
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
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(jobs_router)

    @app.get("/health", tags=["system"], summary="Liveness probe")
    def health() -> dict[str, str]:
        """Return a static liveness response (no dependency on job state)."""
        return {"status": "ok", "service": APP_TITLE}

    @app.get(
        "/static/artifacts/{job_id}/{artifact_id:path}",
        response_class=FileResponse,
        tags=["artifacts"],
        summary="Download one artifact file",
    )
    def serve_artifact(job_id: str, artifact_id: str, manager: JobsManager = MANAGER_DEPENDENCY) -> FileResponse:
        """Serve one artifact file, restricted to the job's manifest (fail-closed).

        The requested relative ``artifact_id`` must exactly match an entry in
        the job's validated artifact manifest. The
        manifest is populated exclusively by the dispatcher after a successful
        execution, so arbitrary filesystem paths can never be downloaded
        through this route.

        Args:
            job_id: Job identifier owning the artifact.
            artifact_id: Safe relative identifier from the job's artifact manifest.

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

        allowed = dict(manager.get_job_artifacts(job_id))
        target = allowed.get(artifact_id)
        if target is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=sanitize_log_text(f"Artifact '{artifact_id}' not found for job '{job_id}'"),
            )

        filename = Path(artifact_id).name
        media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return FileResponse(target, media_type=media_type, filename=filename)

    @app.get("/favicon.ico", include_in_schema=False, tags=["system"])
    def favicon() -> FileResponse:
        """Serve the official mascot favicon (frontend/favicon.png) at /favicon.ico."""
        return FileResponse(FRONTEND_DIR / "favicon.png", media_type="image/png")

    # Serve the zero-build verification console (frontend/) at the web root.
    # Registered last, so every API, health, docs and artifact route declared
    # above keeps precedence over the static mount.
    # Optional React console (web/dist) served alongside the zero-build console.
    REACT_DIST = Path(__file__).resolve().parent / "web" / "dist"
    if REACT_DIST.is_dir():
        app.mount("/console", StaticFiles(directory=REACT_DIST, html=True), name="react-console")
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main_engine:app",
        host=os.environ.get(ENGINE_HOST_ENV, "127.0.0.1"),
        port=int(os.environ.get(ENGINE_PORT_ENV, "8000")),
    )
