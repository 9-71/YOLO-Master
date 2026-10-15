"""Mainline FastAPI engine using the Studio Runtime. Run with ``python main_engine.py``.

A single FastAPI lifespan creates and closes JobsManager. No frontend, launcher,
Gradio or Agent dependencies are required. The supported entry point propagates
ASGI startup/shutdown failures to a nonzero exit; raw uvicorn CLI may not do so.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import signal
import threading
import traceback
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import anyio
import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.v1.jobs import download_router, get_jobs_manager, router
from studio.job_logs import sanitize_log_text
from studio.jobs_manager import JobsManager

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
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


DEFAULT_CORS_ORIGINS = DEV_CORS_ORIGINS
LOGGER = logging.getLogger("uvicorn.error")
CONFIGURATION_ERROR = "Invalid Studio Service configuration"
LEGACY_SERVICE_NAME = "YOLO-Master F1 Task Engine"


def local_origin(value):
    """Allow existing file console and explicit local HTTP(S) origins."""
    if value == "null":
        return True  # Preserve the existing trusted file:// console contract.
    parsed = urlsplit(value)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in LOCAL_HOSTS
        and not parsed.username
        and not parsed.password
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
        and parsed.port != 0
    )


@dataclass(frozen=True)
class ServiceSettings:
    """Immutable server-owned HTTP settings; clients cannot widen deployment scope."""

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    http_drain_seconds: float = 5

    def __post_init__(self):
        if self.host not in LOCAL_HOSTS:
            raise ValueError("Studio Service must bind to localhost, 127.0.0.1 or ::1")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("Service port must be between 1 and 65535")
        if not 0 <= self.http_drain_seconds <= 30:
            raise ValueError("HTTP drain budget must be finite and between 0 and 30 seconds")
        if any(not local_origin(value) for value in self.cors_origins):
            raise ValueError("CORS origins must be explicit local HTTP(S) origins")

    @classmethod
    def from_environment(cls):
        """Read trusted configuration without importing UI or constructing a manager."""
        try:
            raw = configuration_value("CORS_ORIGINS")
            return cls(
                host=configuration_value("ENGINE_HOST", "127.0.0.1"),
                port=int(configuration_value("ENGINE_PORT", "8000")),
                cors_origins=tuple(value.strip() for value in raw.split(",") if value.strip())
                if raw is not None
                else DEFAULT_CORS_ORIGINS,
                http_drain_seconds=float(configuration_value("HTTP_DRAIN_SECONDS", "5")),
            )
        except ValueError:
            # Numeric and URL parsers can include the original value in errors.
            raise ValueError(CONFIGURATION_ERROR) from None


class LocalBoundary:
    """Enforce local peer, literal Host and allowed Origin before requests reach an owner."""

    def __init__(self, app, origins):
        self.app, self.origins = app, frozenset(origins)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope["headers"]}
        try:
            peer = scope.get("client")
            host = urlsplit("http://" + headers.get("host", ""))
            allowed = (
                peer is not None
                and ipaddress.ip_address(peer[0]).is_loopback
                and host.hostname in LOCAL_HOSTS
                and host.username is None
                and host.password is None
                and not host.path
                and not host.query
                and not host.fragment
                and (scope["method"] == "OPTIONS" or "origin" not in headers or headers["origin"] in self.origins)
            )
            _ = host.port  # Validate malformed ports even when the hostname looks local.
        except ValueError:
            allowed = False
        if not allowed:
            await JSONResponse(
                {"detail": {"code": "LOCAL_ONLY", "message": "Service accepts trusted local requests only"}},
                status_code=403,
            )(scope, receive, send)
            return

        started = False

        async def secure_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                message["headers"] = list(message.get("headers", [])) + [
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                ]
            await send(message)

        try:
            await self.app(scope, receive, secure_send)
        except Exception:  # noqa: BLE001 - prevent the server from logging an unsanitized exception
            LOGGER.error("Studio request failed: %s", sanitize_log_text(traceback.format_exc()))
            if started:
                raise RuntimeError("Studio response stream failed") from None
            await JSONResponse(
                {"detail": {"code": "INTERNAL_ERROR", "message": "Service request failed"}}, status_code=500
            )(scope, receive, secure_send)


def configuration_value(suffix, default=None):
    """Keep legacy F1 aliases; reject contradictory dual configuration without echo."""
    modern, legacy = os.environ.get("STUDIO_" + suffix), os.environ.get("F1_" + suffix)
    if suffix in {"CORS_ORIGINS", "MODEL_ROOTS", "DATA_ROOTS", "OUTPUT_ROOT", "JOBS_STATE_PATH"} and legacy == "":
        legacy = None  # Mainline treated empty legacy settings as their defaults.
    if modern is not None and legacy is not None and modern.strip() != legacy.strip():
        raise ValueError(CONFIGURATION_ERROR) from None
    return modern if modern is not None else legacy if legacy is not None else default


def build_cors_origins():
    """Preserve the mainline default origins and explicit empty allowlists."""
    return list(ServiceSettings.from_environment().cors_origins)


def production_manager():
    """Construct Studio's sole production owner with compatible trusted configuration."""

    def roots(suffix, default):
        raw = configuration_value(suffix, default)
        entries = [value for value in raw.split(os.pathsep) if value.strip()]
        if not entries:
            raise ValueError(CONFIGURATION_ERROR)
        return entries

    try:
        return JobsManager(
            storage_path=configuration_value("JOBS_STATE_PATH") or "runs/jobs_state.json",
            model_roots=roots("MODEL_ROOTS", str(Path.cwd())),
            data_roots=roots("DATA_ROOTS", str(Path.cwd())),
            output_root=roots("OUTPUT_ROOT", str(Path.cwd() / "runs"))[0],
            network_input_hosts=configuration_value("NETWORK_INPUT_HOSTS", "").split(","),
            cpu_concurrency=int(configuration_value("CPU_CONCURRENCY", "2")),
            gpu_concurrency=int(configuration_value("GPU_CONCURRENCY", "1")),
            max_pending_jobs=int(configuration_value("MAX_PENDING_JOBS", "100")),
            stop_grace_seconds=float(configuration_value("STOP_GRACE_SECONDS", "2")),
            shutdown_grace_seconds=float(configuration_value("SHUTDOWN_GRACE_SECONDS", "60")),
        )
    except ValueError:
        raise ValueError(CONFIGURATION_ERROR) from None


def create_app(*, settings=None, manager_factory=None):
    """Create a pure Service; the trusted factory seam is for explicit tests/embedding."""
    settings = settings or ServiceSettings.from_environment()
    factory = manager_factory or production_manager

    @asynccontextmanager
    async def lifespan(application):
        if not application.state.lifespan_lock.acquire(blocking=False):
            raise RuntimeError("Service already has an owned manager")
        provider = application.dependency_overrides.get(get_jobs_manager, factory)
        try:
            manager = await anyio.to_thread.run_sync(provider)
        except Exception as exc:  # noqa: BLE001 - preserve failure and sanitize the ASGI boundary
            application.state.lifespan_lock.release()
            application.state.service_failure = "STARTUP_FAILED"
            LOGGER.error("Studio startup failed: %s", sanitize_log_text(str(exc)))
            raise RuntimeError("Studio startup failed") from None
        application.state.jobs_manager = manager
        application.state.ready = True
        application.state.service_failure = None
        application.state.runtime_shutdown_budget_seconds = manager.shutdown_budget_seconds
        try:
            yield
        finally:
            application.state.ready = False
            try:
                # Runtime owns its absolute cooperation deadline and bounded join.
                # Shield the cleanup thread from request/task cancellation; do not
                # abandon it via wait_for or clear the owner on a timeout/failure.
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(manager.shutdown)
            except Exception as exc:  # noqa: BLE001 - preserve failure and sanitize the ASGI boundary
                application.state.service_failure = "SHUTDOWN_FAILED"
                LOGGER.error("Studio shutdown failed; ownership retained: %s", sanitize_log_text(str(exc)))
                raise RuntimeError("Studio shutdown failed; ownership retained") from None
            else:
                application.state.jobs_manager = None
                application.state.lifespan_lock.release()

    application = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)
    application.state.lifespan_lock = threading.Lock()
    application.state.jobs_manager = None
    application.state.ready = False
    application.state.service_failure = None
    application.state.settings = settings
    application.include_router(router)
    application.include_router(download_router)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(LocalBoundary, origins=settings.cors_origins)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not echo input or arbitrary ctx: invalid payloads may contain secrets.
        details = [
            {"loc": list(item["loc"]), "type": item["type"], "msg": sanitize_log_text(item["msg"])}
            for item in exc.errors()
        ]
        return JSONResponse({"detail": details}, status_code=422)

    @application.get("/health", tags=["system"], summary="Liveness probe")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": APP_TITLE}

    # Keep main's existing UI entry points optional so the installed service
    # remains usable without Gradio, frontend or a built React bundle.
    if (FRONTEND_DIR / "favicon.png").is_file():

        @application.get("/favicon.ico", include_in_schema=False)
        def favicon():
            return FileResponse(FRONTEND_DIR / "favicon.png", media_type="image/png")

    react_dist = Path(__file__).resolve().parent / "web" / "dist"
    if react_dist.is_dir():
        application.mount("/console", StaticFiles(directory=react_dist, html=True), name="react-console")
    if FRONTEND_DIR.is_dir():
        application.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return application


try:
    app = create_app()
except ValueError:
    if __name__ == "__main__":
        LOGGER.error("%s", CONFIGURATION_ERROR)
        raise SystemExit(1) from None
    raise


class StudioServer(uvicorn.Server):
    """Finish signal-driven ASGI shutdown before deciding the service exit outcome."""

    @contextmanager
    def capture_signals(self):
        # Uvicorn replays captured signals after cleanup. A default SIGTERM/
        # SIGBREAK action can exit before the caller checks lifespan failure.
        # Keep its shutdown handler and restore prior handlers, without replay.
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        signals = [signal.SIGINT, signal.SIGTERM]
        if os.name == "nt":
            signals.append(signal.SIGBREAK)
        previous = {sig: signal.signal(sig, self.handle_exit) for sig in signals}
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def run_service(application=None):
    """Run one local server and propagate ASGI lifespan failures to the caller."""
    application = app if application is None else application
    settings = application.state.settings
    server = StudioServer(
        uvicorn.Config(
            application,
            host=settings.host,
            port=settings.port,
            workers=1,
            lifespan="on",
            proxy_headers=False,
            access_log=False,  # Default access logs echo untrusted query strings without sanitization.
            timeout_graceful_shutdown=settings.http_drain_seconds,
        )
    )
    server.run()
    if (
        application.state.service_failure
        or server.lifespan.startup_failed
        or server.lifespan.shutdown_failed
        or application.state.jobs_manager is not None
    ):
        raise RuntimeError("Studio Service lifespan failed")


if __name__ == "__main__":
    try:
        run_service()
    except RuntimeError as exc:
        LOGGER.error("%s", sanitize_log_text(str(exc)))
        raise SystemExit(1) from None
