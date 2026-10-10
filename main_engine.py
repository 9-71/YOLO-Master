"""Independent localhost Studio Service. Run with ``python main_engine.py``.

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
from urllib.parse import urlsplit

import anyio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.v1.jobs import download_router, get_jobs_manager, router
from studio.job_logs import sanitize_log_text
from studio.jobs_manager import JobsManager

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
DEFAULT_CORS_ORIGINS = ("http://localhost:8000", "http://127.0.0.1:8000")
LOGGER = logging.getLogger("uvicorn.error")
CONFIGURATION_ERROR = "Invalid Studio Service configuration"
LEGACY_SERVICE_NAME = "YOLO-Master F1 Task Engine"


def local_origin(value):
    """Reject opaque origins, wildcard hosts, credentials and remote origins."""
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
            raw = os.environ.get("STUDIO_CORS_ORIGINS")
            return cls(
                host=os.environ.get("STUDIO_ENGINE_HOST", "127.0.0.1"),
                port=int(os.environ.get("STUDIO_ENGINE_PORT", "8000")),
                cors_origins=tuple(value.strip() for value in raw.split(",") if value.strip())
                if raw is not None
                else DEFAULT_CORS_ORIGINS,
                http_drain_seconds=float(os.environ.get("STUDIO_HTTP_DRAIN_SECONDS", "5")),
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
                and ("origin" not in headers or headers["origin"] in self.origins)
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


def production_manager():
    """Construct the only production owner, exclusively called by app lifespan."""
    return JobsManager(storage_path=os.environ.get("STUDIO_JOBS_STATE_PATH") or "runs/jobs_state.json")


def create_app(*, settings=None, manager_factory=None):
    """Create a pure Service; the trusted factory seam is for explicit tests/embedding."""
    settings = settings or ServiceSettings.from_environment()
    factory = manager_factory or production_manager

    @asynccontextmanager
    async def lifespan(application):
        if application.state.jobs_manager is not None:
            raise RuntimeError("Service already has an owned manager")
        try:
            manager = await anyio.to_thread.run_sync(factory)
        except Exception as exc:  # noqa: BLE001 - preserve failure and sanitize the ASGI boundary
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

    application = FastAPI(title="YOLO-Master Studio Task Service", version="0.1.0", lifespan=lifespan)
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
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
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

    @application.get("/health")
    def health(request: Request):
        manager = get_jobs_manager(request)
        snapshot = manager.get_service_snapshot()
        return {
            "status": "closing" if snapshot["closing"] else "ok",
            "service": LEGACY_SERVICE_NAME,
            "persistence_error": snapshot["persistence_error"],
            "runtime_shutdown_budget_seconds": snapshot["shutdown_budget_seconds"],
            "http_drain_seconds": settings.http_drain_seconds,
        }

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
