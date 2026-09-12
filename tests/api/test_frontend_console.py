"""Static console hosting integration tests for the standalone FastAPI engine (P2 Step 3).

This module verifies the zero-build verification console integration:
``GET /`` serves ``frontend/index.html`` (with its ``app.js`` asset) while the
API, health and docs routes declared before the static mount keep precedence,
unknown paths 404, and the ``null`` origin used by ``file://`` pages as well
as both engine origin spellings (``localhost``/``127.0.0.1``) pass the CORS
middleware. All tests are CPU-only: submissions run against an
in-memory :class:`f1.jobs_manager.JobsManager` whose dispatcher is stubbed,
so no model is loaded and no accelerator is touched.

Run:
    pytest tests/api/test_frontend_console.py -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.v1.jobs import get_jobs_manager
from core.schema import JobRequest
from f1.jobs_manager import JobsManager
from main_engine import CORS_ORIGINS_ENV, create_app


def _job_payload(job_id: str, task_type: str = "predict") -> dict[str, object]:
    """Build a valid ``JobRequest`` JSON payload with the given identity.

    Args:
        job_id: Unique job identifier for the payload.
        task_type: Task type value (``predict``, ``diagnose``, ...).

    Returns:
        dict[str, object]: JSON-serializable submission payload.
    """
    return {
        "job_id": job_id,
        "task_type": task_type,
        "params": {"model_path": "yolov8n.pt", "data_source": "bus.jpg", "conf": 0.25, "device": "cpu"},
        "output": {"output_dir": "runs/predict"},
        "security_constraints": {"allow_shell": False, "path_whitelisted": True, "allowed_paths": ["runs", "."]},
    }


def _noop_execute(job: JobRequest) -> JobRequest:
    """Dispatcher stub parking a job in PENDING: no engine work, no hardware."""
    return job


@pytest.fixture
def api_manager() -> JobsManager:
    """Return a fresh in-memory JobsManager with a no-engine dispatcher stub."""
    manager = JobsManager()
    manager.dispatcher.execute = _noop_execute  # type: ignore[method-assign]
    return manager


@pytest.fixture
def client(api_manager: JobsManager, monkeypatch: pytest.MonkeyPatch):
    """Yield a TestClient over a fresh app wired to the stubbed manager."""
    monkeypatch.delenv(CORS_ORIGINS_ENV, raising=False)
    app = create_app()
    app.dependency_overrides[get_jobs_manager] = lambda: api_manager
    with TestClient(app) as test_client:
        yield test_client


def test_root_serves_console_html(client: TestClient) -> None:
    """``GET /`` serves the verification console HTML, and ``/app.js`` its script."""
    index = client.get("/")
    assert index.status_code == 200
    assert index.headers["content-type"].startswith("text/html")
    assert "Verification Console" in index.text
    assert "app.js" in index.text
    script = client.get("/app.js")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]


def test_static_mount_does_not_shadow_api(client: TestClient) -> None:
    """API, health, docs and artifact routes keep precedence over the root mount."""
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "service": "YOLO-Master F1 Task Engine"}
    listed = client.get("/api/v1/jobs/")
    assert listed.status_code == 200
    assert listed.json()["jobs"] == []
    submitted = client.post("/api/v1/jobs/", json=_job_payload("shadow-001"))
    assert submitted.status_code == 201
    assert client.get("/api/v1/jobs/shadow-001").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_unknown_paths_404(client: TestClient) -> None:
    """Paths outside the API surface and frontend assets return 404."""
    assert client.get("/missing-page").status_code == 404
    assert client.get("/api/v1/jobs/unknown-job").status_code == 404
    assert client.get("/frontend/index.html").status_code == 404


def test_file_protocol_null_origin_cors(client: TestClient) -> None:
    """The ``null`` origin of ``file://`` console pages passes preflight and reads."""
    preflight = client.options(
        "/api/v1/jobs/",
        headers={"Origin": "null", "Access-Control-Request-Method": "POST"},
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "null"
    assert "POST" in preflight.headers["access-control-allow-methods"]
    read = client.get("/api/v1/jobs/", headers={"Origin": "null"})
    assert read.status_code == 200
    assert read.headers["access-control-allow-origin"] == "null"


def test_engine_origin_spelling_cors(client: TestClient) -> None:
    """Console pages served at one engine spelling may call the API via the other.

    The console is served at ``http://127.0.0.1:8000/`` but its default API
    base URL is ``http://localhost:8000``; browsers treat that pair as
    cross-origin and preflight it, so both spellings must pass CORS.
    """
    for origin in ("http://127.0.0.1:8000", "http://localhost:8000"):
        preflight = client.options(
            "/health",
            headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == origin
        assert "GET" in preflight.headers["access-control-allow-methods"]
    listed = client.get("/api/v1/jobs/", headers={"Origin": "http://localhost:8000"})
    assert listed.status_code == 200
    assert listed.headers["access-control-allow-origin"] == "http://localhost:8000"
