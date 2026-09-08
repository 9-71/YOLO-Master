"""REST API integration tests for the standalone FastAPI engine (P2).

This module persists the 48 verification checks executed against
``main_engine.app`` while the P2 core engine was validated interactively.
The checks are organized into ten test functions — one per verified
behavior, six assertions each — and are intentionally CPU-only: every
submission runs against an in-memory :class:`f1.jobs_manager.JobsManager`
whose dispatcher is replaced by a stub, so no model is loaded and no
accelerator is ever touched.

Fixture wiring:
    ``client`` builds a fresh app with :func:`main_engine.create_app` and
    overrides the shared ``get_jobs_manager`` dependency (consumed by both
    the ``/api/v1/jobs`` router and the ``/static/artifacts`` route) with
    the stubbed manager from ``api_manager``.

Run:
    pytest tests/api/test_api_v1.py -v
"""

from __future__ import annotations

import re
import threading
import time

import pytest
from fastapi.testclient import TestClient

from api.v1.jobs import get_jobs_manager
from core.schema import ErrorInfo, JobRequest, JobStatus, TaskType
from f1.jobs_manager import JobsManager
from main_engine import CORS_ORIGINS_ENV, create_app

#: ISO 8601 timestamp shape stamped server-side by ``submit_job_request``.
_ISO_8601 = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(\+\d{2}:\d{2}|Z)?")


def _job_payload(job_id: str, task_type: str = "predict", **overrides: object) -> dict[str, object]:
    """Build a valid ``JobRequest`` JSON payload with the given identity.

    Args:
        job_id: Unique job identifier for the payload.
        task_type: Task type value (``predict``, ``diagnose``, ...).
        **overrides: Top-level field replacements (e.g. ``security_constraints``).

    Returns:
        dict[str, object]: JSON-serializable submission payload.
    """
    payload: dict[str, object] = {
        "job_id": job_id,
        "task_type": task_type,
        "params": {"model_path": "yolov8n.pt", "data_source": "bus.jpg", "conf": 0.25, "device": "cpu"},
        "output": {"output_dir": "runs/predict"},
        "security_constraints": {
            "allow_shell": False,
            "path_whitelisted": True,
            "allowed_paths": ["runs", "."],
        },
    }
    payload.update(overrides)
    return payload


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


def test_submit_job_success(client: TestClient) -> None:
    """Valid ``predict``/``diagnose`` submissions return 201 with the enforced schema."""
    predict = client.post("/api/v1/jobs/", json=_job_payload("predict-001", task_type="predict"))
    assert predict.status_code == 201
    body = predict.json()
    assert body["job_id"] == "predict-001"
    assert body["task_type"] == "predict"
    assert body["status"] == "pending"
    assert _ISO_8601.fullmatch(body["metadata"]["created_at"])
    diagnose = client.post("/api/v1/jobs/", json=_job_payload("diagnose-001", task_type="diagnose"))
    assert diagnose.status_code == 201 and diagnose.json()["task_type"] == "diagnose"


def test_submit_job_validation_and_conflict(client: TestClient) -> None:
    """Invalid payloads are rejected with 422 and duplicate ``job_id`` conflicts with 409."""
    bad_enum = _job_payload("bad-001", task_type="explode")
    assert client.post("/api/v1/jobs/", json=bad_enum).status_code == 422
    missing_field = {"job_id": "bad-002"}
    assert client.post("/api/v1/jobs/", json=missing_field).status_code == 422
    bad_type = _job_payload("bad-003", runtime_tracking={"timeout_seconds": "soon"})
    assert client.post("/api/v1/jobs/", json=bad_type).status_code == 422
    first = client.post("/api/v1/jobs/", json=_job_payload("dup-001"))
    assert first.status_code == 201
    duplicate = client.post("/api/v1/jobs/", json=_job_payload("dup-001"))
    assert duplicate.status_code == 409
    assert "dup-001" in duplicate.json()["detail"]


def test_collection_endpoints_accept_both_slash_spellings(client: TestClient) -> None:
    """``/api/v1/jobs`` and ``/api/v1/jobs/`` both list and submit (no redirects)."""
    bare = client.get("/api/v1/jobs")
    assert bare.status_code == 200
    assert bare.json()["jobs"] == []
    slashed = client.get("/api/v1/jobs/")
    assert slashed.status_code == 200
    assert slashed.json()["jobs"] == []
    assert client.post("/api/v1/jobs", json=_job_payload("slash-001")).status_code == 201
    assert client.post("/api/v1/jobs/", json=_job_payload("slash-002")).status_code == 201


def test_security_fail_closed(client: TestClient) -> None:
    """Dangerous payload flags are overridden server-side and traversal always 404s."""
    shell = client.post(
        "/api/v1/jobs/",
        json=_job_payload(
            "sec-001",
            security_constraints={"allow_shell": True, "path_whitelisted": False, "allowed_paths": ["runs"]},
        ),
    )
    assert shell.status_code == 201
    assert shell.json()["security_constraints"]["allow_shell"] is False
    assert shell.json()["security_constraints"]["path_whitelisted"] is True
    preloaded = client.post(
        "/api/v1/jobs/",
        json=_job_payload("sec-002", output={"output_dir": "runs/predict", "artifacts": ["/etc/passwd"]}),
    )
    assert preloaded.status_code == 201 and preloaded.json()["output"]["artifacts"] == []
    assert client.get("/static/artifacts/sec-001/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/static/artifacts/sec-001/../etc/passwd").status_code == 404


def _poll_status(client: TestClient, job_id: str, timeout: float = 10.0) -> dict[str, object]:
    """Poll ``GET /api/v1/jobs/{job_id}`` until it leaves PENDING (or times out)."""
    deadline = time.monotonic() + timeout
    while True:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body["status"] != "pending" or time.monotonic() >= deadline:
            return body
        time.sleep(0.05)


def test_job_status_and_lifecycle(client: TestClient, api_manager: JobsManager) -> None:
    """Status transitions (pending -> completed) and metadata are observable via GET."""
    gate = threading.Event()

    def _gated_execute(job: JobRequest) -> JobRequest:
        """Block until the test releases the gate, then complete the job."""
        gate.wait(timeout=10)
        job.status = JobStatus.COMPLETED
        return job

    api_manager.dispatcher.execute = _gated_execute  # type: ignore[method-assign]
    submitted = client.post("/api/v1/jobs/", json=_job_payload("life-001"))
    assert submitted.status_code == 201
    pending = client.get("/api/v1/jobs/life-001")
    assert pending.status_code == 200 and pending.json()["status"] == "pending"
    assert pending.json()["duration"] == "N/A"
    gate.set()
    body = _poll_status(client, "life-001")
    assert body["status"] == "completed"
    assert body["duration"].endswith("s")
    assert body["artifact_count"] == 0


def test_unhandled_execution_exception_populates_error(client: TestClient, api_manager: JobsManager) -> None:
    """An unhandled dispatcher exception fails the job with a non-null structured error."""

    def _raise(job: JobRequest) -> JobRequest:
        """Dispatcher stub raising an unexpected exception mid-execution."""
        raise KeyError("missing model param 'weights'")

    api_manager.dispatcher.execute = _raise  # type: ignore[method-assign]
    submitted = client.post("/api/v1/jobs/", json=_job_payload("boom-001"))
    assert submitted.status_code == 201
    body = _poll_status(client, "boom-001")
    assert body["status"] == "failed"
    assert body["error_code"] == "EXECUTION_FAILED"
    assert "missing model param" in body["error_message"]
    logs = client.get("/api/v1/jobs/boom-001/logs").json()["logs"]
    assert any("KeyError" in line and "missing model param" in line for line in logs)


def test_job_cancellation(client: TestClient, api_manager: JobsManager) -> None:
    """Cancellation returns 202 for active jobs and 409/404 for terminal or unknown jobs."""
    pending = JobRequest(job_id="cancel-pending", task_type=TaskType.PREDICT)
    completed = JobRequest(job_id="cancel-done", task_type=TaskType.PREDICT, status=JobStatus.COMPLETED)
    failed = JobRequest(
        job_id="cancel-failed",
        task_type=TaskType.PREDICT,
        status=JobStatus.FAILED,
        error=ErrorInfo(code="EXEC_ERR_500", message="boom"),
    )
    for job in (pending, completed, failed):
        api_manager.jobs[job.job_id] = job
        api_manager.job_logs[job.job_id] = []
    accepted = client.post("/api/v1/jobs/cancel-pending/cancel")
    assert accepted.status_code == 202
    assert accepted.json()["status"] == "cancel_requested"
    assert api_manager.jobs["cancel-pending"].runtime_tracking.cancel_requested is True
    assert client.post("/api/v1/jobs/cancel-done/cancel").status_code == 409
    assert client.post("/api/v1/jobs/cancel-failed/cancel").status_code == 409
    assert client.post("/api/v1/jobs/unknown/cancel").status_code == 404


def test_log_streaming_and_sanitization(client: TestClient, api_manager: JobsManager) -> None:
    """Log windows paginate by cursor and redact credential fragments in every line."""
    job_id = "log-001"
    api_manager.jobs[job_id] = JobRequest(job_id=job_id, task_type=TaskType.PREDICT)
    api_manager.job_logs[job_id] = [
        "[t0] Job log-001 submitted",
        "[t1] Exporting API_KEY=secret_key_12345678",
        "[t2] trace line A\nconnecting with Bearer abcdefghijklmnop1234\ntrace line C",
        "[t3] Completed",
    ]
    first = client.get(f"/api/v1/jobs/{job_id}/logs", params={"offset": 0, "limit": 3})
    assert first.status_code == 200
    body = first.json()
    assert body["total"] == 6
    assert body["logs"] == [
        "[t0] Job log-001 submitted",
        "[t1] Exporting API_KEY=***REDACTED***",
        "[t2] trace line A",
    ]
    assert body["next_offset"] == 3
    tail = client.get(f"/api/v1/jobs/{job_id}/logs", params={"offset": body["next_offset"], "limit": 3})
    assert tail.json()["logs"] == ["connecting with Bearer ***REDACTED***", "trace line C", "[t3] Completed"]
    assert tail.json()["next_offset"] is None


def test_artifacts_manifest_and_static_delivery(client: TestClient, api_manager: JobsManager, tmp_path) -> None:
    """Manifest lists real files with download refs; unlisted files 404, listed files stream."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    results = out_dir / "results.txt"
    results.write_text("hello artifacts", encoding="utf-8")
    preview = out_dir / "result.png"
    preview.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (out_dir / "secret.bin").write_bytes(b"unlisted")
    job_id = "art-001"
    job = JobRequest(job_id=job_id, task_type=TaskType.PREDICT, status=JobStatus.COMPLETED)
    job.output.output_dir = str(out_dir)
    job.output.artifacts = [str(results), str(preview)]
    api_manager.jobs[job_id] = job
    api_manager.job_logs[job_id] = []
    manifest = client.get(f"/api/v1/jobs/{job_id}/artifacts")
    assert manifest.status_code == 200
    body = manifest.json()
    assert len(body["artifacts"]) == 2
    assert {entry["filename"] for entry in body["artifacts"] if entry["is_image"]} == {"result.png"}
    assert body["artifacts"][0]["download_url"] == f"/static/artifacts/{job_id}/results.txt"
    assert client.get(f"/static/artifacts/{job_id}/secret.bin").status_code == 404
    delivered = client.get(f"/static/artifacts/{job_id}/results.txt")
    assert delivered.status_code == 200 and delivered.content == b"hello artifacts"


def test_cors_preflight(client: TestClient) -> None:
    """OPTIONS preflight honors the dev-origin allowlist and rejects foreign origins."""
    allowed = {"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
    preflight = client.options("/api/v1/jobs/", headers=allowed)
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "POST" in preflight.headers["access-control-allow-methods"]
    rejected = client.options(
        "/api/v1/jobs/",
        headers={"Origin": "http://evil.example.com", "Access-Control-Request-Method": "POST"},
    )
    assert rejected.status_code == 400
    assert "Disallowed CORS origin" in rejected.text
    posted = client.post(
        "/api/v1/jobs/",
        json=_job_payload("cors-001"),
        headers={"Origin": "http://localhost:5173"},
    )
    assert posted.headers["access-control-allow-origin"] == "http://localhost:5173"
