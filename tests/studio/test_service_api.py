"""Service contract, security and real Runtime IPC checks, separate from model acceptance."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.schema import JobRequest, JobStatus
from main_engine import ServiceSettings, create_app
from studio.jobs_manager import JobsManager

INVALID_SERVICE_SETTINGS = [
    ("STUDIO_ENGINE_PORT", "password=config-port-secret"),
    ("STUDIO_HTTP_DRAIN_SECONDS", "token=config-drain-secret"),
    ("STUDIO_CORS_ORIGINS", "http://localhost:config-cors-secret"),
    ("STUDIO_CORS_ORIGINS", "http://[config-origin-secret"),
    ("STUDIO_ENGINE_HOST", "config-host-secret"),
    ("STUDIO_ENGINE_PORT", "0"),
    ("STUDIO_HTTP_DRAIN_SECONDS", "nan"),
]


def service_executor(job):
    """Importable deterministic executor using real spawned Runtime/log IPC."""
    root = Path(job.output.output_dir)
    (root / f"{job.job_id}.ready").touch()
    if job.params.get("hold"):
        while not (root / f"{job.job_id}.release").exists():
            time.sleep(0.01)
    target = root / job.job_id / "nested" / "result.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("authorized-result", encoding="utf-8")
    (target.parent / "unlisted.txt").write_text("not-authorized", encoding="utf-8")
    for index in range(12):
        job.append_log(f"line-{index:02d}\npassword=test-secret-{index:02d}")
    job.status = JobStatus.COMPLETED
    job.append_log("terminal-tail-one\nterminal-tail-two", terminal=True)
    job.output.artifacts = [str(target)]
    return job


def wait_for(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError("Service condition did not arrive before the test deadline")


def payload(root, job_id="service-job", **params):
    return {
        "job_id": job_id,
        "task_type": "predict",
        "params": {"device": "cpu", **params},
        "output": {"output_dir": str(root)},
        "runtime_tracking": {"timeout_seconds": 120},
    }


def manager_for(root, **options):
    manager = JobsManager(
        storage_path=root / "state.json",
        model_roots=[root],
        data_roots=[root],
        output_root=root,
        cpu_concurrency=1,
        gpu_concurrency=1,
        stop_grace_seconds=0.1,
        shutdown_grace_seconds=0.4,
        **options,
    )
    manager._worker_executor = service_executor
    return manager


def client_for(application, **kwargs):
    return TestClient(application, base_url="http://localhost", client=("127.0.0.1", 40000), **kwargs)


@pytest.fixture
def service(tmp_path):
    managers = []

    def factory():
        manager = manager_for(tmp_path)
        managers.append(manager)
        return manager

    application = create_app(manager_factory=factory)
    with client_for(application) as client:
        yield client, managers[0], application


def test_owner_only_in_lifespan_and_startup_reconciliation(tmp_path):
    calls = []

    def factory():
        calls.append(manager_for(tmp_path))
        return calls[-1]

    application = create_app(manager_factory=factory)
    assert application.state.jobs_manager is None and calls == []
    with client_for(application) as client:
        owner = application.state.jobs_manager
        for _ in range(3):
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["service"] == "YOLO-Master F1 Task Engine"
            assert health.json()["status"] == "ok"
            assert client.get("/api/v1/jobs").status_code == 200
            assert application.state.jobs_manager is owner
        assert len(calls) == 1
        assert client.get("/").status_code == 404
        assert client.get("/frontend/index.html").status_code == 404
        assert client.get("/openapi.json").status_code == 200
    assert application.state.jobs_manager is None
    active = JobRequest.model_validate(payload(tmp_path, "previous-active"))
    active.status = JobStatus.RUNNING
    raw = {"jobs": {active.job_id: active.model_dump(mode="json")}, "job_logs": {}}
    (tmp_path / "state.json").write_text(json.dumps(raw), encoding="utf-8")
    with client_for(application) as client:
        response = client.get("/api/v1/jobs/previous-active").json()
        assert response["status"] == "failed" and response["error_code"] == "SERVICE_RESTARTED"
        assert len(calls) == 2
        assert not (tmp_path / "previous-active.ready").exists()
    persisted = json.loads((tmp_path / "state.json").read_text())
    assert persisted["jobs"][active.job_id]["error"]["code"] == "SERVICE_RESTARTED"


def test_no_lifespan_returns_503_without_lazy_owner(tmp_path):
    application = create_app(manager_factory=lambda: pytest.fail("Request constructed an owner"))
    client = client_for(application)
    assert client.get("/api/v1/jobs").status_code == 503
    assert application.state.jobs_manager is None


def test_rest_completion_logs_manifest_and_replay(service, tmp_path):
    client, manager, _ = service
    submitted = client.post("/api/v1/jobs/", json=payload(tmp_path))
    assert submitted.status_code == 201
    wait_for(lambda: client.get("/api/v1/jobs/service-job").json()["status"] == "completed")
    detail = client.get("/api/v1/jobs/service-job").json()
    assert detail["artifact_count"] == 1 and detail["duration"] >= 0
    assert detail["completed_at"] is not None
    page = client.get("/api/v1/jobs", params={"limit": 1, "offset": 0}).json()
    assert page["total"] == 1 and page["jobs"][0]["status"] == "completed"
    assert client.get("/api/v1/jobs/").json()["total"] == 1
    lines, offset = [], 0
    while True:
        page = client.get("/api/v1/jobs/service-job/logs", params={"offset": offset, "limit": 3}).json()
        lines.extend(page["logs"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] == offset + len(page["logs"])
        offset = page["next_offset"]
    assert [line for line in lines if line.startswith("line-")] == [f"line-{index:02d}" for index in range(12)]
    assert "terminal-tail-one" in lines and "terminal-tail-two" in lines
    assert "test-secret" not in "\n".join(lines)
    tail = client.get("/api/v1/jobs/service-job/logs", params={"offset": len(lines)}).json()
    assert tail["logs"] == [] and tail["next_offset"] is None
    # A real late buffer update extends a previous null cursor without changing terminal state.
    before = manager.get_job("service-job").model_dump()
    manager._append_log("service-job", "late-tail")
    assert client.get("/api/v1/jobs/service-job/logs", params={"offset": len(lines)}).json()["logs"] == ["late-tail"]
    assert manager.get_job("service-job").model_dump() == before
    manifest = client.get("/api/v1/jobs/service-job/artifacts").json()
    assert manifest["artifacts"][0]["artifact_id"] == "nested/result.txt"
    assert str(tmp_path) not in json.dumps(manifest)
    download = client.get(manifest["artifacts"][0]["download_url"])
    assert download.status_code == 200 and download.text == "authorized-result"
    assert download.headers["x-content-type-options"] == "nosniff"
    assert download.headers["cache-control"] == "no-store"
    for identifier in [
        "nested/unlisted.txt",
        "unlisted.txt",
        "nested%2F..%2Funlisted.txt",
        "C:%5Csecret",
        "nested%5Cresult.txt",
    ]:
        assert client.get(f"/static/artifacts/service-job/{identifier}").status_code == 404
    assert client.post("/api/v1/jobs/service-job/cancel").status_code == 200
    assert client.post("/api/v1/jobs", json=payload(tmp_path)).status_code == 409


def test_cancel_codes_pending_and_running_cleanup(service, tmp_path, monkeypatch):
    from studio.worker_runtime import ManagedWorker

    client, manager, _ = service
    stopped, release = threading.Event(), threading.Event()
    original = ManagedWorker.stop

    def gated_stop(worker, grace):
        stopped.set()
        assert release.wait(10)
        return original(worker, grace)

    monkeypatch.setattr(ManagedWorker, "stop", gated_stop)
    try:
        assert client.post("/api/v1/jobs", json=payload(tmp_path, hold=True)).status_code == 201
        wait_for(lambda: (tmp_path / "service-job.ready").exists())
        assert client.post("/api/v1/jobs", json=payload(tmp_path, "pending-job")).status_code == 201
        assert client.post("/api/v1/jobs/pending-job/cancel").status_code == 202
        assert client.get("/api/v1/jobs/pending-job").json()["status"] == "cancelled"
        assert not (tmp_path / "pending-job.ready").exists()
        ack = client.post("/api/v1/jobs/service-job/cancel")
        assert ack.status_code == 202 and ack.json()["status"] == "cancel_requested"
        assert stopped.wait(5)
        during = client.get("/api/v1/jobs/service-job").json()
        assert during["status"] == "running" and during["completed_at"] is None
        assert manager._workers  # Test proves actual owner retention during cleanup.
    finally:
        release.set()
    wait_for(lambda: client.get("/api/v1/jobs/service-job").json()["status"] == "cancelled")
    assert client.get("/api/v1/jobs/service-job").json()["error_code"] == "USER_CANCELLED"
    assert client.post("/api/v1/jobs/service-job/cancel").status_code == 200
    assert client.post("/api/v1/jobs/missing/cancel").status_code == 404


def test_non_cancellable_and_atomic_queue_full(service, tmp_path):
    client, manager, _ = service
    first = payload(tmp_path, hold=True)
    first["runtime_tracking"]["cancellable"] = False
    assert client.post("/api/v1/jobs", json=first).status_code == 201
    wait_for(lambda: (tmp_path / "service-job.ready").exists())
    assert client.post("/api/v1/jobs/service-job/cancel").status_code == 409
    manager.max_pending_jobs = 1
    assert client.post("/api/v1/jobs", json=payload(tmp_path, "pending")).status_code == 201
    response = client.post("/api/v1/jobs", json=payload(tmp_path, "overflow"))
    assert response.status_code == 429 and response.json()["detail"]["code"] == "QUEUE_FULL"
    assert client.get("/api/v1/jobs/overflow").status_code == 404


def test_admission_durability_and_error_sanitization(service, tmp_path, monkeypatch, caplog):
    client, manager, _ = service
    original = manager._store.save

    def broken(*args):
        raise OSError("disk password=secret-value https://user:pass@host/")

    monkeypatch.setattr(manager._store, "save", broken)
    response = client.post("/api/v1/jobs", json=payload(tmp_path))
    assert response.status_code == 503 and response.json()["detail"]["code"] == "PERSISTENCE_FAILED"
    assert client.get("/api/v1/jobs/service-job").status_code == 404
    assert not manager._supervisors and not (tmp_path / "service-job.ready").exists()
    assert "secret-value" not in response.text and "user:pass" not in response.text
    assert client.get("/health").json()["persistence_error"] is not None
    monkeypatch.setattr(manager._store, "save", original)
    assert client.post("/api/v1/jobs", json=payload(tmp_path)).status_code == 201
    wait_for(lambda: client.get("/api/v1/jobs/service-job").json()["status"] == "completed")
    assert client.get("/health").json()["persistence_error"] is None
    invalid = payload(tmp_path, "bad")
    invalid["task_type"] = "password=invalid-secret"
    response = client.post("/api/v1/jobs", json=invalid)
    assert response.status_code == 422 and "invalid-secret" not in response.text
    monkeypatch.setattr(manager, "list_jobs_snapshot", broken)
    response = client.get("/api/v1/jobs")
    assert response.status_code == 500 and "secret-value" not in response.text
    assert "secret-value" not in caplog.text and "user:pass" not in caplog.text


def test_server_owned_policy_and_manifest_symlink_escape(service, tmp_path):
    client, manager, _ = service
    hostile = payload(tmp_path, "hostile")
    hostile.update(
        security_constraints={
            "allow_shell": True,
            "allowed_paths": [str(tmp_path.parent)],
            "allowed_path_patterns": [".*"],
        }
    )
    hostile["output"]["output_dir"] = str(tmp_path.parent / "outside")
    assert client.post("/api/v1/jobs", json=hostile).status_code == 422
    hostile["output"]["output_dir"] = str(tmp_path)
    hostile["params"]["model_path"] = str(tmp_path.parent / "outside.pt")
    assert client.post("/api/v1/jobs", json=hostile).status_code == 422
    hostile["params"].pop("model_path")
    hostile["params"]["data_source"] = "https://remote.invalid/dataset"
    assert client.post("/api/v1/jobs", json=hostile).status_code == 422
    hostile["params"].pop("data_source")
    response = client.post("/api/v1/jobs", json=hostile)
    assert response.status_code == 201
    security = response.json()["security_constraints"]
    assert security["allow_shell"] is False and security["allowed_path_patterns"] == []
    assert security["allowed_paths"] == [str(tmp_path)]
    wait_for(lambda: manager.get_job("hostile").status == JobStatus.COMPLETED)
    target = tmp_path / "hostile" / "nested" / "result.txt"
    outside = tmp_path / "external-secret.txt"
    outside.write_text("do-not-download")
    target.unlink()
    try:
        target.symlink_to(outside)
    except OSError:
        # Windows symlinks may require a privilege; deletion is still fail-closed.
        assert client.get("/static/artifacts/hostile/nested/result.txt").status_code == 404
        return
    assert client.get("/static/artifacts/hostile/nested/result.txt").status_code == 404
    assert client.get("/api/v1/jobs/hostile/artifacts").json()["artifacts"] == []
    outside.unlink()
    assert client.get("/static/artifacts/hostile/nested/result.txt").status_code == 404


@pytest.mark.parametrize("url", ["/api/v1/jobs/missing", "/api/v1/jobs/missing/logs", "/api/v1/jobs/missing/artifacts"])
def test_unknown_job_404(service, url):
    assert service[0].get(url).status_code == 404


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.2", "remote.invalid", "127.0.0.1.evil"])
def test_remote_bind_configuration_rejected(host):
    with pytest.raises(ValueError):
        ServiceSettings(host=host)


@pytest.mark.parametrize(
    "origin",
    [
        "*",
        "null",
        "https://evil.invalid",
        "http://localhost.evil",
        "http://user:pass@localhost",
        "http://localhost/path",
    ],
)
def test_unsafe_cors_configuration_rejected(origin):
    with pytest.raises(ValueError):
        ServiceSettings(cors_origins=(origin,))


def test_local_peer_host_origin_and_cors(service):
    client, _, application = service
    assert client.get("/health", headers={"Host": "evil.invalid"}).status_code == 403
    assert client.get("/health", headers={"Host": "localhost:bad"}).status_code == 403
    for origin in ["null", "https://evil.invalid", "http://localhost.evil"]:
        assert client.post("/api/v1/jobs", headers={"Origin": origin}, json={}).status_code == 403
    with pytest.raises(RuntimeError, match="already has an owned manager"), client_for(application):
        pass


def test_http_remote_peer_without_second_lifespan(service):
    application = service[2]
    remote = TestClient(application, base_url="http://localhost", client=("10.0.0.2", 40000))
    assert remote.get("/health", headers={"X-Forwarded-For": "127.0.0.1"}).status_code == 403
    local = service[0]
    preflight = local.options(
        "/api/v1/jobs", headers={"Origin": "http://localhost:8000", "Access-Control-Request-Method": "POST"}
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:8000"
    assert "access-control-allow-credentials" not in preflight.headers


def test_startup_corruption_and_shutdown_failure_retains_owner(tmp_path):
    (tmp_path / "state.json").write_text("corrupt")
    application = create_app(manager_factory=lambda: manager_for(tmp_path))
    with pytest.raises(RuntimeError, match="startup failed"), client_for(application):
        pass
    assert application.state.service_failure == "STARTUP_FAILED"
    assert application.state.jobs_manager is None
    (tmp_path / "state.json").unlink()

    class FailingManager(JobsManager):
        def shutdown(self):
            raise RuntimeError("cleanup password=unsafe-secret")

    application = create_app(manager_factory=lambda: FailingManager(storage_path=None))
    with pytest.raises(RuntimeError, match="ownership retained"), client_for(application):
        owner = application.state.jobs_manager
    assert application.state.jobs_manager is owner and not application.state.ready
    assert application.state.service_failure == "SHUTDOWN_FAILED"


@pytest.mark.parametrize("key,value", INVALID_SERVICE_SETTINGS)
def test_invalid_environment_configuration_has_fixed_error(monkeypatch, key, value):
    for name in {item[0] for item in INVALID_SERVICE_SETTINGS}:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError) as error:
        ServiceSettings.from_environment()
    assert str(error.value) == "Invalid Studio Service configuration"
    assert error.value.__suppress_context__


@pytest.mark.parametrize("limit,offset", [(10, 0), (2, 1), (0, 0), (10, 20), (-1, -1)])
def test_summary_interfaces_preserve_order_case_and_detachment(tmp_path, limit, offset):
    manager = manager_for(tmp_path)
    for index, status in enumerate(JobStatus):
        job = JobRequest.model_validate(payload(tmp_path, f"summary-{index}"))
        job.status = status
        job.metadata.created_at = f"2026-10-10T00:00:0{index}+00:00"
        job.metadata.started_at = "2026-10-10T00:01:00+00:00"
        job.metadata.completed_at = job.metadata.started_at if index == 0 else None
        manager._jobs[job.job_id] = job  # Detached summary seam; no workers or admission.
    recent = manager.list_recent_jobs()
    page = manager.list_jobs_snapshot(limit=limit, offset=offset)
    assert page["total"] == len(JobStatus) and page["limit"] == limit and page["offset"] == offset
    assert page["jobs"] == [dict(item, status=item["status"].lower()) for item in recent[offset : offset + limit]]
    assert [item["job_id"] for item in recent] == [f"summary-{index}" for index in reversed(range(len(JobStatus)))]
    assert all(item["status"] == item["status"].upper() for item in recent)
    assert recent[-1]["duration"] == 0.0
    recent[0]["status"] = "changed"
    if page["jobs"]:
        page["jobs"][0]["job_id"] = "changed"
    assert manager.list_recent_jobs()[0]["status"] != "changed"
    assert all(item["job_id"] != "changed" for item in manager.list_jobs_snapshot()["jobs"])
