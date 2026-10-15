"""Production wiring, legacy configuration/history and unchanged HTTP contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api.v1.jobs import get_jobs_manager
from core.schema import ErrorInfo, JobRequest, JobStatus
from f1.jobs_manager import JobsManager as LegacyManager
from main_engine import CONFIGURATION_ERROR, ServiceSettings, create_app, production_manager
from studio.jobs_manager import JobsManager
from tests.studio.test_service_api import client_for, manager_for, payload

SUFFIXES = (
    "ENGINE_HOST",
    "ENGINE_PORT",
    "CORS_ORIGINS",
    "HTTP_DRAIN_SECONDS",
    "JOBS_STATE_PATH",
    "MODEL_ROOTS",
    "DATA_ROOTS",
    "OUTPUT_ROOT",
    "NETWORK_INPUT_HOSTS",
    "CPU_CONCURRENCY",
    "GPU_CONCURRENCY",
    "MAX_PENDING_JOBS",
    "STOP_GRACE_SECONDS",
    "SHUTDOWN_GRACE_SECONDS",
)


@pytest.fixture
def legacy_configuration(tmp_path, monkeypatch):
    for suffix in SUFFIXES:
        monkeypatch.delenv("F1_" + suffix, raising=False)
        monkeypatch.delenv("STUDIO_" + suffix, raising=False)
    for suffix, value in {
        "JOBS_STATE_PATH": str(tmp_path / "legacy.json"),
        "MODEL_ROOTS": str(tmp_path),
        "DATA_ROOTS": str(tmp_path),
        "OUTPUT_ROOT": str(tmp_path),
        "NETWORK_INPUT_HOSTS": "localhost",
        "CPU_CONCURRENCY": "1",
        "GPU_CONCURRENCY": "1",
        "MAX_PENDING_JOBS": "7",
        "STOP_GRACE_SECONDS": "0.1",
        "SHUTDOWN_GRACE_SECONDS": "0.4",
    }.items():
        monkeypatch.setenv("F1_" + suffix, value)
    return tmp_path


def test_default_production_lifespan_uses_studio_and_f1_configuration(legacy_configuration, monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Production created a legacy lifecycle owner")

    monkeypatch.setattr(LegacyManager, "__init__", reject)
    monkeypatch.setenv("F1_ENGINE_PORT", "8123")
    application = create_app()
    assert application.state.jobs_manager is None
    assert application.state.settings.port == 8123
    with client_for(application) as client:
        manager = application.state.jobs_manager
        assert type(manager) is JobsManager
        assert manager._store.path == legacy_configuration / "legacy.json"
        assert manager.max_pending_jobs == 7 and manager._limits == {"cpu": 1, "gpu": 1}
        assert manager._admission.network_input_hosts == frozenset({"localhost"})
        for url in ("/health", "/api/v1/jobs", "/api/v1/jobs/", "/openapi.json"):
            assert client.get(url).status_code == 200
            assert application.state.jobs_manager is manager
        assert client.get("/health").json() == {"status": "ok", "service": "YOLO-Master F1 Task Engine"}
    assert application.state.jobs_manager is None
    assert client.get("/api/v1/jobs").status_code == 503


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_conflicting_aliases_fail_closed_without_input(legacy_configuration, monkeypatch, suffix):
    monkeypatch.setenv("F1_" + suffix, "password=legacy-config-secret")
    monkeypatch.setenv("STUDIO_" + suffix, "token=modern-config-secret")
    with pytest.raises(ValueError) as caught:
        ServiceSettings.from_environment() if suffix in SUFFIXES[:4] else production_manager()
    assert str(caught.value) == CONFIGURATION_ERROR
    assert caught.value.__suppress_context__
    assert not (legacy_configuration / "legacy.json").exists()


def test_matching_aliases_and_existing_file_console(legacy_configuration, monkeypatch):
    monkeypatch.setenv("F1_ENGINE_PORT", "8123")
    monkeypatch.setenv("STUDIO_ENGINE_PORT", "8123")
    application = create_app()
    with client_for(application) as client:
        for origin in ("null", "http://localhost:5173", "http://127.0.0.1:3000", "http://localhost:8080"):
            preflight = client.options(
                "/api/v1/jobs",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type",
                },
            )
            assert preflight.status_code == 200
            assert preflight.headers["access-control-allow-origin"] == origin
            assert client.get("/api/v1/jobs", headers={"Origin": origin}).status_code == 200
        assert client.get("/").status_code == 200
        assert client.get("/favicon.ico").status_code == 200


def test_service_without_ui_assets_keeps_api(legacy_configuration, monkeypatch):
    monkeypatch.setattr("main_engine.FRONTEND_DIR", legacy_configuration / "absent-frontend")
    application = create_app()
    with client_for(application) as client:
        assert client.get("/").status_code == 404
        assert client.get("/favicon.ico").status_code == 404
        assert client.get("/api/v1/jobs").status_code == 200
        assert client.get("/docs").status_code == 200


def test_real_f1_snapshot_terminal_fields_and_absolute_manifest_preserved(legacy_configuration):
    root = legacy_configuration
    state = root / "legacy.json"
    legacy = LegacyManager(storage_path=state, model_roots=[root], data_roots=[root], output_root=root)
    legacy.shutdown()  # Offline fixture writer has no active execution.
    for status in JobStatus:
        job = JobRequest.model_validate(payload(root, "history-" + status.value))
        job.status = status
        job.metadata.created_at = "2026-09-11T00:00:00+00:00"
        if status not in {JobStatus.PENDING, JobStatus.RUNNING}:
            job.metadata.started_at = "2026-09-11T00:00:01+00:00"
            job.metadata.completed_at = job.metadata.started_at
            if status != JobStatus.COMPLETED:
                job.error = ErrorInfo(code="HISTORIC_REASON", message="historic result")
        if status == JobStatus.COMPLETED:
            artifact = root / job.job_id / "nested" / "result.txt"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("historic artifact", encoding="utf-8")
            job.output.artifacts = [str(artifact)]
        legacy.jobs[job.job_id] = job
        legacy.job_logs[job.job_id] = ["historic line one\nhistoric line two"]
    with legacy.lock:
        legacy._save()
    original = json.loads(state.read_text(encoding="utf-8"))
    with client_for(create_app()) as client:
        assert client.get("/api/v1/jobs").json()["total"] == len(JobStatus)
        for status in JobStatus:
            job_id = "history-" + status.value
            detail = client.get("/api/v1/jobs/" + job_id).json()
            current = client.app.state.jobs_manager.get_job(job_id).model_dump(mode="json")
            if status in {JobStatus.PENDING, JobStatus.RUNNING}:
                assert detail["status"] == "failed" and detail["error_code"] == "SERVICE_RESTARTED"
            else:
                assert current == original["jobs"][job_id]
                assert detail["duration"] == 0.0
            assert client.get(f"/api/v1/jobs/{job_id}/logs").json()["logs"][:2] == [
                "historic line one",
                "historic line two",
            ]
        manifest = client.get("/api/v1/jobs/history-completed/artifacts").json()
        assert manifest["artifacts"][0]["artifact_id"] == "nested/result.txt"
        assert client.get(manifest["artifacts"][0]["download_url"]).text == "historic artifact"
        assert client.get("/api/v1/jobs/history-completed").json()["artifact_count"] == 1
        assert client.post("/api/v1/jobs/history-completed/cancel").status_code == 200
    restored = json.loads(state.read_text(encoding="utf-8"))
    for status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        key = "history-" + status.value
        assert restored["jobs"][key] == original["jobs"][key]


def test_legacy_duplicate_invalid_and_not_found_error_shapes(tmp_path):
    manager = manager_for(tmp_path)
    manager._start_supervisors = lambda: None
    application = create_app()
    application.dependency_overrides[get_jobs_manager] = lambda: manager
    with client_for(application) as client:
        assert client.post("/api/v1/jobs", json=payload(tmp_path)).status_code == 201
        duplicate = client.post("/api/v1/jobs", json=payload(tmp_path))
        assert duplicate.status_code == 409
        assert isinstance(duplicate.json()["detail"], str) and "service-job" in duplicate.json()["detail"]
        rejected = client.post("/api/v1/jobs", json=payload(tmp_path / "..", "invalid"))
        assert rejected.status_code == 422 and isinstance(rejected.json()["detail"], str)
        for suffix in ("", "/logs", "/artifacts", "/cancel"):
            response = (
                client.post("/api/v1/jobs/missing" + suffix)
                if suffix == "/cancel"
                else client.get("/api/v1/jobs/missing" + suffix)
            )
            assert response.status_code == 404 and response.json() == {"detail": "Job 'missing' not found"}


def test_empty_legacy_settings_keep_mainline_defaults(legacy_configuration, monkeypatch):
    from main_engine import DEV_CORS_ORIGINS

    monkeypatch.chdir(legacy_configuration)
    for suffix in ("CORS_ORIGINS", "MODEL_ROOTS", "DATA_ROOTS", "OUTPUT_ROOT", "JOBS_STATE_PATH"):
        monkeypatch.setenv("F1_" + suffix, "")
    settings = ServiceSettings.from_environment()
    assert settings.cors_origins == DEV_CORS_ORIGINS
    manager = production_manager()
    assert manager._admission.model_roots == (legacy_configuration,)
    assert manager._admission.data_roots == (legacy_configuration,)
    assert manager._admission.output_root == legacy_configuration / "runs"
    assert manager._store.path == Path("runs/jobs_state.json")
    manager.shutdown()
    monkeypatch.setenv("STUDIO_CORS_ORIGINS", "")
    assert ServiceSettings.from_environment().cors_origins == ()


def test_concurrent_startup_cannot_create_second_owner(tmp_path):
    import asyncio
    import threading

    import anyio

    started, release = threading.Event(), threading.Event()
    managers = []

    def factory():
        started.set()
        assert release.wait(10)
        managers.append(manager_for(tmp_path))
        return managers[-1]

    application = create_app(manager_factory=factory)

    async def check():
        context = application.router.lifespan_context(application)
        first = asyncio.create_task(context.__aenter__())
        try:
            assert await anyio.to_thread.run_sync(started.wait, 5)
            with pytest.raises(RuntimeError, match="already has an owned manager"):
                async with application.router.lifespan_context(application):
                    raise AssertionError("Second owner entered")
        finally:
            release.set()
            await first
            await context.__aexit__(None, None, None)
        assert len(managers) == 1 and application.state.jobs_manager is None

    asyncio.run(check())


@pytest.mark.parametrize(
    "requested,expected,status",
    [
        ("bytes=0-3", b"0123", 206),
        ("bytes=-4", b"6789", 206),
        ("bytes=7-", b"789", 206),
        ("bytes=100-", b"", 416),
        ("items=0-3", b"Only support bytes range", 400),
    ],
)
def test_download_ranges_keep_file_response_contract(tmp_path, requested, expected, status):
    manager = manager_for(tmp_path)
    job = JobRequest.model_validate(payload(tmp_path, "range-job"))
    job.status = JobStatus.COMPLETED
    target = tmp_path / job.job_id / "result.txt"
    target.parent.mkdir()
    target.write_bytes(b"0123456789")
    job.output.artifacts = ["result.txt"]
    manager._jobs[job.job_id] = job
    with client_for(create_app(manager_factory=lambda: manager)) as client:
        url = "/static/artifacts/range-job/result.txt"
        whole = client.get(url)
        assert whole.content == b"0123456789" and whole.headers["content-length"] == "10"
        assert whole.headers["accept-ranges"] == "bytes"
        assert "etag" in whole.headers and "last-modified" in whole.headers
        result = client.get(url, headers={"Range": requested})
        assert result.status_code == status and result.content == expected
        if status == 206:
            assert int(result.headers["content-length"]) == len(expected)
            assert result.headers["content-range"].endswith("/10")
        for validator in (whole.headers["etag"], whole.headers["last-modified"]):
            assert client.get(url, headers={"Range": "bytes=0-3", "If-Range": validator}).content == b"0123"
        stale = client.get(url, headers={"Range": "bytes=0-3", "If-Range": '"stale"'})
        assert stale.status_code == 200 and stale.content == whole.content
        multi = client.get(url, headers={"Range": "bytes=0-1,8-9"})
        assert multi.status_code == 206 and "multipart/byteranges" in multi.headers["content-type"]
        assert b"01" in multi.content and b"89" in multi.content
        assert len(multi.content) == int(multi.headers["content-length"])
