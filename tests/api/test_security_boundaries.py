"""Focused CPU-only tests for Studio server-side security boundaries."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.v1.jobs import get_jobs_manager
from core.schema import JobRequest, JobStatus, TaskType
from f1.handlers.predict import PredictHandler
from f1.jobs_manager import JobsManager
from main_engine import create_app


def _request(job_id: str, model: Path, data: Path, output: Path, **extra: object) -> JobRequest:
    params: dict[str, object] = {"model_path": str(model), "data_source": str(data), "device": "cpu"}
    params.update(extra)
    return JobRequest(
        job_id=job_id,
        task_type=TaskType.PREDICT,
        params=params,
        output={"output_dir": str(output)},
        security_constraints={"allowed_paths": ["C:/"], "allowed_path_patterns": [".*"]},
    )


def _require_real_symlink(link: Path) -> None:
    """Skip when symlink creation reported success but the link never landed.

    On some Windows/sandbox execution environments ``os.symlink`` /
    ``CreateSymbolicLinkW`` returns success without the reparse point ever
    being created (``exists``/``is_symlink`` stay False and ``resolve`` does
    not move). Such environments cannot exercise the symlink-escape rejection
    path, so asserting against a plain missing path would be a false failure.
    """
    if not link.is_symlink():
        pytest.skip(
            f"symlink at {link} did not materialize (is_symlink() is False) even though "
            "creation reported success; the current platform/execution environment does "
            "not create real symlinks"
        )


@pytest.mark.parametrize("job_id", ["abc", "job_001", "job-001", "A12_test"])
def test_job_id_accepts_safe_values(job_id: str) -> None:
    assert JobRequest(job_id=job_id, task_type=TaskType.DIAGNOSE).job_id == job_id


@pytest.mark.parametrize(
    "job_id",
    ["", ".", "..", "../x", "/tmp/a", r"C:\temp\a", "a/b", r"a\b", "a\nline", "a\x01b", "x" * 129],
)
def test_job_id_rejects_path_like_or_oversized_values(job_id: str) -> None:
    with pytest.raises(ValidationError):
        JobRequest(job_id=job_id, task_type=TaskType.DIAGNOSE)


def test_server_roots_override_client_whitelist_and_regex(tmp_path: Path) -> None:
    model_root, data_root, output_root, outside = (
        tmp_path / "models",
        tmp_path / "data",
        tmp_path / "outputs",
        tmp_path / "outside",
    )
    for directory in (model_root, data_root, output_root, outside):
        directory.mkdir()
    model = model_root / "model.pt"
    data = data_root / "image.jpg"
    model.touch()
    data.touch()
    manager = JobsManager(
        model_roots=[model_root],
        data_roots=[data_root],
        output_root=output_root,
        network_input_hosts=["camera.example.test"],
    )
    manager._start_supervisors = lambda: None  # This test covers admission, not execution.

    accepted = manager.submit_job_request(_request("safe-job", model, data, output_root / "predict"))
    assert accepted.security_constraints.allowed_path_patterns == []
    assert set(accepted.security_constraints.allowed_paths) == {str(model_root.resolve()), str(data_root.resolve())}
    network_request = _request("network-job", model, data, output_root / "predict")
    network_request.params["data_source"] = "rtsp://camera.example.test/live"
    assert manager.submit_job_request(network_request).params["data_source"].startswith("rtsp://")
    handler = PredictHandler()
    valid, message = handler.validate_params(network_request.params, accepted.security_constraints.model_dump())
    assert valid is True and message is None
    assert handler._normalize_sources(network_request.params["data_source"]) == ["rtsp://camera.example.test/live"]
    blocked_network = _request("blocked-network", model, data, output_root / "predict")
    blocked_network.params["data_source"] = "http://169.254.169.254/latest/meta-data"
    with pytest.raises(ValueError, match="network host"):
        manager.submit_job_request(blocked_network)

    with pytest.raises(ValueError, match="model_path"):
        manager.submit_job_request(_request("bad-model", outside / "model.pt", data, output_root / "predict"))
    with pytest.raises(ValueError, match="data_source"):
        manager.submit_job_request(_request("bad-data", model, outside / "image.jpg", output_root / "predict"))
    with pytest.raises(ValueError, match="output_dir"):
        manager.submit_job_request(_request("bad-output", model, data, outside / "predict"))
    with pytest.raises(ValueError, match="output_dir"):
        manager.submit_job_request(_request("bad-traversal", model, data, output_root / ".." / "outside"))
    with pytest.raises(ValueError, match="model_path"):
        manager.submit_job_request(_request("bad-windows", Path(r"C:\temp\outside.pt"), data, output_root))
    file_url = _request("bad-file-url", model, data, output_root)
    file_url.params["data_source"] = "file:///etc/passwd"
    with pytest.raises(ValueError, match="data_source"):
        manager.submit_job_request(file_url)


def test_symlink_escape_is_rejected_when_supported(tmp_path: Path) -> None:
    model_root, data_root, output_root, outside = (tmp_path / name for name in ("models", "data", "outputs", "outside"))
    for directory in (model_root, data_root, output_root, outside):
        directory.mkdir()
    data = data_root / "image.jpg"
    data.touch()
    link = model_root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable on this Windows environment: {exc}")
    _require_real_symlink(link)
    manager = JobsManager(model_roots=[model_root], data_roots=[data_root], output_root=output_root)
    with pytest.raises(ValueError, match="model_path"):
        manager.submit_job_request(_request("link-escape", link / "model.pt", data, output_root))
    output_link = output_root / "escape"
    try:
        output_link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable on this Windows environment: {exc}")
    _require_real_symlink(output_link)
    with pytest.raises(ValueError, match="output_dir"):
        manager.submit_job_request(_request("output-link-escape", model_root / "model.pt", data, output_link))


def test_artifact_symlink_escape_is_rejected_when_supported(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    job_id = "artifact-link-job"
    job_root = output_root / "predict" / job_id
    job_root.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = job_root / "escape.txt"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable on this Windows environment: {exc}")
    _require_real_symlink(link)
    job = JobRequest(job_id=job_id, task_type=TaskType.PREDICT, status=JobStatus.COMPLETED)
    job.output.output_dir = str(output_root / "predict")
    job.output.artifacts = [str(link)]
    manager = JobsManager(output_root=output_root)

    manager._normalize_job_artifacts(job)

    assert job.output.artifacts == []


def test_artifacts_are_job_contained_and_api_exposes_only_safe_ids(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    job_id = "artifact-job"
    job_root = output_root / "predict" / job_id
    (job_root / "one").mkdir(parents=True)
    (job_root / "two").mkdir()
    first = job_root / "one" / "result.txt"
    second = job_root / "two" / "result.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    job = JobRequest(job_id=job_id, task_type=TaskType.PREDICT, status=JobStatus.COMPLETED)
    job.output.output_dir = str(output_root / "predict")
    job.output.artifacts = [str(first), str(second), str(outside)]
    manager = JobsManager(output_root=output_root)
    manager._normalize_job_artifacts(job)
    manager.jobs[job_id] = job
    manager.job_logs[job_id] = []
    app = create_app()
    app.dependency_overrides[get_jobs_manager] = lambda: manager

    with TestClient(app) as client:
        response = client.get(f"/api/v1/jobs/{job_id}/artifacts")
        assert response.status_code == 200
        body = response.json()
        assert {item["artifact_id"] for item in body["artifacts"]} == {"one/result.txt", "two/result.txt"}
        assert all("source_path" not in item for item in body["artifacts"])
        assert str(tmp_path.resolve()) not in response.text
        assert client.get(f"/static/artifacts/{job_id}/one/result.txt").content == b"one"
        assert client.get(f"/static/artifacts/{job_id}/two/result.txt").content == b"two"
        assert client.get(f"/static/artifacts/{job_id}/../outside.txt").status_code == 404


def test_request_and_logs_are_redacted_before_state_file_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "known-env-secret-998877"
    dotenv_secret = "dotenv-only-secret-445566"
    monkeypatch.setenv("STUDIO_ACCESS_TOKEN", secret)
    monkeypatch.setenv("SHORT_TOKEN", "abc")
    (tmp_path / ".env").write_text(f"PRIVATE_API_KEY={dotenv_secret}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    model_root, data_root, output_root = (tmp_path / name for name in ("models", "data", "outputs"))
    for directory in (model_root, data_root, output_root):
        directory.mkdir()
    model, data = model_root / "model.pt", data_root / "image.jpg"
    model.touch()
    data.touch()
    state_path = tmp_path / "jobs.json"
    manager = JobsManager(
        storage_path=str(state_path), model_roots=[model_root], data_roots=[data_root], output_root=output_root
    )
    manager._start_supervisors = lambda: None  # This test covers persistence, not execution.
    request = _request(
        "redact-job",
        model,
        data,
        output_root / "predict",
        token="request-token-123456",
        api_key="request-api-key-123456",
        apikey="request-apikey-123456",
        password="hunter2-secret",
        passwd="passwd-secret-123456",
        secret="generic-secret-123456",
        authorization=f"Bearer {secret}",
        bearer="bearer-secret-123456",
        access_token="access-secret-123456",
        refresh_token="refresh-secret-123456",
        cookie="request-cookie-123456",
    )
    manager.submit_job_request(request)
    manager._append_log(
        "redact-job",
        f"stdout token={secret} password=hunter2-secret cookie=session-secret env={dotenv_secret}",
    )
    manager._append_log("redact-job", '{"token":"quoted-json-secret-112233"}')
    manager._append_log("redact-job", "short known value abc; alphabet must remain intact")
    deadline = time.monotonic() + 2
    while not state_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    raw = state_path.read_text(encoding="utf-8")
    assert secret not in raw
    assert "request-token-123456" not in raw
    assert "request-api-key-123456" not in raw
    assert "request-apikey-123456" not in raw
    assert "hunter2-secret" not in raw
    assert "passwd-secret-123456" not in raw
    assert "generic-secret-123456" not in raw
    assert "bearer-secret-123456" not in raw
    assert "access-secret-123456" not in raw
    assert "refresh-secret-123456" not in raw
    assert "request-cookie-123456" not in raw
    assert "session-secret" not in raw
    assert "quoted-json-secret-112233" not in raw
    assert "known value abc" not in raw
    assert "alphabet must remain intact" in raw
    assert dotenv_secret not in raw
    assert "***REDACTED***" in raw
    persisted = json.loads(raw)
    assert "env" not in persisted
    assert persisted["jobs"]["redact-job"]["params"]["token"] == "***REDACTED***"
