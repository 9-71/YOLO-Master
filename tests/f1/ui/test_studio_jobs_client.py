"""Regression tests for the stateless Gradio-to-Studio Job API boundary."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import gradio as gr
import pytest
import requests
from fastapi.testclient import TestClient

from api.v1.jobs import get_jobs_manager
from core.schema import JobStatus
from f1.jobs_manager import JobsManager
from f1.ui.jobs_tab import POLL_FAST_SECONDS, create_jobs_tab
from f1.ui.studio_jobs_client import StudioBackendUnavailableError, StudioJobsApiClient
from main_engine import create_app


class _Response:
    def __init__(self, payload: dict[str, Any] | None = None, content: bytes = b"") -> None:
        self._payload = payload
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        if self._payload is None:
            raise ValueError("binary response")
        return self._payload

    def iter_content(self, chunk_size: int):
        yield self._content


class _Session:
    def __init__(self, responses: list[dict[str, Any] | _Response]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        return response if isinstance(response, _Response) else _Response(response)


class _UnavailableSession:
    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        raise requests.ConnectionError("connection refused")


class _TestClientSession:
    def __init__(self, client: TestClient) -> None:
        self.client = client

    def request(self, method: str, url: str, **kwargs: Any):
        kwargs.pop("timeout", None)
        kwargs.pop("stream", None)
        parsed = urlsplit(url)
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        return self.client.request(method, path, **kwargs)


def test_connection_failure_is_structured_platform_error() -> None:
    """An unreachable API remains a platform error rather than a fake Job state."""
    client = StudioJobsApiClient("http://studio.test", session=_UnavailableSession())

    with pytest.raises(StudioBackendUnavailableError) as caught:
        client.submit_job("predict", "yolov8n.pt", "bus.jpg", "runs/predict", 0.25, "cpu", ["."])

    assert caught.value.code == "STUDIO_BACKEND_UNAVAILABLE"
    assert caught.value.path == "/api/v1/jobs"
    assert caught.value.status_code is None
    assert not hasattr(client, "jobs")


def test_submission_reaches_fastapi_manager_when_backend_is_running() -> None:
    """The API client registers a real request only in FastAPI's manager."""
    manager = JobsManager()
    manager._start_supervisors = lambda: None
    app = create_app()
    app.dependency_overrides[get_jobs_manager] = lambda: manager

    with TestClient(app) as api:
        client = StudioJobsApiClient("http://studio.test", session=_TestClientSession(api))
        job_id, _message = client.submit_job(
            "predict",
            "yolov8n.pt",
            "ultralytics/assets/bus.jpg",
            "runs/predict",
            0.25,
            "cpu",
            ["."],
        )
        assert job_id in manager.jobs
        assert manager.get_job_status(job_id)["status"] == "PENDING"
        assert not hasattr(client, "jobs")


@pytest.mark.parametrize(("task_type", "data_name"), [("predict", "image.jpg"), ("train", "dataset.yaml")])
def test_api_backed_ui_transitions_to_completed_with_artifacts(task_type, data_name, tmp_path, monkeypatch) -> None:
    """Train and predict polling refresh terminal state, artifacts and recent rows through FastAPI."""
    output_root = tmp_path / "runs"
    output_dir = output_root / task_type
    model_path = tmp_path / "model.pt"
    data_source = tmp_path / data_name
    manager = JobsManager(model_roots=[tmp_path], data_roots=[tmp_path], output_root=output_root)
    manager._start_supervisors = lambda: None
    app = create_app()
    app.dependency_overrides[get_jobs_manager] = lambda: manager

    with TestClient(app) as api:
        client = StudioJobsApiClient("http://studio.test", session=_TestClientSession(api))
        job_id, _message = client.submit_job(
            task_type,
            str(model_path),
            str(data_source),
            str(output_dir),
            0.25,
            "cpu",
            [str(tmp_path)],
        )
        tab = create_jobs_tab(client, "en")
        poll_timer = next(
            block for block in tab.blocks.values() if isinstance(block, gr.Timer) and block.value == POLL_FAST_SECONDS
        )
        poll_fn = next(bf.fn for bf in tab.fns.values() if bf.fn and (poll_timer._id, "tick") in bf.targets)

        manager.jobs[job_id].status = JobStatus.RUNNING
        running = poll_fn(job_id, "en")

        artifact_dir = output_dir / job_id
        artifact_dir.mkdir(parents=True)
        artifact_names = ["preview.png"]
        if task_type == "train":
            artifact_names.extend(f"artifact-{index:02d}.txt" for index in range(1, 16))
        for artifact_name in artifact_names:
            artifact = artifact_dir / artifact_name
            if artifact.suffix == ".png":
                artifact.write_bytes(
                    base64.b64decode(
                        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                    )
                )
            else:
                artifact.write_text("artifact\n", encoding="utf-8")
        manager.jobs[job_id].output.artifacts = artifact_names
        manager.jobs[job_id].status = JobStatus.COMPLETED
        completed = poll_fn(job_id, "en")

        assert running[0]["status"] == "RUNNING"
        assert running[-1].active is True
        assert completed[0]["status"] == "COMPLETED"
        assert completed[0]["artifact_count"] == len(artifact_names)
        assert len(completed[4]) == len(artifact_names)

        # The REST manifest is the public API boundary: it contains only safe
        # artifact IDs and delivery URLs, never source_path or server paths.
        manifest = api.get(f"/api/v1/jobs/{job_id}/artifacts").json()
        assert manifest["job_id"] == job_id
        assert len(manifest["artifacts"]) == len(artifact_names)
        assert manifest["image_artifacts"] == ["preview.png"]
        assert str(output_root.resolve()) not in str(manifest)
        for entry in manifest["artifacts"]:
            assert "source_path" not in entry
            assert not Path(entry["artifact_id"]).is_absolute()
            assert entry["download_url"].startswith("/static/artifacts/")

        # These dual-path values are local Gradio adapter state, not REST data.
        preview_path = completed[8]["value"]
        artifact_metadata = completed[10]
        assert Path(preview_path).is_file()
        assert not preview_path.startswith(("http://", "https://"))
        assert preview_path != str((artifact_dir / "preview.png").resolve())
        assert gr.Image(type="filepath").postprocess(preview_path).path
        assert completed[11]["value"].endswith("preview.png")
        assert completed[9][0][2] == "COMPLETED"
        assert completed[-1].active is False

        open_button = next(
            block for block in tab.blocks.values() if isinstance(block, gr.Button) and block.value == "📂 Open Folder"
        )
        open_fn = next(bf.fn for bf in tab.fns.values() if bf.fn and (open_button._id, "click") in bf.targets)
        opened: list[str] = []
        monkeypatch.setattr(os, "startfile", lambda path: opened.append(path), raising=False)
        monkeypatch.setattr("f1.ui.jobs_tab.subprocess.Popen", lambda command: opened.append(command[-1]))
        open_fn(job_id, artifact_metadata, "en")
        assert opened == [str(artifact_dir.resolve())]


def test_submit_job_uses_studio_api_without_local_job_state() -> None:
    """Gradio submission posts the contract and keeps no task registry."""
    session = _Session([{"job_id": "predict_api_001", "status": "pending"}])
    client = StudioJobsApiClient("http://studio.test", session=session)

    job_id, _message = client.submit_job(
        task_type="predict",
        model_path="yolov8n.pt",
        data_source="bus.jpg",
        output_dir="runs/predict",
        conf=0.25,
        device="cpu",
        allowed_paths=[".", "runs"],
    )

    assert job_id == "predict_api_001"
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", "http://studio.test/api/v1/jobs")
    assert kwargs["json"]["task_type"] == "predict"
    assert kwargs["json"]["security_constraints"]["allow_shell"] is False
    assert not hasattr(client, "jobs")
    assert not hasattr(client, "job_logs")


def test_monitoring_operations_read_only_from_studio_api() -> None:
    """Status, logs, artifacts, listing and cancellation all use REST routes."""
    session = _Session(
        [
            {
                "status": "running",
                "duration": 1.2,
                "started_at": "2026-09-11T00:00:00+00:00",
                "completed_at": None,
                "artifact_count": 0,
            },
            {"logs": ["submitted", "running"]},
            {
                "artifacts": [
                    {
                        "filename": "result.png",
                        "artifact_id": "nested/result.png",
                        "is_image": True,
                        "download_url": "/static/artifacts/job-1/nested/result.png",
                    }
                ]
            },
            {
                "artifacts": [
                    {
                        "filename": "result.png",
                        "artifact_id": "nested/result.png",
                        "is_image": True,
                        "download_url": "/static/artifacts/job-1/nested/result.png",
                    }
                ]
            },
            _Response(content=b"png"),
            {"jobs": [{"job_id": "job-1", "task_type": "predict", "status": "running", "created_at": "now"}]},
            {"message": "Cancellation requested"},
        ]
    )
    client = StudioJobsApiClient("http://studio.test", session=session)

    status = client.get_job_status("job-1")
    assert status["status"] == "RUNNING"
    assert status["duration"] == 1.2
    assert status["started_at"] == "2026-09-11T00:00:00+00:00"
    assert status["completed_at"] is None
    assert client.get_job_logs("job-1") == "submitted\nrunning"
    assert client.get_job_artifacts("job-1") == [
        ("nested/result.png", "http://studio.test/static/artifacts/job-1/nested/result.png")
    ]
    image_paths = client.get_job_image_artifacts("job-1")
    assert len(image_paths) == 1
    assert Path(image_paths[0]).name == "result.png"
    assert Path(image_paths[0]).read_bytes() == b"png"
    assert not image_paths[0].startswith(("http://", "https://"))
    assert client.list_recent_jobs()[0]["status"] == "RUNNING"
    assert client.cancel_job("job-1") == "Cancellation requested"

    get_calls = [kwargs for method, _url, kwargs in session.calls if method == "GET"]
    assert all(kwargs["headers"]["Cache-Control"] == "no-cache, no-store" for kwargs in get_calls)
    assert all(kwargs["headers"]["Pragma"] == "no-cache" for kwargs in get_calls)

    assert [(method, url.rsplit("http://studio.test", 1)[-1]) for method, url, _ in session.calls] == [
        ("GET", "/api/v1/jobs/job-1"),
        ("GET", "/api/v1/jobs/job-1/logs"),
        ("GET", "/api/v1/jobs/job-1/artifacts"),
        ("GET", "/api/v1/jobs/job-1/artifacts"),
        ("GET", "/static/artifacts/job-1/nested/result.png"),
        ("GET", "/api/v1/jobs"),
        ("POST", "/api/v1/jobs/job-1/cancel"),
    ]
