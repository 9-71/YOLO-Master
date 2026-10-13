"""Ablation tests for terminal immutability and the single Jobs lifecycle owner."""

from __future__ import annotations

import threading

import pytest

import api.v1.jobs as jobs_api
import f1.jobs_manager as jobs_manager_module
from app import YOLO_Master_WebUI
from core.schema import ErrorInfo, JobRequest, JobStatus, TaskType
from f1.jobs_manager import JobsManager
from f1.ui import jobs_tab
from f1.ui.studio_jobs_client import StudioJobsApiClient


@pytest.mark.parametrize(
    ("terminal_case", "status", "error"),
    [
        pytest.param("success", JobStatus.COMPLETED, None, id="success"),
        pytest.param(
            "failed",
            JobStatus.FAILED,
            ErrorInfo(code="ORIGINAL_FAILURE", message="original failure"),
            id="failed",
        ),
        pytest.param(
            "cancelled",
            JobStatus.CANCELLED,
            ErrorInfo(code="USER_CANCELLED", message="cancelled"),
            id="cancelled",
        ),
    ],
)
@pytest.mark.parametrize("late_outcome", ["completion", "exception"])
def test_published_terminal_state_rejects_late_worker_outcome(
    monkeypatch,
    tmp_path,
    terminal_case: str,
    status: JobStatus,
    error: ErrorInfo | None,
    late_outcome: str,
) -> None:
    """Neither a late worker result nor a late exception may replace an established terminal record."""
    import f1.jobs_manager as jobs_manager_module

    manager = JobsManager(output_root=tmp_path, model_roots=[tmp_path], data_roots=[tmp_path])
    job_id = f"{terminal_case}-late-{late_outcome}"
    job = JobRequest(job_id=job_id, task_type=TaskType.PREDICT, output={"output_dir": str(tmp_path)})
    manager.jobs[job_id] = job
    manager.job_logs[job_id] = ["published before late outcome"]

    late_result = job.model_copy(deep=True)
    late_result.status = JobStatus.COMPLETED
    late_result.logs = ["late completion"]
    receive_entered = threading.Event()
    release_outcome = threading.Event()

    class FakeProcess:
        @staticmethod
        def is_alive() -> bool:
            return True

    class FakeWorker:
        def __init__(self, *_args, **_kwargs) -> None:
            self.process = FakeProcess()

        def start(self) -> None:
            return None

        def receive(self):
            receive_entered.set()
            assert release_outcome.wait(timeout=5)
            if late_outcome == "exception":
                raise RuntimeError("late worker exception")
            return "result", late_result.model_dump(mode="json")

        def stop(self, _grace: float) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(jobs_manager_module, "ManagedWorker", FakeWorker)
    execution = threading.Thread(target=manager._execute_job, args=(job_id,), daemon=True)
    execution.start()

    try:
        assert receive_entered.wait(timeout=5)
        with manager.lock:
            current = manager.jobs[job_id]
            current.status = status
            current.error = error
            current.metadata.completed_at = "2026-09-13T00:00:00+00:00"
            current.output.artifacts = ["terminal-owner.txt"]
            if terminal_case == "cancelled":
                current.runtime_tracking.cancel_requested = True
            terminal_snapshot = current.model_dump(mode="json")

        release_outcome.set()
        execution.join(timeout=5)

        assert not execution.is_alive()
        assert manager.get_job(job_id).model_dump(mode="json") == terminal_snapshot
        assert job_id not in manager._workers
    finally:
        release_outcome.set()
        execution.join(timeout=5)
        manager.shutdown()


def test_ui_and_compat_export_do_not_create_a_second_lifecycle_owner(monkeypatch, tmp_path) -> None:
    """UI construction and compatibility lookup remain client-only and side-effect free."""
    constructions = []

    class GuardedJobsManager(JobsManager):
        def __init__(self, *args, **kwargs) -> None:
            constructions.append((args, kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(jobs_manager_module, "JobsManager", GuardedJobsManager)
    monkeypatch.setattr(jobs_api, "JobsManager", GuardedJobsManager)
    monkeypatch.setattr(jobs_api, "_manager", None)

    ui = YOLO_Master_WebUI(str(tmp_path / "checkpoints"))
    assert "JobsManager" in jobs_tab.__all__
    assert jobs_tab.JobsManager is GuardedJobsManager
    assert isinstance(ui.jobs_client, StudioJobsApiClient)
    assert constructions == []
    assert jobs_api._manager is None
    assert not hasattr(ui.jobs_client, "_workers")
    assert not hasattr(ui.jobs_client, "_supervisors")


def test_client_activity_after_api_shutdown_does_not_recreate_manager(monkeypatch) -> None:
    """A UI client request after owner shutdown cannot construct a second in-process manager."""

    class ExistingOwner:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "status": "cancelled",
                "duration": 1.0,
                "error_code": "USER_CANCELLED",
                "error_message": "cancelled",
                "artifact_count": 0,
            }

    class Session:
        @staticmethod
        def request(_method: str, _url: str, **_kwargs):
            return Response()

    owner = ExistingOwner()
    monkeypatch.setattr(jobs_api, "_manager", owner)
    jobs_api.shutdown_jobs_manager()
    assert owner.shutdown_calls == 1
    assert jobs_api._manager is None

    constructions = []

    def forbidden_manager(*args, **kwargs):
        constructions.append((args, kwargs))
        raise AssertionError("UI/client recreated an in-process JobsManager")

    monkeypatch.setattr(jobs_api, "JobsManager", forbidden_manager)
    client = StudioJobsApiClient("http://studio.test", session=Session())

    assert client.get_job_status("job-1")["status"] == "CANCELLED"
    assert constructions == []
    assert jobs_api._manager is None
