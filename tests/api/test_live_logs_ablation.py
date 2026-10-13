"""Ablation tests for incremental Live Logs and cancellation log tails."""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from api.v1.jobs import get_jobs_manager
from core.schema import JobRequest, JobStatus, TaskType
from f1.jobs_manager import JobsManager
from main_engine import create_app


def _api_for(manager: JobsManager) -> TestClient:
    """Build an API client whose sole lifecycle owner is ``manager``."""
    app = create_app()
    app.dependency_overrides[get_jobs_manager] = lambda: manager
    return TestClient(app)


def test_running_log_polls_observe_each_new_append_exactly_once(monkeypatch, tmp_path) -> None:
    """A tail reached while RUNNING is not final: later polls see later child events in order."""
    import f1.jobs_manager as jobs_manager_module

    manager = JobsManager(output_root=tmp_path, model_roots=[tmp_path], data_roots=[tmp_path])
    job_id = "incremental-live-logs"
    job = JobRequest(job_id=job_id, task_type=TaskType.PREDICT, output={"output_dir": str(tmp_path)})
    manager.jobs[job_id] = job
    manager.job_logs[job_id] = ["submitted"]

    result = job.model_copy(deep=True)
    result.status = JobStatus.COMPLETED
    result.logs = ["first", "second", "third", "terminal"]

    release_second = threading.Event()
    release_third = threading.Event()
    release_result = threading.Event()
    second_waiting = threading.Event()
    third_waiting = threading.Event()
    result_waiting = threading.Event()
    first_published = threading.Event()
    second_published = threading.Event()
    third_published = threading.Event()
    receive_index = 0

    def observe_log_publication() -> None:
        published = manager.job_logs[job_id]
        if published[-1:] == ["first"]:
            first_published.set()
        elif published[-1:] == ["second"]:
            second_published.set()
        elif published[-1:] == ["third"]:
            third_published.set()

    monkeypatch.setattr(manager, "_save", observe_log_publication)

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
            nonlocal receive_index
            receive_index += 1
            if receive_index == 1:
                return "log", {"seq": 0, "text": "first", "terminal": False}
            if receive_index == 2:
                second_waiting.set()
                assert release_second.wait(timeout=5)
                return "log", {"seq": 1, "text": "second", "terminal": False}
            if receive_index == 3:
                third_waiting.set()
                assert release_third.wait(timeout=5)
                return "log", {"seq": 2, "text": "third", "terminal": False}
            result_waiting.set()
            assert release_result.wait(timeout=5)
            return "result", result.model_dump(mode="json")

        def stop(self, _grace: float) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(jobs_manager_module, "ManagedWorker", FakeWorker)
    execution = threading.Thread(target=manager._execute_job, args=(job_id,), daemon=True)
    execution.start()

    try:
        with _api_for(manager) as client:
            assert second_waiting.wait(timeout=5)
            assert first_published.wait(timeout=5)
            first = client.get(f"/api/v1/jobs/{job_id}/logs", params={"offset": 1}).json()
            assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "running"
            assert first["logs"] == ["first"]
            assert first["next_offset"] is None

            release_second.set()
            assert third_waiting.wait(timeout=5)
            assert second_published.wait(timeout=5)
            second = client.get(f"/api/v1/jobs/{job_id}/logs", params={"offset": 2}).json()
            assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "running"
            assert second["logs"] == ["second"]

            release_third.set()
            assert result_waiting.wait(timeout=5)
            assert third_published.wait(timeout=5)
            third = client.get(f"/api/v1/jobs/{job_id}/logs", params={"offset": 3}).json()
            assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "running"
            assert third["logs"] == ["third"]

            release_result.set()
            execution.join(timeout=5)
            assert not execution.is_alive()
            terminal = client.get(f"/api/v1/jobs/{job_id}/logs", params={"offset": 4}).json()

        observed = first["logs"] + second["logs"] + third["logs"] + terminal["logs"]
        assert observed == ["first", "second", "third", "terminal"]
        assert len(observed) == len(set(observed))
        assert manager.get_job(job_id).status == JobStatus.COMPLETED
    finally:
        release_second.set()
        release_third.set()
        release_result.set()
        execution.join(timeout=5)
        manager.shutdown()


def test_cancelled_job_keeps_existing_logs_and_accepts_late_log_tail(tmp_path) -> None:
    """Late log writes remain readable after cancellation without reviving the job."""
    manager = JobsManager(output_root=tmp_path, model_roots=[tmp_path], data_roots=[tmp_path])
    job_id = "cancel-late-logs"
    manager.jobs[job_id] = JobRequest(job_id=job_id, task_type=TaskType.PREDICT)
    manager.job_logs[job_id] = ["before cancellation"]

    try:
        assert "Cancellation requested" in manager.cancel_job(job_id)
        terminal_snapshot = manager.get_job(job_id).model_dump(mode="json")

        with _api_for(manager) as client:
            before_late_tail = client.get(f"/api/v1/jobs/{job_id}/logs").json()
            assert "before cancellation" in before_late_tail["logs"]

            manager._append_log(job_id, "late worker log one")
            manager._append_log(job_id, "late worker log two")

            late_tail = client.get(
                f"/api/v1/jobs/{job_id}/logs",
                params={"offset": before_late_tail["total"]},
            ).json()
            status = client.get(f"/api/v1/jobs/{job_id}").json()

        assert late_tail["logs"] == ["late worker log one", "late worker log two"]
        assert status["status"] == "cancelled"
        assert status["error_code"] == "USER_CANCELLED"
        assert manager.get_job(job_id).model_dump(mode="json") == terminal_snapshot
    finally:
        manager.shutdown()
