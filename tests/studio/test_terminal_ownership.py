"""Ablation tests for terminal immutability and the single Jobs lifecycle owner."""

from __future__ import annotations

import threading

import pytest

import studio.jobs_manager as jobs_manager_module
from core.schema import ErrorInfo, JobRequest, JobStatus, TaskType
from studio.jobs_manager import JobsManager


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

    manager = JobsManager(output_root=tmp_path, model_roots=[tmp_path], data_roots=[tmp_path])
    job_id = f"{terminal_case}-late-{late_outcome}"
    job = JobRequest(job_id=job_id, task_type=TaskType.PREDICT, output={"output_dir": str(tmp_path)})
    manager._jobs[job_id] = job
    manager._logs.entries[job_id] = ["published before late outcome"]

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
            current = manager._jobs[job_id]
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
