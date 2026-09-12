"""Focused pending-queue capacity tests for the Studio JobsManager."""

from __future__ import annotations

import threading

import pytest

from core.schema import JobRequest, JobStatus, TaskType
from f1.jobs_manager import JobsManager, QueueFullError


def _request(job_id: str, output_root, device: str = "cpu") -> JobRequest:
    """Build a path-valid request for queue-only tests."""
    return JobRequest(
        job_id=job_id,
        task_type=TaskType.DIAGNOSE,
        params={"device": device},
        output={"output_dir": str(output_root)},
    )


def test_pending_capacity_accepts_until_full_and_rejects_without_registration(tmp_path) -> None:
    """A full pending queue rejects the next request without creating a job."""
    manager = JobsManager(output_root=tmp_path, max_pending_jobs=1)
    manager._start_supervisors = lambda: None

    accepted = manager.submit_job_request(_request("accepted", tmp_path))
    assert accepted.status == JobStatus.PENDING

    with pytest.raises(QueueFullError, match="capacity exhausted") as caught:
        manager.submit_job_request(_request("rejected", tmp_path))

    assert caught.value.code == "QUEUE_FULL"
    assert manager.get_job("rejected") is None
    assert set(manager.jobs) == {"accepted"}


def test_running_jobs_do_not_consume_pending_capacity_and_dispatch_releases_it(tmp_path) -> None:
    """Moving a pending job to RUNNING immediately frees pending capacity."""
    manager = JobsManager(output_root=tmp_path, cpu_concurrency=1, gpu_concurrency=1, max_pending_jobs=1)
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()

    def execute(job_id: str) -> None:
        with manager.lock:
            manager.jobs[job_id].status = JobStatus.RUNNING
        if job_id == "first":
            first_started.set()
            assert release_first.wait(timeout=5)
        elif job_id == "second":
            second_started.set()
            assert release_second.wait(timeout=5)

    manager._execute_job = execute
    try:
        manager.submit_job_request(_request("first", tmp_path))
        assert first_started.wait(timeout=5)

        manager.submit_job_request(_request("second", tmp_path))
        assert manager.get_job("first").status == JobStatus.RUNNING
        assert manager.get_job("second").status == JobStatus.PENDING
        with pytest.raises(QueueFullError):
            manager.submit_job_request(_request("full", tmp_path))

        release_first.set()
        assert second_started.wait(timeout=5)
        assert manager.get_job("second").status == JobStatus.RUNNING
        manager.submit_job_request(_request("after-dispatch", tmp_path))
        assert manager.get_job("after-dispatch").status == JobStatus.PENDING
    finally:
        release_first.set()
        release_second.set()
        manager.shutdown()


def test_concurrent_submissions_share_one_capacity_check(tmp_path) -> None:
    """Concurrent submissions cannot both claim one pending slot."""
    manager = JobsManager(output_root=tmp_path, max_pending_jobs=1)
    manager._start_supervisors = lambda: None
    barrier = threading.Barrier(3)
    accepted: list[str] = []
    rejected: list[str] = []

    def submit(job_id: str) -> None:
        barrier.wait()
        try:
            manager.submit_job_request(_request(job_id, tmp_path))
            accepted.append(job_id)
        except QueueFullError:
            rejected.append(job_id)

    threads = [threading.Thread(target=submit, args=(f"job-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert set(manager.jobs) == set(accepted)


def test_max_pending_jobs_configuration_requires_positive_integer(monkeypatch) -> None:
    """The environment setting is honored and rejects non-positive values."""
    monkeypatch.setenv("F1_MAX_PENDING_JOBS", "2")
    manager = JobsManager()
    assert manager.max_pending_jobs == 2

    monkeypatch.setenv("F1_MAX_PENDING_JOBS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        JobsManager()
