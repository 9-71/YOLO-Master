"""Real truncated-frame failures must never block deadlines or process cleanup."""

import multiprocessing
import os
import struct
import time
from pathlib import Path

import pytest
from test_worker_lifecycle import live, terminal, wait_for

from core.schema import JobRequest, JobStatus
from studio.jobs_manager import JobsManager


def partial_frame_executor(job):
    """Write a real incomplete frame through the worker's private test-only IPC seam."""
    connection = next(
        cell.cell_contents
        for cell in job._log_event_sink.__closure__
        if isinstance(cell.cell_contents, multiprocessing.connection.Connection)
    )
    root = Path(job.output.output_dir)
    (root / "compute.pid").write_text(str(os.getpid()))
    if os.name == "nt":
        # Windows Pipe frames are messages; one short message advertises a longer
        # stream frame only on POSIX. Use a writer that stalls a normal receive
        # for the POSIX-specific protocol regression below.
        return job
    os.write(connection.fileno(), struct.pack("!i", 4096) + b"short")
    (root / "partial-sent").touch()
    if job.params["mode"] == "crash":
        os._exit(17)
    while True:
        time.sleep(0.05)


@pytest.mark.skipif(os.name == "nt", reason="POSIX stream-frame regression; Windows uses message-mode Pipe")
@pytest.mark.parametrize("mode", ["crash", "timeout", "cancel", "shutdown"])
def test_partial_ipc_does_not_block_owner_or_cleanup(tmp_path, mode):
    manager = JobsManager(output_root=tmp_path, stop_grace_seconds=0.2)
    manager._worker_executor = partial_frame_executor
    job = JobRequest(
        job_id="partial",
        task_type="predict",
        params={"mode": mode},
        output={"output_dir": str(tmp_path)},
        runtime_tracking={"timeout_seconds": 12 if mode == "timeout" else 30},
    )
    try:
        manager.submit_job_request(job)
        wait_for(lambda: (tmp_path / "partial-sent").exists())
        compute_pid = int((tmp_path / "compute.pid").read_text())
        if mode == "cancel":
            manager.request_cancel("partial")
        elif mode == "shutdown":
            manager.shutdown()
        result = terminal(manager, "partial")
        assert result.status == (JobStatus.CANCELLED if mode == "cancel" else JobStatus.FAILED)
        assert (
            result.error.code
            == {
                "crash": "WORKER_LOST",
                "timeout": "TIMEOUT",
                "cancel": "USER_CANCELLED",
                "shutdown": "SERVICE_SHUTDOWN",
            }[mode]
        )
        assert not live(compute_pid)
        assert manager._workers == {}
    finally:
        manager.shutdown()


def test_ipc_reader_launch_failure_still_cleans_gated_worker(tmp_path, monkeypatch):
    import threading

    from studio.worker_runtime import ManagedWorker

    manager = JobsManager(output_root=tmp_path, stop_grace_seconds=0.1)
    job = JobRequest(job_id="launch", task_type="diagnose", output={"output_dir": str(tmp_path)})
    manager._jobs[job.job_id] = job
    original_start = threading.Thread.start
    worker_start = ManagedWorker.start
    owned = []

    def fail_reader(thread):
        if thread.name.startswith("studio-ipc-"):
            raise RuntimeError("reader launch failed")
        return original_start(thread)

    def capture_worker(worker):
        try:
            return worker_start(worker)
        finally:
            if worker.process.pid is not None:
                owned.append(worker.process.pid)

    monkeypatch.setattr(threading.Thread, "start", fail_reader)
    monkeypatch.setattr(ManagedWorker, "start", capture_worker)
    manager._execute_job(job.job_id)
    assert manager.get_job(job.job_id).error.code == "EXECUTION_FAILED"
    assert manager._workers == {} and owned and not any(live(pid) for pid in owned)
