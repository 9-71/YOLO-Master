"""Runtime admission, detached snapshots, owner-only outcomes and real stop IPC."""

import ast
import json
import threading
import time
import traceback
from pathlib import Path

import pytest
from test_worker_lifecycle import terminal, wait_for

from core.schema import JobRequest, JobStatus
from studio.jobs_manager import CancelCode, JobsManager, compute_duration


def request(root, job_id="job", **params):
    return JobRequest(
        job_id=job_id, task_type="predict", output={"output_dir": str(root)}, params={"device": "cpu", **params}
    )


def stopped_tail_executor(job):
    """Produce logs only after the real worker stop pipe is delivered."""
    root = Path(job.output.output_dir)
    (root / "started").touch()
    while not job.runtime_tracking.cancel_requested:
        time.sleep(0.01)
    for sequence in range(80):
        job.append_log(f"stop-tail-{sequence} API_KEY=secret_key_12345678")
    (root / "tail-sent").touch()
    job.status = JobStatus.COMPLETED
    job.append_log("late completed result")
    return job


def altered_outcome_executor(job):
    """Attempt to replace parent-owned identity, policy and lifecycle timestamps."""
    root = Path(job.output.output_dir) / job.job_id
    root.mkdir(parents=True)
    (root / "authorized.txt").write_text("result")
    job.output.artifacts = [str(root / "authorized.txt")]
    job.output.output_dir = str(root.parent / "other")
    job.params["device"] = "malicious"
    job.security_constraints.allow_shell = True
    job.metadata.created_at = job.metadata.started_at = job.metadata.completed_at = "untrusted"
    job.metadata.tags = ["untrusted"]
    job.status = JobStatus.COMPLETED
    return job


def test_submit_and_queries_are_detached_sanitized_snapshots(tmp_path, monkeypatch):
    manager = JobsManager(output_root=tmp_path)
    monkeypatch.setattr(manager, "_start_supervisors", lambda: None)
    original = request(tmp_path, api_key="secret_key_12345678")
    returned = manager.submit_job_request(original)
    original.params["device"] = "external"
    returned.output.artifacts.append("injected")
    returned.metadata.tags.append("external")
    queried = manager.get_job("job")
    queried.runtime_tracking.cancel_requested = True
    assert manager.get_job("job").params["device"] == "cpu"
    assert manager.get_job("job").output.artifacts == []
    assert manager.get_job("job").metadata.tags == []
    assert not manager.get_job("job").runtime_tracking.cancel_requested
    assert "secret_key_12345678" not in returned.model_dump_json()
    assert manager._jobs["job"].params["api_key"] == "secret_key_12345678"
    manager.shutdown()


def test_worker_outcome_cannot_replace_parent_record(tmp_path):
    manager = JobsManager(output_root=tmp_path)
    manager._worker_executor = altered_outcome_executor
    try:
        submitted = manager.submit_job_request(request(tmp_path))
        result = terminal(manager, "job")
        assert result.status == JobStatus.COMPLETED
        assert result.params["device"] == "cpu"
        assert result.output.output_dir == str(tmp_path.resolve())
        assert not result.security_constraints.allow_shell
        assert result.metadata.created_at == submitted.metadata.created_at
        assert result.metadata.started_at != "untrusted" and result.metadata.completed_at != "untrusted"
        assert result.metadata.tags == []
        assert result.output.artifacts == ["authorized.txt"]
    finally:
        manager.shutdown()


@pytest.mark.parametrize("reason", ["cancel", "shutdown"])
def test_real_stop_pipe_drains_late_logs_without_accepting_late_result(tmp_path, reason):
    manager = JobsManager(output_root=tmp_path, stop_grace_seconds=1)
    manager._worker_executor = stopped_tail_executor
    try:
        manager.submit_job_request(request(tmp_path))
        wait_for(lambda: (tmp_path / "started").exists())
        if reason == "cancel":
            assert manager.request_cancel("job").code == CancelCode.ACCEPTED
        else:
            manager.shutdown()
        result = terminal(manager, "job")
        assert result.error.code == ("USER_CANCELLED" if reason == "cancel" else "SERVICE_SHUTDOWN")
        assert (tmp_path / "tail-sent").exists()
        lines = manager.get_job_log_lines("job")
        tails = [line for line in lines if line.startswith("stop-tail-")]
        assert tails == [f"stop-tail-{sequence} API_KEY=***REDACTED***" for sequence in range(80)]
        assert "secret_key_12345678" not in json.dumps(lines)
        offset, drained = 0, []
        while True:
            page = manager.get_job_log_page("job", offset, 7)
            drained.extend(page["logs"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        assert drained == lines
        frozen = manager.get_job("job").model_dump()
        manager._append_log("job", "post-terminal tail")
        assert manager.get_job("job").model_dump() == frozen
        assert manager.get_job_log_page("job", len(lines), 7)["logs"] == ["post-terminal tail"]
    finally:
        manager.shutdown()


@pytest.mark.parametrize("path_kind", ["model", "data", "output"])
def test_client_cannot_expand_server_owned_roots(tmp_path, monkeypatch, path_kind):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    outside = tmp_path / "trusted-sibling" / "escape"
    manager = JobsManager(model_roots=[trusted], data_roots=[trusted], output_root=trusted)
    monkeypatch.setattr(manager, "_start_supervisors", lambda: None)
    job = request(trusted)
    job.security_constraints.allowed_paths = [str(tmp_path)]
    job.security_constraints.allowed_path_patterns = [".*", "["]
    job.security_constraints.allow_shell = True
    if path_kind == "output":
        job.output.output_dir = str(outside)
    else:
        job.params["model_path" if path_kind == "model" else "data_source"] = str(outside)
    with pytest.raises(ValueError, match="trusted roots"):
        manager.submit_job_request(job)
    assert manager.get_job("job") is None


def test_admission_discards_client_policy_manifest_and_state(tmp_path, monkeypatch):
    manager = JobsManager(model_roots=[tmp_path], data_roots=[tmp_path], output_root=tmp_path)
    monkeypatch.setattr(manager, "_start_supervisors", lambda: None)
    job = request(tmp_path)
    job.status = JobStatus.COMPLETED
    job.runtime_tracking.cancel_requested = True
    job.security_constraints.allow_shell = True
    job.security_constraints.allowed_path_patterns = [".*"]
    job.security_constraints.allowed_paths = ["/"]
    job.output.artifacts = ["injected.txt"]
    admitted = manager.submit_job_request(job)
    assert admitted.status == JobStatus.PENDING and admitted.output.artifacts == []
    assert not admitted.runtime_tracking.cancel_requested
    assert not admitted.security_constraints.allow_shell
    assert admitted.security_constraints.allowed_paths == [str(tmp_path.resolve())]
    assert admitted.security_constraints.allowed_path_patterns == []
    manager.shutdown()


def test_typed_cancellation_and_cancel_priority_over_closing(tmp_path, monkeypatch):
    manager = JobsManager(output_root=tmp_path)
    monkeypatch.setattr(manager, "_start_supervisors", lambda: None)
    assert manager.request_cancel("missing").code == CancelCode.NOT_FOUND
    manager.submit_job_request(request(tmp_path))
    accepted = manager.request_cancel("job")
    assert accepted.code == CancelCode.ACCEPTED and accepted.job.error.code == "USER_CANCELLED"
    frozen = manager.get_job("job").model_dump()
    assert manager.request_cancel("job").code == CancelCode.ALREADY_TERMINAL
    manager.shutdown()
    assert manager.get_job("job").model_dump() == frozen


def test_runtime_import_direction_is_transport_and_agent_free():
    root = Path(__file__).resolve().parents[2]
    for name in ("jobs_manager", "worker_runtime", "admission", "job_store", "job_logs", "artifacts"):
        tree = ast.parse((root / "studio" / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert not any(
                module.split(".")[0] in {"api", "fastapi", "gradio", "app", "agent", "f1"} for module in modules
            )


def test_duration_excludes_queue_wait_and_preserves_zero():
    assert compute_duration(None, "2026-10-07T00:01:00+00:00") is None
    assert compute_duration("2026-10-07T00:01:00+00:00", "2026-10-07T00:01:00+00:00") == 0.0


def test_queue_wait_does_not_consume_execution_timeout(tmp_path):
    from test_worker_lifecycle import TASK_TIMEOUT, dummy_executor

    manager = JobsManager(output_root=tmp_path, cpu_concurrency=1, gpu_concurrency=1, stop_grace_seconds=0.1)
    manager._worker_executor = dummy_executor
    try:
        first = request(tmp_path, "first", children=False)
        manager.submit_job_request(first)
        wait_for(lambda: (tmp_path / "first.started").exists())
        queued = request(tmp_path, "queued", children=False)
        queued.runtime_tracking.timeout_seconds = TASK_TIMEOUT
        manager.submit_job_request(queued)
        time.sleep(TASK_TIMEOUT + 0.2)
        assert manager.get_job("queued").status == JobStatus.PENDING
        assert manager.get_job("queued").metadata.started_at is None
        (tmp_path / "queued.release").touch()
        (tmp_path / "first.release").touch()
        assert terminal(manager, "queued").status == JobStatus.COMPLETED
    finally:
        manager.shutdown()


def test_accepted_running_cancel_wins_concurrent_service_closing(tmp_path):
    from test_worker_lifecycle import dummy_executor

    manager = JobsManager(output_root=tmp_path, stop_grace_seconds=0.2)
    manager._worker_executor = dummy_executor
    try:
        manager.submit_job_request(request(tmp_path, children=False))
        wait_for(lambda: (tmp_path / "job.started").exists())
        assert manager.request_cancel("job").code == CancelCode.ACCEPTED
        manager.shutdown()
        assert manager.get_job("job").status == JobStatus.CANCELLED
        assert manager.get_job("job").error.code == "USER_CANCELLED"
    finally:
        manager.shutdown()


def test_handle_close_failure_retains_owner_and_slot_until_retry(tmp_path, monkeypatch):
    import studio.jobs_manager as module

    manager = JobsManager(output_root=tmp_path)
    job = request(tmp_path)
    manager._jobs["job"] = job
    failure_seen, allow_close = threading.Event(), threading.Event()
    result = job.model_copy(deep=True)
    result.status = JobStatus.COMPLETED

    class Worker:
        def __init__(self, *_):
            self.calls = 0

        def start(self):
            return None

        def receive(self):
            return "result", result.model_dump(mode="json")

        def stop(self, _):
            return None

        def close(self):
            self.calls += 1
            if self.calls == 1:
                failure_seen.set()
                raise OSError("transient handle close failure")
            assert allow_close.wait(5)

    monkeypatch.setattr(module, "ManagedWorker", Worker)
    supervisor = threading.Thread(target=manager._execute_job, args=("job",), daemon=True)
    supervisor.start()
    try:
        assert failure_seen.wait(5)
        wait_for(lambda: manager.get_job("job").error is not None)
        assert manager.get_job("job").error.code == "WORKER_STOP_FAILED"
        assert manager.get_job("job").status == JobStatus.RUNNING
        assert "job" in manager._workers and supervisor.is_alive()
    finally:
        allow_close.set()
        supervisor.join(5)
    assert not supervisor.is_alive()
    assert manager.get_job("job").status == JobStatus.COMPLETED
    assert manager._workers == {}


@pytest.fixture
def completed_worker(monkeypatch):
    """Complete through the real supervisor while recording cleanup boundaries."""
    import studio.jobs_manager as module

    events = []

    class Worker:
        def __init__(self, job, *_):
            self.result = job.model_copy(deep=True)
            self.job_id = job.job_id

        def start(self):
            root = Path(self.result.output.output_dir) / self.job_id
            root.mkdir(parents=True)
            (root / "authorized.txt").write_text("result")
            self.result.output.artifacts = ["authorized.txt"]
            self.result.logs = ["worker completed"]
            self.result.status = JobStatus.COMPLETED
            events.append((self.job_id, "start"))

        def receive(self):
            return "result", self.result.model_dump(mode="json")

        def stop(self, _):
            events.append((self.job_id, "stop"))

        def close(self):
            events.append((self.job_id, "close"))

    monkeypatch.setattr(module, "ManagedWorker", Worker)
    return events


@pytest.mark.parametrize("failure", ["serialize", "backup_replace", "primary_replace"])
def test_submission_write_failure_rejects_without_registration_or_launch(tmp_path, monkeypatch, failure):
    import studio.job_store as store_module

    manager = JobsManager(storage_path=tmp_path / "state.json", output_root=tmp_path)
    original = request(tmp_path, "history")
    original.status = JobStatus.COMPLETED
    original.metadata.completed_at = "2026-10-07T00:00:00+00:00"
    manager._jobs["history"] = original
    manager._logs.entries["job"] = ["previous orphan log"]
    manager._save()
    before = manager._store.path.read_bytes()
    started = []
    monkeypatch.setattr(manager, "_start_supervisors", lambda: started.append(True))
    with monkeypatch.context() as fault:
        if failure == "serialize":

            def denied(_):
                raise ValueError("injected serialization failure API_KEY=secret_key_12345678")

            fault.setattr(manager._store, "_serialize", denied)
        else:
            original_replace = store_module.os.replace
            target = manager._store.backup if failure == "backup_replace" else manager._store.path

            def denied(source, destination):
                if Path(destination) == target:
                    raise PermissionError("injected replace failure API_KEY=secret_key_12345678")
                return original_replace(source, destination)

            fault.setattr(store_module.os, "replace", denied)
        with pytest.raises(RuntimeError) as error:
            manager.submit_job_request(request(tmp_path))
    assert error.value.code == "PERSISTENCE_FAILED"
    assert "secret_key_12345678" not in str(error.value)
    assert "secret_key_12345678" not in "".join(traceback.format_exception(error.value))
    assert manager.persistence_error is not None
    assert "secret_key_12345678" not in manager.persistence_error
    assert manager.get_job("job") is None
    assert manager.get_job_log_lines("job") == ["previous orphan log"]
    assert manager._store.path.read_bytes() == before
    assert manager._workers == {} and manager._supervisors == [] and started == []
    assert all(pending.empty() for pending in manager._queues.values())
    try:
        accepted = manager.submit_job_request(request(tmp_path))
        assert accepted.status == JobStatus.PENDING
        assert manager.persistence_error is None
        assert "job" in json.loads(manager._store.path.read_text())["jobs"]
    finally:
        manager.shutdown()


@pytest.mark.parametrize("error_type", [RuntimeError, PermissionError, ValueError])
def test_normalization_failure_publishes_complete_failure_after_cleanup(
    tmp_path, monkeypatch, completed_worker, error_type
):
    manager = JobsManager(storage_path=tmp_path / "state.json", output_root=tmp_path, cpu_concurrency=1)
    original_normalize = manager._artifacts.normalize

    def fail_once(job, candidates):
        if job.job_id == "job":
            assert completed_worker == [("job", "start"), ("job", "stop"), ("job", "close")]
            raise error_type("injected normalization failure API_KEY=secret_key_12345678")
        return original_normalize(job, candidates)

    monkeypatch.setattr(manager._artifacts, "normalize", fail_once)
    try:
        manager.submit_job_request(request(tmp_path))
        result = terminal(manager, "job")
        assert result.status == JobStatus.FAILED
        assert result.error.code == "EXECUTION_FAILED"
        assert result.metadata.started_at is not None and result.metadata.completed_at is not None
        assert result.output.artifacts == []
        assert "job" not in manager._workers
        persisted = json.loads(manager._store.path.read_text())["jobs"]["job"]
        assert persisted["status"] == "failed" and persisted["metadata"]["completed_at"] is not None
        assert "secret_key_12345678" not in result.model_dump_json()
        assert "secret_key_12345678" not in json.dumps(manager.get_job_log_lines("job"))
        assert "secret_key_12345678" not in manager._store.path.read_text()
        manager.submit_job_request(request(tmp_path, "next"))
        recovered = terminal(manager, "next")
        assert recovered.status == JobStatus.COMPLETED
        assert recovered.output.artifacts == ["authorized.txt"]
        assert recovered.metadata.completed_at is not None
    finally:
        manager.shutdown()


def test_terminal_write_failure_preserves_execution_fact_and_cleanup(tmp_path, monkeypatch, completed_worker):
    import studio.job_store as store_module

    manager = JobsManager(storage_path=tmp_path / "state.json", output_root=tmp_path)
    original_normalize = manager._artifacts.normalize
    original_replace = store_module.os.replace

    def denied(source, destination):
        if Path(destination) == manager._store.path:
            raise PermissionError("terminal snapshot unavailable")
        return original_replace(source, destination)

    def fail_terminal_save(job, candidates):
        monkeypatch.setattr(store_module.os, "replace", denied)
        return original_normalize(job, candidates)

    monkeypatch.setattr(manager._artifacts, "normalize", fail_terminal_save)
    try:
        manager.submit_job_request(request(tmp_path))
        result = terminal(manager, "job")
        assert result.status == JobStatus.COMPLETED and result.metadata.completed_at is not None
        assert result.output.artifacts == ["authorized.txt"]
        assert completed_worker == [("job", "start"), ("job", "stop"), ("job", "close")]
        assert manager._workers == {} and manager.persistence_error == "terminal snapshot unavailable"
        assert json.loads(manager._store.path.read_text())["jobs"]["job"]["status"] == "running"
        monkeypatch.setattr(store_module.os, "replace", original_replace)
        manager._append_log("job", "retry persistence")
        assert manager.persistence_error is None
        assert json.loads(manager._store.path.read_text())["jobs"]["job"]["status"] == "completed"
        assert manager.get_job("job").model_dump() == result.model_dump()
    finally:
        monkeypatch.setattr(store_module.os, "replace", original_replace)
        manager.shutdown()


def test_cancel_write_failure_keeps_accepted_action_and_reports_durability(tmp_path, monkeypatch):
    import studio.job_store as store_module

    manager = JobsManager(storage_path=tmp_path / "state.json", output_root=tmp_path)
    monkeypatch.setattr(manager, "_start_supervisors", lambda: None)
    manager.submit_job_request(request(tmp_path))
    with monkeypatch.context() as fault:

        def denied(*_):
            raise PermissionError("cancel snapshot unavailable")

        fault.setattr(store_module.os, "replace", denied)
        decision = manager.request_cancel("job")
    assert decision.code == CancelCode.ACCEPTED and decision.job.status == JobStatus.CANCELLED
    assert decision.job.metadata.completed_at is not None
    assert manager._workers == {} and manager.persistence_error == "cancel snapshot unavailable"
    assert json.loads(manager._store.path.read_text())["jobs"]["job"]["status"] == "pending"
    manager._append_log("job", "retry persistence")
    assert manager.persistence_error is None
    assert json.loads(manager._store.path.read_text())["jobs"]["job"]["status"] == "cancelled"
    manager.shutdown()


def test_initial_snapshot_failure_leaves_no_submission_log_or_worker(tmp_path, monkeypatch):
    import studio.job_store as store_module
    from studio.jobs_manager import StatePersistenceError

    manager = JobsManager(storage_path=tmp_path / "new-state.json", output_root=tmp_path)
    with monkeypatch.context() as fault:

        def denied(*_):
            raise PermissionError("initial snapshot unavailable")

        fault.setattr(store_module.os, "replace", denied)
        with pytest.raises(StatePersistenceError):
            manager.submit_job_request(request(tmp_path))
    assert manager.get_job("job") is None and manager.get_job_log_lines("job") == []
    assert manager._workers == {} and manager._supervisors == []
    assert not manager._store.path.exists()
    assert all(pending.empty() for pending in manager._queues.values())
    assert manager.persistence_error == "initial snapshot unavailable"
    manager.shutdown()


def test_expected_artifact_path_error_rejects_candidate_without_failing_job(tmp_path, monkeypatch, completed_worker):
    manager = JobsManager(storage_path=tmp_path / "state.json", output_root=tmp_path)
    original_is_file = Path.is_file

    def denied(path, *args, **kwargs):
        if path.name == "authorized.txt":
            raise PermissionError("candidate cannot be read")
        return original_is_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", denied)
    try:
        manager.submit_job_request(request(tmp_path))
        result = terminal(manager, "job")
        assert result.status == JobStatus.COMPLETED and result.metadata.completed_at is not None
        assert result.error is None and result.output.artifacts == []
        assert completed_worker == [("job", "start"), ("job", "stop"), ("job", "close")]
        assert manager._workers == {} and manager.persistence_error is None
        assert json.loads(manager._store.path.read_text())["jobs"]["job"]["status"] == "completed"
    finally:
        manager.shutdown()
