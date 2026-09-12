"""Real CPU-only process lifecycle tests: no torch, GPU, model or dataset downloads."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from fastapi.testclient import TestClient

from api.v1.jobs import get_jobs_manager
from core.schema import TERMINAL_STATUSES, ErrorInfo, JobRequest, JobStatus, TaskType
from f1.jobs_manager import JobsManager
from f1.worker_runtime import ManagedWorker, execute_job
from main_engine import create_app

# WSL imports dependencies from a Windows mount; leave time for both spawn
# bootstraps before exercising an in-flight timeout rather than a startup timeout.
TASK_TIMEOUT = 12 if sys.platform == "linux" else 3


def dummy_executor(job):
    """Spawn a real child and grandchild, then finish, crash or ignore cancellation."""
    root = Path(job.output.output_dir)
    if os.name != "nt":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if job.params.get("children", True):
        # Constant program + separate argv paths; no shell and no command interpolation.
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import os,signal,subprocess,sys,time; from pathlib import Path; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'], "
                    "start_new_session=(os.name!='nt')); "
                    "Path(sys.argv[1]).write_text(str(os.getpid())+','+str(p.pid)); time.sleep(120)"
                ),
                str(root / f"{job.job_id}.children"),
            ],
            start_new_session=os.name != "nt",  # Match torchrun's detached rank sessions.
        )
    (root / f"{job.job_id}.started").write_text(str(os.getpid()))
    for message in job.params.get("live_logs", []):
        job.append_log(message)
    if job.params.get("crash"):
        deadline = time.monotonic() + 5
        while not (root / f"{job.job_id}.children").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        os._exit(17)
    gate = root / f"{job.job_id}.release"
    while not gate.exists():
        time.sleep(0.02)
    job.status = JobStatus.COMPLETED
    if terminal_log := job.params.get("terminal_log"):
        job.append_log(terminal_log)
    return job


def mock_yolo_executor(job):
    """Exercise the real dispatcher and each real handler using an in-process fake YOLO."""

    class FakeYOLO:
        def __init__(self, *args):
            self.model = SimpleNamespace()

        def compute(self, **kwargs):
            dummy_executor(job)

        train = val = predict = export = compute

    sys.modules["ultralytics"] = SimpleNamespace(YOLO=FakeYOLO)
    from f1.handlers.train import TrainHandler

    TrainHandler._inject_determinism = lambda *args, **kwargs: 42
    return execute_job(job)


def wait_for(predicate, timeout=15):
    """Poll real process/file state with a bounded test deadline."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.03)
    raise AssertionError("Lifecycle condition did not become true before the test deadline")


def live(pid):
    """Zombies have stopped computing; POSIX init is responsible for their final reap."""
    try:
        return psutil.Process(pid).is_running() and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.fixture
def manager(tmp_path):
    instance = JobsManager(
        storage_path=str(tmp_path / "state.json"),
        output_root=tmp_path,
        model_roots=[tmp_path],
        data_roots=[tmp_path],
        cpu_concurrency=2,
        gpu_concurrency=1,
        stop_grace_seconds=0.2,
    )
    instance._worker_executor = dummy_executor
    yield instance
    instance.shutdown()


def submit(manager, root, job_id, device="cpu", timeout=30, **params):
    return manager.submit_job_request(
        JobRequest(
            job_id=job_id,
            task_type=TaskType.PREDICT,
            params={"device": device, **params},
            output={"output_dir": str(root)},
            runtime_tracking={"timeout_seconds": timeout},
        )
    )


def pids(root, job_id):
    wait_for(lambda: (root / f"{job_id}.started").exists() and (root / f"{job_id}.children").exists())
    return [
        int((root / f"{job_id}.started").read_text()),
        *map(int, (root / f"{job_id}.children").read_text().split(",")),
    ]


def terminal(manager, job_id):
    wait_for(lambda: manager.get_job(job_id).status in TERMINAL_STATUSES)
    assert job_id not in manager._workers
    return manager.get_job(job_id)


def test_normal_completion_cleans_leftover_children(manager, tmp_path):
    submit(manager, tmp_path, "normal")
    owned = pids(tmp_path, "normal")
    (tmp_path / "normal.release").touch()
    result = terminal(manager, "normal")
    assert result.status == JobStatus.COMPLETED
    assert result.metadata.started_at is not None
    assert result.metadata.completed_at is not None
    assert manager.get_job_status("normal")["duration"] >= 0
    assert not any(live(pid) for pid in owned)


def test_running_logs_stream_through_api_in_order_without_terminal_duplicates(manager, tmp_path):
    """Structured child logs are visible while RUNNING; terminal tail waits for cleanup."""
    submit(
        manager,
        tmp_path,
        "live-api",
        children=False,
        live_logs=["first live API_KEY=secret_key_12345678", "second live line"],
        terminal_log="natural terminal line",
    )
    wait_for(lambda: manager.get_job("live-api").status == JobStatus.RUNNING)
    wait_for(
        lambda: manager.get_job_log_lines("live-api")[-2:] == ["first live API_KEY=***REDACTED***", "second live line"]
    )

    app = create_app()
    app.dependency_overrides[get_jobs_manager] = lambda: manager
    with TestClient(app) as client:
        assert client.get("/api/v1/jobs/live-api").json()["status"] == "running"
        running = client.get("/api/v1/jobs/live-api/logs", params={"offset": 1, "limit": 500})
        assert running.status_code == 200
        assert running.json()["logs"] == ["first live API_KEY=***REDACTED***", "second live line"]
        assert "natural terminal line" not in running.json()["logs"]

        (tmp_path / "live-api.release").touch()
        result = terminal(manager, "live-api")
        assert result.status == JobStatus.COMPLETED
        final_logs = client.get("/api/v1/jobs/live-api/logs").json()["logs"]

    assert final_logs[1:] == ["first live API_KEY=***REDACTED***", "second live line", "natural terminal line"]
    assert final_logs.count("first live API_KEY=***REDACTED***") == 1
    assert final_logs.count("second live line") == 1
    assert final_logs.count("natural terminal line") == 1


def test_log_sequence_reordering_dedup_and_terminal_tail(monkeypatch, tmp_path):
    """Parent orders out-of-order events, drops duplicate seq values, and merges only the final tail."""
    import f1.jobs_manager as jobs_manager_module

    manager = JobsManager(output_root=tmp_path, model_roots=[tmp_path], data_roots=[tmp_path])
    job_id = "log-sequence"
    job = JobRequest(job_id=job_id, task_type=TaskType.PREDICT, output={"output_dir": str(tmp_path)})
    manager.jobs[job_id] = job
    manager.job_logs[job_id] = ["submitted"]

    result = job.model_copy(deep=True)
    result.logs = ["zero", "one", "terminal"]
    result.status = JobStatus.COMPLETED
    cleanup_started = threading.Event()
    allow_cleanup = threading.Event()
    events = iter(
        [
            ("log", {"seq": 1, "text": "one", "terminal": False}),
            ("log", {"seq": 0, "text": "zero", "terminal": False}),
            ("log", {"seq": 1, "text": "duplicate one", "terminal": False}),
            ("log", {"seq": 2, "text": "terminal", "terminal": True}),
            ("result", result.model_dump(mode="json")),
        ]
    )

    class FakeProcess:
        @staticmethod
        def is_alive():
            return True

    class FakeWorker:
        def __init__(self, *_args, **_kwargs):
            self.process = FakeProcess()

        def start(self):
            return None

        def receive(self):
            return next(events)

        def stop(self, _grace):
            cleanup_started.set()
            assert allow_cleanup.wait(timeout=5)

        def close(self):
            return None

    monkeypatch.setattr(jobs_manager_module, "ManagedWorker", FakeWorker)
    execution = threading.Thread(target=manager._execute_job, args=(job_id,), daemon=True)
    execution.start()
    assert cleanup_started.wait(timeout=5)
    assert manager.jobs[job_id].status == JobStatus.RUNNING
    assert manager.job_logs[job_id] == ["submitted", "zero", "one"]
    allow_cleanup.set()
    execution.join(timeout=5)

    assert not execution.is_alive()
    assert manager.jobs[job_id].status == JobStatus.COMPLETED
    assert manager.job_logs[job_id] == ["submitted", "zero", "one", "terminal"]


@pytest.mark.parametrize("reason", ["cancel", "timeout", "shutdown", "crash"])
def test_stop_confirms_entire_tree_before_persisting(manager, tmp_path, reason):
    live_line = f"{reason} live log"
    submit(
        manager,
        tmp_path,
        reason,
        timeout=TASK_TIMEOUT if reason == "timeout" else 30,
        crash=reason == "crash",
        live_logs=[live_line],
    )
    owned = pids(tmp_path, reason)
    wait_for(lambda: live_line in manager.get_job_log_lines(reason))
    if reason == "cancel":
        manager.cancel_job(reason)
    elif reason == "shutdown":
        manager.shutdown()
    result = terminal(manager, reason)
    expected = {
        "cancel": "USER_CANCELLED",
        "timeout": "TIMEOUT",
        "shutdown": "SERVICE_SHUTDOWN",
        "crash": "WORKER_LOST",
    }
    assert result.error.code == expected[reason]
    assert result.status == (JobStatus.CANCELLED if reason == "cancel" else JobStatus.FAILED)
    assert result.metadata.started_at is not None
    assert result.metadata.completed_at is not None
    assert manager.get_job_status(reason)["duration"] >= 0
    assert manager.get_job_log_lines(reason).count(live_line) == 1
    assert not any(live(pid) for pid in owned)
    saved = json.loads((tmp_path / "state.json").read_text())["jobs"][reason]
    assert saved["status"] == ("cancelled" if reason == "cancel" else "failed")
    assert saved["error"]["code"] == expected[reason]
    assert "duration" not in saved and "duration" not in saved["metadata"]


def test_cpu_gpu_limits_and_pending_cancellation(manager, tmp_path):
    for job_id, device in [
        ("cpu1", "cpu"),
        ("cpu2", "cpu"),
        ("gpu1", "0"),
        ("cpu3", "cpu"),
        ("gpu2", "0,1"),
        ("gpu3", ""),
    ]:
        submit(manager, tmp_path, job_id, device=device)
    owned = {job_id: pids(tmp_path, job_id) for job_id in ("cpu1", "cpu2", "gpu1")}
    assert len(manager._workers) == 3
    assert len(manager._supervisors) == 3
    for job_id in ("cpu3", "gpu2", "gpu3"):
        assert manager.get_job(job_id).status == JobStatus.PENDING
        assert not (tmp_path / f"{job_id}.started").exists()
    manager.cancel_job("gpu2")
    cancelled = terminal(manager, "gpu2")
    assert cancelled.error.code == "USER_CANCELLED"
    assert cancelled.metadata.started_at is None
    assert cancelled.metadata.completed_at is not None
    assert manager.get_job_status("gpu2")["duration"] is None
    for job_id in ("cpu1", "gpu1"):
        manager.cancel_job(job_id)
        terminal(manager, job_id)
        assert not any(live(pid) for pid in owned[job_id])
    pids(tmp_path, "cpu3")
    pids(tmp_path, "gpu3")
    assert len(manager._workers) == 3
    assert not (tmp_path / "gpu2.started").exists()


@pytest.mark.parametrize("device", ["cpu", "0"])
@pytest.mark.parametrize("outcome", ["complete", "cancel", "timeout", "crash"])
def test_capacity_released_only_after_tree_exit(manager, tmp_path, device, outcome):
    manager._limits = {"cpu": 1, "gpu": 1}
    submit(
        manager,
        tmp_path,
        "first",
        device=device,
        timeout=TASK_TIMEOUT if outcome == "timeout" else 30,
        crash=outcome == "crash",
    )
    submit(manager, tmp_path, "next", device=device)
    assert manager.get_job("next").status == JobStatus.PENDING
    owned = pids(tmp_path, "first")
    if outcome == "complete":
        (tmp_path / "first.release").touch()
    elif outcome == "cancel":
        manager.cancel_job("first")
    result = terminal(manager, "first")
    if outcome == "complete":
        assert result.status == JobStatus.COMPLETED
    else:
        assert result.error.code == {"cancel": "USER_CANCELLED", "timeout": "TIMEOUT", "crash": "WORKER_LOST"}[outcome]
    assert not any(live(pid) for pid in owned)
    pids(tmp_path, "next")
    assert len(manager._workers) == 1
    assert manager.get_job("next").status == JobStatus.RUNNING


@pytest.mark.parametrize(
    "device,expected", [("cpu", "cpu"), (0, "gpu"), (None, "gpu"), ("", "gpu"), ("mps", "gpu"), ([0, 1], "gpu")]
)
def test_resource_classification(device, expected):
    job = JobRequest(job_id="device", task_type=TaskType.TRAIN, params={"device": device})
    assert JobsManager._resource_class(job) == expected


def test_server_configuration(monkeypatch):
    monkeypatch.setenv("F1_CPU_CONCURRENCY", "3")
    monkeypatch.setenv("F1_GPU_CONCURRENCY", "2")
    instance = JobsManager()
    assert instance._limits == {"cpu": 3, "gpu": 2}
    monkeypatch.setenv("F1_CPU_CONCURRENCY", "0")
    with pytest.raises(ValueError, match="concurrency"):
        JobsManager()


def test_launch_failure_cleans_gated_process(manager, tmp_path, monkeypatch):
    launched = []

    def fail_after_spawn(worker):
        worker.process.start()
        launched.append(worker.process.pid)
        raise OSError("Containment setup failed")

    monkeypatch.setattr(ManagedWorker, "start", fail_after_spawn)
    submit(manager, tmp_path, "launch-failed")
    assert terminal(manager, "launch-failed").error.code == "EXECUTION_FAILED"
    assert launched and not any(live(pid) for pid in launched)
    assert not (tmp_path / "launch-failed.started").exists()


def test_cleanup_failure_retains_slot_and_nonterminal_state(manager, tmp_path, monkeypatch):
    manager._limits["cpu"] = 1
    submit(manager, tmp_path, "stopping")
    owned = pids(tmp_path, "stopping")
    original_stop = ManagedWorker.stop

    def deferred_stop(worker, grace):
        if not (tmp_path / "allow-cleanup").exists():
            raise OSError("Temporary termination failure")
        return original_stop(worker, grace)

    monkeypatch.setattr(ManagedWorker, "stop", deferred_stop)
    try:
        submit(manager, tmp_path, "waiting")
        manager.cancel_job("stopping")
        wait_for(lambda: manager.get_job("stopping").error is not None)
        assert manager.get_job("stopping").error.code == "WORKER_STOP_FAILED"
        assert manager.get_job("stopping").status == JobStatus.RUNNING
        assert "stopping" in manager._workers
        assert manager.get_job("waiting").status == JobStatus.PENDING
        assert all(live(pid) for pid in owned)
    finally:
        (tmp_path / "allow-cleanup").touch()
    assert terminal(manager, "stopping").error.code == "USER_CANCELLED"
    assert not any(live(pid) for pid in owned)
    pids(tmp_path, "waiting")


@pytest.mark.parametrize("exit_mode", ["kill", "normal"])
def test_parent_exit_cleans_tree_and_restart_is_honest(tmp_path, exit_mode):
    program = (
        "import sys,time; from pathlib import Path; "
        "sys.path.insert(0,sys.argv[1]); "
        "from test_worker_lifecycle import dummy_executor,submit; "
        "from f1.jobs_manager import JobsManager; "
        "root=Path(sys.argv[2]); "
        "manager=JobsManager(storage_path=str(root/'state.json'), output_root=root,stop_grace_seconds=0.1); "
        "manager._worker_executor=dummy_executor; submit(manager,root,'parent'); "
        "\nwhile not (root/'exit').exists(): time.sleep(0.05)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", program, str(Path(__file__).parent), str(tmp_path)])
    identities = []
    try:
        owned = pids(tmp_path, "parent")
        identities = [psutil.Process(pid) for pid in owned]
        if exit_mode == "kill":
            parent.kill()
        else:
            (tmp_path / "exit").touch()
        parent.wait(timeout=15)
        wait_for(lambda: not any(live(pid) for pid in owned))
        restored = JobsManager(storage_path=str(tmp_path / "state.json"))
        assert restored.get_job("parent").error.code == (
            "SERVICE_RESTARTED" if exit_mode == "kill" else "SERVICE_SHUTDOWN"
        )
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        for process in identities:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass


def test_api_cancel_ack_precedes_confirmed_tree_exit(manager, tmp_path):
    app = create_app()
    app.dependency_overrides[get_jobs_manager] = lambda: manager
    with TestClient(app) as client:
        payload = JobRequest(
            job_id="api-cancel",
            task_type=TaskType.PREDICT,
            params={"device": "cpu"},
            output={"output_dir": str(tmp_path)},
        ).model_dump(mode="json")
        assert client.post("/api/v1/jobs", json=payload).status_code == 201
        owned = pids(tmp_path, "api-cancel")
        response = client.post("/api/v1/jobs/api-cancel/cancel")
        assert response.status_code == 202 and response.json()["status"] == "cancel_requested"
        terminal(manager, "api-cancel")
        body = client.get("/api/v1/jobs/api-cancel").json()
        assert body["status"] == "cancelled" and body["error_code"] == "USER_CANCELLED"
        assert not any(live(pid) for pid in owned)


@pytest.mark.parametrize("natural_status", [JobStatus.COMPLETED, JobStatus.FAILED])
def test_natural_terminal_publication_wins_cancel_api_race(monkeypatch, tmp_path, natural_status):
    """A cancel request with a stale RUNNING read cannot overwrite a published terminal result."""
    import f1.jobs_manager as jobs_manager_module

    manager = JobsManager(output_root=tmp_path, model_roots=[tmp_path], data_roots=[tmp_path])
    job_id = f"natural-{natural_status.value}"
    job = JobRequest(job_id=job_id, task_type=TaskType.PREDICT, output={"output_dir": str(tmp_path)})
    manager.jobs[job_id] = job
    manager.job_logs[job_id] = []

    worker_started = threading.Event()
    release_result = threading.Event()
    terminal_published = threading.Event()
    cancel_observed_running = threading.Event()
    allow_cancel_recheck = threading.Event()

    result = job.model_copy(deep=True)
    result.status = natural_status
    result.logs = ["natural live log", "natural terminal log"]
    if natural_status == JobStatus.FAILED:
        result.error = ErrorInfo(code="NATURAL_FAILURE", message="worker failed naturally")

    class FakeProcess:
        @staticmethod
        def is_alive():
            return True

    class FakeWorker:
        def __init__(self, *_args, **_kwargs):
            self.process = FakeProcess()
            self.receive_calls = 0

        def start(self):
            worker_started.set()

        def receive(self):
            self.receive_calls += 1
            if self.receive_calls == 1:
                return "log", {"seq": 0, "text": "natural live log", "terminal": False}
            assert release_result.wait(timeout=5)
            return "result", result.model_dump(mode="json")

        def stop(self, _grace):
            return None

        def close(self):
            return None

    monkeypatch.setattr(jobs_manager_module, "ManagedWorker", FakeWorker)
    original_save = manager._save

    def observe_terminal_save():
        original_save()
        if manager.jobs[job_id].status in TERMINAL_STATUSES:
            terminal_published.set()

    monkeypatch.setattr(manager, "_save", observe_terminal_save)
    original_get_job = manager.get_job
    gate_lock = threading.Lock()
    gate_used = False

    def gate_cancel_after_running_read(requested_job_id):
        nonlocal gate_used
        current = original_get_job(requested_job_id)
        with gate_lock:
            should_gate = not gate_used and current is not None and current.status == JobStatus.RUNNING
            if should_gate:
                gate_used = True
        if should_gate:
            cancel_observed_running.set()
            assert allow_cancel_recheck.wait(timeout=5)
        return current

    monkeypatch.setattr(manager, "get_job", gate_cancel_after_running_read)
    execution = threading.Thread(target=manager._execute_job, args=(job_id,), daemon=True)
    execution.start()
    assert worker_started.wait(timeout=5)
    wait_for(lambda: manager.get_job_log_lines(job_id) == ["natural live log"])

    app = create_app()
    app.dependency_overrides[get_jobs_manager] = lambda: manager
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        cancel_response = pool.submit(client.post, f"/api/v1/jobs/{job_id}/cancel")
        assert cancel_observed_running.wait(timeout=5)
        release_result.set()
        assert terminal_published.wait(timeout=5)
        published_snapshot = manager.jobs[job_id].model_dump(mode="json")
        allow_cancel_recheck.set()
        response = cancel_response.result(timeout=5)

    execution.join(timeout=5)
    assert not execution.is_alive()
    assert response.status_code == 200
    assert response.json()["status"] == natural_status.value
    assert "Cancellation requested" not in response.json()["message"]
    assert manager.jobs[job_id].model_dump(mode="json") == published_snapshot
    assert manager.jobs[job_id].runtime_tracking.cancel_requested is False
    assert manager.jobs[job_id].metadata.started_at is not None
    assert manager.jobs[job_id].metadata.completed_at is not None
    assert manager.get_job_status(job_id)["duration"] >= 0
    assert manager.job_logs[job_id] == ["natural live log", "natural terminal log"]


def test_restart_reconciles_and_persists_structured_failure(tmp_path):
    jobs = {
        state: JobRequest(job_id=state, task_type=TaskType.PREDICT, status=JobStatus(state)).model_dump(mode="json")
        for state in ("pending", "running", "completed", "failed")
    }
    jobs["running"]["metadata"]["started_at"] = jobs["running"]["metadata"]["created_at"]
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"jobs": jobs, "job_logs": {}}))
    instance = JobsManager(storage_path=str(path))
    app = create_app()
    app.dependency_overrides[get_jobs_manager] = lambda: instance
    with TestClient(app) as client:
        for job_id in ("pending", "running"):
            body = client.get(f"/api/v1/jobs/{job_id}").json()
            assert body["status"] == "failed" and body["error_code"] == "SERVICE_RESTARTED"
            assert body["completed_at"] is not None
            if job_id == "pending":
                assert body["started_at"] is None and body["duration"] is None
            else:
                assert body["started_at"] is not None and body["duration"] >= 0
        assert client.get("/api/v1/jobs/completed").json()["status"] == "completed"
    saved = json.loads(path.read_text())["jobs"]
    assert saved["running"]["error"]["code"] == "SERVICE_RESTARTED"


def test_old_persistence_without_execution_timestamps_loads(tmp_path):
    job = JobRequest(job_id="legacy", task_type=TaskType.PREDICT, status=JobStatus.COMPLETED).model_dump(mode="json")
    job["metadata"].pop("started_at")
    job["metadata"].pop("completed_at")
    path = tmp_path / "legacy-state.json"
    path.write_text(json.dumps({"jobs": {"legacy": job}, "job_logs": {}}))

    restored = JobsManager(storage_path=str(path)).get_job("legacy")

    assert restored.status == JobStatus.COMPLETED
    assert restored.metadata.started_at is None
    assert restored.metadata.completed_at is None


@pytest.mark.parametrize("task", ["train", "val", "predict", "export"])
@pytest.mark.parametrize("reason", ["cancel", "timeout"])
def test_real_handlers_are_process_isolated(manager, tmp_path, task, reason):
    manager._worker_executor = mock_yolo_executor
    model, data = tmp_path / "mock.pt", tmp_path / ("data.yaml" if task in ("train", "val") else "image.jpg")
    model.touch()
    data.touch()
    job_id = f"{task}-{reason}"
    request = JobRequest(
        job_id=job_id,
        task_type=TaskType(task),
        params={"model_path": str(model), "data_source": str(data), "device": "cpu", "format": "onnx"},
        output={"output_dir": str(tmp_path)},
        runtime_tracking={"timeout_seconds": TASK_TIMEOUT if reason == "timeout" else 30},
    )
    manager.submit_job_request(request)
    owned = pids(tmp_path, job_id)
    if reason == "cancel":
        manager.cancel_job(job_id)
    result = terminal(manager, job_id)
    assert result.error.code == ("USER_CANCELLED" if reason == "cancel" else "TIMEOUT")
    assert not any(live(pid) for pid in owned)
