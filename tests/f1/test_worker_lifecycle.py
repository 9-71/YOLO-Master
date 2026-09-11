"""Real CPU-only process lifecycle tests: no torch, GPU, model or dataset downloads."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from fastapi.testclient import TestClient

from api.v1.jobs import get_jobs_manager
from core.schema import JobRequest, JobStatus, TaskType
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
    if job.params.get("crash"):
        deadline = time.monotonic() + 5
        while not (root / f"{job.job_id}.children").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        os._exit(17)
    gate = root / f"{job.job_id}.release"
    while not gate.exists():
        time.sleep(0.02)
    job.status = JobStatus.COMPLETED
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
    wait_for(lambda: manager.get_job(job_id).status in (JobStatus.COMPLETED, JobStatus.FAILED))
    assert job_id not in manager._workers
    return manager.get_job(job_id)


def test_normal_completion_cleans_leftover_children(manager, tmp_path):
    submit(manager, tmp_path, "normal")
    owned = pids(tmp_path, "normal")
    (tmp_path / "normal.release").touch()
    assert terminal(manager, "normal").status == JobStatus.COMPLETED
    assert not any(live(pid) for pid in owned)


@pytest.mark.parametrize("reason", ["cancel", "timeout", "shutdown", "crash"])
def test_stop_confirms_entire_tree_before_persisting(manager, tmp_path, reason):
    submit(manager, tmp_path, reason, timeout=TASK_TIMEOUT if reason == "timeout" else 30, crash=reason == "crash")
    owned = pids(tmp_path, reason)
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
    assert not any(live(pid) for pid in owned)
    saved = json.loads((tmp_path / "state.json").read_text())["jobs"][reason]
    assert saved["status"] == "failed" and saved["error"]["code"] == expected[reason]


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
    assert terminal(manager, "gpu2").error.code == "USER_CANCELLED"
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
        assert body["status"] == "failed" and body["error_code"] == "USER_CANCELLED"
        assert not any(live(pid) for pid in owned)


def test_restart_reconciles_and_persists_structured_failure(tmp_path):
    jobs = {
        state: JobRequest(job_id=state, task_type=TaskType.PREDICT, status=JobStatus(state)).model_dump(mode="json")
        for state in ("pending", "running", "completed", "failed")
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"jobs": jobs, "job_logs": {}}))
    instance = JobsManager(storage_path=str(path))
    app = create_app()
    app.dependency_overrides[get_jobs_manager] = lambda: instance
    with TestClient(app) as client:
        for job_id in ("pending", "running"):
            body = client.get(f"/api/v1/jobs/{job_id}").json()
            assert body["status"] == "failed" and body["error_code"] == "SERVICE_RESTARTED"
            assert body["duration"].endswith("s")
        assert client.get("/api/v1/jobs/completed").json()["status"] == "completed"
    saved = json.loads(path.read_text())["jobs"]
    assert saved["running"]["error"]["code"] == "SERVICE_RESTARTED"


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
