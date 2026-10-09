"""Fault injection for JSON backup, validation, sanitization and restart recovery."""

import json

import pytest

from core.schema import JobRequest, JobStatus
from studio.job_store import JobStore, StateRecoveryError
from studio.jobs_manager import JobsManager


def record(job_id="saved", status="completed"):
    return JobRequest(job_id=job_id, task_type="predict", status=status)


def snapshot(job=None):
    job = job or record()
    return {"version": 1, "jobs": {job.job_id: job.model_dump(mode="json")}, "job_logs": {}}


@pytest.mark.parametrize("primary", [None, "broken", '{"jobs": {"wrong": {}}}'])
def test_valid_backup_repairs_missing_or_invalid_primary(tmp_path, primary):
    path = tmp_path / "jobs_state.json"
    if primary is not None:
        path.write_text(primary)
    store = JobStore(path)
    store.backup.write_text(json.dumps(snapshot()))
    loaded = store.load()
    assert loaded["jobs"]["saved"]["status"] == "completed"
    assert json.loads(path.read_text()) == loaded


@pytest.mark.parametrize(
    "primary,backup",
    [("broken", None), (None, "broken"), ("broken", "broken"), ('{"jobs": []}', '{"jobs": {"bad": {}}}')],
)
def test_existing_invalid_state_fails_closed(tmp_path, primary, backup):
    store = JobStore(tmp_path / "jobs_state.json")
    for path, contents in ((store.path, primary), (store.backup, backup)):
        if contents is not None:
            path.write_text(contents)
    with pytest.raises(StateRecoveryError):
        JobsManager(storage_path=store.path)


def test_new_environment_and_valid_primary_precedence(tmp_path):
    store = JobStore(tmp_path / "jobs_state.json")
    assert store.load()["jobs"] == {}
    store.path.write_text(json.dumps(snapshot(record("primary"))))
    store.backup.write_text(json.dumps(snapshot(record("backup"))))
    assert set(store.load()["jobs"]) == {"primary"}


def test_rotation_never_overwrites_good_backup_with_bad_primary(tmp_path):
    store = JobStore(tmp_path / "jobs_state.json")
    store.save({"first": record("first")}, {})
    store.save({"second": record("second")}, {})
    assert set(json.loads(store.backup.read_text())["jobs"]) == {"first"}
    store.path.write_text('{"jobs": {"invalid": {}}}')
    store.save({"third": record("third")}, {})
    assert set(json.loads(store.backup.read_text())["jobs"]) == {"first"}
    assert set(store.load()["jobs"]) == {"third"}


@pytest.mark.parametrize("failure", ["serialize", "backup_replace", "primary_replace"])
def test_write_failure_preserves_recoverable_history(tmp_path, monkeypatch, failure):
    store = JobStore(tmp_path / "jobs_state.json")
    store.save({"first": record("first")}, {})
    original = store.path.read_bytes()
    if failure == "serialize":
        monkeypatch.setattr(store, "_serialize", lambda _: (_ for _ in ()).throw(ValueError("serialize failed")))
    else:
        import studio.job_store as module

        replace = module.os.replace

        def fail_replace(source, target):
            if target == (store.backup if failure == "backup_replace" else store.path):
                raise OSError("replace failed")
            return replace(source, target)

        monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises((ValueError, OSError)):
        store.save({"second": record("second")}, {})
    assert store.path.read_bytes() == original
    assert set(store.load()["jobs"]) == {"first"}


@pytest.mark.parametrize("damage", ["identity", "logs", "version", "record"])
def test_schema_invalid_primary_uses_backup(tmp_path, damage):
    payload = snapshot()
    if damage == "identity":
        payload["jobs"]["saved"]["job_id"] = "different"
    elif damage == "logs":
        payload["job_logs"] = {"saved": [42]}
    elif damage == "version":
        payload["version"] = 999
    else:
        payload["jobs"]["saved"]["status"] = "unknown"
    store = JobStore(tmp_path / "jobs_state.json")
    store.path.write_text(json.dumps(payload))
    store.backup.write_text(json.dumps(snapshot()))
    assert store.load()["jobs"]["saved"]["status"] == "completed"


def test_backup_recovery_reconciles_active_jobs_and_sanitizes_old_records(tmp_path):
    store = JobStore(tmp_path / "jobs_state.json")
    job = record("running", "running")
    job.params["api_key"] = "secret_key_12345678"
    payload = snapshot(job)
    payload["jobs"]["running"]["metadata"].pop("started_at")
    payload["jobs"]["running"]["metadata"].pop("completed_at")
    payload["job_logs"]["running"] = ["API_KEY=secret_key_12345678"]
    store.backup.write_text(json.dumps(payload))
    manager = JobsManager(storage_path=store.path)
    restored = manager.get_job("running")
    assert restored.status == JobStatus.FAILED
    assert restored.error.code == "SERVICE_RESTARTED"
    assert restored.metadata.started_at is None
    assert restored.metadata.completed_at is not None
    assert manager.get_job_status("running")["duration"] is None
    assert manager._workers == {} and manager._supervisors == []
    assert "secret_key_12345678" not in store.path.read_text()
    assert "secret_key_12345678" not in store.backup.read_text()
    assert json.loads(store.path.read_text())["jobs"]["running"]["error"]["code"] == "SERVICE_RESTARTED"


def test_reconciliation_write_failure_blocks_startup(tmp_path, monkeypatch):
    path = tmp_path / "jobs_state.json"
    path.write_text(json.dumps(snapshot(record("active", "pending"))))
    monkeypatch.setattr(JobStore, "save", lambda *_: (_ for _ in ()).throw(OSError("disk unavailable")))
    with pytest.raises(OSError, match="disk unavailable"):
        JobsManager(storage_path=path)


@pytest.mark.parametrize("entry", ["primary", "backup"])
def test_broken_state_symlink_is_existing_invalid_history(tmp_path, entry):
    from test_runtime_security import link

    store = JobStore(tmp_path / "state.json")
    link(store.path if entry == "primary" else store.backup, tmp_path / "missing.json")
    with pytest.raises(StateRecoveryError):
        store.load()


def test_state_existence_permission_error_fails_closed(tmp_path, monkeypatch):
    from pathlib import Path

    store = JobStore(tmp_path / "state.json")
    original = Path.lstat

    def denied(path, *args, **kwargs):
        if path == store.path:
            raise PermissionError("state entry inaccessible")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(StateRecoveryError):
        store.load()


@pytest.mark.parametrize("job_id", ["keyframe", "token-job", "cookie", "authorization", "password-check"])
def test_structural_ids_survive_sanitization_rotation_and_restart(tmp_path, job_id):
    store = JobStore(tmp_path / "state.json")
    job = record(job_id, "running")
    store.save({job_id: job}, {job_id: ["hello"]})
    store.save({job_id: job}, {job_id: ["hello", "next"]})
    assert store.load()["jobs"][job_id]["job_id"] == job_id
    manager = JobsManager(storage_path=store.path)
    assert manager.get_job(job_id).error.code == "SERVICE_RESTARTED"
    assert manager.get_job_log_lines(job_id)[:2] == ["hello", "next"]
    assert json.loads(store.backup.read_text())["jobs"][job_id]["job_id"] == job_id


def test_restored_url_credentials_are_redacted_in_records_and_logs(tmp_path):
    store = JobStore(tmp_path / "state.json")
    job = record()
    job.params["data_source"] = "https://alice:hunter2@example.com/image.jpg"
    store.path.write_text(
        json.dumps(
            {
                "jobs": {job.job_id: job.model_dump(mode="json")},
                "job_logs": {job.job_id: ["connect https://alice:hunter2@example.com/image.jpg"]},
            }
        )
    )
    manager = JobsManager(storage_path=store.path)
    assert "hunter2" not in manager.get_job(job.job_id).model_dump_json()
    assert "hunter2" not in json.dumps(manager.get_job_log_lines(job.job_id))
    assert "hunter2" not in store.path.read_text() and "hunter2" not in store.backup.read_text()
