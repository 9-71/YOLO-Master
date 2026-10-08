"""Server roots and exact manifest authorization, including symlink retargets."""

import pytest

from core.schema import JobRequest, JobStatus
from studio.jobs_manager import JobsManager


def owner(tmp_path):
    manager = JobsManager(output_root=tmp_path)
    job = JobRequest(job_id="job", task_type="predict", status="completed", output={"output_dir": str(tmp_path)})
    manager._jobs["job"] = job
    root = tmp_path / "job"
    root.mkdir()
    return manager, job, root


def link(path, target, directory=False):
    try:
        path.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege unavailable; Linux gate executes this case")
        raise


def test_manifest_is_exact_nested_and_never_inferred_by_directory_scan(tmp_path):
    manager, job, root = owner(tmp_path)
    (root / "nested").mkdir()
    (root / "nested" / "result.txt").write_text("allowed")
    (root / "not-listed.txt").write_text("not authorized")
    job.output.artifacts = ["nested/result.txt"]
    assert manager.resolve_artifact("job", "nested/result.txt") == root / "nested" / "result.txt"
    assert manager.resolve_artifact("job", "result.txt") is None
    assert manager.resolve_artifact("job", "not-listed.txt") is None
    assert [identifier for identifier, _ in manager.get_job_artifacts("job")] == ["nested/result.txt"]


@pytest.mark.parametrize(
    "identifier",
    [
        "../outside.txt",
        "/absolute.txt",
        "nested/../outside.txt",
        "C:/outside.txt",
        "nested\\result.txt",
        "./result.txt",
    ],
)
def test_even_persisted_manifest_cannot_authorize_unsafe_ids(tmp_path, identifier):
    manager, job, _ = owner(tmp_path)
    job.output.artifacts = [identifier]
    assert manager.resolve_artifact("job", identifier) is None
    assert manager.get_job_artifacts("job") == []


@pytest.mark.parametrize("kind", ["escape", "broken", "retarget"])
def test_manifest_symlinks_are_rechecked_on_every_read(tmp_path, kind):
    manager, job, root = owner(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    inside = root / "inside.txt"
    inside.write_text("allowed")
    candidate = root / "link.txt"
    link(candidate, inside if kind == "retarget" else outside if kind == "escape" else tmp_path / "missing.txt")
    job.output.artifacts = ["link.txt"]
    if kind == "retarget":
        assert manager.resolve_artifact("job", "link.txt") == inside
        candidate.unlink()
        candidate.symlink_to(outside)
    assert manager.resolve_artifact("job", "link.txt") is None


@pytest.mark.parametrize("kind", ["escape", "broken"])
def test_admission_rejects_symlink_escape_and_broken_link(tmp_path, kind):
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    candidate = trusted / "link"
    target = tmp_path / "outside"
    if kind == "escape":
        target.mkdir()
    link(candidate, target, directory=True)
    manager = JobsManager(output_root=trusted)
    request = JobRequest(job_id="job", task_type="diagnose", output={"output_dir": str(candidate / "tail")})
    with pytest.raises(ValueError):
        manager.submit_job_request(request)
    assert manager.get_job("job") is None


def test_normalization_does_not_commit_manifest_or_follow_untrusted_candidates(tmp_path):
    manager, job, root = owner(tmp_path)
    (root / "nested").mkdir()
    valid = root / "nested" / "result.txt"
    valid.write_text("ok")
    outside = tmp_path / "job-sibling"
    outside.mkdir()
    (outside / "private.txt").write_text("private")
    candidates = [str(valid), str(valid), str(outside / "private.txt"), "../job-sibling/private.txt"]
    assert manager._artifacts.normalize(job, candidates) == ["nested/result.txt"]
    assert job.output.artifacts == [] and job.status == JobStatus.COMPLETED


def test_network_input_userinfo_and_credential_shaped_ids_are_rejected(tmp_path, monkeypatch):
    manager = JobsManager(output_root=tmp_path, network_input_hosts=["example.com"])
    monkeypatch.setattr(manager, "_start_supervisors", lambda: None)
    for source in ("https://alice:hunter2@example.com/image.jpg", "https://alice@example.com/image.jpg"):
        job = JobRequest(
            job_id="network", task_type="predict", params={"data_source": source}, output={"output_dir": str(tmp_path)}
        )
        with pytest.raises(ValueError, match="credentials"):
            manager.submit_job_request(job)
    job = JobRequest(job_id="sk-1234567890abcdef12345678", task_type="diagnose", output={"output_dir": str(tmp_path)})
    with pytest.raises(ValueError, match="credential-shaped"):
        manager.submit_job_request(job)
    assert manager._jobs == {}


@pytest.mark.parametrize(
    "model",
    [
        "https://example.com/model.pt",
        "grpc://example.com/model",
        "https:/example.com/model.pt",
        "s3://bucket/model.pt",
        "file:///tmp/model.pt",
    ],
)
def test_model_protocols_cannot_bypass_local_model_roots(tmp_path, monkeypatch, model):
    monkeypatch.chdir(tmp_path)
    manager = JobsManager(output_root=tmp_path)
    job = JobRequest(
        job_id="remote-model", task_type="predict", params={"model_path": model}, output={"output_dir": str(tmp_path)}
    )
    with pytest.raises(ValueError, match="local path"):
        manager.submit_job_request(job)
    assert manager.get_job(job.job_id) is None


def test_local_inputs_are_canonicalized_before_worker_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = JobsManager(output_root=tmp_path, network_input_hosts=["example.com"])
    monkeypatch.setattr(manager, "_start_supervisors", lambda: None)
    job = JobRequest(
        job_id="local",
        task_type="predict",
        params={"model_path": "weights.pt", "data_source": ["image.jpg", "https://example.com/image.jpg"]},
        output={"output_dir": str(tmp_path)},
    )
    admitted = manager.submit_job_request(job)
    assert admitted.params["model_path"] == str(tmp_path / "weights.pt")
    assert admitted.params["data_source"] == [str(tmp_path / "image.jpg"), "https://example.com/image.jpg"]
    manager.shutdown()
