"""Test suite for Jobs Tab UI components and JobsManager.

This module validates the Jobs Tab functionality including job submission,
status monitoring, log streaming, artifact capture, and cancellation handling.

Run:
    pytest smoke/f1/ui/test_jobs_tab.py -v
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from smoke.f1.test_f1_smoke import ErrorInfo, JobStatus
from smoke.f1.ui.jobs_tab import JobsManager


@pytest.fixture
def jobs_manager():
    """Create JobsManager instance for testing."""
    return JobsManager()


@pytest.fixture
def sample_job_params():
    """Sample job parameters for predict task."""
    return {
        "task_type": "predict",
        "model_path": "yolov8n.pt",
        "data_source": "ultralytics/assets/bus.jpg",
        "output_dir": "runs/predict",
        "conf": 0.25,
        "device": "cpu",
        "allowed_paths": [".", "ultralytics/assets", "runs"],
    }


class TestJobsManager:
    """Test cases for JobsManager class."""

    def test_submit_job_success(self, jobs_manager, sample_job_params):
        """Test successful job submission."""
        job_id, message = jobs_manager.submit_job(**sample_job_params)

        assert job_id.startswith("predict_")
        assert "submitted successfully" in message
        assert job_id in jobs_manager.jobs
        assert job_id in jobs_manager.job_logs

    def test_submit_job_creates_unique_ids(self, jobs_manager, sample_job_params):
        """Test that each submission generates unique job ID."""
        job_id_1, _ = jobs_manager.submit_job(**sample_job_params)
        job_id_2, _ = jobs_manager.submit_job(**sample_job_params)

        assert job_id_1 != job_id_2

    def test_get_job_status_not_found(self, jobs_manager):
        """Test status retrieval for non-existent job."""
        status = jobs_manager.get_job_status("nonexistent-job-id")

        assert status["status"] == "NOT_FOUND"
        assert "not found" in status["message"].lower()

    def test_get_job_status_valid(self, jobs_manager, sample_job_params):
        """Test status retrieval for valid job."""
        job_id, _ = jobs_manager.submit_job(**sample_job_params)

        # Wait briefly for job to start processing
        time.sleep(0.5)

        status = jobs_manager.get_job_status(job_id)

        assert status["status"] in ["PENDING", "RUNNING", "COMPLETED", "FAILED"]
        assert "duration" in status
        assert "artifact_count" in status

    def test_get_job_logs(self, jobs_manager, sample_job_params):
        """Test log retrieval."""
        job_id, _ = jobs_manager.submit_job(**sample_job_params)

        logs = jobs_manager.get_job_logs(job_id)

        assert "submitted" in logs.lower()
        assert job_id in logs

    def test_get_job_logs_nonexistent(self, jobs_manager):
        """Test log retrieval for non-existent job."""
        logs = jobs_manager.get_job_logs("nonexistent-job-id")

        assert "no logs available" in logs.lower()

    def test_cancel_job_not_found(self, jobs_manager):
        """Test cancellation of non-existent job."""
        result = jobs_manager.cancel_job("nonexistent-job-id")

        assert "not found" in result.lower()

    def test_cancel_job_terminal_state(self, jobs_manager, sample_job_params):
        """Test cancellation of job in terminal state."""
        job_id, _ = jobs_manager.submit_job(**sample_job_params)

        # Wait for job to complete or fail
        max_wait = 10  # seconds
        elapsed = 0
        while elapsed < max_wait:
            status = jobs_manager.get_job_status(job_id)
            if status["status"] in ["COMPLETED", "FAILED"]:
                break
            time.sleep(0.5)
            elapsed += 0.5

        result = jobs_manager.cancel_job(job_id)

        assert "terminal state" in result.lower() or "not found" in result.lower()

    def test_cancel_job_active(self, jobs_manager):
        """Test cancellation request for active job.

        The dispatcher's execute() is mocked with a controllable thread gate so the
        job simulates a RUNNING state without real network downloads or YOLO inference.
        """
        started = threading.Event()
        release = threading.Event()

        def fake_execute(job):
            """Simulate a long-running executor that honors cancellation requests."""
            job.status = JobStatus.RUNNING
            started.set()
            release.wait(timeout=10)
            if job.runtime_tracking.cancel_requested:
                job.status = JobStatus.FAILED
                job.error = ErrorInfo(code="USER_CANCELLED", message="Job execution cancelled by user request")
            else:
                job.status = JobStatus.COMPLETED
            return job

        # Patch only this manager's dispatcher instance to avoid interfering with
        # background jobs submitted by other tests
        with patch.object(jobs_manager.dispatcher, "execute", fake_execute):
            job_id, _ = jobs_manager.submit_job(
                task_type="predict",
                model_path="nonexistent_model.pt",
                data_source="ultralytics/assets/bus.jpg",
                output_dir="runs/predict",
                conf=0.25,
                device="cpu",
                allowed_paths=[".", "runs"],
            )

            # Wait until the background executor reports the job RUNNING
            assert started.wait(timeout=5), "Background executor never reached RUNNING state"
            assert jobs_manager.get_job_status(job_id)["status"] == "RUNNING"

            # Request cancellation while the job is actively running
            result = jobs_manager.cancel_job(job_id)

            assert "cancellation requested" in result.lower()
            assert job_id in result

        # Release the mock executor so the worker thread finishes cleanly
        release.set()

        # Cancellation should drive the job to a terminal FAILED state without deadlock
        deadline = time.time() + 5
        while time.time() < deadline:
            status = jobs_manager.get_job_status(job_id)
            if status["status"] == "FAILED":
                break
            time.sleep(0.05)
        assert status["status"] == "FAILED", "Cancelled job never reached terminal state"
        assert status["error_code"] == "USER_CANCELLED"

    def test_list_recent_jobs(self, jobs_manager, sample_job_params):
        """Test recent jobs listing."""
        # Submit multiple jobs
        job_ids = []
        for _ in range(3):
            job_id, _ = jobs_manager.submit_job(**sample_job_params)
            job_ids.append(job_id)
            time.sleep(0.1)  # Ensure different timestamps

        recent = jobs_manager.list_recent_jobs(limit=5)

        assert len(recent) == 3
        assert all(j["job_id"] in job_ids for j in recent)
        assert all("task_type" in j for j in recent)
        assert all("status" in j for j in recent)

    def test_list_recent_jobs_respects_limit(self, jobs_manager, sample_job_params):
        """Test that recent jobs listing respects limit parameter."""
        # Submit 5 jobs
        for _ in range(5):
            jobs_manager.submit_job(**sample_job_params)
            time.sleep(0.05)

        recent = jobs_manager.list_recent_jobs(limit=3)

        assert len(recent) <= 3

    def test_get_job_artifacts_nonexistent(self, jobs_manager):
        """Test artifact retrieval for non-existent job."""
        artifacts = jobs_manager.get_job_artifacts("nonexistent-job-id")

        assert artifacts == []

    def test_security_constraints_enforced(self, jobs_manager):
        """Test that security constraints are enforced."""
        # Submit job with path traversal attempt
        job_id, _ = jobs_manager.submit_job(
            task_type="predict",
            model_path="yolov8n.pt",
            data_source="../../etc/passwd",  # Path traversal attempt
            output_dir="runs/predict",
            conf=0.25,
            device="cpu",
            allowed_paths=["runs"],  # Restricted whitelist
        )

        # Wait for job to process
        time.sleep(1.0)

        status = jobs_manager.get_job_status(job_id)

        # Job should fail with security error
        assert status["status"] == "FAILED"
        assert status["error_code"] in ["SEC_ERR_001", "PARAM_VALIDATION_FAILED"]

    def test_thread_safety_concurrent_submissions(self, jobs_manager, sample_job_params):
        """Test thread safety with concurrent job submissions."""
        import threading

        job_ids = []
        lock = threading.Lock()

        def submit_job():
            job_id, _ = jobs_manager.submit_job(**sample_job_params)
            with lock:
                job_ids.append(job_id)

        threads = [threading.Thread(target=submit_job) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All job IDs should be unique
        assert len(job_ids) == len(set(job_ids))
        assert all(job_id in jobs_manager.jobs for job_id in job_ids)


class TestJobsManagerIntegration:
    """Integration tests for end-to-end job execution."""

    @pytest.mark.slow
    def test_predict_job_e2e(self, jobs_manager):
        """Test end-to-end predict job execution with real inference."""
        job_id, _ = jobs_manager.submit_job(
            task_type="predict",
            model_path="yolov8n.pt",
            data_source="ultralytics/assets/bus.jpg",
            output_dir="runs/predict",
            conf=0.25,
            device="cpu",
            allowed_paths=[".", "ultralytics/assets", "runs"],
        )

        # Wait for job completion (max 15 seconds)
        max_wait = 15
        elapsed = 0
        while elapsed < max_wait:
            status = jobs_manager.get_job_status(job_id)
            if status["status"] in ["COMPLETED", "FAILED"]:
                break
            time.sleep(0.5)
            elapsed += 0.5

        final_status = jobs_manager.get_job_status(job_id)

        # Job should complete successfully
        assert final_status["status"] == "COMPLETED"
        assert final_status["artifact_count"] > 0

        # Check artifacts exist
        artifacts = jobs_manager.get_job_artifacts(job_id)
        assert len(artifacts) > 0

        for _filename, filepath in artifacts:
            assert Path(filepath).exists()

        # Check logs contain execution info
        logs = jobs_manager.get_job_logs(job_id)
        assert "completed" in logs.lower() or "✅" in logs

    @pytest.mark.slow
    def test_diagnose_job_e2e(self, jobs_manager):
        """Test end-to-end diagnose job execution."""
        job_id, _ = jobs_manager.submit_job(
            task_type="diagnose",
            model_path="",  # Not needed for diagnose
            data_source="",  # Not needed for diagnose
            output_dir="runs/diagnose",
            conf=0.25,
            device="cpu",
            allowed_paths=["runs"],
        )

        # Wait for job completion
        max_wait = 10
        elapsed = 0
        while elapsed < max_wait:
            status = jobs_manager.get_job_status(job_id)
            if status["status"] in ["COMPLETED", "FAILED"]:
                break
            time.sleep(0.5)
            elapsed += 0.5

        final_status = jobs_manager.get_job_status(job_id)

        # Diagnose should complete successfully
        assert final_status["status"] == "COMPLETED"

        # Should generate diagnostic artifacts
        artifacts = jobs_manager.get_job_artifacts(job_id)
        assert len(artifacts) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
