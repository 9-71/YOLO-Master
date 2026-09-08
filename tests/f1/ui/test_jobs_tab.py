"""Test suite for Jobs Tab UI components and JobsManager.

This module validates the Jobs Tab functionality including job submission,
status monitoring, log streaming, artifact capture, and cancellation handling.

Run:
    pytest f1/ui/test_jobs_tab.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import gradio as gr
import pytest

from core.schema import ErrorInfo, JobRequest, JobStatus, OutputConfig, TaskType
from f1.ui.i18n import get_text
from f1.ui.jobs_tab import (
    _DATAFRAME_HEADER_MENU_CSS,
    POLL_SLOW_SECONDS,
    JobsManager,
    compute_poll_state,
    create_jobs_tab,
    format_created_at,
    get_job_image_artifacts,
    recent_jobs_rows,
)


@pytest.fixture
def jobs_manager():
    """Create JobsManager instance for testing."""
    return JobsManager()


def _wired_fn(tab: gr.Blocks, component: gr.Component, event: str) -> Callable[..., Any]:
    """Resolve the live callable Gradio registered for ``(component, event)``."""
    return next(bf.fn for bf in tab.fns.values() if bf.fn and (component._id, event) in bf.targets)


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


def _stub_execute(job):
    """Dispatcher stub marking a job COMPLETED without any engine work."""
    job.status = JobStatus.COMPLETED
    return job


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

    def test_submit_train_job_injects_required_defaults(self, jobs_manager):
        """Train submissions carry epochs=1 and imgsz=640 so TrainHandler.validate_params passes."""
        with patch.object(jobs_manager.dispatcher, "execute", _stub_execute):
            job_id, _ = jobs_manager.submit_job(
                task_type="train",
                model_path="yolov8n.pt",
                data_source="coco8.yaml",
                output_dir="runs/train",
                conf=0.25,
                device="cpu",
                allowed_paths=[".", "runs"],
            )

        params = jobs_manager.jobs[job_id].params
        assert params["epochs"] == 1
        assert params["imgsz"] == 640

    def test_submit_val_job_injects_imgsz_default(self, jobs_manager):
        """Val submissions carry imgsz=640 as a default engine parameter."""
        with patch.object(jobs_manager.dispatcher, "execute", _stub_execute):
            job_id, _ = jobs_manager.submit_job(
                task_type="val",
                model_path="yolov8n.pt",
                data_source="coco8.yaml",
                output_dir="runs/val",
                conf=0.25,
                device="cpu",
                allowed_paths=[".", "runs"],
            )

        params = jobs_manager.jobs[job_id].params
        assert params["imgsz"] == 640

    def test_train_job_with_image_data_source_fails_validation_fast(self, jobs_manager):
        """Train with an image data_source fails fast with PARAM_VALIDATION_FAILED.

        Regression for the E2E hang: the YOLO train engine requires a dataset YAML.
        Passing bus.jpg must be rejected during validation (before RUNNING) instead
        of raising "Not a YAML file" mid-execution or hanging in dataset resolution.
        """
        job_id, _ = jobs_manager.submit_job(
            task_type="train",
            model_path="yolov8n.pt",
            data_source="ultralytics/assets/bus.jpg",
            output_dir="runs/train",
            conf=0.25,
            device="cpu",
            allowed_paths=[".", "runs", "ultralytics/assets"],
        )

        deadline = time.time() + 10
        status = jobs_manager.get_job_status(job_id)
        while time.time() < deadline and status["status"] not in ("COMPLETED", "FAILED"):
            time.sleep(0.1)
            status = jobs_manager.get_job_status(job_id)

        assert status["status"] == "FAILED"
        assert status["error_code"] == "PARAM_VALIDATION_FAILED"
        assert "dataset YAML file" in status["error_message"]

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

    def test_list_recent_jobs_newest_first_with_per_job_timestamps(self, jobs_manager, sample_job_params):
        """Recent jobs are sorted newest-first and each row carries its own raw created_at.

        Backend rows keep raw ISO UTC strings; local-time formatting happens only at
        the UI presentation layer (compute_poll_state).
        """
        job_ids = []
        for _ in range(3):
            job_id, _ = jobs_manager.submit_job(**sample_job_params)
            job_ids.append(job_id)
            time.sleep(0.1)  # Ensure distinct creation timestamps

        recent = jobs_manager.list_recent_jobs(limit=5)

        # Most recently submitted job appears first
        assert [j["job_id"] for j in recent] == list(reversed(job_ids))
        # Every row uses that job's actual creation timestamp (not a shared constant)
        assert len({j["created_at"] for j in recent}) == len(job_ids)
        for row in recent:
            assert row["created_at"] == jobs_manager.jobs[row["job_id"]].metadata.created_at

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


class TestTimestampFormatting:
    """UI-layer local-time formatting for Recent Jobs timestamps."""

    def test_format_created_at_converts_utc_to_local_time(self):
        raw = "2026-09-02T10:14:44.136175+00:00"

        expected = datetime.fromisoformat(raw).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        assert format_created_at(raw) == expected
        assert "T" not in format_created_at(raw)

    def test_format_created_at_fallbacks(self):
        assert format_created_at("") == "-"
        assert format_created_at(None) == "-"
        assert format_created_at("not-a-timestamp") == "not-a-timestamp"


class TestArtifactVisualizer:
    """Test cases for Artifact Visualizer — image artifact scanning and preview.

    Validates that ``get_job_image_artifacts`` (the module-level scanner) and
    ``JobsManager.get_job_image_artifacts`` (the thread-safe wrapper) correctly:
    hide artifacts until a job reaches COMPLETED, return only image files sorted
    by filename, defend against path traversal, and degrade gracefully.
    """

    @staticmethod
    def _write_minimal_png(path: Path) -> None:
        """Write a minimal valid 1x1 RGB PNG file (real image, not just a stub).

        Args:
            path: Destination file path.
        """
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
            b"\x00\x00\x00\x03\x00\x01\xba\x1b\xe3\x82\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        path.write_bytes(png_bytes)

    @staticmethod
    def _write_minimal_jpg(path: Path) -> None:
        """Write a minimal valid 1x1 JPEG file (real image, not just a stub).

        Args:
            path: Destination file path.
        """
        jpg_bytes = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n"
            b"\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a"
            b"\x1c\x1c $. \x1c\x1c(7),01444\x1f'9=82<.342"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04"
            b'\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142'
            b"\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a"
            b"%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88"
            b"\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7"
            b"\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6"
            b"\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4"
            b"\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa"
            b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xd2\x21\xff\xd9"
        )
        path.write_bytes(jpg_bytes)

    @staticmethod
    def _write_minimal_webp(path: Path) -> None:
        """Write a minimal valid WebP file (RIFF + VP8 lossy, 1x1 frame).

        Args:
            path: Destination file path.
        """
        vp8_data = b"\x10\x00\x00\x9d\x01\x2a\x01\x00\x01\x00\x02\x00"
        vp8_size = len(vp8_data)
        riff_size = 4 + 8 + vp8_size  # "WEBP" + "VP8 " + size_field + vp8_data
        webp_bytes = (
            b"RIFF" + riff_size.to_bytes(4, "little") + b"WEBP" + b"VP8 " + vp8_size.to_bytes(4, "little") + vp8_data
        )
        path.write_bytes(webp_bytes)

    def test_get_image_artifacts_status_not_completed(self, tmp_path: Path) -> None:
        """Non-COMPLETED jobs must never expose unready artifacts.

        Even when real PNG files already exist under ``output_dir`` (e.g. from a
        previous partial run or an external writer), a PENDING or RUNNING job
        must return an empty list so the UI never shows half-baked previews.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        self._write_minimal_png(output_dir / "result.png")

        job = JobRequest(
            job_id="test-pending-artifacts",
            task_type=TaskType.PREDICT,
            status=JobStatus.PENDING,
            output=OutputConfig(output_dir=str(output_dir)),
        )

        assert get_job_image_artifacts(job) == []

    def test_get_image_artifacts_success(self, tmp_path: Path) -> None:
        """COMPLETED job returns only image files, sorted by filename ascending.

        Three images (c.jpg, a.png, b.webp) and one non-image file (readme.txt)
        are placed in the output directory.  The scanner must:
        - include exactly the three image files
        - exclude the non-image file
        - return filenames in strict ascending order: a.png, b.webp, c.jpg
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Write files in intentionally scrambled order
        self._write_minimal_jpg(output_dir / "c.jpg")
        self._write_minimal_png(output_dir / "a.png")
        self._write_minimal_webp(output_dir / "b.webp")
        (output_dir / "readme.txt").write_text("not an image")

        job = JobRequest(
            job_id="test-completed-artifacts",
            task_type=TaskType.PREDICT,
            status=JobStatus.COMPLETED,
            output=OutputConfig(output_dir=str(output_dir)),
        )

        result = get_job_image_artifacts(job)

        assert len(result) == 3
        assert Path(result[0]).name == "a.png"
        assert Path(result[1]).name == "b.webp"
        assert Path(result[2]).name == "c.jpg"

    def test_get_image_artifacts_path_traversal_protection(self, tmp_path: Path) -> None:
        """Images placed outside ``output_dir`` must never be scanned or returned.

        The scanner resolves every candidate path and verifies it lives under
        the resolved output directory root.  A sibling image in the parent
        directory must be silently dropped.
        """
        output_dir = tmp_path / "job_output"
        output_dir.mkdir()
        self._write_minimal_png(output_dir / "inside.png")

        # Place a decoy image in the parent directory (outside output_dir)
        self._write_minimal_png(tmp_path / "outside.png")

        job = JobRequest(
            job_id="test-traversal-protection",
            task_type=TaskType.PREDICT,
            status=JobStatus.COMPLETED,
            output=OutputConfig(output_dir=str(output_dir)),
        )

        result = get_job_image_artifacts(job)

        assert len(result) == 1
        assert Path(result[0]).name == "inside.png"
        assert all("outside" not in p for p in result)

    def test_jobs_manager_wrapper(self, jobs_manager: JobsManager, tmp_path: Path) -> None:
        """JobsManager.get_job_image_artifacts wraps the scanner safely.

        Verifies three behaviours:
        1. A non-existent job_id returns an empty list (no KeyError raised).
        2. A completed job with image artifacts returns the expected files.
        3. Edge cases (e.g. missing output directory on a COMPLETED job) are
           handled gracefully by returning ``[]`` instead of raising.
        """
        # 1. Non-existent job → empty list
        assert jobs_manager.get_job_image_artifacts("nonexistent-job-id") == []

        # 2. Completed job with images → returns artifact paths
        output_dir = tmp_path / "artifacts"
        output_dir.mkdir()
        self._write_minimal_png(output_dir / "preview.png")

        def fake_execute(job: JobRequest) -> JobRequest:
            job.status = JobStatus.COMPLETED
            job.output.output_dir = str(output_dir)
            return job

        with patch.object(jobs_manager.dispatcher, "execute", fake_execute):
            job_id, _ = jobs_manager.submit_job(
                task_type="predict",
                model_path="nonexistent.pt",
                data_source="ultralytics/assets/bus.jpg",
                output_dir=str(output_dir),
                conf=0.25,
                device="cpu",
                allowed_paths=[".", "ultralytics/assets", str(tmp_path)],
            )
            # Give the background thread a moment to mark the job COMPLETED
            time.sleep(0.5)

        images = jobs_manager.get_job_image_artifacts(job_id)
        assert isinstance(images, list)
        assert len(images) == 1
        assert Path(images[0]).name == "preview.png"

        # 3. Edge case: COMPLETED job whose output_dir vanished → safe fallback
        vanished_dir = tmp_path / "vanished"
        vanished_dir.mkdir()
        self._write_minimal_png(vanished_dir / "orphan.png")

        def fake_vanished(job: JobRequest) -> JobRequest:
            job.status = JobStatus.COMPLETED
            job.output.output_dir = str(vanished_dir)
            return job

        with patch.object(jobs_manager.dispatcher, "execute", fake_vanished):
            job_id2, _ = jobs_manager.submit_job(
                task_type="predict",
                model_path="nonexistent.pt",
                data_source="ultralytics/assets/bus.jpg",
                output_dir=str(vanished_dir),
                conf=0.25,
                device="cpu",
                allowed_paths=[".", "ultralytics/assets", str(tmp_path)],
            )
            time.sleep(0.5)

        # Remove the directory after the job is marked COMPLETED
        shutil.rmtree(vanished_dir)

        fallback = jobs_manager.get_job_image_artifacts(job_id2)
        assert fallback == []


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


class TestArtifactTableSelectGuard:
    """Regression guards for the artifacts-table select callback.

    Invokes the wired handler exactly as Gradio does (``BlockFunction.fn``) so
    both the image-preview path and the non-image guard run against real code.
    """

    ARTIFACTS_LABEL = "Generated Artifacts"  # get_text("en", "df.artifacts")

    def _build(self) -> tuple[JobsManager, gr.Blocks, gr.Dataframe, Callable[..., Any]]:
        manager = JobsManager()
        tab = create_jobs_tab(manager, "en")
        dataframe = next(
            b for b in tab.blocks.values() if isinstance(b, gr.Dataframe) and b.label == self.ARTIFACTS_LABEL
        )
        return manager, tab, dataframe, _wired_fn(tab, dataframe, "select")

    def _insert_completed_job(self, manager: JobsManager, job_id: str, artifacts: list[str]) -> None:
        job = JobRequest(
            job_id=job_id,
            task_type=TaskType.PREDICT,
            status=JobStatus.COMPLETED,
            output=OutputConfig(output_dir="runs/predict", artifacts=artifacts),
        )
        manager.jobs[job_id] = job

    def test_image_row_updates_preview_and_selector(self, tmp_path: Path) -> None:
        """Selecting an image row syncs the selector and loads the large preview."""
        manager, _tab, _df, select_fn = self._build()
        image = tmp_path / "a.jpg"
        image.write_bytes(b"jpg")
        manifest = tmp_path / "args.yaml"
        manifest.write_text("conf: 0.25\n", encoding="utf-8")
        self._insert_completed_job(manager, "sel-img-001", [str(manifest), str(image)])

        result = select_fn(SimpleNamespace(index=(1,)), "sel-img-001", "en")

        assert result[0].get("value") == "a.jpg"  # selector synced to the filename
        assert result[1].get("value") == str(image)
        assert "a.jpg" in result[1]["label"]

    def test_non_image_row_keeps_preview_and_warns_bilingually(self, tmp_path: Path, monkeypatch: Any) -> None:
        """A .pt/.yaml row triggers a localized gr.Info and never touches the preview."""
        manager, _tab, _df, select_fn = self._build()
        image = tmp_path / "a.jpg"
        image.write_bytes(b"jpg")
        weights = tmp_path / "best.pt"
        weights.write_bytes(b"pt")
        self._insert_completed_job(manager, "sel-pt-001", [str(image), str(weights)])

        infos: list[str] = []
        monkeypatch.setattr("gradio.Info", lambda message: infos.append(message))
        result = select_fn(SimpleNamespace(index=(1,)), "sel-pt-001", "zh")

        assert len(infos) == 1
        assert "best.pt" in infos[0]
        assert "不是图片文件" in infos[0]
        # Preview state must be preserved: no value overwrite, no reset to None
        assert result[0].get("value") is None
        assert result[1].get("value") is None

        infos.clear()
        result = select_fn(SimpleNamespace(index=(1,)), "sel-pt-001", "en")
        assert len(infos) == 1
        assert "is not an image file" in infos[0]
        assert result[0].get("value") is None

    def test_out_of_range_row_is_ignored_silently(self, tmp_path: Path) -> None:
        """Rows beyond the artifact list (or empty selection) yield no-op updates."""
        manager, _tab, _df, select_fn = self._build()
        image = tmp_path / "a.jpg"
        image.write_bytes(b"jpg")
        self._insert_completed_job(manager, "sel-oob-001", [str(image)])

        for evt in (SimpleNamespace(index=(99,)), SimpleNamespace(index=())):
            result = select_fn(evt, "sel-oob-001", "en")
            assert result[0].get("value") is None
            assert result[1].get("value") is None


class TestOpenOutputFolderNavigation:
    """The open-folder button must reveal exactly the job-specific root folder."""

    def _build(self) -> tuple[JobsManager, Callable[..., Any]]:
        manager = JobsManager()
        tab = create_jobs_tab(manager, "en")
        button = next(b for b in tab.blocks.values() if isinstance(b, gr.Button) and b.value == "📂 Open Folder")
        return manager, _wired_fn(tab, button, "click")

    def _insert_job(self, manager: JobsManager, job_id: str, output_dir: Path, artifacts: list[str]) -> None:
        job = JobRequest(
            job_id=job_id,
            task_type=TaskType.PREDICT,
            status=JobStatus.COMPLETED,
            output=OutputConfig(output_dir=str(output_dir), artifacts=artifacts),
        )
        manager.jobs[job_id] = job

    @pytest.mark.skipif(os.name != "nt", reason="os.startfile is Windows-only")
    def test_deep_artifact_opens_job_root(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Artifacts nested in ``<job_root>/weights/`` open the job root, not the subdir."""
        manager, open_fn = self._build()
        job_root = tmp_path / "runs" / "train" / "predict_open001"
        (job_root / "weights").mkdir(parents=True)
        best = job_root / "weights" / "best.pt"
        best.write_bytes(b"pt")
        self._insert_job(manager, "predict_open001", tmp_path / "runs" / "train", [str(best.resolve())])

        opened: list[str] = []

        def _fake_startfile(path: str) -> None:
            opened.append(path)

        monkeypatch.setattr(os, "startfile", _fake_startfile, raising=False)
        open_fn("predict_open001", "en")

        # Exactly the directory whose name contains the job_id — never deeper,
        # never the parent runs/train.
        assert opened == [str(job_root)]

    def test_posix_branch_uses_xdg_open_on_job_root(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Non-Windows platforms shell out with the job root as the sole argument."""
        manager, open_fn = self._build()
        job_root = tmp_path / "runs" / "val" / "val_open002"
        job_root.mkdir(parents=True)
        report = job_root / "results.csv"
        report.write_text("epoch\n", encoding="utf-8")
        self._insert_job(manager, "val_open002", tmp_path / "runs" / "val", [str(report.resolve())])

        commands: list[list[str]] = []

        class _FakePopen:
            def __init__(self, cmd: list[str], *args: Any, **kwargs: Any) -> None:
                commands.append(cmd)

        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(subprocess, "Popen", _FakePopen)
        open_fn("val_open002", "en")

        assert commands == [["xdg-open", str(job_root)]]

    def test_missing_directory_warns_without_opening(self, tmp_path: Path, monkeypatch: Any) -> None:
        """A dangling manifest or unknown job id degrades to a localized warning."""
        manager, open_fn = self._build()
        self._insert_job(
            manager,
            "predict_gone001",
            tmp_path / "runs" / "predict",
            [str(tmp_path / "runs" / "predict" / "predict_gone001" / "ghost.jpg")],
        )

        warnings: list[str] = []
        opened: list[str] = []
        monkeypatch.setattr("gradio.Warning", lambda message: warnings.append(message))
        monkeypatch.setattr(os, "startfile", lambda path: opened.append(path), raising=False)

        open_fn("predict_gone001", "en")
        open_fn("no-such-job", "zh")

        assert len(warnings) == 2
        assert "Output directory does not exist." in warnings
        assert "输出目录尚不存在。" in warnings
        assert opened == []  # never touches the OS on a missing folder


class TestJobsPersistence:
    """Persistence layer tests for JobsManager (storage_path opt-in)."""

    @staticmethod
    def _wait_terminal(manager: JobsManager, job_id: str, timeout: float = 10.0) -> None:
        """Block until the background thread reaches a terminal state."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if manager.jobs[job_id].status in (JobStatus.COMPLETED, JobStatus.FAILED):
                return
            time.sleep(0.05)
        raise AssertionError(f"Job {job_id} never reached a terminal state")

    def test_persist_writes_json_roundtrip(self, tmp_path: Path, sample_job_params) -> None:
        """Submitting a job with storage_path writes a well-formed JSON state file."""
        storage = tmp_path / "state" / "jobs_state.json"
        manager = JobsManager(storage_path=str(storage))

        with patch.object(manager.dispatcher, "execute", _stub_execute):
            job_id, _ = manager.submit_job(**sample_job_params)
        self._wait_terminal(manager, job_id)

        assert storage.is_file(), "state file was not created"
        payload = json.loads(storage.read_text(encoding="utf-8"))
        assert payload["version"] == 1
        assert job_id in payload["jobs"]
        assert payload["jobs"][job_id]["job_id"] == job_id
        assert payload["jobs"][job_id]["task_type"] == "predict"
        assert payload["jobs"][job_id]["status"] == "completed"
        assert job_id in payload["job_logs"]
        assert any("submitted" in line for line in payload["job_logs"][job_id])
        # Round-trip: the persisted payload re-validates into a JobRequest.
        restored = JobRequest.model_validate(payload["jobs"][job_id])
        assert restored.status == JobStatus.COMPLETED

    def test_new_instance_restores_history(self, tmp_path: Path, sample_job_params) -> None:
        """A second JobsManager on the same storage_path sees prior jobs and logs."""
        storage = tmp_path / "jobs_state.json"
        manager = JobsManager(storage_path=str(storage))

        with patch.object(manager.dispatcher, "execute", _stub_execute):
            job_id, _ = manager.submit_job(**sample_job_params)
        self._wait_terminal(manager, job_id)

        reloaded = JobsManager(storage_path=str(storage))
        assert job_id in reloaded.jobs
        assert reloaded.jobs[job_id].status == JobStatus.COMPLETED
        assert reloaded.jobs[job_id].task_type == TaskType.PREDICT
        assert job_id in reloaded.job_logs
        assert any("submitted" in line for line in reloaded.job_logs[job_id])

    def test_orphaned_active_jobs_heal_to_failed(self, tmp_path: Path) -> None:
        """PENDING/RUNNING jobs in the state file are reset to FAILED on load."""
        storage = tmp_path / "jobs_state.json"
        pending = JobRequest(job_id="predict_orphan_pending", task_type=TaskType.PREDICT, status=JobStatus.PENDING)
        running = JobRequest(job_id="predict_orphan_running", task_type=TaskType.PREDICT, status=JobStatus.RUNNING)
        completed = JobRequest(
            job_id="predict_orphan_completed", task_type=TaskType.PREDICT, status=JobStatus.COMPLETED
        )
        payload = {
            "version": 1,
            "jobs": {j.job_id: j.model_dump(mode="json") for j in (pending, running, completed)},
            "job_logs": {pending.job_id: ["stale log line"], running.job_id: []},
        }
        storage.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        manager = JobsManager(storage_path=str(storage))
        assert manager.jobs[pending.job_id].status == JobStatus.FAILED
        assert manager.jobs[running.job_id].status == JobStatus.FAILED
        # Terminal jobs are restored untouched.
        assert manager.jobs[completed.job_id].status == JobStatus.COMPLETED
        assert manager.job_logs[pending.job_id] == ["stale log line"]


class TestRecentJobsInitialValue:
    """Recent Jobs table must pre-populate from restored persistence on UI build.

    Regression for the first-render timing bug: the backend restored persisted
    state via ``_load()``, but the ``gr.Dataframe`` mounted with an empty value,
    so the table only filled after the next submit/poll. The initial ``value``
    must now carry the restored history with no user interaction.
    """

    @staticmethod
    def _make_job(job_id: str, task_type: TaskType, status: JobStatus, created_at: str) -> JobRequest:
        """Build a persisted job with an explicit, deterministic creation time."""
        job = JobRequest(job_id=job_id, task_type=task_type, status=status)
        job.metadata.created_at = created_at
        return job

    @staticmethod
    def _write_state(storage: Path, jobs: list[JobRequest]) -> None:
        """Write a ``jobs_state.json`` payload exactly as ``JobsManager._save`` does."""
        payload = {
            "version": 1,
            "jobs": {j.job_id: j.model_dump(mode="json") for j in jobs},
            "job_logs": {j.job_id: [f"log {j.job_id}"] for j in jobs},
        }
        storage.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _find_recent_table(tab: gr.Blocks) -> gr.Dataframe:
        """Return the Recent Jobs dataframe by its localized label."""
        return next(
            b for b in tab.blocks.values() if isinstance(b, gr.Dataframe) and b.label == get_text("en", "df.recent")
        )

    @staticmethod
    def _rows(value: Any) -> list[list[str]]:
        """Normalize a ``gr.Dataframe`` value (dict / DataFrame / list) to rows."""
        if isinstance(value, dict):
            return value["data"]
        if hasattr(value, "values"):  # pandas DataFrame
            return value.values.tolist()
        return value

    def test_table_prepopulates_with_persisted_history(self, tmp_path: Path) -> None:
        """A UI built over an existing jobs_state.json shows history immediately."""
        storage = tmp_path / "jobs_state.json"
        jobs = [
            self._make_job("predict_older", TaskType.PREDICT, JobStatus.COMPLETED, "2026-09-01T10:00:00+00:00"),
            self._make_job("train_newer", TaskType.TRAIN, JobStatus.FAILED, "2026-09-02T10:00:00+00:00"),
        ]
        self._write_state(storage, jobs)

        # Backend restores the two persisted jobs.
        manager = JobsManager(storage_path=str(storage))
        assert set(manager.jobs) == {"predict_older", "train_newer"}

        # Building the UI must bind the restored history as the table's initial value.
        tab = create_jobs_tab(manager, "en")
        rows = self._rows(self._find_recent_table(tab).value)

        assert rows, "Recent Jobs table must be non-empty on first render"
        # Strictly matches the persisted records: newest first, local-time formatted.
        expected = [
            ["train_newer", "train", "FAILED", format_created_at("2026-09-02T10:00:00+00:00")],
            ["predict_older", "predict", "COMPLETED", format_created_at("2026-09-01T10:00:00+00:00")],
        ]
        assert rows == expected
        # The component value is exactly what the shared formatting helper produces,
        # so first render and every later poll refresh stay consistent.
        assert rows == recent_jobs_rows(manager, limit=20)


class TestPollIdlePreservesRecentJobs:
    """Idle polling must never blank the Recent Jobs table.

    Regression for the auto-clearing table bug: with no active job selected,
    ``compute_poll_state`` used to return a snapshot whose ``recent`` field was
    the empty default, and the always-on slow sync timer wrote that empty list
    into ``recent_jobs_table``, wiping the rows pre-populated from persistence.
    """

    @staticmethod
    def _make_job(job_id: str, task_type: TaskType, status: JobStatus, created_at: str) -> JobRequest:
        """Build a persisted job with an explicit, deterministic creation time."""
        job = JobRequest(job_id=job_id, task_type=task_type, status=status)
        job.metadata.created_at = created_at
        return job

    @staticmethod
    def _write_state(storage: Path, jobs: list[JobRequest]) -> None:
        """Write a ``jobs_state.json`` payload exactly as ``JobsManager._save`` does."""
        payload = {
            "version": 1,
            "jobs": {j.job_id: j.model_dump(mode="json") for j in jobs},
            "job_logs": {j.job_id: [] for j in jobs},
        }
        storage.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _persisted_manager(tmp_path: Path) -> tuple[JobsManager, list[list[str]]]:
        """Return a manager restored from two persisted jobs and its expected rows."""
        storage = tmp_path / "jobs_state.json"
        jobs = [
            TestPollIdlePreservesRecentJobs._make_job(
                "predict_older", TaskType.PREDICT, JobStatus.COMPLETED, "2026-09-01T10:00:00+00:00"
            ),
            TestPollIdlePreservesRecentJobs._make_job(
                "train_newer", TaskType.TRAIN, JobStatus.FAILED, "2026-09-02T10:00:00+00:00"
            ),
        ]
        TestPollIdlePreservesRecentJobs._write_state(storage, jobs)
        manager = JobsManager(storage_path=str(storage))
        return manager, recent_jobs_rows(manager, limit=20)

    def test_compute_poll_state_idle_keeps_persisted_rows(self, tmp_path: Path) -> None:
        """The no-selection branch still carries the full persisted history."""
        manager, expected = self._persisted_manager(tmp_path)

        state = compute_poll_state(manager, "", "en")

        assert state.status == {"status": "NO_SELECTION"}
        assert state.recent == expected
        assert len(state.recent) == 2
        # Rows are the exact formatted rows, never an empty list.
        assert state.recent != []

    def test_sync_timer_tick_keeps_recent_rows(self, tmp_path: Path) -> None:
        """A slow-sync tick with no active job writes the persisted rows, not ``[]``."""
        manager, expected = self._persisted_manager(tmp_path)
        tab = create_jobs_tab(manager, "en")

        # Resolve the always-on slow sync timer and its tick handler exactly as
        # Gradio invokes it (BlockFunction.fn).
        sync_timer = next(b for b in tab.blocks.values() if isinstance(b, gr.Timer) and b.value == POLL_SLOW_SECONDS)
        sync_fn = _wired_fn(tab, sync_timer, "tick")

        # Simulate an idle tick: no active job selected.
        result = sync_fn("", "en")

        # The Recent Jobs output is the final element of the handler's tuple.
        recent = result[-1]
        assert recent == expected
        assert len(recent) == 2

    def test_poll_handler_idle_keeps_recent_rows(self, tmp_path: Path) -> None:
        """The fast lifecycle poll (idle) also keeps the persisted rows intact."""
        manager, expected = self._persisted_manager(tmp_path)
        tab = create_jobs_tab(manager, "en")

        # Fast timer is inactive on idle, but its tick handler is still wired.
        poll_timer = next(b for b in tab.blocks.values() if isinstance(b, gr.Timer) and b.value != POLL_SLOW_SECONDS)
        poll_fn = _wired_fn(tab, poll_timer, "tick")

        result = poll_fn("", "en")

        # poll_handler tuple: recent is the second-to-last element, timer update last.
        recent = result[-2]
        assert recent == expected
        assert len(recent) == 2


class TestDataframeHeaderMenuHidden:
    """The Dataframe column-header three-dot menu must be hidden via local CSS."""

    def test_css_constant_targets_cell_menu_button(self) -> None:
        """The injected CSS hides the Dataframe header options button."""
        assert ".cell-menu-button" in _DATAFRAME_HEADER_MENU_CSS
        assert "display: none !important" in _DATAFRAME_HEADER_MENU_CSS

    def test_tab_injects_header_menu_css(self) -> None:
        """Building the Jobs tab emits an HTML component carrying the hiding CSS."""
        tab = create_jobs_tab(JobsManager(), "en")

        html_values = [b.value for b in tab.blocks.values() if isinstance(b, gr.HTML)]
        assert _DATAFRAME_HEADER_MENU_CSS in html_values


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
