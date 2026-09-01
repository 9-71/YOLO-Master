"""Unit tests for refactored JobDispatcher with dynamic handler resolution.

This test suite validates the Step 3-1 refactoring requirements:
    1. Dynamic polymorphic dispatch (no hardcoded if-else)
    2. Strict lifecycle and state machine enforcement
    3. Phase 1 cancellation contract (cancel_requested checking)
    4. Exception safety (all errors -> FAILED state)

Test Coverage:
    - Successful dispatch to registered handlers (predict, diagnose)
    - Security policy enforcement (shell execution, path whitelisting)
    - Cancellation signal handling (cancel_requested)
    - Unknown task type rejection
    - Parameter validation failures
    - Handler execution exceptions
    - State machine transition enforcement
"""

from __future__ import annotations

import pytest

from smoke.f1.dispatcher import JobDispatcherStateMachine
from smoke.f1.handlers.registry import TaskHandlerRegistry
from smoke.f1.test_f1_smoke import (
    JobRequest,
    JobStatus,
    OutputConfig,
    RuntimeTracking,
    SecurityConstraints,
    TaskType,
)


class TestJobDispatcherDynamicDispatch:
    """Test suite for dynamic handler resolution (Step 3-1 Requirement 1)."""

    def test_dispatch_to_predict_handler(self):
        """Verify dispatcher dynamically resolves and executes PredictHandler."""
        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-predict-001",
            task_type=TaskType.PREDICT,
            params={
                "model_path": "yolov8n.pt",
                "data_source": "ultralytics/assets/bus.jpg",
                "device": "cpu",
                "conf": 0.25,
            },
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=[".", "ultralytics/assets", "runs"],
            ),
            output=OutputConfig(output_dir="runs/predict"),
        )

        result = dispatcher.execute(job)

        assert result.status == JobStatus.COMPLETED
        assert len(result.output.artifacts) > 0
        assert result.error is None
        print(f"✓ PredictHandler dispatched successfully. Artifacts: {len(result.output.artifacts)}")

    def test_dispatch_to_diagnose_handler(self):
        """Verify dispatcher dynamically resolves and executes DiagnoseHandler."""
        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-diagnose-001",
            task_type=TaskType.DIAGNOSE,
            params={},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=["runs"],
            ),
            output=OutputConfig(output_dir="runs/diagnose"),
        )

        result = dispatcher.execute(job)

        assert result.status == JobStatus.COMPLETED
        assert len(result.output.artifacts) == 2  # JSON + TXT reports
        assert result.error is None
        print(f"✓ DiagnoseHandler dispatched successfully. Artifacts: {result.output.artifacts}")

    def test_reject_unknown_task_type(self, monkeypatch):
        """Verify dispatcher rejects task types not registered in TaskHandlerRegistry."""
        dispatcher = JobDispatcherStateMachine()

        # All four TaskType enum values (predict/train/export/diagnose) now have
        # registered handlers, so simulate an unregistered task type by making the
        # registry lookup raise ValueError (e.g., handler module failed to import).
        def unregistered_lookup(cls, task_type: str):
            raise ValueError(f"Task type '{task_type}' is not registered.")

        monkeypatch.setattr(TaskHandlerRegistry, "get", classmethod(unregistered_lookup))

        # Create a job with TRAIN task type
        job = JobRequest(
            job_id="test-unknown-001",
            task_type=TaskType.TRAIN,
            params={},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=["."],
            ),
        )

        result = dispatcher.execute(job)

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert result.error.code == "TASK_TYPE_UNKNOWN"
        assert "train" in result.error.message.lower()
        print(f"✓ Unknown task type rejected: {result.error.message}")


class TestJobDispatcherSecurityEnforcement:
    """Test suite for security policy enforcement (Step 3-1 Requirement 2)."""

    def test_block_shell_execution(self):
        """Verify dispatcher rejects jobs with allow_shell=True (SEC_ERR_001)."""
        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-security-shell-001",
            task_type=TaskType.PREDICT,
            params={"model_path": "yolov8n.pt"},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=True,  # Security violation
                allowed_paths=["."],
            ),
        )

        result = dispatcher.execute(job)

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert result.error.code == "SEC_ERR_001"
        assert "shell execution" in result.error.message.lower()
        print(f"✓ Shell execution blocked: {result.error.message}")

    def test_block_disabled_path_whitelist(self):
        """Verify dispatcher rejects jobs with path_whitelisted=False (SEC_ERR_001)."""
        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-security-whitelist-001",
            task_type=TaskType.PREDICT,
            params={"model_path": "yolov8n.pt"},
            security_constraints=SecurityConstraints(
                path_whitelisted=False,  # Security violation
                allow_shell=False,
                allowed_paths=[],
            ),
        )

        result = dispatcher.execute(job)

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert result.error.code == "SEC_ERR_001"
        assert "path whitelisting" in result.error.message.lower()
        print(f"✓ Path whitelist enforcement: {result.error.message}")

    def test_block_path_traversal_attack(self):
        """Verify handler-level path validation rejects directory traversal attacks."""
        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-security-traversal-001",
            task_type=TaskType.PREDICT,
            params={
                "model_path": "yolov8n.pt",
                "data_source": "../../etc/passwd",  # Path traversal attempt
            },
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=["ultralytics/assets"],
            ),
        )

        result = dispatcher.execute(job)

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert result.error.code == "PARAM_VALIDATION_FAILED"
        assert "not within allowed_paths" in result.error.message
        print(f"✓ Path traversal blocked: {result.error.message}")


class TestJobDispatcherCancellation:
    """Test suite for Phase 1 cancellation contract (Step 3-1 Requirement 3)."""

    def test_cancel_before_execution(self):
        """Verify dispatcher respects cancel_requested signal and skips execution."""
        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-cancel-001",
            task_type=TaskType.PREDICT,
            params={"model_path": "yolov8n.pt", "data_source": "ultralytics/assets/bus.jpg"},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=[".", "ultralytics/assets"],
            ),
            runtime_tracking=RuntimeTracking(
                cancel_requested=True,  # User cancellation signal
            ),
        )

        result = dispatcher.execute(job)

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert result.error.code == "USER_CANCELLED"
        assert "cancelled by user" in result.error.message.lower()
        assert len(result.output.artifacts) == 0  # No execution occurred
        print(f"✓ Cancellation respected: {result.error.message}")


class TestJobDispatcherExceptionSafety:
    """Test suite for exception handling and atomic FAILED transitions (Step 3-1 Requirement 4)."""

    def test_handler_validation_failure(self):
        """Verify dispatcher handles handler validation failures gracefully."""
        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-validation-001",
            task_type=TaskType.PREDICT,
            params={
                # Missing required parameter: model_path
                "data_source": "ultralytics/assets/bus.jpg",
            },
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=["."],
            ),
        )

        result = dispatcher.execute(job)

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert result.error.code == "PARAM_VALIDATION_FAILED"
        assert "model_path" in result.error.message
        print(f"✓ Validation failure handled: {result.error.message}")

    def test_handler_execution_exception(self):
        """Verify dispatcher catches unhandled exceptions and transitions to FAILED."""
        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-exception-001",
            task_type=TaskType.PREDICT,
            params={
                "model_path": "nonexistent_model.pt",  # Will cause FileNotFoundError
                "data_source": "ultralytics/assets/bus.jpg",
                "device": "cpu",
            },
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=[".", "ultralytics/assets"],
            ),
        )

        result = dispatcher.execute(job)

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        # Handler returns controlled failure with HANDLER_EXEC_FAILED or uncontrolled with EXEC_ERR_500
        assert result.error.code in ["HANDLER_EXEC_FAILED", "EXEC_ERR_500"]
        print(f"✓ Exception caught and handled: {result.error.code} - {result.error.message}")


class TestJobDispatcherStateMachine:
    """Test suite for strict state machine transition enforcement."""

    def test_illegal_transition_completed_to_running(self):
        """Verify state machine prevents COMPLETED -> RUNNING transition."""
        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-state-001",
            task_type=TaskType.PREDICT,
            params={},
        )
        job.status = JobStatus.COMPLETED

        with pytest.raises(ValueError, match="Illegal state transition"):
            dispatcher.transition(job, JobStatus.RUNNING)

        print("✓ Illegal transition COMPLETED->RUNNING prevented")

    def test_illegal_transition_pending_to_completed(self):
        """Verify state machine prevents PENDING -> COMPLETED skip."""
        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-state-002",
            task_type=TaskType.PREDICT,
            params={},
        )

        with pytest.raises(ValueError, match="Illegal state transition"):
            dispatcher.transition(job, JobStatus.COMPLETED)

        print("✓ Illegal transition PENDING->COMPLETED prevented")

    def test_valid_transition_sequence(self):
        """Verify state machine allows valid PENDING -> RUNNING -> COMPLETED sequence."""
        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-state-003",
            task_type=TaskType.PREDICT,
            params={},
        )

        assert job.status == JobStatus.PENDING

        dispatcher.transition(job, JobStatus.RUNNING)
        assert job.status == JobStatus.RUNNING

        dispatcher.transition(job, JobStatus.COMPLETED)
        assert job.status == JobStatus.COMPLETED

        print("✓ Valid state transition sequence: PENDING->RUNNING->COMPLETED")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("JobDispatcher Refactoring Test Suite (Step 3-1)")
    print("=" * 80 + "\n")

    # Run tests with pytest
    exit_code = pytest.main([__file__, "-v", "--tb=short", "--color=yes"])

    if exit_code == 0:
        print("\n" + "=" * 80)
        print("✓ ALL DISPATCHER TESTS PASSED - Step 3-1 Refactoring Complete")
        print("=" * 80 + "\n")
    else:
        print("\n" + "=" * 80)
        print("✗ SOME DISPATCHER TESTS FAILED - Review output above")
        print("=" * 80 + "\n")
