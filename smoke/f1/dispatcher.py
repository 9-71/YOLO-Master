"""Job dispatcher with dynamic handler resolution and state machine enforcement.

This module implements the refactored JobDispatcher that eliminates hardcoded task_type
branching logic. It dynamically resolves handlers via TaskHandlerRegistry and enforces
strict security policies, state transitions, and cancellation semantics.

Architecture Principle:
    The dispatcher is a pure orchestration layer. It does NOT contain task-specific logic.
    All task execution is delegated to registered handlers via polymorphic dispatch.

Refactoring Goals (Step 3-1):
    1. Dynamic polymorphic dispatch via TaskHandlerRegistry.get()
    2. Strict lifecycle and state machine enforcement
    3. Phase 1 contract support (cancel_requested checking)
    4. Exception safety with atomic FAILED transitions
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smoke.f1.test_f1_smoke import JobRequest, JobStatus

from smoke.f1.handlers.registry import TaskHandlerRegistry


class JobDispatcherStateMachine:
    """Refactored job dispatcher with dynamic handler resolution.

    This dispatcher eliminates all hardcoded task_type branching. Handler selection
    is performed dynamically at runtime via TaskHandlerRegistry.get(task_type).

    Guarantees:
        - Security policies are enforced BEFORE execution begins
        - State transitions follow strict FSM rules (no illegal jumps)
        - Cancellation requests are checked before execution
        - All exceptions are caught and converted to FAILED state
        - Handler execution is isolated (no cross-task coupling)

    Example:
        >>> from smoke.f1.test_f1_smoke import JobRequest, TaskType
        >>> dispatcher = JobDispatcherStateMachine()
        >>> job = JobRequest(
        ...     job_id="test-001",
        ...     task_type=TaskType.PREDICT,
        ...     params={"model_path": "yolov8n.pt", "data_source": "bus.jpg"},
        ...     security_constraints={"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]},
        ... )
        >>> result = dispatcher.execute(job)
        >>> print(result.status)
        JobStatus.COMPLETED
    """

    def __init__(self) -> None:
        """Initialize state machine with valid transition rules."""
        from smoke.f1.test_f1_smoke import JobStatus

        self.valid_transitions: dict[JobStatus, list[JobStatus]] = {
            JobStatus.PENDING: [JobStatus.RUNNING, JobStatus.FAILED],
            JobStatus.RUNNING: [JobStatus.COMPLETED, JobStatus.FAILED],
            JobStatus.COMPLETED: [],
            JobStatus.FAILED: [],
        }

    def transition(
        self,
        job: JobRequest,
        target_status: JobStatus,
        err_code: str | None = None,
        err_msg: str | None = None,
    ) -> None:
        """Enforce strict state transitions with FSM validation.

        Args:
            job: JobRequest to transition
            target_status: Target state (must be in valid_transitions for current state)
            err_code: Error code for FAILED transitions (e.g., "SEC_ERR_001", "USER_CANCELLED")
            err_msg: Human-readable error message

        Raises:
            ValueError: If transition is not allowed by FSM rules

        Example:
            >>> dispatcher = JobDispatcherStateMachine()
            >>> job = JobRequest(job_id="test", task_type=TaskType.PREDICT)
            >>> dispatcher.transition(job, JobStatus.RUNNING)  # PENDING -> RUNNING (allowed)
            >>> dispatcher.transition(job, JobStatus.COMPLETED)  # RUNNING -> COMPLETED (allowed)
            >>> dispatcher.transition(job, JobStatus.PENDING)  # COMPLETED -> PENDING (raises ValueError)
            Traceback (most recent call last):
                ...
            ValueError: Illegal state transition: completed -> pending
        """
        from smoke.f1.test_f1_smoke import ErrorInfo

        if target_status not in self.valid_transitions[job.status]:
            raise ValueError(f"Illegal state transition: {job.status.value} -> {target_status.value}")

        job.status = target_status

        if err_code:
            job.error = ErrorInfo(
                code=err_code,
                message=err_msg or "",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        print(f"  [StateMachine] Job {job.job_id} transitioned to: {job.status.value.upper()}")

    def execute(self, job: JobRequest) -> JobRequest:
        """Execute job with dynamic handler resolution and state machine enforcement.

        Execution Flow:
            1. Security Policy Enforcement (pre-execution guard)
            2. Cancellation Check (cancel_requested signal)
            3. Dynamic Handler Resolution (TaskHandlerRegistry.get)
            4. Parameter Validation (handler.validate_params)
            5. State Transition (PENDING -> RUNNING)
            6. Handler Execution (handler.execute)
            7. Artifact Capture & State Transition (RUNNING -> COMPLETED)
            8. Exception Handling (any error -> FAILED with error code)

        Args:
            job: JobRequest with task_type, params, security_constraints, runtime_tracking

        Returns:
            JobRequest: Updated job with final status (COMPLETED or FAILED) and artifacts

        Security Constraints Enforced:
            - allow_shell MUST be False (SEC_ERR_001)
            - path_whitelisted MUST be True (SEC_ERR_001)
            - All paths MUST be within allowed_paths whitelist (delegated to handler)

        Cancellation Semantics:
            - If runtime_tracking.cancel_requested is True, job transitions to FAILED
            - Error code: USER_CANCELLED
            - No handler execution occurs (early exit)

        Exception Safety:
            - All exceptions are caught and logged
            - Job atomically transitions to FAILED with EXEC_ERR_500
            - Error message includes exception type and message

        Example:
            >>> dispatcher = JobDispatcherStateMachine()
            >>> job = JobRequest(
            ...     job_id="test-001",
            ...     task_type=TaskType.DIAGNOSE,
            ...     security_constraints={"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]},
            ... )
            >>> result = dispatcher.execute(job)
            >>> print(result.status, result.error)
            JobStatus.COMPLETED None
        """
        from smoke.f1.test_f1_smoke import JobStatus

        # =====================================================================
        # Step 1: Security Policy Enforcement (Pre-Execution Guard)
        # =====================================================================
        # Rule 1: Shell execution prohibited
        if job.security_constraints.allow_shell:
            self.transition(
                job,
                JobStatus.FAILED,
                err_code="SEC_ERR_001",
                err_msg="Security policy violation: Shell execution not permitted",
            )
            return job

        # Rule 2: Path whitelisting required
        if not job.security_constraints.path_whitelisted:
            self.transition(
                job,
                JobStatus.FAILED,
                err_code="SEC_ERR_001",
                err_msg="Security policy violation: Path whitelisting must be enabled",
            )
            return job

        # =====================================================================
        # Step 2: Phase 1 Contract - Cancellation Check
        # =====================================================================
        if job.runtime_tracking.cancel_requested:
            self.transition(
                job,
                JobStatus.FAILED,
                err_code="USER_CANCELLED",
                err_msg="Job execution cancelled by user request",
            )
            print(f"  [Dispatcher] Job {job.job_id} cancelled before execution (cancel_requested=True)")
            return job

        # =====================================================================
        # Step 3: Dynamic Handler Resolution (NO HARDCODED IF-ELSE)
        # =====================================================================
        try:
            handler_class = TaskHandlerRegistry.get(job.task_type.value)
            handler = handler_class()
            print(f"  [Dispatcher] Resolved handler: {handler_class.__name__} for task_type={job.task_type.value}")
        except ValueError as e:
            # Task type not registered in TaskHandlerRegistry
            self.transition(
                job,
                JobStatus.FAILED,
                err_code="TASK_TYPE_UNKNOWN",
                err_msg=f"Unknown task type '{job.task_type.value}': {e}",
            )
            return job

        # =====================================================================
        # Step 4: Parameter Validation (Handler-Specific)
        # =====================================================================
        is_valid, validation_err = handler.validate_params(
            params=job.params,
            security_constraints=job.security_constraints.model_dump(),
        )

        if not is_valid:
            self.transition(
                job,
                JobStatus.FAILED,
                err_code="PARAM_VALIDATION_FAILED",
                err_msg=f"Parameter validation failed: {validation_err}",
            )
            return job

        # =====================================================================
        # Step 5: State Transition (PENDING -> RUNNING)
        # =====================================================================
        self.transition(job, JobStatus.RUNNING)

        # =====================================================================
        # Step 6: Handler Execution with Exception Safety
        # =====================================================================
        try:
            execution_result = handler.execute(
                job_id=job.job_id,
                params=job.params,
                output_dir=job.output.output_dir,
            )

            # Capture execution results
            if execution_result["success"]:
                job.output.artifacts = execution_result.get("artifacts", [])
                print(f"  [Dispatcher] Execution successful. Artifacts: {len(job.output.artifacts)} files captured")
                self.transition(job, JobStatus.COMPLETED)
            else:
                # Handler returned success=False (controlled failure)
                error_msg = execution_result.get("error", "Unknown handler error")
                self.transition(
                    job,
                    JobStatus.FAILED,
                    err_code="HANDLER_EXEC_FAILED",
                    err_msg=error_msg,
                )

        except Exception as e:  # noqa: BLE001 - Dispatcher must catch all execution exceptions
            # Uncontrolled exception during handler execution
            self.transition(
                job,
                JobStatus.FAILED,
                err_code="EXEC_ERR_500",
                err_msg=f"Unhandled execution exception: {type(e).__name__}: {e}",
            )
            print(f"  [Dispatcher] Exception caught: {type(e).__name__}: {e}")

        return job
