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

Phase 1 Runtime Supervision:
    - Deadline supervision: jobs exceeding runtime_tracking.timeout_seconds transition
      to FAILED with error code TIMEOUT. Handler execution runs in a daemon worker
      thread joined with the remaining deadline.
    - Cooperative cancellation: cancel_requested is checked at dispatcher checkpoints
      (pre-execution, post-validation, in-flight polling, post-execution) AND by
      handlers at their own checkpoints via BaseTaskHandler._check_cancelled(), which
      raises CooperativeCancellationError mapped to FAILED + USER_CANCELLED.

Security Red Line (Log & Env Sanitization):
    - Every error message attached to ``job.error`` is routed through
      :func:`core.security.sanitize_log_text` inside ``transition()``.
    - Handler exception tracebacks are captured, sanitized and appended to
      ``job.logs`` via the ``JobRequest.append_log`` interface (which sanitizes).
    - A sanitized environment audit line (redacted variable names only, never
      values) is recorded per executed job via ``sanitize_env_dict``.
"""

from __future__ import annotations

import os
import time
import traceback
from datetime import datetime, timezone
from queue import Queue
from threading import Thread
from typing import Any

from core.schema import ErrorInfo, JobRequest, JobStatus
from core.security import REDACTED, sanitize_env_dict, sanitize_log_text
from f1.handlers.base import CooperativeCancellationError, PathWhitelistViolationError
from f1.handlers.registry import TaskHandlerRegistry

# How often the dispatcher polls cancel_requested while a job is in flight (seconds).
# A small interval keeps cancellation latency low without busy-waiting.
CANCELLATION_POLL_INTERVAL = 0.05


class JobDispatcherStateMachine:
    """Refactored job dispatcher with dynamic handler resolution.

    This dispatcher eliminates all hardcoded task_type branching. Handler selection
    is performed dynamically at runtime via TaskHandlerRegistry.get(task_type).

    Guarantees:
        - Security policies are enforced BEFORE execution begins
        - State transitions follow strict FSM rules (no illegal jumps)
        - Cancellation requests are checked before, during, and after execution
        - Execution deadline (timeout_seconds) is enforced via worker-thread supervision
        - All exceptions are caught and converted to FAILED state
        - Handler execution is isolated (no cross-task coupling)

    Example:
        >>> from core.schema import JobRequest, TaskType
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
            >>> from core.schema import JobRequest, JobStatus, TaskType
            >>> dispatcher = JobDispatcherStateMachine()
            >>> job = JobRequest(job_id="test", task_type=TaskType.PREDICT)
            >>> dispatcher.transition(job, JobStatus.RUNNING)  # PENDING -> RUNNING (allowed)
            >>> dispatcher.transition(job, JobStatus.COMPLETED)  # RUNNING -> COMPLETED (allowed)
            >>> dispatcher.transition(job, JobStatus.PENDING)  # COMPLETED -> PENDING (raises ValueError)
            Traceback (most recent call last):
                ...
            ValueError: Illegal state transition: completed -> pending
        """
        if target_status not in self.valid_transitions[job.status]:
            raise ValueError(f"Illegal state transition: {job.status.value} -> {target_status.value}")

        job.status = target_status
        job.append_log(f"[StateMachine] Job {job.job_id} transitioned to: {target_status.value.upper()}")

        if err_code:
            # Sanitize BEFORE the message is attached to the job: ErrorInfo is
            # consumed by the UI and external queries, so plaintext credentials
            # must never reach it. ``append_log`` sanitizes its own input.
            job.error = ErrorInfo(
                code=err_code,
                message=sanitize_log_text(err_msg or ""),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            job.append_log(f"[{err_code}] {err_msg or ''}")

        print(f"  [StateMachine] Job {job.job_id} transitioned to: {job.status.value.upper()}")

    def _record_env_audit(self, job: JobRequest) -> None:
        """Record a sanitized environment snapshot audit line into ``job.logs``.

        The process environment is transformed by
        :func:`core.security.sanitize_env_dict` first; only the NAMES of
        variables whose key or value was redacted are logged (never values),
        giving an audit trail that proves the red line held without leaking
        credential material into the log buffer.

        Args:
            job: JobRequest whose sanitized log buffer receives the audit line.
        """
        sanitized_env = sanitize_env_dict(os.environ)
        redacted_keys = sorted(key for key, value in sanitized_env.items() if value == REDACTED)
        detail = f": {', '.join(redacted_keys)}" if redacted_keys else ""
        job.append_log(
            f"[SecurityAudit] Environment snapshot sanitized: "
            f"{len(redacted_keys)} sensitive variable(s) redacted{detail}"
        )

    def execute(self, job: JobRequest) -> JobRequest:
        """Execute job with dynamic handler resolution and state machine enforcement.

        Execution Flow:
            1. Security Policy Enforcement (pre-execution guard)
            2. Cancellation Check (cancel_requested signal)
            3. Dynamic Handler Resolution (TaskHandlerRegistry.get)
            4. Parameter Validation (handler.validate_params)
            5. State Transition (PENDING -> RUNNING)
            6. Deadline-Supervised Handler Execution in Worker Thread
               - Deadline = now + runtime_tracking.timeout_seconds
               - In-flight polling of cancel_requested (cooperative cancellation)
               - Deadline exceeded -> FAILED + TIMEOUT
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
            - Dispatch-time short-circuit (before execution) and in-flight cooperative
              cancellation (polling during execution + handler checkpoints) are supported
            - The in-flight worker thread is daemonized; it is detached after the
              terminal FAILED transition and its late result is discarded

        Timeout Semantics:
            - If handler execution exceeds runtime_tracking.timeout_seconds, the job
              transitions to FAILED with error code TIMEOUT
            - Timeout never produces a partial COMPLETED transition

        Exception Safety:
            - All exceptions are caught and logged
            - Job atomically transitions to FAILED with EXEC_ERR_500
            - Error message includes exception type and message

        Example:
            >>> from core.schema import JobRequest, TaskType
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
        try:
            is_valid, validation_err = handler.validate_params(
                params=job.params,
                security_constraints=job.security_constraints.model_dump(),
            )
        except PathWhitelistViolationError as e:
            # A path whitelist failure is a security event, not a parameter problem:
            # map it to SEC_ERR_001 so the UI and callers can distinguish it from
            # PARAM_VALIDATION_FAILED (which stays reserved for plain parameter issues).
            self.transition(
                job,
                JobStatus.FAILED,
                err_code="SEC_ERR_001",
                err_msg=f"Security policy violation: {e}",
            )
            print(f"  [Dispatcher] Security violation blocked: {sanitize_log_text(str(e))}")
            return job

        if not is_valid:
            self.transition(
                job,
                JobStatus.FAILED,
                err_code="PARAM_VALIDATION_FAILED",
                err_msg=f"Parameter validation failed: {validation_err}",
            )
            return job

        # =====================================================================
        # Step 4.5: Cooperative Cancellation Checkpoint (post-validation)
        # A cancellation request may arrive while validation is running.
        # =====================================================================
        if job.runtime_tracking.cancel_requested:
            self.transition(
                job,
                JobStatus.FAILED,
                err_code="USER_CANCELLED",
                err_msg="Job execution cancelled by user request",
            )
            print(f"  [Dispatcher] Job {job.job_id} cancelled after validation (cancel_requested=True)")
            return job

        # =====================================================================
        # Step 5: State Transition (PENDING -> RUNNING)
        # =====================================================================
        self.transition(job, JobStatus.RUNNING)

        # Security red line: record a sanitized environment audit trail. Only
        # the NAMES of redacted variables are logged; values never leave
        # ``sanitize_env_dict``'s redacted mapping.
        self._record_env_audit(job)

        # =====================================================================
        # Step 6: Deadline-Supervised Handler Execution with Cooperative
        #         Cancellation (Phase 1 runtime supervision)
        # =====================================================================
        timeout_seconds = float(job.runtime_tracking.timeout_seconds)
        deadline = time.monotonic() + max(timeout_seconds, 0.0)

        # Inject the job's runtime tracking so the handler can honor cooperative
        # cancellation checkpoints between long-running work items.
        handler._runtime_tracking = job.runtime_tracking

        # Execute the handler in a daemon worker thread so the dispatcher can
        # enforce the deadline and poll cancel_requested while the job is in flight.
        execution_queue: Queue[dict[str, Any]] = Queue(maxsize=1)

        def _run_handler() -> None:
            try:
                execution_queue.put(
                    {
                        "result": handler.execute(
                            job_id=job.job_id,
                            params=job.params,
                            output_dir=job.output.output_dir,
                        )
                    }
                )
            except Exception as exc:  # noqa: BLE001 - worker captures exceptions for main-thread mapping
                # Capture the full traceback so the main thread can log a
                # sanitized copy; raw tracebacks may embed credential strings
                # raised by downstream engines (DB clients, SDKs, etc.).
                execution_queue.put({"error": exc, "traceback": traceback.format_exc()})

        worker = Thread(target=_run_handler, daemon=True, name=f"f1-job-{job.job_id}")
        worker.start()

        # Deadline supervision + in-flight cooperative cancellation polling.
        # While the worker is alive and the deadline has not passed, poll
        # cancel_requested so an in-flight user cancellation aborts promptly.
        while worker.is_alive() and time.monotonic() < deadline:
            if job.runtime_tracking.cancel_requested:
                self.transition(
                    job,
                    JobStatus.FAILED,
                    err_code="USER_CANCELLED",
                    err_msg="Job cancelled in-flight by user request",
                )
                print(f"  [Dispatcher] Job {job.job_id} cancelled in-flight (cancel_requested=True)")
                return job
            worker.join(timeout=CANCELLATION_POLL_INTERVAL)

        # Deadline supervision: the handler is still running past its deadline.
        if worker.is_alive():
            self.transition(
                job,
                JobStatus.FAILED,
                err_code="TIMEOUT",
                err_msg=f"Job execution exceeded timeout of {timeout_seconds}s",
            )
            print(f"  [Dispatcher] Job {job.job_id} timed out after {timeout_seconds}s (worker detached)")
            return job

        # Worker finished within the deadline; collect its outcome.
        execution_payload = execution_queue.get_nowait()

        # =====================================================================
        # Step 6.5: Cooperative Cancellation Checkpoint (post-execution)
        # A cancellation request arriving after the handler finished but before
        # the COMPLETED transition still wins over a successful result.
        # =====================================================================
        if job.runtime_tracking.cancel_requested:
            self.transition(
                job,
                JobStatus.FAILED,
                err_code="USER_CANCELLED",
                err_msg="Job cancelled by user request after execution",
            )
            print(f"  [Dispatcher] Job {job.job_id} cancelled after execution (cancel_requested=True)")
            return job

        # Map worker exceptions: cooperative cancellation first, then generic errors.
        worker_error = execution_payload.get("error")
        if worker_error is not None:
            if isinstance(worker_error, CooperativeCancellationError):
                self.transition(
                    job,
                    JobStatus.FAILED,
                    err_code="USER_CANCELLED",
                    err_msg=str(worker_error),
                )
                print(f"  [Dispatcher] Job {job.job_id} cooperatively cancelled by handler checkpoint")
            else:
                # Uncontrolled exception during handler execution. The raw
                # traceback may carry plaintext credentials inside exception
                # messages or source lines, so it is sanitized BEFORE entering
                # the job's log buffer (append_log sanitizes again, by design).
                raw_traceback = execution_payload.get("traceback", "")
                if raw_traceback:
                    job.append_log(f"[Traceback]\n{raw_traceback}")
                error_summary = f"{type(worker_error).__name__}: {worker_error}"
                self.transition(
                    job,
                    JobStatus.FAILED,
                    err_code="EXEC_ERR_500",
                    err_msg=f"Unhandled execution exception: {error_summary}",
                )
                print(f"  [Dispatcher] Exception caught: {sanitize_log_text(error_summary)}")
            return job

        # Capture execution results
        execution_result = execution_payload["result"]
        if execution_result["success"]:
            job.output.artifacts = execution_result.get("artifacts", [])
            job.append_log(f"[Dispatcher] Execution successful. Artifacts: {len(job.output.artifacts)} files captured")
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

        return job
