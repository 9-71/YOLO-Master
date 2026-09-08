"""Base handler abstraction for F1 task execution.

This module defines the core abstraction for YOLO-Master F1 platform task handlers.
Each task type (predict, train, export, diagnose) implements BaseTaskHandler to provide
validation and execution logic while maintaining strict security and state machine constraints.

Architecture Principle:
    Studio bridges the product orchestration layer; it does NOT rewrite training/inference engines.
    Handlers delegate to Ultralytics YOLO engine and enforce security/state policies at dispatch boundary.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class CooperativeCancellationError(Exception):
    """Raised when a handler observes cancel_requested at a cooperative checkpoint.

    The dispatcher injects the job's RuntimeTracking object into the handler's
    ``_runtime_tracking`` attribute immediately before execution. Handlers call
    :meth:`BaseTaskHandler._check_cancelled` between long-running work items (e.g.,
    between batch chunks); when the user has requested cancellation, this exception
    propagates to the dispatcher, which transitions the job to FAILED with error
    code USER_CANCELLED.

    This deliberately subclasses Exception directly (NOT RuntimeError) so it is not
    swallowed by the engine-facing ``except (OSError, ImportError, RuntimeError,
    ValueError)`` guards inside handler execute() methods — those guards convert
    engine errors into structured ``success=False`` results, whereas cancellation
    must abort execution entirely.
    """


class PathWhitelistViolationError(Exception):
    """Raised by handler ``validate_params`` when a path fails the security whitelist.

    Ordinary parameter problems (missing fields, out-of-range values, wrong types)
    are reported by returning ``(False, message)``, which the dispatcher maps to
    ``PARAM_VALIDATION_FAILED``. A path whitelist failure is a SECURITY event, not a
    parameter problem: handlers raise this exception instead, and the dispatcher maps
    it to ``FAILED`` with ``error_code="SEC_ERR_001"``.

    Deliberately subclasses Exception directly (NOT ValueError) so generic parameter
    handling can never reclassify a security violation as a plain validation failure.
    """


class BaseTaskHandler(ABC):
    """Abstract base class for F1 task execution handlers.

    Each concrete handler (PredictHandler, TrainHandler, etc.) implements validation
    and execution logic for a specific task_type while adhering to security constraints
    and the JobRequest contract schema.

    Contract:
        - validate_params: Pre-execution security and parameter validation
        - execute: Task execution with artifact capture and error handling

    Security Model:
        Handlers MUST NOT bypass security constraints. Path whitelisting, shell execution
        restrictions, and timeout enforcement are mandatory at validation stage.
    """

    # Injected by the dispatcher immediately before execute() so handlers can honor
    # cooperative cancellation checkpoints between long-running work items. Without
    # injection (direct handler usage), _check_cancelled() is a no-op.
    _runtime_tracking: Any | None = None

    @abstractmethod
    def validate_params(self, params: dict[str, Any], security_constraints: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate task parameters against security constraints.

        This method enforces security policies BEFORE task execution begins:
        - Path whitelisting (model_path, data_source, output_dir)
        - Shell execution prohibition
        - Resource limit verification (timeout, memory)

        Args:
            params: Task-specific parameters from JobRequest.params
                Common keys: model_path, data_source, device, conf_threshold
            security_constraints: Security policy from JobRequest.security_constraints
                Required keys: path_whitelisted, allow_shell, allowed_paths

        Returns:
            tuple[bool, str | None]: (is_valid, error_message)
                - (True, None) if validation passes
                - (False, "error description") if a parameter problem is found

        Raises:
            PathWhitelistViolationError: If a path fails the whitelist containment /
                regex check (a security event the dispatcher maps to SEC_ERR_001).

        Security Rules:
            1. allow_shell MUST be False (no arbitrary shell execution)
            2. path_whitelisted MUST be True
            3. All file paths MUST resolve to allowed_paths roots
            4. Empty allowed_paths rejects ALL paths (fail-closed policy)

        Example:
            >>> handler = ConcreteTaskHandler()
            >>> params = {"model_path": "yolov8n.pt", "data_source": "../../etc/passwd"}
            >>> constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}
            >>> try:
            ...     handler.validate_params(params, constraints)
            ... except PathWhitelistViolationError as e:
            ...     print("security violation:", e)
            security violation: data_source '../../etc/passwd' is not within allowed_paths whitelist
        """

    @abstractmethod
    def execute(self, job_id: str, params: dict[str, Any], output_dir: str) -> dict[str, Any]:
        """Execute the task and return execution results with artifacts.

        This method performs the actual task execution (predict, train, etc.) by delegating
        to the Ultralytics YOLO engine. It captures output artifacts, execution metadata,
        and converts any engine exceptions into structured error responses.

        Args:
            job_id: Unique job identifier for artifact isolation (e.g., "job_20260824_f1_001")
            params: Validated task parameters (already passed validate_params)
                Task-specific keys vary by handler type
            output_dir: Base directory for saving results (e.g., "runs/predict")

        Returns:
            dict[str, Any]: Execution result containing:
                - "success": bool (True if execution completed without errors)
                - "artifacts": list[str] (paths to generated output files)
                - "metadata": dict (task-specific execution details)
                - "error": str | None (error message if success=False)

        Raises:
            RuntimeError: If validation was bypassed or handler encounters unrecoverable state

        Contract Guarantee:
            - Output artifacts MUST be isolated per job_id to prevent collision
            - Artifact paths MUST be absolute and verified to exist
            - GPU memory MUST be released after execution (use context managers)
            - Execution MUST respect timeout_seconds from runtime_tracking

        Example:
            >>> handler = PredictHandler()
            >>> result = handler.execute(
            ...     job_id="test-001",
            ...     params={"model_path": "yolov8n.pt", "data_source": "bus.jpg", "device": "0"},
            ...     output_dir="runs/predict",
            ... )
            >>> print(result["success"], len(result["artifacts"]))
            True 1
        """

    def _is_path_safe(
        self, target_path: str, allowed_roots: list[str], allowed_patterns: list[str] | None = None
    ) -> bool:
        """Validate that target_path is within one of the allowed_roots directories or matches an allowed regex pattern.

        This is a concrete helper method implementing the path containment check.
        Handlers SHOULD use this method in their validate_params implementation.

        Regex whitelisting (P1):
            Entries in ``allowed_roots`` starting with ``^``, plus every entry in
            ``allowed_patterns``, are treated as regex patterns instead of directory
            roots. Each pattern is matched with ``re.match`` against the NORMALIZED
            RESOLVED POSIX path string (symlinks and ``..`` already resolved), and
            the match must consume the ENTIRE path string — a prefix-only pattern
            can never whitelist unintended siblings (no substring bypass).

        Args:
            target_path: Path to validate (can be relative or absolute)
            allowed_roots: List of allowed directory roots and/or regex patterns
                (an entry starting with ``^`` is interpreted as a regex pattern)
            allowed_patterns: Optional additional regex patterns to accept

        Returns:
            bool: True if target_path is contained within any allowed_roots or
                fully matches an allowed pattern, False otherwise

        Security Properties:
            - Resolves symlinks and relative paths (../../) to absolute paths
            - Checks containment via parent chain traversal
            - Regex patterns are evaluated against the resolved path only, so
              traversal/symlink escapes that rewrite the resolved path also
              defeat any pattern written for the literal path
            - Malformed regex entries are skipped (fail-closed, never raise)
            - Returns False when BOTH the whitelist and the pattern set are empty
            - Returns False on invalid paths (ValueError, OSError)

        Example:
            >>> handler = BaseTaskHandler()
            >>> handler._is_path_safe("ultralytics/assets/bus.jpg", [".", "runs"])
            True
            >>> handler._is_path_safe("../../etc/passwd", ["ultralytics/assets"])
            False
        """
        # Partition regex patterns out of the directory roots: entries starting with
        # ``^`` are patterns, everything else is a containment root. Patterns never
        # pass through Path().resolve() — their metacharacters are not filesystem
        # paths and resolving them could raise on platforms like Windows.
        dir_entries = [entry for entry in allowed_roots if not entry.startswith("^")]
        pattern_entries = [entry for entry in allowed_roots if entry.startswith("^")]
        if allowed_patterns:
            pattern_entries.extend(allowed_patterns)
        if not dir_entries and not pattern_entries:
            return False
        try:
            resolved = Path(target_path).resolve()
        except (ValueError, OSError):
            return False
        # 1) Directory containment (unchanged legacy semantics)
        for root in dir_entries:
            try:
                root_resolved = Path(root).resolve()
            except (ValueError, OSError):
                continue  # unusable entry contributes nothing (fail-closed)
            if resolved == root_resolved or root_resolved in resolved.parents:
                return True
        # 2) Regex whitelisting against the normalized resolved POSIX path string
        resolved_str = resolved.as_posix()
        for pattern in pattern_entries:
            try:
                match = re.match(pattern, resolved_str)
            except re.error:
                continue  # malformed pattern contributes nothing (fail-closed)
            # re.match anchors only at the start; require the match to consume the
            # whole path so partial/substring patterns cannot whitelist siblings.
            if match is not None and match.end() == len(resolved_str):
                return True
        return False

    def _check_cancelled(self) -> None:
        """Cooperative cancellation checkpoint: raise if the user requested cancellation.

        The dispatcher injects the job's RuntimeTracking object into ``_runtime_tracking``
        before calling execute(). Handlers call this helper between long-running work
        items (e.g., between batch chunks of a multi-source prediction) so an in-flight
        cancellation request aborts execution promptly instead of waiting for the
        current engine call to finish.

        Raises:
            CooperativeCancellationError: If the injected runtime tracking has
                cancel_requested=True.

        Notes:
            - Without an injected tracking token (direct handler usage), this is a no-op.
            - The dispatcher maps the raised exception to FAILED + USER_CANCELLED.

        Example:
            >>> BaseTaskHandler._runtime_tracking  # No token injected by default
            None
        """
        tracking = getattr(self, "_runtime_tracking", None)
        if tracking is not None and getattr(tracking, "cancel_requested", False):
            raise CooperativeCancellationError("Job execution cancelled by user request (cooperative checkpoint)")
