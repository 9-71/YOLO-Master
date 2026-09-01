"""Base handler abstraction for F1 task execution.

This module defines the core abstraction for YOLO-Master F1 platform task handlers.
Each task type (predict, train, export, diagnose) implements BaseTaskHandler to provide
validation and execution logic while maintaining strict security and state machine constraints.

Architecture Principle:
    Studio bridges the product orchestration layer; it does NOT rewrite training/inference engines.
    Handlers delegate to Ultralytics YOLO engine and enforce security/state policies at dispatch boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


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
                - (False, "error description") if validation fails

        Security Rules:
            1. allow_shell MUST be False (no arbitrary shell execution)
            2. path_whitelisted MUST be True
            3. All file paths MUST resolve to allowed_paths roots
            4. Empty allowed_paths rejects ALL paths (fail-closed policy)

        Example:
            >>> handler = ConcreteTaskHandler()
            >>> params = {"model_path": "yolov8n.pt", "data_source": "../../etc/passwd"}
            >>> constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}
            >>> is_valid, err = handler.validate_params(params, constraints)
            >>> print(is_valid, err)
            False "data_source path '../../etc/passwd' not in whitelist"
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

    def _is_path_safe(self, target_path: str, allowed_roots: list[str]) -> bool:
        """Validate that target_path is within one of the allowed_roots directories.

        This is a concrete helper method implementing the path containment check.
        Handlers SHOULD use this method in their validate_params implementation.

        Args:
            target_path: Path to validate (can be relative or absolute)
            allowed_roots: List of allowed directory roots

        Returns:
            bool: True if target_path is contained within any allowed_roots, False otherwise

        Security Properties:
            - Resolves symlinks and relative paths (../../) to absolute paths
            - Checks containment via parent chain traversal
            - Returns False on empty allowed_roots (fail-closed)
            - Returns False on invalid paths (ValueError, OSError)

        Example:
            >>> handler = BaseTaskHandler()
            >>> handler._is_path_safe("ultralytics/assets/bus.jpg", [".", "runs"])
            True
            >>> handler._is_path_safe("../../etc/passwd", ["ultralytics/assets"])
            False
        """
        if not allowed_roots:
            return False
        try:
            resolved = Path(target_path).resolve()
            for root in allowed_roots:
                root_resolved = Path(root).resolve()
                if resolved == root_resolved or root_resolved in resolved.parents:
                    return True
            return False
        except (ValueError, OSError):
            return False
