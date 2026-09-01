"""Agent Skills wrapper for YOLO-Master F1 platform.

This module provides the Agent-facing API layer that bridges high-level Agent capabilities
with the backend JobDispatcherStateMachine and TaskHandlerRegistry. Skills expose standard
tool metadata (name, description, typed parameters schema) and translate skill calls into
strongly-typed JobRequest instances.

Architecture:
    Agent Layer → Skills (this module) → Dispatcher → Handlers → YOLO Engine

Security Model:
    - Fail-closed path whitelisting with dynamic boundary construction
    - No shell execution permitted
    - All paths validated against allowed_paths whitelist
    - Security constraints enforced at skill boundary before dispatch

Example:
    >>> skill = SystemDoctorSkill()
    >>> result = skill.execute(output_dir="runs/diagnose")
    >>> print(result["status"], result["summary"]["python_version"])
    success 3.11.0

    >>> skill = PredictSkill()
    >>> result = skill.execute(
    ...     model_path="yolov8n.pt",
    ...     data_source="bus.jpg",
    ...     allowed_paths=["."],
    ... )
    >>> print(result["status"], len(result["artifacts"]))
    success 3
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from smoke.f1.dispatcher import JobDispatcherStateMachine
from smoke.f1.test_f1_smoke import JobRequest, JobStatus, SecurityConstraints, TaskType


class SystemDoctorSkill:
    """Agent skill for system environment diagnostics.

    This skill wraps the DiagnoseHandler to provide Agent-friendly system diagnostics
    collection. It gathers hardware/software information (Python, PyTorch, CUDA, GPU)
    and returns structured diagnostic summaries.

    Tool Metadata:
        - name: "system_doctor"
        - description: "Collect system diagnostics for debugging and environment verification"
        - parameters: output_dir (optional), job_id (optional)

    Returns:
        Structured diagnostic payload with summary, artifacts, and status code

    Example:
        >>> skill = SystemDoctorSkill()
        >>> result = skill.execute(output_dir="runs/diagnose")
        >>> print(result["status"])
        success
    """

    @property
    def name(self) -> str:
        """Skill name for agent tool registration."""
        return "system_doctor"

    @property
    def description(self) -> str:
        """Human-readable skill description."""
        return (
            "Collect comprehensive system environment diagnostics including Python version, "
            "PyTorch version, CUDA availability, GPU specifications, and Ultralytics version. "
            "Returns structured diagnostic report suitable for debugging and environment verification."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for skill parameters."""
        return {
            "type": "object",
            "properties": {
                "output_dir": {
                    "type": "string",
                    "description": "Base directory for saving diagnostic reports",
                    "default": "runs/diagnose",
                },
                "job_id": {
                    "type": "string",
                    "description": "Optional job identifier for artifact isolation (auto-generated if not provided)",
                    "default": None,
                },
            },
            "required": [],
        }

    def execute(
        self,
        output_dir: str = "runs/diagnose",
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute system diagnostics collection.

        Args:
            output_dir: Base directory for saving reports (default: "runs/diagnose")
            job_id: Optional job identifier for artifact isolation (auto-generated if None)

        Returns:
            dict[str, Any]: Structured result with keys:
                - status (str): "success" or "failed"
                - summary (dict): Diagnostic summary (python_version, pytorch_version, cuda_available, gpu_count)
                - artifacts (list[str]): Absolute paths to generated reports (JSON, TXT)
                - job_id (str): Job identifier used for this execution
                - error_code (str | None): Error code if status="failed"
                - error_message (str | None): Human-readable error message

        Example:
            >>> skill = SystemDoctorSkill()
            >>> result = skill.execute()
            >>> print(result["status"], result["summary"]["python_version"])
            success 3.11.0
        """
        # Generate job_id if not provided
        if job_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            job_id = f"diag_{timestamp}_{uuid.uuid4().hex[:8]}"

        # Construct JobRequest with security constraints
        # Diagnose task requires no input paths, only output directory access
        job_request = JobRequest(
            job_id=job_id,
            task_type=TaskType.DIAGNOSE,
            params={},
            output={"output_dir": output_dir},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=[output_dir],  # Only allow writing to output_dir
            ),
        )

        # Dispatch to handler via state machine
        dispatcher = JobDispatcherStateMachine()
        result = dispatcher.execute(job_request)

        # Parse result into agent-friendly format
        if result.status == JobStatus.COMPLETED:
            # Extract artifacts from output
            if hasattr(result.output, "artifacts"):
                artifacts = result.output.artifacts
            else:
                artifacts = []

            # Extract diagnostic summary from first JSON artifact
            summary = {}
            if artifacts:
                json_artifact = next((a for a in artifacts if a.endswith(".json")), None)
                if json_artifact:
                    try:
                        import json
                        from pathlib import Path

                        with Path(json_artifact).open(encoding="utf-8") as f:
                            diagnostic_data = json.load(f)
                            # Build summary from diagnostic data
                            summary = {
                                "python_version": diagnostic_data.get("python", {}).get("version", "N/A"),
                                "pytorch_version": diagnostic_data.get("pytorch", {}).get("version", "N/A"),
                                "cuda_available": diagnostic_data.get("pytorch", {}).get("cuda_available", False),
                                "gpu_count": diagnostic_data.get("gpu", {}).get("device_count", 0),
                            }
                    except (OSError, json.JSONDecodeError):
                        pass

            return {
                "status": "success",
                "summary": summary,
                "artifacts": artifacts,
                "job_id": job_id,
                "error_code": None,
                "error_message": None,
            }
        else:
            # Extract error information
            error_code = result.error.code if result.error else "UNKNOWN_ERROR"
            error_message = result.error.message if result.error else "Unknown execution failure"

            return {
                "status": "failed",
                "summary": {},
                "artifacts": [],
                "job_id": job_id,
                "error_code": error_code,
                "error_message": error_message,
            }


class PredictSkill:
    """Agent skill for YOLO object detection inference.

    This skill wraps the PredictHandler to provide Agent-friendly inference execution.
    It validates input paths against dynamic whitelist boundaries, delegates to the
    Ultralytics YOLO engine, and returns structured predictions with artifact paths.

    Tool Metadata:
        - name: "yolo_predict"
        - description: "Execute YOLO object detection inference on images/videos"
        - parameters: model_path, data_source, allowed_paths, conf, device, output_dir, job_id

    Security:
        - Enforces fail-closed path whitelisting
        - model_path and data_source MUST be within allowed_paths
        - No shell execution permitted

    Returns:
        Structured prediction result with artifacts, metadata, and status code

    Example:
        >>> skill = PredictSkill()
        >>> result = skill.execute(
        ...     model_path="yolov8n.pt",
        ...     data_source="bus.jpg",
        ...     allowed_paths=["."],
        ... )
        >>> print(result["status"], len(result["artifacts"]))
        success 3
    """

    @property
    def name(self) -> str:
        """Skill name for agent tool registration."""
        return "yolo_predict"

    @property
    def description(self) -> str:
        """Human-readable skill description."""
        return (
            "Execute YOLO object detection inference on images, videos, or directories. "
            "Validates security constraints (path whitelisting), delegates to Ultralytics YOLO engine, "
            "and returns structured predictions with annotated artifacts (images, labels, metadata)."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for skill parameters."""
        return {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to YOLO model weights (.pt file, e.g., 'yolov8n.pt')",
                },
                "data_source": {
                    "type": "string",
                    "description": "Path to input image, video, or directory",
                },
                "allowed_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Whitelist of allowed directory roots for path validation",
                    "minItems": 1,
                },
                "conf": {
                    "type": "number",
                    "description": "Confidence threshold for detections (0.0, 1.0]",
                    "default": 0.25,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "exclusiveMinimum": True,
                },
                "device": {
                    "type": "string",
                    "description": "Device specification ('0' for GPU 0, 'cpu', 'mps')",
                    "default": "cpu",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Base directory for saving prediction results",
                    "default": "runs/predict",
                },
                "job_id": {
                    "type": "string",
                    "description": "Optional job identifier for artifact isolation (auto-generated if not provided)",
                    "default": None,
                },
            },
            "required": ["model_path", "data_source", "allowed_paths"],
        }

    def execute(
        self,
        model_path: str,
        data_source: str,
        allowed_paths: list[str],
        conf: float = 0.25,
        device: str = "cpu",
        output_dir: str = "runs/predict",
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute YOLO object detection inference.

        Args:
            model_path: Path to YOLO model weights (.pt file)
            data_source: Path to input image/video/directory
            allowed_paths: Whitelist of allowed directory roots
            conf: Confidence threshold (0.0, 1.0], default 0.25
            device: Device specification ("0", "cpu", "mps"), default "cpu"
            output_dir: Base directory for results, default "runs/predict"
            job_id: Optional job identifier (auto-generated if None)

        Returns:
            dict[str, Any]: Structured result with keys:
                - status (str): "success" or "failed"
                - artifacts (list[str]): Absolute paths to annotated images/labels
                - metadata (dict): Execution details (model, source, device, conf, num_results)
                - job_id (str): Job identifier used for this execution
                - error_code (str | None): Error code if status="failed"
                - error_message (str | None): Human-readable error message

        Security Validation:
            - model_path MUST be within allowed_paths
            - data_source MUST be within allowed_paths
            - allowed_paths MUST NOT be empty
            - conf MUST be in range (0.0, 1.0]

        Example:
            >>> skill = PredictSkill()
            >>> result = skill.execute(
            ...     model_path="yolov8n.pt",
            ...     data_source="ultralytics/assets/bus.jpg",
            ...     allowed_paths=[".", "runs"],
            ...     conf=0.25,
            ...     device="cpu",
            ... )
            >>> print(result["status"])
            success
        """
        # Input validation at skill boundary
        if not allowed_paths:
            return {
                "status": "failed",
                "artifacts": [],
                "metadata": {},
                "job_id": job_id or "N/A",
                "error_code": "PARAM_VALIDATION_FAILED",
                "error_message": "allowed_paths cannot be empty (fail-closed security policy)",
            }

        # Validate confidence threshold
        try:
            conf_float = float(conf)
            if not (0.0 < conf_float <= 1.0):
                return {
                    "status": "failed",
                    "artifacts": [],
                    "metadata": {},
                    "job_id": job_id or "N/A",
                    "error_code": "PARAM_VALIDATION_FAILED",
                    "error_message": f"conf must be in range (0.0, 1.0], got {conf}",
                }
        except (ValueError, TypeError) as e:
            return {
                "status": "failed",
                "artifacts": [],
                "metadata": {},
                "job_id": job_id or "N/A",
                "error_code": "PARAM_VALIDATION_FAILED",
                "error_message": f"conf must be a valid float, got {conf}: {e}",
            }

        # Generate job_id if not provided
        if job_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            job_id = f"predict_{timestamp}_{uuid.uuid4().hex[:8]}"

        # Construct dynamic whitelist: include model_path, data_source roots, and output_dir
        # This ensures the handler can access input files AND write to output
        dynamic_whitelist = list(set(allowed_paths + [output_dir]))

        # Construct JobRequest with security constraints
        job_request = JobRequest(
            job_id=job_id,
            task_type=TaskType.PREDICT,
            params={
                "model_path": model_path,
                "data_source": data_source,
                "conf": conf_float,
                "device": device,
            },
            output={"output_dir": output_dir},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=dynamic_whitelist,
            ),
        )

        # Dispatch to handler via state machine
        dispatcher = JobDispatcherStateMachine()
        result = dispatcher.execute(job_request)

        # Parse result into agent-friendly format
        if result.status == JobStatus.COMPLETED:
            # Extract artifacts from output
            if hasattr(result.output, "artifacts"):
                artifacts = result.output.artifacts
            else:
                artifacts = []

            # Reconstruct metadata from params (handler execution result not stored in job)
            metadata = {
                "model": model_path,
                "source": data_source,
                "device": device,
                "conf": conf_float,
                "num_artifacts": len(artifacts),
            }

            return {
                "status": "success",
                "artifacts": artifacts,
                "metadata": metadata,
                "job_id": job_id,
                "error_code": None,
                "error_message": None,
            }
        else:
            # Extract error information
            error_code = result.error.code if result.error else "UNKNOWN_ERROR"
            error_message = result.error.message if result.error else "Unknown execution failure"

            return {
                "status": "failed",
                "artifacts": [],
                "metadata": {"model": model_path, "source": data_source},
                "job_id": job_id,
                "error_code": error_code,
                "error_message": error_message,
            }
