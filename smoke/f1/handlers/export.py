"""Export task handler for YOLO-Master F1 platform.

This module implements the ExportHandler for converting YOLO models to deployment
formats (ONNX, TorchScript, TensorRT, etc.). It validates input parameters against
security constraints, delegates to the Ultralytics YOLO export engine, and isolates
exported artifacts per job_id.

Security Model:
    - Enforces path whitelisting for model_path
    - Validates export format against a closed allowlist (fail-closed policy)
    - Isolates output artifacts per job_id to prevent collision
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from smoke.f1.handlers.base import BaseTaskHandler
from smoke.f1.handlers.registry import TaskHandlerRegistry

# Closed allowlist of Ultralytics YOLO export formats supported by the F1 platform.
# Fail-closed policy: any format outside this set is rejected at validation stage
# before reaching the export engine.
SUPPORTED_EXPORT_FORMATS: frozenset[str] = frozenset(
    {
        "torchscript",
        "onnx",
        "openvino",
        "engine",
        "coreml",
        "saved_model",
        "pb",
        "tflite",
        "edgetpu",
        "tfjs",
        "paddle",
        "ncnn",
        "imx",
        "rknn",
        "mnn",
    }
)


@TaskHandlerRegistry.register("export")
class ExportHandler(BaseTaskHandler):
    """Handler for YOLO model export tasks.

    This handler converts YOLO models to deployment formats (ONNX, TorchScript, etc.)
    using the Ultralytics export engine. It validates security constraints, isolates
    exported artifacts per job_id, and returns structured execution results.

    Example:
        >>> handler = ExportHandler()
        >>> params = {
        ...     "model_path": "yolov8n.pt",
        ...     "format": "onnx",
        ...     "imgsz": 320,
        ...     "device": "cpu",
        ... }
        >>> constraints = {
        ...     "path_whitelisted": True,
        ...     "allow_shell": False,
        ...     "allowed_paths": [".", "runs"],
        ... }
        >>> is_valid, err = handler.validate_params(params, constraints)
        >>> if is_valid:
        ...     result = handler.execute("export-001", params, "runs/export")
        ...     print(result["success"], result["error"])
        True None
    """

    def validate_params(self, params: dict[str, Any], security_constraints: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate export task parameters against security constraints.

        Validation Rules:
            1. model_path (required): Must be within allowed_paths whitelist
            2. format (optional): Must be in SUPPORTED_EXPORT_FORMATS allowlist
            3. imgsz (optional): If present, must be int > 0
            4. device (optional): No validation (passed directly to YOLO engine)
            5. half / int8 (optional): If present, must be bool
            6. allow_shell: Must be False (inherited security constraint)

        Args:
            params: Export task parameters with keys:
                - model_path (str): Path to YOLO model weights (.pt file)
                - format (str, optional): Export format, default "onnx"
                - imgsz (int, optional): Export input image size, default 640
                - device (str, optional): Device specification ("0", "cpu", "mps")
                - half (bool, optional): FP16 export flag, default False
                - int8 (bool, optional): INT8 quantization flag, default False
            security_constraints: Security policy containing:
                - path_whitelisted (bool): Must be True
                - allow_shell (bool): Must be False
                - allowed_paths (list[str]): Whitelist of allowed directory roots

        Returns:
            tuple[bool, str | None]: (is_valid, error_message)
                - (True, None) if all validations pass
                - (False, error_description) on first validation failure

        Example:
            >>> handler = ExportHandler()
            >>> params = {"model_path": "yolov8n.pt", "format": "exe"}
            >>> constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}
            >>> is_valid, err = handler.validate_params(params, constraints)
            >>> print(is_valid)
            False
        """
        # Enforce security policy: shell execution prohibited
        if security_constraints.get("allow_shell", False):
            return False, "Shell execution is not allowed for export tasks"

        # Enforce security policy: path whitelisting required
        if not security_constraints.get("path_whitelisted", False):
            return False, "Path whitelisting must be enabled"

        allowed_paths = security_constraints.get("allowed_paths", [])
        if not allowed_paths:
            return False, "allowed_paths cannot be empty when path_whitelisted=True"

        # Validate required parameter: model_path
        if "model_path" not in params:
            return False, "Required parameter 'model_path' is missing"

        model_path = params["model_path"]
        if not self._is_path_safe(model_path, allowed_paths):
            return False, f"model_path '{model_path}' is not within allowed_paths whitelist"

        # Validate optional parameter: format (closed allowlist, fail-closed)
        if "format" in params:
            fmt = str(params["format"]).lower()
            if fmt not in SUPPORTED_EXPORT_FORMATS:
                return False, f"export format '{params['format']}' is not supported"

        # Validate optional parameter: imgsz
        if "imgsz" in params:
            try:
                imgsz = int(params["imgsz"])
                if imgsz <= 0:
                    return False, f"imgsz must be > 0, got {imgsz}"
            except (ValueError, TypeError) as e:
                return False, f"imgsz must be a valid integer, got {params['imgsz']}: {e}"

        # Validate optional parameter: half (FP16 flag)
        if "half" in params and not isinstance(params["half"], bool):
            return False, f"half must be a boolean, got {params['half']!r}"

        # Validate optional parameter: int8 (INT8 quantization flag)
        if "int8" in params and not isinstance(params["int8"], bool):
            return False, f"int8 must be a boolean, got {params['int8']!r}"

        return True, None

    def execute(self, job_id: str, params: dict[str, Any], output_dir: str) -> dict[str, Any]:
        """Execute YOLO model export and capture artifacts.

        This method performs the following steps:
            1. Copy model weights into the job-specific output directory
            2. Initialize YOLO model from the copied weights
            3. Invoke YOLO.export() with validated format and engine parameters
            4. Collect generated artifacts (exported model files)
            5. Return execution result with artifact paths

        Artifact Isolation:
            The model weights are copied into output_dir / job_id BEFORE export so
            the Ultralytics engine writes the exported artifact next to the copied
            weights — inside the job-specific directory. This prevents collisions
            between concurrent jobs and never mutates the original model directory.

        Args:
            job_id: Unique job identifier for artifact isolation (e.g., "export_20260901_001")
            params: Validated parameters containing:
                - model_path (str): Path to YOLO model weights
                - format (str, optional): Export format, default "onnx"
                - imgsz (int, optional): Export input image size, default 640
                - device (str, optional): Device specification, default "cpu"
                - half (bool, optional): FP16 export flag, default False
                - int8 (bool, optional): INT8 quantization flag, default False
            output_dir: Base directory for saving results (e.g., "runs/export")

        Returns:
            dict[str, Any]: Execution result with structure:
                - success (bool): True if export completed without errors
                - artifacts (list[str]): Absolute paths to generated export files
                - metadata (dict): Execution details (model, format, imgsz, device, exported_path)
                - error (str | None): Error message if success=False

        Raises:
            RuntimeError: If YOLO engine encounters unrecoverable errors

        Example:
            >>> handler = ExportHandler()
            >>> result = handler.execute(
            ...     job_id="export-001",
            ...     params={"model_path": "yolov8n.pt", "format": "onnx", "imgsz": 320, "device": "cpu"},
            ...     output_dir="runs/export",
            ... )
            >>> print(result["success"])
            True
        """
        try:
            from ultralytics import YOLO

            # Create job-specific output directory
            job_output_dir = Path(output_dir) / job_id
            job_output_dir.mkdir(parents=True, exist_ok=True)

            # Copy model weights into the job directory so the exported artifact
            # is generated inside the job-specific directory (artifact isolation)
            model_src = Path(params["model_path"])
            model_copy = job_output_dir / model_src.name
            shutil.copy2(model_src, model_copy)

            # Load YOLO model from the job-local copy
            model = YOLO(str(model_copy))

            # Extract export parameters
            fmt = params.get("format", "onnx")
            imgsz = int(params.get("imgsz", 640))
            device = params.get("device", "cpu")
            half = bool(params.get("half", False))
            int8 = bool(params.get("int8", False))

            # Delegate to Ultralytics export engine
            exported_path = model.export(format=fmt, imgsz=imgsz, device=device, half=half, int8=int8)

            # Collect generated artifacts (all files except the copied input weights)
            artifacts = sorted(
                str(p.resolve()) for p in job_output_dir.iterdir() if p.is_file() and p.name != model_copy.name
            )

            # Build execution metadata
            metadata = {
                "model": params["model_path"],
                "format": fmt,
                "imgsz": imgsz,
                "device": device,
                "half": half,
                "int8": int8,
                "exported_path": exported_path,
            }

            return {
                "success": True,
                "artifacts": artifacts,
                "metadata": metadata,
                "error": None,
            }

        except (OSError, ImportError, RuntimeError, ValueError) as e:
            return {
                "success": False,
                "artifacts": [],
                "metadata": {"model": params.get("model_path"), "format": params.get("format")},
                "error": f"Export failed: {type(e).__name__}: {e}",
            }
