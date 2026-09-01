"""Predict task handler for YOLO-Master F1 platform.

This module implements the PredictHandler for executing YOLO object detection inference tasks.
It validates input parameters against security constraints, delegates to the Ultralytics YOLO
engine for inference, and captures generated artifacts (annotated images, labels, metadata).

Security Model:
    - Enforces path whitelisting for model_path and data_source
    - Validates confidence threshold bounds [0.0, 1.0]
    - Isolates output artifacts per job_id to prevent collision
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smoke.f1.handlers.base import BaseTaskHandler
from smoke.f1.handlers.registry import TaskHandlerRegistry


@TaskHandlerRegistry.register("predict")
class PredictHandler(BaseTaskHandler):
    """Handler for YOLO object detection inference tasks.

    This handler executes real-time object detection on images/videos using pre-trained
    YOLO models. It validates security constraints, invokes the Ultralytics YOLO engine,
    and persists annotated results to the output directory.

    Example:
        >>> handler = PredictHandler()
        >>> params = {
        ...     "model_path": "yolov8n.pt",
        ...     "data_source": "ultralytics/assets/bus.jpg",
        ...     "device": "0",
        ...     "conf": 0.25,
        ... }
        >>> constraints = {
        ...     "path_whitelisted": True,
        ...     "allow_shell": False,
        ...     "allowed_paths": [".", "runs"],
        ... }
        >>> is_valid, err = handler.validate_params(params, constraints)
        >>> if is_valid:
        ...     result = handler.execute("job-001", params, "runs/predict")
        ...     print(result["success"], len(result["artifacts"]))
        True 1
    """

    def validate_params(self, params: dict[str, Any], security_constraints: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate predict task parameters against security constraints.

        Validation Rules:
            1. model_path (required): Must be within allowed_paths whitelist
            2. data_source (required): Must be within allowed_paths whitelist
            3. conf_threshold (optional): Must be float in range (0.0, 1.0]
            4. device (optional): No validation (passed directly to YOLO engine)
            5. allow_shell: Must be False (inherited security constraint)

        Args:
            params: Predict task parameters with keys:
                - model_path (str): Path to YOLO model weights (.pt file)
                - data_source (str): Path to input image/video/directory
                - conf (float, optional): Confidence threshold, default 0.25
                - device (str, optional): Device specification ("0", "cpu", "mps")
            security_constraints: Security policy containing:
                - path_whitelisted (bool): Must be True
                - allow_shell (bool): Must be False
                - allowed_paths (list[str]): Whitelist of allowed directory roots

        Returns:
            tuple[bool, str | None]: (is_valid, error_message)
                - (True, None) if all validations pass
                - (False, error_description) on first validation failure

        Example:
            >>> handler = PredictHandler()
            >>> params = {"model_path": "yolov8n.pt", "data_source": "../../etc/passwd"}
            >>> constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}
            >>> is_valid, err = handler.validate_params(params, constraints)
            >>> print(is_valid)
            False
        """
        # Enforce security policy: shell execution prohibited
        if security_constraints.get("allow_shell", False):
            return False, "Shell execution is not allowed for predict tasks"

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

        # Validate required parameter: data_source
        if "data_source" not in params:
            return False, "Required parameter 'data_source' is missing"

        data_source = params["data_source"]
        if not self._is_path_safe(data_source, allowed_paths):
            return False, f"data_source '{data_source}' is not within allowed_paths whitelist"

        # Validate optional parameter: conf (confidence threshold)
        if "conf" in params:
            try:
                conf = float(params["conf"])
                if not (0.0 < conf <= 1.0):
                    return False, f"conf must be in range (0.0, 1.0], got {conf}"
            except (ValueError, TypeError) as e:
                return False, f"conf must be a valid float, got {params['conf']}: {e}"

        return True, None

    def execute(self, job_id: str, params: dict[str, Any], output_dir: str) -> dict[str, Any]:
        """Execute YOLO object detection inference and capture artifacts.

        This method performs the following steps:
            1. Initialize YOLO model from params["model_path"]
            2. Configure job-specific output directory (output_dir / job_id)
            3. Invoke YOLO.predict() with device, conf, and save parameters
            4. Collect generated artifacts (annotated images, labels, metadata)
            5. Return execution result with artifact paths

        Args:
            job_id: Unique job identifier for artifact isolation (e.g., "job_20260901_001")
            params: Validated parameters containing:
                - model_path (str): Path to YOLO model weights
                - data_source (str): Path to input image/video/directory
                - device (str, optional): Device specification, default "cpu"
                - conf (float, optional): Confidence threshold, default 0.25
            output_dir: Base directory for saving results (e.g., "runs/predict")

        Returns:
            dict[str, Any]: Execution result with structure:
                - success (bool): True if inference completed without errors
                - artifacts (list[str]): Absolute paths to generated files
                - metadata (dict): Execution details (model, device, source, conf)
                - error (str | None): Error message if success=False

        Raises:
            RuntimeError: If YOLO engine encounters unrecoverable errors

        Example:
            >>> handler = PredictHandler()
            >>> result = handler.execute(
            ...     job_id="test-001",
            ...     params={"model_path": "yolov8n.pt", "data_source": "bus.jpg", "device": "cpu"},
            ...     output_dir="runs/predict",
            ... )
            >>> print(result["success"])
            True
        """
        try:
            from ultralytics import YOLO

            # Create job-specific output directory
            job_output_dir = Path(output_dir) / job_id
            job_output_dir.mkdir(parents=True, exist_ok=True)

            # Load YOLO model
            model_path = params["model_path"]
            model = YOLO(model_path)

            # Extract prediction parameters
            data_source = params["data_source"]
            device = params.get("device", "cpu")
            conf = params.get("conf", 0.25)

            # Execute prediction with artifact saving
            results = model.predict(
                source=data_source,
                device=device,
                conf=conf,
                save=True,
                project=str(job_output_dir.parent.resolve()),
                name=job_id,
                exist_ok=True,
            )

            # Collect generated artifacts from actual YOLO save directory
            artifacts = []
            if results and len(results) > 0:
                actual_save_dir = Path(results[0].save_dir)
                if actual_save_dir.exists():
                    for artifact_path in actual_save_dir.rglob("*"):
                        if artifact_path.is_file():
                            artifacts.append(str(artifact_path.resolve()))

            # Build execution metadata
            metadata = {
                "model": model_path,
                "source": data_source,
                "device": device,
                "conf": conf,
                "num_results": len(results) if results else 0,
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
                "metadata": {"model": params.get("model_path"), "source": params.get("data_source")},
                "error": f"Prediction failed: {type(e).__name__}: {e}",
            }
