"""Validation task handler for YOLO-Master F1 platform.

This module implements the ValHandler for executing YOLO model validation tasks.
It validates input parameters against security constraints, delegates to the Ultralytics
YOLO validation engine, parses detection metrics (mAP50, mAP50-95, mAP75, precision,
recall), and captures generated artifacts (results.csv, confusion matrix and curve
plots) under the job-specific output directory.

Security Model:
    - Enforces path whitelisting for model_path (checkpoint) and data_source (data.yaml)
    - Validates imgsz (int > 0), batch_size (int > 0), conf ((0.0, 1.0])
    - Isolates output artifacts per job_id to prevent collision
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smoke.f1.handlers.base import BaseTaskHandler, CooperativeCancellationError, PathWhitelistViolationError
from smoke.f1.handlers.registry import TaskHandlerRegistry


@TaskHandlerRegistry.register("val")
class ValHandler(BaseTaskHandler):
    """Handler for YOLO model validation tasks.

    This handler executes model validation (mAP evaluation) on a dataset described by
    a data.yaml configuration. It validates security constraints, invokes the
    Ultralytics YOLO val() engine with plotting enabled, parses the returned metrics,
    and collects the generated reports into the job-specific output directory.

    Example:
        >>> handler = ValHandler()
        >>> params = {
        ...     "model_path": "yolov8n.pt",
        ...     "data_source": "coco8.yaml",
        ...     "device": "cpu",
        ...     "imgsz": 320,
        ... }
        >>> constraints = {
        ...     "path_whitelisted": True,
        ...     "allow_shell": False,
        ...     "allowed_paths": [".", "runs"],
        ... }
        >>> is_valid, err = handler.validate_params(params, constraints)
        >>> if is_valid:
        ...     result = handler.execute("val-001", params, "runs/val")
        ...     print(result["success"], "mAP50" in result["metadata"]["metrics"])
        True True
    """

    def validate_params(self, params: dict[str, Any], security_constraints: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate val task parameters against security constraints.

        Validation Rules:
            1. model_path (required): Must be within allowed_paths whitelist (checkpoint)
            2. data_source (required): Must be a dataset YAML (.yaml/.yml) within allowed_paths whitelist
            3. imgsz (optional): If present, must be int > 0
            4. batch_size (optional): If present, must be int > 0
            5. conf (optional): If present, must be float in range (0.0, 1.0]
            6. device (optional): No validation (passed directly to YOLO engine)
            7. allow_shell: Must be False (inherited security constraint)

        Args:
            params: Val task parameters with keys:
                - model_path (str): Path to YOLO model checkpoint (.pt file)
                - data_source (str): Path to dataset configuration file (data.yaml)
                - imgsz (int, optional): Validation input image size, default 640
                - batch_size (int, optional): Validation batch size, default 16
                - conf (float, optional): Confidence threshold, default 0.001
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
            >>> handler = ValHandler()
            >>> params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml"}
            >>> constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}
            >>> is_valid, err = handler.validate_params(params, constraints)
            >>> print(is_valid)
            True
        """
        # Enforce security policy: shell execution prohibited
        if security_constraints.get("allow_shell", False):
            return False, "Shell execution is not allowed for val tasks"

        # Enforce security policy: path whitelisting required
        if not security_constraints.get("path_whitelisted", False):
            return False, "Path whitelisting must be enabled"

        allowed_paths = security_constraints.get("allowed_paths", [])
        allowed_patterns = security_constraints.get("allowed_path_patterns", [])
        if not allowed_paths and not allowed_patterns:
            return False, "allowed_paths cannot be empty when path_whitelisted=True"

        # Validate required parameter: model_path (checkpoint, empty string = not provided)
        if not params.get("model_path"):
            return False, "Required parameter 'model_path' is missing"

        model_path = params["model_path"]
        if not self._is_path_safe(model_path, allowed_paths, allowed_patterns):
            raise PathWhitelistViolationError(f"model_path '{model_path}' is not within allowed_paths whitelist")

        # Validate required parameter: data_source (data.yaml)
        if "data_source" not in params:
            return False, "Required parameter 'data_source' is missing"

        data_source = params["data_source"]
        if not self._is_path_safe(data_source, allowed_paths, allowed_patterns):
            raise PathWhitelistViolationError(f"data_source '{data_source}' is not within allowed_paths whitelist")

        # val requires a dataset configuration YAML: a non-YAML source (e.g. an
        # image) hangs the engine in dataset/weight resolution, so reject it up
        # front during validation instead of leaving the job stuck in RUNNING.
        if not data_source or not str(data_source).lower().endswith((".yaml", ".yml")):
            return False, "data_source for val must be a dataset YAML file (e.g., coco8.yaml)"

        # Validate optional parameter: imgsz
        if "imgsz" in params:
            try:
                imgsz = int(params["imgsz"])
                if imgsz <= 0:
                    return False, f"imgsz must be > 0, got {imgsz}"
            except (ValueError, TypeError) as e:
                return False, f"imgsz must be a valid integer, got {params['imgsz']}: {e}"

        # Validate optional parameter: batch_size
        if "batch_size" in params:
            try:
                batch_size = int(params["batch_size"])
                if batch_size <= 0:
                    return False, f"batch_size must be > 0, got {batch_size}"
            except (ValueError, TypeError) as e:
                return False, f"batch_size must be a valid integer, got {params['batch_size']}: {e}"

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
        """Execute YOLO model validation and capture artifacts.

        This method performs the following steps:
            1. Cooperative cancellation checkpoint (before engine work)
            2. Initialize YOLO model from params["model_path"]
            3. Configure job-specific output directory (output_dir / job_id)
            4. Invoke YOLO.val() with plots enabled (confusion matrix, curve plots)
            5. Parse validation metrics from the returned results object
            6. Collect generated artifacts deterministically (sorted absolute paths)
            7. Return execution result with artifacts and metrics

        Args:
            job_id: Unique job identifier for artifact isolation (e.g., "val_20260901_001")
            params: Validated parameters containing:
                - model_path (str): Path to YOLO model checkpoint
                - data_source (str): Path to dataset configuration file (data.yaml)
                - imgsz (int, optional): Validation input image size, default 640
                - batch_size (int, optional): Validation batch size, default 16
                - conf (float, optional): Confidence threshold, default 0.001
                - device (str, optional): Device specification, default "cpu"
            output_dir: Base directory for saving results (e.g., "runs/val")

        Returns:
            dict[str, Any]: Execution result with structure:
                - success (bool): True if validation completed without errors
                - artifacts (list[str]): Sorted absolute paths to generated files
                    (results.csv, confusion_matrix.png, F1/PR curve plots)
                - metadata (dict): Execution details including parsed "metrics"
                    (mAP50, mAP50-95, mAP75, precision, recall, speed_ms)
                - error (str | None): Error message if success=False

        Raises:
            CooperativeCancellationError: If cancellation is requested at a checkpoint

        Example:
            >>> handler = ValHandler()
            >>> result = handler.execute(
            ...     job_id="val-001",
            ...     params={"model_path": "yolov8n.pt", "data_source": "coco8.yaml", "device": "cpu"},
            ...     output_dir="runs/val",
            ... )
            >>> print(result["success"])
            True
        """
        try:
            from ultralytics import YOLO

            # Cooperative cancellation checkpoint: abort before any engine work
            self._check_cancelled()

            # Create job-specific output directory
            job_output_dir = Path(output_dir) / job_id
            job_output_dir.mkdir(parents=True, exist_ok=True)

            # Load YOLO model from the checkpoint
            model_path = params["model_path"]
            model = YOLO(model_path)

            # Extract validation parameters
            data_source = params["data_source"]
            device = params.get("device", "cpu")

            # Execute validation with plotting enabled so the engine generates
            # results.csv, confusion_matrix.png, and F1/PR curve plots
            val_kwargs: dict[str, Any] = {
                "data": data_source,
                "device": device,
                "project": str(job_output_dir.parent.resolve()),
                "name": job_id,
                "exist_ok": True,
                "plots": True,
            }
            if "imgsz" in params:
                val_kwargs["imgsz"] = int(params["imgsz"])
            if "batch_size" in params:
                val_kwargs["batch"] = int(params["batch_size"])
            if "conf" in params:
                val_kwargs["conf"] = float(params["conf"])

            results = model.val(**val_kwargs)

            # Cooperative cancellation checkpoint: after the engine call
            self._check_cancelled()

            # Collect generated artifacts deterministically (sorted absolute paths)
            artifacts = sorted(str(p.resolve()) for p in job_output_dir.rglob("*") if p.is_file())

            # Build execution metadata with parsed validation metrics
            metadata = {
                "model": model_path,
                "data": data_source,
                "device": device,
                "metrics": self._parse_val_metrics(results),
            }

            return {
                "success": True,
                "artifacts": artifacts,
                "metadata": metadata,
                "error": None,
            }

        except CooperativeCancellationError:
            # Cancellation is a control-flow signal, not an engine failure; re-raise
            # so the dispatcher can map it to FAILED + USER_CANCELLED.
            raise
        except Exception as e:  # noqa: BLE001 - engine errors of any type must become structured failures
            return {
                "success": False,
                "artifacts": [],
                "metadata": {
                    "model": params.get("model_path"),
                    "data": params.get("data_source"),
                },
                "error": f"Validation failed: {type(e).__name__}: {e}",
            }

    def _parse_val_metrics(self, results: Any) -> dict[str, Any]:
        """Parse validation metrics from the Ultralytics val() results object.

        Extraction is defensive: it reads the ``box`` metrics namespace when
        available (mAP50, mAP75, mAP50-95, per-class precision/recall) and falls
        back to ``results_dict`` for older engine versions. Torch tensors are
        converted to plain float lists so the metadata stays JSON-serializable.

        Args:
            results: The results object returned by YOLO.val() (or None).

        Returns:
            dict[str, Any]: Metrics dictionary with keys such as:
                - mAP50 (float): Mean Average Precision at IoU=0.5
                - mAP50-95 (float): Mean Average Precision at IoU=0.5:0.95
                - mAP75 (float): Mean Average Precision at IoU=0.75
                - precision (list[float]): Per-class precision values
                - recall (list[float]): Per-class recall values
                - speed_ms (dict): Engine timing breakdown, if reported

        Example:
            >>> handler = ValHandler()
            >>> from types import SimpleNamespace
            >>> box = SimpleNamespace(map50=0.62, map75=0.41, map=0.45, p=[0.9], r=[0.8])
            >>> metrics = handler._parse_val_metrics(SimpleNamespace(box=box))
            >>> metrics["mAP50"], metrics["mAP50-95"]
            (0.62, 0.45)
        """
        metrics: dict[str, Any] = {}
        if results is None:
            return metrics

        # Primary source: results.box metrics namespace (Ultralytics DetMetrics)
        box = getattr(results, "box", None)
        if box is not None:
            for attr, key in (("map50", "mAP50"), ("map75", "mAP75"), ("map", "mAP50-95")):
                value = getattr(box, attr, None)
                if isinstance(value, (int, float)):
                    metrics[key] = round(float(value), 5)

            precision = self._as_serializable(getattr(box, "p", None))
            recall = self._as_serializable(getattr(box, "r", None))
            if precision is not None:
                metrics["precision"] = precision
            if recall is not None:
                metrics["recall"] = recall

        # Fallback source: results.results_dict (older engine versions)
        if not metrics:
            results_dict = getattr(results, "results_dict", None)
            if isinstance(results_dict, dict):
                for raw_key, key in (
                    ("metrics/mAP50(B)", "mAP50"),
                    ("metrics/mAP50-95(B)", "mAP50-95"),
                    ("metrics/mAP75(B)", "mAP75"),
                ):
                    value = results_dict.get(raw_key)
                    if isinstance(value, (int, float)):
                        metrics[key] = round(float(value), 5)

        # Timing breakdown, when reported by the engine
        speed = getattr(results, "speed", None)
        if isinstance(speed, dict):
            metrics["speed_ms"] = {
                name: round(float(value), 3) for name, value in speed.items() if isinstance(value, (int, float))
            }

        return metrics

    @staticmethod
    def _as_serializable(value: Any) -> Any:
        """Convert engine metric values (tensors/arrays/lists/scalars) to JSON-safe types.

        Args:
            value: Metric value from the engine (torch.Tensor, list, or scalar).

        Returns:
            Any: Rounded float/list of floats, or None if the value is not numeric.

        Example:
            >>> ValHandler._as_serializable([0.123456, 0.654321])
            [0.12346, 0.65432]
        """
        if value is None:
            return None
        # torch.Tensor and numpy arrays expose tolist(); convert element-wise
        if hasattr(value, "tolist"):
            try:
                return [round(float(x), 5) for x in value.tolist()]
            except (TypeError, ValueError):
                return None
        if isinstance(value, (list, tuple)):
            return [round(float(x), 5) for x in value]
        if isinstance(value, (int, float)):
            return round(float(value), 5)
        return None
