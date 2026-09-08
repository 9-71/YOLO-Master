"""Predict task handler for YOLO-Master F1 platform.

This module implements the PredictHandler for executing YOLO object detection inference tasks.
It validates input parameters against security constraints, delegates to the Ultralytics YOLO
engine for inference, and captures generated artifacts (annotated images, labels, metadata).

Batch Inference Support (Phase 1):
    data_source accepts a single file path, a directory of media files, or a sliced
    list of file paths. Sources are normalized into a deterministic file list, split
    into chunks of ``batch_size``, and processed one chunk per engine call so
    cooperative cancellation checkpoints run between chunks.

Security Model:
    - Enforces path whitelisting for model_path and every data_source entry
    - Validates confidence threshold bounds (0.0, 1.0] and batch_size bounds (> 0)
    - Isolates output artifacts per job_id to prevent collision
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from f1.handlers.base import BaseTaskHandler, PathWhitelistViolationError
from f1.handlers.registry import TaskHandlerRegistry

# Media file extensions recognized when expanding a directory data_source.
# Fail-closed policy: files with other extensions inside a directory are ignored.
SUPPORTED_MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".ts",
    }
)


@TaskHandlerRegistry.register("predict")
class PredictHandler(BaseTaskHandler):
    """Handler for YOLO object detection inference tasks.

    This handler executes real-time object detection on images/videos using pre-trained
    YOLO models. It validates security constraints, invokes the Ultralytics YOLO engine
    (per batch chunk for multi-source inputs), and persists annotated results to the
    output directory.

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
            2. data_source (required): Must be a path string or a list of path strings,
               each within the allowed_paths whitelist (batch inference support)
            3. conf (optional): Must be float in range (0.0, 1.0]
            4. batch_size (optional): Must be int > 0 (chunk size for batch inference)
            5. device (optional): No validation (passed directly to YOLO engine)
            6. allow_shell: Must be False (inherited security constraint)

        Args:
            params: Predict task parameters with keys:
                - model_path (str): Path to YOLO model weights (.pt file)
                - data_source (str | list[str]): Path to input image/video, a directory
                  of media files, or a sliced list of file paths
                - conf (float, optional): Confidence threshold, default 0.25
                - batch_size (int, optional): Inputs per engine call, default 8
                - device (str, optional): Device specification ("0", "cpu", "mps")
            security_constraints: Security policy containing:
                - path_whitelisted (bool): Must be True
                - allow_shell (bool): Must be False
                - allowed_paths (list[str]): Whitelist of allowed directory roots

        Returns:
            tuple[bool, str | None]: (is_valid, error_message)
                - (True, None) if all validations pass
                - (False, error_description) on first parameter validation failure

        Raises:
            PathWhitelistViolationError: If model_path or any data_source entry fails
                the path whitelist check (mapped to SEC_ERR_001 by the dispatcher).

        Example:
            >>> handler = PredictHandler()
            >>> params = {"model_path": "yolov8n.pt", "data_source": "../../etc/passwd"}
            >>> constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}
            >>> try:
            ...     handler.validate_params(params, constraints)
            ... except PathWhitelistViolationError as e:
            ...     print("blocked:", "data_source" in str(e))
            blocked: True
        """
        # Enforce security policy: shell execution prohibited
        if security_constraints.get("allow_shell", False):
            return False, "Shell execution is not allowed for predict tasks"

        # Enforce security policy: path whitelisting required
        if not security_constraints.get("path_whitelisted", False):
            return False, "Path whitelisting must be enabled"

        allowed_paths = security_constraints.get("allowed_paths", [])
        allowed_patterns = security_constraints.get("allowed_path_patterns", [])
        if not allowed_paths and not allowed_patterns:
            return False, "allowed_paths cannot be empty when path_whitelisted=True"

        # Validate required parameter: model_path (empty string = not provided;
        # reject it as a parameter problem before the path whitelist check, which
        # would otherwise misclassify it as a security violation)
        if not params.get("model_path"):
            return False, "Required parameter 'model_path' is missing"

        model_path = params["model_path"]
        if not self._is_path_safe(model_path, allowed_paths, allowed_patterns):
            raise PathWhitelistViolationError(f"model_path '{model_path}' is not within allowed_paths whitelist")

        # Validate required parameter: data_source (single path or batch file list)
        if "data_source" not in params:
            return False, "Required parameter 'data_source' is missing"

        data_source = params["data_source"]
        if isinstance(data_source, str):
            data_sources = [data_source]
        elif isinstance(data_source, (list, tuple)):
            if not data_source:
                return False, "data_source list cannot be empty"
            for idx, item in enumerate(data_source):
                if not isinstance(item, str):
                    return False, f"data_source[{idx}] must be a string, got {item!r}"
            data_sources = list(data_source)
        else:
            return False, f"data_source must be a string or list of strings, got {type(data_source).__name__}"

        # Every source entry must be contained in the whitelist (batch-safe)
        for source in data_sources:
            if not self._is_path_safe(source, allowed_paths, allowed_patterns):
                raise PathWhitelistViolationError(f"data_source '{source}' is not within allowed_paths whitelist")

        # Validate optional parameter: conf (confidence threshold)
        if "conf" in params:
            try:
                conf = float(params["conf"])
                if not (0.0 < conf <= 1.0):
                    return False, f"conf must be in range (0.0, 1.0], got {conf}"
            except (ValueError, TypeError) as e:
                return False, f"conf must be a valid float, got {params['conf']}: {e}"

        # Validate optional parameter: batch_size (chunk size for batch inference)
        if "batch_size" in params:
            try:
                batch_size = int(params["batch_size"])
                if batch_size <= 0:
                    return False, f"batch_size must be > 0, got {batch_size}"
            except (ValueError, TypeError) as e:
                return False, f"batch_size must be a valid integer, got {params['batch_size']}: {e}"

        return True, None

    def execute(self, job_id: str, params: dict[str, Any], output_dir: str) -> dict[str, Any]:
        """Execute YOLO object detection inference and capture artifacts.

        This method performs the following steps:
            1. Cooperative cancellation checkpoint (before engine work)
            2. Initialize YOLO model from params["model_path"]
            3. Normalize data_source into a deterministic list of input files
               (directory entries are expanded and sorted; file lists are preserved)
            4. Split sources into chunks of batch_size and run one engine call per
               chunk, each saved under output_dir / job_id / batch_NNN
            5. Cooperative cancellation checkpoint between chunks
            6. Collect all generated artifacts deterministically (sorted absolute paths)
            7. Return execution result with artifact paths and batch metadata

        Args:
            job_id: Unique job identifier for artifact isolation (e.g., "job_20260901_001")
            params: Validated parameters containing:
                - model_path (str): Path to YOLO model weights
                - data_source (str | list[str]): Input image/video path(s) or directory
                - device (str, optional): Device specification, default "cpu"
                - conf (float, optional): Confidence threshold, default 0.25
                - batch_size (int, optional): Inputs per engine call, default 8
            output_dir: Base directory for saving results (e.g., "runs/predict")

        Returns:
            dict[str, Any]: Execution result with structure:
                - success (bool): True if inference completed without errors
                - artifacts (list[str]): Sorted absolute paths to generated files
                - metadata (dict): Execution details (model, sources, num_inputs,
                    num_batches, batch_size, device, conf, num_results)
                - error (str | None): Error message if success=False

        Raises:
            CooperativeCancellationError: If cancellation is requested at a checkpoint

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

            # Cooperative cancellation checkpoint: abort before any engine work
            self._check_cancelled()

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
            batch_size = int(params.get("batch_size", 8))

            # Normalize input into a deterministic file list (batch inference support)
            sources = self._normalize_sources(data_source)
            if not sources:
                return {
                    "success": False,
                    "artifacts": [],
                    "metadata": {
                        "model": model_path,
                        "source": data_source,
                        "device": device,
                        "conf": conf,
                    },
                    "error": f"Prediction failed: no supported media files found in data_source '{data_source}'",
                }

            # Partition sources into batch chunks; each chunk is one engine call
            chunks = [sources[i : i + batch_size] for i in range(0, len(sources), batch_size)]

            # Execute prediction per chunk with per-chunk artifact isolation
            num_results = 0
            for chunk_index, chunk in enumerate(chunks):
                # Cooperative cancellation checkpoint between chunks
                self._check_cancelled()

                results = model.predict(
                    source=chunk,
                    device=device,
                    conf=conf,
                    save=True,
                    project=str(job_output_dir.parent.resolve()),
                    name=f"{job_id}/batch_{chunk_index:03d}",
                    exist_ok=True,
                )
                num_results += len(results) if results else 0

            # Cooperative cancellation checkpoint: after the final chunk
            self._check_cancelled()

            # Collect all generated artifacts deterministically (sorted absolute paths)
            artifacts = sorted(str(p.resolve()) for p in job_output_dir.rglob("*") if p.is_file())

            # Build execution metadata including batch structure
            metadata = {
                "model": model_path,
                "source": data_source,
                "sources": sources,
                "num_inputs": len(sources),
                "num_batches": len(chunks),
                "batch_size": batch_size,
                "device": device,
                "conf": conf,
                "num_results": num_results,
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

    def _normalize_sources(self, data_source: str | list[str]) -> list[str]:
        """Normalize data_source into a deterministic list of individual input paths.

        Expansion Rules:
            - A string file path becomes a single-entry list.
            - A string directory is expanded to the sorted list of media files
              (SUPPORTED_MEDIA_EXTENSIONS) directly inside it. Deterministic ordering
              makes artifact collection and batch partitioning reproducible.
            - A list/tuple of strings is expanded entry by entry, preserving order.

        Args:
            data_source: Raw data_source value (validated by validate_params).

        Returns:
            list[str]: Deterministic list of input paths. Directories with no supported
                media files contribute no entries.

        Example:
            >>> handler = PredictHandler()
            >>> handler._normalize_sources(["ultralytics/assets/bus.jpg", "ultralytics/assets/zidane.jpg"])
            ['ultralytics/assets/bus.jpg', 'ultralytics/assets/zidane.jpg']
        """
        entries = [data_source] if isinstance(data_source, str) else list(data_source)

        sources: list[str] = []
        for entry in entries:
            entry_path = Path(entry)
            if entry_path.is_dir():
                media = sorted(
                    str(p)
                    for p in entry_path.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS
                )
                sources.extend(media)
            else:
                sources.append(str(entry))

        return sources
