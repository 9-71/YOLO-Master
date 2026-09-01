"""Diagnose task handler for YOLO-Master F1 platform.

This module implements the DiagnoseHandler for collecting system environment diagnostics.
It gathers hardware and software information (Python, PyTorch, CUDA, GPU specs, Ultralytics)
and persists diagnostic reports in both JSON and human-readable text formats.

Security Model:
    - No external data access required (reads system state only)
    - Validates output_dir writability
    - Returns diagnostic reports as artifacts
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

from smoke.f1.handlers.base import BaseTaskHandler
from smoke.f1.handlers.registry import TaskHandlerRegistry


@TaskHandlerRegistry.register("diagnose")
class DiagnoseHandler(BaseTaskHandler):
    """Handler for system environment diagnostics collection.

    This handler collects comprehensive system information for debugging and
    environment verification: Python version, PyTorch version, CUDA availability,
    GPU specifications, memory status, and Ultralytics version.

    Example:
        >>> handler = DiagnoseHandler()
        >>> params = {}
        >>> constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["runs"]}
        >>> is_valid, err = handler.validate_params(params, constraints)
        >>> if is_valid:
        ...     result = handler.execute("diag-001", params, "runs/diagnose")
        ...     print(result["success"], len(result["artifacts"]))
        True 2
    """

    def validate_params(self, params: dict[str, Any], security_constraints: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate diagnose task parameters against security constraints.

        Validation Rules:
            1. No path parameters required (diagnose reads system state only)
            2. allow_shell: Must be False (inherited security constraint)
            3. Output directory writability will be checked during execute()

        Args:
            params: Diagnose task parameters (typically empty dict)
            security_constraints: Security policy containing:
                - allow_shell (bool): Must be False
                - path_whitelisted (bool): Ignored (no input paths)
                - allowed_paths (list[str]): Ignored (no input paths)

        Returns:
            tuple[bool, str | None]: (is_valid, error_message)
                - (True, None) if validation passes
                - (False, error_description) on validation failure

        Example:
            >>> handler = DiagnoseHandler()
            >>> params = {}
            >>> constraints = {"allow_shell": False}
            >>> is_valid, err = handler.validate_params(params, constraints)
            >>> print(is_valid)
            True
        """
        # Enforce security policy: shell execution prohibited
        if security_constraints.get("allow_shell", False):
            return False, "Shell execution is not allowed for diagnose tasks"

        # Diagnose task requires no input parameters
        return True, None

    def execute(self, job_id: str, params: dict[str, Any], output_dir: str) -> dict[str, Any]:
        """Collect system diagnostics and persist reports to output directory.

        This method performs the following steps:
            1. Gather system environment information (Python, PyTorch, CUDA, GPU)
            2. Create job-specific output directory (output_dir / job_id)
            3. Persist diagnostics as JSON (machine-readable) and TXT (human-readable)
            4. Return execution result with artifact paths

        Args:
            job_id: Unique job identifier for artifact isolation (e.g., "diag_20260901_001")
            params: Task parameters (empty dict for diagnose tasks)
            output_dir: Base directory for saving reports (e.g., "runs/diagnose")

        Returns:
            dict[str, Any]: Execution result with structure:
                - success (bool): True if diagnostics completed without errors
                - artifacts (list[str]): Absolute paths to generated reports
                - metadata (dict): Summary of collected diagnostics
                - error (str | None): Error message if success=False

        Diagnostic Information Collected:
            - System: OS, architecture, hostname
            - Python: Version, executable path
            - PyTorch: Version, CUDA support, device count
            - GPU: Model name, total/free memory (if CUDA available)
            - Ultralytics: Library version

        Example:
            >>> handler = DiagnoseHandler()
            >>> result = handler.execute("diag-001", {}, "runs/diagnose")
            >>> print(result["success"], "system_diagnostics.json" in result["artifacts"][0])
            True True
        """
        try:
            # Create job-specific output directory
            job_output_dir = Path(output_dir) / job_id
            job_output_dir.mkdir(parents=True, exist_ok=True)

            # Collect system diagnostics
            diagnostics = self._collect_diagnostics()

            # Persist diagnostics as JSON
            json_path = job_output_dir / "system_diagnostics.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(diagnostics, f, indent=2, ensure_ascii=False)

            # Persist diagnostics as formatted text
            txt_path = job_output_dir / "system_diagnostics.txt"
            with txt_path.open("w", encoding="utf-8") as f:
                f.write(self._format_diagnostics(diagnostics))

            # Return execution result
            artifacts = [str(json_path.resolve()), str(txt_path.resolve())]

            metadata = {
                "python_version": diagnostics["python"]["version"],
                "pytorch_version": diagnostics["pytorch"]["version"],
                "cuda_available": diagnostics["pytorch"]["cuda_available"],
                "gpu_count": diagnostics["gpu"]["device_count"],
            }

            return {
                "success": True,
                "artifacts": artifacts,
                "metadata": metadata,
                "error": None,
            }

        except (OSError, ImportError, RuntimeError) as e:
            return {
                "success": False,
                "artifacts": [],
                "metadata": {},
                "error": f"Diagnostics collection failed: {type(e).__name__}: {e}",
            }

    def _collect_diagnostics(self) -> dict[str, Any]:
        """Collect comprehensive system and environment diagnostics.

        Returns:
            dict[str, Any]: Diagnostic information with keys:
                - system: OS, architecture, hostname
                - python: Version, executable path
                - pytorch: Version, CUDA support, device count
                - gpu: Device names, memory stats (if CUDA available)
                - ultralytics: Library version
        """
        diagnostics: dict[str, Any] = {
            "system": {
                "os": platform.system(),
                "os_version": platform.version(),
                "architecture": platform.machine(),
                "hostname": platform.node(),
            },
            "python": {
                "version": sys.version,
                "executable": sys.executable,
            },
        }

        # Collect PyTorch diagnostics
        try:
            import torch

            diagnostics["pytorch"] = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }

            # Collect GPU diagnostics if CUDA is available
            if torch.cuda.is_available():
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    gpu_props = torch.cuda.get_device_properties(i)
                    total_memory_gb = gpu_props.total_memory / (1024**3)
                    free_memory_gb = (torch.cuda.mem_get_info(i)[0]) / (1024**3)

                    gpu_info.append(
                        {
                            "device_id": i,
                            "name": gpu_props.name,
                            "total_memory_gb": round(total_memory_gb, 2),
                            "free_memory_gb": round(free_memory_gb, 2),
                            "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                        }
                    )

                diagnostics["gpu"] = {
                    "device_count": len(gpu_info),
                    "devices": gpu_info,
                }
            else:
                diagnostics["gpu"] = {
                    "device_count": 0,
                    "devices": [],
                }

        except ImportError:
            diagnostics["pytorch"] = {"version": "not_installed", "cuda_available": False, "device_count": 0}
            diagnostics["gpu"] = {"device_count": 0, "devices": []}

        # Collect Ultralytics version
        try:
            import ultralytics

            diagnostics["ultralytics"] = {"version": ultralytics.__version__}
        except ImportError:
            diagnostics["ultralytics"] = {"version": "not_installed"}

        return diagnostics

    def _format_diagnostics(self, diagnostics: dict[str, Any]) -> str:
        """Format diagnostics dictionary as human-readable text report.

        Args:
            diagnostics: Diagnostic information from _collect_diagnostics()

        Returns:
            str: Formatted text report with sections for system, Python, PyTorch, GPU, Ultralytics
        """
        lines = ["=" * 80, "YOLO-Master F1 Platform - System Diagnostics Report", "=" * 80, ""]

        # System section
        lines.append("[SYSTEM]")
        lines.append(f"  OS: {diagnostics['system']['os']} {diagnostics['system']['os_version']}")
        lines.append(f"  Architecture: {diagnostics['system']['architecture']}")
        lines.append(f"  Hostname: {diagnostics['system']['hostname']}")
        lines.append("")

        # Python section
        lines.append("[PYTHON]")
        lines.append(f"  Version: {diagnostics['python']['version']}")
        lines.append(f"  Executable: {diagnostics['python']['executable']}")
        lines.append("")

        # PyTorch section
        lines.append("[PYTORCH]")
        pytorch = diagnostics.get("pytorch", {})
        lines.append(f"  Version: {pytorch.get('version', 'N/A')}")
        lines.append(f"  CUDA Available: {pytorch.get('cuda_available', False)}")
        if pytorch.get("cuda_available"):
            lines.append(f"  CUDA Version: {pytorch.get('cuda_version', 'N/A')}")
            lines.append(f"  Device Count: {pytorch.get('device_count', 0)}")
        lines.append("")

        # GPU section
        lines.append("[GPU]")
        gpu = diagnostics.get("gpu", {})
        if gpu.get("device_count", 0) > 0:
            for device in gpu.get("devices", []):
                lines.append(f"  Device {device['device_id']}: {device['name']}")
                lines.append(f"    Total Memory: {device['total_memory_gb']} GB")
                lines.append(f"    Free Memory: {device['free_memory_gb']} GB")
                lines.append(f"    Compute Capability: {device['compute_capability']}")
        else:
            lines.append("  No CUDA devices available")
        lines.append("")

        # Ultralytics section
        lines.append("[ULTRALYTICS]")
        ultralytics_info = diagnostics.get("ultralytics", {})
        lines.append(f"  Version: {ultralytics_info.get('version', 'N/A')}")
        lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)
