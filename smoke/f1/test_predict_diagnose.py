"""Unit tests for PredictHandler and DiagnoseHandler implementations.

This test suite validates:
1. Handler registration via @TaskHandlerRegistry.register()
2. Parameter validation (security constraints, path whitelisting, type checking)
3. Execution flow (artifact generation, error handling)
4. Security enforcement (path traversal prevention, shell execution blocking)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from smoke.f1.handlers import TaskHandlerRegistry
from smoke.f1.handlers.diagnose import DiagnoseHandler
from smoke.f1.handlers.predict import PredictHandler


class TestPredictHandlerRegistration:
    """Test suite for PredictHandler registration and instantiation."""

    def test_predict_handler_is_registered(self):
        """Verify that PredictHandler is registered under 'predict' task_type."""
        handler_class = TaskHandlerRegistry.get("predict")
        assert handler_class is PredictHandler

    def test_predict_handler_instantiation(self):
        """Verify that PredictHandler can be instantiated."""
        handler = PredictHandler()
        assert handler is not None
        assert hasattr(handler, "validate_params")
        assert hasattr(handler, "execute")


class TestPredictHandlerValidation:
    """Test suite for PredictHandler parameter validation."""

    def setup_method(self):
        """Initialize handler instance for each test."""
        self.handler = PredictHandler()

    def test_validation_requires_model_path(self):
        """Verify that validation fails when model_path is missing."""
        params = {"data_source": "bus.jpg"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "model_path" in err
        assert "missing" in err.lower()

    def test_validation_requires_data_source(self):
        """Verify that validation fails when data_source is missing."""
        params = {"model_path": "yolov8n.pt"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "data_source" in err
        assert "missing" in err.lower()

    def test_validation_rejects_path_outside_whitelist(self):
        """Verify that validation fails when paths are outside allowed_paths."""
        params = {"model_path": "ultralytics/assets/yolov8n.pt", "data_source": "../../etc/passwd"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["ultralytics/assets"]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "data_source" in err
        assert "whitelist" in err.lower()

    def test_validation_accepts_valid_paths(self):
        """Verify that validation passes for paths within allowed_paths."""
        params = {
            "model_path": "yolov8n.pt",
            "data_source": "ultralytics/assets/bus.jpg",
            "conf": 0.25,
            "device": "cpu",
        }
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": [".", "ultralytics"]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is True
        assert err is None

    def test_validation_rejects_invalid_conf_threshold(self):
        """Verify that validation fails for conf outside (0.0, 1.0] range."""
        params = {"model_path": "yolov8n.pt", "data_source": "bus.jpg", "conf": 1.5}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "conf" in err
        assert "range" in err.lower()

    def test_validation_rejects_zero_conf_threshold(self):
        """Verify that validation fails for conf=0.0 (must be > 0.0)."""
        params = {"model_path": "yolov8n.pt", "data_source": "bus.jpg", "conf": 0.0}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "conf" in err

    def test_validation_rejects_non_numeric_conf(self):
        """Verify that validation fails for non-numeric conf values."""
        params = {"model_path": "yolov8n.pt", "data_source": "bus.jpg", "conf": "invalid"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "conf" in err
        assert "float" in err.lower()

    def test_validation_rejects_shell_execution(self):
        """Verify that validation fails when allow_shell=True."""
        params = {"model_path": "yolov8n.pt", "data_source": "bus.jpg"}
        constraints = {"path_whitelisted": True, "allow_shell": True, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "shell" in err.lower()

    def test_validation_requires_path_whitelisting(self):
        """Verify that validation fails when path_whitelisted=False."""
        params = {"model_path": "yolov8n.pt", "data_source": "bus.jpg"}
        constraints = {"path_whitelisted": False, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "path whitelisting" in err.lower()

    def test_validation_rejects_empty_allowed_paths(self):
        """Verify that validation fails when allowed_paths is empty."""
        params = {"model_path": "yolov8n.pt", "data_source": "bus.jpg"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": []}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "allowed_paths" in err
        assert "empty" in err.lower()


class TestPredictHandlerExecution:
    """Test suite for PredictHandler execution logic."""

    def setup_method(self):
        """Initialize handler instance for each test."""
        self.handler = PredictHandler()

    def test_execute_returns_structured_result(self):
        """Verify that execute() returns a dict with required keys."""
        params = {"model_path": "yolov8n.pt", "data_source": "bus.jpg", "device": "cpu"}

        # Note: This test will fail if yolov8n.pt doesn't exist, but validates structure
        result = self.handler.execute("test-001", params, "runs/predict")

        assert isinstance(result, dict)
        assert "success" in result
        assert "artifacts" in result
        assert "metadata" in result
        assert "error" in result
        assert isinstance(result["artifacts"], list)
        assert isinstance(result["metadata"], dict)

    def test_execute_handles_missing_model_gracefully(self):
        """Verify that execute() returns success=False when model doesn't exist."""
        params = {"model_path": "nonexistent_model.pt", "data_source": "bus.jpg", "device": "cpu"}

        result = self.handler.execute("test-002", params, "runs/predict")

        assert result["success"] is False
        assert result["error"] is not None
        assert len(result["artifacts"]) == 0


class TestDiagnoseHandlerRegistration:
    """Test suite for DiagnoseHandler registration and instantiation."""

    def test_diagnose_handler_is_registered(self):
        """Verify that DiagnoseHandler is registered under 'diagnose' task_type."""
        handler_class = TaskHandlerRegistry.get("diagnose")
        assert handler_class is DiagnoseHandler

    def test_diagnose_handler_instantiation(self):
        """Verify that DiagnoseHandler can be instantiated."""
        handler = DiagnoseHandler()
        assert handler is not None
        assert hasattr(handler, "validate_params")
        assert hasattr(handler, "execute")


class TestDiagnoseHandlerValidation:
    """Test suite for DiagnoseHandler parameter validation."""

    def setup_method(self):
        """Initialize handler instance for each test."""
        self.handler = DiagnoseHandler()

    def test_validation_accepts_empty_params(self):
        """Verify that validation passes with empty params (no input required)."""
        params: dict[str, Any] = {}
        constraints = {"allow_shell": False}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is True
        assert err is None

    def test_validation_rejects_shell_execution(self):
        """Verify that validation fails when allow_shell=True."""
        params: dict[str, Any] = {}
        constraints = {"allow_shell": True}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "shell" in err.lower()

    def test_validation_ignores_path_whitelisting(self):
        """Verify that path_whitelisted constraint is not enforced (no input paths)."""
        params: dict[str, Any] = {}
        constraints = {"allow_shell": False, "path_whitelisted": False}

        is_valid, _ = self.handler.validate_params(params, constraints)
        assert is_valid is True  # DiagnoseHandler doesn't require path whitelisting


class TestDiagnoseHandlerExecution:
    """Test suite for DiagnoseHandler execution logic."""

    def setup_method(self):
        """Initialize handler instance for each test."""
        self.handler = DiagnoseHandler()

    def test_execute_returns_structured_result(self, tmp_path):
        """Verify that execute() returns a dict with required keys."""
        params: dict[str, Any] = {}
        output_dir = str(tmp_path / "diagnose")

        result = self.handler.execute("diag-001", params, output_dir)

        assert isinstance(result, dict)
        assert "success" in result
        assert "artifacts" in result
        assert "metadata" in result
        assert "error" in result
        assert result["success"] is True
        assert result["error"] is None

    def test_execute_generates_json_artifact(self, tmp_path):
        """Verify that execute() generates system_diagnostics.json."""
        params: dict[str, Any] = {}
        output_dir = str(tmp_path / "diagnose")

        result = self.handler.execute("diag-002", params, output_dir)

        assert result["success"] is True
        assert len(result["artifacts"]) == 2

        json_artifact = next((a for a in result["artifacts"] if a.endswith(".json")), None)
        assert json_artifact is not None
        assert Path(json_artifact).exists()

        # Validate JSON structure
        with Path(json_artifact).open("r", encoding="utf-8") as f:
            diagnostics = json.load(f)
            assert "system" in diagnostics
            assert "python" in diagnostics
            assert "pytorch" in diagnostics
            assert "gpu" in diagnostics
            assert "ultralytics" in diagnostics

    def test_execute_generates_txt_artifact(self, tmp_path):
        """Verify that execute() generates system_diagnostics.txt."""
        params: dict[str, Any] = {}
        output_dir = str(tmp_path / "diagnose")

        result = self.handler.execute("diag-003", params, output_dir)

        assert result["success"] is True
        txt_artifact = next((a for a in result["artifacts"] if a.endswith(".txt")), None)
        assert txt_artifact is not None
        assert Path(txt_artifact).exists()

        # Validate text content
        with Path(txt_artifact).open("r", encoding="utf-8") as f:
            content = f.read()
            assert "System Diagnostics Report" in content
            assert "[SYSTEM]" in content
            assert "[PYTHON]" in content
            assert "[PYTORCH]" in content
            assert "[GPU]" in content
            assert "[ULTRALYTICS]" in content

    def test_execute_metadata_contains_summary(self, tmp_path):
        """Verify that execute() returns metadata with diagnostic summary."""
        params: dict[str, Any] = {}
        output_dir = str(tmp_path / "diagnose")

        result = self.handler.execute("diag-004", params, output_dir)

        assert result["success"] is True
        metadata = result["metadata"]
        assert "python_version" in metadata
        assert "pytorch_version" in metadata
        assert "cuda_available" in metadata
        assert "gpu_count" in metadata

    def test_execute_isolates_output_by_job_id(self, tmp_path):
        """Verify that execute() creates job-specific output directories."""
        params: dict[str, Any] = {}
        output_dir = str(tmp_path / "diagnose")

        result1 = self.handler.execute("diag-005", params, output_dir)
        result2 = self.handler.execute("diag-006", params, output_dir)

        # Both jobs should succeed
        assert result1["success"] is True
        assert result2["success"] is True

        # Artifacts should be in separate directories
        job1_dir = Path(output_dir) / "diag-005"
        job2_dir = Path(output_dir) / "diag-006"
        assert job1_dir.exists()
        assert job2_dir.exists()
        assert len(list(job1_dir.iterdir())) == 2  # JSON + TXT
        assert len(list(job2_dir.iterdir())) == 2  # JSON + TXT


class TestHandlerRegistryIntegration:
    """End-to-end integration tests for both handlers."""

    def test_both_handlers_are_registered(self):
        """Verify that both predict and diagnose handlers are registered."""
        registered = TaskHandlerRegistry.list_registered()
        assert "predict" in registered
        assert "diagnose" in registered

    def test_dispatcher_can_retrieve_both_handlers(self):
        """Simulate dispatcher workflow retrieving and using both handlers."""
        # Retrieve handlers
        predict_class = TaskHandlerRegistry.get("predict")
        diagnose_class = TaskHandlerRegistry.get("diagnose")

        assert predict_class is PredictHandler
        assert diagnose_class is DiagnoseHandler

        # Instantiate handlers
        predict_handler = predict_class()
        diagnose_handler = diagnose_class()

        assert isinstance(predict_handler, PredictHandler)
        assert isinstance(diagnose_handler, DiagnoseHandler)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
