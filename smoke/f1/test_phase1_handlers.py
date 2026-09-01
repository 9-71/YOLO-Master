"""Unit tests for phase-1 handlers (TrainHandler and ExportHandler).

This test suite validates:
1. Handler registration via @TaskHandlerRegistry.register()
2. Parameter validation (security constraints, path whitelisting, type checking)
3. Execution flow (artifact generation, error handling, job isolation)
4. Security enforcement (path traversal prevention, shell execution blocking)
5. Registry integration across all four task types

Train execution is verified with a mocked Ultralytics engine (real training is out of
scope for unit tests); export additionally runs one real ONNX export against the smoke
checkpoint to prove the E2E artifact pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from smoke.f1.handlers import TaskHandlerRegistry
from smoke.f1.handlers.export import SUPPORTED_EXPORT_FORMATS, ExportHandler
from smoke.f1.handlers.train import TrainHandler

# Smoke checkpoint shipped with the F1 entry smoke test (smoke/f1/yolov8n.pt)
SMOKE_MODEL = Path(__file__).parent / "yolov8n.pt"


class TestPhase1HandlerRegistration:
    """Test suite for TrainHandler and ExportHandler registration."""

    def test_train_handler_is_registered(self):
        """Verify that TrainHandler is registered under 'train' task_type."""
        handler_class = TaskHandlerRegistry.get("train")
        assert handler_class is TrainHandler

    def test_export_handler_is_registered(self):
        """Verify that ExportHandler is registered under 'export' task_type."""
        handler_class = TaskHandlerRegistry.get("export")
        assert handler_class is ExportHandler

    def test_train_handler_instantiation(self):
        """Verify that TrainHandler can be instantiated."""
        handler = TrainHandler()
        assert handler is not None
        assert hasattr(handler, "validate_params")
        assert hasattr(handler, "execute")

    def test_export_handler_instantiation(self):
        """Verify that ExportHandler can be instantiated."""
        handler = ExportHandler()
        assert handler is not None
        assert hasattr(handler, "validate_params")
        assert hasattr(handler, "execute")


class TestTrainHandlerValidation:
    """Test suite for TrainHandler parameter validation."""

    def setup_method(self):
        """Initialize handler instance for each test."""
        self.handler = TrainHandler()

    def test_validation_requires_model_path(self):
        """Verify that validation fails when model_path is missing."""
        params = {"data_source": "coco8.yaml", "epochs": 10}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "model_path" in err
        assert "missing" in err.lower()

    def test_validation_requires_data_source(self):
        """Verify that validation fails when data_source is missing."""
        params = {"model_path": "yolov8n.pt", "epochs": 10}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "data_source" in err
        assert "missing" in err.lower()

    def test_validation_requires_epochs(self):
        """Verify that validation fails when epochs is missing."""
        params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "epochs" in err
        assert "missing" in err.lower()

    def test_validation_rejects_non_positive_epochs(self):
        """Verify that validation fails for epochs <= 0."""
        params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml", "epochs": 0}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "epochs" in err

    def test_validation_rejects_non_numeric_epochs(self):
        """Verify that validation fails for non-numeric epochs."""
        params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml", "epochs": "ten"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "epochs" in err
        assert "integer" in err.lower()

    def test_validation_rejects_non_positive_batch_size(self):
        """Verify that validation fails for batch_size <= 0."""
        params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml", "epochs": 10, "batch_size": -1}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "batch_size" in err

    def test_validation_rejects_negative_seed(self):
        """Verify that validation fails for seed < 0."""
        params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml", "epochs": 10, "seed": -1}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "seed" in err

    def test_validation_rejects_path_outside_whitelist(self):
        """Verify that validation fails when model_path is outside allowed_paths."""
        params = {"model_path": "../../etc/passwd", "data_source": "coco8.yaml", "epochs": 10}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["ultralytics/assets"]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "model_path" in err
        assert "whitelist" in err.lower()

    def test_validation_rejects_shell_execution(self):
        """Verify that validation fails when allow_shell=True."""
        params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml", "epochs": 10}
        constraints = {"path_whitelisted": True, "allow_shell": True, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "shell" in err.lower()

    def test_validation_requires_path_whitelisting(self):
        """Verify that validation fails when path_whitelisted=False."""
        params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml", "epochs": 10}
        constraints = {"path_whitelisted": False, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "path whitelisting" in err.lower()

    def test_validation_rejects_empty_allowed_paths(self):
        """Verify that validation fails when allowed_paths is empty."""
        params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml", "epochs": 10}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": []}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "allowed_paths" in err
        assert "empty" in err.lower()

    def test_validation_accepts_valid_params_with_seed(self):
        """Verify that validation passes for a complete valid parameter set."""
        params = {
            "model_path": "yolov8n.pt",
            "data_source": "coco8.yaml",
            "epochs": 10,
            "batch_size": 16,
            "device": "cpu",
            "seed": 42,
        }
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": [".", "runs"]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is True
        assert err is None


class TestTrainHandlerExecution:
    """Test suite for TrainHandler execution logic."""

    def setup_method(self):
        """Initialize handler instance for each test."""
        self.handler = TrainHandler()

    def test_execute_handles_missing_model_gracefully(self):
        """Verify that execute() returns success=False when model doesn't exist."""
        params = {"model_path": "nonexistent_model.pt", "data_source": "coco8.yaml", "epochs": 1}

        result = self.handler.execute("train-001", params, "runs/train")

        assert result["success"] is False
        assert result["error"] is not None
        assert len(result["artifacts"]) == 0

    def test_execute_happy_path_with_mocked_engine(self, tmp_path, monkeypatch):
        """Verify execute() flow with a mocked YOLO engine (no real training).

        The fake engine writes best.pt/last.pt/results.csv into the job directory,
        exercising artifact collection, seed determinism injection, and metadata.
        """

        class FakeTrainModel:
            def __init__(self, path):
                self.path = path
                self.trainer = None

            def train(self, **kwargs):
                job_dir = Path(kwargs["project"]) / kwargs["name"]
                weights_dir = job_dir / "weights"
                weights_dir.mkdir(parents=True, exist_ok=True)
                (weights_dir / "best.pt").write_bytes(b"fake-best")
                (weights_dir / "last.pt").write_bytes(b"fake-last")
                (job_dir / "results.csv").write_text("epoch,loss\n1,0.1\n")

        monkeypatch.setattr("ultralytics.YOLO", FakeTrainModel)

        params = {
            "model_path": str(SMOKE_MODEL),
            "data_source": "coco8.yaml",
            "epochs": 1,
            "batch_size": 16,
            "device": "cpu",
            "seed": 42,
        }
        result = self.handler.execute("train-001", params, str(tmp_path))

        assert result["success"] is True
        assert result["error"] is None
        assert len(result["artifacts"]) == 3  # best.pt + last.pt + results.csv
        assert any(p.endswith("best.pt") for p in result["artifacts"])
        assert any(p.endswith("last.pt") for p in result["artifacts"])

        # Discussion #244 Defect B: seed determinism must be recorded in metadata
        metadata = result["metadata"]
        assert metadata["seed"] == 42
        assert metadata["defect_mitigations"]["seed_determinism_applied"] is True

    def test_inject_determinism_returns_applied_seed(self):
        """Verify _inject_determinism locks torch/numpy state and returns the seed."""
        applied = self.handler._inject_determinism(42)
        assert applied == 42


class TestExportHandlerValidation:
    """Test suite for ExportHandler parameter validation."""

    def setup_method(self):
        """Initialize handler instance for each test."""
        self.handler = ExportHandler()

    def test_validation_requires_model_path(self):
        """Verify that validation fails when model_path is missing."""
        params: dict[str, Any] = {"format": "onnx"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "model_path" in err
        assert "missing" in err.lower()

    def test_validation_rejects_path_outside_whitelist(self):
        """Verify that validation fails when model_path is outside allowed_paths."""
        params = {"model_path": "../../etc/passwd", "format": "onnx"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["ultralytics/assets"]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "model_path" in err
        assert "whitelist" in err.lower()

    def test_validation_rejects_unsupported_format(self):
        """Verify that validation fails for formats outside the closed allowlist."""
        params = {"model_path": "yolov8n.pt", "format": "exe"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "format" in err
        assert "not supported" in err.lower()

    def test_validation_accepts_format_case_insensitively(self):
        """Verify that format validation is case-insensitive."""
        params = {"model_path": "yolov8n.pt", "format": "ONNX"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is True
        assert err is None

    def test_validation_rejects_non_positive_imgsz(self):
        """Verify that validation fails for imgsz <= 0."""
        params = {"model_path": "yolov8n.pt", "format": "onnx", "imgsz": 0}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "imgsz" in err

    def test_validation_rejects_non_numeric_imgsz(self):
        """Verify that validation fails for non-numeric imgsz."""
        params = {"model_path": "yolov8n.pt", "format": "onnx", "imgsz": "large"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "imgsz" in err
        assert "integer" in err.lower()

    def test_validation_rejects_non_boolean_half(self):
        """Verify that validation fails for non-bool half flag."""
        params = {"model_path": "yolov8n.pt", "format": "onnx", "half": "yes"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "half" in err
        assert "boolean" in err.lower()

    def test_validation_rejects_shell_execution(self):
        """Verify that validation fails when allow_shell=True."""
        params = {"model_path": "yolov8n.pt", "format": "onnx"}
        constraints = {"path_whitelisted": True, "allow_shell": True, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "shell" in err.lower()

    def test_validation_requires_path_whitelisting(self):
        """Verify that validation fails when path_whitelisted=False."""
        params = {"model_path": "yolov8n.pt", "format": "onnx"}
        constraints = {"path_whitelisted": False, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "path whitelisting" in err.lower()

    def test_validation_rejects_empty_allowed_paths(self):
        """Verify that validation fails when allowed_paths is empty."""
        params = {"model_path": "yolov8n.pt", "format": "onnx"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": []}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "allowed_paths" in err
        assert "empty" in err.lower()

    def test_validation_accepts_valid_params(self):
        """Verify that validation passes for a complete valid parameter set."""
        params = {
            "model_path": "yolov8n.pt",
            "format": "onnx",
            "imgsz": 320,
            "device": "cpu",
            "half": False,
            "int8": False,
        }
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": [".", "runs"]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is True
        assert err is None

    def test_all_allowlisted_formats_pass_validation(self):
        """Verify every entry in the closed allowlist passes format validation."""
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}
        for fmt in sorted(SUPPORTED_EXPORT_FORMATS):
            is_valid, err = self.handler.validate_params({"model_path": "yolov8n.pt", "format": fmt}, constraints)
            assert is_valid is True, f"format '{fmt}' should be allowed: {err}"


class TestExportHandlerExecution:
    """Test suite for ExportHandler execution logic."""

    def setup_method(self):
        """Initialize handler instance for each test."""
        self.handler = ExportHandler()

    def test_execute_handles_missing_model_gracefully(self):
        """Verify that execute() returns success=False when model doesn't exist."""
        params = {"model_path": "nonexistent_model.pt", "format": "onnx", "device": "cpu"}

        result = self.handler.execute("export-001", params, "runs/export")

        assert result["success"] is False
        assert result["error"] is not None
        assert len(result["artifacts"]) == 0

    def test_execute_happy_path_with_mocked_engine(self, tmp_path, monkeypatch):
        """Verify execute() flow with a mocked YOLO export engine.

        The fake engine writes the exported artifact next to the job-local model
        copy, exercising the copy-isolation and artifact collection pipeline.
        """

        class FakeExportModel:
            def __init__(self, path):
                self.path = Path(path)

            def export(self, **kwargs):
                artifact = self.path.with_suffix(f".{kwargs['format']}")
                artifact.write_bytes(b"fake-export")
                return str(artifact)

        monkeypatch.setattr("ultralytics.YOLO", FakeExportModel)

        params = {"model_path": str(SMOKE_MODEL), "format": "onnx", "imgsz": 320, "device": "cpu"}
        result = self.handler.execute("export-001", params, str(tmp_path))

        assert result["success"] is True
        assert result["error"] is None
        assert len(result["artifacts"]) == 1
        assert result["artifacts"][0].endswith(".onnx")
        assert Path(result["artifacts"][0]).exists()

        metadata = result["metadata"]
        assert metadata["format"] == "onnx"
        assert metadata["imgsz"] == 320
        assert metadata["exported_path"].endswith(".onnx")

    def test_execute_isolates_output_by_job_id(self, tmp_path, monkeypatch):
        """Verify that execute() writes artifacts into job-specific directories."""

        class FakeExportModel:
            def __init__(self, path):
                self.path = Path(path)

            def export(self, **kwargs):
                artifact = self.path.with_suffix(f".{kwargs['format']}")
                artifact.write_bytes(b"fake-export")
                return str(artifact)

        monkeypatch.setattr("ultralytics.YOLO", FakeExportModel)

        params = {"model_path": str(SMOKE_MODEL), "format": "torchscript", "device": "cpu"}
        result1 = self.handler.execute("export-001", params, str(tmp_path))
        result2 = self.handler.execute("export-002", params, str(tmp_path))

        assert result1["success"] is True
        assert result2["success"] is True

        job1_dir = Path(tmp_path) / "export-001"
        job2_dir = Path(tmp_path) / "export-002"
        assert job1_dir.exists()
        assert job2_dir.exists()

        # Each job dir contains the copied input weights + its own export artifact
        assert (job1_dir / "yolov8n.torchscript").exists()
        assert (job2_dir / "yolov8n.torchscript").exists()
        assert result1["artifacts"][0] != result2["artifacts"][0]

    def test_execute_real_onnx_export(self, tmp_path):
        """Verify a real ONNX export E2E against the smoke checkpoint.

        This is the only test in the suite that runs the actual Ultralytics export
        engine (CPU, reduced imgsz to keep runtime bounded).
        """
        params = {
            "model_path": str(SMOKE_MODEL),
            "format": "onnx",
            "imgsz": 320,
            "device": "cpu",
        }
        result = self.handler.execute("export-001", params, str(tmp_path))

        assert result["success"] is True
        assert result["error"] is None
        assert len(result["artifacts"]) >= 1

        onnx_artifact = next(a for a in result["artifacts"] if a.endswith(".onnx"))
        assert Path(onnx_artifact).exists()
        assert Path(onnx_artifact).stat().st_size > 0
        assert result["metadata"]["format"] == "onnx"
        assert result["metadata"]["exported_path"] is not None


class TestRegistryIntegration:
    """End-to-end integration tests across all four registered handlers."""

    def test_all_four_handlers_are_registered(self):
        """Verify predict, train, export, and diagnose are all registered."""
        registered = TaskHandlerRegistry.list_registered()
        assert registered == ["diagnose", "export", "predict", "train"]  # Alphabetically sorted

    def test_dispatcher_can_retrieve_phase1_handlers(self):
        """Simulate dispatcher workflow retrieving and using phase-1 handlers."""
        train_class = TaskHandlerRegistry.get("train")
        export_class = TaskHandlerRegistry.get("export")

        assert train_class is TrainHandler
        assert export_class is ExportHandler

        train_handler = train_class()
        export_handler = export_class()

        assert isinstance(train_handler, TrainHandler)
        assert isinstance(export_handler, ExportHandler)

    def test_phase1_handlers_pass_security_validation(self):
        """Verify both phase-1 handlers enforce the shared security contract."""
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}
        insecure = {"path_whitelisted": True, "allow_shell": True, "allowed_paths": ["."]}

        train_handler = TrainHandler()
        export_handler = ExportHandler()

        valid_params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml", "epochs": 10}
        is_valid, _ = train_handler.validate_params(valid_params, constraints)
        assert is_valid is True
        is_valid, err = train_handler.validate_params(valid_params, insecure)
        assert is_valid is False
        assert "shell" in err.lower()

        valid_params = {"model_path": "yolov8n.pt", "format": "onnx"}
        is_valid, _ = export_handler.validate_params(valid_params, constraints)
        assert is_valid is True
        is_valid, err = export_handler.validate_params(valid_params, insecure)
        assert is_valid is False
        assert "shell" in err.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
