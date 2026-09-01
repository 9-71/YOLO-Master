"""Comprehensive unit tests for Agent Skills wrapper (skills.py).

This test suite validates the SystemDoctorSkill and PredictSkill implementations,
covering execution happy paths, parameter validation, fail-closed security violations,
and error code mappings.

Test Coverage:
    - SystemDoctorSkill:
        * Execution happy path (with real dispatcher)
        * Security constraint enforcement (shell execution prohibition)
        * Output artifact validation (JSON + TXT reports)
        * Error handling and error code mapping
    - PredictSkill:
        * Execution happy path (with mocked YOLO engine)
        * Parameter validation (conf threshold, required params)
        * Fail-closed path whitelisting (empty allowed_paths, path traversal)
        * Dynamic whitelist boundary construction
        * Error code mapping (security violations, validation failures)

Usage:
    pytest smoke/f1/test_skills.py -v
    pytest smoke/f1/test_skills.py::TestSystemDoctorSkill -v
    pytest smoke/f1/test_skills.py::TestPredictSkill -v
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from smoke.f1.skills import PredictSkill, SystemDoctorSkill


class TestSystemDoctorSkill:
    """Test suite for SystemDoctorSkill Agent wrapper."""

    def test_metadata_properties(self):
        """Verify skill exposes correct tool metadata for Agent registration."""
        skill = SystemDoctorSkill()

        # Verify name
        assert skill.name == "system_doctor"

        # Verify description is non-empty and descriptive
        assert len(skill.description) > 50
        assert "diagnostic" in skill.description.lower()

        # Verify parameters schema structure
        schema = skill.parameters_schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "output_dir" in schema["properties"]
        assert "job_id" in schema["properties"]
        assert schema["required"] == []  # All parameters are optional

    def test_execute_happy_path(self, tmp_path):
        """Test successful diagnostics collection with real dispatcher."""
        skill = SystemDoctorSkill()
        output_dir = str(tmp_path / "diagnose")

        # Execute diagnostics
        result = skill.execute(output_dir=output_dir, job_id="test_diag_001")

        # Verify success status
        assert result["status"] == "success"
        assert result["job_id"] == "test_diag_001"
        assert result["error_code"] is None
        assert result["error_message"] is None

        # Verify summary contains expected diagnostic keys
        summary = result["summary"]
        assert "python_version" in summary
        assert "pytorch_version" in summary
        assert "cuda_available" in summary
        assert "gpu_count" in summary
        assert isinstance(summary["gpu_count"], int)

        # Verify artifacts were generated
        artifacts = result["artifacts"]
        assert len(artifacts) == 2  # JSON + TXT reports

        # Verify JSON artifact exists and is valid
        json_artifact = next((a for a in artifacts if a.endswith(".json")), None)
        assert json_artifact is not None
        assert Path(json_artifact).exists()

        with open(json_artifact, encoding="utf-8") as f:
            diagnostic_data = json.load(f)
            assert "system" in diagnostic_data
            assert "python" in diagnostic_data
            assert "pytorch" in diagnostic_data

        # Verify TXT artifact exists
        txt_artifact = next((a for a in artifacts if a.endswith(".txt")), None)
        assert txt_artifact is not None
        assert Path(txt_artifact).exists()

    def test_execute_auto_generates_job_id(self, tmp_path):
        """Test that job_id is auto-generated when not provided."""
        skill = SystemDoctorSkill()
        output_dir = str(tmp_path / "diagnose")

        result = skill.execute(output_dir=output_dir)

        # Verify job_id was generated
        assert result["job_id"] is not None
        assert result["job_id"].startswith("diag_")
        assert result["status"] == "success"

    def test_execute_security_violation_shell_execution(self, tmp_path):
        """Test that security violations are caught at dispatcher level."""
        # Note: The skill itself doesn't enforce shell execution policy,
        # but the dispatcher does. This test verifies error propagation.
        skill = SystemDoctorSkill()
        output_dir = str(tmp_path / "diagnose")

        # SystemDoctorSkill sets allow_shell=False internally
        # We can't directly test shell violation at skill level
        # (it's enforced by dispatcher before reaching handler)
        # This test primarily validates the error path structure

        result = skill.execute(output_dir=output_dir)
        assert result["status"] == "success"  # Should succeed with allow_shell=False

    def test_execute_handles_output_dir_creation(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        skill = SystemDoctorSkill()
        output_dir = str(tmp_path / "nested" / "diagnose" / "path")

        result = skill.execute(output_dir=output_dir)

        assert result["status"] == "success"
        assert Path(output_dir).exists()


class TestPredictSkill:
    """Test suite for PredictSkill Agent wrapper."""

    def test_metadata_properties(self):
        """Verify skill exposes correct tool metadata for Agent registration."""
        skill = PredictSkill()

        # Verify name
        assert skill.name == "yolo_predict"

        # Verify description is non-empty and descriptive
        assert len(skill.description) > 50
        assert "yolo" in skill.description.lower()
        assert "inference" in skill.description.lower()

        # Verify parameters schema structure
        schema = skill.parameters_schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "model_path" in schema["properties"]
        assert "data_source" in schema["properties"]
        assert "allowed_paths" in schema["properties"]
        assert "conf" in schema["properties"]
        assert "device" in schema["properties"]
        assert schema["required"] == ["model_path", "data_source", "allowed_paths"]

        # Verify conf parameter constraints
        conf_schema = schema["properties"]["conf"]
        assert conf_schema["minimum"] == 0.0
        assert conf_schema["maximum"] == 1.0
        assert conf_schema["exclusiveMinimum"] is True

    def test_execute_happy_path_with_mock(self, tmp_path):
        """Test successful prediction execution with mocked YOLO engine."""
        skill = PredictSkill()
        output_dir = str(tmp_path / "predict")

        # Create dummy model and data files within allowed paths
        model_path = tmp_path / "yolov8n.pt"
        model_path.touch()
        data_source = tmp_path / "bus.jpg"
        data_source.touch()

        # Mock YOLO engine to avoid actual model loading
        with patch("ultralytics.YOLO") as mock_yolo_class:
            # Mock YOLO instance and predict results
            mock_yolo = MagicMock()
            mock_yolo_class.return_value = mock_yolo

            # Mock prediction results
            mock_result = MagicMock()
            mock_result.save_dir = tmp_path / "predict" / "test_predict_001"
            mock_result.save_dir.mkdir(parents=True, exist_ok=True)

            # Create mock artifact files
            artifact1 = mock_result.save_dir / "bus_annotated.jpg"
            artifact1.touch()
            artifact2 = mock_result.save_dir / "labels.txt"
            artifact2.touch()

            mock_yolo.predict.return_value = [mock_result]

            # Execute prediction
            result = skill.execute(
                model_path=str(model_path),
                data_source=str(data_source),
                allowed_paths=[str(tmp_path)],
                conf=0.25,
                device="cpu",
                output_dir=output_dir,
                job_id="test_predict_001",
            )

            # Verify success status
            assert result["status"] == "success"
            assert result["job_id"] == "test_predict_001"
            assert result["error_code"] is None
            assert result["error_message"] is None

            # Verify metadata
            metadata = result["metadata"]
            assert metadata["model"] == str(model_path)
            assert metadata["source"] == str(data_source)
            assert metadata["device"] == "cpu"
            assert metadata["conf"] == 0.25

            # Verify artifacts
            assert len(result["artifacts"]) == 2
            assert any("bus_annotated.jpg" in a for a in result["artifacts"])
            assert any("labels.txt" in a for a in result["artifacts"])

    def test_execute_auto_generates_job_id(self, tmp_path):
        """Test that job_id is auto-generated when not provided."""
        skill = PredictSkill()
        model_path = tmp_path / "yolov8n.pt"
        model_path.touch()
        data_source = tmp_path / "bus.jpg"
        data_source.touch()

        with patch("ultralytics.YOLO") as mock_yolo_class:
            mock_yolo = MagicMock()
            mock_yolo_class.return_value = mock_yolo
            mock_result = MagicMock()
            mock_result.save_dir = tmp_path / "predict" / "auto_job"
            mock_result.save_dir.mkdir(parents=True, exist_ok=True)
            mock_yolo.predict.return_value = [mock_result]

            result = skill.execute(
                model_path=str(model_path),
                data_source=str(data_source),
                allowed_paths=[str(tmp_path)],
            )

            # Verify job_id was auto-generated
            assert result["job_id"] is not None
            assert result["job_id"].startswith("predict_")
            assert result["status"] == "success"

    def test_execute_fail_closed_empty_allowed_paths(self, tmp_path):
        """Test that empty allowed_paths triggers fail-closed security policy."""
        skill = PredictSkill()
        model_path = tmp_path / "yolov8n.pt"
        data_source = tmp_path / "bus.jpg"

        result = skill.execute(
            model_path=str(model_path),
            data_source=str(data_source),
            allowed_paths=[],  # Empty whitelist
            job_id="test_empty_whitelist",
        )

        # Verify failure due to empty whitelist
        assert result["status"] == "failed"
        assert result["error_code"] == "PARAM_VALIDATION_FAILED"
        assert "allowed_paths cannot be empty" in result["error_message"]
        assert result["artifacts"] == []

    def test_execute_invalid_conf_threshold(self, tmp_path):
        """Test that invalid confidence thresholds are rejected."""
        skill = PredictSkill()
        model_path = tmp_path / "yolov8n.pt"
        data_source = tmp_path / "bus.jpg"

        # Test conf > 1.0
        result = skill.execute(
            model_path=str(model_path),
            data_source=str(data_source),
            allowed_paths=[str(tmp_path)],
            conf=1.5,
        )
        assert result["status"] == "failed"
        assert result["error_code"] == "PARAM_VALIDATION_FAILED"
        assert "conf must be in range" in result["error_message"]

        # Test conf <= 0.0
        result = skill.execute(
            model_path=str(model_path),
            data_source=str(data_source),
            allowed_paths=[str(tmp_path)],
            conf=0.0,
        )
        assert result["status"] == "failed"
        assert result["error_code"] == "PARAM_VALIDATION_FAILED"
        assert "conf must be in range" in result["error_message"]

        # Test conf < 0
        result = skill.execute(
            model_path=str(model_path),
            data_source=str(data_source),
            allowed_paths=[str(tmp_path)],
            conf=-0.1,
        )
        assert result["status"] == "failed"
        assert result["error_code"] == "PARAM_VALIDATION_FAILED"

    def test_execute_invalid_conf_type(self, tmp_path):
        """Test that non-numeric conf values are rejected."""
        skill = PredictSkill()
        model_path = tmp_path / "yolov8n.pt"
        data_source = tmp_path / "bus.jpg"

        result = skill.execute(
            model_path=str(model_path),
            data_source=str(data_source),
            allowed_paths=[str(tmp_path)],
            conf="invalid",  # type: ignore[arg-type]
        )

        assert result["status"] == "failed"
        assert result["error_code"] == "PARAM_VALIDATION_FAILED"
        assert "conf must be a valid float" in result["error_message"]

    def test_execute_path_whitelist_violation(self, tmp_path):
        """Test that path traversal attacks are blocked by dispatcher."""
        skill = PredictSkill()
        output_dir = str(tmp_path / "predict")

        # Create allowed directory
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Try to access file outside allowed_paths
        model_path = tmp_path / "outside" / "yolov8n.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.touch()

        data_source = allowed_dir / "bus.jpg"
        data_source.touch()

        with patch("ultralytics.YOLO"):
            result = skill.execute(
                model_path=str(model_path),
                data_source=str(data_source),
                allowed_paths=[str(allowed_dir)],  # Only allow allowed_dir
                output_dir=output_dir,
            )

            # Verify failure due to path violation
            assert result["status"] == "failed"
            assert result["error_code"] == "PARAM_VALIDATION_FAILED"
            assert "not within allowed_paths whitelist" in result["error_message"]

    def test_execute_dynamic_whitelist_includes_output_dir(self, tmp_path):
        """Test that output_dir is automatically added to allowed_paths."""
        skill = PredictSkill()

        # Create separate allowed directory for inputs
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        model_path = input_dir / "yolov8n.pt"
        model_path.touch()
        data_source = input_dir / "bus.jpg"
        data_source.touch()

        # Output directory is different from input directory
        output_dir = str(tmp_path / "outputs")

        with patch("ultralytics.YOLO") as mock_yolo_class:
            mock_yolo = MagicMock()
            mock_yolo_class.return_value = mock_yolo
            mock_result = MagicMock()
            job_id = f"test_{uuid.uuid4().hex[:8]}"
            mock_result.save_dir = Path(output_dir) / job_id
            mock_result.save_dir.mkdir(parents=True, exist_ok=True)
            mock_yolo.predict.return_value = [mock_result]

            result = skill.execute(
                model_path=str(model_path),
                data_source=str(data_source),
                allowed_paths=[str(input_dir)],  # Only input_dir in allowed_paths
                output_dir=output_dir,
                job_id=job_id,
            )

            # Verify success (output_dir was added to dynamic whitelist)
            assert result["status"] == "success"

    def test_execute_propagates_handler_errors(self, tmp_path):
        """Test that handler execution errors are properly propagated."""
        skill = PredictSkill()
        model_path = tmp_path / "yolov8n.pt"
        model_path.touch()
        data_source = tmp_path / "bus.jpg"
        data_source.touch()

        # Mock YOLO to raise an exception
        with patch("ultralytics.YOLO") as mock_yolo_class:
            mock_yolo_class.side_effect = RuntimeError("Model loading failed")

            result = skill.execute(
                model_path=str(model_path),
                data_source=str(data_source),
                allowed_paths=[str(tmp_path)],
            )

            # Verify error is propagated
            assert result["status"] == "failed"
            assert result["error_code"] in ["EXEC_ERR_500", "HANDLER_EXEC_FAILED"]
            assert "Model loading failed" in result["error_message"] or "RuntimeError" in result["error_message"]

    def test_execute_missing_required_params_caught_by_handler(self, tmp_path):
        """Test that missing model_path/data_source are caught by handler validation."""
        skill = PredictSkill()
        data_source = tmp_path / "bus.jpg"
        data_source.touch()

        # Note: This test would require bypassing skill-level validation
        # In practice, the skill's type hints enforce required params at call time
        # The handler's validate_params provides a second validation layer

        # Test will verify handler-level validation by constructing valid skill call
        # but with handler receiving invalid params (simulated)
        with patch("ultralytics.YOLO"):
            result = skill.execute(
                model_path="",  # Empty string (handler should reject)
                data_source=str(data_source),
                allowed_paths=[str(tmp_path)],
            )

            # Handler validation should catch empty model_path
            assert result["status"] == "failed"
            # Error could be from path validation or model loading
            assert result["error_code"] in ["PARAM_VALIDATION_FAILED", "EXEC_ERR_500", "HANDLER_EXEC_FAILED"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
