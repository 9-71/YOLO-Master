"""Unit tests for F1 Task Handler Framework.

This test suite validates:
1. BaseTaskHandler abstract interface enforcement
2. TaskHandlerRegistry registration and lookup
3. Path safety validation logic
4. Error handling for unregistered task types
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from smoke.f1.handlers import BaseTaskHandler, TaskHandlerRegistry


class TestBaseTaskHandler:
    """Test suite for BaseTaskHandler abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Verify that BaseTaskHandler cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseTaskHandler()  # type: ignore[abstract]

    def test_concrete_handler_must_implement_all_methods(self):
        """Verify that incomplete implementations raise TypeError."""

        class IncompleteHandler(BaseTaskHandler):
            def validate_params(self, params, security_constraints):
                return True, None

            # Missing execute() implementation

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteHandler()  # type: ignore[abstract]

    def test_path_safety_validation_baseline(self):
        """Test _is_path_safe helper method with various path scenarios."""

        class ConcreteHandler(BaseTaskHandler):
            def validate_params(self, params, security_constraints):
                return True, None

            def execute(self, job_id, params, output_dir):
                return {"success": True, "artifacts": []}

        handler = ConcreteHandler()

        # Case 1: Valid path within allowed root
        assert handler._is_path_safe("ultralytics/assets/bus.jpg", [".", "runs"]) is True

        # Case 2: Path traversal attempt
        assert handler._is_path_safe("../../etc/passwd", ["ultralytics/assets"]) is False

        # Case 3: Empty allowed_roots (fail-closed)
        assert handler._is_path_safe("valid_file.txt", []) is False

        # Case 4: Absolute path within allowed root
        current_dir = Path.cwd()
        assert handler._is_path_safe(str(current_dir / "test.txt"), [str(current_dir)]) is True

        # Case 5: Symlink resolution (if OS supports)
        assert handler._is_path_safe("./././ultralytics/../ultralytics/assets", ["."]) is True


class TestTaskHandlerRegistry:
    """Test suite for TaskHandlerRegistry registration and lookup."""

    def setup_method(self):
        """Backup registry state before each test for isolation."""
        self._registry_backup = TaskHandlerRegistry._handlers.copy()
        TaskHandlerRegistry.clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        TaskHandlerRegistry._handlers = self._registry_backup

    def test_register_and_retrieve_handler(self):
        """Verify handler registration and retrieval via get()."""

        @TaskHandlerRegistry.register("test_predict")
        class TestPredictHandler(BaseTaskHandler):
            def validate_params(self, params, security_constraints):
                return True, None

            def execute(self, job_id, params, output_dir):
                return {"success": True, "artifacts": []}

        # Retrieve the registered handler
        handler_class = TaskHandlerRegistry.get("test_predict")
        assert handler_class is TestPredictHandler

        # Verify instantiation works
        handler = handler_class()
        assert isinstance(handler, BaseTaskHandler)

    def test_duplicate_registration_raises_error(self):
        """Verify that registering the same task_type twice raises ValueError."""

        @TaskHandlerRegistry.register("duplicate_task")
        class FirstHandler(BaseTaskHandler):
            def validate_params(self, params, security_constraints):
                return True, None

            def execute(self, job_id, params, output_dir):
                return {"success": True}

        # Attempt duplicate registration
        with pytest.raises(ValueError, match="Task type 'duplicate_task' is already registered"):

            @TaskHandlerRegistry.register("duplicate_task")
            class SecondHandler(BaseTaskHandler):
                def validate_params(self, params, security_constraints):
                    return True, None

                def execute(self, job_id, params, output_dir):
                    return {"success": True}

    def test_get_unregistered_task_type_raises_error(self):
        """Verify that retrieving an unregistered task_type raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Task type 'unknown_task' is not registered"):
            TaskHandlerRegistry.get("unknown_task")

    def test_register_non_basehandler_raises_error(self):
        """Verify that registering a non-BaseTaskHandler class raises TypeError."""

        with pytest.raises(TypeError, match="must inherit from BaseTaskHandler"):

            @TaskHandlerRegistry.register("invalid_handler")
            class NotAHandler:
                pass

    def test_list_registered_returns_sorted_keys(self):
        """Verify that list_registered() returns sorted task types."""

        @TaskHandlerRegistry.register("train")
        class TrainHandler(BaseTaskHandler):
            def validate_params(self, params, security_constraints):
                return True, None

            def execute(self, job_id, params, output_dir):
                return {"success": True}

        @TaskHandlerRegistry.register("predict")
        class PredictHandler(BaseTaskHandler):
            def validate_params(self, params, security_constraints):
                return True, None

            def execute(self, job_id, params, output_dir):
                return {"success": True}

        registered = TaskHandlerRegistry.list_registered()
        assert registered == ["predict", "train"]  # Alphabetically sorted


class TestEndToEndIntegration:
    """End-to-end integration tests simulating dispatcher usage."""

    def setup_method(self):
        """Backup registry state and register mock handlers."""
        self._registry_backup = TaskHandlerRegistry._handlers.copy()
        TaskHandlerRegistry.clear()

        @TaskHandlerRegistry.register("predict")
        class MockPredictHandler(BaseTaskHandler):
            def validate_params(
                self, params: dict[str, Any], security_constraints: dict[str, Any]
            ) -> tuple[bool, str | None]:
                # Enforce shell execution prohibition
                if security_constraints.get("allow_shell"):
                    return False, "Shell execution not permitted"

                # Validate paths
                allowed_paths = security_constraints.get("allowed_paths", [])
                model_path = params.get("model_path", "")
                if model_path and not self._is_path_safe(model_path, allowed_paths):
                    return False, f"Model path '{model_path}' not in whitelist"

                return True, None

            def execute(self, job_id: str, params: dict[str, Any], output_dir: str) -> dict[str, Any]:
                return {
                    "success": True,
                    "artifacts": [f"{output_dir}/{job_id}/result.jpg"],
                    "metadata": {"detected_count": 5},
                    "error": None,
                }

    def teardown_method(self):
        """Restore registry state after each test."""
        TaskHandlerRegistry._handlers = self._registry_backup

    def test_dispatcher_workflow_happy_path(self):
        """Simulate dispatcher retrieving handler and executing job."""
        handler_class = TaskHandlerRegistry.get("predict")
        handler = handler_class()

        params = {"model_path": "yolov8n.pt", "data_source": "bus.jpg"}
        security = {"allow_shell": False, "allowed_paths": ["."]}

        # Validation
        is_valid, err = handler.validate_params(params, security)
        assert is_valid is True
        assert err is None

        # Execution
        result = handler.execute("job-001", params, "runs/predict")
        assert result["success"] is True
        assert len(result["artifacts"]) > 0

    def test_dispatcher_workflow_security_violation(self):
        """Simulate dispatcher detecting security violation at validation stage."""
        handler_class = TaskHandlerRegistry.get("predict")
        handler = handler_class()

        params = {"model_path": "../../etc/passwd"}
        security = {"allow_shell": False, "allowed_paths": ["ultralytics/assets"]}

        # Validation should fail
        is_valid, err = handler.validate_params(params, security)
        assert is_valid is False
        assert "not in whitelist" in err


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
