"""Unit tests for F1 Task Handler Framework.

This test suite validates:
1. BaseTaskHandler abstract interface enforcement
2. TaskHandlerRegistry registration and lookup
3. Path safety validation logic
4. Error handling for unregistered task types
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from f1.handlers import BaseTaskHandler, TaskHandlerRegistry


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


class TestPathSafetyRegexWhitelist:
    """Regex-enhanced path whitelisting unit tests (P1)."""

    @staticmethod
    def _make_handler() -> BaseTaskHandler:
        class ConcreteHandler(BaseTaskHandler):
            def validate_params(self, params, security_constraints):
                return True, None

            def execute(self, job_id, params, output_dir):
                return {"success": True, "artifacts": []}

        return ConcreteHandler()

    @staticmethod
    def _posix_root(path: Path) -> str:
        """Regex-escaped normalized resolved POSIX form used to build patterns."""
        return re.escape(path.resolve().as_posix())

    def test_regex_pattern_match_succeeds(self, tmp_path):
        """A path fully matching an allowed regex pattern is accepted."""
        handler = self._make_handler()
        root = self._posix_root(tmp_path)
        target = tmp_path / "models" / "yolov8n.pt"
        assert handler._is_path_safe(str(target), [], [f"^{root}/models/.*\\.pt$"]) is True

    def test_regex_pattern_resolving_outside_allowed_boundaries_fails(self, tmp_path):
        """A path that textually matches the pattern but resolves outside it is rejected."""
        handler = self._make_handler()
        root = self._posix_root(tmp_path)
        # Textual prefix of this path matches "^{root}/sub/.*", but the resolved
        # path (tmp_path's parent + file) does not -> fail-closed rejection.
        target = tmp_path / "sub" / ".." / ".." / "escape.pt"
        assert handler._is_path_safe(str(target), [], [f"^{root}/sub/.*\\.pt$"]) is False

    def test_regex_substring_partial_match_bypass_rejected(self, tmp_path):
        """Prefix/substring matches must not whitelist sibling paths."""
        handler = self._make_handler()
        root = self._posix_root(tmp_path)
        # Pattern with no tail anchor must NOT whitelist a sibling directory
        # whose name merely starts with the same prefix.
        sibling = tmp_path / "models_evil" / "weight.pt"
        assert handler._is_path_safe(str(sibling), [], [f"^{root}/models"]) is False
        # Same for a child of the intended directory: the pattern must fully
        # describe the path, e.g. with an explicit wildcard tail.
        child = tmp_path / "models" / "weight.pt"
        assert handler._is_path_safe(str(child), [], [f"^{root}/models"]) is False
        assert handler._is_path_safe(str(child), [], [f"^{root}/models/.*"]) is True

    def test_regex_entry_detected_inside_allowed_paths(self, tmp_path):
        """Entries in allowed_paths starting with '^' are treated as regex patterns."""
        handler = self._make_handler()
        root = self._posix_root(tmp_path)
        allowed_roots = [f"^{root}/cache/.*\\.pt$"]
        assert handler._is_path_safe(str(tmp_path / "cache" / "yolov8n.pt"), allowed_roots) is True
        assert handler._is_path_safe(str(tmp_path / "other" / "yolov8n.pt"), allowed_roots) is False

    def test_malformed_regex_fails_closed_without_crash(self, tmp_path):
        """Malformed patterns never raise; they contribute nothing and a mixed
        list with one malformed pattern must not poison the valid ones."""
        handler = self._make_handler()
        root = self._posix_root(tmp_path)
        target = str(tmp_path / "models" / "yolov8n.pt")
        # Malformed-only: fail closed, no exception.
        assert handler._is_path_safe(target, [], ["^[unclosed("]) is False
        # Malformed entry beside a valid pattern: valid pattern still applies.
        patterns = ["^[unclosed(", f"^{root}/models/.*\\.pt$"]
        assert handler._is_path_safe(target, [], patterns) is True

    def test_empty_whitelist_and_patterns_reject_all(self, tmp_path):
        """Fail-closed: with both sets empty every path is rejected."""
        handler = self._make_handler()
        assert handler._is_path_safe(str(tmp_path / "valid_file.txt"), [], []) is False

    def test_regex_symlink_escape_fails(self, tmp_path):
        """A symlink escape resolves outside the pattern's boundary and is rejected."""
        handler = self._make_handler()
        root = self._posix_root(tmp_path)
        try:
            link = tmp_path / "link_out"
            link.symlink_to(tmp_path.parent, target_is_directory=True)
        except OSError:
            pytest.skip("symlink creation not permitted on this platform")
        if not link.is_symlink():
            pytest.skip(
                "symlink creation reported success but the link did not materialize "
                "(link.is_symlink() is False); the current platform/execution "
                "environment does not create real symlinks"
            )
        # Literal path matches "^{root}/.*"; the resolved path (outside tmp_path)
        # does not -> rejected.
        target = link / "secret.pt"
        assert handler._is_path_safe(str(target), [], [f"^{root}/.*\\.pt$"]) is False


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
