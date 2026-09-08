"""Phase 1 test suite: ValHandler, timeout supervision, cooperative cancellation, batch prediction.

This test suite validates the Phase 1 protocol unification deliverables:
1. ValHandler registration, parameter validation, execution, and artifact capture
2. Dispatcher deadline supervision (TIMEOUT) and terminal state assertion
3. Cooperative cancellation checkpoints (handler-side raises + dispatcher-side polling)
4. Batch prediction input normalization, chunked execution, deterministic artifact lists

Val execution is verified with a mocked Ultralytics engine (a real val run would
download a dataset); batch prediction additionally runs one real multi-image
inference E2E against the smoke checkpoint.
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.schema import JobRequest, JobStatus, OutputConfig, RuntimeTracking, SecurityConstraints, TaskType
from f1.dispatcher import JobDispatcherStateMachine
from f1.handlers import TaskHandlerRegistry
from f1.handlers.base import BaseTaskHandler, CooperativeCancellationError, PathWhitelistViolationError
from f1.handlers.predict import PredictHandler
from f1.handlers.val import ValHandler

# Smoke checkpoint shipped with the F1 entry smoke test (smoke/f1/yolov8n.pt)
SMOKE_MODEL = Path(__file__).parent / "yolov8n.pt"

# Reference image for real inference tests (bundled with Ultralytics)
BUS_IMAGE = project_root / "ultralytics" / "assets" / "bus.jpg"

# Loose upper bound used to prove dispatcher-side supervision returns promptly
# instead of waiting for the slow handler to finish on its own.
SUPERVISION_RETURN_BOUND = 3.0


# ---------------------------------------------------------------------------
# Test helper doubles
# ---------------------------------------------------------------------------
class FakeValModel:
    """Mocked Ultralytics engine for val: writes reports and returns metrics."""

    def __init__(self, path: str) -> None:
        self.path = path

    def val(self, **kwargs: Any) -> Any:
        job_dir = Path(kwargs["project"]) / kwargs["name"]
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "results.csv").write_text("epoch,metrics/mAP50(B)\n", encoding="utf-8")
        (job_dir / "confusion_matrix.png").write_bytes(b"fake-png")
        box = SimpleNamespace(map50=0.62, map75=0.41, map=0.45, p=[0.9, 0.8], r=[0.7, 0.6])
        return SimpleNamespace(box=box, speed={"inference": 1.234, "preprocess": 0.12})


class FakeBatchPredictModel:
    """Mocked Ultralytics engine for predict: records chunk sizes and writes one artifact per source."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.predict_calls: list[list[str]] = []

    def predict(self, **kwargs: Any) -> list[Any]:
        sources = list(kwargs["source"])
        self.predict_calls.append(sources)
        batch_dir = Path(kwargs["project"]) / kwargs["name"]
        batch_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for source in sources:
            annotated = batch_dir / f"{Path(source).stem}.annotated.jpg"
            annotated.write_bytes(b"fake-annotated-image")
            results.append(SimpleNamespace(save_dir=str(batch_dir)))
        return results


class SlowHandler(BaseTaskHandler):
    """Handler that sleeps past the 1s test deadline and never cancels on its own.

    Runs in the dispatcher's daemon worker thread, so the 2s sleep never blocks
    the test: the deadline supervision returns after ~1s and the detached daemon
    thread is discarded at process exit.
    """

    def validate_params(self, params: dict[str, Any], security_constraints: dict[str, Any]) -> tuple[bool, str | None]:
        return True, None

    def execute(self, job_id: str, params: dict[str, Any], output_dir: str) -> dict[str, Any]:
        time.sleep(2)
        return {"success": True, "artifacts": [], "metadata": {}, "error": None}


class CooperativeLoopHandler(BaseTaskHandler):
    """Handler that cooperates: checks the cancellation checkpoint on every loop iteration."""

    def validate_params(self, params: dict[str, Any], security_constraints: dict[str, Any]) -> tuple[bool, str | None]:
        return True, None

    def execute(self, job_id: str, params: dict[str, Any], output_dir: str) -> dict[str, Any]:
        for _ in range(100):
            time.sleep(0.02)
            self._check_cancelled()
        return {"success": True, "artifacts": [], "metadata": {}, "error": None}


class SelfCancellingHandler(BaseTaskHandler):
    """Handler that raises CooperativeCancellationError at its checkpoint (deterministic)."""

    def validate_params(self, params: dict[str, Any], security_constraints: dict[str, Any]) -> tuple[bool, str | None]:
        return True, None

    def execute(self, job_id: str, params: dict[str, Any], output_dir: str) -> dict[str, Any]:
        raise CooperativeCancellationError("Job execution cancelled by user request (cooperative checkpoint)")


def _registry_get_stub(handler_class: type[BaseTaskHandler]):
    """Build a classmethod stub for TaskHandlerRegistry.get returning handler_class."""

    def _stub(cls, task_type: str) -> type[BaseTaskHandler]:
        return handler_class

    return classmethod(_stub)


def _flip_cancel_after(job: JobRequest, delay: float) -> None:
    """Simulate a user pressing cancel mid-flight by flipping the flag after a delay."""
    time.sleep(delay)
    job.runtime_tracking.cancel_requested = True


# ---------------------------------------------------------------------------
# ValHandler
# ---------------------------------------------------------------------------
class TestValHandlerRegistration:
    """Test suite for ValHandler registration and TaskType.VAL contract."""

    def test_val_handler_is_registered(self):
        """Verify that ValHandler is registered under the 'val' task_type."""
        assert TaskHandlerRegistry.get("val") is ValHandler

    def test_val_task_type_enum_value(self):
        """Verify TaskType.VAL maps to the 'val' string used by the registry."""
        assert TaskType.VAL.value == "val"
        assert TaskHandlerRegistry.get(TaskType.VAL.value) is ValHandler

    def test_val_handler_instantiation(self):
        """Verify that ValHandler can be instantiated with the full contract."""
        handler = ValHandler()
        assert hasattr(handler, "validate_params")
        assert hasattr(handler, "execute")
        assert hasattr(handler, "_check_cancelled")


class TestValHandlerValidation:
    """Test suite for ValHandler parameter validation."""

    def setup_method(self):
        self.handler = ValHandler()

    def test_validation_accepts_valid_params(self):
        """Verify that validation passes for a complete valid parameter set."""
        params = {
            "model_path": "yolov8n.pt",
            "data_source": "coco8.yaml",
            "imgsz": 320,
            "batch_size": 8,
            "device": "cpu",
        }
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": [".", "runs"]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is True
        assert err is None

    def test_validation_requires_model_path(self):
        """Verify that validation fails when the checkpoint (model_path) is missing."""
        params = {"data_source": "coco8.yaml"}
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

    def test_validation_rejects_non_yaml_data_source(self):
        """Verify that an image (or any non-YAML) data_source is rejected.

        Regression for the E2E val hang: a non-YAML source hangs the engine in
        dataset/weight resolution, so validation must reject it before execution.
        """
        params = {"model_path": "yolov8n.pt", "data_source": "ultralytics/assets/bus.jpg"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "dataset YAML file" in err

    def test_validation_rejects_empty_data_source(self):
        """Verify that an empty data_source is rejected as a non-YAML source."""
        params = {"model_path": "yolov8n.pt", "data_source": ""}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "dataset YAML file" in err

    def test_validation_accepts_yaml_and_yml_data_sources(self):
        """Verify that both .yaml and .yml dataset configurations pass validation."""
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}
        for source in ("coco8.yaml", "dataset.yml"):
            params = {"model_path": "yolov8n.pt", "data_source": source}
            is_valid, err = self.handler.validate_params(params, constraints)
            assert is_valid is True, f"data_source '{source}' should be accepted: {err}"
            assert err is None

    def test_validation_rejects_path_outside_whitelist(self):
        """Verify path containment rejection for the checkpoint path raises the
        dedicated security exception (dispatcher signal for SEC_ERR_001)."""
        params = {"model_path": "../../etc/passwd", "data_source": "coco8.yaml"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["ultralytics/assets"]}

        with pytest.raises(PathWhitelistViolationError, match="model_path") as exc_info:
            self.handler.validate_params(params, constraints)
        assert "whitelist" in str(exc_info.value).lower()

    def test_validation_rejects_data_source_outside_whitelist(self):
        """Verify path containment rejection for the dataset configuration."""
        params = {"model_path": "ultralytics/assets/yolov8n.pt", "data_source": "../../etc/shadow"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["ultralytics/assets"]}

        with pytest.raises(PathWhitelistViolationError, match="data_source") as exc_info:
            self.handler.validate_params(params, constraints)
        assert "whitelist" in str(exc_info.value).lower()

    def test_validation_rejects_shell_execution(self):
        """Verify that validation fails when allow_shell=True."""
        params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml"}
        constraints = {"path_whitelisted": True, "allow_shell": True, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "shell" in err.lower()

    def test_validation_requires_path_whitelisting(self):
        """Verify that validation fails when path_whitelisted=False."""
        params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml"}
        constraints = {"path_whitelisted": False, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "path whitelisting" in err.lower()

    def test_validation_rejects_empty_allowed_paths(self):
        """Verify fail-closed policy: empty allowed_paths rejects everything."""
        params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml"}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": []}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "allowed_paths" in err
        assert "empty" in err.lower()

    def test_validation_rejects_non_positive_imgsz(self):
        """Verify that validation fails for imgsz <= 0."""
        params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml", "imgsz": 0}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "imgsz" in err

    def test_validation_rejects_non_positive_batch_size(self):
        """Verify that validation fails for batch_size <= 0."""
        params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml", "batch_size": -1}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "batch_size" in err

    def test_validation_rejects_invalid_conf(self):
        """Verify that validation fails for conf outside (0.0, 1.0]."""
        params = {"model_path": "yolov8n.pt", "data_source": "coco8.yaml", "conf": 1.5}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "conf" in err


class TestValHandlerExecution:
    """Test suite for ValHandler execution, metric parsing, and artifact capture."""

    def setup_method(self):
        self.handler = ValHandler()

    def test_execute_handles_missing_checkpoint_gracefully(self, tmp_path):
        """Verify that execute() returns success=False when the checkpoint is missing.

        A local junk file is used instead of a nonexistent name so the engine fails
        locally without attempting a model download.
        """
        junk_checkpoint = tmp_path / "not_a_model.pt"
        junk_checkpoint.write_bytes(b"not a checkpoint")

        params = {"model_path": str(junk_checkpoint), "data_source": "coco8.yaml", "device": "cpu"}

        result = self.handler.execute("val-001", params, str(tmp_path))

        assert result["success"] is False
        assert result["error"] is not None
        assert "Validation failed" in result["error"]
        assert len(result["artifacts"]) == 0

    def test_execute_happy_path_with_mocked_engine(self, tmp_path, monkeypatch):
        """Verify execute() captures reports and parses metrics with a mocked engine."""
        monkeypatch.setattr("ultralytics.YOLO", FakeValModel)

        params = {
            "model_path": str(SMOKE_MODEL),
            "data_source": "coco8.yaml",
            "imgsz": 320,
            "batch_size": 4,
            "device": "cpu",
        }
        result = self.handler.execute("val-001", params, str(tmp_path))

        assert result["success"] is True
        assert result["error"] is None

        # Deterministic artifact collection: sorted absolute paths, all verified
        artifacts = result["artifacts"]
        assert len(artifacts) == 2
        assert artifacts == sorted(artifacts)
        assert any(a.endswith("results.csv") for a in artifacts)
        assert any(a.endswith("confusion_matrix.png") for a in artifacts)
        assert all(Path(a).is_file() for a in artifacts)

        # Parsed metrics from the results object
        metrics = result["metadata"]["metrics"]
        assert metrics["mAP50"] == 0.62
        assert metrics["mAP75"] == 0.41
        assert metrics["mAP50-95"] == 0.45
        assert metrics["precision"] == [0.9, 0.8]
        assert metrics["recall"] == [0.7, 0.6]
        assert metrics["speed_ms"]["inference"] == 1.234

    def test_execute_isolates_output_by_job_id(self, tmp_path, monkeypatch):
        """Verify that artifacts land in job-specific directories."""
        monkeypatch.setattr("ultralytics.YOLO", FakeValModel)

        params = {"model_path": str(SMOKE_MODEL), "data_source": "coco8.yaml", "device": "cpu"}
        result1 = self.handler.execute("val-001", params, str(tmp_path))
        result2 = self.handler.execute("val-002", params, str(tmp_path))

        assert result1["success"] is True
        assert result2["success"] is True
        assert (tmp_path / "val-001" / "results.csv").exists()
        assert (tmp_path / "val-002" / "results.csv").exists()
        assert result1["artifacts"] != result2["artifacts"]

    def test_parse_val_metrics_none_results(self):
        """Verify defensive metric parsing for a None results object."""
        assert self.handler._parse_val_metrics(None) == {}

    def test_parse_val_metrics_results_dict_fallback(self):
        """Verify the results_dict fallback path for older engine versions."""
        results = SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.51, "metrics/mAP50-95(B)": 0.33})
        metrics = self.handler._parse_val_metrics(results)
        assert metrics["mAP50"] == 0.51
        assert metrics["mAP50-95"] == 0.33

    def test_dispatcher_val_end_to_end(self, tmp_path, monkeypatch):
        """Verify dispatcher -> ValHandler flow through the supervised worker path."""
        monkeypatch.setattr("ultralytics.YOLO", FakeValModel)

        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="val-e2e",
            task_type=TaskType.VAL,
            params={"model_path": str(SMOKE_MODEL), "data_source": "coco8.yaml", "device": "cpu"},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=[str(project_root)],
            ),
            output=OutputConfig(output_dir=str(tmp_path)),
        )

        result = dispatcher.execute(job)

        assert result.status == JobStatus.COMPLETED
        assert result.error is None
        assert len(result.output.artifacts) == 2
        assert all(Path(a).is_file() for a in result.output.artifacts)

    def test_dispatcher_rejects_non_yaml_data_source_before_execution(self, tmp_path):
        """Verify a non-YAML data_source fails fast with PARAM_VALIDATION_FAILED.

        The engine is never invoked: validation rejects the source up front, so the
        job transitions straight from PENDING to FAILED instead of hanging in RUNNING.
        """
        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="val-bad-data",
            task_type=TaskType.VAL,
            params={"model_path": str(SMOKE_MODEL), "data_source": str(BUS_IMAGE), "device": "cpu"},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=[str(project_root)],
            ),
            output=OutputConfig(output_dir=str(tmp_path)),
        )

        result = dispatcher.execute(job)

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert result.error.code == "PARAM_VALIDATION_FAILED"
        assert "dataset YAML file" in result.error.message
        assert len(result.output.artifacts) == 0


# ---------------------------------------------------------------------------
# Timeout supervision
# ---------------------------------------------------------------------------
class TestTimeoutSupervision:
    """Test suite for dispatcher deadline supervision (Phase 1 TIMEOUT contract)."""

    def test_timeout_transitions_job_to_failed_with_timeout_code(self, monkeypatch):
        """Verify a handler exceeding timeout_seconds yields FAILED + TIMEOUT promptly."""
        monkeypatch.setattr(TaskHandlerRegistry, "get", _registry_get_stub(SlowHandler))

        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-timeout-001",
            task_type=TaskType.PREDICT,
            params={},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=["."],
            ),
            runtime_tracking=RuntimeTracking(timeout_seconds=1),
        )

        started = time.monotonic()
        result = dispatcher.execute(job)
        elapsed = time.monotonic() - started

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert result.error.code == "TIMEOUT"
        assert "timeout" in result.error.message.lower()
        assert elapsed < SUPERVISION_RETURN_BOUND  # returned before the handler finished
        assert len(result.output.artifacts) == 0

    def test_zero_timeout_fails_immediately(self, monkeypatch):
        """Verify timeout_seconds=0 fails immediately without waiting for the handler."""
        monkeypatch.setattr(TaskHandlerRegistry, "get", _registry_get_stub(SlowHandler))

        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-timeout-000",
            task_type=TaskType.PREDICT,
            params={},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=["."],
            ),
            runtime_tracking=RuntimeTracking(timeout_seconds=0),
        )

        started = time.monotonic()
        result = dispatcher.execute(job)
        elapsed = time.monotonic() - started

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert result.error.code == "TIMEOUT"
        assert elapsed < SUPERVISION_RETURN_BOUND

    def test_failed_state_is_terminal_after_timeout(self, monkeypatch):
        """Verify the FAILED state produced by a timeout accepts no further transitions."""
        monkeypatch.setattr(TaskHandlerRegistry, "get", _registry_get_stub(SlowHandler))

        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-timeout-term",
            task_type=TaskType.PREDICT,
            params={},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=["."],
            ),
            runtime_tracking=RuntimeTracking(timeout_seconds=1),
        )

        result = dispatcher.execute(job)
        assert result.status == JobStatus.FAILED

        with pytest.raises(ValueError, match="Illegal state transition"):
            dispatcher.transition(result, JobStatus.COMPLETED)


# ---------------------------------------------------------------------------
# Cooperative cancellation
# ---------------------------------------------------------------------------
class TestCooperativeCancellation:
    """Test suite for cooperative cancellation checkpoints (handler-side and dispatcher-side)."""

    def test_check_cancelled_is_noop_without_injected_token(self):
        """Verify _check_cancelled() is a no-op for direct handler usage."""
        handler = ValHandler()
        handler._check_cancelled()  # must not raise

    def test_check_cancelled_raises_when_token_requests_cancel(self):
        """Verify _check_cancelled() raises CooperativeCancellationError when requested."""
        handler = ValHandler()
        handler._runtime_tracking = RuntimeTracking(cancel_requested=True)
        with pytest.raises(CooperativeCancellationError):
            handler._check_cancelled()

    def test_handler_cancellation_exception_maps_to_user_cancelled(self, monkeypatch):
        """Verify a handler-side CooperativeCancellationError maps to FAILED + USER_CANCELLED."""
        monkeypatch.setattr(TaskHandlerRegistry, "get", _registry_get_stub(SelfCancellingHandler))

        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-cancel-checkpoint",
            task_type=TaskType.PREDICT,
            params={},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=["."],
            ),
        )

        result = dispatcher.execute(job)

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert result.error.code == "USER_CANCELLED"
        assert "cancel" in result.error.message.lower()
        assert len(result.output.artifacts) == 0

    def test_inflight_cancellation_polled_by_dispatcher(self, monkeypatch):
        """Verify in-flight cancellation on a non-cooperative handler via dispatcher polling."""
        monkeypatch.setattr(TaskHandlerRegistry, "get", _registry_get_stub(SlowHandler))

        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-cancel-inflight",
            task_type=TaskType.PREDICT,
            params={},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=["."],
            ),
            runtime_tracking=RuntimeTracking(timeout_seconds=300),
        )

        threading.Thread(target=_flip_cancel_after, args=(job, 0.2), daemon=True).start()

        started = time.monotonic()
        result = dispatcher.execute(job)
        elapsed = time.monotonic() - started

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert result.error.code == "USER_CANCELLED"
        assert elapsed < SUPERVISION_RETURN_BOUND  # aborted well before the 2s handler
        assert len(result.output.artifacts) == 0

    def test_cooperative_loop_handler_aborts_on_cancel(self, monkeypatch):
        """Verify a handler honoring checkpoints aborts promptly when cancelled mid-loop."""
        monkeypatch.setattr(TaskHandlerRegistry, "get", _registry_get_stub(CooperativeLoopHandler))

        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-cancel-loop",
            task_type=TaskType.PREDICT,
            params={},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=["."],
            ),
            runtime_tracking=RuntimeTracking(timeout_seconds=300),
        )

        threading.Thread(target=_flip_cancel_after, args=(job, 0.15), daemon=True).start()

        started = time.monotonic()
        result = dispatcher.execute(job)
        elapsed = time.monotonic() - started

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert result.error.code == "USER_CANCELLED"
        assert elapsed < SUPERVISION_RETURN_BOUND  # aborted instead of running the full loop
        assert len(result.output.artifacts) == 0

    def test_predict_handler_respects_cancellation_checkpoint(self, tmp_path):
        """Verify PredictHandler raises at its checkpoint when cancellation is requested."""
        handler = PredictHandler()
        handler._runtime_tracking = RuntimeTracking(cancel_requested=True)

        with pytest.raises(CooperativeCancellationError):
            handler.execute(
                job_id="batch-cancel",
                params={"model_path": str(SMOKE_MODEL), "data_source": str(BUS_IMAGE)},
                output_dir=str(tmp_path),
            )


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------
class TestBatchPredictionValidation:
    """Test suite for batch-aware PredictHandler parameter validation."""

    def setup_method(self):
        self.handler = PredictHandler()

    def test_validation_accepts_sliced_file_list(self, tmp_path):
        """Verify that a list of whitelisted file paths passes validation."""
        img_a = tmp_path / "a.jpg"
        img_b = tmp_path / "b.jpg"
        img_a.write_bytes(b"a")
        img_b.write_bytes(b"b")

        params = {
            "model_path": "yolov8n.pt",
            "data_source": [str(img_a), str(img_b)],
            "batch_size": 2,
        }
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": [str(tmp_path), "."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is True
        assert err is None

    def test_validation_accepts_directory_input(self, tmp_path):
        """Verify that a directory data_source passes whitelist validation."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        params = {"model_path": str(tmp_path / "yolov8n.pt"), "data_source": str(images_dir)}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": [str(tmp_path)]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is True
        assert err is None

    def test_validation_rejects_empty_file_list(self):
        """Verify that an empty data_source list is rejected."""
        params = {"model_path": "yolov8n.pt", "data_source": []}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "data_source" in err
        assert "empty" in err.lower()

    def test_validation_rejects_non_string_list_entry(self):
        """Verify that non-string entries in the file list are rejected."""
        params = {"model_path": "yolov8n.pt", "data_source": ["a.jpg", 42]}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "data_source[1]" in err

    def test_validation_rejects_list_entry_outside_whitelist(self, tmp_path):
        """Verify that one out-of-whitelist entry fails the whole list."""
        img_a = tmp_path / "a.jpg"
        img_a.write_bytes(b"a")

        params = {
            "model_path": "yolov8n.pt",
            "data_source": [str(img_a), "../../etc/passwd"],
        }
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": [str(tmp_path)]}

        with pytest.raises(PathWhitelistViolationError) as exc_info:
            self.handler.validate_params(params, constraints)
        assert "whitelist" in str(exc_info.value).lower()

    def test_validation_rejects_non_positive_batch_size(self):
        """Verify that batch_size <= 0 is rejected."""
        params = {"model_path": "yolov8n.pt", "data_source": "bus.jpg", "batch_size": 0}
        constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}

        is_valid, err = self.handler.validate_params(params, constraints)
        assert is_valid is False
        assert "batch_size" in err


class TestBatchPredictionExecution:
    """Test suite for batch prediction execution, artifact persistence, and determinism."""

    def setup_method(self):
        self.handler = PredictHandler()

    def test_batch_execution_chunks_and_persists_artifacts(self, tmp_path, monkeypatch):
        """Verify chunked execution, deterministic artifact list, and persistence.

        Five inputs with batch_size=2 produce chunks of [2, 2, 1]; every generated
        artifact must exist on disk and the artifact list must match the expected
        set exactly in sorted order.
        """
        monkeypatch.setattr("ultralytics.YOLO", FakeBatchPredictModel)

        sources = []
        for idx in range(5):
            source_file = tmp_path / f"img_{idx:02d}.jpg"
            source_file.write_bytes(b"fake-image")
            sources.append(str(source_file))

        params = {
            "model_path": str(SMOKE_MODEL),
            "data_source": sources,
            "device": "cpu",
            "batch_size": 2,
        }
        result = self.handler.execute("batch-001", params, str(tmp_path))

        assert result["success"] is True
        assert result["error"] is None

        # Batch structure metadata
        metadata = result["metadata"]
        assert metadata["num_inputs"] == 5
        assert metadata["num_batches"] == 3
        assert metadata["batch_size"] == 2
        assert metadata["num_results"] == 5
        assert metadata["sources"] == sources

        # Artifact persistence: every artifact exists and matches expectations
        artifacts = result["artifacts"]
        expected_names = sorted(f"img_{idx:02d}.annotated.jpg" for idx in range(5))
        actual_names = sorted(Path(a).name for a in artifacts)
        assert len(artifacts) == 5
        assert actual_names == expected_names
        assert artifacts == sorted(artifacts)  # deterministic ordering
        assert all(Path(a).is_file() for a in artifacts)

        # Per-chunk isolation directories
        for chunk_index in range(3):
            assert (tmp_path / "batch-001" / f"batch_{chunk_index:03d}").exists()

    def test_batch_execution_chunk_sizes_match_batch_size(self, tmp_path, monkeypatch):
        """Verify the engine receives exactly batch_size inputs per call."""
        fake_model = FakeBatchPredictModel(str(SMOKE_MODEL))
        monkeypatch.setattr("ultralytics.YOLO", lambda path: fake_model)

        sources = []
        for idx in range(5):
            source_file = tmp_path / f"img_{idx:02d}.jpg"
            source_file.write_bytes(b"fake-image")
            sources.append(str(source_file))

        params = {
            "model_path": str(SMOKE_MODEL),
            "data_source": sources,
            "device": "cpu",
            "batch_size": 2,
        }
        result = self.handler.execute("batch-002", params, str(tmp_path))
        assert result["success"] is True

        # The recorded engine calls must be sized [2, 2, 1]
        assert [len(call) for call in fake_model.predict_calls] == [2, 2, 1]

    def test_directory_input_expands_deterministically(self, tmp_path, monkeypatch):
        """Verify directory input expands to a sorted media file list (noise ignored)."""
        fake_model = FakeBatchPredictModel(str(SMOKE_MODEL))
        monkeypatch.setattr("ultralytics.YOLO", lambda path: fake_model)

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "b.jpg").write_bytes(b"b")
        (images_dir / "a.jpg").write_bytes(b"a")
        (images_dir / "c.png").write_bytes(b"c")
        (images_dir / "notes.txt").write_bytes(b"not media")  # must be ignored

        params = {
            "model_path": str(SMOKE_MODEL),
            "data_source": str(images_dir),
            "device": "cpu",
            "batch_size": 8,
        }
        result = self.handler.execute("batch-dir", params, str(tmp_path))

        assert result["success"] is True
        metadata = result["metadata"]
        assert metadata["num_inputs"] == 3
        assert metadata["num_batches"] == 1
        assert metadata["sources"] == sorted(metadata["sources"])
        assert [Path(s).name for s in metadata["sources"]] == ["a.jpg", "b.jpg", "c.png"]
        assert "notes.txt" not in " ".join(metadata["sources"])
        assert len(fake_model.predict_calls[0]) == 3

    def test_execute_reports_error_when_no_media_found(self, tmp_path, monkeypatch):
        """Verify that a directory without supported media yields a controlled failure."""
        monkeypatch.setattr("ultralytics.YOLO", FakeBatchPredictModel)

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        params = {
            "model_path": str(SMOKE_MODEL),
            "data_source": str(empty_dir),
            "device": "cpu",
        }
        result = self.handler.execute("batch-empty", params, str(tmp_path))

        assert result["success"] is False
        assert "no supported media files" in result["error"]
        assert len(result["artifacts"]) == 0

    def test_real_batch_inference_end_to_end(self, tmp_path):
        """Verify real multi-image inference E2E with per-chunk processing.

        Two copies of the bundled bus.jpg are processed with batch_size=1 so the
        multi-chunk engine path executes for real; artifacts must persist on disk.
        """
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for name in ("bus_a.jpg", "bus_b.jpg"):
            shutil.copy2(BUS_IMAGE, images_dir / name)

        params = {
            "model_path": str(SMOKE_MODEL),
            "data_source": str(images_dir),
            "device": "cpu",
            "batch_size": 1,
        }
        result = self.handler.execute("batch-real", params, str(tmp_path))

        assert result["success"] is True
        assert result["error"] is None
        metadata = result["metadata"]
        assert metadata["num_inputs"] == 2
        assert metadata["num_batches"] == 2
        assert metadata["num_results"] == 2

        artifacts = result["artifacts"]
        assert len(artifacts) >= 2
        assert artifacts == sorted(artifacts)
        assert all(Path(a).is_file() for a in artifacts)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
