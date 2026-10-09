"""Core schema, import direction, parameter propagation and audit regressions."""

from __future__ import annotations

import ast
import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from core.schema import JobRequest, TaskType
from core.security import REDACTED, sanitize_for_persistence
from studio.handlers import TaskHandlerRegistry
from studio.handlers.train import TrainHandler

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("job_id", ["", "..", "../escape", "a/b", "a\\b", "a.b", "a" * 129])
def test_schema_rejects_path_like_identifiers(job_id):
    with pytest.raises(ValidationError):
        JobRequest(job_id=job_id, task_type=TaskType.PREDICT)


def test_job_mutable_defaults_and_sanitized_sink():
    first = JobRequest(job_id="first", task_type="predict")
    second = JobRequest(job_id="second", task_type="train")
    first.metadata.tags.append("first-only")
    first.output.artifacts.append("candidate")
    first.security_constraints.allowed_paths.append("trusted")
    first.runtime_tracking.cancel_requested = True
    events = []
    first._set_log_event_sink(lambda *event: events.append(event))
    first.append_log("password=secret-value", terminal=True)
    assert second.metadata.tags == second.output.artifacts == second.security_constraints.allowed_paths == []
    assert second.runtime_tracking.cancel_requested is False
    assert events == [(0, first.logs[0], True)]
    assert "secret-value" not in first.logs[0]
    assert sanitize_for_persistence({"token": "secret-value", "nested": ["password=secret-value"]}) == {
        "token": REDACTED,
        "nested": [f"password={REDACTED}"],
    }


def test_registry_ready_before_concurrent_lookup():
    assert TaskHandlerRegistry.list_registered() == sorted(task.value for task in TaskType)
    with ThreadPoolExecutor(max_workers=5) as pool:
        classes = list(pool.map(TaskHandlerRegistry.get, [task.value for task in TaskType] * 4))
    assert all(cls.__module__.startswith("studio.handlers.") for cls in classes)


def test_core_import_direction():
    forbidden = {"api", "agent", "fastapi", "gradio", "frontend", "web", "skills"}
    forbidden_studio = {"jobs_manager", "worker_runtime", "admission", "job_store", "job_logs", "artifacts", "ui"}
    core_paths = [
        *(ROOT / "core").glob("*.py"),
        ROOT / "studio/__init__.py",
        ROOT / "studio/dispatcher.py",
        *(ROOT / "studio/handlers").rglob("*.py"),
    ]
    for path in core_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, (path, node.lineno)
                names = [node.module or ""]
            else:
                continue
            for name in names:
                parts = name.split(".")
                assert parts[0] not in forbidden, (path, node.lineno, name)
                assert not (parts[0] == "studio" and len(parts) > 1 and parts[1] in forbidden_studio), name
    for name in ("core.schema", "core.security", "core.path_safety", "studio.dispatcher"):
        assert Path(importlib.import_module(name).__file__).is_relative_to(ROOT)


@pytest.mark.parametrize("imgsz", [0, -32, "invalid", None])
def test_train_rejects_invalid_imgsz(imgsz, tmp_path):
    params = {
        "model_path": str(tmp_path / "model.pt"),
        "data_source": str(tmp_path / "data.yaml"),
        "epochs": 1,
        "imgsz": imgsz,
    }
    valid, error = TrainHandler().validate_params(
        params, {"allow_shell": False, "path_whitelisted": True, "allowed_paths": [str(tmp_path)]}
    )
    assert not valid and "imgsz" in error


@pytest.mark.parametrize("imgsz", [None, 64, "96"])
def test_train_imgsz_reaches_engine(imgsz, tmp_path, monkeypatch):
    calls = []

    class FakeTrain:
        trainer = None

        def __init__(self, path):
            pass

        def train(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("ultralytics.YOLO", FakeTrain)
    params = {"model_path": "local-model.pt", "data_source": "local-data.yaml", "epochs": 1}
    if imgsz is not None:
        params["imgsz"] = imgsz
    result = TrainHandler().execute("train", params, str(tmp_path))
    assert result["success"]
    if imgsz is None:
        assert "imgsz" not in calls[0]
    else:
        assert calls[0]["imgsz"] == int(imgsz)


def test_train_reuses_upstream_audit_without_rescan():
    import torch

    from ultralytics.optim import audit_optimizer_param_groups

    model = torch.nn.Module()
    model.register_parameter("lora_A", torch.nn.Parameter(torch.ones(1)))
    model.register_parameter("router_bias", torch.nn.Parameter(torch.ones(1)))
    optimizer = torch.optim.SGD(
        [
            {"params": [model.lora_A], "param_group": "adapter", "weight_decay": 0.0},
            {"params": [model.router_bias], "param_group": "bias", "weight_decay": 0.0},
        ],
        lr=0.1,
    )
    audit = audit_optimizer_param_groups(model, optimizer)
    # No trainer.model or optimizer is exposed: the handler must use the snapshot.
    trainer = SimpleNamespace(optimizer_group_audit=audit)
    result = TrainHandler()._audit_optimizer_param_groups(SimpleNamespace(trainer=trainer))
    assert result == {
        "audited": True,
        "total_param_groups": 2,
        "has_lora_params": True,
        "has_moe_params": True,
        "bias_wd_correct": True,
        "warnings": [],
    }
    audit["missing_trainable_count"] = 1
    assert (
        "missing_trainable_count=1"
        in TrainHandler()._audit_optimizer_param_groups(SimpleNamespace(trainer=trainer))["warnings"]
    )


def test_missing_audit_does_not_claim_success():
    result = TrainHandler()._audit_optimizer_param_groups(SimpleNamespace(trainer=None))
    assert result["audited"] is False
    assert result["bias_wd_correct"] is None
