"""F15 MultiTask Foundation representation-transfer contracts."""

import copy
from types import SimpleNamespace

import pytest
import torch

from tests.test_foundation_distill_model import DummyTeacher, TinyStudent, config
from ultralytics.cfg import YAML, get_cfg
from ultralytics.nn.foundation_distill_model import (
    FoundationDistillationModel,
    rebuild_foundation_distillation_wrapper,
    strip_foundation_distillation_model,
)


class MultiTaskTinyStudent(TinyStudent):
    """Small two-task student exposing the existing MultiTask metadata contract."""

    def __init__(self, positive_tasks=("detect", "segment")):
        super().__init__()
        self.active_tasks = set(positive_tasks)
        self.model[-1].active_tasks = sorted(self.active_tasks)
        self.model[-1]._task_router_names = sorted(self.active_tasks)
        self.positive_tasks = set(positive_tasks)

    def loss(self, batch, preds=None):
        # Stable MultiTaskLoss order: box, cls, dfl, seg, pose, cls_global, depth, normal, semantic.
        values = torch.zeros(9, device=batch["img"].device)
        if "detect" in self.positive_tasks:
            values[:3] = 0.2
        if "segment" in self.positive_tasks:
            values[3] = 0.3
        if "pose" in self.positive_tasks:
            values[4] = 0.4
        total = values.sum().reshape(1) + self.model[0].weight.square().mean() * 0.01
        return total, values


def f15_config(**overrides):
    values = vars(
        config(
            foundation_multitask=True,
            foundation_multitask_tasks=["detect", "segment"],
            foundation_multitask_negative_transfer_threshold=2.0,
            task="multitask",
        )
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_f15_shared_kd_and_two_task_gate_report_positive_transfer_evidence():
    wrapper = FoundationDistillationModel(MultiTaskTinyStudent(), DummyTeacher(), f15_config())
    wrapper.train()
    total, items = wrapper({"img": torch.rand(2, 3, 64, 64)})
    metrics = wrapper.foundation_metrics()

    assert total.shape == (2,)
    assert items.shape[0] >= 10  # nine stable task items plus optional routed aux and Foundation items
    assert metrics["foundation_multitask_active_tasks"] == 2.0
    assert metrics["foundation_multitask_supervised_tasks"] == 2.0
    assert metrics["foundation_multitask_representation_transfer_ready"] == 1.0
    assert metrics["foundation_multitask_task_loss_detect"] > 0
    assert metrics["foundation_multitask_task_loss_segment"] > 0

    total.sum().backward()
    assert wrapper.projector.student_proj[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in wrapper.teacher_manager.parameters())


def test_f15_task_imbalance_is_observation_only_and_task_router_name_is_preserved():
    wrapper = FoundationDistillationModel(
        MultiTaskTinyStudent(positive_tasks=("detect", "segment")), DummyTeacher(), f15_config()
    )
    wrapper.train()
    wrapper({"img": torch.rand(2, 3, 64, 64)})
    metrics = wrapper.foundation_metrics()
    assert metrics["foundation_multitask_task_loss_imbalance"] == pytest.approx(2.0)
    assert metrics["foundation_multitask_negative_transfer_risk"] == 0.0
    assert wrapper.student_model.model[-1]._task_router_names == ["detect", "segment"]


def test_f15_teacher_boundary_checkpoint_resume_and_export_strip():
    wrapper = FoundationDistillationModel(MultiTaskTinyStudent(), DummyTeacher(), f15_config())
    copied = copy.deepcopy(wrapper)
    assert copied.teacher_manager is None
    assert copied.checkpoint_metadata()["multitask"]["active_tasks"] == ["detect", "segment"]
    assert all("teacher_model" not in key.lower() for key in copied.state_dict())

    rebuilt = rebuild_foundation_distillation_wrapper(
        MultiTaskTinyStudent(), f15_config(), checkpoint_model=copied, teacher_manager=DummyTeacher()
    )
    assert isinstance(rebuilt, FoundationDistillationModel)
    assert rebuilt.multitask_active_tasks == ("detect", "segment")
    assert strip_foundation_distillation_model(rebuilt) is rebuilt.student_model


def test_f15_disabled_path_is_exact_noop_and_config_requires_multitask_mode():
    student = MultiTaskTinyStudent()
    assert (
        __import__(
            "ultralytics.nn.foundation_distill_model", fromlist=["build_foundation_distillation_wrapper"]
        ).build_foundation_distillation_wrapper(student, f15_config(foundation_enabled=False))
        is student
    )
    with pytest.raises(ValueError, match="task='multitask'"):
        get_cfg(
            overrides={
                "mode": "train",
                "task": "detect",
                "foundation_enabled": True,
                "foundation_multitask": True,
                "foundation_teacher": "dinov3",
                "foundation_model": "dinov3-local",
                "foundation_loss_weight": 0.1,
            }
        )


def test_f15_recipe_is_explicit_about_three_supervised_tasks():
    from pathlib import Path

    recipe = YAML.load(
        Path(__file__).resolve().parents[1] / "ultralytics/cfg/experiments/foundation/f15-foundation-multitask.yaml"
    )
    assert recipe["task"] == "multitask"
    assert recipe["foundation_multitask"] is True
    assert set(recipe["foundation_multitask_tasks"]) == {"detect", "segment", "pose"}
