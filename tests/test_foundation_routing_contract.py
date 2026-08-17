"""F11 Foundation Teacher Router and routing-KD contracts."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.test_foundation_distill_model import DummyTeacher, config
from ultralytics.nn.foundation import (
    FoundationFeatures,
    FoundationTeacherRouter,
    foundation_teacher_summary,
    routing_kd_loss,
)
from ultralytics.nn.foundation_distill_model import FoundationDistillationModel
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.latent_mixture import LatentMixture
from ultralytics.nn.modules.routing_protocol import clear_aux_records, collect_aux_loss, get_aux_record
from ultralytics.utils.checkpoint_compat import checkpoint_runtime_metadata


class RoutedTinyStudent(nn.Module):
    """Small YOLO-like graph with one image-level LatentMixture route."""

    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList(
            [
                nn.Conv2d(3, 8, 1),
                nn.Conv2d(8, 8, 1),
                LatentMixture([8], 8, num_experts=3, residual_init=0.01),
                nn.Conv2d(8, 16, 1),
            ]
        )
        head = Detect(nc=2, ch=(8, 8, 16))
        head.f, head.i = [1, 2, 3], 4
        self.model.append(head)
        self.yaml = {"channels": 3}
        self.stride = torch.tensor([32])
        self.nc = 2
        self.names = {0: "zero", 1: "one"}
        self.args = SimpleNamespace(imgsz=64)
        self.criterion = None

    def predict(self, x, *args, **kwargs):
        outputs = []
        for index, layer in enumerate(self.model):
            if index == self.model[-1].i:
                return layer([outputs[source] for source in layer.f])
            x = layer([outputs[0]]) if isinstance(layer, LatentMixture) else layer(x)
            outputs.append(x)
        raise AssertionError("unreachable")

    def forward(self, x, *args, **kwargs):
        return self.predict(x, *args, **kwargs)

    def loss(self, batch, preds=None):
        return self.model[0].weight.square().mean().reshape(1), torch.ones(3, device=batch["img"].device)

    def init_criterion(self):
        return None

    def set_head_attr(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.model[-1], key, value)


def route_config(**overrides):
    values = vars(config())
    values.update(
        foundation_router_distill=True,
        foundation_router_loss_weight=0.25,
        foundation_router_temperature=2.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_teacher_summary_contains_pooled_mean_and_spatial_std():
    dense = torch.arange(2 * 4 * 2 * 2, dtype=torch.float32).reshape(2, 4, 2, 2)
    pooled = dense.mean((2, 3)) + 1
    summary = foundation_teacher_summary(FoundationFeatures(dense={"p4": dense}, pooled=pooled))
    assert summary.shape == (2, 12)
    assert torch.allclose(summary[:, :4], pooled)
    assert torch.isfinite(summary).all()


def test_routing_kd_uses_t_squared_and_only_student_gets_grad():
    student = torch.randn(2, 3, requires_grad=True)
    teacher = torch.randn(2, 3, requires_grad=True)
    loss = routing_kd_loss(student, teacher, temperature=2.0)
    loss.backward()
    assert loss.isfinite()
    assert student.grad is not None
    assert teacher.grad is None


def test_frozen_teacher_router_is_deterministic_and_has_no_trainable_params():
    a = FoundationTeacherRouter(8, 12, 3, seed=77)
    b = FoundationTeacherRouter(8, 12, 3, seed=77)
    x, y = torch.randn(2, 8), torch.randn(2, 12)
    assert torch.allclose(a(x, y), b(x, y))
    assert all(not parameter.requires_grad for parameter in a.parameters())


def test_wrapper_publishes_foundation_route_and_backpropagates_student_router():
    clear_aux_records(step=701)
    wrapper = FoundationDistillationModel(RoutedTinyStudent(), DummyTeacher(), route_config())
    wrapper.train()
    total, items = wrapper({"img": torch.rand(2, 3, 64, 64)})
    assert total.shape == (2,)
    assert items.shape == (4,)
    metrics = wrapper.foundation_metrics()
    assert metrics["foundation_router_modules"] == 1.0
    assert metrics["foundation_router_loss"] > 0.0
    record = get_aux_record(wrapper)
    assert record is not None and record.kind == "foundation_route"
    collected = collect_aux_loss(wrapper, include_kinds=("foundation_route",))
    assert torch.allclose(collected.detach(), torch.tensor(metrics["foundation_router_loss"]))
    total.sum().backward()
    latent = wrapper.student_model.model[2]
    assert latent.router.expert_head.weight.grad is not None
    assert all("route_teachers" not in key for key in wrapper.state_dict())


def test_route_teacher_heads_are_training_only_and_strip_with_wrapper_copy():
    wrapper = FoundationDistillationModel(RoutedTinyStudent(), DummyTeacher(), route_config())
    wrapper.train()
    wrapper({"img": torch.rand(2, 3, 64, 64)})
    copied = copy.deepcopy(wrapper)
    assert copied.__dict__.get("_route_teachers") == {}
    assert copied.teacher_manager is None
    assert all("route_teacher" not in key for key in copied.state_dict())


def test_route_checkpoint_metadata_describes_static_latent_routes():
    wrapper = FoundationDistillationModel(RoutedTinyStudent(), DummyTeacher(), route_config())
    metadata = checkpoint_runtime_metadata(wrapper)["foundation"]
    assert metadata["router_kind"] == "latent_mixture_image_level"
    assert len(metadata["router_specs"]) == 1
    assert metadata["router_specs"][0]["num_experts"] == 3


def test_route_temperature_must_be_positive():
    with pytest.raises(ValueError, match="router_temperature"):
        FoundationDistillationModel(RoutedTinyStudent(), DummyTeacher(), route_config(foundation_router_temperature=0))
