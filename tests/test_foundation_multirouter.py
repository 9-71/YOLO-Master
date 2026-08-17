"""F14 Multi-Foundation Router contracts."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.test_foundation_distill_model import DummyTeacher, config
from tests.test_foundation_routing_contract import RoutedTinyStudent
from ultralytics.cfg import get_cfg
from ultralytics.nn.foundation import FoundationFeatures, MultiFoundationTeacher, foundation_multiteacher_summary
from ultralytics.nn.foundation_distill_model import (
    FoundationDistillationModel,
    rebuild_foundation_distillation_wrapper,
    strip_foundation_distillation_model,
)


class DummySigLIPTeacher(nn.Module):
    """Offline SigLIP2-shaped teacher with dense and semantic outputs."""

    name = "siglip2"

    def __init__(self, channels: int = 6):
        super().__init__()
        self.anchor = nn.Parameter(torch.arange(channels, dtype=torch.float32) + 1.0)
        self.calls = 0
        self.freeze_calls = 0
        self.hidden_size = channels

    def freeze(self):
        self.freeze_calls += 1
        self.eval()
        self.anchor.requires_grad_(False)

    def encode(self, images):
        self.calls += 1
        dense = self.anchor.view(1, -1, 1, 1).expand(images.shape[0], -1, 2, 2)
        pooled = dense.mean((2, 3))
        semantic = torch.nn.functional.normalize(pooled, dim=-1)
        return FoundationFeatures(dense={"p4": dense}, pooled=pooled, semantic=semantic)

    def encode_text(self, prompts):
        return torch.ones(len(prompts), self.anchor.numel())


def multi_config(**overrides):
    values = vars(
        config(
            foundation_teacher="multi",
            foundation_router_distill=True,
            foundation_router_loss_weight=0.2,
            foundation_router_temperature=2.0,
        )
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_multi_wrapper(**overrides):
    teacher = MultiFoundationTeacher(DummyTeacher(), DummySigLIPTeacher())
    wrapper = FoundationDistillationModel(RoutedTinyStudent(), teacher, multi_config(**overrides))
    wrapper.train()
    return wrapper


def test_multiteacher_summary_separates_spatial_and_semantic_blocks():
    dino = FoundationFeatures(dense={"p4": torch.randn(2, 4, 2, 2)}, pooled=torch.randn(2, 4))
    siglip = FoundationFeatures(
        dense={"p4": torch.randn(2, 6, 2, 2)},
        pooled=torch.randn(2, 6),
        semantic=torch.randn(2, 6),
    )
    summary = foundation_multiteacher_summary({"dinov3": dino, "siglip2": siglip})
    assert summary.shape == (2, 3 * 4 + 3 * 6 + 2)
    assert summary.isfinite().all()
    assert not summary.requires_grad


def test_multirouter_training_freezes_both_teachers_and_routes_gradients():
    wrapper = make_multi_wrapper()
    total, _ = wrapper({"img": torch.rand(2, 3, 64, 64)})
    metrics = wrapper.foundation_metrics()
    assert metrics["foundation_router_modules"] == 1.0
    assert metrics["foundation_router_loss"] > 0
    total.sum().backward()
    latent = wrapper.student_model.model[2]
    assert latent.router.expert_head.weight.grad is not None
    manager = wrapper.teacher_manager
    assert manager is not None
    assert manager.dinov3.training is False and manager.siglip2.training is False
    assert all(not parameter.requires_grad for parameter in manager.dinov3.parameters())
    assert all(not parameter.requires_grad for parameter in manager.siglip2.parameters())
    assert all("route_teacher" not in key and "_teacher_manager" not in key for key in wrapper.state_dict())


def test_multirouter_metadata_resume_and_export_strip():
    wrapper = make_multi_wrapper()
    wrapper({"img": torch.rand(2, 3, 64, 64)})
    metadata = wrapper.checkpoint_metadata()
    assert metadata["router_kind"] == "multi_foundation_image_level"
    assert metadata["router_teachers"] == ["dinov3", "siglip2"]
    assert metadata["router_native_state"] is True
    assert metadata["router_input_dims"]["teacher"]

    copied = copy.deepcopy(wrapper)
    assert copied.teacher_manager is None
    assert copied.checkpoint_metadata()["router_kind"] == "multi_foundation_image_level"
    rebuilt = rebuild_foundation_distillation_wrapper(
        RoutedTinyStudent(),
        multi_config(),
        checkpoint_model=copied,
        teacher_manager=MultiFoundationTeacher(DummyTeacher(), DummySigLIPTeacher()),
    )
    assert isinstance(rebuilt, FoundationDistillationModel)
    assert rebuilt.checkpoint_metadata()["router_kind"] == "multi_foundation_image_level"
    stripped = strip_foundation_distillation_model(rebuilt)
    assert stripped is rebuilt.student_model if isinstance(rebuilt, FoundationDistillationModel) else True


def test_multirouter_requires_both_named_teachers_and_semantic_capability():
    with pytest.raises(ValueError, match="both named outputs"):
        foundation_multiteacher_summary({"dinov3": FoundationFeatures(dense={"p4": torch.randn(1, 4, 2, 2)})})
    with pytest.raises(ValueError, match="semantic features"):
        foundation_multiteacher_summary(
            {
                "dinov3": FoundationFeatures(dense={"p4": torch.randn(1, 4, 2, 2)}),
                "siglip2": FoundationFeatures(dense={"p4": torch.randn(1, 6, 2, 2)}),
            }
        )


def test_builder_rejects_single_injected_teacher_for_multi():
    with pytest.raises((TypeError, ValueError), match="dinov3.*siglip2|both"):
        from ultralytics.nn.foundation_distill_model import build_foundation_distillation_wrapper

        build_foundation_distillation_wrapper(
            RoutedTinyStudent(), multi_config(), teacher_manager={"dinov3": DummyTeacher()}
        )


def test_config_restricts_f14_teacher_order_and_requires_router():
    with pytest.raises(ValueError, match="requires foundation_router_teachers"):
        get_cfg(
            overrides={
                "foundation_enabled": True,
                "foundation_teacher": "multi",
                "foundation_model": "dinov3-local",
                "foundation_router_distill": True,
                "foundation_router_teachers": ["siglip2", "dinov3"],
                "foundation_loss_weight": 0.0,
                "foundation_router_loss_weight": 0.1,
            }
        )
    with pytest.raises(ValueError, match="requires foundation_router_distill"):
        get_cfg(
            overrides={
                "foundation_enabled": True,
                "foundation_teacher": "multi",
                "foundation_model": "dinov3-local",
                "foundation_loss_weight": 1.0,
            }
        )
