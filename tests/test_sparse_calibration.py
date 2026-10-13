"""Data isolation and statistic-only calibration safeguards."""

import copy

import pytest
import torch

from scripts.calibrate_sparse_esmoe import assert_disjoint, distill_block, recalibrate_bn
from scripts.evaluate_router_choices import RouterChoice
from ultralytics.nn.modules.moe.modules import ES_MOE


@pytest.mark.parametrize("image, digest", [("/other/a.jpg", "other"), ("/other/b.jpg", "same")])
def test_overlap_rejected_by_identity_or_content(image, digest):
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint([{"image": image, "image_sha256": digest}], [{"image": "/eval/a.jpg", "image_sha256": "same"}])


def test_bn_changes_statistics_only_and_restores_modes(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.BatchNorm2d(3)).eval()
    parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    monkeypatch.setattr("scripts.calibrate_sparse_esmoe.image_tensor", lambda *args: torch.ones(1, 3, 8, 8))
    recalibrate_bn(model, ["a", "b"], 8, "cpu")
    assert model[1].num_batches_tracked == 2
    assert all(not m.training for m in model.modules())
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, parameters[name])


def test_sparse_distillation_updates_target_without_bn_or_teacher_mutation(monkeypatch):
    torch.manual_seed(4)
    teacher = torch.nn.Sequential(ES_MOE(3, 3, num_experts=3, top_k=3)).eval().requires_grad_(False)
    student = copy.deepcopy(teacher)
    original = copy.deepcopy(teacher.state_dict())
    monkeypatch.setattr("scripts.calibrate_sparse_esmoe.image_tensor", lambda *args: torch.ones(1, 3, 8, 8))
    with RouterChoice(student[0], "native_top2"):
        losses = distill_block(teacher, student, "0", ["a", "b"], 8, "cpu", 2, 1e-3)
    assert len(losses) == 2 and all(torch.isfinite(torch.tensor(losses)))
    assert any(not torch.equal(p, original[k]) for k, p in student.state_dict().items() if p.is_floating_point())
    for name, p in teacher.state_dict().items():
        torch.testing.assert_close(p, original[name])
    for name, p in student.state_dict().items():
        if "running_" in name or "num_batches_tracked" in name:
            torch.testing.assert_close(p, original[name])
    assert not teacher[0]._forward_hooks and not student[0]._forward_pre_hooks


def test_bn_scope_leaves_upstream_statistics_unchanged(monkeypatch):
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.BatchNorm2d(3)).eval()
    upstream = copy.deepcopy(model[0].state_dict())
    monkeypatch.setattr("scripts.calibrate_sparse_esmoe.image_tensor", lambda *args: torch.ones(1, 3, 8, 8))
    recalibrate_bn(model, ["a", "b"], 8, "cpu", scope=model[1])
    assert model[1].num_batches_tracked == 2
    for key, value in model[0].state_dict().items():
        torch.testing.assert_close(value, upstream[key])
