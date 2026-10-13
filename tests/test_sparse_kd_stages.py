"""Semantic and gradient contracts for the staged sparse distillation experiment."""

import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.evaluate_router_choices import RouterChoice
from scripts.finetune_sparse_detector import STAGE_MODES, train_target
from scripts.prepare_sparse_supervised_data import select_trainval_splits
from scripts.sparse_kd_losses import (
    feature_distillation,
    localization_distillation,
    oracle_pair_targets,
    region_weights,
    router_distillation,
)
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel


def batch():
    return {
        "img": torch.zeros(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }


def test_official_train_val_sources_are_seeded_disjoint_and_exclude_history():
    train, val = [f"t{i}" for i in range(10)], [f"v{i}" for i in range(10)]
    result = select_trainval_splits(train, val, {"t0", "v0"}, (5, 3, 3), 42)
    assert result == select_trainval_splits(train[::-1], val[::-1], {"t0", "v0"}, (5, 3, 3), 42)
    assert set(result["train"]) <= set(train) - {"t0"}
    assert set(result["dev"] + result["test"]) <= set(val) - {"v0"}
    assert not set(result["dev"]) & set(result["test"])
    with pytest.raises(ValueError):
        select_trainval_splits(train, train, set(), (5, 3, 3), 42)


def test_region_weights_small_boxes_overlap_and_empty_background():
    data = batch()
    weights = region_weights(data, (8, 8))
    assert weights[0, 0, 3:5, 3:5].eq(8).all()
    assert weights[0, 0, 0, 0] == 1
    data["bboxes"] = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 1.0, 1.0]])
    data["batch_idx"] = torch.zeros(2)
    weights = region_weights(data, (8, 8))
    assert weights[0, 0, 0, 0] == 4
    assert weights[0, 0, 3, 3] == 8
    data["bboxes"] = torch.empty(0, 4)
    data["batch_idx"] = torch.empty(0)
    assert region_weights(data, (8, 8)).eq(1).all()


def test_weighted_feature_loss_prioritizes_foreground_and_detaches_teacher():
    teacher = torch.ones(1, 1, 2, 2, requires_grad=True)
    student = teacher.detach().clone().requires_grad_()
    with torch.no_grad():
        student[0, 0, 0, 0] += 1
    plain = feature_distillation(student, teacher)
    torch.testing.assert_close(plain, torch.tensor(0.25))
    weighted = feature_distillation(student, teacher, torch.tensor([[[[8.0, 1.0], [1.0, 1.0]]]]))
    assert weighted > plain
    weighted.backward()
    assert student.grad is not None and teacher.grad is None


def test_localization_identity_gradient_and_empty_selection():
    data = batch()
    target = torch.randn(1, 16, 4, requires_grad=True)
    prediction = target.detach().clone().requires_grad_()
    teacher = {"boxes": target, "scores": torch.zeros(1, 2, 4), "feats": [torch.zeros(1, 8, 2, 2)]}
    student = {**teacher, "boxes": prediction}
    identity = localization_distillation(student, teacher, data)
    assert abs(identity.item()) < 1e-6
    student["boxes"] = torch.randn(1, 16, 4, requires_grad=True)
    loss = localization_distillation(student, teacher, data)
    assert loss > 0
    loss.backward()
    assert student["boxes"].grad.abs().sum() > 0 and target.grad is None
    data["bboxes"] = torch.empty(0, 4)
    data["batch_idx"] = torch.empty(0)
    teacher["scores"] = torch.full((1, 2, 4), -100.0)
    assert localization_distillation(student, teacher, data).item() == 0
    student["feats"] = [torch.zeros(1, 8, 1, 4)]
    with pytest.raises(ValueError, match="grids"):
        localization_distillation(student, teacher, data)


def test_oracle_pairs_and_raw_router_gradient_reach_low_probability_expert():
    block = SimpleNamespace(training=False, num_experts=3, norm=torch.nn.Identity())
    experts = [torch.full((1, 1, 2, 2), x) for x in (0.0, 2.0, 10.0)]
    routing = torch.full((1, 3, 1, 1), 1 / 3)
    for expected, value in enumerate((1.0, 5.0, 6.0)):
        target = torch.full((1, 1, 2, 2), value)
        labels = oracle_pair_targets(block, experts, routing, target)
        assert labels.tolist() == [expected]  # Each pair can win; no hard-coded first-pair fallback.
    logits = torch.tensor([[[[3.0]], [[2.0]], [[-3.0]]]], requires_grad=True)
    loss, _ = router_distillation(logits.softmax(1), torch.tensor([1]))  # Desired support (0, 2).
    loss.backward()
    assert logits.grad[0, 2, 0, 0] < 0


@pytest.mark.parametrize("mode", list(STAGE_MODES.values()))
def test_real_stages_update_only_target_keep_teacher_bn_frozen_and_remove_hooks(mode):
    torch.manual_seed(42)
    cfg = {
        "nc": 2,
        "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "ES_MOE", [8, 3, 8, 3]]],
        "head": [[[1], 1, "Detect", [2]]],
    }
    model = DetectionModel(cfg, verbose=False).eval()
    model.args = get_cfg()
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    data = batch()
    data["img"] = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    with RouterChoice(model.model[1], "native_top2") as route:
        route.active = True
        history = train_target(model, teacher, "model.1", [data], 1, 1e-3, 1, detection_weight=0, kd_mode=mode)
        assert route.summary()["executed_experts_per_sample"] == 2
    changed = [key for key, value in model.state_dict().items() if not torch.equal(value, before[key])]
    assert changed and all(key.startswith("model.1.") for key in changed)
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(not module._forward_hooks for module in model.modules())
    assert all(not module._forward_hooks for module in teacher.modules())
    assert history[0]["detection_loss"] == 0
    if "localization" in mode:
        assert history[0]["localization_kd_loss"] >= 0
    if "router" in mode:
        assert sum(history[0]["oracle_pair_counts"]) == 1
        assert history[0]["router_kd_loss"] > 0
