"""Held-out selection and actual detection-gradient sparse training contracts."""

import copy
import hashlib
from pathlib import Path

import pytest
import torch

from scripts.evaluate_router_choices import RouterChoice
from scripts.finetune_sparse_detector import choose_candidate, train_target, verify_manifest
from scripts.prepare_sparse_supervised_data import select_splits
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel


def test_split_assignment_is_seeded_disjoint_and_excludes_prior_images():
    names = [f"{index:012}.jpg" for index in range(12)]
    splits = select_splits(names, names[:2], (4, 3, 3), 42)
    assert splits == select_splits(list(reversed(names)), names[:2], (4, 3, 3), 42)
    flattened = [name for split in splits.values() for name in split]
    assert len(set(flattened)) == 10
    assert not set(flattened) & set(names[:2])
    with pytest.raises(ValueError):
        select_splits(names, names[:2], (8, 3, 3), 42)


def test_candidate_choice_ignores_test_metrics_and_dense_control():
    cases = [
        {"name": "sparse_det", "dev_ap": 0.4, "test_ap": 0.8},
        {"name": "sparse_det_kd", "dev_ap": 0.5, "test_ap": 0.1},
        {"name": "dense_det", "dev_ap": 0.9},
    ]
    assert choose_candidate(cases) == "sparse_det_kd"
    cases[1]["dev_ap"] = float("nan")
    with pytest.raises(ValueError):
        choose_candidate(cases)


def test_manifest_rejects_label_mutation_and_cross_split_content_overlap(tmp_path):
    splits = {}
    for index, name in enumerate(("train", "dev", "test")):
        image, label = tmp_path / f"{name}.jpg", tmp_path / f"{name}.txt"
        image.write_bytes(bytes([index]))
        label.write_text("0 0.5 0.5 0.2 0.2")
        splits[name] = [
            {
                "image": str(image),
                "label": str(label),
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
            }
        ]
    verify_manifest({"splits": splits})
    label = Path(splits["train"][0]["label"])
    label.write_text("1 0.5 0.5 0.2 0.2")
    with pytest.raises(ValueError, match="label identity changed"):
        verify_manifest({"splits": splits})
    splits["train"][0]["label_sha256"] = hashlib.sha256(label.read_bytes()).hexdigest()
    splits["dev"][0]["image"] = splits["train"][0]["image"]
    splits["dev"][0]["image_sha256"] = splits["train"][0]["image_sha256"]
    with pytest.raises(ValueError, match="overlap"):
        verify_manifest({"splits": splits})


@pytest.mark.parametrize(
    "kd_weight, detection_weight, batch_size", [(0.0, 1.0, 1), (1.0, 1.0, 1), (1.0, 0.0, 1), (1.0, 1.0, 4)]
)
def test_real_detection_loss_updates_only_sparse_target_and_keeps_bn_frozen(
    kd_weight, detection_weight, batch_size, monkeypatch
):
    torch.manual_seed(42)
    cfg = {
        "nc": 2,
        "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "ES_MOE", [8, 3, 8, 3]]],
        "head": [[[1], 1, "Detect", [2]]],
    }
    model = DetectionModel(cfg, verbose=False).eval()
    model.args = get_cfg()
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    before = {name: p.clone() for name, p in model.named_parameters()}
    if not detection_weight:

        def reject_detection_loss(*args):
            raise AssertionError("pure KD must not consume detection labels")

        monkeypatch.setattr(model, "loss", reject_detection_loss)
    batch = {
        "img": torch.randint(0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8),
        "batch_idx": torch.arange(batch_size).float(),
        "cls": torch.zeros(batch_size, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]).repeat(batch_size, 1),
    }
    with RouterChoice(model.model[1], "native_top2") as routing:
        routing.active = True
        history = train_target(
            model,
            teacher if kd_weight else None,
            "model.1",
            [batch],
            1,
            1e-3,
            kd_weight,
            detection_weight=detection_weight,
        )
        assert routing.summary()["executed_experts_per_sample"] == 2
    if detection_weight:
        assert history[0]["detection_loss"] > 0
    else:
        assert history[0]["detection_loss"] == 0
    changed = [name for name, p in model.named_parameters() if not torch.equal(p, before[name])]
    assert changed and all(name.startswith("model.1.") for name in changed)
    for name, p in teacher.named_parameters():
        torch.testing.assert_close(p, before[name])
    assert not model.model[1]._forward_hooks
    assert not teacher.model[1]._forward_hooks


def test_three_arm_selection_requires_every_declared_development_score():
    names = ("sparse_det", "sparse_kd", "sparse_det_kd")
    cases = [{"name": name, "dev_ap": score} for name, score in zip(names, (0.4, 0.6, 0.5))]
    assert choose_candidate(cases, names) == "sparse_kd"
    with pytest.raises(ValueError, match="present and finite"):
        choose_candidate(cases[:2], names)


@pytest.mark.parametrize("det_weight, kd_weight", [(0, 0), (-1, 1), (1, -1)])
def test_invalid_loss_weights_fail_before_mutating_model(det_weight, kd_weight):
    with pytest.raises(ValueError, match="loss weights"):
        train_target(None, None, "model.1", [], 1, 1e-3, kd_weight, detection_weight=det_weight)
