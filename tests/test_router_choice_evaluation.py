"""Counterfactual expert choices must execute the chosen expert and restore native behavior."""

import pytest
import torch

from scripts.evaluate_router_choices import RouterChoice, prepare_subset, summarize_choices
from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.mot import MoTBlock


@pytest.mark.parametrize("kind", ["moe", "mot"])
def test_fixed_expert_intervention_restores_native_output_and_flags(kind):
    torch.manual_seed(0)
    block = (ES_MOE(8, 8, num_experts=3, top_k=3) if kind == "moe" else MoTBlock(8, num_heads=2, top_k=2)).eval()
    x = torch.randn(2, 8, 4, 6)
    with torch.inference_mode():
        original = block(x)
        baseline = original[0] if kind == "mot" else original
        top_k = block.top_k
        with RouterChoice(block, "fixed:1") as choice:
            choice.active = True
            block(x)
            assert choice.observed["sample_expert_pairs"] == [0, 2, 0]
            assert choice.observed["router_samples"] == 2
            assert choice.summary()["mean_router_weight"] == [0, 1, 0]
        assert block.top_k == top_k
        restored = block(x)
        torch.testing.assert_close(restored[0] if kind == "mot" else restored, baseline)


def test_intervention_exception_removes_hooks_and_restores_flags():
    block = ES_MOE(8, 8, num_experts=3, top_k=None).eval()
    with pytest.raises(RuntimeError, match="fixture"), RouterChoice(block, "native_top1"):
        raise RuntimeError("fixture")
    assert block.top_k == 3
    assert not block.use_top_k
    assert not block.routing._forward_hooks
    assert all(not expert._forward_pre_hooks for expert in block.experts)


def test_legacy_checkpoint_missing_sparse_flag_gets_temporary_override():
    block = ES_MOE(8, 8, num_experts=3, top_k=None).eval()
    del block.use_top_k
    with RouterChoice(block, "fixed:2") as choice, torch.inference_mode():
        choice.active = True
        block(torch.randn(1, 8, 4, 6))
        assert choice.observed["sample_expert_pairs"] == [0, 0, 1]
    assert not hasattr(block, "use_top_k")


@pytest.mark.parametrize("kind", ["moe", "mot"])
def test_uniform_control_runs_all_experts_with_equal_weights(kind):
    block = (ES_MOE(8, 8, num_experts=3, top_k=1) if kind == "moe" else MoTBlock(8, num_heads=2, top_k=1)).eval()
    with RouterChoice(block, "uniform") as choice, torch.inference_mode():
        choice.active = True
        block(torch.randn(2, 8, 4, 6))
        assert choice.observed["sample_expert_pairs"] == [2, 2, 2]
        assert choice.summary()["mean_router_weight"] == pytest.approx([1 / 3] * 3)
    assert block.top_k == 1


@pytest.mark.parametrize("kind", ["moe", "mot"])
def test_top2_control_keeps_two_experts_and_restores_threshold(kind):
    block = (
        ES_MOE(8, 8, num_experts=3, top_k=3, dynamic_threshold=0.9)
        if kind == "moe"
        else MoTBlock(8, num_heads=2, top_k=3, use_spatial_router=False)
    ).eval()
    with RouterChoice(block, "native_top2") as choice, torch.inference_mode():
        choice.active = True
        block(torch.randn(2, 8, 4, 6))
        assert sum(choice.observed["sample_expert_pairs"]) == 4
        assert sum(choice.summary()["mean_router_weight"]) == pytest.approx(1.0)
    assert block.top_k == 3
    if kind == "moe":
        assert block.dynamic_threshold == 0.9


@pytest.mark.parametrize("policy", ["fixed:3", "fixed:-1", "unknown"])
def test_invalid_policy_rejected(policy):
    with pytest.raises(ValueError):
        RouterChoice(ES_MOE(8, 8, num_experts=3).eval(), policy)


def test_summary_keeps_failed_cases_out_of_comparison():
    cases = [
        {"policy": "native", "status": "ok", "metrics": {"metrics/mAP50-95(B)": 0.5}},
        {"policy": "fixed:0", "status": "ok", "metrics": {"metrics/mAP50-95(B)": 0.4}},
        {"policy": "fixed:1", "status": "failed"},
    ]
    result = summarize_choices(cases)["map50_95_delta_points_vs_native"]
    assert result == pytest.approx({"native": 0, "fixed:0": -10})


def test_baseline_repeat_detects_metric_drift():
    cases = [
        {"policy": "native", "status": "ok", "metrics": {"metrics/mAP50-95(B)": 0.5}},
        {"policy": "native_repeat", "status": "ok", "metrics": {"metrics/mAP50-95(B)": 0.4}},
    ]
    result = summarize_choices(cases)
    assert not result["baseline_repeat_within_tolerance"]
    assert result["baseline_repeat_max_metric_abs_delta"] == pytest.approx(0.1)


def test_subset_fingerprints_image_and_label_changes(tmp_path, monkeypatch):
    images = tmp_path / "images/val"
    labels = tmp_path / "labels/val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "b.jpg").write_bytes(b"fixture-b")
    (images / "a.jpg").write_bytes(b"fixture-a")
    label = labels / "a.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n")
    monkeypatch.setattr(
        "ultralytics.data.utils.check_det_dataset",
        lambda data, autodownload: {"val": str(images), "names": {0: "object"}},
    )
    output = tmp_path / "output"
    output.mkdir()
    _, before = prepare_subset("fixture.yaml", "val", 1, output)
    label.write_text("0 0.5 0.5 0.4 0.4\n")
    _, after = prepare_subset("fixture.yaml", "val", 1, output)
    assert before[0]["image"].endswith("a.jpg")
    assert before[0]["image_sha256"] == after[0]["image_sha256"]
    assert before[0]["label_sha256"] != after[0]["label_sha256"]
