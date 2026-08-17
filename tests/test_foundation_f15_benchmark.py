"""Tests for the reproducible F15 paired training-signal benchmark."""

import json

import pytest

from scripts.foundation_f15_paired_benchmark import run_benchmark


LOCAL_DINOV3 = (
    "/Users/gatilin/.cache/huggingface/hub/models--Tooony133--dinov3-vits16-pretrain-lvd1689m/"
    "snapshots/fc6921f7a0b44d5b33ab4482cfed5443db6ccd81"
)


@pytest.mark.slow
def test_f15_paired_benchmark_records_foundation_effects(tmp_path):
    """Baseline/Foundation branches share initialization and expose the F15 gate."""
    try:
        result = run_benchmark(LOCAL_DINOV3, steps=2, align_dim=8)
    except OSError as exc:
        pytest.skip(f"local DINOv3 cache unavailable: {exc}")
    assert result["synthetic_batch"] is True
    assert result["real_accuracy_claim"] is False
    assert result["summary"]["foundation_supervised_task_gate"] is True
    assert len(result["baseline"]) == len(result["foundation"]) == 2
    assert all(row["foundation_loss"] > 0 for row in result["foundation"])
    assert all(row["supervised_tasks"] >= 2 for row in result["foundation"])
    assert all(row["p4_grad_norm"] > 0 for row in result["foundation"])
    assert result["summary"]["foundation_mean_step_s"] > 0

    json.dumps(result)
