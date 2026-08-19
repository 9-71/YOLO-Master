"""Tests for the reproducible F15 paired training-signal benchmark."""

import json
import os
from pathlib import Path

import pytest

from scripts.foundation_f15_paired_benchmark import run_benchmark


def _local_dinov3_snapshot() -> str:
    """Resolve a local DINOv3-ViT-S16 snapshot: env override first, then the standard HF cache."""
    override = os.environ.get("YOLO_MASTER_DINOV3_PATH")
    if override:
        return override
    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Tooony133--dinov3-vits16-pretrain-lvd1689m"
        / "snapshots"
    )
    snapshots = sorted(p for p in snapshots_dir.glob("*") if p.is_dir()) if snapshots_dir.is_dir() else []
    return str(snapshots[-1]) if snapshots else ""


@pytest.mark.slow
def test_f15_paired_benchmark_records_foundation_effects(tmp_path):
    """Baseline/Foundation branches share initialization and expose the F15 gate."""
    local_dinov3 = _local_dinov3_snapshot()
    if not local_dinov3:
        pytest.skip("local DINOv3 cache unavailable: set YOLO_MASTER_DINOV3_PATH to a local snapshot")
    try:
        result = run_benchmark(local_dinov3, steps=2, align_dim=8)
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
