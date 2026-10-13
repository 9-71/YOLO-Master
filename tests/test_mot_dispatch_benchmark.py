"""Guard the distinction between logical routing and executed expert work."""

import pytest
import torch

from benchmarks.benchmark_mot_dispatch import run_case
from ultralytics.nn.modules.mot import MoTBlock


@pytest.mark.parametrize(
    ("pattern", "calls", "pairs"),
    [("single", 1, 3), ("sample_split", 3, 3), ("spatial_union", 3, 9)],
)
def test_matched_dispatch_executes_expected_sample_expert_pairs(pattern, calls, pairs):
    torch.manual_seed(0)
    block = MoTBlock(8, num_heads=2, top_k=1).eval()
    result = run_case(block, torch.randn(3, 8, 4, 6), pattern, warmup=0, steps=1)
    sparse = result["modes"]["sample_sparse"]
    dense = result["modes"]["dense_masked_reference"]
    assert sparse["expert_calls"] == calls
    assert sparse["sample_expert_pairs"] == pairs
    assert dense["expert_calls"] == 3
    assert dense["sample_expert_pairs"] == 9
    assert result["token_mask_sparsity"] == pytest.approx(2 / 3)
    assert result["sample_expert_pair_reduction"] == pytest.approx(1 - pairs / 9)
    assert result["max_abs_error"] < 1e-5


def test_dispatch_diagnostic_rejects_training_mode():
    with pytest.raises(ValueError, match="eval"):
        run_case(MoTBlock(8, num_heads=2), torch.randn(1, 8, 4, 6), "single")
