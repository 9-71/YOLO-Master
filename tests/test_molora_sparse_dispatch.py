import torch
import torch.nn as nn

from ultralytics.nn.peft.molora.layer import MoLoRALayer


def test_molora_grouped_dispatch_records_actual_calls():
    torch.manual_seed(0)
    layer = MoLoRALayer(nn.Linear(16, 16), r=2, num_experts=4, top_k=1).eval()
    layer(torch.randn(8, 16))
    stats = layer._last_dispatch_stats
    assert stats["mode"] == "grouped_sparse"
    assert 1 <= stats["expert_calls"] <= 4
    assert stats["selected_samples"] == 8


def test_molora_small_batch_keeps_dense_fast_path():
    layer = MoLoRALayer(nn.Linear(16, 16), r=2, num_experts=4, top_k=1).eval()
    layer(torch.randn(2, 16))
    assert layer._last_dispatch_stats["mode"] == "dense_small_batch"


def test_molora_all_linear_experts_use_batched_dispatch_without_changing_values():
    torch.manual_seed(1)
    layer = MoLoRALayer(nn.Linear(8, 6), r=2, num_experts=3, top_k=3).eval()
    x = torch.randn(5, 8)
    weights = torch.softmax(torch.randn(5, 3), dim=-1)
    indices = torch.tensor([[0, 1, 2]]).expand(5, -1)

    batched = layer._compute_sparse_experts(x, weights, indices, torch.zeros(5, 6))
    reference = sum(weights[:, i : i + 1] * layer.experts[i](x) for i in range(3))

    assert layer._last_dispatch_stats["mode"] == "batched_dense_linear"
    assert torch.allclose(batched, reference, atol=1e-6, rtol=1e-5)
