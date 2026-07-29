"""Production MoE diagnostics and pruning contract tests."""

import torch

from ultralytics.nn.modules.moe.diagnostics import routing_runtime_metrics
from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.moe.pruning import MoEPruner


def test_runtime_metrics_reports_usage_gini_and_dispatch_mode():
    module = ES_MOE(16, 16, num_experts=3, top_k=1).train()
    with torch.no_grad():
        for _ in range(10):
            module(torch.randn(4, 16, 8, 8))
    metrics = routing_runtime_metrics(module)
    assert metrics["routed_layers"] >= 1
    assert metrics["expert_calls"] >= 1
    row = next(iter(metrics["layers"].values()))
    assert len(row["expert_usage"]) == 3
    assert 0.0 <= row["gini"] <= 1.0
    assert row["dispatch_mode"] in {"sparse", "dense", "grouped_sparse"}


def test_pruning_plan_keep_top_m_is_deterministic():
    pruner = MoEPruner("dummy.pt", keep_top_m=2)
    pruner.usage_stats = {
        "0.routing": {
            0: type("S", (), {"hits": 1.0, "avg_weight": 0.2})(),
            1: type("S", (), {"hits": 8.0, "avg_weight": 0.5})(),
            2: type("S", (), {"hits": 4.0, "avg_weight": 0.3})(),
        }
    }
    pruner.model = type("M", (), {"model": ES_MOE(16, 16, num_experts=3, top_k=1)})()
    pruner._create_pruning_plan()
    assert pruner.pruning_plan["0.routing"] == [1, 2]
