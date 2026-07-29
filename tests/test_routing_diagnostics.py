"""Routing diagnostics, capability declarations, and wrapper propagation tests."""

import pytest
import torch

from ultralytics.nn.modules.moa import C2fMoA, MoABlock
from ultralytics.nn.modules.moa.router import _moa_router_aux_loss
from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.utils.errors import MoERouterError


def test_nonfinite_moa_aux_preserves_finite_graph_and_reports_boundary():
    weights = torch.full((1, 3, 2, 2), 1.0 / 3.0, requires_grad=True)
    logits = torch.full((1, 3, 2, 2), float("nan"))

    loss, diagnostics = _moa_router_aux_loss(weights, logits, 0.01, return_diagnostics=True)

    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert diagnostics["first_nonfinite_boundary"] == "router_logits"
    assert diagnostics["logits_nonfinite_count"] == logits.numel()
    loss.backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


def test_moa_block_snapshot_keeps_pre_fallback_nonfinite_diagnostics(monkeypatch):
    block = MoABlock(24, num_heads=3).train()

    def bad_router(x, return_logits=False):
        weights = x[:, :1].repeat(1, 3, 1, 1) * 0.0 + (1.0 / 3.0)
        logits = torch.full_like(weights, float("nan"))
        return (weights, logits) if return_logits else weights

    monkeypatch.setattr(block.router, "forward", bad_router)
    _ = block(torch.randn(1, 24, 4, 4, requires_grad=True))

    diagnostics = block.last_routing_snapshot["finite_diagnostics"]
    assert diagnostics["first_nonfinite_boundary"] == "router_logits"
    assert diagnostics["logits_finite"] is False
    assert torch.isfinite(block.aux_loss)


def test_es_moe_preserves_router_failure_diagnostics():
    module = ES_MOE(16, 16, num_experts=3, top_k=1).train()
    with torch.no_grad():
        module.routing.routing_network[-1].bias.fill_(float("nan"))

    with pytest.raises(MoERouterError, match="internal output"):
        module(torch.randn(1, 16, 4, 4))

    diagnostics = module.last_routing_diagnostics
    assert diagnostics["first_nonfinite_boundary"] == "router_logits"
    assert diagnostics["logits_finite"] is False


def test_routed_modules_declare_sparse_export_boundary():
    moa = MoABlock(24, num_heads=3).export_capabilities()
    mot = MoTBlock(24, num_heads=3, top_k=2).export_capabilities()
    moe = ES_MOE(16, 16, num_experts=3, top_k=1).export_capabilities()

    assert moa["routing_kind"] == "moa"
    assert moa["eager_sparse_dispatch"] is False
    for capabilities in (mot, moe):
        assert capabilities["eager_sparse_dispatch"] is True
        assert capabilities["onnx_sparse_dispatch"] is False
        assert capabilities["torchscript_trace_sparse_dispatch"] is False
        assert capabilities["exact_sparse_export"] is False
        assert "dense" in capabilities["sparse_export_limitation"].lower()


def test_c2f_moa_propagates_sequential_head_configuration():
    module = C2fMoA(48, 48, n=3, num_heads=3, sequential_heads=True)

    assert len(module.m) == 3
    assert all(block.sequential_heads for block in module.m)
