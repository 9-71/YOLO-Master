"""
MoE Unified API Compatibility Layer (P1-1)

Provides a consistent interface across all STABLE MoE classes without
modifying their internals. This allows downstream code (trainers, callbacks,
diagnostics) to interact with any MoE variant through a single API.

Key unified properties:
    - get_expert_usage() → Tensor
    - get_aux_loss() → Tensor
    - get_routing_weights() → Tensor | None
    - get_expert_module() → nn.Module (the expert container)
    - reset_runtime_state() → None

Usage:
    from ultralytics.nn.modules.moe.api import moe_info, get_aux_loss_unified
    for name, m in model.named_modules():
        if is_core_moe_block(m):
            info = moe_info(m)
            loss = get_aux_loss_unified(m)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .utils import is_core_moe_block


@dataclass
class MoEInfo:
    """Unified info snapshot from any MoE module."""

    class_name: str
    num_experts: int
    top_k: int
    has_routing: bool
    has_experts: bool
    expert_backend: str  # "experts", "fused_experts", or "unknown"
    has_balance_loss: bool
    aux_loss_value: float
    expert_usage: list[float]


def _get_expert_module(m: nn.Module) -> nn.Module | None:
    """Return the expert container module, regardless of attribute name."""
    for attr in ("experts", "fused_experts", "expert_group"):
        mod = getattr(m, attr, None)
        if mod is not None:
            return mod
    return None


def _get_routing_module(m: nn.Module) -> nn.Module | None:
    """Return the routing module, regardless of attribute name."""
    for attr in ("routing", "router", "gate"):
        mod = getattr(m, attr, None)
        if mod is not None and isinstance(mod, nn.Module):
            return mod
    return None


def get_aux_loss_unified(m: nn.Module) -> torch.Tensor:
    """Get auxiliary loss from any MoE module.

    Tries multiple attribute names for compatibility.
    """
    # Direct attribute access
    for attr in ("aux_loss", "moe_aux_loss", "_aux_loss"):
        val = getattr(m, attr, None)
        if val is not None:
            return val if torch.is_tensor(val) else torch.tensor(float(val))
    # Via MoELoss wrapper
    moe_loss_fn = getattr(m, "moe_loss_fn", None)
    if moe_loss_fn is not None:
        last_loss = getattr(moe_loss_fn, "last_loss", None)
        if last_loss is not None:
            return last_loss if torch.is_tensor(last_loss) else torch.tensor(float(last_loss))
    # Via load_balancing_loss buffer
    lb = getattr(m, "load_balancing_loss", None)
    if lb is not None:
        return lb if torch.is_tensor(lb) else torch.tensor(float(lb))
    return torch.tensor(0.0)


def get_expert_usage_unified(m: nn.Module) -> torch.Tensor:
    """Get expert usage counts from any MoE module.

    Returns a 1-D tensor of length num_experts. Falls back to zeros
    if the module does not track usage.
    """
    # Direct buffer
    usage = getattr(m, "expert_usage_counts", None)
    if usage is not None and torch.is_tensor(usage):
        return usage.detach().float()
    # Via routing snapshot
    snapshot = getattr(m, "last_routing_snapshot", None)
    if isinstance(snapshot, dict) and "expert_weights" in snapshot:
        w = snapshot["expert_weights"]
        if torch.is_tensor(w):
            return w.detach().float()
    # Via MoELoss
    moe_loss_fn = getattr(m, "moe_loss_fn", None)
    if moe_loss_fn is not None:
        usage = getattr(moe_loss_fn, "expert_usage", None)
        if usage is not None and torch.is_tensor(usage):
            return usage.detach().float()
    # Fallback: zeros
    num_experts = getattr(m, "num_experts", 4)
    return torch.zeros(num_experts)


def get_routing_weights_unified(m: nn.Module) -> torch.Tensor | None:
    """Get last routing weights from any MoE module."""
    snapshot = getattr(m, "last_routing_snapshot", None)
    if isinstance(snapshot, dict):
        for key in ("routing_weights", "expert_weights", "gate_weights", "weights"):
            val = snapshot.get(key)
            if val is not None and torch.is_tensor(val):
                return val.detach()
    # Via routing module
    routing = _get_routing_module(m)
    if routing is not None:
        for attr in ("last_weights", "routing_weights", "gate_output"):
            val = getattr(routing, attr, None)
            if val is not None and torch.is_tensor(val):
                return val.detach()
    return None


def reset_runtime_state_unified(m: nn.Module) -> None:
    """Reset runtime state (non-persistent buffers) on any MoE module."""
    # Reset usage counts
    usage = getattr(m, "expert_usage_counts", None)
    if usage is not None and torch.is_tensor(usage):
        usage.zero_()
    # Reset load balancing loss
    lb = getattr(m, "load_balancing_loss", None)
    if lb is not None and torch.is_tensor(lb):
        lb.zero_()
    # Reset routing snapshot
    snapshot = getattr(m, "last_routing_snapshot", None)
    if isinstance(snapshot, dict):
        snapshot.clear()
    # Reset training step
    ts = getattr(m, "training_step", None)
    if ts is not None and torch.is_tensor(ts):
        ts.fill_(0)
    # Reset MoELoss internal state
    moe_loss_fn = getattr(m, "moe_loss_fn", None)
    if moe_loss_fn is not None:
        last_loss = getattr(moe_loss_fn, "last_loss", None)
        if last_loss is not None and torch.is_tensor(last_loss):
            last_loss.zero_()


def moe_info(m: nn.Module) -> MoEInfo:
    """Build a unified info snapshot from any MoE module."""
    if not is_core_moe_block(m):
        raise TypeError(f"Module {type(m).__name__} is not a recognized MoE block")

    expert_mod = _get_expert_module(m)
    routing_mod = _get_routing_module(m)

    # Determine expert backend name
    for attr in ("experts", "fused_experts", "expert_group"):
        if hasattr(m, attr):
            expert_backend = attr
            break
    else:
        expert_backend = "unknown"

    # Get aux loss
    try:
        aux = get_aux_loss_unified(m)
        aux_val = float(aux.detach()) if torch.is_tensor(aux) else float(aux)
    except Exception:
        aux_val = 0.0

    # Get expert usage
    try:
        usage = get_expert_usage_unified(m)
        usage_list = usage.tolist() if usage.numel() > 0 else []
    except Exception:
        usage_list = []

    return MoEInfo(
        class_name=type(m).__name__,
        num_experts=getattr(m, "num_experts", 0),
        top_k=getattr(m, "top_k", getattr(m, "use_top_k", False) and getattr(m, "top_k", 0) or 0),
        has_routing=routing_mod is not None,
        has_experts=expert_mod is not None,
        expert_backend=expert_backend,
        has_balance_loss=hasattr(m, "balance_loss_coeff") or hasattr(m, "load_balancing_loss"),
        aux_loss_value=aux_val,
        expert_usage=usage_list,
    )


def collect_all_moe_info(model: nn.Module) -> dict[str, MoEInfo]:
    """Collect MoEInfo from every MoE module in the model.

    Returns:
        Dict mapping module name → MoEInfo.
    """
    results = {}
    for name, m in model.named_modules():
        if is_core_moe_block(m):
            results[name] = moe_info(m)
    return results


def get_balance_loss_coeff_unified(m: nn.Module) -> float:
    """Get balance loss coefficient from any MoE module."""
    return float(getattr(m, "balance_loss_coeff", 1.0))


def set_balance_loss_coeff_unified(m: nn.Module, value: float) -> None:
    """Set balance loss coefficient on any MoE module."""
    m.balance_loss_coeff = value
    moe_loss_fn = getattr(m, "moe_loss_fn", None)
    if moe_loss_fn is not None:
        moe_loss_fn.balance_loss_coeff = value
