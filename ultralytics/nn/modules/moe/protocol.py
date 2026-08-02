"""Minimal adapters for MoE routing snapshots and derived metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import torch


@dataclass(frozen=True)
class RoutingMetrics:
    """JSON-safe metrics derived from one normalized routing snapshot."""

    expert_usage: list[float]
    topk_counts: list[float]
    num_experts: int
    top_k: int
    aux_loss: float
    gini: float
    dominant_expert: int
    dominant_share: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().reshape(-1).cpu().tolist()
    if value is None:
        return []
    return [float(item) for item in value]


def normalize_routing_snapshot(
    snapshot: Mapping[str, Any] | None, *, num_experts: int = 0, top_k: int = 0
) -> dict[str, Any]:
    """Normalize legacy/current snapshot keys without mutating the producer payload."""
    source = snapshot or {}
    usage = _float_list(source.get("expert_usage", source.get("usage")))
    counts = _float_list(source.get("topk_counts", source.get("counts")))
    size = int(source.get("num_experts", num_experts or len(usage)))
    if size > 0:
        usage = (usage + [0.0] * size)[:size]
        counts = (counts + [0.0] * size)[:size]
    return {
        **source,
        "expert_usage": usage,
        "topk_counts": counts,
        "num_experts": size,
        "top_k": int(source.get("top_k", top_k)),
        "aux_loss": float(source.get("aux_loss", 0.0)),
    }


def usage_gini(usage: Iterable[float] | torch.Tensor) -> float:
    """Return one canonical bounded Gini coefficient for expert usage."""
    values = torch.as_tensor(_float_list(usage), dtype=torch.float32).clamp_min(0.0)
    if values.numel() == 0 or float(values.sum()) <= 0.0:
        return 0.0
    diff = torch.abs(values[:, None] - values[None, :]).sum()
    return float((diff / (2 * values.numel() * values.sum())).item())


def routing_metrics(snapshot: Mapping[str, Any] | None, *, num_experts: int = 0, top_k: int = 0) -> RoutingMetrics:
    """Adapt a routing snapshot to the shared scheduler/diagnostic metric vocabulary."""
    normalized = normalize_routing_snapshot(snapshot, num_experts=num_experts, top_k=top_k)
    usage = normalized["expert_usage"]
    dominant = max(range(len(usage)), key=usage.__getitem__) if usage else -1
    share = usage[dominant] if dominant >= 0 else 0.0
    return RoutingMetrics(
        expert_usage=usage,
        topk_counts=normalized["topk_counts"],
        num_experts=normalized["num_experts"],
        top_k=normalized["top_k"],
        aux_loss=normalized["aux_loss"],
        gini=usage_gini(usage),
        dominant_expert=dominant,
        dominant_share=share,
    )
