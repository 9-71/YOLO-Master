"""Versioned interchange contract between V-PEFT solvers and LoRA injection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch.nn as nn


def _model_fingerprint(model: nn.Module) -> str:
    """Build the same stable model binding hash used by the LoRA API."""
    entries = []
    for name, module in model.named_modules():
        if name:
            entries.append((name, module.__class__.__qualname__))
    for name, parameter in model.named_parameters():
        entries.append((f"param:{name}", tuple(parameter.shape), str(parameter.dtype)))
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PlacementTarget:
    """One adapter placement target emitted by a planner."""

    name: str
    variant: str = "lora"
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "variant": self.variant, "rank": int(self.rank)}


@dataclass(frozen=True)
class PlacementPlan:
    """Serializable, auditable planner result consumed by the adapter layer."""

    model_fingerprint: str
    planner_backend: str
    solver: str
    budget: dict[str, int]
    targets: tuple[PlacementTarget, ...] = ()
    constraints: dict[str, list[str]] = field(default_factory=lambda: {"hard": [], "soft": []})
    predicted_delta: float | None = None
    confidence: float | None = None
    status: str = "FALLBACK"
    refusal_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported PlacementPlan schema_version={self.schema_version}")
        if self.status not in {"ADAPT", "ACCEPT", "REFUSE", "FALLBACK"}:
            raise ValueError(f"invalid PlacementPlan status={self.status!r}")
        if int(self.budget.get("max_adapter_params", 0)) < 0:
            raise ValueError("max_adapter_params must be non-negative")

    @property
    def fingerprint(self) -> str:
        """Return a stable hash of the plan payload."""
        payload = json.dumps(self.to_dict(include_fingerprint=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "model_fingerprint": self.model_fingerprint,
            "planner_backend": self.planner_backend,
            "solver": self.solver,
            "budget": dict(self.budget),
            "targets": [target.to_dict() for target in self.targets],
            "constraints": {key: list(value) for key, value in self.constraints.items()},
            "predicted_delta": self.predicted_delta,
            "confidence": self.confidence,
            "status": self.status,
            "refusal_reason": self.refusal_reason,
            "metadata": dict(self.metadata),
        }
        if include_fingerprint:
            payload["plan_fingerprint"] = self.fingerprint
        return payload

    def validate_model(self, model: nn.Module, *, require_targets: bool = True) -> None:
        """Validate that this plan is safe to consume for ``model``.

        V-PEFT plans are serialized artifacts. Checking the binding fingerprint
        and every target before adapter injection prevents silently applying a
        stale plan to a structurally similar but incompatible model.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model)!r}")
        actual_fingerprint = _model_fingerprint(model)
        if self.model_fingerprint and actual_fingerprint != self.model_fingerprint:
            raise ValueError("PlacementPlan model fingerprint mismatch; rebuild the plan for the current model")
        if require_targets and not self.targets:
            raise ValueError("PlacementPlan contains no adapter targets")
        for target in self.targets:
            try:
                module = model.get_submodule(target.name)
            except (AttributeError, RuntimeError):
                module = None
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                raise ValueError(
                    f"PlacementPlan target {target.name!r} is missing or unsupported; expected Conv2d/Linear"
                )
            rank = int(target.rank)
            if rank <= 0:
                raise ValueError(f"PlacementPlan target {target.name!r} must have a positive rank")
            capacity = min(
                int(module.in_channels if isinstance(module, nn.Conv2d) else module.in_features),
                int(module.out_channels if isinstance(module, nn.Conv2d) else module.out_features),
            )
            if rank > capacity:
                raise ValueError(f"PlacementPlan rank {rank} for {target.name!r} exceeds layer capacity {capacity}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlacementPlan":
        targets = tuple(
            PlacementTarget(str(item["name"]), str(item.get("variant", "lora")), int(item.get("rank", 0)))
            for item in payload.get("targets", [])
        )
        plan = cls(
            schema_version=int(payload.get("schema_version", 1)),
            model_fingerprint=str(payload.get("model_fingerprint", "")),
            planner_backend=str(payload.get("planner_backend", "legacy")),
            solver=str(payload.get("solver", "none")),
            budget={key: int(value) for key, value in dict(payload.get("budget", {})).items()},
            targets=targets,
            constraints={key: list(value) for key, value in dict(payload.get("constraints", {})).items()},
            predicted_delta=payload.get("predicted_delta"),
            confidence=payload.get("confidence"),
            status=str(payload.get("status", "FALLBACK")),
            refusal_reason=payload.get("refusal_reason"),
            metadata=dict(payload.get("metadata", {})),
        )
        expected = payload.get("plan_fingerprint")
        if expected is not None and expected != plan.fingerprint:
            raise ValueError("PlacementPlan fingerprint mismatch")
        return plan


__all__ = ["PlacementPlan", "PlacementTarget"]
