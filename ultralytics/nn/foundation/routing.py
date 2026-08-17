"""Training-only routing helpers for the F11 Foundation Teacher Router.

The routing teacher deliberately has a small, frozen head.  It consumes a detached
summary of the YOLO latent and a DINO spatial/pooled summary, and produces one
logit vector for the configured number of LatentMixture experts.  The head is
kept outside the student module tree by :class:`FoundationDistillationModel`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .protocol import FoundationFeatures


def _as_feature(features: FoundationFeatures | Mapping[str, Any]) -> torch.Tensor:
    """Extract and validate the DINO dense P4 tensor used by the route target."""

    dense = features.dense if isinstance(features, FoundationFeatures) else features.get("dense")
    if not isinstance(dense, Mapping) or "p4" not in dense:
        raise KeyError("Foundation Teacher output must contain dense['p4'] for routing distillation.")
    feature = dense["p4"]
    if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
        raise ValueError(
            "Foundation Teacher dense['p4'] must be a BCHW tensor, "
            f"got {type(feature).__name__} {getattr(feature, 'shape', None)}."
        )
    if not feature.is_floating_point() or not bool(torch.isfinite(feature).all().item()):
        raise ValueError("Foundation Teacher dense['p4'] must be finite floating point features.")
    return feature


def foundation_teacher_summary(features: FoundationFeatures | Mapping[str, Any]) -> torch.Tensor:
    """Build a compact, deterministic image-level DINO route summary.

    The summary contains the teacher pooled embedding, spatial mean, and spatial
    standard deviation.  DINO implementations that do not expose ``pooled`` use
    the spatial mean for that slot, keeping the protocol backend agnostic.
    """

    dense = _as_feature(features)
    spatial_mean = dense.float().mean(dim=(2, 3))
    spatial_std = dense.float().flatten(2).std(dim=2, unbiased=False)
    pooled = features.pooled if isinstance(features, FoundationFeatures) else features.get("pooled")
    if pooled is None:
        pooled = spatial_mean
    if not isinstance(pooled, torch.Tensor) or pooled.ndim != 2:
        raise ValueError(f"Foundation Teacher pooled summary must be [B,C], got {getattr(pooled, 'shape', None)}.")
    pooled = pooled.float()
    if pooled.shape[0] != dense.shape[0] or pooled.shape[1] != dense.shape[1]:
        raise ValueError(
            "Foundation Teacher pooled summary must match dense['p4'] batch/channels: "
            f"pooled={tuple(pooled.shape)}, dense={tuple(dense.shape)}."
        )
    summary = torch.cat((pooled, spatial_mean, spatial_std), dim=1)
    if not bool(torch.isfinite(summary).all().item()):
        raise ValueError("Foundation Teacher route summary contains NaN or Inf values.")
    return summary.detach()


def _nested_teacher_features(features: FoundationFeatures | Mapping[str, Any], name: str) -> Any:
    """Resolve one named teacher response from a multi-teacher transport object."""

    if isinstance(features, Mapping):
        teachers = features.get("teachers", features.get("teacher_features"))
        if isinstance(teachers, Mapping) and name in teachers:
            return teachers[name]
        if name in features:
            return features[name]
    elif isinstance(features, FoundationFeatures):
        metadata = features.metadata
        if isinstance(metadata, Mapping):
            teachers = metadata.get("teachers", metadata.get("teacher_features"))
            if isinstance(teachers, Mapping) and name in teachers:
                return teachers[name]
    return None


def _semantic_feature(features: FoundationFeatures | Mapping[str, Any]) -> torch.Tensor:
    """Extract the normalized SigLIP2 semantic embedding required by F14."""

    semantic = features.semantic if isinstance(features, FoundationFeatures) else features.get("semantic")
    if not isinstance(semantic, torch.Tensor) or semantic.ndim != 2 or semantic.shape[1] <= 0:
        raise ValueError(
            "Multi-Foundation routing requires SigLIP2 semantic features with shape [B,C]; "
            f"got {getattr(semantic, 'shape', None)}."
        )
    if not semantic.is_floating_point() or not bool(torch.isfinite(semantic).all().item()):
        raise ValueError("SigLIP2 semantic features for routing must be finite floating point values.")
    return semantic.float()


def _pooled_or_dense_summary(features: FoundationFeatures | Mapping[str, Any], semantic: torch.Tensor) -> torch.Tensor:
    """Build a fixed-width SigLIP2 route summary even for pooled-only test backends."""

    dense = features.dense if isinstance(features, FoundationFeatures) else features.get("dense")
    if isinstance(dense, Mapping) and isinstance(dense.get("p4"), torch.Tensor):
        feature = dense["p4"]
        if feature.ndim != 4 or feature.shape[0] != semantic.shape[0]:
            raise ValueError(
                "SigLIP2 dense['p4'] must be BCHW with the same batch as semantic features; "
                f"dense={getattr(feature, 'shape', None)}, semantic={tuple(semantic.shape)}."
            )
        if not feature.is_floating_point() or not bool(torch.isfinite(feature).all().item()):
            raise ValueError("SigLIP2 dense['p4'] for routing must be finite floating point values.")
        spatial_mean = feature.float().mean(dim=(2, 3))
        spatial_std = feature.float().flatten(2).std(dim=2, unbiased=False)
    else:
        # A pooled-only injected backend remains useful for route-unit tests, but
        # receives explicit zero spatial complexity instead of an accidental
        # shape-dependent fallback.
        spatial_mean = torch.zeros_like(semantic)
        spatial_std = torch.zeros_like(semantic)
    logits = semantic.float()
    probs = F.softmax(logits, dim=-1)
    entropy = (-(probs * probs.clamp_min(1e-12).log()).sum(dim=1, keepdim=True)).to(dtype=semantic.dtype)
    variance = logits.std(dim=1, unbiased=False, keepdim=True)
    return torch.cat((semantic, spatial_mean, spatial_std, entropy, variance), dim=1)


def foundation_multiteacher_summary(
    features: FoundationFeatures | Mapping[str, Any],
    *,
    teacher_order: tuple[str, ...] = ("dinov3", "siglip2"),
) -> torch.Tensor:
    """Build the F14 image-level route target from DINOv3 and SigLIP2.

    DINOv3 contributes its pooled/spatial mean/std summary (a compact proxy for
    spatial complexity). SigLIP2 contributes semantic, spatial mean/std, entropy,
    and variance (semantic uncertainty). The two blocks are concatenated in the
    declared order and detached, so the frozen teachers never receive gradients.

    ``features`` may be a mapping containing named responses, or a
    :class:`FoundationFeatures` whose metadata contains ``teachers`` or
    ``teacher_features``. A missing teacher or SigLIP2 semantic capability is an
    explicit error; F14 never silently falls back to a single teacher.
    """

    order = tuple(str(name).lower() for name in teacher_order)
    if order != ("dinov3", "siglip2"):
        raise ValueError("F14 multi-teacher routing currently requires teacher_order=('dinov3', 'siglip2').")
    dino = _nested_teacher_features(features, "dinov3")
    siglip = _nested_teacher_features(features, "siglip2")
    if dino is None or siglip is None:
        raise ValueError(
            "F14 multi-teacher routing requires both named outputs: 'dinov3' and 'siglip2'. "
            "A single available teacher is not silently accepted."
        )
    dino_summary = foundation_teacher_summary(dino)
    semantic = _semantic_feature(siglip)
    siglip_summary = _pooled_or_dense_summary(siglip, semantic)
    if dino_summary.shape[0] != siglip_summary.shape[0]:
        raise ValueError(
            "DINOv3 and SigLIP2 route summaries must have matching batch sizes: "
            f"{dino_summary.shape[0]} vs {siglip_summary.shape[0]}."
        )
    summary = torch.cat((dino_summary, siglip_summary), dim=1)
    if not bool(torch.isfinite(summary).all().item()):
        raise ValueError("Multi-Foundation route summary contains NaN or Inf values.")
    return summary.detach()


class FoundationTeacherRouter(nn.Module):
    """Frozen image-level teacher head used to generate route targets.

    ``student_summary`` is included as context, matching the F11 design while
    remaining detached from the routing-KD graph.  Parameters are initialized
    deterministically from ``seed`` so resume can recreate an identical target
    without serializing the training-only teacher head.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        num_experts: int,
        *,
        hidden_dim: int | None = None,
        seed: int = 1101,
    ) -> None:
        super().__init__()
        student_dim = int(student_dim)
        teacher_dim = int(teacher_dim)
        num_experts = int(num_experts)
        if student_dim <= 0 or teacher_dim <= 0 or num_experts <= 0:
            raise ValueError("FoundationTeacherRouter dimensions and num_experts must be positive.")
        hidden = int(hidden_dim or max(32, min(256, 2 * max(student_dim, teacher_dim))))
        if hidden <= 0:
            raise ValueError("FoundationTeacherRouter hidden_dim must be positive.")
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden
        self.seed = int(seed)
        self.student_proj = nn.Linear(student_dim, hidden, bias=False)
        self.teacher_proj = nn.Linear(teacher_dim, hidden, bias=False)
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_experts),
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        self.requires_grad_(False)
        self.eval()

    def forward(self, student_summary: torch.Tensor, teacher_summary: torch.Tensor) -> torch.Tensor:
        """Return frozen teacher route logits with shape ``[B, num_experts]``."""

        if not isinstance(student_summary, torch.Tensor) or student_summary.ndim != 2:
            raise ValueError("student_summary must be a [B,D] tensor.")
        if not isinstance(teacher_summary, torch.Tensor) or teacher_summary.ndim != 2:
            raise ValueError("teacher_summary must be a [B,D] tensor.")
        if student_summary.shape[0] != teacher_summary.shape[0]:
            raise ValueError("student_summary and teacher_summary batch sizes must match.")
        if student_summary.shape[1] != self.student_dim:
            raise ValueError(
                f"student_summary dim {student_summary.shape[1]} does not match configured {self.student_dim}."
            )
        if teacher_summary.shape[1] != self.teacher_dim:
            raise ValueError(
                f"teacher_summary dim {teacher_summary.shape[1]} does not match configured {self.teacher_dim}."
            )
        student = torch.nan_to_num(student_summary.float(), nan=0.0, posinf=0.0, neginf=0.0)
        teacher = torch.nan_to_num(teacher_summary.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return self.fuse(torch.cat((self.student_proj(student), self.teacher_proj(teacher)), dim=1))


def routing_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute F11 temperature-scaled KL routing distillation.

    The teacher is detached by contract; only ``student_logits`` receives
    gradients.  ``T²`` preserves the conventional distillation gradient scale.
    """

    if not isinstance(student_logits, torch.Tensor) or not isinstance(teacher_logits, torch.Tensor):
        raise TypeError("routing logits must be tensors.")
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError(
            "student and teacher routing logits must both be [B,E] with matching shape, "
            f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}."
        )
    temperature = float(temperature)
    if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0.0:
        raise ValueError(f"routing temperature must be finite and positive, got {temperature}.")
    student = student_logits.float()
    teacher = teacher_logits.detach().float()
    loss = F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="batchmean",
    )
    loss = loss * (temperature * temperature)
    if not bool(torch.isfinite(loss).all().item()):
        raise ValueError("routing distillation loss is NaN or Inf.")
    return loss.reshape(())


__all__ = [
    "FoundationTeacherRouter",
    "foundation_teacher_summary",
    "foundation_multiteacher_summary",
    "routing_kd_loss",
]
