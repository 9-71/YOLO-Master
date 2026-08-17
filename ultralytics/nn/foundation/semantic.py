"""Region-level semantic distillation helpers for F13.

The first F13 implementation intentionally operates on positive P4 locations
returned by the native task-aligned assigner.  Background tokens never enter
the semantic loss, which keeps the SigLIP2 prior focused on object regions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _as_labels_by_image(batch: Mapping[str, torch.Tensor], batch_size: int, device: torch.device) -> list[torch.Tensor]:
    """Reconstruct per-image GT class order used by the native preprocessor."""
    classes = batch.get("cls")
    batch_idx = batch.get("batch_idx")
    if not isinstance(classes, torch.Tensor) or not isinstance(batch_idx, torch.Tensor):
        return [torch.empty(0, dtype=torch.long, device=device) for _ in range(batch_size)]
    classes = classes.reshape(-1).to(device=device, dtype=torch.long)
    batch_idx = batch_idx.reshape(-1).to(device=device, dtype=torch.long)
    result = []
    for image_index in range(batch_size):
        result.append(classes[batch_idx == image_index])
    return result


def positive_region_pool(
    feature: torch.Tensor,
    fg_mask: torch.Tensor,
    target_gt_idx: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    level_index: int = 1,
    feature_shapes: Sequence[Any] | None = None,
    source_level_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool positive matched feature locations from one FPN level.

    Args:
        feature: Student BCHW feature for the requested level (P4 by default).
        fg_mask: Native assigner foreground mask, shaped ``(B, sum(H_l W_l))``.
        target_gt_idx: Native matched GT indices, shaped like ``fg_mask``.
        batch: Detection batch containing ``cls`` and ``batch_idx``.
        level_index: Zero-based feature level index in ``preds['feats']``.

    Returns:
        Region vectors ``(N,C)``, image indices ``(N,)``, class labels ``(N,)``,
        and flattened P4 token indices ``(N,)``.  Empty selections preserve the
        feature device and dtype.
    """
    if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
        raise ValueError(f"feature must be BCHW, got {getattr(feature, 'shape', None)}")
    if not isinstance(fg_mask, torch.Tensor) or not isinstance(target_gt_idx, torch.Tensor):
        raise TypeError("fg_mask and target_gt_idx must be tensors")
    if fg_mask.ndim != 2 or target_gt_idx.shape != fg_mask.shape:
        raise ValueError("fg_mask and target_gt_idx must have matching shape (B, anchors)")
    batch_size, channels, height, width = feature.shape
    if fg_mask.shape[0] != batch_size:
        raise ValueError("feature batch and assigner batch must match")

    feats = feature_shapes
    if not isinstance(feats, Sequence) or level_index >= len(feats):
        # The caller may provide the level sizes separately through this private
        # transport key.  A missing key is an explicit no-positive result.
        empty = feature.new_empty((0, channels))
        return (
            empty,
            torch.empty(0, dtype=torch.long, device=feature.device),
            torch.empty(0, dtype=torch.long, device=feature.device),
            torch.empty(0, dtype=torch.long, device=feature.device),
        )
    level_sizes = []
    level_hw = []
    for item in feats:
        shape = tuple(item.shape) if isinstance(item, torch.Tensor) else tuple(item)
        if len(shape) < 2:
            raise ValueError("feature_shapes entries must expose height and width")
        level_sizes.append(int(shape[-2] * shape[-1]))
        level_hw.append((int(shape[-2]), int(shape[-1])))
    if level_index < 0 or level_index >= len(level_sizes):
        raise ValueError(f"level_index must be in [0, {len(level_sizes)}), got {level_index}")
    offset = sum(level_sizes[:level_index])
    count = level_sizes[level_index]
    if fg_mask.shape[1] < offset + count:
        raise ValueError("assigner mask has fewer anchors than the requested feature level")
    local_positive = fg_mask[:, offset : offset + count].bool()
    if not local_positive.any():
        empty = feature.new_empty((0, channels))
        return (
            empty,
            torch.empty(0, dtype=torch.long, device=feature.device),
            torch.empty(0, dtype=torch.long, device=feature.device),
            torch.empty(0, dtype=torch.long, device=feature.device),
        )

    matches = target_gt_idx[:, offset : offset + count].to(device=feature.device, dtype=torch.long)
    image_indices, local_indices = local_positive.nonzero(as_tuple=True)
    source_level = level_index if source_level_index is None else int(source_level_index)
    if source_level < 0 or source_level >= len(level_sizes):
        raise ValueError(f"source_level_index must be in [0, {len(level_sizes)}), got {source_level}")
    source_h, source_w = feature.shape[-2:]
    if source_level != level_index:
        source_h, source_w = level_hw[source_level]
        if (source_h, source_w) != tuple(feature.shape[-2:]):
            raise ValueError("feature spatial size must match source_level_index")
    requested_h, requested_w = level_hw[level_index]
    requested_y = torch.div(local_indices, requested_w, rounding_mode="floor")
    requested_x = local_indices.remainder(requested_w)
    source_y = (requested_y.float() * source_h / requested_h).floor().long().clamp_max(source_h - 1)
    source_x = (requested_x.float() * source_w / requested_w).floor().long().clamp_max(source_w - 1)
    source_indices = source_y * source_w + source_x
    vectors = feature.permute(0, 2, 3, 1).reshape(batch_size, source_h * source_w, channels)[
        image_indices, source_indices
    ]
    gt_slots = matches[image_indices, local_indices]
    labels_by_image = _as_labels_by_image(batch, batch_size, feature.device)
    labels = torch.full_like(gt_slots, -1)
    for image_index in image_indices.unique().tolist():
        selected = image_indices == int(image_index)
        image_labels = labels_by_image[int(image_index)]
        slots = gt_slots[selected]
        valid = (slots >= 0) & (slots < image_labels.numel())
        if valid.any():
            labels[selected.nonzero(as_tuple=True)[0][valid]] = image_labels[slots[valid]]
    valid = labels >= 0
    return vectors[valid], image_indices[valid], labels[valid], source_indices[valid]


class RegionSemanticProjector(nn.Module):
    """Trainable student-region projection into the frozen teacher semantic space."""

    def __init__(self, student_channels: int, semantic_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if isinstance(student_channels, bool) or not isinstance(student_channels, int) or student_channels <= 0:
            raise ValueError("student_channels must be a positive integer")
        if isinstance(semantic_dim, bool) or not isinstance(semantic_dim, int) or semantic_dim <= 0:
            raise ValueError("semantic_dim must be a positive integer")
        hidden = int(hidden_dim or min(semantic_dim, max(64, student_channels)))
        if hidden <= 0:
            raise ValueError("hidden_dim must be positive")
        self.student_channels = student_channels
        self.semantic_dim = semantic_dim
        self.hidden_dim = hidden
        self.proj = nn.Sequential(nn.Linear(student_channels, hidden), nn.GELU(), nn.Linear(hidden, semantic_dim))

    def forward(self, regions: torch.Tensor) -> torch.Tensor:
        if regions.ndim != 2 or regions.shape[1] != self.student_channels:
            raise ValueError(f"regions must have shape (N,{self.student_channels}), got {tuple(regions.shape)}")
        return F.normalize(self.proj(regions), dim=-1)


def region_text_loss(
    region_embeddings: torch.Tensor,
    class_labels: torch.Tensor,
    text_prototypes: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Classify positive region embeddings against normalized text prototypes."""
    if region_embeddings.ndim != 2 or text_prototypes.ndim != 2:
        raise ValueError("region_embeddings and text_prototypes must be 2D")
    if region_embeddings.shape[0] == 0:
        return region_embeddings.sum() * 0.0
    if region_embeddings.shape[1] != text_prototypes.shape[1]:
        raise ValueError("region and text prototype dimensions must match")
    labels = class_labels.to(device=region_embeddings.device, dtype=torch.long).reshape(-1)
    if labels.numel() != region_embeddings.shape[0] or labels.min() < 0 or labels.max() >= text_prototypes.shape[0]:
        raise ValueError("class_labels must index text_prototypes")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or float(temperature) <= 0:
        raise ValueError("temperature must be a positive number")
    logits = F.normalize(region_embeddings.float(), dim=-1) @ F.normalize(text_prototypes.float(), dim=-1).t()
    return F.cross_entropy(logits / float(temperature), labels)


def region_image_loss(region_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
    """Align positive regions with the frozen image semantic representation."""
    if region_embeddings.ndim != 2 or image_embeddings.ndim != 2:
        raise ValueError("region_embeddings and image_embeddings must be 2D")
    if region_embeddings.shape[0] == 0:
        return region_embeddings.sum() * 0.0
    if (
        image_embeddings.shape[0] != region_embeddings.shape[0]
        or image_embeddings.shape[1] != region_embeddings.shape[1]
    ):
        raise ValueError("region and image embedding shapes must match")
    similarity = F.cosine_similarity(region_embeddings.float(), image_embeddings.detach().float(), dim=-1)
    return (1.0 - similarity).mean()


def semantic_distillation_loss(
    region_embeddings: torch.Tensor,
    class_labels: torch.Tensor,
    image_embeddings: torch.Tensor,
    text_prototypes: torch.Tensor | None,
    *,
    text_weight: float = 1.0,
    image_weight: float = 1.0,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, text, and image components for F13 region semantic KD."""
    zero = region_embeddings.sum() * 0.0
    text = zero
    if text_weight > 0:
        if text_prototypes is None:
            raise ValueError("text_prototypes are required when text_weight is positive")
        text = region_text_loss(region_embeddings, class_labels, text_prototypes, temperature=temperature)
    image = region_image_loss(region_embeddings, image_embeddings) if image_weight > 0 else zero
    total = float(text_weight) * text + float(image_weight) * image
    if not torch.isfinite(total):
        raise ValueError("semantic distillation loss is NaN or Inf")
    return total, text, image


__all__ = [
    "RegionSemanticProjector",
    "positive_region_pool",
    "region_text_loss",
    "region_image_loss",
    "semantic_distillation_loss",
]
