"""Numerically stable feature distillation losses for Foundation KD."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.nn.modules._numeric import disabled_autocast


_RELATION_MODES = frozenset({"sampled", "full"})


def foreground_token_weights(
    batch: dict[str, torch.Tensor],
    *,
    height: int,
    width: int,
    image_height: int,
    image_width: int,
    foreground_weight: float = 1.5,
    boundary_weight: float = 1.0,
    background_weight: float = 0.25,
) -> torch.Tensor:
    """Build detached P4-token weights from normalized GT boxes.

    Boxes are interpreted as ``xywh`` normalized to the input image.  A token whose center lies inside any box gets
    ``foreground_weight``; tokens in a one-cell dilated box boundary get ``boundary_weight`` and all other tokens
    get ``background_weight``. Empty or malformed target tensors fall back to an all-background map.
    """
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not torch.isfinite(torch.tensor(float(value)))
        or float(value) < 0
        for value in (foreground_weight, boundary_weight, background_weight)
    ):
        raise ValueError("foreground, boundary, and background weights must be finite non-negative numbers.")
    if height <= 0 or width <= 0 or image_height <= 0 or image_width <= 0:
        raise ValueError("Token and image dimensions must be positive.")
    images = batch.get("img")
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("batch['img'] must be a BCHW tensor to build foreground token weights.")
    batch_size = int(images.shape[0])
    device = images.device
    dtype = torch.float32
    weights = torch.full((batch_size, height, width), float(background_weight), device=device, dtype=dtype)
    boxes = batch.get("bboxes")
    batch_idx = batch.get("batch_idx")
    if not isinstance(boxes, torch.Tensor) or boxes.numel() == 0 or not isinstance(batch_idx, torch.Tensor):
        return weights.reshape(batch_size, height * width).detach()
    boxes = boxes.to(device=device, dtype=dtype).reshape(-1, 4)
    batch_idx = batch_idx.to(device=device, dtype=torch.long).reshape(-1)
    count = min(boxes.shape[0], batch_idx.numel())
    if count == 0:
        return weights.reshape(batch_size, height * width).detach()
    boxes, batch_idx = boxes[:count], batch_idx[:count]
    valid = (
        torch.isfinite(boxes).all(dim=1) & (batch_idx >= 0) & (batch_idx < batch_size) & (boxes[:, 2:] > 0).all(dim=1)
    )
    boxes, batch_idx = boxes[valid], batch_idx[valid]
    if boxes.numel() == 0:
        return weights.reshape(batch_size, height * width).detach()

    # Convert normalized xywh boxes to image coordinates and clip to valid image bounds.
    cx, cy, bw, bh = boxes.unbind(dim=1)
    x1 = ((cx - bw / 2) * image_width).clamp(0, image_width)
    y1 = ((cy - bh / 2) * image_height).clamp(0, image_height)
    x2 = ((cx + bw / 2) * image_width).clamp(0, image_width)
    y2 = ((cy + bh / 2) * image_height).clamp(0, image_height)
    cell_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * image_width / width
    cell_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * image_height / height
    grid_y, grid_x = torch.meshgrid(cell_y, cell_x, indexing="ij")
    boundary_x = image_width / width
    boundary_y = image_height / height
    for index in range(boxes.shape[0]):
        sample = int(batch_idx[index])
        inside = (grid_x >= x1[index]) & (grid_x <= x2[index]) & (grid_y >= y1[index]) & (grid_y <= y2[index])
        expanded = (
            (grid_x >= x1[index] - boundary_x)
            & (grid_x <= x2[index] + boundary_x)
            & (grid_y >= y1[index] - boundary_y)
            & (grid_y <= y2[index] + boundary_y)
        )
        sample_weights = weights[sample]
        sample_weights[expanded] = torch.maximum(sample_weights[expanded], sample_weights.new_tensor(boundary_weight))
        sample_weights[inside] = torch.maximum(sample_weights[inside], sample_weights.new_tensor(foreground_weight))
    return weights.reshape(batch_size, height * width).detach()


def _validate_feature_pair(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> tuple[int, int, int, int]:
    """Validate aligned BCHW feature tensors and return their shape."""
    for name, feature in (("student_feat", student_feat), ("teacher_feat", teacher_feat)):
        if not isinstance(feature, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(feature).__name__}.")
        if feature.ndim != 4:
            raise ValueError(f"{name} must have shape (B, C, H, W), got {tuple(feature.shape)}.")
        if feature.shape[0] <= 0 or feature.shape[1] <= 0 or feature.shape[2] <= 0 or feature.shape[3] <= 0:
            raise ValueError(f"{name} must have positive BCHW dimensions, got {tuple(feature.shape)}.")
        if not torch.isfinite(feature).all():
            raise ValueError(f"{name} contains NaN or Inf values.")
    if student_feat.shape != teacher_feat.shape:
        raise ValueError(
            "student_feat and teacher_feat must have identical aligned shapes, got "
            f"{tuple(student_feat.shape)} and {tuple(teacher_feat.shape)}."
        )
    if student_feat.device != teacher_feat.device:
        raise ValueError(
            f"student_feat and teacher_feat must be on the same device, got {student_feat.device} and "
            f"{teacher_feat.device}."
        )
    return tuple(student_feat.shape)


def _validate_weight(name: str, value: float) -> float:
    """Validate a finite non-negative scalar loss weight."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number, got {value!r}.")
    value = float(value)
    if not torch.isfinite(torch.tensor(value)) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}.")
    return value


def cosine_kd_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    *,
    eps: float = 1e-6,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-spatial-token cosine distillation loss in FP32.

    Args:
        student_feat (torch.Tensor): Aligned student features in ``(B, C, H, W)`` layout.
        teacher_feat (torch.Tensor): Aligned teacher features in ``(B, C, H, W)`` layout.
        eps (float): Positive denominator floor used by cosine similarity.

    Returns:
        (torch.Tensor): A scalar loss tensor with gradient to ``student_feat`` only when the teacher is detached.
    """
    _validate_feature_pair(student_feat, teacher_feat)
    if (
        not isinstance(eps, (int, float))
        or isinstance(eps, bool)
        or not torch.isfinite(torch.tensor(float(eps)))
        or eps <= 0
    ):
        raise ValueError(f"eps must be a finite positive number, got {eps!r}.")

    batch, channels, height, width = student_feat.shape
    device_type = student_feat.device.type
    with disabled_autocast(device_type):
        student = student_feat.float().reshape(batch, channels, height * width)
        teacher = teacher_feat.detach().float().reshape(batch, channels, height * width)
        similarity = F.cosine_similarity(student, teacher, dim=1, eps=float(eps))
        if token_weights is None:
            loss = 1.0 - similarity.mean()
        else:
            weights = _validate_token_weights(token_weights, batch, height * width, student_feat.device)
            loss = 1.0 - (similarity * weights).sum() / weights.sum().clamp_min(1e-6)
    if not torch.isfinite(loss):
        raise ValueError("cosine KD loss is NaN or Inf.")
    return loss


def _resolve_relation_indices(
    num_tokens: int,
    num_samples: int | None,
    sample_indices: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Resolve deterministic or random token indices for sampled relational KD."""
    if sample_indices is not None:
        if not isinstance(sample_indices, torch.Tensor):
            raise TypeError(f"sample_indices must be a torch.Tensor, got {type(sample_indices).__name__}.")
        if sample_indices.ndim != 1 or sample_indices.numel() == 0:
            raise ValueError("sample_indices must be a non-empty 1D tensor.")
        if sample_indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError(f"sample_indices must contain integer indices, got {sample_indices.dtype}.")
        indices = sample_indices.to(device=device, dtype=torch.long)
        if indices.min() < 0 or indices.max() >= num_tokens:
            raise ValueError(f"sample_indices must be in [0, {num_tokens}), got {indices.tolist()}.")
        return indices
    if num_samples is None:
        return torch.arange(num_tokens, device=device)
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError(f"num_samples must be a positive integer or None, got {num_samples!r}.")
    if num_samples >= num_tokens:
        return torch.arange(num_tokens, device=device)
    return torch.randperm(num_tokens, device=device)[:num_samples]


def _validate_token_weights(weights: torch.Tensor, batch: int, tokens: int, device: torch.device) -> torch.Tensor:
    """Validate and flatten non-negative per-token weights."""
    if not isinstance(weights, torch.Tensor):
        raise TypeError(f"token_weights must be a torch.Tensor, got {type(weights).__name__}.")
    if weights.numel() != batch * tokens:
        raise ValueError(f"token_weights must contain {(batch, tokens)} values, got {tuple(weights.shape)}.")
    weights = weights.to(device=device, dtype=torch.float32).reshape(batch, tokens)
    if not torch.isfinite(weights).all() or (weights < 0).any() or float(weights.sum()) <= 0:
        raise ValueError("token_weights must be finite, non-negative, and have a positive sum.")
    return weights.detach()


def relational_kd_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    *,
    mode: str = "sampled",
    num_samples: int = 256,
    sample_indices: torch.Tensor | None = None,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match normalized spatial-token Gram matrices with bounded sampled memory.

    Args:
        student_feat (torch.Tensor): Aligned student features in ``(B, C, H, W)`` layout.
        teacher_feat (torch.Tensor): Aligned teacher features in ``(B, C, H, W)`` layout.
        mode (str): ``sampled`` to cap token pairs or ``full`` to use the complete spatial grid.
        num_samples (int): Maximum number of shared spatial tokens in sampled mode.
        sample_indices (torch.Tensor | None): Optional deterministic token indices overriding random sampling.

    Returns:
        (torch.Tensor): Mean absolute difference between student and teacher relation matrices.
    """
    _validate_feature_pair(student_feat, teacher_feat)
    if mode not in _RELATION_MODES:
        raise ValueError(f"mode must be one of {sorted(_RELATION_MODES)}, got {mode!r}.")
    if mode == "full" and sample_indices is not None:
        raise ValueError("sample_indices is only supported when mode='sampled'.")

    batch, channels, height, width = student_feat.shape
    num_tokens = height * width
    weights = (
        _validate_token_weights(token_weights, batch, num_tokens, student_feat.device)
        if token_weights is not None
        else None
    )
    indices = _resolve_relation_indices(
        num_tokens,
        None if mode == "full" else num_samples,
        sample_indices,
        student_feat.device,
    )
    device_type = student_feat.device.type
    with disabled_autocast(device_type):
        student = student_feat.float().reshape(batch, channels, num_tokens)[:, :, indices]
        teacher = teacher_feat.detach().float().reshape(batch, channels, num_tokens)[:, :, indices]
        student = F.normalize(student, p=2, dim=1, eps=1e-6)
        teacher = F.normalize(teacher, p=2, dim=1, eps=1e-6)
        student_gram = torch.bmm(student.transpose(1, 2), student)
        teacher_gram = torch.bmm(teacher.transpose(1, 2), teacher)
        difference = (student_gram - teacher_gram).abs()
        if weights is None:
            loss = difference.mean()
        else:
            pair_weights = weights[:, indices].unsqueeze(2) * weights[:, indices].unsqueeze(1)
            loss = (difference * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)
    if not torch.isfinite(loss):
        raise ValueError("relational KD loss is NaN or Inf.")
    return loss


def hybrid_kd_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    *,
    cosine_weight: float = 1.0,
    relation_weight: float = 1.0,
    relation_mode: str = "sampled",
    relation_samples: int = 256,
    sample_indices: torch.Tensor | None = None,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine cosine and relational Foundation KD losses."""
    cosine_weight = _validate_weight("cosine_weight", cosine_weight)
    relation_weight = _validate_weight("relation_weight", relation_weight)
    _validate_feature_pair(student_feat, teacher_feat)
    if cosine_weight == 0 and relation_weight == 0:
        return student_feat.sum() * 0.0
    cosine = (
        cosine_kd_loss(student_feat, teacher_feat, token_weights=token_weights)
        if cosine_weight
        else student_feat.sum() * 0.0
    )
    relation = (
        relational_kd_loss(
            student_feat,
            teacher_feat,
            mode=relation_mode,
            num_samples=relation_samples,
            sample_indices=sample_indices,
            token_weights=token_weights,
        )
        if relation_weight
        else student_feat.sum() * 0.0
    )
    loss = cosine_weight * cosine + relation_weight * relation
    if not torch.isfinite(loss):
        raise ValueError("hybrid KD loss is NaN or Inf.")
    return loss


__all__ = ["cosine_kd_loss", "foreground_token_weights", "hybrid_kd_loss", "relational_kd_loss"]
