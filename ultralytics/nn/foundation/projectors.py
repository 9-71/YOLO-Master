"""Feature alignment projectors for Foundation distillation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _validate_channels(name: str, value: int) -> int:
    """Validate a positive channel count or alignment dimension."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _validate_feature(name: str, feature: torch.Tensor, channels: int) -> None:
    """Validate a feature tensor before it enters a projector."""
    if not isinstance(feature, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(feature).__name__}.")
    if feature.ndim != 4:
        raise ValueError(f"{name} must have shape (B, C, H, W), got {tuple(feature.shape)}.")
    if feature.shape[1] != channels:
        raise ValueError(f"{name} has {feature.shape[1]} channels; expected {channels}.")
    if feature.shape[-2] <= 0 or feature.shape[-1] <= 0:
        raise ValueError(f"{name} must have positive spatial dimensions, got {tuple(feature.shape[-2:])}.")


class P4AlignmentProjector(nn.Module):
    """Align student and Foundation teacher P4 features in a wrapper-owned space.

    The student path is trainable. The teacher path is detached and uses a fixed, bias-free projection when its
    channel count differs from ``align_dim``. Spatial alignment resizes only the teacher feature to the student's
    resolution, preserving the student P4 grid for later distillation losses.

    Args:
        student_channels (int): Number of channels in the captured student P4 feature.
        teacher_channels (int): Number of channels in the Foundation teacher P4 feature.
        align_dim (int): Shared feature dimension returned by both paths.
        use_norm (bool): Add BatchNorm2d after the trainable student projection.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        align_dim: int = 256,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.student_channels = _validate_channels("student_channels", student_channels)
        self.teacher_channels = _validate_channels("teacher_channels", teacher_channels)
        self.align_dim = _validate_channels("align_dim", align_dim)
        if not isinstance(use_norm, bool):
            raise TypeError(f"use_norm must be a bool, got {type(use_norm).__name__}.")
        self.use_norm = use_norm

        self.student_proj = nn.Sequential(
            nn.Conv2d(self.student_channels, self.align_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.align_dim) if use_norm else nn.Identity(),
        )
        if self.teacher_channels == self.align_dim:
            self.teacher_proj: nn.Module = nn.Identity()
        else:
            self.teacher_proj = nn.Conv2d(self.teacher_channels, self.align_dim, kernel_size=1, bias=False)
            self.teacher_proj.requires_grad_(False)

        self._alignment: dict[str, Any] = {
            "student_size": None,
            "teacher_size": None,
            "target_size": None,
            "teacher_resized": False,
            "resize_ratio": None,
        }

    @property
    def teacher_projection_frozen(self) -> bool:
        """Return whether every teacher projection parameter is non-trainable."""
        return all(not parameter.requires_grad for parameter in self.teacher_proj.parameters())

    @property
    def alignment(self) -> dict[str, Any]:
        """Return metadata for the most recent spatial alignment operation."""
        return dict(self._alignment)

    def _project_teacher(self, teacher_feat: torch.Tensor) -> torch.Tensor:
        """Apply the frozen teacher projection with a compatible device and dtype."""
        parameters = list(self.teacher_proj.parameters())
        if parameters:
            parameter = parameters[0]
            if teacher_feat.device != parameter.device or teacher_feat.dtype != parameter.dtype:
                teacher_feat = teacher_feat.to(device=parameter.device, dtype=parameter.dtype)
        return self.teacher_proj(teacher_feat)

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return aligned student and detached teacher features in BCHW layout."""
        _validate_feature("student_feat", student_feat, self.student_channels)
        _validate_feature("teacher_feat", teacher_feat, self.teacher_channels)
        if student_feat.shape[0] != teacher_feat.shape[0]:
            raise ValueError(
                f"student_feat and teacher_feat batch sizes must match, got {student_feat.shape[0]} and "
                f"{teacher_feat.shape[0]}."
            )

        student_size = tuple(student_feat.shape[-2:])
        teacher_size = tuple(teacher_feat.shape[-2:])
        teacher_resized = student_size != teacher_size
        resize_ratio = (
            (student_size[0] / teacher_size[0], student_size[1] / teacher_size[1]) if teacher_resized else None
        )
        if teacher_resized:
            teacher_feat = F.interpolate(teacher_feat.detach(), size=student_size, mode="bilinear", align_corners=False)
        else:
            teacher_feat = teacher_feat.detach()
        self._alignment = {
            "student_size": student_size,
            "teacher_size": teacher_size,
            "target_size": student_size,
            "teacher_resized": teacher_resized,
            "resize_ratio": resize_ratio,
        }

        student_aligned = self.student_proj(student_feat)
        teacher_aligned = self._project_teacher(teacher_feat)
        if teacher_aligned.device != student_aligned.device or teacher_aligned.dtype != student_aligned.dtype:
            teacher_aligned = teacher_aligned.to(device=student_aligned.device, dtype=student_aligned.dtype)
        return student_aligned, teacher_aligned

    def train(self, mode: bool = True):
        """Set student modules to the requested mode while preserving frozen teacher parameters."""
        super().train(mode)
        self.teacher_proj.requires_grad_(False)
        return self


__all__ = ["P4AlignmentProjector"]
