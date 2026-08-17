"""Shared protocol and output container for Foundation Teacher backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class FoundationFeatures:
    """Normalized output returned by every Foundation Teacher.

    Args:
        dense (dict[str, torch.Tensor]): Named spatial features in ``(B, C, H, W)`` layout.
        pooled (torch.Tensor | None): Optional global representation in ``(B, C)`` layout.
        semantic (torch.Tensor | None): Optional semantic representation in ``(B, C)`` layout.
        metadata (dict[str, Any]): Backend-specific shape and preprocessing metadata.
    """

    dense: dict[str, torch.Tensor]
    pooled: torch.Tensor | None = None
    semantic: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the transport container without inspecting model-specific output details."""
        if not isinstance(self.dense, dict):
            raise TypeError(f"dense must be a dict[str, torch.Tensor], got {type(self.dense).__name__}.")
        for name, feature in self.dense.items():
            if not isinstance(name, str) or not name:
                raise TypeError("dense feature names must be non-empty strings.")
            if not isinstance(feature, torch.Tensor):
                raise TypeError(f"dense['{name}'] must be a torch.Tensor, got {type(feature).__name__}.")
            if feature.ndim != 4:
                raise ValueError(f"dense['{name}'] must have shape (B, C, H, W), got {tuple(feature.shape)}.")
        for name, value in (("pooled", self.pooled), ("semantic", self.semantic)):
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor or None, got {type(value).__name__}.")
            if value is not None and value.ndim != 2:
                raise ValueError(f"{name} must have shape (B, C), got {tuple(value.shape)}.")
        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be a dict, got {type(self.metadata).__name__}.")


@runtime_checkable
class FoundationTeacher(Protocol):
    """Structural protocol implemented by training-only Foundation Teacher backends."""

    name: str

    def freeze(self) -> None:
        """Freeze teacher parameters and keep the backend in evaluation mode."""

    def preprocess(self, images: torch.Tensor) -> Any:
        """Convert a YOLO image batch into the backend's input representation."""

    def encode(self, images: torch.Tensor) -> FoundationFeatures:
        """Encode images and return normalized Foundation features."""

    def to(self, device=None, dtype=None):
        """Move the teacher to a device and/or dtype and return the backend."""


__all__ = ["FoundationFeatures", "FoundationTeacher"]
