"""Multi-Foundation Teacher manager used by the F14 routing phase."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

import torch
import torch.nn as nn

from ..protocol import FoundationFeatures


def _as_features(value: Any, name: str) -> FoundationFeatures | Mapping[str, Any]:
    """Validate a named teacher response without imposing backend-specific classes."""

    if isinstance(value, FoundationFeatures):
        return value
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{name} teacher encode() must return FoundationFeatures or a mapping, got {type(value).__name__}.")


class MultiFoundationTeacher(nn.Module):
    """Training-only manager combining frozen DINOv3 and SigLIP2 backends.

    The backends are deliberately stored in ``__dict__`` rather than registered
    as child modules. This keeps the manager safe when it is attached to the
    Foundation wrapper as an external training-only object and makes the
    no-teacher checkpoint/export contract explicit.
    """

    name = "multi"

    def __init__(self, dinov3: Any, siglip2: Any) -> None:
        super().__init__()
        if dinov3 is None or siglip2 is None:
            raise ValueError("F14 MultiFoundationTeacher requires both dinov3 and siglip2 backends.")
        self.__dict__["_dinov3"] = dinov3
        self.__dict__["_siglip2"] = siglip2

    @property
    def dinov3(self) -> Any:
        """Return the DINOv3 backend."""

        return self.__dict__["_dinov3"]

    @property
    def siglip2(self) -> Any:
        """Return the SigLIP2 backend."""

        return self.__dict__["_siglip2"]

    @property
    def teachers(self) -> dict[str, Any]:
        """Return named teacher backends for diagnostics and metadata only."""

        return {"dinov3": self.dinov3, "siglip2": self.siglip2}

    def freeze(self) -> None:
        """Freeze both backends and force evaluation mode."""

        for teacher in self.teachers.values():
            freeze = getattr(teacher, "freeze", None)
            if callable(freeze):
                freeze()
            else:
                train = getattr(teacher, "train", None)
                if callable(train):
                    train(False)
                if isinstance(teacher, nn.Module):
                    teacher.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True):
        """Keep both Foundation backends frozen/eval when the student trains."""

        self.freeze()
        return self

    def to(self, device=None, dtype=None, *args, **kwargs):
        """Move both external backends and retain the manager API."""

        for teacher in self.teachers.values():
            mover = getattr(teacher, "to", None)
            if callable(mover):
                try:
                    mover(device=device, dtype=dtype, *args, **kwargs)
                except TypeError:
                    mover(device, *args, **kwargs)
        self.freeze()
        return self

    def encode(self, images: torch.Tensor) -> FoundationFeatures:
        """Encode one batch with both teachers and expose named responses in metadata."""

        dino = _as_features(self.dinov3.encode(images), "dinov3")
        siglip = _as_features(self.siglip2.encode(images), "siglip2")
        dino_dense = dino.dense if isinstance(dino, FoundationFeatures) else dino.get("dense")
        dino_pooled = dino.pooled if isinstance(dino, FoundationFeatures) else dino.get("pooled")
        siglip_semantic = siglip.semantic if isinstance(siglip, FoundationFeatures) else siglip.get("semantic")
        if not isinstance(dino_dense, Mapping) or "p4" not in dino_dense:
            raise ValueError("F14 DINOv3 backend must expose dense['p4'] features.")
        if not isinstance(siglip_semantic, torch.Tensor) or siglip_semantic.ndim != 2:
            raise ValueError("F14 SigLIP2 backend must expose semantic features with shape [B,C].")
        metadata = {
            "teacher_names": ("dinov3", "siglip2"),
            "teachers": {"dinov3": dino, "siglip2": siglip},
            "dinov3_metadata": dict(dino.metadata) if isinstance(dino, FoundationFeatures) else {},
            "siglip2_metadata": dict(siglip.metadata) if isinstance(siglip, FoundationFeatures) else {},
        }
        return FoundationFeatures(
            dense=dict(dino_dense),
            pooled=dino_pooled if isinstance(dino_pooled, torch.Tensor) else None,
            semantic=siglip_semantic,
            metadata=metadata,
        )

    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        """Delegate closed-set text prototypes to SigLIP2."""

        encode_text = getattr(self.siglip2, "encode_text", None)
        if not callable(encode_text):
            raise AttributeError("F14 SigLIP2 backend must expose encode_text() for semantic prototypes.")
        return encode_text(prompts)


__all__ = ["MultiFoundationTeacher"]
