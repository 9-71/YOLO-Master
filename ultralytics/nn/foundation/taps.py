"""Student-side feature capture utilities for Foundation distillation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ultralytics.nn.modules.head import Detect


_FEATURE_LEVEL_OFFSETS = {"p3": 0, "p4": 1, "p5": 2}


class StudentFeatureTap:
    """Capture one FPN feature from a YOLO Detect head without changing the student graph.

    The source layer is resolved from the Detect head's ``f`` attribute in ``[P3, P4, P5]`` order. Captured
    tensors are intentionally not detached so a later distillation loss can backpropagate through the student.

    Args:
        student_model (nn.Module): A YOLO model, DetectionModel, or module container exposing ``.model``.
        target (str): Feature level to capture: ``p3``, ``p4``, or ``p5``.
    """

    def __init__(self, student_model: nn.Module, target: str = "p4") -> None:
        if not isinstance(student_model, nn.Module):
            raise TypeError(f"student_model must be an nn.Module, got {type(student_model).__name__}.")
        if not isinstance(target, str) or target.lower() not in _FEATURE_LEVEL_OFFSETS:
            raise ValueError(f"target must be one of {sorted(_FEATURE_LEVEL_OFFSETS)}, got {target!r}.")

        self.target = target.lower()
        self._layers = self._resolve_layers(student_model)
        self.head_index, self._source_indices = self._find_detect_sources(self._layers)
        self.source_index = self._source_indices[_FEATURE_LEVEL_OFFSETS[self.target]]
        self.source_layer = self._layers[self.source_index]
        self._captured: torch.Tensor | None = None
        self._closed = False
        self._hook_handle = self.source_layer.register_forward_hook(self._hook_fn)

    @staticmethod
    def _resolve_layers(student_model: nn.Module) -> Sequence[nn.Module]:
        """Resolve a YOLO layer container from a model wrapper or bare sequential container."""
        candidate: Any = student_model
        for _ in range(3):
            if isinstance(candidate, (nn.Sequential, nn.ModuleList)):
                return candidate
            nested = getattr(candidate, "model", None)
            if nested is None or nested is candidate:
                break
            candidate = nested
        if isinstance(candidate, (nn.Sequential, nn.ModuleList)):
            return candidate
        if isinstance(candidate, Sequence) and all(isinstance(layer, nn.Module) for layer in candidate):
            return candidate
        raise TypeError("student_model must expose a Sequential/ModuleList layer container via '.model'.")

    @classmethod
    def _find_detect_sources(cls, layers: Sequence[nn.Module]) -> tuple[int, list[int]]:
        """Find a Detect head and resolve its feature-source indices in model-graph coordinates."""
        for head_index, layer in enumerate(layers):
            if not isinstance(layer, Detect):
                continue
            raw_sources = getattr(layer, "f", None)
            if isinstance(raw_sources, bool) or isinstance(raw_sources, int) or raw_sources is None:
                raise ValueError("Detect head must expose a sequence of at least three feature-source indices in 'f'.")
            try:
                raw_sources = list(raw_sources)
            except TypeError as exc:
                raise ValueError("Detect head 'f' must be an iterable of feature-source indices.") from exc
            if len(raw_sources) < 3:
                raise ValueError(
                    f"Detect head must expose at least three feature-source indices in 'f', got {raw_sources!r}."
                )

            source_indices = []
            for raw_index in raw_sources:
                if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                    raise TypeError(f"Detect head feature-source indices must be integers, got {raw_index!r}.")
                index = head_index + raw_index if raw_index < 0 else raw_index
                if index < 0 or index >= len(layers) or index == head_index:
                    raise ValueError(
                        f"Detect head feature-source index {raw_index!r} resolves outside the student layers."
                    )
                source_indices.append(index)
            return head_index, source_indices
        raise ValueError("No Detect head found in student model.")

    @staticmethod
    def _as_feature(output: Any) -> torch.Tensor:
        """Extract a tensor output while rejecting ambiguous multi-output modules."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)):
            tensors = [item for item in output if isinstance(item, torch.Tensor)]
            if len(tensors) == 1:
                return tensors[0]
            raise TypeError("Student P-level source output must contain exactly one tensor.")
        raise TypeError(f"Student P-level source output must be a torch.Tensor, got {type(output).__name__}.")

    def _hook_fn(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        """Store the live feature tensor, preserving its autograd graph for distillation."""
        self._captured = self._as_feature(output)

    @property
    def source_indices(self) -> tuple[int, ...]:
        """Return resolved source layer indices in P3/P4/P5 order."""
        return tuple(self._source_indices)

    @property
    def has_feature(self) -> bool:
        """Return whether a forward pass has populated the requested feature."""
        return self._captured is not None

    @property
    def feature(self) -> torch.Tensor:
        """Return the captured BCHW feature, or raise if no valid feature is available."""
        if self._captured is None:
            raise RuntimeError(f"No feature captured for {self.target}; run the student forward pass first.")
        if self._captured.ndim != 4:
            raise RuntimeError(
                f"Student feature '{self.target}' must have shape (B, C, H, W), got {tuple(self._captured.shape)}."
            )
        return self._captured

    def clear(self) -> None:
        """Clear the previous capture before a new student forward pass."""
        self._captured = None

    def close(self) -> None:
        """Remove the forward hook and release the captured feature."""
        if not self._closed:
            self._hook_handle.remove()
            self._closed = True
        self.clear()

    def __enter__(self) -> "StudentFeatureTap":
        """Return the active tap for context-manager usage."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Remove the hook when leaving a context manager."""
        self.close()

    def __del__(self):
        """Best-effort hook cleanup for short-lived taps."""
        try:
            self.close()
        except (AttributeError, RuntimeError):
            pass


__all__ = ["StudentFeatureTap"]
