"""
MoE-Aware Quantization Utility (P2-2)

Provides post-training quantization (PTQ) helpers that are aware of MoE
routing layers — ensuring router weights retain higher precision while
expert weights can be aggressively quantized.

Usage:
    from ultralytics.nn.modules.moe.quantize import quantize_moe_model
    quantize_moe_model(model, backend="onnx", calibration_loader=loader)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER

from .api import collect_all_moe_info, get_routing_weights_unified
from .utils import is_core_moe_block


# Layers that should retain FP16 or higher precision during quantization
_SENSITIVE_LAYERS = (
    "routing", "router", "gate", "se_gate",
    "moe_loss_fn", "balance_loss_coeff",
    "complexity_estimator", "static_net",
)


def _is_routing_layer(name: str) -> bool:
    """Check if a parameter belongs to a routing/sensitive layer."""
    return any(s in name for s in _SENSITIVE_LAYERS)


def get_quantization_plan(model: nn.Module) -> dict[str, str]:
    """Generate a per-layer quantization plan for a MoE model.

    Routing layers → "fp16" (retain precision)
    Expert weights  → "int8" (aggressive quantization)
    Other layers    → "int8" (standard)

    Returns:
        Dict mapping parameter name → quantization dtype.
    """
    plan: dict[str, str] = {}
    for name, param in model.named_parameters():
        if _is_routing_layer(name):
            plan[name] = "fp16"
        elif "expert" in name:
            plan[name] = "int8"
        else:
            plan[name] = "int8"
    return plan


def _onnx_routing_nodes(onnx_path: str | Path) -> list[str]:
    """Return ONNX node names that belong to router/control paths.

    Parameter names are not preserved one-to-one in an exported graph, so the
    ONNX pass uses the same conservative routing tokens as the PyTorch plan.
    Unnamed nodes are ignored rather than accidentally excluding the whole
    graph.
    """
    try:
        import onnx
    except ImportError:
        return []
    try:
        graph = onnx.load(str(onnx_path)).graph
    except Exception as exc:
        LOGGER.warning(f"[Quantize] Could not inspect ONNX graph for routing nodes: {exc}")
        return []
    tokens = tuple(token.lower() for token in _SENSITIVE_LAYERS)
    return sorted({node.name for node in graph.node if node.name and any(token in node.name.lower() for token in tokens)})


def quantize_moe_model(
    model: nn.Module,
    backend: str = "onnx",
    calibration_loader: Iterator[torch.Tensor] | None = None,
    output_path: str | Path | None = None,
    dynamic_quantize: bool = True,
) -> Path | nn.Module:
    """Apply MoE-aware quantization to a model.

    Args:
        model: PyTorch model with MoE layers.
        backend: "onnx" for ONNX Runtime quantization, "torch" for PyTorch native.
        calibration_loader: Data loader for calibration (PTQ). If None, uses random data.
        output_path: Save path for quantized model (ONNX backend).
        dynamic_quantize: If True, use dynamic quantization (weight-only).

    Returns:
        Path to quantized model file (ONNX) or quantized nn.Module (torch).
    """
    plan = get_quantization_plan(model)
    routing_params = sum(1 for v in plan.values() if v == "fp16")
    expert_params = sum(1 for v in plan.values() if v == "int8")
    LOGGER.info(f"[Quantize] Plan: {routing_params} routing params (fp16), "
          f"{expert_params} expert/other params (int8)")

    if backend == "onnx":
        return _quantize_onnx(model, calibration_loader, output_path, dynamic_quantize)
    elif backend == "torch":
        return _quantize_torch(model, plan)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'onnx' or 'torch'.")


def _quantize_onnx(
    model: nn.Module,
    calibration_loader: Iterator[torch.Tensor] | None,
    output_path: str | Path | None,
    dynamic_quantize: bool,
) -> Path:
    """Export to ONNX and apply quantization."""
    from ultralytics.engine.exporter import Exporter

    # First export to ONNX
    if output_path is None:
        output_path = "model_quantized.onnx"
    output_path = Path(output_path)

    # Export model
    import io
    buffer = io.BytesIO()
    dummy = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        model, dummy, buffer,
        input_names=["images"],
        output_names=["output0"],
        opset=13,
        dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
    )

    # Apply quantization
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, CalibrationDataReader
    except ImportError:
        LOGGER.warning("[Quantize] onnxruntime not installed — returning unquantized ONNX")
        buffer.seek(0)
        output_path.write_bytes(buffer.read())
        return output_path

    # Save temp ONNX
    temp_path = output_path.parent / "model_temp.onnx"
    buffer.seek(0)
    temp_path.write_bytes(buffer.read())
    excluded_nodes = _onnx_routing_nodes(temp_path)
    if excluded_nodes:
        LOGGER.info(f"[Quantize] Preserving {len(excluded_nodes)} routing/control ONNX nodes at full precision.")

    if dynamic_quantize:
        # Dynamic quantization (weight-only, no calibration needed)
        # Exclude routing nodes from quantization
        quantize_dynamic(
            str(temp_path), str(output_path),
            weight_type=QuantType.QInt8,
            nodes_to_exclude=excluded_nodes,
        )
    else:
        # Static quantization with calibration
        class _CalibReader(CalibrationDataReader):
            def __init__(self, loader, max_items=100):
                self._items = []
                count = 0
                if loader is not None:
                    for batch in loader:
                        if count >= max_items:
                            break
                        if isinstance(batch, dict):
                            batch = batch.get("img", batch.get("images", batch.get("image")))
                        elif isinstance(batch, (tuple, list)):
                            batch = batch[0] if batch else None
                        if isinstance(batch, torch.Tensor):
                            self._items.append({"images": batch.detach().cpu().float().numpy()})
                        count += 1
                if not self._items:
                    raise ValueError(
                        "Static MoE quantization requires a non-empty representative "
                        "calibration_loader; random calibration is disabled."
                    )

            def get_next(self):
                return self._items.pop(0) if self._items else None

        reader = _CalibReader(calibration_loader)
        quantize_static(
            str(temp_path), str(output_path),
            calibration_data_reader=reader,
            nodes_to_exclude=excluded_nodes,
        )

    temp_path.unlink(missing_ok=True)
    LOGGER.info(f"[Quantize] ONNX quantized model saved to {output_path}")
    return output_path


def _quantize_torch(model: nn.Module, plan: dict[str, str]) -> nn.Module:
    """Apply PyTorch native dynamic quantization with MoE-awareness.

    Note: PyTorch dynamic quantization only supports nn.Linear and nn.LSTM.
    For conv layers, this falls back to FP16 for routing and keeps INT8 for
    nothing (PyTorch limitation). Use ONNX backend for full int8 support.
    """
    # Collect Linear module names that are NOT routing layers
    non_routing_linears = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not _is_routing_layer(name):
            non_routing_linears.append(name)

    if non_routing_linears:
        from torch.quantization import quantize_dynamic
        q_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        LOGGER.info(f"[Quantize] Torch dynamic quantization applied to {len(non_routing_linears)} Linear layers")
        return q_model
    else:
        LOGGER.info("[Quantize] No quantizable Linear layers found — model unchanged")
        return model


def estimate_size_reduction(model: nn.Module) -> dict[str, float]:
    """Estimate model size reduction from quantization.

    Returns:
        Dict with fp32_size_mb, estimated_int8_size_mb, reduction_pct.
    """
    total_params = sum(p.numel() for p in model.parameters())
    routing_params = sum(
        p.numel() for name, p in model.named_parameters()
        if _is_routing_layer(name)
    )
    expert_params = total_params - routing_params

    # FP32: 4 bytes/param, FP16: 2 bytes/param, INT8: 1 byte/param
    fp32_bytes = total_params * 4
    mixed_bytes = (routing_params * 2) + (expert_params * 1)  # fp16 routing + int8 experts
    reduction = (1 - mixed_bytes / fp32_bytes) * 100

    return {
        "fp32_size_mb": fp32_bytes / (1024 * 1024),
        "estimated_mixed_size_mb": mixed_bytes / (1024 * 1024),
        "reduction_pct": reduction,
        "routing_params": routing_params,
        "expert_params": expert_params,
    }
