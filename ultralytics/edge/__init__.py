"""
Edge Deployment Unified API (P2-1)

Wraps the existing exporter with a simplified, profile-aware API for
one-command edge deployment across ONNX/NCNN/MNN/TFLite/EdgeTPU.

Usage:
    from ultralytics.edge import deploy_for_edge
    deploy_for_edge("yolov8n.pt", formats=["onnx", "ncnn"], profile="visdrone")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ultralytics.utils import LOGGER


@dataclass
class EdgeDeployProfile:
    """Pre-configured deployment profile for a specific edge scenario."""

    name: str
    imgsz: int = 640
    half: bool = False
    int8: bool = False
    simplify: bool = True
    opset: int = 12
    conf: float = 0.25
    iou: float = 0.45
    metadata: dict[str, str] = field(default_factory=dict)


# Built-in profiles
BUILTIN_PROFILES: dict[str, EdgeDeployProfile] = {
    "default": EdgeDeployProfile(name="default"),
    "visdrone": EdgeDeployProfile(
        name="visdrone",
        imgsz=960,
        conf=0.20,
        iou=0.55,
        metadata={"scenario": "drone aerial detection"},
    ),
    "sku110k": EdgeDeployProfile(
        name="sku110k",
        imgsz=1280,
        conf=0.25,
        iou=0.60,
        metadata={"scenario": "dense retail shelf detection"},
    ),
    "rpi": EdgeDeployProfile(
        name="rpi",
        imgsz=320,
        half=False,
        int8=True,
        metadata={"scenario": "Raspberry Pi real-time"},
    ),
    "jetson": EdgeDeployProfile(
        name="jetson",
        imgsz=640,
        half=True,
        metadata={"scenario": "Jetson Nano/Orin FP16"},
    ),
    "mobile": EdgeDeployProfile(
        name="mobile",
        imgsz=416,
        int8=True,
        simplify=True,
        metadata={"scenario": "mobile NCNN/MNN"},
    ),
}

SUPPORTED_FORMATS = ("onnx", "ncnn", "mnn", "tflite", "edgetpu", "openvino", "engine", "coreml")


def get_profile(name: str) -> EdgeDeployProfile:
    """Get a built-in or custom edge deployment profile."""
    if name not in BUILTIN_PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {sorted(BUILTIN_PROFILES)}")
    return BUILTIN_PROFILES[name]


def deploy_for_edge(
    model_path: str | Path,
    formats: list[str] | None = None,
    profile: str | EdgeDeployProfile = "default",
    output_dir: str | Path | None = None,
    **overrides: Any,
) -> dict[str, Path]:
    """Export a model for edge deployment with a single call.

    Args:
        model_path: Path to the YOLO model checkpoint (.pt).
        formats: List of export formats (default: ["onnx"]).
        profile: Profile name or EdgeDeployProfile instance.
        output_dir: Directory for exported models (default: alongside model).
        **overrides: Override profile fields (e.g., imgsz=416, half=True).

    Returns:
        Dict mapping format → exported file path.
    """
    from ultralytics import YOLO

    if isinstance(profile, str):
        profile = get_profile(profile)

    # Apply overrides
    for key, val in overrides.items():
        if hasattr(profile, key):
            setattr(profile, key, val)

    formats = formats or ["onnx"]
    invalid = [f for f in formats if f not in SUPPORTED_FORMATS]
    if invalid:
        raise ValueError(f"Unsupported formats: {invalid}. Supported: {SUPPORTED_FORMATS}")

    model = YOLO(str(model_path))
    results: dict[str, Path] = {}

    for fmt in formats:
        export_kwargs = {
            "format": fmt,
            "imgsz": profile.imgsz,
            "half": profile.half,
            "int8": profile.int8,
        }
        if fmt == "onnx":
            export_kwargs["opset"] = profile.opset
            export_kwargs["simplify"] = profile.simplify

        LOGGER.info(f"[Edge Deploy] Exporting {model_path} → {fmt} (profile={profile.name})")
        try:
            path = model.export(**export_kwargs)
            results[fmt] = Path(path)
            LOGGER.info(f"[Edge Deploy] ✓ {fmt}: {path}")
        except Exception as exc:
            LOGGER.error(f"[Edge Deploy] ✗ {fmt} failed: {exc}")

    return results


def benchmark_edge_model(
    model_path: str | Path,
    imgsz: int = 640,
    warmup: int = 10,
    runs: int = 100,
) -> dict[str, float]:
    """Benchmark an exported model's inference latency.

    Args:
        model_path: Path to exported model (ONNX, NCNN, etc.).
        imgsz: Input image size.
        warmup: Number of warmup iterations.
        runs: Number of timed iterations.

    Returns:
        Dict with latency statistics (mean_ms, p50_ms, p95_ms, fps).
    """
    import time
    import numpy as np

    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup):
        model.predict(dummy, verbose=False)

    # Timed runs
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    mean_ms = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    return {
        "mean_ms": mean_ms,
        "p50_ms": p50,
        "p95_ms": p95,
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "runs": runs,
    }
