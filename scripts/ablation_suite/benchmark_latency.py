"""
推理延迟基准脚本 — YOLO-Master Baseline / LoRA / MoLoRA 多后端对比
=====================================================================

目的：系统测量 Baseline、标准 LoRA 与 MoLoRA 在 merged / unmerged 状态下
      于 PyTorch eager、ONNX Runtime、TensorRT 三种后端上的端到端推理延迟。

测试矩阵（3 方法 × 2 merge 状态 × 3 后端 = 18 个组合，部分组合不可行）：
  ┌──────────┬─────────┬─────────────────┬─────────────────┬─────────────────┐
  │  方法    │  merge  │ PyTorch eager   │ ONNX Runtime    │ TensorRT        │
  ├──────────┼─────────┼─────────────────┼─────────────────┼─────────────────┤
  │ Baseline │    —    │        ✓        │        ✓        │        ✓        │
  │ LoRA     │ unmerged│        ✓        │  尝试(可能失败) │  尝试(可能失败) │
  │ LoRA     │  merged │        ✓        │        ✓        │        ✓        │
  │ MoLoRA   │ unmerged│        ✓        │  尝试(可能失败) │  尝试(可能失败) │
  │ MoLoRA   │  merged │        ✓        │        ✓        │        ✓        │
  └──────────┴─────────┴─────────────────┴─────────────────┴─────────────────┘

环境：
    - 设备优先 MPS (Apple Silicon), 回退 CUDA / CPU
    - 禁用 WandB、YOLO autoinstall/verbose
    - 数据集: coco128.yaml（仅用于基准环境校验，不实际训练）
    - 模型: YOLO-Master-EsMoE-N.pt（项目根目录）

输出：
    - JSON 结果文件（实时写入）
    - 控制台汇总表（含 mean ± std / median / p95 / FPS）

用法：
    python scripts/benchmark_latency.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 环境初始化 —— 必须在 import ultralytics 之前完成
# ═══════════════════════════════════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")

# 所有导入在设置 sys.path 之后
import numpy as np
import torch
import torch.nn as nn

# 关闭 ultralytics 的 wandb 上报
from ultralytics.utils import SETTINGS

SETTINGS["wandb"] = False

from ultralytics import YOLO
from ultralytics.utils import ASSETS

# MoLoRA 基础设施
from ultralytics.nn.peft.molora import (
    MoLoRAConfig,
    get_peft_molora_model,
)
from ultralytics.nn.peft.molora.layer import MoLoRALayer

# LoRA 基础设施
from ultralytics.utils.lora import apply_lora, merge_lora_weights
from ultralytics.utils.lora.config import LoRAConfig

# 确认加载的是当前仓库的 ultralytics
import ultralytics

assert str(REPO_ROOT) in ultralytics.__file__, f"加载的不是当前仓库的 ultralytics！got {ultralytics.__file__}"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 全局配置
# ═══════════════════════════════════════════════════════════════════════════════

HERE = Path(__file__).parent
MODEL_PATH = str(REPO_ROOT / "YOLO-Master-EsMoE-N.pt")
DATA_YAML = "coco128.yaml"  # ultralytics 内置数据集，用于环境校验
RESULTS_JSON = HERE / "benchmark_latency_results.json"

# 图像尺寸与批次
IMGSZ = 320
BATCH = 1

# 延迟测量参数
WARMUP_RUNS = 10  # 预热轮数
TIMED_RUNS = 50  # 正式测量轮数
MIN_TIME_SEC = 10.0  # 最少测量时长（秒）

# 设备选择: MPS优先 → CUDA → CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

# 强制行缓冲，便于实时观察后台输出
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# PEFT / MoLoRA 公共超参
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 工具函数
# ═══════════════════════════════════════════════════════════════════════════════


def count_params(m: nn.Module) -> Tuple[int, int]:
    """统计模型总参数量与可训练参数量。"""
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def detect_adapter_signature(m: nn.Module) -> Dict[str, Any]:
    """
    检测模型中各类型 PEFT adapter 的存在性与数量。
    返回结构化字典，用于验证 PEFT 是否正确注入。
    """
    names = [n for n, _ in m.named_parameters()]
    return {
        "has_lora_A": any("lora_A" in n for n in names),
        "has_lora_B": any("lora_B" in n for n in names),
        "has_molora": any("molora" in n.lower() or "router" in n.lower() for n in names),
        "n_lora_params": sum(1 for n in names if "lora_" in n.lower()),
        "n_router_params": sum(1 for n in names if "router" in n.lower()),
    }


def set_merged_state(model: nn.Module, merged: bool) -> None:
    """
    统一设置模型的 merged / unmerged 状态。

    - LoRA (PEFT): 调用 merge_lora_weights() 进行 merge；unmerge 需要重新加载。
      由于 PEFT 的 merge_and_unload 是不可逆的，本脚本对 LoRA 的 unmerged 状态
      采用"重新应用 LoRA"的策略。
    - MoLoRA: 遍历所有 MoLoRALayer 调用 merge_weights() / unmerge_weights()。
    - Baseline: 无操作。
    """
    # MoLoRA: 支持 merge / unmerge
    molora_layers = [m for m in model.modules() if isinstance(m, MoLoRALayer)]
    if molora_layers:
        for layer in molora_layers:
            if merged:
                layer.merge_weights()
            else:
                layer.unmerge_weights()
        return

    # LoRA (fallback / manual): 目前不支持可逆 unmerge，由调用方重新加载模型处理
    # PEFT LoRA: merge_and_unload 不可逆，同样由调用方重新加载处理
    # 因此本函数对 LoRA 不做任何操作，状态切换在模型准备阶段完成。


def is_merged(model: nn.Module) -> bool:
    """检查模型当前是否处于 merged 状态。"""
    molora_layers = [m for m in model.modules() if isinstance(m, MoLoRALayer)]
    if molora_layers:
        return all(getattr(m, "merged", False) for m in molora_layers)
    return False


def _sigma_clip(data: np.ndarray, sigma: float = 2.0, max_iters: int = 3) -> np.ndarray:
    """迭代 sigma clipping，剔除异常值。"""
    data = np.array(data)
    for _ in range(max_iters):
        mean, std = np.mean(data), np.std(data)
        clipped = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
        if len(clipped) == len(data):
            break
        data = clipped
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 模型准备 —— Baseline / LoRA / MoLoRA
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_baseline() -> YOLO:
    """加载 Baseline 模型（无 PEFT）。"""
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    model.model.eval()
    return model


def prepare_lora(merged: bool = False) -> Optional[YOLO]:
    """
    加载并应用标准 LoRA，可选 merge。

    Args:
        merged: 是否将 LoRA adapter 合并到基权重中。

    Returns:
        应用了 LoRA 的 YOLO 模型；若失败返回 None。
    """
    try:
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)

        # 使用 ultralytics 内置 LoRA 应用器
        lora_cfg = LoRAConfig(
            r=LORA_R,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT,
            backend="peft",
        )
        model = apply_lora(model.model, lora_cfg)

        # apply_lora 返回的是 DetectionModel，需要重新包装
        if not isinstance(model, YOLO):
            yolo_model = YOLO(MODEL_PATH)
            yolo_model.model = model
            model = yolo_model

        model.to(DEVICE)
        model.model.eval()

        if merged:
            merge_lora_weights(model.model)

        return model
    except Exception as exc:
        print(f"[WARN] LoRA 准备失败 (merged={merged}): {exc}")
        traceback.print_exc()
        return None


def prepare_molora(merged: bool = False) -> Optional[YOLO]:
    """
    加载并应用标准 MoLoRA，可选 merge。

    Args:
        merged: 是否将所有 MoLoRALayer 的 expert delta 合并到基权重中。

    Returns:
        应用了 MoLoRA 的 YOLO 模型；若失败返回 None。
    """
    try:
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)

        cfg = MoLoRAConfig(
            r=LORA_R,
            alpha=LORA_ALPHA,
            num_experts=4,
            top_k=2,
            router_type="linear",
            dropout=LORA_DROPOUT,
            use_rslora=True,
            balance_loss_coef=0.01,
            z_loss_coef=0.001,
        )
        get_peft_molora_model(model.model, cfg)

        model.to(DEVICE)
        model.model.eval()

        if merged:
            for m in model.model.modules():
                if isinstance(m, MoLoRALayer):
                    m.merge_weights()

        return model
    except Exception as exc:
        print(f"[WARN] MoLoRA 准备失败 (merged={merged}): {exc}")
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 延迟测量 —— 三种后端
# ═══════════════════════════════════════════════════════════════════════════════


def _get_dummy_input_tensor(device: str, imgsz: int = IMGSZ) -> torch.Tensor:
    """构造与真实输入同形状的 dummy tensor 用于延迟测量。"""
    # YOLO 输入格式: [B, 3, H, W] (CHW)
    return torch.randn(BATCH, 3, imgsz, imgsz, device=device)


def measure_pytorch_eager(model: YOLO, imgsz: int = IMGSZ) -> Optional[Dict[str, float]]:
    """
    测量 PyTorch eager 模式下的单张图像推理延迟。

    测量方式：
      1. 构造 dummy tensor（与真实图片同 shape）
      2. 预热 WARMUP_RUNS 次
      3. 正式测量 TIMED_RUNS 次（或至少 MIN_TIME_SEC 秒）
      4. 返回 mean / std / median / p95 / min / max / fps

    Returns:
        含延迟指标的字典；若失败返回 None。
    """
    try:
        device = next(model.model.parameters()).device
        x = _get_dummy_input_tensor(str(device), imgsz)
        model.model.eval()

        # 预热
        with torch.no_grad():
            for _ in range(WARMUP_RUNS):
                _ = model.model(x)

        # 正式测量
        times = []
        t_start = time.perf_counter()
        with torch.no_grad():
            while True:
                t0 = time.perf_counter()
                _ = model.model(x)
                # MPS 需要同步才能获取准确时间
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)  # ms

                if len(times) >= TIMED_RUNS and (time.perf_counter() - t_start) >= MIN_TIME_SEC:
                    break

        times_arr = _sigma_clip(np.array(times))
        mean_ms = float(np.mean(times_arr))
        std_ms = float(np.std(times_arr))
        median_ms = float(np.median(times_arr))
        p95_ms = float(np.percentile(times_arr, 95))
        min_ms = float(np.min(times_arr))
        max_ms = float(np.max(times_arr))
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        return {
            "mean_ms": round(mean_ms, 3),
            "std_ms": round(std_ms, 3),
            "median_ms": round(median_ms, 3),
            "p95_ms": round(p95_ms, 3),
            "min_ms": round(min_ms, 3),
            "max_ms": round(max_ms, 3),
            "fps": round(fps, 2),
            "n_runs": len(times_arr),
        }
    except Exception as exc:
        print(f"[WARN] PyTorch eager 测量失败: {exc}")
        traceback.print_exc()
        return None


def measure_onnx_runtime(model: YOLO, imgsz: int = IMGSZ) -> Optional[Dict[str, float]]:
    """
    导出 ONNX 并使用 ONNX Runtime 测量延迟。

    流程：
      1. 调用 model.export(format="onnx", imgsz=imgsz, simplify=True)
      2. 创建 ort.InferenceSession
      3. 用 dummy numpy 输入进行 WARMUP + TIMED 测量

    Returns:
        含延迟指标的字典；若导出或推理失败返回 None。
    """
    try:
        # 1. 导出 ONNX
        onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True, verbose=False)
        if onnx_path is None or not Path(onnx_path).exists():
            print("[WARN] ONNX 导出失败或文件不存在")
            return None

        # 2. 创建 ONNX Runtime session
        try:
            import onnxruntime as ort
        except ImportError:
            print("[WARN] onnxruntime 未安装，跳过 ONNX 测量")
            return None

        # 选择 provider: CUDA > CPU (MPS 没有 ORT provider)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(onnx_path), sess_options, providers=providers)

        # 3. 构造输入
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        # 处理动态 shape
        if any(not isinstance(dim, int) or dim < 0 for dim in input_shape):
            input_shape = (BATCH, 3, imgsz, imgsz)
        x_np = np.random.rand(*input_shape).astype(np.float32)

        # 4. 预热
        for _ in range(WARMUP_RUNS):
            sess.run(None, {input_name: x_np})

        # 5. 正式测量
        times = []
        t_start = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            sess.run(None, {input_name: x_np})
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

            if len(times) >= TIMED_RUNS and (time.perf_counter() - t_start) >= MIN_TIME_SEC:
                break

        times_arr = _sigma_clip(np.array(times))
        mean_ms = float(np.mean(times_arr))
        std_ms = float(np.std(times_arr))
        median_ms = float(np.median(times_arr))
        p95_ms = float(np.percentile(times_arr, 95))
        min_ms = float(np.min(times_arr))
        max_ms = float(np.max(times_arr))
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        # 清理导出的 ONNX 文件（可选，保持 workspace 干净）
        try:
            Path(onnx_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "mean_ms": round(mean_ms, 3),
            "std_ms": round(std_ms, 3),
            "median_ms": round(median_ms, 3),
            "p95_ms": round(p95_ms, 3),
            "min_ms": round(min_ms, 3),
            "max_ms": round(max_ms, 3),
            "fps": round(fps, 2),
            "n_runs": len(times_arr),
            "onnx_path": str(onnx_path),
        }
    except Exception as exc:
        print(f"[WARN] ONNX Runtime 测量失败: {exc}")
        traceback.print_exc()
        return None


def measure_tensorrt(model: YOLO, imgsz: int = IMGSZ) -> Optional[Dict[str, float]]:
    """
    导出 TensorRT engine 并测量延迟。

    流程：
      1. 调用 model.export(format="engine", imgsz=imgsz)
      2. 用 YOLO(engine_file) 进行推理测量

    注意：TensorRT 仅在 NVIDIA GPU 上可用；MPS/CPU 环境会自动跳过。

    Returns:
        含延迟指标的字典；若导出或推理失败返回 None。
    """
    # TensorRT 仅 CUDA 可用
    if not torch.cuda.is_available():
        print("[INFO] 非 CUDA 环境，跳过 TensorRT 测量")
        return None

    try:
        # 1. 导出 engine
        engine_path = model.export(format="engine", imgsz=imgsz, verbose=False)
        if engine_path is None or not Path(engine_path).exists():
            print("[WARN] TensorRT engine 导出失败或文件不存在")
            return None

        # 2. 加载 engine 模型
        trt_model = YOLO(str(engine_path))

        # 3. 使用 bus.jpg 作为标准输入（与 ultralytics benchmark 保持一致）
        test_img = str(ASSETS / "bus.jpg")
        if not Path(test_img).exists():
            # fallback: dummy
            test_img = None

        # 4. 预热
        for _ in range(WARMUP_RUNS):
            if test_img:
                trt_model.predict(test_img, imgsz=imgsz, verbose=False)
            else:
                # dummy 路径不太可靠，优先用真实图片
                pass

        # 5. 正式测量：用 predict 返回结果的 speed["inference"]
        times = []
        t_start = time.perf_counter()
        while True:
            if test_img:
                results = trt_model.predict(test_img, imgsz=imgsz, verbose=False)
                inference_ms = results[0].speed.get("inference", 0.0)
                if inference_ms > 0:
                    times.append(inference_ms)
            else:
                break

            if len(times) >= TIMED_RUNS and (time.perf_counter() - t_start) >= MIN_TIME_SEC:
                break

        if not times:
            print("[WARN] TensorRT 未收集到有效延迟数据")
            return None

        times_arr = _sigma_clip(np.array(times))
        mean_ms = float(np.mean(times_arr))
        std_ms = float(np.std(times_arr))
        median_ms = float(np.median(times_arr))
        p95_ms = float(np.percentile(times_arr, 95))
        min_ms = float(np.min(times_arr))
        max_ms = float(np.max(times_arr))
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        # 清理 engine 文件
        try:
            Path(engine_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "mean_ms": round(mean_ms, 3),
            "std_ms": round(std_ms, 3),
            "median_ms": round(median_ms, 3),
            "p95_ms": round(p95_ms, 3),
            "min_ms": round(min_ms, 3),
            "max_ms": round(max_ms, 3),
            "fps": round(fps, 2),
            "n_runs": len(times_arr),
            "engine_path": str(engine_path),
        }
    except Exception as exc:
        print(f"[WARN] TensorRT 测量失败: {exc}")
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 实验变体定义与执行
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkSpec:
    """单个基准测试的规格定义。"""

    method: str  # baseline | lora | molora
    merged: bool  # merged / unmerged (baseline 无视此字段)
    backends: List[str] = field(default_factory=lambda: ["pytorch", "onnx", "tensorrt"])
    description: str = ""


BENCHMARK_MATRIX: List[BenchmarkSpec] = [
    BenchmarkSpec(method="baseline", merged=False, description="Baseline (no PEFT)"),
    BenchmarkSpec(method="lora", merged=False, description="LoRA unmerged (adapter active)"),
    BenchmarkSpec(method="lora", merged=True, description="LoRA merged (adapter fused)"),
    BenchmarkSpec(method="molora", merged=False, description="MoLoRA unmerged (top-k routing active)"),
    BenchmarkSpec(method="molora", merged=True, description="MoLoRA merged (experts fused into base)"),
]


# 后端名称映射
BACKEND_LABELS = {
    "pytorch": "PyTorch eager",
    "onnx": "ONNX Runtime",
    "tensorrt": "TensorRT",
}


def run_single_benchmark(spec: BenchmarkSpec, backend: str) -> Dict[str, Any]:
    """
    执行单个 (method × merged × backend) 组合的延迟测量。

    Returns:
        结构化结果字典，包含延迟指标、错误信息和元数据。
    """
    label = f"{spec.method}_{'merged' if spec.merged else 'unmerged'}_{backend}"
    print(f"\n[Benchmark] {label} — {spec.description} | {BACKEND_LABELS[backend]}")

    record: Dict[str, Any] = {
        "method": spec.method,
        "merged": spec.merged,
        "backend": backend,
        "ok": False,
        "error": None,
        "latency": None,
        "params_total": 0,
        "params_trainable": 0,
        "adapter_sig": {},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    model = None
    try:
        # ── 准备模型 ──
        if spec.method == "baseline":
            model = prepare_baseline()
        elif spec.method == "lora":
            model = prepare_lora(merged=spec.merged)
        elif spec.method == "molora":
            model = prepare_molora(merged=spec.merged)
        else:
            raise ValueError(f"Unknown method: {spec.method}")

        if model is None:
            raise RuntimeError("模型准备返回 None")

        # 参数统计
        total, trainable = count_params(model.model)
        record["params_total"] = total
        record["params_trainable"] = trainable
        record["adapter_sig"] = detect_adapter_signature(model.model)
        print(f"  [Params] total={total:,} trainable={trainable:,}")

        # ── 选择后端测量 ──
        if backend == "pytorch":
            latency = measure_pytorch_eager(model)
        elif backend == "onnx":
            latency = measure_onnx_runtime(model)
        elif backend == "tensorrt":
            latency = measure_tensorrt(model)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        if latency is not None:
            record["latency"] = latency
            record["ok"] = True
            print(
                f"  [Latency] mean={latency['mean_ms']:.2f}±{latency['std_ms']:.2f} ms  "
                f"median={latency['median_ms']:.2f} ms  p95={latency['p95_ms']:.2f} ms  "
                f"FPS={latency['fps']:.1f}"
            )
        else:
            record["error"] = "Latency measurement returned None"
            print("  [Latency] FAILED")

    except Exception as exc:
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  [ERROR] {record['error']}")
        traceback.print_exc()

    finally:
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        # 释放模型引用
        del model

    return record


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 结果汇总与输出
# ═══════════════════════════════════════════════════════════════════════════════


def print_header():
    """打印实验环境信息。"""
    print("\n" + "=" * 88)
    print("  YOLO-Master Inference Latency Benchmark")
    print("=" * 88)
    print(f"  Model      : {MODEL_PATH}")
    print(f"  Device     : {DEVICE}")
    print(f"  Image size : {IMGSZ}")
    print(f"  Batch      : {BATCH}")
    print(f"  Warmup     : {WARMUP_RUNS}")
    print(f"  Timed runs : {TIMED_RUNS} (min_time={MIN_TIME_SEC}s)")
    print(f"  Output     : {RESULTS_JSON}")
    print("=" * 88)
    print(f"\n[Boot] ultralytics: {ultralytics.__file__}")
    print(f"[Boot] version     : {ultralytics.__version__}")
    print(f"[Boot] torch       : {torch.__version__}  device={DEVICE}")
    print()


def print_summary_table(all_records: List[Dict[str, Any]]):
    """打印控制台汇总表。"""
    header = (
        f"{'Method':<12} {'Merged':<8} {'Backend':<16} {'OK':<3} "
        f"{'Mean(ms)':>10} {'Std(ms)':>9} {'Median(ms)':>11} {'P95(ms)':>9} {'FPS':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in all_records:
        lat = r.get("latency")
        if lat:
            mean_str = f"{lat['mean_ms']:.2f}"
            std_str = f"{lat['std_ms']:.2f}"
            med_str = f"{lat['median_ms']:.2f}"
            p95_str = f"{lat['p95_ms']:.2f}"
            fps_str = f"{lat['fps']:.1f}"
        else:
            mean_str = std_str = med_str = p95_str = fps_str = "N/A"

        merged_label = "merged" if r["merged"] else "unmerged" if r["method"] != "baseline" else "—"
        print(
            f"{r['method']:<12} {merged_label:<8} {BACKEND_LABELS.get(r['backend'], r['backend']):<16} "
            f"{'Y' if r['ok'] else 'N':<3} "
            f"{mean_str:>10} {std_str:>9} {med_str:>11} {p95_str:>9} {fps_str:>8}"
        )

    print("=" * len(header))


def save_results(all_records: List[Dict[str, Any]]) -> bool:
    """将结果实时写入 JSON 文件。"""
    try:
        RESULTS_JSON.write_text(
            json.dumps(all_records, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return True
    except Exception as e:
        print(f"[WARN] 结果写入失败: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 主流程
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """主入口：遍历所有变体与后端，实时持久化结果。"""
    print_header()
    HERE.mkdir(exist_ok=True, parents=True)

    all_records: List[Dict[str, Any]] = []
    total_jobs = sum(len(spec.backends) for spec in BENCHMARK_MATRIX)
    job_idx = 0

    for spec in BENCHMARK_MATRIX:
        for backend in spec.backends:
            job_idx += 1
            print(
                f"\n[Progress] {job_idx}/{total_jobs} — {spec.method} {'merged' if spec.merged else 'unmerged'} {backend}"
            )
            rec = run_single_benchmark(spec, backend)
            all_records.append(rec)

            # 实时落盘：单个失败不丢之前结果
            save_results(all_records)

    # ── 最终汇总 ──
    print_summary_table(all_records)

    print(f"\n{'=' * 88}")
    print(f"全部 {total_jobs} 个基准测试组合运行完毕。")
    print(f"详细结果 JSON: {RESULTS_JSON}")
    print("=" * 88)


if __name__ == "__main__":
    main()
