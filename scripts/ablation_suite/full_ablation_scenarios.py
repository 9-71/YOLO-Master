"""
================================================================================
YOLO-Master Full Ablation Scenarios — Scene Dataset Extension
================================================================================
在 VisDrone、SKU110K、Cityscapes→Foggy 三个场景数据集上运行核心 PEFT 变体，
支持 domain 标签用于持续学习，报告 mAP、mAP_s/m/l、参数量、训练时间。

核心 PEFT 变体:
    0. full          — 全量微调 Baseline
    1. lora_r16      — 标准 LoRA (r=16, alpha=32, RS-LoRA)
    2. molora_4e2k   — MoLoRA linear router
    3. molora_4e2k_spatial — MoLoRA spatial router
    4. molora_4e2k_hybrid   — MoLoRA hybrid router
    5. molora_aware  — MoLoRA + MoE-aware (per-expert rank)
    6. molora_calib  — MoLoRA + Router Calibration (ΔW_r)

场景数据集:
    • visdrone      — 航拍小目标检测 (10 classes, domain=None)
    • sku110k       — 极端密集零售检测 (1 class, ~100 instances/img, domain=None)
    • cityscapes    — 城市街景 (30 classes, domain="day")
    • foggy_cityscapes — 雾域偏移 (30 classes, domain="fog")

用法:
    python scripts/full_ablation_scenarios.py [--seed SEED] [--epochs EPOCHS] [--batch BATCH]

输出:
    - 结构化 JSON 结果文件 (实时写入): scripts/full_ablation_scenarios_results.json
    - 控制台逐数据集 / 逐变体进度 + 最终汇总表
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 环境初始化 —— 必须在 import ultralytics 之前完成
# ═══════════════════════════════════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parents[2]

# 插入项目根目录到 sys.path 以导入 full_ablation_spec
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ablation_suite"))

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")

import torch
import torch.nn as nn

# 关闭 ultralytics 的 wandb 上报
from ultralytics.utils import SETTINGS
SETTINGS["wandb"] = False

from ultralytics import YOLO

# MoLoRA 基础设施
from ultralytics.nn.peft.molora import (
    MoLoRAConfig,
    MoLoRAMoEAwareConfig,
    MoLoRAConfigBuilder,
    get_peft_molora_model,
    build_moe_aware_layer,
    mark_only_molora_as_trainable,
)
from ultralytics.nn.peft.molora.model import _parent_child_name, _get_submodule

# 标准 LoRA / DoRA / IA3 等 PEFT 注入
from ultralytics.utils.lora import apply_lora
from ultralytics.utils.lora.config import LoRAConfig

# 确认加载的是当前仓库的 ultralytics (避免 pip 安装的版本干扰)
import ultralytics
assert str(REPO_ROOT) in ultralytics.__file__, (
    f"加载的不是当前仓库的 ultralytics！got {ultralytics.__file__}"
)

# 从统一规范导入数据结构
from full_ablation_spec import (
    DatasetConfig,
    VariantConfig,
    ExperimentResult,
    DATASET_REGISTRY,
    FULL_ABLATION_VARIANTS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 全局配置
# ═══════════════════════════════════════════════════════════════════════════════

HERE = Path(__file__).parent
MODEL_PATH = str(REPO_ROOT / "YOLO-Master-EsMoE-N.pt")
PROJECT_DIR = HERE / "runs_full_ablation_scenarios"
RESULTS_JSON = HERE / "full_ablation_scenarios_results.json"

# 设备选择: MPS 优先 → CUDA → CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

# 强制行缓冲，便于实时观察后台输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# 从 FULL_ABLATION_VARIANTS 中筛选核心场景变体
CORE_VARIANT_NAMES = [
    "full",
    "lora_r16",
    "molora_4e2k",
    "molora_4e2k_spatial",
    "molora_4e2k_hybrid",
    "molora_aware",
    "molora_calib",
]

VARIANTS_BY_NAME: Dict[str, VariantConfig] = {v.name: v for v in FULL_ABLATION_VARIANTS}
CORE_VARIANTS: List[VariantConfig] = []
for name in CORE_VARIANT_NAMES:
    if name in VARIANTS_BY_NAME:
        CORE_VARIANTS.append(VARIANTS_BY_NAME[name])
    else:
        warnings.warn(f"核心变体 '{name}' 在 FULL_ABLATION_VARIANTS 中未找到，已跳过。")

# 场景数据集列表
SCENARIO_DATASETS: List[str] = [
    "visdrone",
    "sku110k",
    "cityscapes",
    "foggy_cityscapes",
]

# 持续学习: Cityscapes → FoggyCityscapes 的域序列
CONTINUOUS_LEARNING_SEQUENCE: List[str] = [
    "cityscapes",
    "foggy_cityscapes",
]


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
    """
    names = [n for n, _ in m.named_parameters()]
    return {
        "has_lora_A": any("lora_A" in n for n in names),
        "has_lora_B": any("lora_B" in n for n in names),
        "has_dora_magnitude": any("magnitude_vector" in n for n in names),
        "has_loha": any("hada" in n.lower() for n in names),
        "has_ia3": any("ia3" in n.lower() for n in names),
        "has_adalora": any("adalora" in n.lower() or "s_vector" in n.lower() for n in names),
        "has_molora": any("molora" in n.lower() or "router" in n.lower() for n in names),
        "has_router_calibration": any("router_calibration" in n.lower() for n in names),
        "n_lora_params": sum(1 for n in names if "lora_" in n.lower()),
        "n_router_params": sum(1 for n in names if "router" in n.lower()),
    }


def collect_molora_diagnostics(model: nn.Module) -> Dict[str, Any]:
    """
    收集 MoLoRA 特有的诊断信息。
    """
    diag: Dict[str, Any] = {
        "molora_enabled": getattr(model, "molora_enabled", False),
        "molora_config": str(getattr(model, "molora_config", "N/A")),
        "per_expert_ranks": None,
        "router_calib_present": False,
        "router_types": {},
        "expert_counts": {},
        "top_k_values": {},
    }

    for name, module in model.named_modules():
        if hasattr(module, "_expert_ranks") and module._expert_ranks is not None:
            if diag["per_expert_ranks"] is None:
                diag["per_expert_ranks"] = module._expert_ranks

        if hasattr(module, "router_calibration") and module.router_calibration is not None:
            diag["router_calib_present"] = True

        if hasattr(module, "router"):
            router_type = type(module.router).__name__
            diag["router_types"][router_type] = diag["router_types"].get(router_type, 0) + 1

        if hasattr(module, "num_experts"):
            key = f"{name}:E{module.num_experts}"
            diag["expert_counts"][key] = diag["expert_counts"].get(key, 0) + 1
        if hasattr(module, "top_k"):
            key = f"{name}:K{module.top_k}"
            diag["top_k_values"][key] = diag["top_k_values"].get(key, 0) + 1

    return diag


def extract_final_metrics(results) -> Dict[str, float]:
    """从 ultralytics Results 对象中提取最终评估指标。"""
    final_metrics: Dict[str, float] = {}
    if results is None:
        return final_metrics

    if hasattr(results, "results_dict") and results.results_dict:
        final_metrics = {
            k: float(v) for k, v in results.results_dict.items()
            if isinstance(v, (int, float))
        }
    elif hasattr(results, "metrics") and results.metrics:
        final_metrics = {
            k: float(v) for k, v in results.metrics.items()
            if isinstance(v, (int, float))
        }

    return final_metrics


def extract_map_per_scale(results) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    尝试从 results 中提取 mAP_small, mAP_medium, mAP_large。
    ultralytics 不同版本可能存放位置不同，这里做兼容性处理。
    """
    m_s = m_m = m_l = None

    # 方式 1: results.results_dict 中的 keys
    if hasattr(results, "results_dict") and results.results_dict:
        rd = results.results_dict
        for k, v in rd.items():
            if isinstance(v, (int, float)):
                if "small" in k.lower() or "map_s" in k.lower() or "mAP50-95_s" in k:
                    m_s = float(v)
                elif "medium" in k.lower() or "map_m" in k.lower() or "mAP50-95_m" in k:
                    m_m = float(v)
                elif "large" in k.lower() or "map_l" in k.lower() or "mAP50-95_l" in k:
                    m_l = float(v)

    # 方式 2: results.metrics 中可能包含 mAP_s/m/l
    if hasattr(results, "metrics") and results.metrics:
        metrics = results.metrics
        if m_s is None:
            m_s = float(metrics.get("mAP_s", metrics.get("map_s", metrics.get("metrics/mAP50-95(B)_small", float("nan")))))
            if m_s != m_s:  # nan check
                m_s = None
        if m_m is None:
            m_m = float(metrics.get("mAP_m", metrics.get("map_m", metrics.get("metrics/mAP50-95(B)_medium", float("nan")))))
            if m_m != m_m:
                m_m = None
        if m_l is None:
            m_l = float(metrics.get("mAP_l", metrics.get("map_l", metrics.get("metrics/mAP50-95(B)_large", float("nan")))))
            if m_l != m_l:
                m_l = None

    return m_s, m_m, m_l


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PEFT 应用器
# ═══════════════════════════════════════════════════════════════════════════════

def apply_peft_via_train_args(model: YOLO, kwargs: Dict[str, Any]) -> YOLO:
    """
    对于标准 PEFT 方法 (LoRA, DoRA, IA3 等)，
    显式使用 apply_lora 注入 adapter。
    """
    lora_type = kwargs.get("lora_type", "lora")
    lora_r = kwargs.get("lora_r", 8)
    lora_alpha = kwargs.get("lora_alpha", 16)
    lora_dropout = kwargs.get("lora_dropout", 0.05)
    lora_backend = kwargs.get("lora_backend", "peft")
    use_dora = kwargs.get("lora_use_dora", False)

    cfg = LoRAConfig(
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        backend=lora_backend,
        variant=lora_type,
        use_dora=use_dora,
        peft_type=lora_type,
    )
    model = apply_lora(model.model, cfg)
    if not isinstance(model, YOLO):
        yolo_model = YOLO(MODEL_PATH)
        yolo_model.model = model
        model = yolo_model
    return model


def apply_molora_standard(model: YOLO, config: Dict[str, Any]) -> YOLO:
    """应用标准 MoLoRA (Mixture-of-LoRA)。"""
    cfg = MoLoRAConfig(
        r=config.get("r", 8),
        alpha=config.get("alpha", 16),
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 2),
        router_type=config.get("router_type", "linear"),
        dropout=config.get("dropout", 0.05),
        use_rslora=config.get("use_rslora", True),
        balance_loss_coef=config.get("balance_loss_coef", 0.01),
        z_loss_coef=config.get("z_loss_coef", 0.001),
    )
    get_peft_molora_model(model.model, cfg)
    return model


def apply_molora_moe_aware(model: YOLO, config: Dict[str, Any]) -> YOLO:
    """应用 MoLoRA + MoE-aware (per-expert rank, frequency-based)。"""
    cfg = MoLoRAMoEAwareConfig(
        r=config.get("r", 8),
        alpha=config.get("alpha", 16),
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 2),
        router_type=config.get("router_type", "linear"),
        dropout=config.get("dropout", 0.05),
        use_rslora=config.get("use_rslora", True),
        balance_loss_coef=config.get("balance_loss_coef", 0.01),
        z_loss_coef=config.get("z_loss_coef", 0.001),
        per_expert_rank=True,
        rank_allocator_mode="frequency",
        rank_budget_total=config.get("rank_budget_total", 32),
        rank_min=config.get("rank_min", 2),
    )

    target_modules = MoLoRAConfigBuilder.auto_detect_targets(
        model.model, r=cfg.r, include_moe=True, only_backbone=False
    )

    modules_dict = dict(model.model.named_modules())
    for name in target_modules:
        if name not in modules_dict:
            continue
        base_layer = modules_dict[name]
        if not isinstance(base_layer, (nn.Conv2d, nn.Linear)):
            continue

        parent_name, child_name = _parent_child_name(name)
        parent = _get_submodule(model.model, parent_name) if parent_name else model.model
        if parent is None or not hasattr(parent, child_name):
            continue

        num_experts = cfg.num_experts
        if num_experts == 4:
            usage_history = torch.tensor([0.5, 0.2, 0.2, 0.1])
        else:
            x = torch.linspace(0, 1, num_experts)
            usage_history = torch.exp(-3 * x)
            usage_history = usage_history / usage_history.sum()

        layer = build_moe_aware_layer(base_layer, cfg, usage_history=usage_history)
        setattr(parent, child_name, layer)

    model.model.molora_config = cfg
    model.model.molora_enabled = True
    mark_only_molora_as_trainable(model.model)
    return model


def apply_molora_router_calibration(model: YOLO, config: Dict[str, Any]) -> YOLO:
    """应用 MoLoRA + Router Calibration (ΔW_r)。"""
    cfg = MoLoRAMoEAwareConfig(
        r=config.get("r", 8),
        alpha=config.get("alpha", 16),
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 2),
        router_type=config.get("router_type", "linear"),
        dropout=config.get("dropout", 0.05),
        use_rslora=config.get("use_rslora", True),
        balance_loss_coef=config.get("balance_loss_coef", 0.01),
        z_loss_coef=config.get("z_loss_coef", 0.001),
        router_calibration=True,
        router_calib_rank=config.get("router_calib_rank", 4),
    )

    target_modules = MoLoRAConfigBuilder.auto_detect_targets(
        model.model, r=cfg.r, include_moe=True, only_backbone=False
    )

    modules_dict = dict(model.model.named_modules())
    for name in target_modules:
        if name not in modules_dict:
            continue
        base_layer = modules_dict[name]
        if not isinstance(base_layer, (nn.Conv2d, nn.Linear)):
            continue

        parent_name, child_name = _parent_child_name(name)
        parent = _get_submodule(model.model, parent_name) if parent_name else model.model
        if parent is None or not hasattr(parent, child_name):
            continue

        layer = build_moe_aware_layer(base_layer, cfg, usage_history=None)
        setattr(parent, child_name, layer)

    model.model.molora_config = cfg
    model.model.molora_enabled = True
    mark_only_molora_as_trainable(model.model)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 单变体执行逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def run_variant(
    dataset_cfg: DatasetConfig,
    variant_cfg: VariantConfig,
    seed: int = 42,
    previous_model_state: Optional[Dict[str, Any]] = None,
    apply_peft: bool = True,
) -> Tuple[ExperimentResult, Optional[Dict[str, Any]]]:
    """
    运行单个数据集 × 变体组合。
    previous_model_state 用于持续学习：从上一个域的模型状态继续训练。
    apply_peft 控制是否在本轮应用 PEFT（持续学习序列中仅在首个域应用）。
    返回: (ExperimentResult, 模型 state_dict 或 None)
    """
    print(f"\n{'='*78}")
    print(f"=== Dataset: {dataset_cfg.name.upper()} | Variant: {variant_cfg.name.upper()} {'='*35}")
    print(f"{'='*78}")
    print(f"Description: {dataset_cfg.description}")
    print(f"Domain     : {dataset_cfg.domain}")
    print(f"PEFT type  : {variant_cfg.peft_type}")
    print(f"Epochs     : {variant_cfg.epochs} | Batch: {variant_cfg.batch} | Imgsz: {variant_cfg.imgsz}")

    t0 = time.time()
    result = ExperimentResult(
        dataset=dataset_cfg.name,
        variant=variant_cfg.name,
        seed=seed,
        ok=False,
    )

    try:
        # ── 4.1 加载模型 ──
        if previous_model_state is not None and variant_cfg.peft_type != "full":
            # 持续学习: 从上一个域的 PEFT 状态恢复
            model = YOLO(MODEL_PATH)
            missing, unexpected = model.model.load_state_dict(previous_model_state, strict=False)
            if missing:
                print(f"[CL] 恢复模型状态: missing keys={len(missing)}")
            if unexpected:
                print(f"[CL] 恢复模型状态: unexpected keys={len(unexpected)}")
        else:
            model = YOLO(MODEL_PATH)

        base_total, base_train = count_params(model.model)
        print(f"[Pre-train]  total={base_total:,}  trainable={base_train:,}  ({base_train/base_total*100:.2f}%)")

        # ── 4.2 应用 PEFT（仅在 apply_peft=True 时执行） ──
        if apply_peft:
            if variant_cfg.peft_type == "peft":
                model = apply_peft_via_train_args(model, variant_cfg.train_kwargs)
            elif variant_cfg.peft_type == "molora":
                model = apply_molora_standard(model, variant_cfg.molora_config)
            elif variant_cfg.peft_type == "molora_aware":
                model = apply_molora_moe_aware(model, variant_cfg.molora_config)
            elif variant_cfg.peft_type == "molora_calib":
                model = apply_molora_router_calibration(model, variant_cfg.molora_config)
            elif variant_cfg.peft_type == "full":
                # 全量微调: 不做 PEFT 注入，所有参数可训练
                pass
            else:
                raise ValueError(f"Unknown peft_type: {variant_cfg.peft_type}")
        else:
            print("[CL] 跳过 PEFT 应用，直接继承上一域模型状态。")

        # ── 4.3 训练前统计 ──
        post_total, post_train = count_params(model.model)
        sig = detect_adapter_signature(model.model)
        molora_diag = collect_molora_diagnostics(model.model)

        print(f"[Post-wrap]  total={post_total:,}  trainable={post_train:,}  ({post_train/post_total*100:.2f}%)")
        print(f"[Adapter]    {sig}")
        if molora_diag.get("molora_enabled"):
            print(f"[MoLoRA]     enabled={molora_diag['molora_enabled']}, "
                  f"router_calib={molora_diag['router_calib_present']}, "
                  f"ranks={molora_diag['per_expert_ranks']}")

        # ── 4.4 构建训练参数 ──
        train_kwargs = {
            "data": dataset_cfg.yaml,
            "epochs": variant_cfg.epochs,
            "batch": variant_cfg.batch,
            "imgsz": variant_cfg.imgsz,
            "device": DEVICE,
            "project": str(PROJECT_DIR),
            "name": f"{dataset_cfg.name}_{variant_cfg.name}",
            "exist_ok": True,
            "verbose": False,
            "workers": 2,
            "patience": 0,
            "plots": False,
            "save": False,
            "seed": seed,
        }
        if variant_cfg.peft_type == "peft" and variant_cfg.train_kwargs:
            train_kwargs.update(variant_cfg.train_kwargs)

        # 若数据集配置了 domain，传入作为自定义参数（可被后续 MoLoRA 持续学习模块消费）
        if dataset_cfg.domain:
            train_kwargs["domain"] = dataset_cfg.domain

        # ── 4.5 训练 ──
        results = model.train(**train_kwargs)

        # ── 4.6 提取指标 ──
        final_metrics = extract_final_metrics(results)
        m_s, m_m, m_l = extract_map_per_scale(results)

        result.ok = True
        result.final_metrics = final_metrics
        result.map_small = m_s
        result.map_medium = m_m
        result.map_large = m_l
        result.params_total = post_total
        result.params_trainable = post_train
        result.trainable_pct = round(post_train / post_total * 100, 4) if post_total else 0.0
        result.adapter_sig = sig
        result.molora_diagnostics = molora_diag
        result.elapsed_sec = round(time.time() - t0, 1)

        print(f"[Final metrics] {json.dumps(final_metrics, indent=2)}")
        print(f"[mAP per scale] s={m_s}, m={m_m}, l={m_l}")

    except Exception as exc:
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_sec = round(time.time() - t0, 1)
        print(f"[ERROR] {result.error}")
        traceback.print_exc()

    # 兜底统计（即使训练失败也尝试记录模型参数）
    try:
        if "model" in dir() and model is not None:
            post_total, post_train = count_params(model.model)
            result.params_total = post_total
            result.params_trainable = post_train
            result.trainable_pct = round(post_train / post_total * 100, 4) if post_total else 0.0
            result.adapter_sig = detect_adapter_signature(model.model)
            result.molora_diagnostics = collect_molora_diagnostics(model.model)
    except Exception:
        pass

    print(f"[Done] elapsed={result.elapsed_sec}s  ok={result.ok}")

    # 返回模型状态，供持续学习下一域继承
    model_state = None
    if result.ok and "model" in dir() and model is not None:
        try:
            model_state = {k: v.cpu().clone() for k, v in model.model.state_dict().items()}
        except Exception:
            pass
    return result, model_state


def run_continuous_learning(
    datasets: List[str],
    variant_cfg: VariantConfig,
    seed: int = 42,
) -> List[ExperimentResult]:
    """
    执行持续学习序列：在多个域上依次训练，后一个域继承前一个域的模型状态。
    仅对非 full 变体有效（full 微调会改变所有权重，不适合简单 state_dict 继承）。
    首个域应用 PEFT，后续域直接加载上一域的模型状态。
    """
    results: List[ExperimentResult] = []
    previous_state: Optional[Dict[str, Any]] = None

    for idx, ds_name in enumerate(datasets):
        ds_cfg = DATASET_REGISTRY.get(ds_name)
        if ds_cfg is None:
            print(f"[WARN] 数据集 '{ds_name}' 在注册表中未找到，已跳过。")
            continue

        apply_peft = (idx == 0)  # 仅在第一个域应用 PEFT
        result, state = run_variant(
            ds_cfg, variant_cfg, seed=seed,
            previous_model_state=previous_state, apply_peft=apply_peft
        )
        results.append(result)

        if result.ok and variant_cfg.peft_type != "full":
            previous_state = state
            if previous_state is not None:
                print(f"[CL] 已保存 {ds_name} 的模型状态用于下一域继承 ({len(previous_state)} keys)")
        else:
            previous_state = None

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def print_header(args: argparse.Namespace):
    """打印实验环境信息。"""
    print("\n" + "=" * 78)
    print("  YOLO-Master Full Ablation Scenarios — Scene Dataset Extension")
    print("=" * 78)
    print(f"  Model      : {MODEL_PATH}")
    print(f"  Device     : {DEVICE}")
    print(f"  Seed       : {args.seed}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch      : {args.batch}")
    print(f"  Variants   : {len(CORE_VARIANTS)}")
    print(f"  Datasets   : {', '.join(SCENARIO_DATASETS)}")
    print(f"  CL sequence: {', '.join(CONTINUOUS_LEARNING_SEQUENCE)}")
    print(f"  Output dir : {PROJECT_DIR}")
    print(f"  Results    : {RESULTS_JSON}")
    print("=" * 78)
    print(f"\n[Boot] ultralytics: {ultralytics.__file__}")
    print(f"[Boot] version     : {ultralytics.__version__}")
    print(f"[Boot] torch device: {DEVICE}")
    print()


def print_summary_table(all_results: List[ExperimentResult]):
    """打印控制台汇总表。"""
    header = (
        f"{'Dataset':<14} {'Variant':<14} {'OK':<3} {'Total':>12} {'Trainable':>12} {'%':>7} "
        f"{'mAP50-95':>10} {'mAP_s':>8} {'mAP_m':>8} {'mAP_l':>8} {'Time(s)':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in all_results:
        m = r.final_metrics.get("metrics/mAP50-95(B)", float("nan"))
        m_str = f"{m:.4f}" if isinstance(m, float) else "N/A"
        s_str = f"{r.map_small:.4f}" if r.map_small is not None else "N/A"
        m_str2 = f"{r.map_medium:.4f}" if r.map_medium is not None else "N/A"
        l_str = f"{r.map_large:.4f}" if r.map_large is not None else "N/A"

        print(
            f"{r.dataset:<14} "
            f"{r.variant:<14} "
            f"{'Y' if r.ok else 'N':<3} "
            f"{r.params_total:>12,} "
            f"{r.params_trainable:>12,} "
            f"{r.trainable_pct:>7.3f} "
            f"{m_str:>10} "
            f"{s_str:>8} "
            f"{m_str2:>8} "
            f"{l_str:>8} "
            f"{r.elapsed_sec:>8.1f}"
        )

    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(
        description="YOLO-Master Full Ablation Scenarios — Scene Dataset Extension"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all variants")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size for all variants")
    parser.add_argument("--cl-only", action="store_true", help="仅运行持续学习序列 (Cityscapes→Foggy)")
    parser.add_argument("--dataset", type=str, default=None, help="仅运行指定数据集 (如 visdrone)")
    parser.add_argument("--variant", type=str, default=None, help="仅运行指定变体 (如 lora_r16)")
    args = parser.parse_args()

    print_header(args)
    PROJECT_DIR.mkdir(exist_ok=True, parents=True)

    all_results: List[ExperimentResult] = []

    # 确定要运行的数据集
    if args.cl_only:
        datasets_to_run = CONTINUOUS_LEARNING_SEQUENCE
    elif args.dataset:
        datasets_to_run = [args.dataset]
    else:
        datasets_to_run = SCENARIO_DATASETS

    # 确定要运行的变体
    if args.variant:
        variants_to_run = [v for v in CORE_VARIANTS if v.name == args.variant]
        if not variants_to_run:
            print(f"[ERROR] 变体 '{args.variant}' 不在核心变体列表中，退出。")
            sys.exit(1)
    else:
        variants_to_run = CORE_VARIANTS

    # 若用户指定了 --epochs 或 --batch，覆盖所有变体配置
    if args.epochs is not None or args.batch is not None:
        for v in variants_to_run:
            if args.epochs is not None:
                v.epochs = args.epochs
            if args.batch is not None:
                v.batch = args.batch

    # 主实验循环
    total_experiments = len(datasets_to_run) * len(variants_to_run)
    exp_idx = 0

    for ds_name in datasets_to_run:
        ds_cfg = DATASET_REGISTRY.get(ds_name)
        if ds_cfg is None:
            print(f"[WARN] 数据集 '{ds_name}' 在 DATASET_REGISTRY 中未找到，已跳过。")
            continue

        for var_cfg in variants_to_run:
            exp_idx += 1
            print(f"\n[Progress] {exp_idx}/{total_experiments} — 开始运行 {ds_name} × {var_cfg.name}")

            result, _ = run_variant(ds_cfg, var_cfg, seed=args.seed)
            all_results.append(result)

            # 实时写入 JSON
            try:
                RESULTS_JSON.write_text(
                    json.dumps([r.to_dict() for r in all_results], indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"[WARN] 结果 JSON 写入失败: {e}")

    # 若未使用 --cl-only 且 Cityscapes 和 Foggy 都在运行列表中，额外运行持续学习实验
    if not args.cl_only and not args.dataset and not args.variant:
        print("\n" + "=" * 78)
        print("  额外运行持续学习序列: Cityscapes → FoggyCityscapes")
        print("=" * 78)
        for var_cfg in variants_to_run:
            if var_cfg.peft_type == "full":
                # full 微调不适合简单 state_dict 继承，跳过或单独运行
                print(f"[CL-Skip] {var_cfg.name} 为 full 微调，不纳入持续学习序列。")
                continue
            cl_results = run_continuous_learning(
                CONTINUOUS_LEARNING_SEQUENCE, var_cfg, seed=args.seed
            )
            all_results.extend(cl_results)
            try:
                RESULTS_JSON.write_text(
                    json.dumps([r.to_dict() for r in all_results], indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"[WARN] 结果 JSON 写入失败: {e}")

    # ── 最终汇总 ──
    print_summary_table(all_results)

    print(f"\n{'='*78}")
    print(f"全部 {len(all_results)} 条实验记录运行完毕。")
    print(f"详细结果 JSON: {RESULTS_JSON}")
    print(f"训练日志目录: {PROJECT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
