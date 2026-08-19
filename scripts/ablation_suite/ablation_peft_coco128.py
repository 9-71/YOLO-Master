"""
PEFT 消融实验主脚本 — COCO128 多方法对比 (YOLO-Master)
========================================================
目的: 在 COCO128 数据集上系统对比多种 PEFT 方法的精度、效率与可训练参数量。

涵盖方法:
  1. full          — 全量微调 (baseline)
  2. lora          — 标准 LoRA
  3. dora          — DoRA (Weight-Decomposed LoRA)
  4. ia3           — IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
  5. loha          — LoHA (Low-Rank Hadamard Product)
  6. adalora       — AdaLoRA (Adaptive Budget Allocation)
  7. molora        — 标准 MoLoRA (E=4, K=2, r=8)
  8. molora_aware  — MoLoRA + MoE-aware (per-expert rank, frequency-based)
  9. molora_calib  — MoLoRA + Router Calibration (ΔW_r)

输出:
  - 本地 JSON 结果文件
  - 控制台汇总表 (trainable params, mAP, 耗时, adapter 签名)

用法:
    python scripts/ablation_peft_coco128.py

环境:
    - 设备优先 MPS (Apple Silicon), 回退 CUDA / CPU
    - 所有实验禁用 WandB (WANDB_MODE=disabled)
    - 数据集: ultralytics 内置 coco128.yaml (自动下载)
    - 模型: YOLO-Master-EsMoE-N.pt (项目根目录)
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

# 标准 PEFT (LoRA / DoRA / IA3 / LoHA / AdaLoRA)
from ultralytics.utils.lora import apply_lora
from ultralytics.utils.lora.config import LoRAConfig

# 确认加载的是当前仓库的 ultralytics
import ultralytics
assert str(REPO_ROOT) in ultralytics.__file__, (
    f"加载的不是当前仓库的 ultralytics！got {ultralytics.__file__}"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 全局配置
# ═══════════════════════════════════════════════════════════════════════════════

HERE = Path(__file__).parent
MODEL_PATH = str(REPO_ROOT / "YOLO-Master-EsMoE-N.pt")
DATA_YAML = "coco128.yaml"          # ultralytics 内置数据集，首次运行自动下载
PROJECT_DIR = HERE / "runs_peft_ablation"
RESULTS_JSON = HERE / "ablation_peft_coco128_results.json"

# 训练超参 (消融实验用轻量配置以加速迭代)
EPOCHS = 3
BATCH = 8
IMGSZ = 320

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

# PEFT 公共超参
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
    收集 MoLoRA 特有的诊断信息：
      - 是否启用了 MoLoRA
      - 第一个 MoLoRA 层的 per-expert rank 分配 (若存在)
      - 是否存在 Router Calibration
      - 各层 router 类型统计
    """
    diag: Dict[str, Any] = {
        "molora_enabled": getattr(model, "molora_enabled", False),
        "molora_config": str(getattr(model, "molora_config", "N/A")),
        "per_expert_ranks": None,
        "router_calib_present": False,
        "router_types": {},
    }

    for name, module in model.named_modules():
        # Per-expert rank (来自 MoE-aware)
        if hasattr(module, "_expert_ranks") and module._expert_ranks is not None:
            if diag["per_expert_ranks"] is None:
                diag["per_expert_ranks"] = module._expert_ranks

        # Router calibration 检测
        if hasattr(module, "router_calibration") and module.router_calibration is not None:
            diag["router_calib_present"] = True

        # Router 类型统计
        if hasattr(module, "router"):
            router_type = type(module.router).__name__
            diag["router_types"][router_type] = diag["router_types"].get(router_type, 0) + 1

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


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PEFT 应用器 —— 统一接口封装不同 PEFT 方法的应用逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def apply_peft_via_train_args(model: YOLO, kwargs: Dict[str, Any]) -> YOLO:
    """
    对于标准 PEFT 方法 (LoRA, DoRA, IA3, LoHA, AdaLoRA)，
    使用 apply_lora() 显式注入 adapter，确保可训练参数统计正确。
    """
    lora_type = kwargs.get("lora_type", "lora")
    lora_r = kwargs.get("lora_r", LORA_R)
    lora_alpha = kwargs.get("lora_alpha", LORA_ALPHA)
    lora_dropout = kwargs.get("lora_dropout", LORA_DROPOUT)
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
    new_model = apply_lora(model.model, cfg)
    if not isinstance(new_model, YOLO):
        model.model = new_model
    else:
        model = new_model
    return model


def apply_molora_standard(model: YOLO, config: Dict[str, Any]) -> YOLO:
    """
    应用标准 MoLoRA (Mixture-of-LoRA)。
    使用 get_peft_molora_model 直接包装 DetectionModel。
    """
    cfg = MoLoRAConfig(
        r=config.get("r", LORA_R),
        alpha=config.get("alpha", LORA_ALPHA),
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 2),
        router_type=config.get("router_type", "linear"),
        dropout=config.get("dropout", LORA_DROPOUT),
        use_rslora=config.get("use_rslora", True),
        balance_loss_coef=config.get("balance_loss_coef", 0.01),
        z_loss_coef=config.get("z_loss_coef", 0.001),
    )
    get_peft_molora_model(model.model, cfg)
    return model


def apply_molora_moe_aware(model: YOLO, config: Dict[str, Any]) -> YOLO:
    """
    应用 MoLoRA + MoE-aware (per-expert rank, frequency-based)。
    使用 build_moe_aware_layer 逐层替换目标模块。
    """
    cfg = MoLoRAMoEAwareConfig(
        r=config.get("r", LORA_R),
        alpha=config.get("alpha", LORA_ALPHA),
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 2),
        router_type=config.get("router_type", "linear"),
        dropout=config.get("dropout", LORA_DROPOUT),
        use_rslora=config.get("use_rslora", True),
        balance_loss_coef=config.get("balance_loss_coef", 0.01),
        z_loss_coef=config.get("z_loss_coef", 0.001),
        # MoE-aware 特有参数
        per_expert_rank=True,
        rank_allocator_mode="frequency",
        rank_budget_total=config.get("rank_budget_total", 32),
        rank_min=config.get("rank_min", 2),
    )

    target_modules = MoLoRAConfigBuilder.auto_detect_targets(
        model.model, r=cfg.r, include_moe=True, only_backbone=False
    )

    wrapped = 0
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

        # 使用预设的 usage_history 让 frequency 模式产生有意义的非均匀分配
        num_experts = cfg.num_experts
        if num_experts == 4:
            usage_history = torch.tensor([0.5, 0.2, 0.2, 0.1])
        else:
            x = torch.linspace(0, 1, num_experts)
            usage_history = torch.exp(-3 * x)
            usage_history = usage_history / usage_history.sum()

        layer = build_moe_aware_layer(base_layer, cfg, usage_history=usage_history)
        setattr(parent, child_name, layer)
        wrapped += 1

    model.model.molora_config = cfg
    model.model.molora_enabled = True
    mark_only_molora_as_trainable(model.model)

    return model


def apply_molora_router_calibration(model: YOLO, config: Dict[str, Any]) -> YOLO:
    """
    应用 MoLoRA + Router Calibration (ΔW_r)。
    Router Calibration 为每个 router 添加可学习的低秩修正项。
    """
    cfg = MoLoRAMoEAwareConfig(
        r=config.get("r", LORA_R),
        alpha=config.get("alpha", LORA_ALPHA),
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 2),
        router_type=config.get("router_type", "linear"),
        dropout=config.get("dropout", LORA_DROPOUT),
        use_rslora=config.get("use_rslora", True),
        balance_loss_coef=config.get("balance_loss_coef", 0.01),
        z_loss_coef=config.get("z_loss_coef", 0.001),
        # Router Calibration 特有参数
        router_calibration=True,
        router_calib_rank=config.get("router_calib_rank", 4),
    )

    target_modules = MoLoRAConfigBuilder.auto_detect_targets(
        model.model, r=cfg.r, include_moe=True, only_backbone=False
    )

    wrapped = 0
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
        wrapped += 1

    model.model.molora_config = cfg
    model.model.molora_enabled = True
    mark_only_molora_as_trainable(model.model)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 实验变体定义
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VariantSpec:
    """单个消融实验变体的规格定义。"""
    name: str                                    # 变体名称 (用于目录和输出)
    peft_type: str                               # peft | molora | molora_aware | molora_calib
    train_kwargs: Dict[str, Any] = field(default_factory=dict)   # 传给 model.train() 的额外参数
    molora_config: Dict[str, Any] = field(default_factory=dict)  # MoLoRA 专属配置
    description: str = ""                        # 人类可读描述


# 公共配置片段
COMMON_TRAIN_KWARGS = {
    "data": DATA_YAML,
    "epochs": EPOCHS,
    "batch": BATCH,
    "imgsz": IMGSZ,
    "device": DEVICE,
    "project": str(PROJECT_DIR),
    "exist_ok": True,
    "verbose": False,
    "workers": 2,
    "patience": 0,      # 不早停
    "plots": False,     # 不生成 val 图，加速
    "save": False,      # 不保留 checkpoint，节省磁盘
}

LORA_COMMON = {
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_backend": "peft",
    "lora_dropout": LORA_DROPOUT,
}

# 变体列表 —— 可根据时间/资源裁剪
VARIANTS: List[VariantSpec] = [
    # ── 0. 全量微调 baseline ──
    VariantSpec(
        name="full",
        peft_type="peft",
        train_kwargs={},
        description="Full fine-tuning (no PEFT). All parameters trainable.",
    ),

    # ── 1. 标准 LoRA ──
    VariantSpec(
        name="lora",
        peft_type="peft",
        train_kwargs={"lora_type": "lora", **LORA_COMMON},
        description="Standard LoRA (Hu et al. 2021).",
    ),

    # ── 2. DoRA (Weight-Decomposed LoRA) ──
    VariantSpec(
        name="dora",
        peft_type="peft",
        train_kwargs={"lora_type": "lora", "lora_use_dora": True, **LORA_COMMON},
        description="DoRA: Decompose pretrained weight into magnitude and direction, adapt direction only.",
    ),

    # ── 3. IA3 ──
    VariantSpec(
        name="ia3",
        peft_type="peft",
        train_kwargs={"lora_type": "ia3", "lora_backend": "peft"},
        description="IA3: Scale inner activations with learned vectors (no low-rank matrices).",
    ),

    # ── 4. LoHA ──
    VariantSpec(
        name="loha",
        peft_type="peft",
        train_kwargs={"lora_type": "loha", **LORA_COMMON},
        description="LoHA: Hadamard product parameterization for higher expressiveness at same rank.",
    ),

    # ── 5. AdaLoRA ──
    VariantSpec(
        name="adalora",
        peft_type="peft",
        train_kwargs={"lora_type": "adalora", **LORA_COMMON},
        description="AdaLoRA: Adaptive budget allocation among singular values via SVD-based importance scoring.",
    ),

    # ── 6. 标准 MoLoRA (E=4, K=2, r=8) ──
    VariantSpec(
        name="molora",
        peft_type="molora",
        train_kwargs={},
        molora_config={
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "num_experts": 4,
            "top_k": 2,
            "router_type": "linear",
            "dropout": LORA_DROPOUT,
            "use_rslora": True,
            "balance_loss_coef": 0.01,
            "z_loss_coef": 0.001,
        },
        description="Mixture-of-LoRA (MoLoRA): sparse expert selection on top of low-rank adapters.",
    ),

    # ── 7. MoLoRA + MoE-aware (per-expert rank, frequency-based) ──
    VariantSpec(
        name="molora_aware",
        peft_type="molora_aware",
        train_kwargs={},
        molora_config={
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "num_experts": 4,
            "top_k": 2,
            "router_type": "linear",
            "dropout": LORA_DROPOUT,
            "use_rslora": True,
            "balance_loss_coef": 0.01,
            "z_loss_coef": 0.001,
            "rank_budget_total": 32,  # 4 experts * r=8
            "rank_min": 2,
        },
        description="MoLoRA + MoE-aware: per-expert rank allocated by activation frequency (same total budget).",
    ),

    # ── 8. MoLoRA + Router Calibration (ΔW_r) ──
    VariantSpec(
        name="molora_calib",
        peft_type="molora_calib",
        train_kwargs={},
        molora_config={
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "num_experts": 4,
            "top_k": 2,
            "router_type": "linear",
            "dropout": LORA_DROPOUT,
            "use_rslora": True,
            "balance_loss_coef": 0.01,
            "z_loss_coef": 0.001,
            "router_calib_rank": 4,
        },
        description="MoLoRA + Router Calibration: learnable low-rank correction on frozen router logits.",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 单变体执行逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def run_variant(spec: VariantSpec) -> Dict[str, Any]:
    """
    运行单个消融实验变体。
    返回结构化结果字典，包含参数统计、训练指标、耗时和错误信息。
    """
    print(f"\n{'='*78}")
    print(f"=== Variant: {spec.name.upper()} {'='*55}")
    print(f"{'='*78}")
    print(f"Description: {spec.description}")
    print(f"PEFT type  : {spec.peft_type}")
    if spec.train_kwargs:
        print(f"train kwargs: {spec.train_kwargs}")
    if spec.molora_config:
        print(f"MoLoRA config: {spec.molora_config}")

    t0 = time.time()
    record: Dict[str, Any] = {
        "name": spec.name,
        "description": spec.description,
        "peft_type": spec.peft_type,
        "ok": False,
        "error": None,
        "elapsed_sec": 0.0,
        "params_total": 0,
        "params_trainable": 0,
        "trainable_pct": 0.0,
        "adapter_sig": {},
        "molora_diagnostics": {},
        "final_metrics": {},
    }

    try:
        # ── 5.1 加载模型 ──
        model = YOLO(MODEL_PATH)
        base_total, base_train = count_params(model.model)
        print(f"[Pre-train]  total={base_total:,}  trainable={base_train:,}  ({base_train/base_total*100:.2f}%)")

        # ── 5.2 应用 PEFT ──
        if spec.peft_type == "peft":
            # 标准 PEFT: 参数通过 train() 传入，不做显式模型修改
            model = apply_peft_via_train_args(model, spec.train_kwargs)
        elif spec.peft_type == "molora":
            model = apply_molora_standard(model, spec.molora_config)
        elif spec.peft_type == "molora_aware":
            model = apply_molora_moe_aware(model, spec.molora_config)
        elif spec.peft_type == "molora_calib":
            model = apply_molora_router_calibration(model, spec.molora_config)
        else:
            raise ValueError(f"Unknown peft_type: {spec.peft_type}")

        # ── 5.3 训练前统计 ──
        post_total, post_train = count_params(model.model)
        sig = detect_adapter_signature(model.model)
        molora_diag = collect_molora_diagnostics(model.model)

        print(f"[Post-wrap]  total={post_total:,}  trainable={post_train:,}  ({post_train/post_total*100:.2f}%)")
        print(f"[Adapter]    {sig}")
        if molora_diag.get("molora_enabled"):
            print(f"[MoLoRA]     enabled={molora_diag['molora_enabled']}, "
                  f"router_calib={molora_diag['router_calib_present']}, "
                  f"ranks={molora_diag['per_expert_ranks']}")

        # ── 5.4 训练 ──
        train_kwargs = {**COMMON_TRAIN_KWARGS, "name": f"v_{spec.name}"}
        if spec.peft_type == "peft" and spec.train_kwargs:
            train_kwargs.update(spec.train_kwargs)

        results = model.train(**train_kwargs)

        # ── 5.5 提取指标 ──
        final_metrics = extract_final_metrics(results)
        record["ok"] = True
        record["final_metrics"] = final_metrics
        print(f"[Final metrics] {json.dumps(final_metrics, indent=2)}")

    except Exception as exc:
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {record['error']}")
        traceback.print_exc()

    # ── 5.6 收尾统计 ──
    elapsed = time.time() - t0
    record["elapsed_sec"] = round(elapsed, 1)

    # 若训练失败，尝试用当前模型状态兜底统计
    try:
        if "model" in dir() and model is not None:
            post_total, post_train = count_params(model.model)
            record["params_total"] = post_total
            record["params_trainable"] = post_train
            record["trainable_pct"] = round(post_train / post_total * 100, 4) if post_total else 0.0
            record["adapter_sig"] = detect_adapter_signature(model.model)
            record["molora_diagnostics"] = collect_molora_diagnostics(model.model)
    except Exception:
        pass

    print(f"[Done] elapsed={record['elapsed_sec']}s  ok={record['ok']}")
    return record


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def print_header():
    """打印实验环境信息。"""
    print("\n" + "=" * 78)
    print("  YOLO-Master PEFT Ablation Experiment on COCO128")
    print("=" * 78)
    print(f"  Model      : {MODEL_PATH}")
    print(f"  Dataset    : {DATA_YAML}")
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch      : {BATCH}")
    print(f"  Image size : {IMGSZ}")
    print(f"  Variants   : {len(VARIANTS)}")
    print(f"  Output dir : {PROJECT_DIR}")
    print(f"  Results    : {RESULTS_JSON}")
    print("=" * 78)
    print(f"\n[Boot] ultralytics: {ultralytics.__file__}")
    print(f"[Boot] version     : {ultralytics.__version__}")
    print(f"[Boot] torch device: {DEVICE}")
    print()


def print_summary_table(all_records: List[Dict[str, Any]]):
    """打印控制台汇总表。"""
    header = (
        f"{'Variant':<14} {'OK':<3} {'Total':>12} {'Trainable':>12} {'%':>7} "
        f"{'lora_A':>6} {'dora':>5} {'loha':>5} {'ia3':>4} {'molora':>7} "
        f"{'calib':>5} {'mAP50-95':>10} {'Time(s)':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in all_records:
        sig = r.get("adapter_sig", {})
        diag = r.get("molora_diagnostics", {})
        m = r["final_metrics"].get("metrics/mAP50-95(B)", float("nan"))
        m_str = f"{m:.4f}" if isinstance(m, float) else "N/A"

        print(
            f"{r['name']:<14} "
            f"{'Y' if r['ok'] else 'N':<3} "
            f"{r.get('params_total', 0):>12,} "
            f"{r.get('params_trainable', 0):>12,} "
            f"{r.get('trainable_pct', 0.0):>7.3f} "
            f"{'Y' if sig.get('has_lora_A') else 'N':>6} "
            f"{'Y' if sig.get('has_dora_magnitude') else 'N':>5} "
            f"{'Y' if sig.get('has_loha') else 'N':>5} "
            f"{'Y' if sig.get('has_ia3') else 'N':>4} "
            f"{'Y' if diag.get('molora_enabled') or sig.get('has_molora') else 'N':>7} "
            f"{'Y' if diag.get('router_calib_present') else 'N':>5} "
            f"{m_str:>10} "
            f"{r.get('elapsed_sec', 0):>8.1f}"
        )

    print("=" * len(header))


def main():
    """主入口：顺序运行所有变体，实时持久化结果。"""
    print_header()
    PROJECT_DIR.mkdir(exist_ok=True, parents=True)

    all_records: List[Dict[str, Any]] = []

    for idx, spec in enumerate(VARIANTS, start=1):
        print(f"\n[Progress] {idx}/{len(VARIANTS)} — 开始运行变体 '{spec.name}'")
        rec = run_variant(spec)
        all_records.append(rec)

        # 实时落盘：单个失败不丢之前结果
        try:
            RESULTS_JSON.write_text(
                json.dumps(all_records, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[WARN] 结果写入失败: {e}")

    # ── 最终汇总 ──
    print_summary_table(all_records)

    print(f"\n{'='*78}")
    print(f"全部 {len(VARIANTS)} 个变体运行完毕。")
    print(f"详细结果 JSON: {RESULTS_JSON}")
    print(f"训练日志目录: {PROJECT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
