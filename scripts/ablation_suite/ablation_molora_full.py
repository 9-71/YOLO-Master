"""
MoLoRA 完整消融实验脚本 — COCO128 (YOLO-Master)
==================================================

目的:
    在 COCO128 数据集上系统开展 MoLoRA (Mixture-of-LoRA) 的完整消融实验，
    覆盖四个核心实验模块 (M1-M4)，验证 MoLoRA 各组件的有效性与设计选择。

实验模块:
    ┌────────┬───────────────────────────────────────────────────────────────┐
    │ M1     │ 与标准 LoRA 对比: full / lora / molora / molora_aware /     │
    │        │ molora_calib                                                  │
    ├────────┼───────────────────────────────────────────────────────────────┤
    │ M2     │ Router 类型消融: linear / spatial / hybrid                    │
    ├────────┼───────────────────────────────────────────────────────────────┤
    │ M3     │ Expert 数量扫描: E=2/4/8/16 × K=1/2/4 (12 组合)              │
    ├────────┼───────────────────────────────────────────────────────────────┤
    │ M4     │ Merge/Unmerge 精度验证: 训练后 merge 权重，对比 merge/unmerge │
    │        │ 状态下 mAP 差异                                               │
    └────────┴───────────────────────────────────────────────────────────────┘

输出:
    - 结构化 JSON 结果文件 (实时写入)
    - 控制台逐模块汇总表 (参数统计、精度、耗时、adapter 签名)

用法:
    python scripts/ablation_molora_full.py

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
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 环境初始化 —— 必须在 import ultralytics 之前完成
# ═══════════════════════════════════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# 环境变量: 禁用 WandB、避免库冲突、禁用自动安装、关闭冗余输出
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

# 标准 PEFT (LoRA / DoRA) 基础设施
from ultralytics.utils.lora import apply_lora
from ultralytics.utils.lora.config import LoRAConfig

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

# 确认加载的是当前仓库的 ultralytics (避免 pip 安装的版本干扰)
import ultralytics

assert str(REPO_ROOT) in ultralytics.__file__, f"加载的不是当前仓库的 ultralytics！got {ultralytics.__file__}"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 全局配置
# ═══════════════════════════════════════════════════════════════════════════════

HERE = Path(__file__).parent
MODEL_PATH = str(REPO_ROOT / "YOLO-Master-EsMoE-N.pt")
DATA_YAML = "coco128.yaml"  # ultralytics 内置数据集，首次运行自动下载
PROJECT_DIR = HERE / "runs_molora_ablation"
RESULTS_JSON = HERE / "ablation_molora_full_results.json"

# 训练超参 (消融实验用轻量配置以加速迭代)
EPOCHS = 3
BATCH = 8
IMGSZ = 320

# 设备选择: MPS优先 → CUDA → CPU (严格使用 torch.backends.mps.is_available())
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
      - 各层 expert 数量与 top_k 配置
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

        # Expert 数量与 top_k 统计
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
        final_metrics = {k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))}
    elif hasattr(results, "metrics") and results.metrics:
        final_metrics = {k: float(v) for k, v in results.metrics.items() if isinstance(v, (int, float))}

    return final_metrics


def evaluate_model_on_coco128(model: YOLO, imgsz: int = IMGSZ, device: str = DEVICE) -> Dict[str, float]:
    """
    在 COCO128 验证集上评估模型，返回指标字典。
    用于 M4 Merge/Unmerge 精度验证等场景。
    """
    try:
        val_results = model.val(data=DATA_YAML, imgsz=imgsz, device=device, verbose=False, plots=False)
        return extract_final_metrics(val_results)
    except Exception as exc:
        return {"error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PEFT 应用器 —— 统一接口封装不同实验变体的模型修改逻辑
# ═══════════════════════════════════════════════════════════════════════════════


def apply_peft_via_train_args(model: YOLO, kwargs: Dict[str, Any]) -> YOLO:
    """
    对于标准 PEFT 方法 (LoRA, DoRA 等)，显式注入 LoRA adapter。
    当 train_kwargs 中包含 lora_type 时才执行注入；否则返回原始模型 (全量微调)。
    """
    if not kwargs.get("lora_type"):
        return model

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
    model.model = apply_lora(model.model, cfg)
    return model


def apply_molora_standard(model: YOLO, config: Dict[str, Any]) -> YOLO:
    """
    应用标准 MoLoRA (Mixture-of-LoRA)。
    使用 get_peft_molora_model 直接包装 DetectionModel。

    参数:
        config: 必须包含 num_experts, top_k, router_type 等 MoLoRA 配置项
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

    参数:
        config: 必须包含 rank_budget_total, rank_min 等 MoE-aware 特有参数
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

    参数:
        config: 必须包含 router_calib_rank 等 Router Calibration 特有参数
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


def _merge_all_molora_layers(model: nn.Module) -> None:
    """
    遍历模型所有模块，对带 merge() 方法的 MoLoRA 层执行权重 merge。
    merge 后 adapter 权重被合并回 base weight，推理等价于无 adapter。
    """
    for name, module in model.named_modules():
        if hasattr(module, "merge") and callable(getattr(module, "merge")):
            try:
                module.merge()
            except Exception as exc:
                warnings.warn(f"merge() failed for {name}: {exc}")


def _unmerge_all_molora_layers(model: nn.Module) -> None:
    """
    遍历模型所有模块，对带 unmerge() 方法的 MoLoRA 层执行权重 unmerge。
    unmerge 将 base weight 恢复为原始状态，重新启用 adapter 分支。
    """
    for name, module in model.named_modules():
        if hasattr(module, "unmerge") and callable(getattr(module, "unmerge")):
            try:
                module.unmerge()
            except Exception as exc:
                warnings.warn(f"unmerge() failed for {name}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 实验变体定义 —— 四大模块 (M1-M4)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class VariantSpec:
    """单个消融实验变体的规格定义。"""

    name: str  # 变体名称 (用于目录和输出)
    module: str  # 所属模块 M1/M2/M3/M4
    peft_type: str  # peft | molora | molora_aware | molora_calib
    train_kwargs: Dict[str, Any] = field(default_factory=dict)  # 传给 model.train() 的额外参数
    molora_config: Dict[str, Any] = field(default_factory=dict)  # MoLoRA 专属配置
    description: str = ""  # 人类可读描述
    post_train_action: str = "none"  # 训练后额外操作: none | merge_eval | unmerge_eval


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
    "patience": 0,  # 不早停
    "plots": False,  # 不生成 val 图，加速
    "save": False,  # 不保留 checkpoint，节省磁盘
}

LORA_COMMON = {
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_backend": "peft",
    "lora_dropout": LORA_DROPOUT,
}

# ─────────────────────────────────────────────────────────────────────────────
# M1: 与标准 LoRA 对比 (baseline + LoRA + MoLoRA 变体)
# ─────────────────────────────────────────────────────────────────────────────
M1_VARIANTS = [
    # ── M1.0 全量微调 baseline ──
    VariantSpec(
        name="M1_full",
        module="M1",
        peft_type="peft",
        train_kwargs={},
        description="Full fine-tuning (no PEFT). All parameters trainable.",
    ),
    # ── M1.1 标准 LoRA ──
    VariantSpec(
        name="M1_lora",
        module="M1",
        peft_type="peft",
        train_kwargs={"lora_type": "lora", **LORA_COMMON},
        description="Standard LoRA (Hu et al. 2021) for direct comparison.",
    ),
    # ── M1.2 标准 MoLoRA (E=4, K=2, r=8, linear router) ──
    VariantSpec(
        name="M1_molora",
        module="M1",
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
        description="Standard MoLoRA (E=4, K=2, r=8, linear router).",
    ),
    # ── M1.3 MoLoRA + MoE-aware (per-expert rank, frequency-based) ──
    VariantSpec(
        name="M1_molora_aware",
        module="M1",
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
            "rank_budget_total": 32,
            "rank_min": 2,
        },
        description="MoLoRA + MoE-aware: per-expert rank by activation frequency.",
    ),
    # ── M1.4 MoLoRA + Router Calibration (ΔW_r) ──
    VariantSpec(
        name="M1_molora_calib",
        module="M1",
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
        description="MoLoRA + Router Calibration: learnable low-rank correction.",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# M2: Router 类型消融 (linear / spatial / hybrid)
#     固定 E=4, K=2, r=8，仅变化 router_type
# ─────────────────────────────────────────────────────────────────────────────
M2_VARIANTS = [
    VariantSpec(
        name="M2_router_linear",
        module="M2",
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
        description="MoLoRA with linear router (scalar gating per expert).",
    ),
    VariantSpec(
        name="M2_router_spatial",
        module="M2",
        peft_type="molora",
        train_kwargs={},
        molora_config={
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "num_experts": 4,
            "top_k": 2,
            "router_type": "spatial",
            "dropout": LORA_DROPOUT,
            "use_rslora": True,
            "balance_loss_coef": 0.01,
            "z_loss_coef": 0.001,
        },
        description="MoLoRA with spatial router (spatial-aware gating maps).",
    ),
    VariantSpec(
        name="M2_router_hybrid",
        module="M2",
        peft_type="molora",
        train_kwargs={},
        molora_config={
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "num_experts": 4,
            "top_k": 2,
            "router_type": "hybrid",
            "dropout": LORA_DROPOUT,
            "use_rslora": True,
            "balance_loss_coef": 0.01,
            "z_loss_coef": 0.001,
        },
        description="MoLoRA with hybrid router (combines linear + spatial gates).",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# M3: Expert 数量扫描 (E=2/4/8/16 × K=1/2/4)
#     固定 linear router, r=8，扫描 (E, K) 组合
#     排除无效组合: K 必须 <= E
# ─────────────────────────────────────────────────────────────────────────────
M3_E_VALUES = [2, 4, 8, 16]
M3_K_VALUES = [1, 2, 4]

M3_VARIANTS: List[VariantSpec] = []
for E in M3_E_VALUES:
    for K in M3_K_VALUES:
        if K > E:
            continue  # top_k 不能超过 expert 数量
        M3_VARIANTS.append(
            VariantSpec(
                name=f"M3_E{E}_K{K}",
                module="M3",
                peft_type="molora",
                train_kwargs={},
                molora_config={
                    "r": LORA_R,
                    "alpha": LORA_ALPHA,
                    "num_experts": E,
                    "top_k": K,
                    "router_type": "linear",
                    "dropout": LORA_DROPOUT,
                    "use_rslora": True,
                    "balance_loss_coef": 0.01,
                    "z_loss_coef": 0.001,
                },
                description=f"MoLoRA expert sweep: E={E}, K={K}, linear router.",
            )
        )

# ─────────────────────────────────────────────────────────────────────────────
# M4: Merge/Unmerge 精度验证
#     训练 MoLoRA 后，分别评估 merge 状态和 unmerge 状态下的 mAP
# ─────────────────────────────────────────────────────────────────────────────
M4_VARIANTS = [
    VariantSpec(
        name="M4_molora_baseline",
        module="M4",
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
        description="MoLoRA baseline for merge/unmerge validation.",
        post_train_action="merge_eval",
    ),
]

# 合并所有变体
ALL_VARIANTS: List[VariantSpec] = M1_VARIANTS + M2_VARIANTS + M3_VARIANTS + M4_VARIANTS


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 单变体执行逻辑
# ═══════════════════════════════════════════════════════════════════════════════


def run_variant(spec: VariantSpec) -> Dict[str, Any]:
    """
    运行单个消融实验变体。
    返回结构化结果字典，包含参数统计、训练指标、耗时和错误信息。
    对于 M4，训练后还会执行 merge/unmerge 双重评估。
    """
    print(f"\n{'=' * 78}")
    print(f"=== Variant: {spec.name.upper()} {'=' * 50}")
    print(f"{'=' * 78}")
    print(f"Module     : {spec.module}")
    print(f"Description: {spec.description}")
    print(f"PEFT type  : {spec.peft_type}")
    if spec.train_kwargs:
        print(f"train kwargs: {spec.train_kwargs}")
    if spec.molora_config:
        print(f"MoLoRA config: {spec.molora_config}")
    if spec.post_train_action != "none":
        print(f"Post-train : {spec.post_train_action}")

    t0 = time.time()
    record: Dict[str, Any] = {
        "name": spec.name,
        "module": spec.module,
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
        # M4 专用字段
        "merge_metrics": {},
        "unmerge_metrics": {},
    }

    try:
        # ── 5.1 加载模型 ──
        model = YOLO(MODEL_PATH)
        base_total, base_train = count_params(model.model)
        print(f"[Pre-train]  total={base_total:,}  trainable={base_train:,}  ({base_train / base_total * 100:.2f}%)")

        # ── 5.2 应用 PEFT ──
        if spec.peft_type == "peft":
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

        print(f"[Post-wrap]  total={post_total:,}  trainable={post_train:,}  ({post_train / post_total * 100:.2f}%)")
        print(f"[Adapter]    {sig}")
        if molora_diag.get("molora_enabled"):
            print(
                f"[MoLoRA]     enabled={molora_diag['molora_enabled']}, "
                f"router_calib={molora_diag['router_calib_present']}, "
                f"ranks={molora_diag['per_expert_ranks']}"
            )
            if molora_diag.get("router_types"):
                print(f"[Router]     types={molora_diag['router_types']}")

        # ── 5.4 训练 ──
        train_kwargs = {**COMMON_TRAIN_KWARGS, "name": f"v_{spec.name}"}
        if spec.peft_type == "peft" and spec.train_kwargs:
            train_kwargs.update(spec.train_kwargs)

        results = model.train(**train_kwargs)

        # ── 5.5 提取训练后指标 ──
        final_metrics = extract_final_metrics(results)
        record["ok"] = True
        record["final_metrics"] = final_metrics
        print(f"[Final metrics] {json.dumps(final_metrics, indent=2)}")

        # ── 5.6 M4 专用: Merge/Unmerge 精度验证 ──
        if spec.module == "M4" and spec.post_train_action == "merge_eval":
            print("\n[M4] 执行 Merge/Unmerge 精度验证...")

            # 5.6.1 Unmerge 状态评估 (训练刚结束，模型处于默认 unmerge 状态)
            unmerge_metrics = evaluate_model_on_coco128(model)
            record["unmerge_metrics"] = unmerge_metrics
            print(f"[M4] Unmerge mAP50-95: {unmerge_metrics.get('metrics/mAP50-95(B)', 'N/A')}")

            # 5.6.2 Merge 状态评估
            _merge_all_molora_layers(model.model)
            merge_metrics = evaluate_model_on_coco128(model)
            record["merge_metrics"] = merge_metrics
            print(f"[M4] Merge mAP50-95: {merge_metrics.get('metrics/mAP50-95(B)', 'N/A')}")

            # 5.6.3 恢复 unmerge 状态 (避免后续实验受影响)
            _unmerge_all_molora_layers(model.model)
            print("[M4] 已恢复 unmerge 状态")

            # 计算 merge 精度损失
            um_map = record["unmerge_metrics"].get("metrics/mAP50-95(B)", float("nan"))
            m_map = record["merge_metrics"].get("metrics/mAP50-95(B)", float("nan"))
            if isinstance(um_map, float) and isinstance(m_map, float) and not (um_map != um_map or m_map != m_map):
                record["merge_gap"] = round(um_map - m_map, 6)
                print(f"[M4] Merge gap (unmerge - merge): {record['merge_gap']:.6f}")

    except Exception as exc:
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {record['error']}")
        traceback.print_exc()

    # ── 5.7 收尾统计 ──
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
    print("  YOLO-Master MoLoRA Full Ablation Experiment on COCO128")
    print("=" * 78)
    print(f"  Model      : {MODEL_PATH}")
    print(f"  Dataset    : {DATA_YAML}")
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch      : {BATCH}")
    print(f"  Image size : {IMGSZ}")
    print(f"  Variants   : {len(ALL_VARIANTS)}")
    print(f"    - M1 (vs LoRA)    : {len(M1_VARIANTS)}")
    print(f"    - M2 (Router)     : {len(M2_VARIANTS)}")
    print(f"    - M3 (Expert scan): {len(M3_VARIANTS)}")
    print(f"    - M4 (Merge/Unmerge): {len(M4_VARIANTS)}")
    print(f"  Output dir : {PROJECT_DIR}")
    print(f"  Results    : {RESULTS_JSON}")
    print("=" * 78)
    print(f"\n[Boot] ultralytics: {ultralytics.__file__}")
    print(f"[Boot] version     : {ultralytics.__version__}")
    print(f"[Boot] torch device: {DEVICE}")
    print()


def print_module_summary(all_records: List[Dict[str, Any]], module_name: str):
    """打印单个模块的汇总表。"""
    module_recs = [r for r in all_records if r.get("module") == module_name]
    if not module_recs:
        return

    header = (
        f"{'Variant':<22} {'OK':<3} {'Total':>12} {'Trainable':>12} {'%':>7} "
        f"{'molora':>7} {'calib':>5} {'mAP50-95':>10} {'Time(s)':>8}"
    )
    # M4 增加 merge/unmerge 列
    if module_name == "M4":
        header = (
            f"{'Variant':<22} {'OK':<3} {'Total':>12} {'Trainable':>12} {'%':>7} "
            f"{'Unmerge mAP':>12} {'Merge mAP':>12} {'Gap':>10} {'Time(s)':>8}"
        )

    print(f"\n{'─' * len(header)}")
    print(f"  Module {module_name}  {'─' * (len(header) - 12 - len(module_name))}")
    print(header)
    print("-" * len(header))

    for r in module_recs:
        diag = r.get("molora_diagnostics", {})
        sig = r.get("adapter_sig", {})
        m = r["final_metrics"].get("metrics/mAP50-95(B)", float("nan"))
        m_str = f"{m:.4f}" if isinstance(m, float) else "N/A"

        if module_name == "M4":
            um = r.get("unmerge_metrics", {}).get("metrics/mAP50-95(B)", float("nan"))
            mm = r.get("merge_metrics", {}).get("metrics/mAP50-95(B)", float("nan"))
            gap = r.get("merge_gap", float("nan"))
            um_str = f"{um:.4f}" if isinstance(um, float) and um == um else "N/A"
            mm_str = f"{mm:.4f}" if isinstance(mm, float) and mm == mm else "N/A"
            gap_str = f"{gap:.6f}" if isinstance(gap, float) and gap == gap else "N/A"
            print(
                f"{r['name']:<22} "
                f"{'Y' if r['ok'] else 'N':<3} "
                f"{r.get('params_total', 0):>12,} "
                f"{r.get('params_trainable', 0):>12,} "
                f"{r.get('trainable_pct', 0.0):>7.3f} "
                f"{um_str:>12} "
                f"{mm_str:>12} "
                f"{gap_str:>10} "
                f"{r.get('elapsed_sec', 0):>8.1f}"
            )
        else:
            print(
                f"{r['name']:<22} "
                f"{'Y' if r['ok'] else 'N':<3} "
                f"{r.get('params_total', 0):>12,} "
                f"{r.get('params_trainable', 0):>12,} "
                f"{r.get('trainable_pct', 0.0):>7.3f} "
                f"{'Y' if diag.get('molora_enabled') or sig.get('has_molora') else 'N':>7} "
                f"{'Y' if diag.get('router_calib_present') else 'N':>5} "
                f"{m_str:>10} "
                f"{r.get('elapsed_sec', 0):>8.1f}"
            )

    print("=" * len(header))


def print_summary_table(all_records: List[Dict[str, Any]]):
    """打印所有模块的控制台汇总表。"""
    for mod in ["M1", "M2", "M3", "M4"]:
        print_module_summary(all_records, mod)


def main():
    """主入口：顺序运行所有变体，实时持久化结果。"""
    print_header()
    PROJECT_DIR.mkdir(exist_ok=True, parents=True)

    all_records: List[Dict[str, Any]] = []

    for idx, spec in enumerate(ALL_VARIANTS, start=1):
        print(f"\n[Progress] {idx}/{len(ALL_VARIANTS)} — 开始运行变体 '{spec.name}' (模块 {spec.module})")
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

    print(f"\n{'=' * 78}")
    print(f"全部 {len(ALL_VARIANTS)} 个变体运行完毕。")
    print(f"  - M1 (vs LoRA)    : {len(M1_VARIANTS)} 个")
    print(f"  - M2 (Router)     : {len(M2_VARIANTS)} 个")
    print(f"  - M3 (Expert scan): {len(M3_VARIANTS)} 个")
    print(f"  - M4 (Merge/Unmerge): {len(M4_VARIANTS)} 个")
    print(f"详细结果 JSON: {RESULTS_JSON}")
    print(f"训练日志目录: {PROJECT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
