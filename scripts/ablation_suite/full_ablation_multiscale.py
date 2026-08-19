"""
多分辨率+多尺度消融脚本 — COCO128 (YOLO-Master)
========================================================
目的：在 COCO128 上对比 320/640/1280 三种分辨率下 LoRA 和 MoLoRA 的性能差异，
      同时收集 mAP_small、medium、large 分解，验证 structure-aware placement
      在 neck vs backbone 的差异化策略在不同尺度目标上的效果。

输出：
  - 实时 JSON 结果文件 (ExperimentResult 结构)
  - 控制台汇总表 (含分辨率、placement、scale-wise mAP)
  - 分辨率-精度 trade-off 曲线 (PNG)

用法：
    python scripts/full_ablation_multiscale.py

环境：
  - 设备优先 MPS (Apple Silicon) → CUDA → CPU
  - 所有实验禁用 WandB (WANDB_MODE=disabled)
  - 数据集: ultralytics 内置 coco128.yaml
  - 模型: YOLO-Master-EsMoE-N.pt (项目根目录)
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 环境初始化 —— 必须在 import ultralytics 之前完成
# ═══════════════════════════════════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ablation_suite"))

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")

import torch
import torch.nn as nn

# 关闭 ultralytics 的 wandb 上报
from ultralytics.utils import SETTINGS, LOGGER
SETTINGS["wandb"] = False

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import check_requirements

# MoLoRA 基础设施
from ultralytics.nn.peft.molora import (
    MoLoRAConfig,
    MoLoRAConfigBuilder,
    get_peft_molora_model,
)

from ultralytics.utils.lora import apply_lora
from ultralytics.utils.lora.config import LoRAConfig

# 确认加载的是当前仓库的 ultralytics
import ultralytics
assert str(REPO_ROOT) in ultralytics.__file__, (
    f"加载的不是当前仓库的 ultralytics！got {ultralytics.__file__}"
)

# 导入统一数据结构规范
from full_ablation_spec import (
    VariantConfig,
    ExperimentResult,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 全局配置
# ═══════════════════════════════════════════════════════════════════════════════

HERE = Path(__file__).parent
MODEL_PATH = str(REPO_ROOT / "YOLO-Master-EsMoE-N.pt")
DATA_YAML = "coco128.yaml"
PROJECT_DIR = HERE / "runs_multiscale_ablation"
RESULTS_JSON = HERE / "full_ablation_multiscale_results.json"
PLOT_PATH = HERE / "multiscale_tradeoff_curve.png"

# 训练超参 (消融实验用轻量配置以加速迭代)
EPOCHS = 3

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
    """检测模型中各类型 PEFT adapter 的存在性与数量。"""
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
    """收集 MoLoRA 特有的诊断信息。"""
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


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Scale-Aware Validator —— 捕获 mAP small/medium/large
# ═══════════════════════════════════════════════════════════════════════════════

_original_eval_json = DetectionValidator.eval_json


def _patched_eval_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Monkey-patch DetectionValidator.eval_json 以额外捕获 scale-wise mAP。"""
    stats = _original_eval_json(self, stats)
    self._scale_metrics = {}  # type: ignore
    if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
        try:
            check_requirements("faster-coco-eval>=1.6.7")
            from faster_coco_eval import COCO, COCOeval_faster

            pred_json = self.save_dir / "predictions.json"
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )
            for x in (pred_json, anno_json):
                assert x.is_file(), f"{x} not found"

            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            for iou_type in ["bbox"]:
                val = COCOeval_faster(
                    anno,
                    pred,
                    iouType=iou_type,
                    lvis_style=self.is_lvis,
                    print_function=lambda _msg: None,
                )
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]
                val.evaluate()
                val.accumulate()
                val.summarize()

                self._scale_metrics = {  # type: ignore
                    "AP_small": val.stats_as_dict.get("AP_small"),
                    "AP_medium": val.stats_as_dict.get("AP_medium"),
                    "AP_large": val.stats_as_dict.get("AP_large"),
                    "AP_all": val.stats_as_dict.get("AP_all"),
                    "AP_50": val.stats_as_dict.get("AP_50"),
                }
        except Exception as exc:
            self._scale_metrics = {"error": str(exc)}  # type: ignore
    return stats


DetectionValidator.eval_json = _patched_eval_json


def run_scale_aware_validation(model: YOLO, imgsz: int, name: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    运行 scale-aware 验证，返回 (metrics_dict, scale_metrics_dict)。
    直接创建 DetectionValidator 实例并运行，避免 model.val() 返回值的限制。
    """
    val_args = {
        **model.overrides,
        "rect": True,
        "data": DATA_YAML,
        "imgsz": imgsz,
        "device": DEVICE,
        "save_json": True,
        "mode": "val",
        "project": str(PROJECT_DIR),
        "name": f"val_{name}",
        "verbose": False,
        "plots": False,
    }
    validator = DetectionValidator(args=val_args, _callbacks=model.callbacks)
    validator(model=model.model)
    metrics = validator.metrics
    scale_metrics = getattr(validator, "_scale_metrics", {})
    return extract_final_metrics(metrics), scale_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PEFT 应用器
# ═══════════════════════════════════════════════════════════════════════════════


def apply_peft_via_train_args(model: YOLO, kwargs: Dict[str, Any]) -> YOLO:
    """标准 PEFT：显式注入 LoRA adapter。"""
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
    wrapped = apply_lora(model.model, cfg)
    if not isinstance(wrapped, YOLO):
        model.model = wrapped
    return model


def _resolve_target_modules(model: nn.Module, placement: str, r: int) -> List[str]:
    """根据 placement 策略解析目标模块列表。"""
    if placement == "backbone":
        return MoLoRAConfigBuilder.auto_detect_targets(
            model, r=r, include_moe=True, only_backbone=True
        )
    elif placement == "neck":
        full_targets = set(
            MoLoRAConfigBuilder.auto_detect_targets(
                model, r=r, include_moe=True, only_backbone=False
            )
        )
        backbone_targets = set(
            MoLoRAConfigBuilder.auto_detect_targets(
                model, r=r, include_moe=True, only_backbone=True
            )
        )
        return sorted(full_targets - backbone_targets)
    else:  # "full" or None
        return MoLoRAConfigBuilder.auto_detect_targets(
            model, r=r, include_moe=True, only_backbone=False
        )


def apply_molora(model: YOLO, config: Dict[str, Any]) -> YOLO:
    """
    应用 MoLoRA，支持 structure-aware placement (full / backbone / neck)。
    placement 通过 config["placement"] 指定，默认为 "full"。
    """
    placement = config.get("placement", "full")
    r = config.get("r", LORA_R)
    alpha = config.get("alpha", LORA_ALPHA)

    cfg = MoLoRAConfig(
        r=r,
        alpha=alpha,
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 2),
        router_type=config.get("router_type", "linear"),
        dropout=config.get("dropout", LORA_DROPOUT),
        use_rslora=config.get("use_rslora", True),
        balance_loss_coef=config.get("balance_loss_coef", 0.01),
        z_loss_coef=config.get("z_loss_coef", 0.001),
    )

    # 如果 placement 不是 full，手动指定 target_modules
    if placement in ("backbone", "neck"):
        target_modules = _resolve_target_modules(model.model, placement, r)
        if not target_modules:
            LOGGER.warning(f"[MoLoRA] placement='{placement}' 未找到任何目标模块，回退到 full")
        else:
            cfg.target_modules = target_modules
            LOGGER.info(f"[MoLoRA] placement='{placement}' 选中 {len(target_modules)} 个目标模块")

    get_peft_molora_model(model.model, cfg)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 实验变体定义
# ═══════════════════════════════════════════════════════════════════════════════

RESOLUTIONS = [320, 640, 1280]
PLACEMENTS = ["full", "backbone", "neck"]


def build_variants() -> List[VariantConfig]:
    """构建多分辨率+多尺度+多 placement 的消融变体列表。"""
    variants: List[VariantConfig] = []
    for imgsz in RESOLUTIONS:
        batch = 8 if imgsz <= 640 else 4  # 1280 分辨率降低 batch 避免 OOM

        # ── LoRA baseline (每种分辨率一个) ──
        variants.append(
            VariantConfig(
                name=f"lora_{imgsz}",
                peft_type="peft",
                description=f"Standard LoRA (r=16) at {imgsz}px — full placement baseline",
                train_kwargs={
                    "lora_type": "lora",
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "lora_backend": "peft",
                    "lora_dropout": 0.05,
                },
                epochs=EPOCHS,
                batch=batch,
                imgsz=imgsz,
            )
        )

        # ── MoLoRA with different placements ──
        for placement in PLACEMENTS:
            variants.append(
                VariantConfig(
                    name=f"molora_{placement}_{imgsz}",
                    peft_type="molora",
                    description=f"MoLoRA (E=4,K=2,r=8) at {imgsz}px — placement={placement}",
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
                        "placement": placement,
                    },
                    epochs=EPOCHS,
                    batch=batch,
                    imgsz=imgsz,
                )
            )
    return variants


VARIANTS = build_variants()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 单变体执行逻辑
# ═══════════════════════════════════════════════════════════════════════════════


def run_variant(variant: VariantConfig, seed: int = 42) -> ExperimentResult:
    """
    运行单个消融实验变体，返回 ExperimentResult 结构。
    """
    print(f"\n{'='*78}")
    print(f"=== Variant: {variant.name.upper()} {'='*55}")
    print(f"{'='*78}")
    print(f"Description : {variant.description}")
    print(f"PEFT type   : {variant.peft_type}")
    print(f"Resolution  : {variant.imgsz}px")
    print(f"Batch       : {variant.batch}")
    print(f"Epochs      : {variant.epochs}")
    if variant.train_kwargs:
        print(f"train kwargs: {variant.train_kwargs}")
    if variant.molora_config:
        print(f"MoLoRA config: {variant.molora_config}")

    t0 = time.time()
    result = ExperimentResult(
        dataset="coco128",
        variant=variant.name,
        seed=seed,
        ok=False,
    )

    try:
        # ── 6.1 加载模型 ──
        model = YOLO(MODEL_PATH)
        base_total, base_train = count_params(model.model)
        print(f"[Pre-train]  total={base_total:,}  trainable={base_train:,}  ({base_train/base_total*100:.2f}%)")

        # ── 6.2 应用 PEFT ──
        if variant.peft_type == "peft":
            model = apply_peft_via_train_args(model, variant.train_kwargs)
        elif variant.peft_type == "molora":
            model = apply_molora(model, variant.molora_config)
        else:
            raise ValueError(f"Unknown peft_type: {variant.peft_type}")

        # ── 6.3 训练前统计 ──
        post_total, post_train = count_params(model.model)
        sig = detect_adapter_signature(model.model)
        molora_diag = collect_molora_diagnostics(model.model)

        print(f"[Post-wrap]  total={post_total:,}  trainable={post_train:,}  ({post_train/post_total*100:.2f}%)")
        print(f"[Adapter]    {sig}")
        if molora_diag.get("molora_enabled"):
            print(f"[MoLoRA]     enabled={molora_diag['molora_enabled']}, "
                  f"router_calib={molora_diag['router_calib_present']}, "
                  f"ranks={molora_diag['per_expert_ranks']}")
            if molora_diag.get("router_types"):
                print(f"[Router]     types={molora_diag['router_types']}")

        # ── 6.4 训练 ──
        train_kwargs = {
            "data": DATA_YAML,
            "epochs": variant.epochs,
            "batch": variant.batch,
            "imgsz": variant.imgsz,
            "device": DEVICE,
            "project": str(PROJECT_DIR),
            "name": f"train_{variant.name}",
            "exist_ok": True,
            "verbose": False,
            "workers": 2,
            "patience": 0,
            "plots": False,
            "save": False,
        }
        if variant.peft_type == "peft" and variant.train_kwargs:
            train_kwargs.update(variant.train_kwargs)

        train_results = model.train(**train_kwargs)
        final_metrics = extract_final_metrics(train_results)

        # ── 6.5 训练后 scale-aware 验证 ──
        print("[Scale-aware validation] 运行 COCO 评估以获取 small/medium/large mAP...")
        val_metrics, scale_metrics = run_scale_aware_validation(
            model, imgsz=variant.imgsz, name=variant.name
        )
        # 将验证指标合并到 final_metrics（若有重复，val 覆盖 train）
        final_metrics.update(val_metrics)

        result.ok = True
        result.final_metrics = final_metrics
        result.map_small = scale_metrics.get("AP_small")
        result.map_medium = scale_metrics.get("AP_medium")
        result.map_large = scale_metrics.get("AP_large")

        print(f"[Final metrics] {json.dumps(final_metrics, indent=2)}")
        print(f"[Scale mAP]    small={result.map_small}, medium={result.map_medium}, large={result.map_large}")

    except Exception as exc:
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {result.error}")
        traceback.print_exc()

    # ── 6.6 收尾统计 ──
    elapsed = time.time() - t0
    result.elapsed_sec = round(elapsed, 1)

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
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 主流程
# ═══════════════════════════════════════════════════════════════════════════════


def print_header():
    """打印实验环境信息。"""
    print("\n" + "=" * 78)
    print("  YOLO-Master Multi-Resolution + Multi-Scale Ablation on COCO128")
    print("=" * 78)
    print(f"  Model      : {MODEL_PATH}")
    print(f"  Dataset    : {DATA_YAML}")
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Resolutions: {RESOLUTIONS}")
    print(f"  Placements : {PLACEMENTS}")
    print(f"  Variants   : {len(VARIANTS)}")
    print(f"  Output dir : {PROJECT_DIR}")
    print(f"  Results    : {RESULTS_JSON}")
    print(f"  Plot       : {PLOT_PATH}")
    print("=" * 78)
    print(f"\n[Boot] ultralytics: {ultralytics.__file__}")
    print(f"[Boot] version     : {ultralytics.__version__}")
    print(f"[Boot] torch device: {DEVICE}")
    print()


def print_summary_table(all_records: List[ExperimentResult]):
    """打印控制台汇总表，包含 scale-wise mAP。"""
    header = (
        f"{'Variant':<22} {'OK':<3} {'Imgsz':>6} {'Total':>12} {'Trainable':>12} {'%':>7} "
        f"{'mAP50-95':>10} {'mAPsmall':>10} {'mAPmedium':>10} {'mAPlarge':>10} {'Time(s)':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in all_records:
        m = r.final_metrics.get("metrics/mAP50-95(B)", float("nan"))
        m_str = f"{m:.4f}" if isinstance(m, float) and m == m else "N/A"
        ms_str = f"{r.map_small:.4f}" if r.map_small is not None and r.map_small == r.map_small else "N/A"
        mm_str = f"{r.map_medium:.4f}" if r.map_medium is not None and r.map_medium == r.map_medium else "N/A"
        ml_str = f"{r.map_large:.4f}" if r.map_large is not None and r.map_large == r.map_large else "N/A"

        print(
            f"{r.variant:<22} "
            f"{'Y' if r.ok else 'N':<3} "
            f"{r.final_metrics.get('imgsz', 0):>6} "  # 注意：imgsz 不在 final_metrics 中，这里用 variant 的
            f"{r.params_total:>12,} "
            f"{r.params_trainable:>12,} "
            f"{r.trainable_pct:>7.3f} "
            f"{m_str:>10} "
            f"{ms_str:>10} "
            f"{mm_str:>10} "
            f"{ml_str:>10} "
            f"{r.elapsed_sec:>8.1f}"
        )

    print("=" * len(header))


def print_multiscale_summary(all_records: List[ExperimentResult]):
    """按分辨率分组打印多尺度对比表。"""
    for imgsz in RESOLUTIONS:
        recs = [r for r in all_records if f"_{imgsz}" in r.variant]
        if not recs:
            continue
        print(f"\n{'─'*78}")
        print(f"  Resolution: {imgsz}px  {'─'*(70-len(str(imgsz)))}")
        header = (
            f"{'Variant':<22} {'OK':<3} {'Total':>12} {'Trainable':>12} {'%':>7} "
            f"{'mAP50-95':>10} {'mAPsmall':>10} {'mAPmedium':>10} {'mAPlarge':>10}"
        )
        print(header)
        print("-" * len(header))
        for r in recs:
            m = r.final_metrics.get("metrics/mAP50-95(B)", float("nan"))
            m_str = f"{m:.4f}" if isinstance(m, float) and m == m else "N/A"
            ms_str = f"{r.map_small:.4f}" if r.map_small is not None and r.map_small == r.map_small else "N/A"
            mm_str = f"{r.map_medium:.4f}" if r.map_medium is not None and r.map_medium == r.map_medium else "N/A"
            ml_str = f"{r.map_large:.4f}" if r.map_large is not None and r.map_large == r.map_large else "N/A"
            print(
                f"{r.variant:<22} "
                f"{'Y' if r.ok else 'N':<3} "
                f"{r.params_total:>12,} "
                f"{r.params_trainable:>12,} "
                f"{r.trainable_pct:>7.3f} "
                f"{m_str:>10} "
                f"{ms_str:>10} "
                f"{mm_str:>10} "
                f"{ml_str:>10}"
            )
        print("=" * len(header))


def generate_tradeoff_plot(all_records: List[ExperimentResult]):
    """生成分辨率-精度 trade-off 曲线图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 不可用，跳过绘图")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Multi-Resolution vs Multi-Scale Ablation Trade-off Curves", fontsize=14, fontweight="bold")

    metrics = [
        ("mAP50-95", "metrics/mAP50-95(B)", axes[0, 0]),
        ("mAP-small", "map_small", axes[0, 1]),
        ("mAP-medium", "map_medium", axes[1, 0]),
        ("mAP-large", "map_large", axes[1, 1]),
    ]

    for title, metric_key, ax in metrics:
        for placement in PLACEMENTS + ["lora"]:
            xs = []
            ys = []
            for imgsz in RESOLUTIONS:
                if placement == "lora":
                    variant_name = f"lora_{imgsz}"
                else:
                    variant_name = f"molora_{placement}_{imgsz}"
                rec = next((r for r in all_records if r.variant == variant_name and r.ok), None)
                if rec is None:
                    continue
                if metric_key.startswith("map_"):
                    val = getattr(rec, metric_key, None)
                else:
                    val = rec.final_metrics.get(metric_key)
                if val is not None and val == val:  # 排除 NaN
                    xs.append(imgsz)
                    ys.append(val)
            if xs:
                label = "LoRA" if placement == "lora" else f"MoLoRA ({placement})"
                marker = "o" if placement == "lora" else "s"
                linestyle = "-" if placement == "lora" else "--"
                ax.plot(xs, ys, marker=marker, linestyle=linestyle, label=label, markersize=8)

        ax.set_xlabel("Input Resolution (px)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(RESOLUTIONS)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Trade-off curve saved to {PLOT_PATH}")


def main():
    """主入口：顺序运行所有变体，实时持久化结果，最后生成图表。"""
    print_header()
    PROJECT_DIR.mkdir(exist_ok=True, parents=True)

    all_records: List[ExperimentResult] = []

    for idx, variant in enumerate(VARIANTS, start=1):
        print(f"\n[Progress] {idx}/{len(VARIANTS)} — 开始运行变体 '{variant.name}'")
        rec = run_variant(variant)
        all_records.append(rec)

        # 实时落盘：单个失败不丢之前结果
        try:
            RESULTS_JSON.write_text(
                json.dumps([r.to_dict() for r in all_records], indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[WARN] 结果写入失败: {e}")

    # ── 最终汇总 ──
    print_multiscale_summary(all_records)
    print_summary_table(all_records)

    # ── 生成 trade-off 曲线 ──
    generate_tradeoff_plot(all_records)

    print(f"\n{'='*78}")
    print(f"全部 {len(VARIANTS)} 个变体运行完毕。")
    print(f"详细结果 JSON: {RESULTS_JSON}")
    print(f"训练日志目录: {PROJECT_DIR}")
    print(f"Trade-off 曲线: {PLOT_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
