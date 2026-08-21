"""
Few-shot 消融实验脚本 — K-shot 目标检测微调协议 (YOLO-Master)
================================================================
目的: 在 COCO128 数据集上系统对比 Full Fine-tuning / LoRA / MoLoRA 在少数据
      场景下的性能，验证 K-shot (K=1,5,10) 目标检测微调协议的有效性。

实验协议:
  - 从 COCO128 训练集中随机采样 K 张图像作为训练子集。
  - 验证集使用 COCO128 全部 128 张图像（标准 few-shot 评估协议）。
  - 每种 (K, method) 组合在 3 个不同随机种子下运行，报告均值±标准差。
  - 所有实验禁用 WandB，使用轻量训练配置以加速迭代。

对比方法:
  1. full    — 全量微调 (baseline，所有参数可训练)
  2. lora    — 标准 LoRA (r=8, alpha=16, peft backend)
  3. molora  — MoLoRA (E=4, K=2, r=8, 标准 Mixture-of-LoRA)

输出指标:
  - mAP50      (metrics/mAP50(B))
  - mAP50-95   (metrics/mAP50-95(B))
  - 训练时间   (秒)
  - 可训练参数量及占比

用法:
    python scripts/ablation_fewshot.py

环境:
  - 设备优先 MPS (Apple Silicon), 回退 CUDA / CPU
  - 所有实验禁用 WandB (WANDB_MODE=disabled)
  - 数据集: ultralytics 内置 coco128.yaml (首次运行自动下载)
  - 模型: YOLO-Master-EsMoE-N.pt (项目根目录)
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
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

import torch
import torch.nn as nn

# 关闭 ultralytics 的 wandb 上报
from ultralytics.utils import SETTINGS

SETTINGS["wandb"] = False

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

# MoLoRA 基础设施
from ultralytics.nn.peft.molora import (
    MoLoRAConfig,
    get_peft_molora_model,
)

# 标准 PEFT (LoRA) 基础设施
from ultralytics.utils.lora import apply_lora
from ultralytics.utils.lora.config import LoRAConfig

# MoE 诊断基础设施
from ultralytics.nn.modules.moe.analysis import (
    ExpertUsageTracker,
    RoutingCollapseDetector,
)
from ultralytics.nn.modules.moe.diagnostics import (
    collect_moe_diagnostics,
    diagnostics_to_dict,
)

# 确认加载的是当前仓库的 ultralytics
import ultralytics

assert str(REPO_ROOT) in ultralytics.__file__, f"加载的不是当前仓库的 ultralytics！got {ultralytics.__file__}"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 全局配置
# ═══════════════════════════════════════════════════════════════════════════════

HERE = Path(__file__).parent
MODEL_PATH = str(REPO_ROOT / "YOLO-Master-EsMoE-N.pt")
DATA_YAML = "coco128.yaml"  # ultralytics 内置数据集，首次运行自动下载
PROJECT_DIR = HERE / "runs_fewshot_ablation"
RESULTS_JSON = HERE / "ablation_fewshot_results.json"
TEMP_ROOT = HERE / ".temp_kshot"  # K-shot 临时数据集目录

# 训练超参 (few-shot 场景下使用更多 epoch 以补偿数据不足)
EPOCHS = 20
BATCH = 8
IMGSZ = 320

# Few-shot 配置
K_SHOTS = [1, 5, 10]  # K-shot 采样数量
SEEDS = [42, 123, 456]  # 随机种子，用于统计稳定性
METHODS = ["full", "lora", "molora"]

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
# 2. K-shot 数据集采样工具
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_coco128() -> Dict[str, Any]:
    """
    确保 COCO128 数据集已下载并返回解析后的数据集信息字典。

    利用 ultralytics.data.utils.check_det_dataset 自动处理下载与路径解析，
    返回包含 path / train / val / nc / names 等键的字典。
    """
    print(f"[Dataset] 检查并解析 {DATA_YAML} ...")
    info = check_det_dataset(DATA_YAML, autodownload=True)
    print(f"[Dataset] 根目录: {info['path']}")
    print(f"[Dataset] 训练集: {info['train']} ({len(list(Path(info['train']).glob('*.jpg')))} images)")
    print(f"[Dataset] 验证集: {info['val']}")
    return info


def create_kshot_subset(
    k: int,
    seed: int,
    coco_info: Dict[str, Any],
    temp_root: Path = TEMP_ROOT,
) -> Path:
    """
    从 COCO128 中创建 K-shot 训练子集，返回临时 data.yaml 的路径。

    实现细节:
      1. 读取 COCO128 训练集所有图像路径。
      2. 使用给定随机种子采样 K 张图像（不放回）。
      3. 在临时目录下构建标准 YOLO 数据集结构:
         temp_dir/images/train/   — K 张训练图像
         temp_dir/labels/train/   — 对应的 YOLO 格式标签
         temp_dir/images/val/     → 符号链接到原始验证集
         temp_dir/labels/val/     → 符号链接到原始标签
      4. 生成 data.yaml，val 指向原始 COCO128 的完整验证集。

    Args:
        k: 采样图像数量。
        seed: 随机种子，保证实验可复现。
        coco_info: check_det_dataset 返回的数据集信息字典。
        temp_root: 临时数据集根目录。

    Returns:
        生成的临时 data.yaml 绝对路径。

    Raises:
        ValueError: 当 K 大于训练集总图像数时抛出。
    """
    rng = random.Random(seed)
    coco_root = Path(coco_info["path"])

    # 收集所有训练图像
    train_path = Path(coco_info["train"])
    if train_path.is_dir():
        all_images = sorted(train_path.glob("*.jpg"))
        if not all_images:
            all_images = sorted(train_path.glob("*.png"))
    else:
        # 若为 .txt 文件（图像路径列表）
        with open(train_path, "r", encoding="utf-8") as fh:
            all_images = [Path(line.strip()) for line in fh if line.strip()]

    if len(all_images) < k:
        raise ValueError(f"K-shot 采样失败: 请求 K={k}，但训练集仅有 {len(all_images)} 张图像。")

    # 随机采样 K 张（不放回）
    selected = rng.sample(all_images, k)
    selected_stems = {p.stem for p in selected}
    print(f"[K-shot] K={k}, seed={seed} — 采样 {len(selected)} 张训练图像")

    # 创建临时目录结构
    temp_dir = temp_root / f"k{k}_seed{seed}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    train_img_dir = temp_dir / "images" / "train"
    train_lbl_dir = temp_dir / "labels" / "train"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 复制训练图像与对应标签
    for img_path in selected:
        shutil.copy2(img_path, train_img_dir / img_path.name)
        # 标签可能在 labels/train2017/ 或 labels/train/
        lbl_name = f"{img_path.stem}.txt"
        copied = False
        for lbl_dir in [
            coco_root / "labels" / "train2017",
            coco_root / "labels" / "train",
        ]:
            lbl_path = lbl_dir / lbl_name
            if lbl_path.exists():
                shutil.copy2(lbl_path, train_lbl_dir / lbl_name)
                copied = True
                break
        if not copied:
            print(f"[WARN] 未找到标签: {img_path.name}")

    # 验证集：创建符号链接指向原始完整数据（COCO128 的 val 即 train2017）
    val_img_src = Path(coco_info["val"])
    val_img_dst = temp_dir / "images" / "val"
    val_lbl_dst = temp_dir / "labels" / "val"

    if val_img_src.is_dir():
        val_img_dst.symlink_to(val_img_src, target_is_directory=True)
        # 推断标签目录
        val_lbl_src = None
        for cand in [
            coco_root / "labels" / "train2017",
            coco_root / "labels" / "val",
        ]:
            if cand.exists():
                val_lbl_src = cand
                break
        if val_lbl_src is not None:
            val_lbl_dst.symlink_to(val_lbl_src, target_is_directory=True)
    else:
        # 若 val 为文件列表，直接复制
        shutil.copy2(val_img_src, temp_dir / "val.txt")

    # 生成 data.yaml
    yaml_path = temp_dir / "data.yaml"
    names = coco_info.get("names", {})
    nc = coco_info.get("nc", len(names))

    # 将 names 序列化为 YAML 格式
    names_lines = ""
    if isinstance(names, dict):
        for idx in sorted(names.keys(), key=int):
            names_lines += f"  {idx}: {names[idx]}\n"
    elif isinstance(names, list):
        for idx, name in enumerate(names):
            names_lines += f"  {idx}: {name}\n"

    yaml_content = (
        f"# Auto-generated K-shot COCO128 subset (K={k}, seed={seed})\n"
        f"path: {temp_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {nc}\n"
        f"names:\n"
        f"{names_lines}"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def cleanup_temp_datasets() -> None:
    """
    清理所有临时 K-shot 数据集目录。
    注册为 atexit 回调，确保脚本退出时释放磁盘空间。
    """
    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT)
        print(f"[Cleanup] 已清理临时目录: {TEMP_ROOT}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 工具函数
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


def run_moe_diagnostics(model: YOLO, data_yaml: str, batch: int = 8) -> Dict[str, Any]:
    """
    对 MoLoRA 模型运行 MoE 诊断，收集专家使用统计与路由健康度。

    诊断流程:
      1. 使用 ExpertUsageTracker 注册 router forward hooks。
      2. 在验证集上运行一次 inference，收集专家选择统计。
      3. 使用 RoutingCollapseDetector 检测路由崩溃与 dead experts。
      4. 使用 collect_moe_diagnostics 收集结构化诊断快照。

    Args:
        model: 已训练好的 YOLO 模型（含 MoLoRA 层）。
        data_yaml: 数据集 YAML 路径。
        batch: 验证 batch size。

    Returns:
        结构化诊断字典，包含 usage_stats / collapse / dead_experts / snapshots。
    """
    diag: Dict[str, Any] = {"usage_stats": {}, "collapse": False, "dead_experts": {}, "snapshots": []}

    try:
        # 1. ExpertUsageTracker — 收集 router 前向传播统计
        with ExpertUsageTracker(model.model) as tracker:
            model.val(data=data_yaml, batch=batch, imgsz=IMGSZ, device=DEVICE, verbose=False)

        # 序列化 usage_stats（转换为 JSON-safe 格式）
        diag["usage_stats"] = {
            layer_name: {
                str(eid): {"hits": stats.hits, "avg_weight": stats.avg_weight} for eid, stats in experts.items()
            }
            for layer_name, experts in tracker.usage_stats.items()
        }
        diag["total_tokens"] = tracker.total_tokens

        # 2. RoutingCollapseDetector — 检测崩溃与 dead experts
        detector = RoutingCollapseDetector(collapse_threshold=0.8, dead_threshold=0.05)
        diagnosis = detector.diagnose(model.model)
        diag["collapse"] = any(d.get("collapsed", False) for d in diagnosis.values())
        diag["dead_experts"] = {name: d.get("dead_experts", []) for name, d in diagnosis.items()}
        diag["max_usage"] = {name: d.get("max_usage", 0.0) for name, d in diagnosis.items()}

        # 3. collect_moe_diagnostics — 结构化快照
        snapshots = collect_moe_diagnostics(model.model, collapse_threshold=0.8)
        diag["snapshots"] = diagnostics_to_dict(snapshots)

    except Exception as exc:
        diag["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    return diag


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PEFT 应用器
# ═══════════════════════════════════════════════════════════════════════════════


def apply_full_finetuning(model: YOLO) -> YOLO:
    """
    全量微调：不做任何 PEFT 修改，保持所有参数可训练。

    Args:
        model: 加载好的 YOLO 模型。

    Returns:
        原模型（未修改）。
    """
    return model


def apply_lora_via_train_args(model: YOLO) -> YOLO:
    """
    标准 LoRA：在模型层面预注入 adapter。
    使用 apply_lora() 与 LoRAConfig 显式注入，确保 trainable_pct > 0。

    Args:
        model: 加载好的 YOLO 模型。

    Returns:
        已注入 LoRA adapter 并冻结 backbone 的模型。
    """
    cfg = LoRAConfig(
        r=LORA_R,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
        backend="peft",
        variant="lora",
        peft_type="lora",
    )
    model = apply_lora(model.model, cfg)
    if not isinstance(model, YOLO):
        yolo_model = YOLO(MODEL_PATH)
        yolo_model.model = model
        model = yolo_model
    return model


def apply_molora(model: YOLO, config: Optional[Dict[str, Any]] = None) -> YOLO:
    """
    应用标准 MoLoRA (Mixture-of-LoRA)。
    使用 get_peft_molora_model 直接包装 DetectionModel，并自动冻结非 adapter 参数。

    Args:
        model: 加载好的 YOLO 模型。
        config: 可选的 MoLoRA 配置字典；默认使用标准预设 (E=4, K=2, r=8)。

    Returns:
        已注入 MoLoRA 层并冻结 backbone 的模型。
    """
    cfg = MoLoRAConfig(
        r=config.get("r", LORA_R) if config else LORA_R,
        alpha=config.get("alpha", LORA_ALPHA) if config else LORA_ALPHA,
        num_experts=config.get("num_experts", 4) if config else 4,
        top_k=config.get("top_k", 2) if config else 2,
        router_type=config.get("router_type", "linear") if config else "linear",
        dropout=config.get("dropout", LORA_DROPOUT) if config else LORA_DROPOUT,
        use_rslora=config.get("use_rslora", True) if config else True,
        balance_loss_coef=config.get("balance_loss_coef", 0.01) if config else 0.01,
        z_loss_coef=config.get("z_loss_coef", 0.001) if config else 0.001,
    )
    get_peft_molora_model(model.model, cfg)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 实验变体定义与执行逻辑
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ExperimentSpec:
    """单个 few-shot 实验的配置规格。"""

    method: str  # full | lora | molora
    k: int  # K-shot 数量
    seed: int  # 随机种子
    epochs: int = EPOCHS
    batch: int = BATCH
    imgsz: int = IMGSZ
    description: str = ""


@dataclass
class CommonTrainKwargs:
    """公共训练参数字典模板。"""

    epochs: int = EPOCHS
    imgsz: int = IMGSZ
    device: str = DEVICE
    project: str = str(PROJECT_DIR)
    exist_ok: bool = True
    verbose: bool = False
    workers: int = 2
    patience: int = 0  # 不早停
    plots: bool = False  # 不生成 val 图，加速
    save: bool = False  # 不保留 checkpoint，节省磁盘

    def to_dict(self, name: str, data_yaml: str, batch: int) -> Dict[str, Any]:
        return {
            "data": data_yaml,
            "epochs": self.epochs,
            "batch": batch,
            "imgsz": self.imgsz,
            "device": self.device,
            "project": self.project,
            "name": name,
            "exist_ok": self.exist_ok,
            "verbose": self.verbose,
            "workers": self.workers,
            "patience": self.patience,
            "plots": self.plots,
            "save": self.save,
        }


def run_experiment(spec: ExperimentSpec, data_yaml: Path) -> Dict[str, Any]:
    """
    运行单个 few-shot 实验变体。

    执行流程:
      1. 加载预训练模型 YOLO-Master-EsMoE-N.pt。
      2. 根据 method 应用对应的 PEFT 策略（full / lora / molora）。
      3. 统计参数并打印 adapter 签名。
      4. 执行训练（K-shot 子集）。
      5. 提取 mAP 指标与训练时间。
      6. 若为 MoLoRA，附加运行 MoE 诊断。

    Args:
        spec: 实验规格。
        data_yaml: K-shot 临时数据集的 YAML 路径。

    Returns:
        结构化结果字典，包含 metrics / timing / params / diagnostics / error。
    """
    print(f"\n{'=' * 78}")
    print(f"=== K={spec.k:2d} | Method={spec.method.upper():8s} | Seed={spec.seed:3d} {'=' * 35}")
    print(f"{'=' * 78}")
    print(f"Description: {spec.description}")
    print(f"Data YAML  : {data_yaml}")

    t0 = time.time()
    record: Dict[str, Any] = {
        "method": spec.method,
        "k": spec.k,
        "seed": spec.seed,
        "ok": False,
        "error": None,
        "elapsed_sec": 0.0,
        "params_total": 0,
        "params_trainable": 0,
        "trainable_pct": 0.0,
        "adapter_sig": {},
        "final_metrics": {},
        "moe_diagnostics": {},
    }

    try:
        # ── 5.1 加载模型 ──
        model = YOLO(MODEL_PATH)
        base_total, base_train = count_params(model.model)
        print(f"[Pre-train]  total={base_total:,}  trainable={base_train:,}  ({base_train / base_total * 100:.2f}%)")

        # ── 5.2 应用 PEFT ──
        if spec.method == "full":
            model = apply_full_finetuning(model)
        elif spec.method == "lora":
            model = apply_lora_via_train_args(model)
        elif spec.method == "molora":
            model = apply_molora(model)
        else:
            raise ValueError(f"Unknown method: {spec.method}")

        # ── 5.3 训练前统计 ──
        post_total, post_train = count_params(model.model)
        sig = detect_adapter_signature(model.model)
        print(f"[Post-wrap]  total={post_total:,}  trainable={post_train:,}  ({post_train / post_total * 100:.2f}%)")
        print(f"[Adapter]    {sig}")

        # ── 5.4 训练 ──
        common = CommonTrainKwargs(epochs=spec.epochs, imgsz=spec.imgsz)
        # Few-shot 场景下 batch 不应超过 K
        effective_batch = min(spec.batch, spec.k)
        train_kwargs = common.to_dict(
            name=f"k{spec.k}_{spec.method}_s{spec.seed}",
            data_yaml=str(data_yaml),
            batch=effective_batch,
        )

        if spec.method == "lora":
            train_kwargs.update(
                {
                    "lora_type": "lora",
                    "lora_r": LORA_R,
                    "lora_alpha": LORA_ALPHA,
                    "lora_backend": "peft",
                    "lora_dropout": LORA_DROPOUT,
                }
            )

        results = model.train(**train_kwargs)

        # ── 5.5 提取指标 ──
        final_metrics = extract_final_metrics(results)
        record["ok"] = True
        record["final_metrics"] = final_metrics
        print(f"[Final metrics] {json.dumps(final_metrics, indent=2)}")

        # ── 5.6 MoE 诊断（仅 MoLoRA）──
        if spec.method == "molora":
            print("[MoE-Diag] 运行专家使用诊断 ...")
            moe_diag = run_moe_diagnostics(model, str(data_yaml), batch=min(8, spec.k))
            record["moe_diagnostics"] = moe_diag
            if moe_diag.get("collapse"):
                print("[MoE-Diag] ⚠️ 检测到路由崩溃！")
            dead = moe_diag.get("dead_experts", {})
            if any(dead.values()):
                print(f"[MoE-Diag] ⚠️ 检测到 dead experts: {dead}")

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
    except Exception:
        pass

    print(f"[Done] elapsed={record['elapsed_sec']}s  ok={record['ok']}")
    return record


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 汇总与报告
# ═══════════════════════════════════════════════════════════════════════════════


def compute_summary(all_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按 (K, method) 聚合多次运行的结果，计算均值与标准差。

    聚合字段:
      - mAP50_mean / mAP50_std
      - mAP50_95_mean / mAP50_95_std
      - elapsed_mean / elapsed_std
      - trainable_pct_mean
      - success_rate (成功运行次数 / 总运行次数)
    """
    from collections import defaultdict
    import statistics

    groups = defaultdict(list)
    for rec in all_records:
        key = (rec["k"], rec["method"])
        groups[key].append(rec)

    summary = []
    for (k, method), records in sorted(groups.items()):
        ok_records = [r for r in records if r["ok"]]
        n_total = len(records)
        n_ok = len(ok_records)

        entry: Dict[str, Any] = {
            "k": k,
            "method": method,
            "runs": n_total,
            "success_rate": round(n_ok / n_total, 2) if n_total else 0.0,
        }

        if ok_records:
            map50_vals = [r["final_metrics"].get("metrics/mAP50(B)", float("nan")) for r in ok_records]
            map50_95_vals = [r["final_metrics"].get("metrics/mAP50-95(B)", float("nan")) for r in ok_records]
            elapsed_vals = [r["elapsed_sec"] for r in ok_records]
            pct_vals = [r.get("trainable_pct", 0.0) for r in ok_records]

            def _mean_std(vals):
                clean = [v for v in vals if isinstance(v, (int, float)) and not (v != v)]  # NaN 检查
                if not clean:
                    return float("nan"), float("nan")
                return round(statistics.mean(clean), 4), round(statistics.stdev(clean), 4) if len(clean) > 1 else 0.0

            m50, s50 = _mean_std(map50_vals)
            m95, s95 = _mean_std(map50_95_vals)
            met, set_ = _mean_std(elapsed_vals)
            mpct, _ = _mean_std(pct_vals)

            entry.update(
                {
                    "mAP50_mean": m50,
                    "mAP50_std": s50,
                    "mAP50_95_mean": m95,
                    "mAP50_95_std": s95,
                    "elapsed_mean": met,
                    "elapsed_std": set_,
                    "trainable_pct_mean": mpct,
                }
            )

        summary.append(entry)

    return summary


def print_header():
    """打印实验环境信息。"""
    print("\n" + "=" * 78)
    print("  YOLO-Master Few-Shot Ablation Experiment — K-shot Protocol")
    print("=" * 78)
    print(f"  Model      : {MODEL_PATH}")
    print(f"  Base Data  : {DATA_YAML}")
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch      : {BATCH}")
    print(f"  Image size : {IMGSZ}")
    print(f"  K shots    : {K_SHOTS}")
    print(f"  Seeds      : {SEEDS}")
    print(f"  Methods    : {METHODS}")
    print(f"  Output dir : {PROJECT_DIR}")
    print(f"  Results    : {RESULTS_JSON}")
    print("=" * 78)
    print(f"\n[Boot] ultralytics: {ultralytics.__file__}")
    print(f"[Boot] version     : {ultralytics.__version__}")
    print(f"[Boot] torch device: {DEVICE}")
    print()


def print_summary_table(all_records: List[Dict[str, Any]], summary: List[Dict[str, Any]]):
    """打印控制台汇总表：先逐条结果，再聚合统计。"""
    # ── 逐条明细 ──
    print("\n" + "=" * 100)
    print("  DETAILED RESULTS")
    print("=" * 100)
    header = (
        f"{'K':>3} {'Method':>8} {'Seed':>5} {'OK':>3} {'Trainable%':>10} {'mAP50':>10} {'mAP50-95':>10} {'Time(s)':>8}"
    )
    print(header)
    print("-" * 100)
    for r in all_records:
        m50 = r["final_metrics"].get("metrics/mAP50(B)", float("nan"))
        m95 = r["final_metrics"].get("metrics/mAP50-95(B)", float("nan"))
        m50_str = f"{m50:.4f}" if isinstance(m50, (int, float)) and m50 == m50 else "N/A"
        m95_str = f"{m95:.4f}" if isinstance(m95, (int, float)) and m95 == m95 else "N/A"
        print(
            f"{r['k']:>3} {r['method']:>8} {r['seed']:>5} "
            f"{'Y' if r['ok'] else 'N':>3} "
            f"{r.get('trainable_pct', 0.0):>10.3f} "
            f"{m50_str:>10} {m95_str:>10} {r.get('elapsed_sec', 0):>8.1f}"
        )
    print("=" * 100)

    # ── 聚合统计 ──
    print("\n" + "=" * 100)
    print("  AGGREGATED SUMMARY (mean ± std)")
    print("=" * 100)
    header2 = (
        f"{'K':>3} {'Method':>8} {'Runs':>5} {'OK%':>6} "
        f"{'Trainable%':>10} {'mAP50':>14} {'mAP50-95':>14} {'Time(s)':>10}"
    )
    print(header2)
    print("-" * 100)
    for s in summary:
        m50_str = (
            f"{s['mAP50_mean']:.4f}±{s['mAP50_std']:.4f}"
            if isinstance(s.get("mAP50_mean"), (int, float)) and s["mAP50_mean"] == s["mAP50_mean"]
            else "N/A"
        )
        m95_str = (
            f"{s['mAP50_95_mean']:.4f}±{s['mAP50_95_std']:.4f}"
            if isinstance(s.get("mAP50_95_mean"), (int, float)) and s["mAP50_95_mean"] == s["mAP50_95_mean"]
            else "N/A"
        )
        t_str = (
            f"{s['elapsed_mean']:.1f}±{s['elapsed_std']:.1f}"
            if isinstance(s.get("elapsed_mean"), (int, float)) and s["elapsed_mean"] == s["elapsed_mean"]
            else "N/A"
        )
        print(
            f"{s['k']:>3} {s['method']:>8} {s['runs']:>5} "
            f"{s['success_rate'] * 100:>5.0f}% "
            f"{s.get('trainable_pct_mean', 0.0):>10.3f} "
            f"{m50_str:>14} {m95_str:>14} {t_str:>10}"
        )
    print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 主流程
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """
    主入口：
      1. 解析 COCO128 数据集并确保已下载。
      2. 对每个 K ∈ K_SHOTS、每个 seed ∈ SEEDS、每个 method ∈ METHODS，
         创建 K-shot 子集并运行训练/评估。
      3. 每个实验完成后实时持久化结果到 JSON。
      4. 训练结束后打印聚合汇总表。
      5. 清理临时数据集目录。
    """
    print_header()
    PROJECT_DIR.mkdir(exist_ok=True, parents=True)
    TEMP_ROOT.mkdir(exist_ok=True, parents=True)

    # 注册退出清理
    import atexit

    atexit.register(cleanup_temp_datasets)

    # 获取 COCO128 数据集信息
    try:
        coco_info = prepare_coco128()
    except Exception as exc:
        print(f"[FATAL] 数据集准备失败: {exc}")
        traceback.print_exc()
        return

    # 生成全部实验规格
    specs: List[Tuple[ExperimentSpec, Path]] = []
    for k in K_SHOTS:
        for seed in SEEDS:
            data_yaml = create_kshot_subset(k, seed, coco_info)
            for method in METHODS:
                desc = {
                    "full": "Full fine-tuning (all params trainable)",
                    "lora": f"Standard LoRA (r={LORA_R}, alpha={LORA_ALPHA})",
                    "molora": f"MoLoRA (E=4, K=2, r={LORA_R})",
                }.get(method, "")
                specs.append(
                    (
                        ExperimentSpec(method=method, k=k, seed=seed, description=desc),
                        data_yaml,
                    )
                )

    total = len(specs)
    print(f"\n[Plan] 共 {total} 个实验待运行 ({len(K_SHOTS)} K × {len(SEEDS)} seeds × {len(METHODS)} methods)")

    all_records: List[Dict[str, Any]] = []

    for idx, (spec, data_yaml) in enumerate(specs, start=1):
        print(f"\n[Progress] {idx}/{total} — 开始运行实验")
        rec = run_experiment(spec, data_yaml)
        all_records.append(rec)

        # 实时落盘：单个失败不丢之前结果
        try:
            RESULTS_JSON.write_text(
                json.dumps(
                    {
                        "records": all_records,
                        "summary": compute_summary(all_records),
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[WARN] 结果写入失败: {e}")

    # ── 最终汇总 ──
    summary = compute_summary(all_records)
    print_summary_table(all_records, summary)

    # 最终保存带 summary 的完整结果
    try:
        RESULTS_JSON.write_text(
            json.dumps(
                {
                    "records": all_records,
                    "summary": summary,
                    "config": {
                        "model": MODEL_PATH,
                        "data": DATA_YAML,
                        "device": DEVICE,
                        "epochs": EPOCHS,
                        "batch": BATCH,
                        "imgsz": IMGSZ,
                        "k_shots": K_SHOTS,
                        "seeds": SEEDS,
                        "methods": METHODS,
                    },
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[WARN] 最终结果写入失败: {e}")

    print(f"\n{'=' * 78}")
    print(f"全部 {total} 个实验运行完毕。")
    print(f"详细结果 JSON: {RESULTS_JSON}")
    print(f"训练日志目录: {PROJECT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
