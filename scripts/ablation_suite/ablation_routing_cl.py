"""
路由诊断 + 持续学习 消融实验脚本 — YOLO-Master MoLoRA
============================================================

目的:
  (a) 路由诊断 — 在 MoLoRA 训练过程中收集专家激活频率、Gini 系数、
      负载均衡曲线、router collapse 检测;
  (b) 持续学习 — 模拟 Day→Night→Fog 三域顺序训练，报告每域 mAP、
      Backward Transfer (BWT)、灾难性遗忘度。

输出:
  - JSON 结果文件 (实时写入)
  - 路由诊断可视化图表
  - 持续学习指标可视化图表

用法:
    python scripts/ablation_routing_cl.py

环境:
  - 设备优先 MPS (Apple Silicon), 回退 CUDA / CPU
  - 所有实验禁用 WandB
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

import numpy as np
import torch
import torch.nn as nn

# 关闭 ultralytics 的 wandb 上报
from ultralytics.utils import SETTINGS

SETTINGS["wandb"] = False

from ultralytics import YOLO

# MoLoRA 基础设施
from ultralytics.nn.peft.molora import (
    MoLoRAConfig,
    get_peft_molora_model,
    allocate_domain_experts,
)
from ultralytics.nn.peft.molora.layer import MoLoRALayer
from ultralytics.nn.peft.molora.model import MoLoRAModel

# 诊断基础设施
from ultralytics.nn.modules.moe.diagnostics import (
    MoELayerDiagnostic,
    RoutingCollapseDetector,
)
from ultralytics.nn.modules.moe.history import MoEDiagnosticsRecorder
from ultralytics.nn.modules.moe.scheduler import compute_gini

# 确认加载的是当前仓库的 ultralytics
import ultralytics

assert str(REPO_ROOT) in ultralytics.__file__, f"加载的不是当前仓库的 ultralytics！got {ultralytics.__file__}"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 全局配置
# ═══════════════════════════════════════════════════════════════════════════════

HERE = Path(__file__).parent
MODEL_PATH = str(REPO_ROOT / "YOLO-Master-EsMoE-N.pt")
DATA_YAML = "coco128.yaml"  # ultralytics 内置数据集，首次运行自动下载
PROJECT_DIR = HERE / "runs_routing_cl"
RESULTS_JSON = HERE / "ablation_routing_cl_results.json"

# 训练超参 (消融实验用轻量配置以加速迭代)
EPOCHS_PER_DOMAIN = 3  # 每个域训练 epoch 数
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

# MoLoRA 公共超参
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
NUM_EXPERTS = 8  # 使用 8 个专家以支持三域分配
TOP_K = 2

# 三域配置 (使用 augmentation 参数模拟不同域)
DOMAINS = ["day", "night", "fog"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 工具函数
# ═══════════════════════════════════════════════════════════════════════════════


def count_params(m: nn.Module) -> Tuple[int, int]:
    """统计模型总参数量与可训练参数量。"""
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


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


def get_map50_95(metrics: Dict[str, float]) -> float:
    """从 metrics 字典中提取 mAP50-95 值。"""
    for key in ("metrics/mAP50-95(B)", "mAP50-95", "mAP"):
        if key in metrics:
            return float(metrics[key])
    return 0.0


def collect_molora_routing_stats(model: nn.Module) -> List[Dict[str, Any]]:
    """
    遍历模型中所有 MoLoRALayer，收集当前路由统计信息。

    返回列表，每个元素为一个 MoLoRA 层的统计字典，包含:
      - layer_name: 层名称
      - num_experts: 专家数量
      - top_k: Top-K 值
      - expert_usage: 专家使用频率分布 (list)
      - dominant_expert: 主导专家索引
      - dominant_share: 主导专家占比
    """
    stats: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, MoLoRALayer):
            continue

        # 读取 last_routing_stats (由 forward 更新)
        routing_stats = getattr(module, "_last_routing_stats", None)
        if routing_stats is None:
            continue

        expert_usage = routing_stats.get("expert_usage")
        if expert_usage is None:
            continue

        # 确保是 CPU 上的 list
        if isinstance(expert_usage, torch.Tensor):
            usage_list = [float(v) for v in expert_usage.detach().cpu().tolist()]
        else:
            usage_list = [float(v) for v in expert_usage]

        if not usage_list or sum(usage_list) == 0:
            continue

        dominant_expert = int(max(range(len(usage_list)), key=usage_list.__getitem__))
        dominant_share = float(usage_list[dominant_expert])

        stats.append(
            {
                "layer_name": name,
                "num_experts": module.num_experts,
                "top_k": module.top_k,
                "expert_usage": usage_list,
                "dominant_expert": dominant_expert,
                "dominant_share": dominant_share,
            }
        )

    return stats


def compute_layer_gini(usage_list: List[float]) -> float:
    """计算单个层的专家使用 Gini 系数。"""
    if not usage_list or sum(usage_list) <= 0:
        return 0.0
    usage_t = torch.tensor(usage_list, dtype=torch.float32)
    return compute_gini(usage_t)


def compute_overall_gini(all_stats: List[Dict[str, Any]]) -> float:
    """计算所有层的平均 Gini 系数。"""
    if not all_stats:
        return 0.0
    ginis = [compute_layer_gini(s["expert_usage"]) for s in all_stats]
    return float(np.mean(ginis)) if ginis else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 路由诊断收集器 (训练回调)
# ═══════════════════════════════════════════════════════════════════════════════


class RoutingDiagnosticsCollector:
    """
    在 MoLoRA 训练过程中收集路由诊断数据。

    收集内容:
      - 每 epoch 专家激活频率 (expert_usage)
      - 每 epoch Gini 系数
      - 每 epoch 负载均衡指标 (dominant_share, dead_experts)
      - Router collapse 检测 (基于连续 epoch 的主导专家占比)

    用法:
        collector = RoutingDiagnosticsCollector(save_dir)
        model.add_callback("on_train_epoch_end", collector.on_train_epoch_end)
        # 训练完成后:
        summary = collector.get_summary()
    """

    def __init__(
        self,
        save_dir: str | Path,
        collapse_threshold: float = 0.8,
        dead_threshold: float = 0.05,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.collapse_threshold = collapse_threshold
        self.dead_threshold = dead_threshold

        # 持久化记录器
        self.recorder = MoEDiagnosticsRecorder(
            save_dir=self.save_dir / "routing_diag",
            collapse_threshold=collapse_threshold,
            dead_threshold=dead_threshold,
        )

        # 运行时状态
        self.epoch_history: List[Dict[str, Any]] = []
        self.step_counter = 0
        self.collapse_detector = RoutingCollapseDetector(
            collapse_threshold=collapse_threshold,
            dead_threshold=dead_threshold,
        )

    def on_train_epoch_end(self, trainer) -> None:
        """
        ultralytics 训练回调 — 在每个 epoch 结束时触发。

        Args:
            trainer: ultralytics 的 Trainer 对象，可通过 trainer.model 访问模型。
        """
        try:
            model = getattr(trainer, "model", None)
            if model is None:
                return

            # 兼容 trainer.model 可能是 DetectionModel 或 YOLO 包装
            detection_model = model
            if hasattr(model, "model"):
                detection_model = model.model

            epoch = int(getattr(trainer, "epoch", len(self.epoch_history)))
            self.step_counter += 1

            # 收集 MoLoRA 层路由统计
            stats = collect_molora_routing_stats(detection_model)
            if not stats:
                return

            # 转换为 MoELayerDiagnostic 供 recorder 使用
            diagnostics: List[MoELayerDiagnostic] = []
            for s in stats:
                diag = MoELayerDiagnostic(
                    name=s["layer_name"],
                    module_type="MoLoRALayer",
                    num_experts=s["num_experts"],
                    top_k=s["top_k"],
                    aux_loss=0.0,  # 回调中不收集 aux_loss
                    usage=s["expert_usage"],
                    counts=[0.0] * s["num_experts"],
                    dominant_expert=s["dominant_expert"],
                    dominant_share=s["dominant_share"],
                    mean_router_probs=None,
                    mean_topk_weight=None,
                    collapse_flag=s["dominant_share"] >= self.collapse_threshold,
                )
                diagnostics.append(diag)

            # 记录到持久化存储
            self.recorder.record(
                step=self.step_counter,
                epoch=epoch,
                diagnostics=diagnostics,
                stage="train",
            )

            # 计算全局 Gini
            overall_gini = compute_overall_gini(stats)

            # 检测 collapse (直接基于 stats，因为 RoutingCollapseDetector.diagnose
            # 查找的是 last_routing_snapshot 而非 MoLoRALayer 的 _last_routing_stats)
            collapsed_layers = [s["layer_name"] for s in stats if s["dominant_share"] >= self.collapse_threshold]

            # 汇总当前 epoch
            epoch_summary = {
                "epoch": epoch,
                "step": self.step_counter,
                "num_molora_layers": len(stats),
                "overall_gini": round(overall_gini, 6),
                "collapsed_layers": collapsed_layers,
                "num_collapsed": len(collapsed_layers),
                "layer_stats": stats,
            }
            self.epoch_history.append(epoch_summary)

        except Exception as exc:
            # 回调中不抛出异常，避免中断训练
            print(f"[WARN] RoutingDiagnosticsCollector callback error: {exc}")

    def get_summary(self) -> Dict[str, Any]:
        """返回路由诊断汇总。"""
        if not self.epoch_history:
            return {"status": "no_data"}

        ginis = [e["overall_gini"] for e in self.epoch_history]
        total_collapses = sum(e["num_collapsed"] for e in self.epoch_history)

        return {
            "status": "ok",
            "num_epochs_recorded": len(self.epoch_history),
            "gini_mean": round(float(np.mean(ginis)), 6) if ginis else 0.0,
            "gini_std": round(float(np.std(ginis)), 6) if ginis else 0.0,
            "gini_trend": ginis,
            "total_collapse_events": total_collapses,
            "epoch_history": self.epoch_history,
        }

    def export_plots(self) -> List[Path]:
        """导出诊断图表。"""
        return self.recorder.export_plots()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 持续学习管理器
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DomainConfig:
    """单个域的训练配置。"""

    name: str
    epochs: int
    augment_kwargs: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


class ContinualLearningManager:
    """
    管理 Day→Night→Fog 三域顺序持续学习。

    功能:
      - 为每个域分配专属专家 (allocate_domain_experts)
      - 顺序训练: 每域训练后保存 replay buffer
      - 评估: 每域训练后在所有域上评估 mAP
      - 计算 BWT (Backward Transfer) 和灾难性遗忘度

    BWT 定义:
      BWT_i = (在域 i 上训练完所有域后的 mAP) - (仅在域 i 上训练后的 mAP)
      负值表示遗忘，正值表示正向迁移。

    灾难性遗忘度定义:
      CF_i = max(0, (仅在域 i 上的 mAP) - (所有域训练后在域 i 上的 mAP))
      即遗忘量，范围 [0, 1]。
    """

    def __init__(
        self,
        model: YOLO,
        domains: List[DomainConfig],
        base_train_kwargs: Dict[str, Any],
        device: str,
        project_dir: Path,
    ) -> None:
        self.model = model
        self.domains = domains
        self.base_train_kwargs = base_train_kwargs
        self.device = device
        self.project_dir = project_dir

        # 初始化 MoLoRAModel 包装器以使用持续学习 API
        self.molora_wrapper = MoLoRAModel(model.model, model.model.molora_config)

        # 域专家分配
        num_experts = getattr(model.model.molora_config, "num_experts", NUM_EXPERTS)
        self.domain_experts = allocate_domain_experts(num_experts, [d.name for d in domains])
        print(f"[CL] Domain expert allocation: {self.domain_experts}")

        # 记录每域训练后的 mAP (在所有域上)
        # map_matrix[i][j] = 在 domain i 训练后，domain j 上的 mAP
        self.map_matrix: Dict[str, Dict[str, float]] = {}
        # map_after_single[i] = 仅在 domain i 上训练后，domain i 上的 mAP
        self.map_after_single: Dict[str, float] = {}

        # Replay buffers
        self.replay_buffers: Dict[str, Dict[str, Any]] = {}

        # 训练历史
        self.training_history: List[Dict[str, Any]] = []

    def _set_domain(self, domain: str) -> None:
        """设置当前活动域。"""
        for m in self.model.model.modules():
            if isinstance(m, MoLoRALayer):
                m.domain_experts = self.domain_experts
                m.set_domain(domain)

    def _clear_domain(self) -> None:
        """清除域限制。"""
        for m in self.model.model.modules():
            if isinstance(m, MoLoRALayer):
                m.clear_domain()

    def _freeze_previous_experts(self, current_domain_idx: int) -> None:
        """冻结之前域的专家，防止遗忘。"""
        if current_domain_idx <= 0:
            return
        frozen = []
        for i in range(current_domain_idx):
            prev_domain = self.domains[i].name
            frozen.extend(self.domain_experts.get(prev_domain, []))
        if frozen:
            self.molora_wrapper.freeze_experts(frozen)
            print(f"[CL] Frozen previous experts: {frozen}")

    def _unfreeze_all_experts(self) -> None:
        """解冻所有专家。"""
        self.molora_wrapper.unfreeze_experts()

    def train_domain(self, domain_idx: int) -> Dict[str, Any]:
        """
        在指定域上训练。

        Args:
            domain_idx: 域索引。

        Returns:
            训练结果字典。
        """
        domain_cfg = self.domains[domain_idx]
        domain_name = domain_cfg.name
        print(f"\n{'=' * 78}")
        print(f"=== CL Stage {domain_idx + 1}/{len(self.domains)}: {domain_name.upper()} {'=' * 50}")
        print(f"{'=' * 78}")
        print(f"Description: {domain_cfg.description}")
        print(f"Augment    : {domain_cfg.augment_kwargs}")

        # 设置当前域
        self._set_domain(domain_name)

        # 冻结之前域的专家
        self._freeze_previous_experts(domain_idx)

        # 构建训练参数
        train_kwargs = {
            **self.base_train_kwargs,
            "name": f"cl_{domain_name}",
            "epochs": domain_cfg.epochs,
        }
        train_kwargs.update(domain_cfg.augment_kwargs)

        t0 = time.time()
        try:
            results = self.model.train(**train_kwargs)
            final_metrics = extract_final_metrics(results)
            ok = True
            error = None
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            final_metrics = {}
            traceback.print_exc()

        elapsed = time.time() - t0

        # 保存 replay buffer
        try:
            buffer = self.molora_wrapper.save_expert_replay_buffer(domain_name)
            self.replay_buffers[domain_name] = buffer
        except Exception as exc:
            print(f"[WARN] Failed to save replay buffer for {domain_name}: {exc}")

        record = {
            "domain": domain_name,
            "domain_idx": domain_idx,
            "ok": ok,
            "error": error,
            "elapsed_sec": round(elapsed, 1),
            "final_metrics": final_metrics,
            "mAP50_95": get_map50_95(final_metrics),
        }
        self.training_history.append(record)

        print(f"[CL] {domain_name} training done: mAP50-95={record['mAP50_95']:.4f}, elapsed={elapsed:.1f}s")
        return record

    def evaluate_all_domains(self, after_domain: str) -> Dict[str, float]:
        """
        在当前模型状态下评估所有域的 mAP。

        Args:
            after_domain: 当前已训练到的域名称 (用于记录)。

        Returns:
            字典: {domain_name: mAP50-95}
        """
        print(f"\n[CL] Evaluating all domains after training on '{after_domain}'...")
        results_map: Dict[str, float] = {}

        for domain_cfg in self.domains:
            domain_name = domain_cfg.name
            try:
                # 切换到目标域
                self._set_domain(domain_name)

                # 运行验证
                val_results = self.model.val(
                    data=DATA_YAML,
                    batch=BATCH,
                    imgsz=IMGSZ,
                    device=self.device,
                    verbose=False,
                    plots=False,
                )
                metrics = extract_final_metrics(val_results)
                map_val = get_map50_95(metrics)
                results_map[domain_name] = map_val
                print(f"  [Eval] {domain_name}: mAP50-95={map_val:.4f}")
            except Exception as exc:
                print(f"  [Eval] {domain_name}: ERROR {exc}")
                results_map[domain_name] = 0.0

        self._clear_domain()
        self.map_matrix[after_domain] = results_map
        return results_map

    def evaluate_single_domain_baseline(self, domain_idx: int) -> float:
        """
        在仅训练单个域后，评估该域的 mAP (用于计算 BWT 基线)。

        注意: 这会创建临时模型副本，不影响当前主模型。
        """
        domain_cfg = self.domains[domain_idx]
        domain_name = domain_cfg.name
        print(f"\n[CL] Computing single-domain baseline for '{domain_name}'...")

        try:
            # 重新加载模型并仅在该域上训练
            temp_model = YOLO(MODEL_PATH)
            temp_cfg = MoLoRAConfig(
                r=LORA_R,
                alpha=LORA_ALPHA,
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                router_type="linear",
                dropout=LORA_DROPOUT,
                use_rslora=True,
                balance_loss_coef=0.01,
                z_loss_coef=0.001,
            )
            get_peft_molora_model(temp_model.model, temp_cfg)

            # 分配域专家
            for m in temp_model.model.modules():
                if isinstance(m, MoLoRALayer):
                    m.domain_experts = self.domain_experts
                    m.set_domain(domain_name)

            train_kwargs = {
                **self.base_train_kwargs,
                "name": f"baseline_{domain_name}",
                "epochs": domain_cfg.epochs,
            }
            train_kwargs.update(domain_cfg.augment_kwargs)

            temp_model.train(**train_kwargs)

            # 评估
            val_results = temp_model.val(
                data=DATA_YAML,
                batch=BATCH,
                imgsz=IMGSZ,
                device=self.device,
                verbose=False,
                plots=False,
            )
            metrics = extract_final_metrics(val_results)
            map_val = get_map50_95(metrics)

            # 清理
            del temp_model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            elif DEVICE == "mps":
                torch.mps.empty_cache()

            print(f"  [Baseline] {domain_name}: mAP50-95={map_val:.4f}")
            self.map_after_single[domain_name] = map_val
            return map_val

        except Exception as exc:
            print(f"  [Baseline] {domain_name}: ERROR {exc}")
            traceback.print_exc()
            self.map_after_single[domain_name] = 0.0
            return 0.0

    def compute_bwt(self) -> Dict[str, float]:
        """
        计算 Backward Transfer (BWT)。

        BWT_i = mAP(所有域训练后, 域 i) - mAP(仅在域 i 训练后, 域 i)
        """
        bwt: Dict[str, float] = {}
        last_domain = self.domains[-1].name

        if last_domain not in self.map_matrix:
            return bwt

        final_map = self.map_matrix[last_domain]

        for domain in self.domains:
            dname = domain.name
            single_map = self.map_after_single.get(dname, 0.0)
            final_d_map = final_map.get(dname, 0.0)
            bwt[dname] = round(final_d_map - single_map, 6)

        return bwt

    def compute_forgetting(self) -> Dict[str, float]:
        """
        计算灾难性遗忘度 (Catastrophic Forgetting)。

        CF_i = max(0, mAP(仅在域 i 训练后, 域 i) - mAP(所有域训练后, 域 i))
        """
        cf: Dict[str, float] = {}
        last_domain = self.domains[-1].name

        if last_domain not in self.map_matrix:
            return cf

        final_map = self.map_matrix[last_domain]

        for domain in self.domains:
            dname = domain.name
            single_map = self.map_after_single.get(dname, 0.0)
            final_d_map = final_map.get(dname, 0.0)
            cf[dname] = round(max(0.0, single_map - final_d_map), 6)

        return cf

    def get_summary(self) -> Dict[str, Any]:
        """返回持续学习汇总。"""
        bwt = self.compute_bwt()
        cf = self.compute_forgetting()

        return {
            "domain_experts": self.domain_experts,
            "map_matrix": self.map_matrix,
            "map_after_single": self.map_after_single,
            "bwt": bwt,
            "bwt_average": round(float(np.mean(list(bwt.values()))), 6) if bwt else 0.0,
            "catastrophic_forgetting": cf,
            "cf_average": round(float(np.mean(list(cf.values()))), 6) if cf else 0.0,
            "training_history": self.training_history,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 可视化
# ═══════════════════════════════════════════════════════════════════════════════


def plot_routing_diagnostics(
    epoch_history: List[Dict[str, Any]],
    save_dir: Path,
) -> List[Path]:
    """
    绘制路由诊断可视化图表。

    图表:
      1. Gini 系数随 epoch 变化曲线
      2. 每层主导专家占比随 epoch 变化曲线
      3. Collapse 事件统计
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    if not epoch_history:
        return written

    epochs = [e["epoch"] for e in epoch_history]
    ginis = [e["overall_gini"] for e in epoch_history]
    collapse_counts = [e["num_collapsed"] for e in epoch_history]

    # ── 图 1: Gini 系数趋势 ──
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, ginis, marker="o", linewidth=2, color="#2E86AB")
    plt.axhline(y=0.25, color="green", linestyle="--", linewidth=1, label="Target Gini (0.25)")
    plt.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="Warning Gini (0.5)")
    plt.fill_between(epochs, ginis, alpha=0.2, color="#2E86AB")
    plt.title("Gini Coefficient per Epoch", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Gini Coefficient", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path1 = save_dir / "gini_per_epoch.png"
    plt.savefig(path1, dpi=160)
    plt.close()
    written.append(path1)

    # ── 图 2: Collapse 事件 ──
    plt.figure(figsize=(10, 5))
    plt.bar(epochs, collapse_counts, color="#E94F37", alpha=0.7, edgecolor="darkred")
    plt.title("Router Collapse Events per Epoch", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Num Collapsed Layers", fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path2 = save_dir / "collapse_per_epoch.png"
    plt.savefig(path2, dpi=160)
    plt.close()
    written.append(path2)

    # ── 图 3: 每层主导专家占比 (所有层取平均) ──
    avg_dominant_shares = []
    for e in epoch_history:
        layer_stats = e.get("layer_stats", [])
        if layer_stats:
            shares = [s["dominant_share"] for s in layer_stats]
            avg_dominant_shares.append(float(np.mean(shares)))
        else:
            avg_dominant_shares.append(0.0)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, avg_dominant_shares, marker="s", linewidth=2, color="#F18F01")
    plt.axhline(y=0.8, color="red", linestyle="--", linewidth=1, label="Collapse Threshold (0.8)")
    plt.fill_between(epochs, avg_dominant_shares, alpha=0.2, color="#F18F01")
    plt.title("Average Dominant Expert Share per Epoch", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Dominant Share", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path3 = save_dir / "dominant_share_per_epoch.png"
    plt.savefig(path3, dpi=160)
    plt.close()
    written.append(path3)

    return written


def plot_continual_learning_metrics(
    map_matrix: Dict[str, Dict[str, float]],
    map_after_single: Dict[str, float],
    bwt: Dict[str, float],
    forgetting: Dict[str, float],
    domains: List[str],
    save_dir: Path,
) -> List[Path]:
    """
    绘制持续学习可视化图表。

    图表:
      1. mAP 热图 (每阶段后在各域上的 mAP)
      2. BWT 柱状图
      3. 灾难性遗忘度柱状图
      4. mAP 折线图 (每域在各训练阶段后的表现)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    domain_list = domains
    n_domains = len(domain_list)

    # ── 图 1: mAP 热图 ──
    if map_matrix:
        matrix = np.zeros((n_domains, n_domains))
        for i, train_domain in enumerate(domain_list):
            row = map_matrix.get(train_domain, {})
            for j, eval_domain in enumerate(domain_list):
                matrix[i, j] = row.get(eval_domain, 0.0)

        plt.figure(figsize=(8, 6))
        im = plt.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(im, label="mAP50-95")
        plt.xticks(range(n_domains), [f"Eval:\n{d}" for d in domain_list], fontsize=10)
        plt.yticks(range(n_domains), [f"After:\n{d}" for d in domain_list], fontsize=10)
        plt.title("mAP Matrix: After Training Domain × Evaluating Domain", fontsize=13, fontweight="bold")

        # 添加数值标注
        for i in range(n_domains):
            for j in range(n_domains):
                plt.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black" if matrix[i, j] > 0.5 else "white",
                    fontsize=10,
                )
        plt.tight_layout()
        path1 = save_dir / "cl_map_heatmap.png"
        plt.savefig(path1, dpi=160)
        plt.close()
        written.append(path1)

    # ── 图 2: BWT 柱状图 ──
    if bwt:
        plt.figure(figsize=(8, 5))
        colors = ["#2E86AB" if v >= 0 else "#E94F37" for v in bwt.values()]
        bars = plt.bar(bwt.keys(), bwt.values(), color=colors, alpha=0.8, edgecolor="black")
        plt.axhline(y=0, color="black", linewidth=1)
        plt.title("Backward Transfer (BWT) per Domain", fontsize=14, fontweight="bold")
        plt.xlabel("Domain", fontsize=12)
        plt.ylabel("BWT (mAP difference)", fontsize=12)
        plt.grid(True, alpha=0.3, axis="y")
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
            )
        plt.tight_layout()
        path2 = save_dir / "cl_bwt.png"
        plt.savefig(path2, dpi=160)
        plt.close()
        written.append(path2)

    # ── 图 3: 灾难性遗忘度 ──
    if forgetting:
        plt.figure(figsize=(8, 5))
        bars = plt.bar(forgetting.keys(), forgetting.values(), color="#F18F01", alpha=0.8, edgecolor="black")
        plt.title("Catastrophic Forgetting per Domain", fontsize=14, fontweight="bold")
        plt.xlabel("Domain", fontsize=12)
        plt.ylabel("Forgetting (mAP drop)", fontsize=12)
        plt.ylim(0, max(max(forgetting.values()) * 1.2, 0.1))
        plt.grid(True, alpha=0.3, axis="y")
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0, height, f"{height:.4f}", ha="center", va="bottom", fontsize=10
            )
        plt.tight_layout()
        path3 = save_dir / "cl_forgetting.png"
        plt.savefig(path3, dpi=160)
        plt.close()
        written.append(path3)

    # ── 图 4: 每域在各阶段后的 mAP 折线 ──
    if map_matrix:
        plt.figure(figsize=(10, 6))
        for j, eval_domain in enumerate(domain_list):
            values = []
            for i, train_domain in enumerate(domain_list):
                row = map_matrix.get(train_domain, {})
                values.append(row.get(eval_domain, 0.0))
            plt.plot(domain_list, values, marker="o", linewidth=2, label=f"Eval: {eval_domain}")

        plt.title("mAP Trajectory Across Training Stages", fontsize=14, fontweight="bold")
        plt.xlabel("Training Stage", fontsize=12)
        plt.ylabel("mAP50-95", fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path4 = save_dir / "cl_map_trajectory.png"
        plt.savefig(path4, dpi=160)
        plt.close()
        written.append(path4)

    return written


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 主流程
# ═══════════════════════════════════════════════════════════════════════════════


def build_domain_configs() -> List[DomainConfig]:
    """
    构建 Day / Night / Fog 三域配置。

    使用 ultralytics 的 augmentation 参数模拟不同域的光照/天气条件:
      - Day: 默认增强 (正常日光条件)
      - Night: 大幅降低亮度 (hsv_v↓), 增加对比度
      - Fog: 降低饱和度与亮度, 模拟雾天低对比度
    """
    return [
        DomainConfig(
            name="day",
            epochs=EPOCHS_PER_DOMAIN,
            augment_kwargs={
                # 默认增强 — 正常日光
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
            },
            description="Normal daylight conditions (default augmentation).",
        ),
        DomainConfig(
            name="night",
            epochs=EPOCHS_PER_DOMAIN,
            augment_kwargs={
                # 夜间 — 大幅降低亮度, 提高饱和度补偿
                "hsv_h": 0.01,
                "hsv_s": 0.9,
                "hsv_v": 0.1,  # 显著降低亮度
            },
            description="Night conditions — low brightness, high contrast.",
        ),
        DomainConfig(
            name="fog",
            epochs=EPOCHS_PER_DOMAIN,
            augment_kwargs={
                # 雾天 — 降低饱和度与亮度, 低对比度
                "hsv_h": 0.005,
                "hsv_s": 0.3,  # 低饱和度
                "hsv_v": 0.2,  # 低亮度
            },
            description="Fog conditions — low saturation, low brightness, low contrast.",
        ),
    ]


def print_header() -> None:
    """打印实验环境信息。"""
    print("\n" + "=" * 78)
    print("  YOLO-Master Routing Diagnostics + Continual Learning Ablation")
    print("=" * 78)
    print(f"  Model      : {MODEL_PATH}")
    print(f"  Dataset    : {DATA_YAML}")
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs/domain: {EPOCHS_PER_DOMAIN}")
    print(f"  Batch      : {BATCH}")
    print(f"  Image size : {IMGSZ}")
    print(f"  Num experts: {NUM_EXPERTS}")
    print(f"  Top-K      : {TOP_K}")
    print(f"  Domains    : {DOMAINS}")
    print(f"  Output dir : {PROJECT_DIR}")
    print(f"  Results    : {RESULTS_JSON}")
    print("=" * 78)
    print(f"\n[Boot] ultralytics: {ultralytics.__file__}")
    print(f"[Boot] version     : {ultralytics.__version__}")
    print(f"[Boot] torch device: {DEVICE}")
    print()


def main() -> None:
    """主入口: 执行 (a) 路由诊断实验 和 (b) 持续学习实验。"""
    print_header()
    PROJECT_DIR.mkdir(exist_ok=True, parents=True)

    overall_results: Dict[str, Any] = {
        "experiment": "routing_diagnostics + continual_learning",
        "model": MODEL_PATH,
        "dataset": DATA_YAML,
        "device": DEVICE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "routing_diag": {},
        "continual_learning": {},
        "plots": [],
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # (a) 路由诊断实验
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("  PART (a): ROUTING DIAGNOSTICS")
    print("=" * 78)

    diag_save_dir = PROJECT_DIR / "routing_diag"
    collector = RoutingDiagnosticsCollector(save_dir=diag_save_dir)

    try:
        # 加载模型
        model = YOLO(MODEL_PATH)
        base_total, base_train = count_params(model.model)
        print(f"[Pre-wrap] total={base_total:,} trainable={base_train:,}")

        # 应用 MoLoRA
        molora_cfg = MoLoRAConfig(
            r=LORA_R,
            alpha=LORA_ALPHA,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            router_type="linear",
            dropout=LORA_DROPOUT,
            use_rslora=True,
            balance_loss_coef=0.01,
            z_loss_coef=0.001,
        )
        get_peft_molora_model(model.model, molora_cfg)

        post_total, post_train = count_params(model.model)
        print(f"[Post-wrap] total={post_total:,} trainable={post_train:,} ({post_train / post_total * 100:.3f}%)")

        # 注册路由诊断回调
        model.add_callback("on_train_epoch_end", collector.on_train_epoch_end)

        # 训练 (Day 域作为路由诊断的训练数据)
        train_kwargs = {
            "data": DATA_YAML,
            "epochs": EPOCHS_PER_DOMAIN,
            "batch": BATCH,
            "imgsz": IMGSZ,
            "device": DEVICE,
            "project": str(PROJECT_DIR),
            "name": "routing_diag_train",
            "exist_ok": True,
            "verbose": False,
            "workers": 2,
            "patience": 0,
            "plots": False,
            "save": False,
        }

        print("[RoutingDiag] Starting training with MoLoRA...")
        t0 = time.time()
        results = model.train(**train_kwargs)
        diag_elapsed = time.time() - t0

        diag_metrics = extract_final_metrics(results)
        print(f"[RoutingDiag] Training done in {diag_elapsed:.1f}s")
        print(f"[RoutingDiag] Final mAP50-95: {get_map50_95(diag_metrics):.4f}")

        # 汇总路由诊断
        diag_summary = collector.get_summary()
        diag_summary["elapsed_sec"] = round(diag_elapsed, 1)
        diag_summary["final_metrics"] = diag_metrics
        overall_results["routing_diag"] = diag_summary

        # 导出路由诊断图表
        plot_paths = collector.export_plots()
        extra_plots = plot_routing_diagnostics(
            collector.epoch_history,
            save_dir=diag_save_dir / "plots",
        )
        plot_paths.extend(extra_plots)
        overall_results["plots"].extend([str(p) for p in plot_paths])
        print(f"[RoutingDiag] Plots saved: {[p.name for p in plot_paths]}")

        # 实时落盘
        RESULTS_JSON.write_text(
            json.dumps(overall_results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    except Exception as exc:
        overall_results["routing_diag"] = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"[ERROR] Routing diagnostics failed: {exc}")
        traceback.print_exc()
        RESULTS_JSON.write_text(
            json.dumps(overall_results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # (b) 持续学习实验
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("  PART (b): CONTINUAL LEARNING (Day → Night → Fog)")
    print("=" * 78)

    try:
        # 重新加载模型以确保干净状态
        cl_model = YOLO(MODEL_PATH)
        cl_cfg = MoLoRAConfig(
            r=LORA_R,
            alpha=LORA_ALPHA,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            router_type="linear",
            dropout=LORA_DROPOUT,
            use_rslora=True,
            balance_loss_coef=0.01,
            z_loss_coef=0.001,
        )
        get_peft_molora_model(cl_model.model, cl_cfg)

        base_train_kwargs = {
            "data": DATA_YAML,
            "batch": BATCH,
            "imgsz": IMGSZ,
            "device": DEVICE,
            "project": str(PROJECT_DIR),
            "exist_ok": True,
            "verbose": False,
            "workers": 2,
            "patience": 0,
            "plots": False,
            "save": False,
        }

        domain_configs = build_domain_configs()
        cl_manager = ContinualLearningManager(
            model=cl_model,
            domains=domain_configs,
            base_train_kwargs=base_train_kwargs,
            device=DEVICE,
            project_dir=PROJECT_DIR,
        )

        # 顺序训练三域
        for idx, domain_cfg in enumerate(domain_configs):
            # 训练当前域
            cl_manager.train_domain(idx)

            # 在当前阶段后评估所有域
            after_domain = domain_cfg.name
            cl_manager.evaluate_all_domains(after_domain)

            # 实时落盘
            overall_results["continual_learning"] = cl_manager.get_summary()
            RESULTS_JSON.write_text(
                json.dumps(overall_results, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        # 计算单域基线 (可选，耗时较长，使用简化版本)
        print("\n[CL] Computing single-domain baselines for BWT...")
        for idx, domain_cfg in enumerate(domain_configs):
            cl_manager.evaluate_single_domain_baseline(idx)
            # 实时更新
            overall_results["continual_learning"] = cl_manager.get_summary()
            RESULTS_JSON.write_text(
                json.dumps(overall_results, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        # 最终汇总
        cl_summary = cl_manager.get_summary()
        overall_results["continual_learning"] = cl_summary

        # 绘制持续学习图表
        cl_plots = plot_continual_learning_metrics(
            map_matrix=cl_summary.get("map_matrix", {}),
            map_after_single=cl_summary.get("map_after_single", {}),
            bwt=cl_summary.get("bwt", {}),
            forgetting=cl_summary.get("catastrophic_forgetting", {}),
            domains=DOMAINS,
            save_dir=PROJECT_DIR / "cl_plots",
        )
        overall_results["plots"].extend([str(p) for p in cl_plots])
        print(f"[CL] Plots saved: {[p.name for p in cl_plots]}")

        # 最终落盘
        RESULTS_JSON.write_text(
            json.dumps(overall_results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        # 控制台汇总
        print("\n" + "=" * 78)
        print("  CONTINUAL LEARNING SUMMARY")
        print("=" * 78)
        print(f"Domain Expert Allocation: {cl_summary['domain_experts']}")
        print("\nmAP Matrix (After Training → Evaluating):")
        for train_d, eval_map in cl_summary.get("map_matrix", {}).items():
            row = "  ".join(f"{k}={v:.4f}" for k, v in eval_map.items())
            print(f"  After {train_d:>5}: {row}")

        print(f"\nBackward Transfer (BWT): {cl_summary['bwt']}")
        print(f"  Average BWT: {cl_summary['bwt_average']:.6f}")
        print(f"\nCatastrophic Forgetting: {cl_summary['catastrophic_forgetting']}")
        print(f"  Average CF : {cl_summary['cf_average']:.6f}")
        print("=" * 78)

    except Exception as exc:
        overall_results["continual_learning"] = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"[ERROR] Continual learning failed: {exc}")
        traceback.print_exc()
        RESULTS_JSON.write_text(
            json.dumps(overall_results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # 最终汇总
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("  EXPERIMENT COMPLETE")
    print("=" * 78)
    print(f"Results JSON : {RESULTS_JSON}")
    print(f"Output dir   : {PROJECT_DIR}")
    print(f"Plots        : {len(overall_results['plots'])} files")
    print("=" * 78)


if __name__ == "__main__":
    main()
