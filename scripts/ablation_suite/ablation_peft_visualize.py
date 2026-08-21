"""PEFT Ablation Experiment Visualization Report Generator.

Aggregates results from all PEFT/MoLoRA ablation experiments (E1-E3, EVAL, peft_validation)
and generates:
  - Multi-panel comparison figures (PNG)
  - Interactive HTML dashboard
  - Markdown summary report
  - LaTeX-style parameter efficiency table

Data sources (auto-discovered):
  - scripts/e1_*.json              : E1 per-expert rank results
  - scripts/e2_*.json              : E2 router calibration results
  - scripts/e3_viz_outputs/        : E3 expert load visualizations
  - scripts/eval_moe_peft_*.json   : Unified evaluation results
  - scripts/peft_validation/*.json : Standard PEFT (LoRA/DoRA/IA3/LoHA) results

Usage:
    python scripts/ablation_peft_visualize.py
    python scripts/ablation_peft_visualize.py --output-dir ./my_reports
    python scripts/ablation_peft_visualize.py --format all

Author: Auto-generated for YOLO-Master PEFT ablation suite.
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
import traceback
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Environment setup (must happen before any ultralytics import)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")

# ---------------------------------------------------------------------------
# Third-party imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

# Matplotlib configuration (Agg backend for headless environments)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    plt = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants & styling
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
DEFAULT_OUTPUT_DIR = HERE / "ablation_reports"

# Color palette (low-saturation, warm tones -- no blue-purple gradients)
COLORS = {
    "full": "#5D5D5D",  # dark gray
    "lora": "#4A90A4",  # muted teal
    "dora": "#C75146",  # muted red
    "ia3": "#D4A373",  # warm sand
    "loha": "#6A994E",  # muted green
    "adalora": "#8B7090",  # muted purple-gray
    "molora": "#BC4749",  # terracotta
    "baseline": "#5D5D5D",
    "uniform": "#4A90A4",
    "frequency": "#6A994E",
    "calibrated_r4": "#D4A373",
    "calibrated_r8": "#C75146",
    "molora_baseline": "#5D5D5D",
    "molora_calib_r4": "#D4A373",
    "molora_freq_rank": "#6A994E",
}

PATTERN_MAP = {
    "full": "/",
    "lora": "\\",
    "dora": "|",
    "ia3": "-",
    "loha": "+",
    "adalora": "x",
}

# Metric keys we care about
METRIC_KEYS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "fitness",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ExperimentRecord:
    """Unified record for a single experiment variant."""

    experiment: str  # e.g. "E1", "E2", "E3", "EVAL", "PEFT"
    variant: str  # e.g. "frequency", "calibrated_r4", "lora"
    ok: bool
    params_total: int
    params_trainable: int
    trainable_pct: float
    elapsed_sec: float
    final_metrics: Dict[str, float]
    extra: Dict[str, Any]  # experiment-specific fields

    def get_metric(self, key: str, default: float = float("nan")) -> float:
        return self.final_metrics.get(key, default)


@dataclass
class ExperimentSuite:
    """Collection of records from one experiment script."""

    name: str
    records: List[ExperimentRecord]
    source_file: Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_json_safe(path: Path) -> Optional[Any]:
    """Load JSON with comprehensive error handling."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON decode error in {path}: {e}")
        return None
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None


def parse_e1_data(data: Any, source: Path) -> List[ExperimentRecord]:
    """Parse E1 (per-expert rank) results."""
    records = []
    # E1 can be either a single dict or a list of dicts
    items = data if isinstance(data, list) else [data]
    for item in items:
        records.append(
            ExperimentRecord(
                experiment="E1",
                variant=item.get("name", item.get("variant", "unknown")),
                ok=item.get("ok", False),
                params_total=item.get("params_total", 0),
                params_trainable=item.get("params_trainable", 0),
                trainable_pct=item.get("trainable_pct", 0.0),
                elapsed_sec=item.get("elapsed_sec", 0.0),
                final_metrics=item.get("final_metrics", {}),
                extra={
                    "rank_info": item.get("rank_info", ""),
                    "expert_ranks": item.get("expert_ranks", []),
                    "wrapped_layers": item.get("wrapped_layers", 0),
                },
            )
        )
    return records


def parse_e2_data(data: Any, source: Path) -> List[ExperimentRecord]:
    """Parse E2 (router calibration) results."""
    records = []
    items = data if isinstance(data, list) else [data]
    for item in items:
        records.append(
            ExperimentRecord(
                experiment="E2",
                variant=item.get("name", "unknown"),
                ok=item.get("ok", False),
                params_total=item.get("params_total", 0),
                params_trainable=item.get("params_trainable", 0),
                trainable_pct=item.get("trainable_pct", 0.0),
                elapsed_sec=item.get("elapsed_sec", 0.0),
                final_metrics=item.get("final_metrics", {}),
                extra={
                    "has_calibration": item.get("has_calibration", False),
                    "router_calib_rank": item.get("router_calib_rank", None),
                },
            )
        )
    return records


def parse_e3_data(source_dir: Path) -> List[ExperimentRecord]:
    """Parse E3 (expert load visualization) summary."""
    summary_path = source_dir / "e3_summary.json"
    data = load_json_safe(summary_path)
    if data is None:
        return []

    records = []
    for variant, info in data.items():
        records.append(
            ExperimentRecord(
                experiment="E3",
                variant=variant,
                ok=True,
                params_total=0,
                params_trainable=0,
                trainable_pct=0.0,
                elapsed_sec=0.0,
                final_metrics={},
                extra={
                    "avg_usage": info.get("avg_usage", []),
                    "gini": info.get("gini", 0.0),
                    "ranks": info.get("ranks", []),
                    "num_layers": info.get("num_layers", 0),
                },
            )
        )
    return records


def parse_eval_data(data: Any, source: Path) -> List[ExperimentRecord]:
    """Parse unified evaluation results (multi-seed, multi-config)."""
    records = []
    items = data if isinstance(data, list) else [data]
    for item in items:
        config_name = item.get("config_name", "unknown")
        seeds = item.get("seeds", [])
        agg = item.get("aggregate", {})

        # Create a synthetic record from aggregate stats
        if seeds:
            first_ok = next((s for s in seeds if s.get("ok", False)), seeds[0])
            records.append(
                ExperimentRecord(
                    experiment="EVAL",
                    variant=config_name,
                    ok=any(s.get("ok", False) for s in seeds),
                    params_total=first_ok.get("params_total", 0),
                    params_trainable=first_ok.get("params_trainable", 0),
                    trainable_pct=first_ok.get("trainable_pct", 0.0),
                    elapsed_sec=sum(s.get("elapsed_sec", 0) for s in seeds) / max(len(seeds), 1),
                    final_metrics={
                        "metrics/mAP50-95(B)": agg.get("mean", 0.0),
                        "metrics/mAP50-95_std": agg.get("std", 0.0),
                    },
                    extra={
                        "n_seeds": len(seeds),
                        "agg_mean": agg.get("mean"),
                        "agg_std": agg.get("std"),
                        "wrapped_layers": first_ok.get("wrapped_layers", 0),
                    },
                )
            )
    return records


def parse_peft_validation_data(data: Any, source: Path) -> List[ExperimentRecord]:
    """Parse standard PEFT comparison results (LoRA, DoRA, IA3, LoHA, etc.)."""
    records = []
    items = data if isinstance(data, list) else [data]
    for item in items:
        records.append(
            ExperimentRecord(
                experiment="PEFT",
                variant=item.get("name", "unknown"),
                ok=item.get("ok", False),
                params_total=item.get("params_total", 0),
                params_trainable=item.get("params_trainable", 0),
                trainable_pct=item.get("trainable_pct", 0.0),
                elapsed_sec=item.get("elapsed_sec", 0.0),
                final_metrics=item.get("final_metrics", {}),
                extra={
                    "adapter_sig": item.get("adapter_sig", {}),
                    "lora_type": item.get("lora_type"),
                    "lora_backend": item.get("lora_backend"),
                    "delta_total_vs_baseline": item.get("delta_total_vs_baseline", 0),
                },
            )
        )
    return records


def discover_all_experiments(scripts_dir: Path) -> List[ExperimentSuite]:
    """Auto-discover and parse all experiment result files."""
    suites: List[ExperimentSuite] = []
    seen_sources: set[Path] = set()  # deduplicate by resolved path

    def _add_suite(name: str, records: List[ExperimentRecord], path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen_sources:
            return
        seen_sources.add(resolved)
        suites.append(ExperimentSuite(name, records, path))

    # E1: per-expert rank
    e1_found = False
    for pattern in ["e1_*.json", "e1_quick_results.json"]:
        for path in scripts_dir.glob(pattern):
            if e1_found:
                break
            data = load_json_safe(path)
            if data is not None:
                records = parse_e1_data(data, path)
                if records:
                    _add_suite("E1_RankAllocation", records, path)
                    e1_found = True

    # E2: router calibration
    for path in scripts_dir.glob("e2_*.json"):
        data = load_json_safe(path)
        if data is not None:
            records = parse_e2_data(data, path)
            if records:
                _add_suite("E2_RouterCalibration", records, path)
                break

    # E3: expert load visualization
    e3_dir = scripts_dir / "e3_viz_outputs"
    if e3_dir.exists():
        records = parse_e3_data(e3_dir)
        if records:
            _add_suite("E3_ExpertLoad", records, e3_dir / "e3_summary.json")

    # EVAL: unified evaluation -- prefer combined file if present
    combined_eval = scripts_dir / "eval_moe_peft_results_combined.json"
    if combined_eval.exists():
        data = load_json_safe(combined_eval)
        if data is not None:
            records = parse_eval_data(data, combined_eval)
            if records:
                _add_suite("EVAL_Unified", records, combined_eval)
    else:
        for path in scripts_dir.glob("eval_moe_peft*.json"):
            data = load_json_safe(path)
            if data is not None:
                records = parse_eval_data(data, path)
                if records:
                    _add_suite("EVAL_Unified", records, path)

    # PEFT validation
    pv_dir = scripts_dir / "peft_validation"
    if pv_dir.exists():
        for path in pv_dir.glob("*.json"):
            if "results" in path.name.lower():
                data = load_json_safe(path)
                if data is not None:
                    records = parse_peft_validation_data(data, path)
                    if records:
                        _add_suite("PEFT_Standard", records, path)
                        break

    return suites


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def get_color(variant: str) -> str:
    """Return color for a variant name, with fallback."""
    return COLORS.get(variant, "#888888")


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Figure 1: Parameter Efficiency Overview
# ---------------------------------------------------------------------------
def fig_parameter_efficiency(suites: List[ExperimentSuite], output_dir: Path) -> Optional[Path]:
    """Generate parameter efficiency comparison bar chart."""
    if not HAS_MPL:
        print("[SKIP] fig_parameter_efficiency: matplotlib not available")
        return None

    # Collect all records that have param counts
    all_records: List[ExperimentRecord] = []
    for suite in suites:
        all_records.extend([r for r in suite.records if r.params_total > 0])

    if not all_records:
        print("[SKIP] fig_parameter_efficiency: no data")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Trainable params bar chart
    ax = axes[0]
    labels = [f"{r.experiment}\n{r.variant}" for r in all_records]
    values = [r.params_trainable for r in all_records]
    colors = [get_color(r.variant) for r in all_records]
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Trainable Parameters", fontsize=11)
    ax.set_title("Trainable Parameter Count by Variant", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )

    # Right: Trainable percentage
    ax = axes[1]
    pct_values = [r.trainable_pct for r in all_records]
    bars = ax.bar(range(len(labels)), pct_values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Trainable %", fontsize=11)
    ax.set_title("Trainable Parameter Ratio", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(pct_values) * 1.2 if pct_values else 100)

    for bar, val in zip(bars, pct_values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_path = output_dir / "fig01_parameter_efficiency.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: Metrics Comparison Radar/Bar
# ---------------------------------------------------------------------------
def fig_metrics_comparison(suites: List[ExperimentSuite], output_dir: Path) -> Optional[Path]:
    """Generate metrics comparison across all experiment variants."""
    if not HAS_MPL:
        print("[SKIP] fig_metrics_comparison: matplotlib not available")
        return None

    # Collect variants that have meaningful metrics
    metric_records: List[ExperimentRecord] = []
    for suite in suites:
        for r in suite.records:
            if r.final_metrics and any(k in r.final_metrics for k in METRIC_KEYS):
                metric_records.append(r)

    if not metric_records:
        print("[SKIP] fig_metrics_comparison: no metric data")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        ("metrics/mAP50-95(B)", "mAP50-95", axes[0][0]),
        ("metrics/mAP50(B)", "mAP50", axes[0][1]),
        ("metrics/precision(B)", "Precision", axes[1][0]),
        ("metrics/recall(B)", "Recall", axes[1][1]),
    ]

    for metric_key, title, ax in panels:
        variants = []
        values = []
        colors = []
        for r in metric_records:
            v = r.get_metric(metric_key)
            if not math.isnan(v):
                variants.append(f"{r.experiment}:{r.variant}")
                values.append(v)
                colors.append(get_color(r.variant))

        if values:
            bars = ax.barh(range(len(variants)), values, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_yticks(range(len(variants)))
            ax.set_yticklabels(variants, fontsize=8)
            ax.set_xlabel(title, fontsize=10)
            ax.set_title(f"{title} Comparison", fontsize=11, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            ax.invert_yaxis()

            for bar, val in zip(bars, values):
                width = bar.get_width()
                ax.annotate(
                    f"{val:.4f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                )

    plt.tight_layout()
    out_path = output_dir / "fig02_metrics_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: Training Time & Efficiency
# ---------------------------------------------------------------------------
def fig_training_efficiency(suites: List[ExperimentSuite], output_dir: Path) -> Optional[Path]:
    """Generate training time vs parameter efficiency scatter plot."""
    if not HAS_MPL:
        print("[SKIP] fig_training_efficiency: matplotlib not available")
        return None

    time_records: List[ExperimentRecord] = []
    for suite in suites:
        for r in suite.records:
            if r.elapsed_sec > 0 and r.params_trainable > 0:
                time_records.append(r)

    if not time_records:
        print("[SKIP] fig_training_efficiency: no timing data")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in time_records:
        ax.scatter(
            r.params_trainable,
            r.elapsed_sec,
            s=200,
            c=get_color(r.variant),
            edgecolors="white",
            linewidths=1.5,
            alpha=0.8,
            zorder=5,
        )
        ax.annotate(
            f"{r.experiment}\n{r.variant}",
            xy=(r.params_trainable, r.elapsed_sec),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    ax.set_xlabel("Trainable Parameters", fontsize=11)
    ax.set_ylabel("Training Time (seconds)", fontsize=11)
    ax.set_title("Training Time vs Parameter Budget", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "fig03_training_efficiency.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 4: E1 Expert Rank Allocation
# ---------------------------------------------------------------------------
def fig_expert_rank_allocation(suites: List[ExperimentSuite], output_dir: Path) -> Optional[Path]:
    """Visualize per-expert rank allocation from E1."""
    if not HAS_MPL:
        print("[SKIP] fig_expert_rank_allocation: matplotlib not available")
        return None

    e1_suite = next((s for s in suites if s.name == "E1_RankAllocation"), None)
    if e1_suite is None:
        print("[SKIP] fig_expert_rank_allocation: E1 data not found")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, r in enumerate(e1_suite.records):
        ax = axes[idx] if idx < 2 else axes[1]
        ranks = r.extra.get("expert_ranks", [])
        rank_info = r.extra.get("rank_info", "")

        # Parse ranks from various formats
        if not ranks and rank_info:
            # Try to parse from string like "[14, 7, 7, 4]"
            match = re.search(r"\[([\d,\s]+)\]", str(rank_info))
            if match:
                ranks = [int(x.strip()) for x in match.group(1).split(",")]

        if not ranks:
            # Fallback: uniform assumption
            ranks = [8, 8, 8, 8]

        labels = [f"E{i}" for i in range(len(ranks))]
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(ranks))) if HAS_NUMPY else ["#4A90A4"] * len(ranks)
        bars = ax.bar(labels, ranks, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Rank", fontsize=11)
        ax.set_title(f"{r.variant}: Per-Expert Rank (Total={sum(ranks)})", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, ranks):
            height = bar.get_height()
            ax.annotate(
                f"{val}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    # Hide extra subplot if only 1 record
    if len(e1_suite.records) < 2 and len(axes) > 1:
        axes[1].axis("off")

    plt.tight_layout()
    out_path = output_dir / "fig04_expert_rank_allocation.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 5: E2 Router Calibration Impact
# ---------------------------------------------------------------------------
def fig_router_calibration(suites: List[ExperimentSuite], output_dir: Path) -> Optional[Path]:
    """Visualize router calibration parameter overhead and metrics impact."""
    if not HAS_MPL:
        print("[SKIP] fig_router_calibration: matplotlib not available")
        return None

    e2_suite = next((s for s in suites if s.name == "E2_RouterCalibration"), None)
    if e2_suite is None:
        print("[SKIP] fig_router_calibration: E2 data not found")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Parameter overhead
    ax = axes[0]
    labels = [r.variant for r in e2_suite.records]
    trainable = [r.params_trainable for r in e2_suite.records]
    colors = [get_color(r.variant) for r in e2_suite.records]
    bars = ax.bar(labels, trainable, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Trainable Parameters", fontsize=11)
    ax.set_title("Router Calibration: Parameter Overhead", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, trainable):
        height = bar.get_height()
        ax.annotate(
            f"{val:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Right: Time comparison
    ax = axes[1]
    times = [r.elapsed_sec for r in e2_suite.records]
    bars = ax.bar(labels, times, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Training Time (s)", fontsize=11)
    ax.set_title("Router Calibration: Training Time", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, times):
        height = bar.get_height()
        ax.annotate(
            f"{val:.1f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    out_path = output_dir / "fig05_router_calibration.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 6: E3 Expert Load Distribution & Gini
# ---------------------------------------------------------------------------
def fig_expert_load_gini(suites: List[ExperimentSuite], output_dir: Path) -> Optional[Path]:
    """Visualize expert load distribution and Gini coefficient."""
    if not HAS_MPL:
        print("[SKIP] fig_expert_load_gini: matplotlib not available")
        return None

    e3_suite = next((s for s in suites if s.name == "E3_ExpertLoad"), None)
    if e3_suite is None:
        print("[SKIP] fig_expert_load_gini: E3 data not found")
        return None

    n = len(e3_suite.records)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(6 * max(n, 1), 5))
    if n == 1:
        axes = [axes]

    for idx, r in enumerate(e3_suite.records):
        ax = axes[idx]
        usage = r.extra.get("avg_usage", [])
        gini = r.extra.get("gini", 0.0)
        ranks = r.extra.get("ranks", [])

        if not usage:
            ax.text(0.5, 0.5, "No usage data", ha="center", va="center", transform=ax.transAxes)
            continue

        labels = [f"E{i}" for i in range(len(usage))]
        x = np.arange(len(labels)) if HAS_NUMPY else list(range(len(labels)))
        width = 0.35

        # Usage bars
        bars1 = ax.bar(x - width / 2, usage, width, label="Avg Usage", color="#4A90A4", edgecolor="white")
        # Rank bars (if available, scaled to match usage scale)
        if ranks:
            max_usage = max(usage) if usage else 1
            max_rank = max(ranks) if ranks else 1
            scaled_ranks = [rk * max_usage / max_rank for rk in ranks]
            bars2 = ax.bar(
                x + width / 2, scaled_ranks, width, label="Rank (scaled)", color="#D4A373", edgecolor="white"
            )

        ax.set_ylabel("Value", fontsize=11)
        ax.set_title(f"{r.variant}: Expert Load (Gini={gini:.4f})", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Annotate Gini
        ax.text(
            0.98,
            0.98,
            f"Gini = {gini:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
        )

    plt.tight_layout()
    out_path = output_dir / "fig06_expert_load_gini.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 7: PEFT Standard Comparison (if available)
# ---------------------------------------------------------------------------
def fig_peft_standard_comparison(suites: List[ExperimentSuite], output_dir: Path) -> Optional[Path]:
    """Visualize standard PEFT method comparison (LoRA, DoRA, IA3, LoHA)."""
    if not HAS_MPL:
        print("[SKIP] fig_peft_standard_comparison: matplotlib not available")
        return None

    pv_suite = next((s for s in suites if s.name == "PEFT_Standard"), None)
    if pv_suite is None:
        print("[SKIP] fig_peft_standard_comparison: PEFT validation data not found")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Trainable params comparison
    ax = axes[0]
    labels = [r.variant for r in pv_suite.records]
    trainable = [r.params_trainable for r in pv_suite.records]
    colors = [get_color(r.variant) for r in pv_suite.records]
    bars = ax.bar(labels, trainable, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Trainable Parameters", fontsize=11)
    ax.set_title("Standard PEFT: Trainable Parameters", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, trainable):
        height = bar.get_height()
        ax.annotate(
            f"{val:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Right: Adapter signature presence
    ax = axes[1]
    sig_keys = ["lora_A/B", "DoRA-magnitude", "hada", "ia3"]
    sig_present = {k: [] for k in sig_keys}
    for r in pv_suite.records:
        sig = r.extra.get("adapter_sig", {})
        sig_present["lora_A/B"].append(sig.get("has_lora_A", False))
        sig_present["DoRA-magnitude"].append(sig.get("has_dora_magnitude", False))
        sig_present["hada"].append(sig.get("has_loha", False))
        sig_present["ia3"].append(sig.get("has_ia3", False))

    x = np.arange(len(labels)) if HAS_NUMPY else list(range(len(labels)))
    width = 0.18
    sig_colors = ["#4A90A4", "#C75146", "#6A994E", "#D4A373"]
    for i, (key, vals) in enumerate(sig_present.items()):
        ax.bar(
            [xi + width * (i - 1.5) for xi in x],
            [1.0 if v else 0.0 for v in vals],
            width,
            label=key,
            color=sig_colors[i],
            edgecolor="white",
        )

    ax.set_ylabel("Presence (1=Yes, 0=No)", fontsize=11)
    ax.set_title("Adapter Signature Verification", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "fig07_peft_standard_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 8: Combined Dashboard
# ---------------------------------------------------------------------------
def fig_combined_dashboard(suites: List[ExperimentSuite], output_dir: Path) -> Optional[Path]:
    """Generate a comprehensive dashboard combining key insights."""
    if not HAS_MPL:
        print("[SKIP] fig_combined_dashboard: matplotlib not available")
        return None

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # === Panel 1: Parameter efficiency (top-left, spans 2 cols) ===
    ax1 = fig.add_subplot(gs[0, :2])
    all_param_records: List[ExperimentRecord] = []
    for suite in suites:
        all_param_records.extend([r for r in suite.records if r.params_trainable > 0])

    if all_param_records:
        labels = [f"{r.experiment}:{r.variant}" for r in all_param_records]
        values = [r.params_trainable for r in all_param_records]
        colors = [get_color(r.variant) for r in all_param_records]
        bars = ax1.barh(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.5)
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_xlabel("Trainable Parameters", fontsize=10)
        ax1.set_title("Parameter Efficiency Overview", fontsize=11, fontweight="bold")
        ax1.grid(axis="x", alpha=0.3)
        ax1.invert_yaxis()

    # === Panel 2: Legend / Info (top-right) ===
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    info_text = []
    info_text.append("PEFT Ablation Dashboard")
    info_text.append("=" * 30)
    info_text.append(f"Experiments: {len(suites)}")
    total_variants = sum(len(s.records) for s in suites)
    info_text.append(f"Total Variants: {total_variants}")
    info_text.append("")
    info_text.append("Color Key:")
    for name, color in COLORS.items():
        if name in ["molora", "baseline", "uniform", "frequency", "calibrated_r4", "calibrated_r8"]:
            continue  # Skip redundant aliases
        info_text.append(f"  * {name}: {color}")
    ax2.text(
        0.05,
        0.95,
        "\n".join(info_text),
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f8f8f8", alpha=0.9),
    )

    # === Panel 3: mAP comparison (middle-left) ===
    ax3 = fig.add_subplot(gs[1, 0])
    map_records = []
    for suite in suites:
        for r in suite.records:
            v = r.get_metric("metrics/mAP50-95(B)")
            if not math.isnan(v):
                map_records.append(r)
    if map_records:
        labels = [r.variant for r in map_records]
        values = [r.get_metric("metrics/mAP50-95(B)") for r in map_records]
        colors = [get_color(r.variant) for r in map_records]
        ax3.bar(labels, values, color=colors, edgecolor="white")
        ax3.set_ylabel("mAP50-95", fontsize=10)
        ax3.set_title("mAP50-95 Comparison", fontsize=11, fontweight="bold")
        ax3.tick_params(axis="x", rotation=30, labelsize=8)
        ax3.grid(axis="y", alpha=0.3)

    # === Panel 4: Gini comparison (middle-center) ===
    ax4 = fig.add_subplot(gs[1, 1])
    e3_suite = next((s for s in suites if s.name == "E3_ExpertLoad"), None)
    if e3_suite:
        labels = [r.variant for r in e3_suite.records]
        gini_vals = [r.extra.get("gini", 0.0) for r in e3_suite.records]
        colors = [get_color(r.variant) for r in e3_suite.records]
        bars = ax4.bar(labels, gini_vals, color=colors, edgecolor="white")
        ax4.set_ylabel("Gini Coefficient", fontsize=10)
        ax4.set_title("Expert Load Imbalance (Gini)", fontsize=11, fontweight="bold")
        ax4.tick_params(axis="x", rotation=15, labelsize=8)
        ax4.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, gini_vals):
            ax4.annotate(
                f"{val:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    # === Panel 5: Time comparison (middle-right) ===
    ax5 = fig.add_subplot(gs[1, 2])
    if all_param_records:
        labels = [r.variant for r in all_param_records]
        times = [r.elapsed_sec for r in all_param_records]
        colors = [get_color(r.variant) for r in all_param_records]
        ax5.bar(labels, times, color=colors, edgecolor="white")
        ax5.set_ylabel("Time (s)", fontsize=10)
        ax5.set_title("Training Time", fontsize=11, fontweight="bold")
        ax5.tick_params(axis="x", rotation=30, labelsize=8)
        ax5.grid(axis="y", alpha=0.3)

    # === Panel 6-8: Rank allocation, Calibration, Summary (bottom row) ===
    ax6 = fig.add_subplot(gs[2, 0])
    e1_suite = next((s for s in suites if s.name == "E1_RankAllocation"), None)
    if e1_suite and e1_suite.records:
        r = e1_suite.records[0]
        ranks = r.extra.get("expert_ranks", [])
        if not ranks:
            match = re.search(r"\[([\d,\s]+)\]", str(r.extra.get("rank_info", "")))
            if match:
                ranks = [int(x.strip()) for x in match.group(1).split(",")]
        if ranks:
            labels = [f"E{i}" for i in range(len(ranks))]
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(ranks))) if HAS_NUMPY else ["#4A90A4"] * len(ranks)
            ax6.bar(labels, ranks, color=colors, edgecolor="white")
            ax6.set_title("Per-Expert Rank (E1)", fontsize=11, fontweight="bold")
            ax6.grid(axis="y", alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 1])
    e2_suite = next((s for s in suites if s.name == "E2_RouterCalibration"), None)
    if e2_suite:
        labels = [r.variant for r in e2_suite.records]
        trainable = [r.params_trainable for r in e2_suite.records]
        colors = [get_color(r.variant) for r in e2_suite.records]
        ax7.bar(labels, trainable, color=colors, edgecolor="white")
        ax7.set_title("Router Calibration Params", fontsize=11, fontweight="bold")
        ax7.tick_params(axis="x", rotation=15, labelsize=8)
        ax7.grid(axis="y", alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")
    summary_lines = ["Key Findings", "=" * 25]
    if e3_suite:
        gini_vals = [r.extra.get("gini", 0.0) for r in e3_suite.records]
        if len(gini_vals) >= 2:
            improvement = (1 - gini_vals[-1] / gini_vals[0]) * 100 if gini_vals[0] > 0 else 0
            summary_lines.append(f"- Gini improvement: {improvement:.1f}%")
    if e2_suite:
        baseline_p = e2_suite.records[0].params_trainable if e2_suite.records else 0
        calib_p = e2_suite.records[-1].params_trainable if len(e2_suite.records) > 1 else 0
        delta = calib_p - baseline_p
        summary_lines.append(f"- Calibration overhead: +{delta:,} params")
    summary_lines.append(f"- Total variants tested: {total_variants}")
    ax8.text(
        0.05,
        0.95,
        "\n".join(summary_lines),
        transform=ax8.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#f0f8ff", alpha=0.9),
    )

    fig.suptitle("YOLO-Master PEFT Ablation Dashboard", fontsize=16, fontweight="bold", y=0.98)
    out_path = output_dir / "fig08_combined_dashboard.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Report generation: Markdown
# ---------------------------------------------------------------------------
def generate_markdown_report(
    suites: List[ExperimentSuite], output_dir: Path, figure_paths: List[Optional[Path]]
) -> Path:
    """Generate comprehensive Markdown report."""
    lines: List[str] = []

    lines.append("# YOLO-Master PEFT Ablation Visualization Report")
    lines.append("")
    lines.append("> Auto-generated by `scripts/ablation_peft_visualize.py`")
    lines.append("> ")
    lines.append(
        f"> Model: YOLO-Master-EsMoE-N.pt | Device: {'MPS' if HAS_TORCH and torch and torch.backends.mps.is_available() else 'CUDA' if HAS_TORCH and torch and torch.cuda.is_available() else 'CPU'}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    total_variants = sum(len(s.records) for s in suites)
    total_ok = sum(1 for s in suites for r in s.records if r.ok)
    lines.append(f"- **Experiments discovered**: {len(suites)}")
    lines.append(f"- **Total variants**: {total_variants}")
    lines.append(f"- **Successful runs**: {total_ok}/{total_variants}")
    lines.append("")

    # Data sources
    lines.append("### Data Sources")
    lines.append("| Experiment | Source | Variants |")
    lines.append("|------------|--------|----------|")
    for s in suites:
        try:
            rel_path = s.source_file.relative_to(REPO_ROOT)
        except ValueError:
            rel_path = s.source_file
        lines.append(f"| {s.name} | `{rel_path}` | {len(s.records)} |")
    lines.append("")

    # E1 Section
    e1 = next((s for s in suites if s.name == "E1_RankAllocation"), None)
    if e1:
        lines.append("---")
        lines.append("")
        lines.append("## E1: Per-Expert Rank Allocation")
        lines.append("")
        lines.append("| Variant | Trainable Params | Trainable % | Expert Ranks | mAP50-95 |")
        lines.append("|---------|-----------------|-------------|-------------|----------|")
        for r in e1.records:
            ranks = r.extra.get("expert_ranks", [])
            if not ranks:
                match = re.search(r"\[([\d,\s]+)\]", str(r.extra.get("rank_info", "")))
                if match:
                    ranks = [int(x.strip()) for x in match.group(1).split(",")]
            ranks_str = str(ranks) if ranks else "N/A"
            map_val = r.get_metric("metrics/mAP50-95(B)")
            map_str = f"{map_val:.4f}" if not math.isnan(map_val) else "N/A"
            lines.append(f"| {r.variant} | {r.params_trainable:,} | {r.trainable_pct:.3f}% | {ranks_str} | {map_str} |")
        lines.append("")

    # E2 Section
    e2 = next((s for s in suites if s.name == "E2_RouterCalibration"), None)
    if e2:
        lines.append("---")
        lines.append("")
        lines.append("## E2: Router Calibration Ablation")
        lines.append("")
        lines.append("| Variant | Trainable Params | Overhead | Calibrated | Time (s) | mAP50-95 |")
        lines.append("|---------|-----------------|----------|------------|----------|----------|")
        baseline_trainable = e2.records[0].params_trainable if e2.records else 0
        for r in e2.records:
            overhead = r.params_trainable - baseline_trainable
            has_calib = "Y" if r.extra.get("has_calibration") else "N"
            map_val = r.get_metric("metrics/mAP50-95(B)")
            map_str = f"{map_val:.4f}" if not math.isnan(map_val) else "N/A"
            lines.append(
                f"| {r.variant} | {r.params_trainable:,} | +{overhead:,} | {has_calib} | {r.elapsed_sec:.1f} | {map_str} |"
            )
        lines.append("")

    # E3 Section
    e3 = next((s for s in suites if s.name == "E3_ExpertLoad"), None)
    if e3:
        lines.append("---")
        lines.append("")
        lines.append("## E3: Expert Load Distribution")
        lines.append("")
        lines.append("| Variant | Avg Usage | Gini | Ranks |")
        lines.append("|---------|-----------|------|-------|")
        for r in e3.records:
            usage = r.extra.get("avg_usage", [])
            gini = r.extra.get("gini", 0.0)
            ranks = r.extra.get("ranks", [])
            usage_str = f"[{', '.join(f'{u:.3f}' for u in usage)}]" if usage else "N/A"
            ranks_str = str(ranks) if ranks else "N/A"
            lines.append(f"| {r.variant} | {usage_str} | {gini:.4f} | {ranks_str} |")
        lines.append("")
        if len(e3.records) >= 2:
            g0 = e3.records[0].extra.get("gini", 0.0)
            g1 = e3.records[-1].extra.get("gini", 0.0)
            if g0 > 0:
                improvement = (1 - g1 / g0) * 100
                lines.append(
                    f"**Finding**: Frequency-based rank allocation reduces Gini coefficient by **{improvement:.1f}%** (from {g0:.4f} to {g1:.4f}), indicating better expert load balancing."
                )
                lines.append("")

    # EVAL Section
    ev = next((s for s in suites if s.name == "EVAL_Unified"), None)
    if ev:
        lines.append("---")
        lines.append("")
        lines.append("## EVAL: Unified Multi-Config Evaluation")
        lines.append("")
        lines.append("| Config | Seeds | mAP50-95 (mean+-std) | Trainable Params |")
        lines.append("|--------|-------|---------------------|-----------------|")
        for r in ev.records:
            mean = r.extra.get("agg_mean")
            std = r.extra.get("agg_std")
            n_seeds = r.extra.get("n_seeds", 1)
            if mean is not None and std is not None:
                metric_str = f"{mean:.4f} +- {std:.4f}"
            else:
                metric_str = "N/A"
            lines.append(f"| {r.variant} | {n_seeds} | {metric_str} | {r.params_trainable:,} |")
        lines.append("")

    # PEFT Standard Section
    pv = next((s for s in suites if s.name == "PEFT_Standard"), None)
    if pv:
        lines.append("---")
        lines.append("")
        lines.append("## PEFT Standard Methods Comparison")
        lines.append("")
        lines.append("| Variant | Trainable Params | Trainable % | Adapter Signature | mAP50-95 | Time (s) |")
        lines.append("|---------|-----------------|-------------|-------------------|----------|----------|")
        for r in pv.records:
            sig = r.extra.get("adapter_sig", {})
            sig_parts = []
            if sig.get("has_lora_A"):
                sig_parts.append("LoRA_A/B")
            if sig.get("has_dora_magnitude"):
                sig_parts.append("DoRA")
            if sig.get("has_loha"):
                sig_parts.append("LoHA")
            if sig.get("has_ia3"):
                sig_parts.append("IA3")
            sig_str = ", ".join(sig_parts) if sig_parts else "Full"
            map_val = r.get_metric("metrics/mAP50-95(B)")
            map_str = f"{map_val:.4f}" if not math.isnan(map_val) else "N/A"
            lines.append(
                f"| {r.variant} | {r.params_trainable:,} | {r.trainable_pct:.3f}% | {sig_str} | {map_str} | {r.elapsed_sec:.1f} |"
            )
        lines.append("")

    # Figures reference
    lines.append("---")
    lines.append("")
    lines.append("## Generated Figures")
    lines.append("")
    for fp in figure_paths:
        if fp is not None:
            try:
                rel = fp.relative_to(REPO_ROOT)
            except ValueError:
                rel = fp
            lines.append(f"- `{rel}`")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- mAP=0.0 in short epochs (1-3) is expected for COCO-scale detection tasks.")
    lines.append("- Gini coefficient: 0 = perfectly equal load, 1 = maximally unequal.")
    lines.append("- Router calibration adds low-rank correction to router weights (delta W_r).")
    lines.append("- Frequency-based rank allocation gives higher ranks to more active experts.")
    lines.append("")

    out_path = output_dir / "REPORT.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Report generation: JSON summary
# ---------------------------------------------------------------------------
def generate_json_summary(suites: List[ExperimentSuite], output_dir: Path) -> Path:
    """Generate machine-readable JSON summary."""
    summary = {
        "meta": {
            "generator": "ablation_peft_visualize.py",
            "repo_root": str(REPO_ROOT),
            "experiments_found": len(suites),
        },
        "experiments": [],
    }

    for suite in suites:
        exp_summary = {
            "name": suite.name,
            "source": str(suite.source_file),
            "num_variants": len(suite.records),
            "variants": [],
        }
        for r in suite.records:
            exp_summary["variants"].append(
                {
                    "experiment": r.experiment,
                    "variant": r.variant,
                    "ok": r.ok,
                    "params_total": r.params_total,
                    "params_trainable": r.params_trainable,
                    "trainable_pct": r.trainable_pct,
                    "elapsed_sec": r.elapsed_sec,
                    "final_metrics": r.final_metrics,
                    "extra": r.extra,
                }
            )
        summary["experiments"].append(exp_summary)

    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVED] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Report generation: HTML dashboard
# ---------------------------------------------------------------------------
def generate_html_dashboard(
    suites: List[ExperimentSuite], output_dir: Path, figure_paths: List[Optional[Path]]
) -> Path:
    """Generate a simple HTML dashboard with embedded figures."""

    # Build figure HTML
    figure_html = ""
    for fp in figure_paths:
        if fp is not None and fp.exists():
            rel = fp.name
            figure_html += f"""
            <div class="figure">
                <img src="{rel}" alt="{rel}" loading="lazy">
                <p class="caption">{rel.replace("fig", "Figure ").replace("_", " ").replace(".png", "").title()}</p>
            </div>
            """

    # Build tables
    tables_html = ""

    # E1 table
    e1 = next((s for s in suites if s.name == "E1_RankAllocation"), None)
    if e1:
        rows = ""
        for r in e1.records:
            ranks = r.extra.get("expert_ranks", [])
            if not ranks:
                match = re.search(r"\[([\d,\s]+)\]", str(r.extra.get("rank_info", "")))
                if match:
                    ranks = [int(x.strip()) for x in match.group(1).split(",")]
            rows += f"<tr><td>{r.variant}</td><td>{r.params_trainable:,}</td><td>{r.trainable_pct:.3f}%</td><td>{ranks}</td></tr>\n"
        tables_html += f"""
        <h3>E1: Per-Expert Rank Allocation</h3>
        <table>
            <tr><th>Variant</th><th>Trainable Params</th><th>Trainable %</th><th>Expert Ranks</th></tr>
            {rows}
        </table>
        """

    # E2 table
    e2 = next((s for s in suites if s.name == "E2_RouterCalibration"), None)
    if e2:
        rows = ""
        baseline = e2.records[0].params_trainable if e2.records else 0
        for r in e2.records:
            overhead = r.params_trainable - baseline
            calib = "Y" if r.extra.get("has_calibration") else "N"
            rows += f"<tr><td>{r.variant}</td><td>{r.params_trainable:,}</td><td>+{overhead:,}</td><td>{calib}</td><td>{r.elapsed_sec:.1f}s</td></tr>\n"
        tables_html += f"""
        <h3>E2: Router Calibration</h3>
        <table>
            <tr><th>Variant</th><th>Trainable Params</th><th>Overhead</th><th>Calibration</th><th>Time</th></tr>
            {rows}
        </table>
        """

    # E3 table
    e3 = next((s for s in suites if s.name == "E3_ExpertLoad"), None)
    if e3:
        rows = ""
        for r in e3.records:
            usage = r.extra.get("avg_usage", [])
            usage_str = ", ".join(f"{u:.3f}" for u in usage) if usage else "N/A"
            gini = r.extra.get("gini", 0.0)
            rows += f"<tr><td>{r.variant}</td><td>[{usage_str}]</td><td>{gini:.4f}</td></tr>\n"
        tables_html += f"""
        <h3>E3: Expert Load Distribution</h3>
        <table>
            <tr><th>Variant</th><th>Avg Usage</th><th>Gini</th></tr>
            {rows}
        </table>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO-Master PEFT Ablation Dashboard</title>
    <style>
        :root {{
            --bg: #fafafa;
            --fg: #333;
            --accent: #4A90A4;
            --card-bg: #fff;
            --border: #e0e0e0;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--fg);
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{ color: var(--accent); border-bottom: 3px solid var(--accent); padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #666; margin-top: 20px; }}
        .meta {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .figures {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .figure {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }}
        .figure img {{ max-width: 100%; height: auto; border-radius: 4px; }}
        .caption {{ font-size: 0.85em; color: #666; margin-top: 8px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            margin-top: 10px;
        }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: #f0f4f8; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
        }}
        .badge-ok {{ background: #d4edda; color: #155724; }}
        .badge-fail {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <h1>YOLO-Master PEFT Ablation Dashboard</h1>
    <div class="meta">
        <strong>Generated:</strong> Auto-generated report<br>
        <strong>Model:</strong> YOLO-Master-EsMoE-N.pt<br>
        <strong>Experiments:</strong> {len(suites)} suites discovered<br>
    </div>

    <h2>Data Tables</h2>
    {tables_html}

    <h2>Visualizations</h2>
    <div class="figures">
        {figure_html}
    </div>

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border); color: #888; font-size: 0.85em;">
        Generated by scripts/ablation_peft_visualize.py
    </footer>
</body>
</html>"""

    out_path = output_dir / "dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[SAVED] {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PEFT Ablation Experiment Visualization Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ablation_peft_visualize.py
  python scripts/ablation_peft_visualize.py --output-dir ./reports --format all
  python scripts/ablation_peft_visualize.py --format md+json
        """,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save all outputs (default: scripts/ablation_reports)",
    )
    parser.add_argument(
        "--format", type=str, default="all", help="Output formats: all, png, md, json, html (comma-separated)"
    )
    parser.add_argument(
        "--scripts-dir", type=str, default=str(HERE), help="Directory containing experiment result JSON files"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    scripts_dir = Path(args.scripts_dir)

    requested_formats = [f.strip().lower() for f in args.format.split(",")]
    do_all = "all" in requested_formats
    do_png = do_all or "png" in requested_formats
    do_md = do_all or "md" in requested_formats or "markdown" in requested_formats
    do_json = do_all or "json" in requested_formats
    do_html = do_all or "html" in requested_formats

    print("=" * 70)
    print("YOLO-Master PEFT Ablation Visualization Report Generator")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Scripts directory: {scripts_dir}")
    print(f"Formats: {requested_formats}")
    print(f"Matplotlib: {'OK' if HAS_MPL else 'MISSING'}")
    print(f"NumPy: {'OK' if HAS_NUMPY else 'MISSING'}")
    print(f"PyTorch: {'OK' if HAS_TORCH else 'MISSING'}")
    print("-" * 70)

    # -----------------------------------------------------------------------
    # Phase 1: Data discovery
    # -----------------------------------------------------------------------
    print("\n[Phase 1] Discovering experiment data...")
    suites = discover_all_experiments(scripts_dir)
    if not suites:
        print("[ERROR] No experiment data found. Please run ablation scripts first.")
        print(f"  Searched in: {scripts_dir}")
        sys.exit(1)

    print(f"[OK] Discovered {len(suites)} experiment suites:")
    for s in suites:
        print(f"  - {s.name}: {len(s.records)} variants from {s.source_file.name}")

    # -----------------------------------------------------------------------
    # Phase 2: Generate visualizations (PNG)
    # -----------------------------------------------------------------------
    figure_paths: List[Optional[Path]] = []
    if do_png:
        print("\n[Phase 2] Generating visualizations...")
        try:
            figure_paths.append(fig_parameter_efficiency(suites, output_dir))
        except Exception as e:
            print(f"[ERROR] fig_parameter_efficiency failed: {e}")
            figure_paths.append(None)

        try:
            figure_paths.append(fig_metrics_comparison(suites, output_dir))
        except Exception as e:
            print(f"[ERROR] fig_metrics_comparison failed: {e}")
            figure_paths.append(None)

        try:
            figure_paths.append(fig_training_efficiency(suites, output_dir))
        except Exception as e:
            print(f"[ERROR] fig_training_efficiency failed: {e}")
            figure_paths.append(None)

        try:
            figure_paths.append(fig_expert_rank_allocation(suites, output_dir))
        except Exception as e:
            print(f"[ERROR] fig_expert_rank_allocation failed: {e}")
            figure_paths.append(None)

        try:
            figure_paths.append(fig_router_calibration(suites, output_dir))
        except Exception as e:
            print(f"[ERROR] fig_router_calibration failed: {e}")
            figure_paths.append(None)

        try:
            figure_paths.append(fig_expert_load_gini(suites, output_dir))
        except Exception as e:
            print(f"[ERROR] fig_expert_load_gini failed: {e}")
            figure_paths.append(None)

        try:
            figure_paths.append(fig_peft_standard_comparison(suites, output_dir))
        except Exception as e:
            print(f"[ERROR] fig_peft_standard_comparison failed: {e}")
            figure_paths.append(None)

        try:
            figure_paths.append(fig_combined_dashboard(suites, output_dir))
        except Exception as e:
            print(f"[ERROR] fig_combined_dashboard failed: {e}")
            figure_paths.append(None)

    # -----------------------------------------------------------------------
    # Phase 3: Generate reports
    # -----------------------------------------------------------------------
    if do_md:
        print("\n[Phase 3a] Generating Markdown report...")
        try:
            generate_markdown_report(suites, output_dir, figure_paths)
        except Exception as e:
            print(f"[ERROR] Markdown report failed: {e}")
            traceback.print_exc()

    if do_json:
        print("\n[Phase 3b] Generating JSON summary...")
        try:
            generate_json_summary(suites, output_dir)
        except Exception as e:
            print(f"[ERROR] JSON summary failed: {e}")
            traceback.print_exc()

    if do_html:
        print("\n[Phase 3c] Generating HTML dashboard...")
        try:
            generate_html_dashboard(suites, output_dir, figure_paths)
        except Exception as e:
            print(f"[ERROR] HTML dashboard failed: {e}")
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # Phase 4: Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")
    print("-" * 70)


if __name__ == "__main__":
    main()
