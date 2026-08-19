"""
MoE Enhanced Diagnostic Visualization (P1-2)

Generates a self-contained HTML dashboard with routing heatmaps, expert
utilization distributions, and alert markers — complementing the existing
matplotlib plots in history.py.

Usage:
    from ultralytics.nn.modules.moe.viz import generate_moe_dashboard
    generate_moe_dashboard(model, save_path="moe_dashboard.html")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .api import collect_all_moe_info
from .diagnostics import collect_moe_diagnostics, format_moe_diagnostics
from .utils import is_core_moe_block


def _expert_usage_bar_html(info_list: dict[str, Any]) -> str:
    """Generate HTML for expert utilization bar charts."""
    cards = []
    for name, info in info_list.items():
        usage = info.expert_usage
        if not usage:
            continue
        max_val = max(usage) if usage else 1.0
        bars = ""
        for idx, val in enumerate(usage):
            height_pct = (val / max_val * 100) if max_val > 0 else 0
            color = "#e74c3c" if val < 0.01 else ("#f39c12" if val < 0.1 else "#2ecc71")
            bars += f'<div class="bar-item"><span class="bar-label">E{idx}</span><div class="bar-track"><div class="bar-fill" style="height:{height_pct:.1f}%;background:{color}"></div></div><span class="bar-val">{val:.3f}</span></div>'
        cards.append(f'''
        <div class="card">
            <h3>{name} <span class="badge">{info.class_name}</span></h3>
            <div class="bar-chart">{bars}</div>
        </div>''')
    return "\n".join(cards) if cards else "<p>No expert usage data available.</p>"


def _routing_heatmap_html(model: nn.Module) -> str:
    """Generate routing weight heatmap HTML from live model state."""
    layers = []
    for name, m in model.named_modules():
        if not is_core_moe_block(m):
            continue
        snapshot = getattr(m, "last_routing_snapshot", None)
        if not isinstance(snapshot, dict):
            continue
        weights = snapshot.get("expert_weights") or snapshot.get("routing_weights") or snapshot.get("gate_weights")
        if weights is None or not torch.is_tensor(weights):
            continue
        w = weights.detach().cpu().float()
        # Average across batch dimension if present
        if w.dim() > 1:
            w = w.mean(dim=0)
        w_list = w.tolist()
        if not w_list:
            continue

        max_w = max(w_list) if w_list else 1.0
        cells = ""
        for idx, val in enumerate(w_list):
            intensity = min(val / max_w, 1.0) if max_w > 0 else 0
            bg = f"rgba(52, 152, 219, {intensity:.3f})"
            cells += f'<div class="heat-cell" style="background:{bg}" title="E{idx}: {val:.4f}">E{idx}<br><span>{val:.3f}</span></div>'
        layers.append(f'''
        <div class="card">
            <h3>{name} <span class="badge">{type(m).__name__}</span></h3>
            <div class="heatmap-grid">{cells}</div>
        </div>''')
    return "\n".join(layers) if layers else "<p>No routing snapshots available (run a forward pass first).</p>"


def _alert_summary_html(diagnostics: list) -> str:
    """Generate alert summary from diagnostics."""
    alerts = []
    for diag in diagnostics:
        if diag.collapse_flag:
            alerts.append(f'<div class="alert alert-collapse">⚠️ {diag.name}: routing collapse — expert E{diag.dominant_expert} dominates ({diag.dominant_share:.1%})</div>')
        for idx, usage in enumerate(diag.usage):
            if usage < 0.01:
                alerts.append(f'<div class="alert alert-dead">💀 {diag.name}: expert E{idx} is dead (usage={usage:.4f})</div>')
    if not alerts:
        return '<div class="alert alert-ok">✅ No routing issues detected.</div>'
    return "\n".join(alerts)


def _summary_stats_html(info_list: dict[str, Any]) -> str:
    """Generate summary statistics cards."""
    total_layers = len(info_list)
    total_experts = sum(i.num_experts for i in info_list.values())
    total_aux = sum(i.aux_loss_value for i in info_list.values())
    collapsed = sum(1 for i in info_list.values() if any(u < 0.01 for u in i.expert_usage))

    return f"""
    <div class="stats-row">
        <div class="stat-card"><div class="stat-val">{total_layers}</div><div class="stat-label">MoE Layers</div></div>
        <div class="stat-card"><div class="stat-val">{total_experts}</div><div class="stat-label">Total Experts</div></div>
        <div class="stat-card"><div class="stat-val">{total_aux:.4f}</div><div class="stat-label">Total Aux Loss</div></div>
        <div class="stat-card"><div class="stat-val">{collapsed}</div><div class="stat-label">Layers w/ Dead Experts</div></div>
    </div>"""


_DASHBOARD_CSS = """
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: #f5f6fa; color: #2c3e50; padding: 20px; }
  h1 { font-size: 24px; margin-bottom: 8px; }
  h2 { font-size: 18px; margin: 24px 0 12px; border-bottom: 2px solid #3498db; padding-bottom: 6px; }
  .timestamp { color: #7f8c8d; font-size: 13px; margin-bottom: 20px; }
  .stats-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
  .stat-card { background: #fff; border-radius: 8px; padding: 16px 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; min-width: 140px; }
  .stat-val { font-size: 28px; font-weight: 700; color: #2c3e50; }
  .stat-label { font-size: 12px; color: #95a5a6; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }
  .card { background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  .card h3 { font-size: 14px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
  .badge { background: #ecf0f1; padding: 2px 8px; border-radius: 4px; font-size: 11px; color: #7f8c8d; font-family: monospace; }
  .bar-chart { display: flex; align-items: flex-end; gap: 8px; height: 120px; padding: 8px 0; }
  .bar-item { display: flex; flex-direction: column; align-items: center; flex: 1; height: 100%; justify-content: flex-end; }
  .bar-label { font-size: 11px; color: #7f8c8d; margin-bottom: 2px; }
  .bar-track { width: 100%; max-width: 40px; height: 100%; background: #f0f0f0; border-radius: 4px 4px 0 0; position: relative; display: flex; align-items: flex-end; }
  .bar-fill { width: 100%; border-radius: 4px 4px 0 0; transition: height 0.3s; min-height: 2px; }
  .bar-val { font-size: 10px; color: #95a5a6; margin-top: 2px; }
  .heatmap-grid { display: flex; gap: 4px; flex-wrap: wrap; }
  .heat-cell { width: 60px; height: 60px; display: flex; flex-direction: column; align-items: center; justify-content: center; border-radius: 6px; font-size: 11px; font-weight: 600; color: #2c3e50; border: 1px solid rgba(0,0,0,0.05); }
  .heat-cell span { font-size: 9px; color: rgba(0,0,0,0.6); margin-top: 2px; }
  .alert { padding: 8px 14px; border-radius: 6px; margin-bottom: 8px; font-size: 13px; }
  .alert-collapse { background: #ffeaa7; border-left: 4px solid #e17055; }
  .alert-dead { background: #fab1a0; border-left: 4px solid #d63031; }
  .alert-ok { background: #dcedc8; border-left: 4px solid #7cb342; }
  .text-summary { background: #1e272e; color: #d2dae2; border-radius: 8px; padding: 16px; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; line-height: 1.6; white-space: pre-wrap; overflow-x: auto; margin-bottom: 16px; }
</style>
"""


def generate_moe_dashboard(
    model: nn.Module,
    save_path: str | Path = "moe_dashboard.html",
    title: str = "MoE Diagnostic Dashboard",
) -> Path:
    """Generate a self-contained HTML diagnostic dashboard for a MoE model.

    Args:
        model: PyTorch model with MoE layers (run a forward pass first for live data).
        save_path: Output HTML file path.
        title: Dashboard title.

    Returns:
        Path to the generated HTML file.
    """
    save_path = Path(save_path)

    # Collect data
    info_list = collect_all_moe_info(model)
    diagnostics = collect_moe_diagnostics(model)
    text_summary = format_moe_diagnostics(diagnostics)

    # Build sections
    stats_html = _summary_stats_html(info_list)
    alert_html = _alert_summary_html(diagnostics)
    usage_bars = _expert_usage_bar_html(info_list)
    heatmaps = _routing_heatmap_html(model)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
{_DASHBOARD_CSS}
</head>
<body>
<h1>🔬 {title}</h1>
<div class="timestamp">Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</div>

<h2>Summary</h2>
{stats_html}

<h2>Alerts</h2>
{alert_html}

<h2>Expert Utilization</h2>
{usage_bars}

<h2>Routing Weights Heatmap</h2>
{heatmaps}

<h2>Text Summary</h2>
<div class="text-summary">{text_summary}</div>
</body>
</html>"""

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html, encoding="utf-8")
    return save_path
