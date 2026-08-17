"""Run a reproducible F15 seed/KD-weight mechanism matrix.

The matrix is deliberately honest about its scope: it uses the paired synthetic
training-signal benchmark and never reports AP or other real-data accuracy.
Use a real COCO trainer experiment for the release effect gate.

Example::

    python scripts/foundation_f15_effect_matrix.py \
        --teacher-model /path/to/dinov3 \
        --seeds 20260813,20260814,20260815 \
        --foundation-loss-weights 0.01,0.05,0.1 \
        --steps 6 \
        --output reports/foundation/v0.1/f15-effect-matrix.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev

# Make direct ``python scripts/...`` invocation resolve this checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.foundation_f15_paired_benchmark import (  # noqa: E402
    DEFAULT_TEACHER,
    run_benchmark,
)


def _csv_ints(value: str) -> list[int]:
    """Parse a comma-separated list of non-negative integer seeds."""
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("seeds must be a non-empty comma-separated list of non-negative integers")
    return values


def _csv_floats(value: str) -> list[float]:
    """Parse a comma-separated list of non-negative Foundation loss weights."""
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("foundation-loss-weights must contain numbers") from exc
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "foundation-loss-weights must be a non-empty comma-separated list of non-negative numbers"
        )
    return values


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate paired runs by Foundation weight without implying accuracy."""
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        key = str(record["foundation_loss_weight"])
        groups.setdefault(key, []).append(record)
    summaries = []
    for weight, group in groups.items():
        deltas = [float(item["summary"]["foundation_task_delta_first_last"]) for item in group]
        baseline_deltas = [float(item["summary"]["baseline_task_delta_first_last"]) for item in group]
        overheads = [float(item["summary"]["foundation_step_overhead_ratio"]) for item in group]
        summaries.append(
            {
                "foundation_loss_weight": float(weight),
                "runs": len(group),
                "foundation_task_delta_mean": round(mean(deltas), 6),
                "foundation_task_delta_std": round(stdev(deltas), 6) if len(deltas) > 1 else 0.0,
                "baseline_task_delta_mean": round(mean(baseline_deltas), 6),
                "baseline_task_delta_std": round(stdev(baseline_deltas), 6) if len(baseline_deltas) > 1 else 0.0,
                "foundation_overhead_mean": round(mean(overheads), 6),
                "all_mechanism_gates": all(
                    bool(item["summary"]["foundation_supervised_task_gate"])
                    and bool(item["summary"]["foundation_nonzero_kd_gate"])
                    and float(item["summary"]["foundation_p4_grad_min"]) > 0
                    for item in group
                ),
            }
        )
    return {"weights": summaries}


def run_matrix(
    teacher_model: str,
    seeds: list[int],
    foundation_loss_weights: list[float],
    steps: int,
    align_dim: int,
) -> dict[str, object]:
    """Run all seed/weight combinations and return a JSON-safe matrix record."""
    records = []
    for foundation_loss_weight in foundation_loss_weights:
        for seed in seeds:
            records.append(
                run_benchmark(
                    teacher_model,
                    steps,
                    align_dim,
                    seed=seed,
                    foundation_loss_weight=foundation_loss_weight,
                )
            )
    return {
        "schema_version": 1,
        "benchmark": "f15_effect_matrix",
        "teacher_model": teacher_model,
        "student_model": records[0]["student_model"] if records else None,
        "tasks": records[0]["tasks"] if records else ["detect", "segment", "pose"],
        "steps": steps,
        "align_dim": align_dim,
        "seeds": seeds,
        "foundation_loss_weights": foundation_loss_weights,
        "synthetic_batch": True,
        "real_accuracy_claim": False,
        "records": records,
        "summary": _summary(records),
    }


def main() -> None:
    """Parse matrix options and persist the mechanism-only report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", default=os.environ.get("YOLO_MASTER_DINOV3_LOCAL", DEFAULT_TEACHER))
    parser.add_argument("--seeds", type=_csv_ints, default=[20260813, 20260814, 20260815])
    parser.add_argument("--foundation-loss-weights", type=_csv_floats, default=[0.01, 0.05, 0.1])
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--align-dim", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("reports/foundation/v0.1/f15-effect-matrix.json"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.align_dim < 1:
        parser.error("--align-dim must be positive")
    result = run_matrix(args.teacher_model, args.seeds, args.foundation_loss_weights, args.steps, args.align_dim)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
