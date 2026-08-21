#!/usr/bin/env python3
"""Run the Foundation x Mixture 2x2 interaction matrix.

The matrix pairs Foundation-disabled and Foundation-enabled training for both
the dense YOLO26 student and the routed YOLO26-Master student.  It is an
engineering/evidence runner: it records observed validation and telemetry
values, computes paired deltas, and never turns a small smoke run into an
accuracy claim.

Examples::

    python scripts/foundation_mixture_interaction.py --dry-run
    python scripts/foundation_mixture_interaction.py --epochs 3 --imgsz 256 \
        --device mps --seeds 0,1 --foundation-loss-weights 0.05,0.1

The default student models are constructed from YAML with ``pretrained=False``
so each architecture has a deterministic, architecture-local baseline.  Use a
separate benchmark when comparing pretrained checkpoints across architectures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 2
ARCHITECTURES = {
    "dense": str(REPO_ROOT / "ultralytics/cfg/models/26/yolo26.yaml"),
    "routed": str(REPO_ROOT / "ultralytics/cfg/models/26/yolo26-master-n.yaml"),
}
FOUNDATION_TEACHER_DEFAULT = "Tooony133/dinov3-vits16-pretrain-lvd1689m"
OBSERVED_METRICS = (
    "metrics/mAP50-95(B)",
    "metrics/mAP50(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP_small(B)",
    "metrics/mAP_medium(B)",
    "metrics/mAP_large(B)",
    "train/foundation_loss",
    "train/foundation_cosine_raw",
    "train/foundation_relational_raw",
    "train/foundation_effective_weight",
    "train/foundation_task_ratio",
    "train/mixture_aux_loss",
)


def _csv_ints(value: str) -> list[int]:
    """Parse a non-empty comma-separated list of non-negative integer seeds."""
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must contain integers") from exc
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("seeds must be a non-empty list of non-negative integers")
    return values


def _csv_floats(value: str) -> list[float]:
    """Parse a non-empty list of non-negative Foundation loss weights."""
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("foundation-loss-weights must contain numbers") from exc
    if not values or any(item < 0 or not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("foundation-loss-weights must be finite and non-negative")
    return values


def _clean_row(row: dict[str | None, str | None]) -> dict[str, str]:
    """Normalize padded result-column names emitted by Ultralytics."""
    return {str(key).strip(): str(value).strip() for key, value in row.items() if key is not None}


def _read_results(path: Path) -> list[dict[str, str]]:
    """Read non-empty rows from a results CSV, tolerating an absent file."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            _clean_row(row) for row in csv.DictReader(handle) if any((value or "").strip() for value in row.values())
        ]


def _finite_float(value: Any) -> float | None:
    """Convert a value to a finite float, returning None for missing telemetry."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """Summarize the final observed row without imputing missing values."""
    rows = _read_results(run_dir / "results.csv")
    row = rows[-1] if rows else {}
    checkpoint_candidates = (run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt")
    checkpoint = next((path for path in checkpoint_candidates if path.is_file()), None)
    observed = {key: value for key in OBSERVED_METRICS if (value := _finite_float(row.get(key))) is not None}
    if (train_elapsed := _finite_float(row.get("time"))) is not None:
        observed["train/elapsed_s"] = train_elapsed
    return {
        "run_dir": str(run_dir),
        "results_csv": str(run_dir / "results.csv"),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "last_epoch": _finite_float(row.get("epoch")),
        "train_elapsed_s": _finite_float(row.get("time")),
        "observed": observed,
        "results_available": bool(row),
    }


def build_run_plan(
    *,
    dataset: str,
    project: str,
    teacher_model: str,
    architectures: dict[str, str] | None = None,
    seeds: list[int],
    foundation_loss_weights: list[float],
    epochs: int,
    fraction: float,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    val: bool,
) -> list[dict[str, Any]]:
    """Build deterministic paired runs for the four cells of the 2x2 matrix."""
    models = dict(architectures or ARCHITECTURES)
    if set(models) != set(ARCHITECTURES):
        raise ValueError(f"architectures must contain exactly {tuple(ARCHITECTURES)}")
    if models["dense"] == models["routed"]:
        raise ValueError("dense and routed model configurations must be distinct")
    plan: list[dict[str, Any]] = []
    for architecture, model in models.items():
        for weight in foundation_loss_weights:
            for seed in seeds:
                pair_name = f"{architecture}-s{seed}-w{weight:g}"
                common = {
                    "architecture": architecture,
                    "model": model,
                    "dataset": dataset,
                    "project": project,
                    "seed": seed,
                    "foundation_loss_weight": weight,
                    "epochs": epochs,
                    "fraction": fraction,
                    "imgsz": imgsz,
                    "batch": batch,
                    "device": device,
                    "workers": workers,
                    "val": val,
                    "initialization_contract": {
                        "pretrained": False,
                        "same_model_config": True,
                        "same_seed": True,
                        "same_dataset_split": True,
                        "same_augmentation_config": True,
                    },
                }
                for foundation in (False, True):
                    name = f"{'foundation' if foundation else 'baseline'}-{pair_name}"
                    overrides = {
                        "model": model,
                        "data": dataset,
                        "task": "detect",
                        "mode": "train",
                        "epochs": epochs,
                        "fraction": fraction,
                        "imgsz": imgsz,
                        "batch": batch,
                        "device": device,
                        "workers": workers,
                        "val": val,
                        "seed": seed,
                        "deterministic": True,
                        "pretrained": False,
                        "project": project,
                        "name": name,
                        "exist_ok": True,
                        "save": True,
                        "plots": False,
                        "foundation_enabled": foundation,
                        "foundation_teacher": "dinov3" if foundation else "none",
                        "foundation_model": teacher_model if foundation else None,
                        "foundation_backend": "transformers",
                        "foundation_teacher_dtype": "fp32",
                        "foundation_teacher_device": device,
                        "foundation_target_levels": ["p4"],
                        "foundation_loss": "hybrid",
                        "foundation_cosine_weight": 1.0,
                        "foundation_relation_weight": 1.0,
                        "foundation_relation_mode": "sampled",
                        "foundation_relation_samples": 16,
                        "foundation_loss_weight": weight if foundation else 0.0,
                    }
                    plan.append(
                        {
                            **common,
                            "foundation": foundation,
                            "name": name,
                            "pair_name": pair_name,
                            "overrides": overrides,
                        }
                    )
    return plan


def _train_one(spec: dict[str, Any]) -> dict[str, Any]:
    """Train one planned run and return its measured summary."""
    import torch
    from ultralytics import YOLO

    torch.manual_seed(int(spec["seed"]))
    started = time.perf_counter()
    model = YOLO(spec["model"])
    model.train(**spec["overrides"])
    fallback = Path(spec["project"]) / spec["name"]
    run_dir = Path(str(getattr(getattr(model, "trainer", None), "save_dir", fallback)))
    summary = summarize_run(run_dir)
    summary.update(
        {
            "architecture": spec["architecture"],
            "foundation": spec["foundation"],
            "foundation_loss_weight": spec["foundation_loss_weight"],
            "seed": spec["seed"],
            "name": spec["name"],
            "elapsed_s": round(time.perf_counter() - started, 4),
        }
    )
    return summary


def _aggregate_observed_fields(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float | int | None]]:
    """Summarize available numeric fields without imputing missing observations."""
    values_by_key: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.get(field, {}).items():
            if (number := _finite_float(value)) is not None:
                values_by_key.setdefault(key, []).append(number)
    return {
        key: {
            "n": len(values),
            "mean": round(statistics.fmean(values), 8),
            "sample_std": round(statistics.stdev(values), 8) if len(values) > 1 else None,
        }
        for key, values in sorted(values_by_key.items())
    }


def _paired_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate within-architecture Foundation deltas and their interaction."""
    grouped: dict[tuple[str, int, float], dict[bool, dict[str, Any]]] = {}
    for record in records:
        key = (str(record["architecture"]), int(record["seed"]), float(record["foundation_loss_weight"]))
        grouped.setdefault(key, {})[bool(record["foundation"])] = record

    pairs = []
    for (architecture, seed, weight), branches in sorted(grouped.items()):
        baseline, foundation = branches.get(False), branches.get(True)
        base_observed = (baseline or {}).get("observed", {})
        foundation_observed = (foundation or {}).get("observed", {})
        deltas = {
            key: round(foundation_observed[key] - base_observed[key], 8)
            for key in sorted(set(base_observed) & set(foundation_observed))
            if _finite_float(base_observed[key]) is not None and _finite_float(foundation_observed[key]) is not None
        }
        pair = {
            "architecture": architecture,
            "seed": seed,
            "foundation_loss_weight": weight,
            "baseline_complete": baseline is not None,
            "foundation_complete": foundation is not None,
            "observed_deltas": deltas,
        }
        pairs.append(pair)

    interactions = []
    observed_keys = sorted({key for pair in pairs for key in pair["observed_deltas"]})
    for weight in sorted({float(pair["foundation_loss_weight"]) for pair in pairs}):
        for seed in sorted({int(pair["seed"]) for pair in pairs}):
            cells = {
                pair["architecture"]: pair
                for pair in pairs
                if pair["seed"] == seed and pair["foundation_loss_weight"] == weight
            }
            dense, routed = cells.get("dense"), cells.get("routed")
            interaction = {
                key: round(routed["observed_deltas"][key] - dense["observed_deltas"][key], 8)
                for key in observed_keys
                if dense and routed and key in dense["observed_deltas"] and key in routed["observed_deltas"]
            }
            complete = bool(
                dense
                and routed
                and dense["baseline_complete"]
                and dense["foundation_complete"]
                and routed["baseline_complete"]
                and routed["foundation_complete"]
            )
            interactions.append(
                {
                    "seed": seed,
                    "foundation_loss_weight": weight,
                    "interaction_definition": "(routed foundation - routed baseline) - (dense foundation - dense baseline)",
                    "observed_metric_interactions": interaction,
                    "complete": complete,
                }
            )
    delta_aggregates = []
    for architecture in sorted({str(pair["architecture"]) for pair in pairs}):
        for weight in sorted({float(pair["foundation_loss_weight"]) for pair in pairs}):
            complete_pairs = [
                pair
                for pair in pairs
                if pair["architecture"] == architecture
                and pair["foundation_loss_weight"] == weight
                and pair["baseline_complete"]
                and pair["foundation_complete"]
            ]
            delta_aggregates.append(
                {
                    "architecture": architecture,
                    "foundation_loss_weight": weight,
                    "complete_pairs": len(complete_pairs),
                    "observed_delta_summary": _aggregate_observed_fields(complete_pairs, "observed_deltas"),
                }
            )
    interaction_aggregates = []
    for weight in sorted({float(interaction["foundation_loss_weight"]) for interaction in interactions}):
        complete_interactions = [
            interaction
            for interaction in interactions
            if interaction["foundation_loss_weight"] == weight and interaction["complete"]
        ]
        interaction_aggregates.append(
            {
                "foundation_loss_weight": weight,
                "complete_interactions": len(complete_interactions),
                "observed_interaction_summary": _aggregate_observed_fields(
                    complete_interactions, "observed_metric_interactions"
                ),
            }
        )
    return {
        "paired_runs": len(pairs),
        "pairs": pairs,
        "interactions": interactions,
        "foundation_delta_aggregates": delta_aggregates,
        "interaction_aggregates": interaction_aggregates,
        "accuracy_claim": False,
        "interpretation": "Observed smoke deltas only; interaction is not an accuracy claim or statistical result.",
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable UTF-8 JSON report, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _plan_specs_by_name(plan: list[dict[str, Any]]) -> dict[str, str]:
    """Return canonical serialized specs for validating resumable plans."""
    return {str(spec["name"]): json.dumps(spec, sort_keys=True, separators=(",", ":")) for spec in plan}


def _migrate_record_timing(record: dict[str, Any]) -> dict[str, Any]:
    """Rename the pre-v2 cumulative-time field without changing its observed value."""
    migrated = dict(record)
    if "epoch_time_s" in migrated:
        migrated["train_elapsed_s"] = migrated.pop("epoch_time_s")
    observed = dict(migrated.get("observed") or {})
    if "train/epoch_time_s" in observed:
        observed["train/elapsed_s"] = observed.pop("train/epoch_time_s")
    migrated["observed"] = observed
    return migrated


def _report_payload(plan: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the complete report payload from the currently observed records."""
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "foundation_mixture_interaction",
        "real_data": True,
        "accuracy_claim": False,
        "matrix": "foundation_enabled x mixture_architecture",
        "plan": plan,
        "records": records,
        "completed_runs": len(records),
        "total_runs": len(plan),
        "summary": _paired_summary(records),
    }


def run_matrix(
    plan: list[dict[str, Any]],
    output: Path,
    *,
    runner: Callable[[dict[str, Any]], dict[str, Any]],
    resume: bool = False,
) -> dict[str, Any]:
    """Execute a plan with per-run checkpoints in the report for safe resume."""
    records: list[dict[str, Any]] = []
    if resume and output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("benchmark") != "foundation_mixture_interaction":
            raise ValueError(f"Cannot resume incompatible report: {output}")
        records = [_migrate_record_timing(record) for record in list(previous.get("records") or [])]
        previous_plan = list(previous.get("plan") or [])
        current_specs = _plan_specs_by_name(plan)
        previous_specs = _plan_specs_by_name(previous_plan)
        missing_names = sorted(set(previous_specs) - set(current_specs))
        changed_names = sorted(
            name for name in set(previous_specs) & set(current_specs) if previous_specs[name] != current_specs[name]
        )
        if missing_names or changed_names:
            details = []
            if missing_names:
                details.append(f"missing previous runs: {', '.join(missing_names[:3])}")
            if changed_names:
                details.append(f"changed run specs: {', '.join(changed_names[:3])}")
            raise ValueError("Cannot resume with a changed or truncated plan (" + "; ".join(details) + ")")
        unknown_records = sorted({str(record.get("name")) for record in records} - set(current_specs))
        if unknown_records:
            raise ValueError(f"Cannot resume report with unknown run records: {', '.join(unknown_records[:3])}")
    completed_names = {record.get("name") for record in records}
    for spec in plan:
        if spec["name"] in completed_names:
            continue
        records.append(runner(spec))
        completed_names.add(spec["name"])
        _write_report(output, _report_payload(plan, records))
    if records:
        _write_report(output, _report_payload(plan, records))
    return json.loads(output.read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse runner arguments and enforce matrix boundaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(REPO_ROOT / "ultralytics/cfg/datasets/coco128.yaml"))
    parser.add_argument("--project", default="runs/foundation/mixture-interaction")
    parser.add_argument("--teacher-model", default=FOUNDATION_TEACHER_DEFAULT)
    parser.add_argument("--seeds", type=_csv_ints, default=[0])
    parser.add_argument("--foundation-loss-weights", type=_csv_floats, default=[0.1])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/foundation/v0.1/foundation-mixture-interaction.json")
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-val", dest="val", action="store_false", help="Disable validation for signal-only diagnostics."
    )
    parser.set_defaults(val=True)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")
    if args.imgsz < 32 or args.batch < 1 or args.workers < 0:
        parser.error("--imgsz must be >= 32, --batch positive, and --workers non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    """Build and optionally execute the Foundation x Mixture matrix."""
    args = parse_args(argv)
    dataset = str(Path(args.dataset).expanduser().resolve())
    project = str(Path(args.project).expanduser().resolve())
    teacher = (
        str(Path(args.teacher_model).expanduser().resolve())
        if Path(args.teacher_model).expanduser().exists()
        else args.teacher_model
    )
    plan = build_run_plan(
        dataset=dataset,
        project=project,
        teacher_model=teacher,
        seeds=args.seeds,
        foundation_loss_weights=args.foundation_loss_weights,
        epochs=args.epochs,
        fraction=args.fraction,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        val=args.val,
    )
    if args.dry_run:
        print(json.dumps({"benchmark": "foundation_mixture_interaction", "dry_run": True, "plan": plan}, indent=2))
        return
    result = run_matrix(plan, args.output, runner=_train_one, resume=args.resume)
    print(json.dumps({"completed_runs": result["completed_runs"], "total_runs": result["total_runs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
