"""Run a reproducible real-COCO F15 baseline/Foundation effect-gate matrix.

This runner owns experiment orchestration and result provenance only. It does
not change the model or trainer implementation, and it never turns a missing
validation result into an accuracy claim. Each Foundation run is paired with a
baseline run using the same model config, seed, data split, and initialization
contract.

Example dry-run::

    python scripts/foundation_f15_real_effect_gate.py \
        --dataset /Users/gatilin/MyWork/datasets/coco2017/unified_multitask_f15/coco2017_mot_multitask.yaml \
        --teacher-model /path/to/dinov3 \
        --seeds 20260813,20260814,20260815 \
        --foundation-loss-weights 0.01,0.05 \
        --epochs 10 --fraction 0.1 --imgsz 256 --batch 2 \
        --dry-run

The same command without ``--dry-run`` starts the paired training runs. Use
``--no-val`` only for a training-signal diagnostic; the effect gate requires
validation evidence from the same COCO split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Make direct ``python scripts/...`` invocation resolve this checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL = str(REPO_ROOT / "ultralytics/cfg/models/26/yolo26-master-mt-n.yaml")
DEFAULT_DATA = "/Users/gatilin/MyWork/datasets/coco2017/unified_multitask_f15/coco2017_mot_multitask.yaml"
DEFAULT_TEACHER = os.environ.get("YOLO_MASTER_DINOV3_LOCAL", "Tooony133/dinov3-vits16-pretrain-lvd1689m")
TASKS = ["detect", "segment", "pose"]
SCHEMA_VERSION = 1


def _csv_ints(value: str) -> list[int]:
    """Parse a non-empty comma-separated list of non-negative integer seeds."""
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must contain integers") from exc
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("seeds must be a non-empty list of non-negative integers")
    return values


def _csv_floats(value: str) -> list[float]:
    """Parse a non-empty comma-separated list of non-negative loss weights."""
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("foundation-loss-weights must contain numbers") from exc
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("foundation-loss-weights must be a non-empty list of non-negative numbers")
    return values


def _clean_row(row: dict[str | None, str | None]) -> dict[str, str]:
    """Normalize the padded column names emitted by Ultralytics results.csv."""
    return {str(key).strip(): str(value).strip() for key, value in row.items() if key is not None}


def read_last_results(results_csv: Path) -> dict[str, str]:
    """Read the last non-empty results.csv row, returning an empty dict when absent."""
    if not results_csv.is_file():
        return {}
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = [
            _clean_row(row) for row in csv.DictReader(handle) if any((value or "").strip() for value in row.values())
        ]
    return rows[-1] if rows else {}


def _numeric_fields(row: dict[str, str], prefixes: tuple[str, ...]) -> dict[str, float]:
    """Extract finite numeric fields with one of the requested prefixes."""
    values: dict[str, float] = {}
    for key, value in row.items():
        if not key.startswith(prefixes):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and abs(number) != float("inf"):
            values[key] = number
    return values


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """Summarize one Ultralytics run without inventing missing metrics."""
    row = read_last_results(run_dir / "results.csv")
    checkpoints = [run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"]
    checkpoint = next((path for path in checkpoints if path.is_file()), None)
    return {
        "run_dir": str(run_dir),
        "results_csv": str(run_dir / "results.csv"),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "last_epoch": row.get("epoch"),
        "train_metrics": _numeric_fields(row, ("train/",)),
        "validation_metrics": _numeric_fields(row, ("metrics/", "val/")),
        "results_available": bool(row),
    }


def build_run_plan(
    *,
    dataset: str,
    model: str,
    teacher_model: str,
    project: str,
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
    """Build deterministic baseline/Foundation pairs in weight-then-seed order."""
    plan: list[dict[str, Any]] = []
    for weight in foundation_loss_weights:
        for seed in seeds:
            pair_name = f"s{seed}-w{weight:g}"
            common = {
                "dataset": dataset,
                "model": model,
                "teacher_model": teacher_model,
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
                "tasks": TASKS,
                "initialization_contract": {
                    "pretrained": False,
                    "same_model_config": True,
                    "same_seed": True,
                    "same_dataset_split": True,
                },
            }
            for foundation in (False, True):
                name = f"{'foundation' if foundation else 'baseline'}-{pair_name}"
                plan.append(
                    {
                        **common,
                        "foundation": foundation,
                        "name": name,
                        "overrides": {
                            "model": model,
                            "data": dataset,
                            "task": "multitask",
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
                            "foundation_multitask": foundation,
                            "foundation_multitask_tasks": TASKS,
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
                        },
                    }
                )
    return plan


def _train_one(spec: dict[str, Any]) -> dict[str, Any]:
    """Train one planned run and return its measured result summary."""
    import torch

    from ultralytics import YOLO

    # Seed model construction explicitly so paired branches share initialization.
    torch.manual_seed(int(spec["seed"]))
    started = time.perf_counter()
    model = YOLO(spec["model"])
    model.train(**spec["overrides"])
    run_dir = Path(str(getattr(getattr(model, "trainer", None), "save_dir", Path(spec["project"]) / spec["name"])))
    summary = summarize_run(run_dir)
    summary.update(
        {
            "elapsed_s": round(time.perf_counter() - started, 4),
            "foundation": spec["foundation"],
            "foundation_loss_weight": spec["foundation_loss_weight"],
            "seed": spec["seed"],
            "name": spec["name"],
        }
    )
    return summary


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    """Persist a JSON report with stable UTF-8 formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate available validation metrics without imputing missing accuracy."""
    grouped: dict[tuple[int, float], dict[bool, dict[str, Any]]] = {}
    for record in records:
        key = (int(record["seed"]), float(record["foundation_loss_weight"]))
        grouped.setdefault(key, {})[bool(record["foundation"])] = record
    pairs = []
    for (seed, weight), branches in sorted(grouped.items()):
        baseline, foundation = branches.get(False), branches.get(True)
        baseline_metrics = (baseline or {}).get("validation_metrics", {})
        foundation_metrics = (foundation or {}).get("validation_metrics", {})
        deltas = {
            key: round(float(foundation_metrics[key]) - float(baseline_metrics[key]), 8)
            for key in sorted(set(baseline_metrics) & set(foundation_metrics))
        }
        pairs.append(
            {
                "seed": seed,
                "foundation_loss_weight": weight,
                "baseline_complete": baseline is not None,
                "foundation_complete": foundation is not None,
                "validation_metric_deltas": deltas,
            }
        )
    return {
        "paired_runs": len(pairs),
        "validation_pairs_with_metrics": sum(bool(pair["validation_metric_deltas"]) for pair in pairs),
        "pairs": pairs,
    }


def run_matrix(
    plan: list[dict[str, Any]],
    output: Path,
    *,
    runner: Callable[[dict[str, Any]], dict[str, Any]],
    resume: bool = False,
) -> dict[str, Any]:
    """Execute a plan and write progress after every run for interruption safety."""
    records: list[dict[str, Any]] = []
    if resume and output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("benchmark") != "f15_real_coco_effect_gate":
            raise ValueError(f"Cannot resume incompatible report: {output}")
        records = list(previous.get("records") or [])
    completed_names = {record.get("name") for record in records}
    for spec in plan:
        if spec["name"] in completed_names:
            continue
        result = runner(spec)
        records.append(result)
        completed_names.add(spec["name"])
        _write_report(
            output,
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark": "f15_real_coco_effect_gate",
                "real_data": True,
                "accuracy_claim": False,
                "tasks": TASKS,
                "plan": plan,
                "records": records,
                "completed_runs": len(records),
                "total_runs": len(plan),
                "summary": _aggregate_records(records),
            },
        )
    return json.loads(output.read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse effect-gate runner arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATA, help="Unified COCO multi-task YAML.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Multi-task student model YAML.")
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER, help="Local or HF DINOv3 model path/id.")
    parser.add_argument("--project", default="runs/multitask/f15-effect-gate", help="Ultralytics project directory.")
    parser.add_argument("--seeds", type=_csv_ints, default=[20260813, 20260814, 20260815])
    parser.add_argument("--foundation-loss-weights", type=_csv_floats, default=[0.01, 0.05])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("reports/foundation/v0.1/f15-real-coco-effect-gate.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and do not train.")
    parser.add_argument("--resume", action="store_true", help="Skip completed run names found in --output.")
    parser.add_argument("--val", dest="val", action="store_true", default=True, help="Run validation (default).")
    parser.add_argument("--no-val", dest="val", action="store_false", help="Disable validation for diagnostics only.")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")
    if args.imgsz < 32 or args.batch < 1 or args.workers < 0:
        parser.error("--imgsz must be >= 32, --batch must be positive, and --workers must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    """Build and optionally execute the real-data F15 effect-gate matrix."""
    args = parse_args(argv)
    plan = build_run_plan(
        dataset=str(Path(args.dataset).expanduser().resolve()),
        model=str(Path(args.model).expanduser().resolve()),
        teacher_model=str(Path(args.teacher_model).expanduser().resolve())
        if Path(args.teacher_model).expanduser().exists()
        else args.teacher_model,
        project=str(Path(args.project).expanduser().resolve()),
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
        print(json.dumps({"benchmark": "f15_real_coco_effect_gate", "dry_run": True, "plan": plan}, indent=2))
        return
    result = run_matrix(plan, args.output, runner=_train_one, resume=args.resume)
    print(json.dumps({"completed_runs": result["completed_runs"], "total_runs": result["total_runs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
