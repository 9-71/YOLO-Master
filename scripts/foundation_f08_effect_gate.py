#!/usr/bin/env python3
"""Run the finite MPS F08 Foundation effect gate.

The gate is intentionally small and reproducible: a deterministic 1% COCO
train split, a fixed 500-image validation split, ten epochs, and three seeds.
It compares one routed student in four arms:

* ``B0``: Foundation disabled;
* ``D0``: cosine KD;
* ``D1``: sampled relational KD;
* ``D2``: hybrid KD.

This is an engineering/effect gate, not a paper benchmark.  Every completed
run is persisted in the report, missing values are left missing, and no
accuracy claim is inferred from a small finite matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1
BENCHMARK = "foundation_f08_mps_effect_gate"
DEFAULT_DATASET_ROOT = Path(os.environ.get("YOLO_MASTER_COCO_ROOT", "/Users/gatilin/MyWork/datasets/coco2017"))
DEFAULT_MODEL = str(REPO_ROOT / "ultralytics/cfg/models/26/yolo26-master-n.yaml")
DEFAULT_TEACHER = os.environ.get("YOLO_MASTER_DINOV3_LOCAL", "Tooony133/dinov3-vits16-pretrain-lvd1689m")
ARMS: dict[str, dict[str, Any]] = {
    "B0": {"foundation_enabled": False, "foundation_loss": None, "description": "routed student, Foundation disabled"},
    "D0": {"foundation_enabled": True, "foundation_loss": "cosine", "description": "cosine Foundation KD"},
    "D1": {
        "foundation_enabled": True,
        "foundation_loss": "relational",
        "description": "sampled relational Foundation KD",
    },
    "D2": {"foundation_enabled": True, "foundation_loss": "hybrid", "description": "hybrid Foundation KD"},
}
REQUIRED_VALIDATION_METRICS = (
    "metrics/mAP50-95(B)",
    "metrics/mAP50(B)",
    "metrics/mAP_small(B)",
    "metrics/mAP_medium(B)",
    "metrics/mAP_large(B)",
)
OBSERVED_TRAIN_METRICS = (
    "train/foundation_loss",
    "train/foundation_cosine_raw",
    "train/foundation_relational_raw",
    "train/foundation_effective_weight",
    "train/foundation_task_ratio",
    "train/mixture_aux_loss",
)
FORBIDDEN_TEACHER_KEY_FRAGMENTS = ("teacher_manager", "_route_teachers", "teacher_model", "dinov3", "siglip")


def _csv_ints(value: str) -> list[int]:
    """Parse a non-empty comma-separated list of non-negative integer seeds."""
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must contain integers") from exc
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("seeds must be a non-empty list of non-negative integers")
    return values


def _finite_float(value: Any) -> float | None:
    """Return a finite float or ``None`` for absent/non-numeric values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clean_row(row: dict[str | None, str | None]) -> dict[str, str]:
    """Normalize padded results.csv headers and values."""
    return {str(key).strip(): str(value).strip() for key, value in row.items() if key is not None}


def read_last_results(path: Path) -> dict[str, str]:
    """Read the final non-empty results.csv row without imputing metrics."""
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [_clean_row(row) for row in csv.DictReader(handle) if any((v or "").strip() for v in row.values())]
    return rows[-1] if rows else {}


def _numeric_fields(row: dict[str, str], keys: tuple[str, ...]) -> dict[str, float]:
    """Extract finite values for the requested exact result columns."""
    values = {}
    for key in keys:
        if (number := _finite_float(row.get(key))) is not None:
            values[key] = number
    return values


def _source_dataset_yaml(dataset_root: Path) -> Path:
    """Resolve the local COCO YAML used as the metadata source."""
    candidates = (dataset_root / "coco2017.yaml", REPO_ROOT / "scripts" / "coco2017.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No COCO2017 dataset YAML found in {dataset_root} or scripts/")


def prepare_fixed_subsets(
    dataset_root: Path,
    output_dir: Path,
    *,
    train_size: int = 1183,
    val_size: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Create/reuse deterministic train and validation image-list YAMLs."""
    if train_size <= 0 or val_size <= 0:
        raise ValueError("train_size and val_size must be positive")
    dataset_root = dataset_root.expanduser().resolve()
    train_images = sorted((dataset_root / "images" / "train2017").glob("*.jpg"))
    val_images = sorted((dataset_root / "images" / "val2017").glob("*.jpg"))
    if len(train_images) < train_size or len(val_images) < val_size:
        raise ValueError(
            f"COCO split is too small: train={len(train_images)} (need {train_size}), "
            f"val={len(val_images)} (need {val_size})"
        )
    source = yaml.safe_load(_source_dataset_yaml(dataset_root).read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    selected_train = sorted(rng.sample(train_images, train_size))
    selected_val = sorted(rng.sample(val_images, val_size))
    train_list = output_dir / f"coco2017_train{train_size}_seed{seed}.txt"
    val_list = output_dir / f"coco2017_val{val_size}_seed{seed}.txt"
    train_yaml = output_dir / f"coco2017_train{train_size}_val{val_size}_seed{seed}.yaml"
    train_list.write_text("\n".join(str(path) for path in selected_train) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(str(path) for path in selected_val) + "\n", encoding="utf-8")
    data = dict(source)
    data["path"] = str(dataset_root)
    data["train"] = str(train_list)
    data["val"] = str(val_list)
    data["subset"] = {
        "benchmark": BENCHMARK,
        "train_split": "train2017",
        "train_size": train_size,
        "val_split": "val2017",
        "val_size": val_size,
        "seed": seed,
    }
    train_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {
        "dataset_root": str(dataset_root),
        "train_list": str(train_list),
        "val_list": str(val_list),
        "dataset_yaml": str(train_yaml),
        "train_size": train_size,
        "val_size": val_size,
        "seed": seed,
    }


def build_run_plan(
    *,
    dataset: str,
    model: str = DEFAULT_MODEL,
    teacher_model: str = DEFAULT_TEACHER,
    project: str,
    seeds: list[int],
    epochs: int = 10,
    fraction: float = 1.0,
    imgsz: int = 640,
    batch: int = 2,
    device: str = "mps",
    workers: int = 0,
    foundation_loss_weight: float = 0.1,
    val: bool = True,
) -> list[dict[str, Any]]:
    """Build deterministic B0/D0/D1/D2 runs in arm-then-seed order."""
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be a non-empty list of non-negative integers")
    if epochs < 1 or not 0 < fraction <= 1 or imgsz < 32 or batch < 1 or workers < 0:
        raise ValueError("invalid epochs/fraction/imgsz/batch/workers")
    if foundation_loss_weight < 0 or not math.isfinite(foundation_loss_weight):
        raise ValueError("foundation_loss_weight must be finite and non-negative")
    plan = []
    for arm, arm_config in ARMS.items():
        for seed in seeds:
            name = f"{arm.lower()}-s{seed}"
            enabled = bool(arm_config["foundation_enabled"])
            loss = arm_config["foundation_loss"]
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
                "optimizer": "SGD",
                "lr0": 0.01,
                "amp": False,
                "project": project,
                "name": name,
                "exist_ok": True,
                "save": True,
                "save_period": 1,
                "plots": False,
                "foundation_enabled": enabled,
                "foundation_teacher": "dinov3" if enabled else "none",
                "foundation_model": teacher_model if enabled else None,
                "foundation_backend": "transformers",
                "foundation_teacher_dtype": "fp32",
                "foundation_teacher_device": device,
                "foundation_target_levels": ["p4"],
                "foundation_align_dim": 32,
                "foundation_loss": loss or "relational",
                "foundation_cosine_weight": 1.0,
                "foundation_relation_weight": 1.0,
                "foundation_relation_mode": "sampled",
                "foundation_relation_samples": 16,
                "foundation_loss_weight": foundation_loss_weight if enabled else 0.0,
                "foundation_weight_schedule": "gate_decay",
            }
            plan.append(
                {
                    "arm": arm,
                    "description": arm_config["description"],
                    "name": name,
                    "seed": seed,
                    "model": model,
                    "dataset": dataset,
                    "project": project,
                    "teacher_model": teacher_model,
                    "foundation": enabled,
                    "foundation_loss": loss,
                    "foundation_loss_weight": overrides["foundation_loss_weight"],
                    "initialization_contract": {
                        "pretrained": False,
                        "model_source": "routed YAML yolo26-master-n.yaml; no routed pretrained checkpoint found",
                        "same_model_config": True,
                        "same_seed": True,
                        "same_dataset_split": True,
                        "same_optimizer": True,
                    },
                    "teacher_contract": {
                        "training_only": enabled,
                        "teacher_family": "dinov3" if enabled else None,
                        "local_files_only": enabled,
                        "hf_hub_offline": True,
                    },
                    "overrides": overrides,
                }
            )
    return plan


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object when absent."""
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def summarize_telemetry(run_dir: Path) -> dict[str, Any]:
    """Keep telemetry measurement semantics while avoiding full rank payloads in the report."""
    path = run_dir / "telemetry.json"
    payload = _read_json(path)
    aggregate = payload.get("aggregation", {})
    ranks = payload.get("ranks", [])
    rank0 = next((record for record in ranks if int(record.get("metadata", {}).get("rank", 0)) == 0), {})
    return {
        "path": str(path),
        "available": bool(payload),
        "aggregation": aggregate,
        "steps": rank0.get("steps", {}),
        "memory": rank0.get("memory", {}),
        "memory_measurement": rank0.get("memory", {}).get("measurement"),
        "memory_is_true_peak": rank0.get("memory", {}).get("is_true_peak"),
    }


def student_only_contract(checkpoint: Path, *, imgsz: int, device: str, latency_runs: int = 10) -> dict[str, Any]:
    """Inspect a checkpoint after stripping Foundation and measure student-only cost."""
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "available": checkpoint.is_file(),
        "teacher_state_keys": [],
        "teacher_state_clean": None,
        "params": None,
        "gflops": None,
        "pytorch_latency_ms": None,
        "latency_measurement": "synchronized inference wall time",
    }
    if not checkpoint.is_file():
        return result
    try:
        import torch
        from ultralytics.nn.foundation_distill_model import strip_foundation_distillation_model
        from ultralytics.nn.tasks import load_checkpoint
        from ultralytics.utils.torch_utils import get_flops, get_num_params

        loaded, _ = load_checkpoint(str(checkpoint), device="cpu")
        student = strip_foundation_distillation_model(loaded).eval()
        keys = [str(key) for key in student.state_dict()]
        banned = sorted(
            {key for key in keys if any(fragment in key.lower() for fragment in FORBIDDEN_TEACHER_KEY_FRAGMENTS)}
        )
        result["teacher_state_keys"] = banned
        result["teacher_state_clean"] = not banned
        result["params"] = int(get_num_params(student))
        result["gflops"] = float(get_flops(student, imgsz=imgsz))
        target = torch.device(device)
        student = student.to(target)
        image = torch.zeros((1, 3, imgsz, imgsz), device=target)
        synchronize = getattr(getattr(torch, "mps", None), "synchronize", None) if target.type == "mps" else None
        if target.type == "cuda":

            def synchronize():
                torch.cuda.synchronize(target)

        with torch.inference_mode():
            for _ in range(2):
                student(image)
            if callable(synchronize):
                synchronize()
            started = time.perf_counter()
            for _ in range(max(1, latency_runs)):
                student(image)
            if callable(synchronize):
                synchronize()
        result["pytorch_latency_ms"] = (time.perf_counter() - started) * 1000.0 / max(1, latency_runs)
    except Exception as exc:  # Keep the training report usable if optional profiling fails.
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def summarize_run(run_dir: Path, *, imgsz: int, device: str) -> dict[str, Any]:
    """Collect final metrics, telemetry, and the deployable student contract."""
    row = read_last_results(run_dir / "results.csv")
    checkpoint_candidates = (run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt")
    checkpoint = next((path for path in checkpoint_candidates if path.is_file()), None)
    observed = _numeric_fields(row, REQUIRED_VALIDATION_METRICS + OBSERVED_TRAIN_METRICS)
    if (elapsed := _finite_float(row.get("time"))) is not None:
        observed["train/elapsed_s"] = elapsed
    return {
        "run_dir": str(run_dir),
        "results_csv": str(run_dir / "results.csv"),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "last_epoch": _finite_float(row.get("epoch")),
        "observed": observed,
        "results_available": bool(row),
        "telemetry": summarize_telemetry(run_dir),
        "student_only": student_only_contract(checkpoint, imgsz=imgsz, device=device) if checkpoint else None,
    }


def _train_one(spec: dict[str, Any]) -> dict[str, Any]:
    """Train one arm, resuming from its last checkpoint when present."""
    import torch
    from ultralytics import YOLO

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("YOLO_TRAIN_TELEMETRY", "1")
    # This gate disables plots. On macOS, malformed system font caches can make
    # the optional Arial check block dataset validation before training starts.
    # Keep the bypass local to this runner and make it explicit/overridable.
    if os.environ.get("YOLO_F08_SKIP_FONT_CHECK", "1").strip().lower() in {"1", "true", "yes", "on"}:
        import ultralytics.data.utils as data_utils

        def _skip_font_check(*args, **kwargs):
            return None

        data_utils.check_font = _skip_font_check
    torch.manual_seed(int(spec["seed"]))
    run_dir = Path(spec["project"]) / spec["name"]
    checkpoints = (run_dir / "weights" / "last.pt", run_dir / "weights" / "last_healthy.pt")
    resume_checkpoint = next((path for path in checkpoints if path.is_file()), None)
    model = YOLO(str(resume_checkpoint) if resume_checkpoint else spec["model"])
    overrides = dict(spec["overrides"])
    if resume_checkpoint:
        overrides["resume"] = str(resume_checkpoint)
    started = time.perf_counter()
    model.train(**overrides)
    actual_dir = Path(str(getattr(getattr(model, "trainer", None), "save_dir", run_dir)))
    summary = summarize_run(actual_dir, imgsz=int(overrides["imgsz"]), device=str(overrides["device"]))
    summary.update(
        {
            "arm": spec["arm"],
            "name": spec["name"],
            "seed": spec["seed"],
            "foundation": spec["foundation"],
            "foundation_loss": spec["foundation_loss"],
            "foundation_loss_weight": spec["foundation_loss_weight"],
            "elapsed_s": round(time.perf_counter() - started, 4),
            "resumed_from_last": resume_checkpoint is not None,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "status": "completed",
        }
    )
    return summary


def _paired_summary(records: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """Calculate paired D-arm deltas against B0 without filling missing fields."""
    by_key = {(int(row["seed"]), str(row["arm"])): row for row in records}
    pairs = []
    for seed in seeds:
        baseline = by_key.get((seed, "B0"))
        baseline_observed = (baseline or {}).get("observed", {})
        for arm in ("D0", "D1", "D2"):
            branch = by_key.get((seed, arm))
            branch_observed = (branch or {}).get("observed", {})
            deltas = {
                key: round(branch_observed[key] - baseline_observed[key], 8)
                for key in sorted(set(branch_observed) & set(baseline_observed))
            }
            pairs.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "baseline_complete": baseline is not None,
                    "arm_complete": branch is not None,
                    "observed_metric_deltas_vs_B0": deltas,
                }
            )
    complete = len(records) == len(seeds) * len(ARMS)
    required_present = complete and all(
        all(metric in row.get("observed", {}) for metric in REQUIRED_VALIDATION_METRICS) for row in records
    )
    return {
        "pairs": pairs,
        "complete_runs": len(records),
        "total_runs": len(seeds) * len(ARMS),
        "all_runs_complete": complete,
        "required_validation_metrics_present": required_present,
        "accuracy_claim": False,
        "interpretation": "Finite paired observations only; no paper-level or accuracy-improvement claim.",
    }


def _report_payload(
    plan: list[dict[str, Any]],
    records: list[dict[str, Any]],
    data_contract: dict[str, Any],
    *,
    interrupted: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the serializable progress report."""
    seeds = [int(spec["seed"]) for spec in plan if spec["arm"] == "B0"]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "real_data": True,
        "device": "mps",
        "accuracy_claim": False,
        "data_contract": data_contract,
        "plan": plan,
        "records": records,
        "interrupted_runs": interrupted or [],
        "completed_runs": len(records),
        "total_runs": len(plan),
        "summary": _paired_summary(records, seeds),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON report, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _spec_fingerprint(plan: list[dict[str, Any]]) -> dict[str, str]:
    """Return stable serialized specs for resume validation."""
    return {str(spec["name"]): json.dumps(spec, sort_keys=True, separators=(",", ":")) for spec in plan}


def run_effect_gate(
    plan: list[dict[str, Any]],
    output: Path,
    *,
    data_contract: dict[str, Any],
    runner: Callable[[dict[str, Any]], dict[str, Any]] = _train_one,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute a resumable plan, persisting progress after every completed run."""
    records: list[dict[str, Any]] = []
    interrupted: list[dict[str, Any]] = []
    if resume and output.is_file():
        previous = _read_json(output)
        if previous.get("benchmark") != BENCHMARK:
            raise ValueError(f"Cannot resume incompatible report: {output}")
        previous_specs = _spec_fingerprint(list(previous.get("plan") or []))
        current_specs = _spec_fingerprint(plan)
        changed = sorted(
            name for name in set(previous_specs) & set(current_specs) if previous_specs[name] != current_specs[name]
        )
        missing = sorted(set(previous_specs) - set(current_specs))
        if changed or missing:
            raise ValueError("Cannot resume with a changed or truncated plan")
        records = list(previous.get("records") or [])
        interrupted = list(previous.get("interrupted_runs") or [])
    elif output.exists() and not resume:
        raise ValueError(f"Report already exists; pass --resume to continue: {output}")
    completed = {str(record.get("name")) for record in records}
    for spec in plan:
        if spec["name"] in completed:
            continue
        try:
            record = runner(spec)
        except KeyboardInterrupt:
            interrupted.append(
                {
                    "name": spec["name"],
                    "arm": spec["arm"],
                    "seed": spec["seed"],
                    "run_dir": str(Path(spec["project"]) / spec["name"]),
                    "status": "interrupted",
                    "resume_hint": "rerun with --resume; the runner will use weights/last.pt or last_healthy.pt",
                }
            )
            _write_report(output, _report_payload(plan, records, data_contract, interrupted=interrupted))
            raise
        records.append(record)
        completed.add(spec["name"])
        interrupted = [item for item in interrupted if item.get("name") != spec["name"]]
        _write_report(output, _report_payload(plan, records, data_contract, interrupted=interrupted))
    payload = _report_payload(plan, records, data_contract, interrupted=interrupted)
    _write_report(output, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse F08 CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument("--project", type=Path, default=Path("runs/foundation/f08-effect-gate-mps"))
    parser.add_argument("--output", type=Path, default=Path("reports/foundation/v0.1/f08-effect-gate-mps.json"))
    parser.add_argument("--data-output-dir", type=Path, default=Path("reports/foundation/v0.1/f08-data"))
    parser.add_argument("--train-size", type=int, default=1183)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seeds", type=_csv_ints, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--foundation-loss-weight", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-val", dest="val", action="store_false")
    parser.set_defaults(val=True)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.train_size < 1 or args.val_size < 1:
        parser.error("epochs/train-size/val-size must be positive")
    if not 0 < args.fraction <= 1 or args.imgsz < 32 or args.batch < 1 or args.workers < 0:
        parser.error("fraction must be in (0,1], imgsz >= 32, batch positive, workers non-negative")
    if args.foundation_loss_weight < 0 or not math.isfinite(args.foundation_loss_weight):
        parser.error("foundation-loss-weight must be finite and non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    """Prepare data, print/execute the finite MPS effect gate."""
    args = parse_args(argv)
    data_contract = prepare_fixed_subsets(
        args.dataset_root,
        args.data_output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.split_seed,
    )
    plan = build_run_plan(
        dataset=data_contract["dataset_yaml"],
        model=str(Path(args.model).expanduser().resolve()),
        teacher_model=str(Path(args.teacher_model).expanduser().resolve())
        if Path(args.teacher_model).expanduser().exists()
        else args.teacher_model,
        project=str(args.project.expanduser().resolve()),
        seeds=args.seeds,
        epochs=args.epochs,
        fraction=args.fraction,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        foundation_loss_weight=args.foundation_loss_weight,
        val=args.val,
    )
    if args.dry_run:
        print(
            json.dumps(
                {"benchmark": BENCHMARK, "dry_run": True, "data_contract": data_contract, "plan": plan}, indent=2
            )
        )
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("YOLO_TRAIN_TELEMETRY", "1")
    os.environ.setdefault("YOLO_F08_SKIP_FONT_CHECK", "1")
    result = run_effect_gate(plan, args.output.expanduser().resolve(), data_contract=data_contract, resume=args.resume)
    print(json.dumps({"completed_runs": result["completed_runs"], "total_runs": result["total_runs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
