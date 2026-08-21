#!/usr/bin/env python3
"""Run the F09 D2-to-D3 foreground-aware Foundation effect gate.

F09 compares sampled hybrid DINOv3 KD without foreground weighting (``D2``)
against the same KD with GT-derived interior/boundary/background token weights
(``D3``). It deliberately admits execution only after the finite F08 report
contains complete, analyzable B0/D2 evidence for three paired seeds.

The runner does not make an accuracy claim. It persists progress after every
completed run, leaves missing values missing, and reuses F08's training,
telemetry, checkpoint-boundary, and resume contracts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import foundation_f08_effect_gate as f08  # noqa: E402

BENCHMARK = "foundation_f09_foreground_effect_gate"
SCHEMA_VERSION = 1
ARMS = ("D2", "D3")


def _finite(value: Any) -> bool:
    """Return whether an arbitrary scalar can be represented as finite."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _check(name: str, passed: bool, details: Any = None) -> dict[str, Any]:
    """Build one JSON-safe F08 admission check."""
    result = {"name": name, "passed": bool(passed)}
    if details is not None:
        result["details"] = details
    return result


def load_f08_report(path: Path) -> dict[str, Any]:
    """Load an F08 report and reject malformed external JSON early."""
    if not path.is_file():
        raise FileNotFoundError(f"F08 effect report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"F08 effect report must contain a JSON object: {path}")
    return payload


def f08_admission(payload: dict[str, Any], *, source: Path | None = None) -> dict[str, Any]:
    """Validate that F08 gives F09 a complete and analyzable D2 comparison.

    F09 does not require a positive F08 mAP delta: the roadmap permits an
    analyzable result. It does require three complete B0/D2 pairs with every
    required detection metric, finite D2 Foundation engagement, and a clean
    deployable-student checkpoint scan. This prevents a foreground ablation
    from being launched on partial, zero-signal, or teacher-leaking evidence.
    """
    records = payload.get("records")
    plan = payload.get("plan")
    data_contract = payload.get("data_contract")
    checks = [
        _check("benchmark", payload.get("benchmark") == f08.BENCHMARK, payload.get("benchmark")),
        _check("real_data", payload.get("real_data") is True, payload.get("real_data")),
        _check("accuracy_claim_false", payload.get("accuracy_claim") is False, payload.get("accuracy_claim")),
        _check("plan_present", isinstance(plan, list) and bool(plan)),
        _check(
            "data_contract",
            isinstance(data_contract, dict) and bool(data_contract.get("dataset_yaml")),
            (data_contract or {}).get("dataset_yaml") if isinstance(data_contract, dict) else None,
        ),
        _check("no_interrupted_runs", not payload.get("interrupted_runs"), payload.get("interrupted_runs", [])),
    ]
    complete = isinstance(records, list) and payload.get("completed_runs") == payload.get("total_runs") == len(records)
    checks.append(
        _check(
            "matrix_complete",
            complete,
            {
                "completed_runs": payload.get("completed_runs"),
                "total_runs": payload.get("total_runs"),
                "record_count": len(records) if isinstance(records, list) else None,
            },
        )
    )

    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    if isinstance(records, list):
        for record in records:
            try:
                indexed[(str(record["arm"]), int(record["seed"]))] = record
            except (KeyError, TypeError, ValueError):
                continue
    seeds = sorted(seed for arm, seed in indexed if arm == "B0")
    seeds = sorted(set(seeds))
    checks.append(_check("three_baseline_seeds", len(seeds) == 3, seeds))

    pair_details = []
    for seed in seeds:
        baseline, hybrid = indexed.get(("B0", seed)), indexed.get(("D2", seed))
        baseline_metrics = (baseline or {}).get("observed", {})
        hybrid_metrics = (hybrid or {}).get("observed", {})
        metrics_ok = all(
            _finite(baseline_metrics.get(metric)) and _finite(hybrid_metrics.get(metric))
            for metric in f08.REQUIRED_VALIDATION_METRICS
        )
        foundation_ok = bool(hybrid and hybrid.get("foundation") and hybrid.get("foundation_loss") == "hybrid")
        foundation_ok = foundation_ok and _finite(hybrid_metrics.get("train/foundation_loss"))
        student_only = (hybrid or {}).get("student_only") or {}
        checkpoint_ok = student_only.get("teacher_state_clean") is True
        pair_details.append(
            {
                "seed": seed,
                "baseline_present": baseline is not None,
                "hybrid_present": hybrid is not None,
                "validation_metrics": metrics_ok,
                "hybrid_engaged": foundation_ok,
                "student_checkpoint_clean": checkpoint_ok,
            }
        )
    checks.extend(
        [
            _check(
                "complete_B0_D2_pairs",
                len(pair_details) == 3
                and all(item["baseline_present"] and item["hybrid_present"] for item in pair_details),
                pair_details,
            ),
            _check(
                "validation_metrics_present",
                bool(pair_details) and all(item["validation_metrics"] for item in pair_details),
                pair_details,
            ),
            _check(
                "hybrid_engaged",
                bool(pair_details) and all(item["hybrid_engaged"] for item in pair_details),
                pair_details,
            ),
            _check(
                "student_checkpoint_clean",
                bool(pair_details) and all(item["student_checkpoint_clean"] for item in pair_details),
                pair_details,
            ),
        ]
    )
    d2_specs = [spec for spec in plan or [] if isinstance(spec, dict) and spec.get("arm") == "D2"]
    d2_spec = d2_specs[0] if d2_specs else {}
    checks.append(_check("D2_plan_present", bool(d2_spec), d2_spec.get("name")))
    return {
        "source": str(source) if source else None,
        "ready": all(check["passed"] for check in checks),
        "checks": checks,
        "seeds": seeds,
        "data_contract": data_contract if isinstance(data_contract, dict) else {},
        "d2_spec": d2_spec,
    }


def build_run_plan(*, admission: dict[str, Any], project: str) -> list[dict[str, Any]]:
    """Build matched D2/D3 runs from the admitted F08 D2 specification."""
    if not admission.get("ready"):
        raise ValueError("F09 is blocked until F08 admission passes")
    d2 = dict(admission["d2_spec"])
    overrides = dict(d2.get("overrides") or {})
    dataset = str(admission["data_contract"]["dataset_yaml"])
    model = str(d2["model"])
    teacher_model = str(d2["teacher_model"])
    seeds = list(admission["seeds"])
    plan = []
    for arm in ARMS:
        for seed in seeds:
            name = f"{arm.lower()}-s{seed}"
            arm_overrides = dict(overrides)
            arm_overrides.update(
                {
                    "model": model,
                    "data": dataset,
                    "project": project,
                    "name": name,
                    "seed": seed,
                    "foundation_enabled": True,
                    "foundation_teacher": "dinov3",
                    "foundation_model": teacher_model,
                    "foundation_target_levels": ["p4"],
                    "foundation_loss": "hybrid",
                    "foundation_relation_mode": "sampled",
                    "foundation_foreground_weighting": arm == "D3",
                    "foundation_foreground_weight": 1.5,
                    "foundation_boundary_weight": 1.0,
                    "foundation_background_weight": 0.25,
                }
            )
            plan.append(
                {
                    "arm": arm,
                    "name": name,
                    "seed": seed,
                    "model": model,
                    "dataset": dataset,
                    "project": project,
                    "teacher_model": teacher_model,
                    "foundation": True,
                    "foundation_loss": "hybrid",
                    "foundation_loss_weight": arm_overrides["foundation_loss_weight"],
                    "foreground_weighting": arm == "D3",
                    "foreground_weights": {
                        "interior": arm_overrides["foundation_foreground_weight"],
                        "boundary": arm_overrides["foundation_boundary_weight"],
                        "background": arm_overrides["foundation_background_weight"],
                    },
                    "initialization_contract": {
                        "pretrained": False,
                        "same_model_config": True,
                        "same_seed": True,
                        "same_dataset_split": True,
                        "same_optimizer": True,
                        "same_foundation_kd": True,
                        "only_foreground_weighting_differs": True,
                    },
                    "overrides": arm_overrides,
                }
            )
    return plan


def _paired_summary(records: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """Summarize D3-minus-D2 observations without filling in absent values."""
    indexed = {(str(record.get("arm")), int(record.get("seed"))): record for record in records}
    pairs = []
    for seed in seeds:
        d2, d3 = indexed.get(("D2", seed)), indexed.get(("D3", seed))
        d2_observed, d3_observed = (d2 or {}).get("observed", {}), (d3 or {}).get("observed", {})
        deltas = {
            key: round(float(d3_observed[key]) - float(d2_observed[key]), 8)
            for key in sorted(set(d2_observed) & set(d3_observed))
            if _finite(d2_observed[key]) and _finite(d3_observed[key])
        }
        pairs.append(
            {
                "seed": seed,
                "D2_complete": d2 is not None,
                "D3_complete": d3 is not None,
                "observed_metric_deltas_D3_minus_D2": deltas,
            }
        )
    complete = len(records) == len(seeds) * len(ARMS)
    required_metrics = complete and all(
        all(metric in record.get("observed", {}) for metric in f08.REQUIRED_VALIDATION_METRICS) for record in records
    )
    return {
        "pairs": pairs,
        "all_runs_complete": complete,
        "required_validation_metrics_present": required_metrics,
        "accuracy_claim": False,
        "interpretation": "Finite D3-minus-D2 observations only; no accuracy-improvement or paper-level claim.",
    }


def _report_payload(
    plan: list[dict[str, Any]],
    records: list[dict[str, Any]],
    admission: dict[str, Any],
    *,
    interrupted: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a stable, resumable F09 effect-gate report."""
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "real_data": True,
        "accuracy_claim": False,
        "parent_f08_admission": admission,
        "plan": plan,
        "records": records,
        "interrupted_runs": interrupted or [],
        "completed_runs": len(records),
        "total_runs": len(plan),
        "summary": _paired_summary(records, list(admission["seeds"])),
    }


def run_effect_gate(
    plan: list[dict[str, Any]],
    output: Path,
    *,
    admission: dict[str, Any],
    runner: Callable[[dict[str, Any]], dict[str, Any]] = f08._train_one,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute and persist a D2/D3 plan, including interruption recovery state."""
    records: list[dict[str, Any]] = []
    interrupted: list[dict[str, Any]] = []
    if resume and output.is_file():
        previous = f08._read_json(output)
        if previous.get("benchmark") != BENCHMARK:
            raise ValueError(f"Cannot resume incompatible report: {output}")
        if f08._spec_fingerprint(list(previous.get("plan") or [])) != f08._spec_fingerprint(plan):
            raise ValueError("Cannot resume with a changed plan")
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
                    "resume_hint": "rerun with --resume; F08 runner restores last.pt or last_healthy.pt",
                }
            )
            f08._write_report(output, _report_payload(plan, records, admission, interrupted=interrupted))
            raise
        records.append(record)
        completed.add(spec["name"])
        interrupted = [item for item in interrupted if item.get("name") != spec["name"]]
        f08._write_report(output, _report_payload(plan, records, admission, interrupted=interrupted))
    payload = _report_payload(plan, records, admission, interrupted=interrupted)
    f08._write_report(output, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse F09 runner arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f08-report", type=Path, default=Path("reports/foundation/v0.1/f08-effect-gate-mps-256.json"))
    parser.add_argument("--project", type=Path, default=Path("runs/foundation/f09-foreground-effect-gate-mps-256"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/foundation/v0.1/f09-foreground-effect-gate-mps-256.json")
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Validate F08 evidence, then print or run the constrained F09 matrix."""
    args = parse_args(argv)
    report_path = args.f08_report.expanduser().resolve()
    admission = f08_admission(load_f08_report(report_path), source=report_path)
    if not admission["ready"]:
        print(json.dumps({"benchmark": BENCHMARK, "ready": False, "admission": admission}, indent=2))
        if not args.dry_run:
            raise SystemExit("F09 is blocked: complete analyzable F08 B0/D2 evidence is required.")
        return
    plan = build_run_plan(admission=admission, project=str(args.project.expanduser().resolve()))
    if args.dry_run:
        print(json.dumps({"benchmark": BENCHMARK, "dry_run": True, "admission": admission, "plan": plan}, indent=2))
        return
    result = run_effect_gate(plan, args.output.expanduser().resolve(), admission=admission, resume=args.resume)
    print(json.dumps({"completed_runs": result["completed_runs"], "total_runs": result["total_runs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
