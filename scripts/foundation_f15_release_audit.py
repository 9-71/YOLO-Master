"""Run the read-only F15 release and artifact audit.

The audit validates provenance and deployment boundaries for the completed
real-COCO effect gates.  It intentionally does not turn zero/low-budget AP
measurements into an accuracy claim: a passing audit means the release
artifacts are internally consistent, not that Foundation improves COCO AP.

Example::

    python scripts/foundation_f15_release_audit.py \
        --effect-report reports/foundation/v0.1/f15-real-coco-effect-gate-seeds.json \
        --effect-report reports/foundation/v0.1/f15-real-coco-effect-gate-w005.json \
        --checkpoint-root runs/multitask/f15-effect-gate-w005 \
        --output reports/foundation/v0.1/f15-release-audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASK_METRICS = ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)", "metrics/mAP50-95(P)")
EXPECTED_TASKS = {"detect", "segment", "pose"}
SCHEMA_VERSION = 1


def _check(name: str, passed: bool, details: Any = None) -> dict[str, Any]:
    """Create a stable, JSON-safe audit check result."""
    result = {"name": name, "passed": bool(passed)}
    if details is not None:
        result["details"] = details
    return result


def _finite(value: Any) -> bool:
    """Return whether a value can be represented as a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object payloads."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def audit_effect_report(path: Path) -> list[dict[str, Any]]:
    """Audit one effect-gate report without requiring positive AP."""
    if not path.is_file():
        return [_check("report_exists", False, str(path))]
    try:
        payload = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [_check("report_json", False, f"{path}: {exc}")]

    checks = [
        _check("report_schema", payload.get("schema_version") == SCHEMA_VERSION, payload.get("schema_version")),
        _check("report_benchmark", payload.get("benchmark") == "f15_real_coco_effect_gate", payload.get("benchmark")),
        _check("real_data", payload.get("real_data") is True, payload.get("real_data")),
        _check("accuracy_claim_false", payload.get("accuracy_claim") is False, payload.get("accuracy_claim")),
    ]
    records = payload.get("records")
    total = payload.get("total_runs")
    completed = payload.get("completed_runs")
    checks.append(
        _check(
            "report_complete",
            isinstance(records, list) and completed == total == len(records),
            {
                "completed_runs": completed,
                "total_runs": total,
                "record_count": len(records) if isinstance(records, list) else None,
            },
        )
    )
    groups: dict[tuple[int, float], set[bool]] = defaultdict(set)
    pair_details = []
    for record in records if isinstance(records, list) else []:
        try:
            key = (int(record["seed"]), float(record["foundation_loss_weight"]))
            groups[key].add(bool(record["foundation"]))
            metrics = record.get("validation_metrics") or {}
            metric_ok = all(key in metrics and _finite(metrics[key]) for key in TASK_METRICS)
        except (KeyError, TypeError, ValueError):
            metric_ok = False
            key = (None, None)
        pair_details.append({"seed": key[0], "foundation_loss_weight": key[1], "validation_metrics": metric_ok})
    complete_pairs = all(branches == {False, True} for branches in groups.values())
    checks.append(_check("paired_baseline_foundation", bool(groups) and complete_pairs, pair_details))
    checks.append(
        _check(
            "validation_metrics_present",
            bool(pair_details) and all(item["validation_metrics"] for item in pair_details),
            pair_details,
        )
    )
    return checks


def _model_state_keys(model: Any) -> list[str]:
    """Read model state keys without assuming a specific wrapper class."""
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        return []
    return [str(key) for key in state_dict().keys()]


def _teacher_state_keys(keys: list[str]) -> list[str]:
    """Find registered teacher state while excluding trainable teacher projectors."""
    return [
        key
        for key in keys
        if "teacher_manager" in key.lower() or ".teacher." in key.lower() or key.lower().startswith("teacher.")
    ]


def audit_checkpoint(path: Path) -> list[dict[str, Any]]:
    """Audit one Foundation checkpoint and its deployment-stripped model."""
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload.get("foundation") or {}
        model = payload.get("ema") or payload.get("model")
        keys = _model_state_keys(model)
        from ultralytics.nn.foundation_distill_model import strip_foundation_distillation_model

        stripped = strip_foundation_distillation_model(model)
        stripped_keys = _model_state_keys(stripped)
        teacher_manager = getattr(model, "teacher_manager", None)
        stripped_forbidden = [
            key
            for key in stripped_keys
            if any(token in key.lower() for token in ("teacher", "projector", "foundation"))
        ]
        checks = [
            _check("checkpoint_metadata_training_only", metadata.get("training_only") is True),
            _check("checkpoint_metadata_teacher", metadata.get("teacher") == "dinov3", metadata.get("teacher")),
            _check("checkpoint_metadata_tasks", set(metadata.get("multitask_active_tasks") or ()) == EXPECTED_TASKS),
            _check(
                "teacher_not_registered",
                teacher_manager is None and not _teacher_state_keys(keys),
                {
                    "teacher_manager": type(teacher_manager).__name__ if teacher_manager is not None else None,
                    "teacher_state_keys": _teacher_state_keys(keys),
                },
            ),
            _check("export_strip_student", type(stripped).__name__ == "MultiTaskModel", type(stripped).__name__),
            _check("export_strip_removes_training_components", not stripped_forbidden, stripped_forbidden),
        ]
        return checks
    except Exception as exc:  # pragma: no cover - defensive boundary for corrupted external artifacts
        return [_check("checkpoint_load_and_strip", False, f"{path}: {type(exc).__name__}: {exc}")]


def audit(
    effect_reports: list[Path],
    checkpoint_roots: list[Path],
) -> dict[str, Any]:
    """Run all F15 release checks and return a JSON-safe report."""
    checks: list[dict[str, Any]] = []
    for path in effect_reports:
        checks.extend({"artifact": str(path), **item} for item in audit_effect_report(path))
    checkpoints = sorted(
        {path for root in checkpoint_roots for path in root.glob("foundation-s*/weights/last_healthy.pt")}
    )
    checks.append(_check("foundation_checkpoints_present", bool(checkpoints), [str(path) for path in checkpoints]))
    for path in checkpoints:
        checks.extend({"artifact": str(path), **item} for item in audit_checkpoint(path))
    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "audit": "f15_release",
        "release_audit_passed": passed,
        "accuracy_claim": False,
        "ready_for_accuracy_claim": False,
        "ready_for_next_phase": False,
        "scope_limit": "3 seeds, 3 epochs, fraction=0.005, imgsz=128; use higher-budget training for AP claims",
        "effect_reports": [str(path) for path in effect_reports],
        "checkpoints": [str(path) for path in checkpoints],
        "checks": checks,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse audit command arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-report", action="append", type=Path, required=True)
    parser.add_argument("--checkpoint-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/foundation/v0.1/f15-release-audit.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run and persist the F15 release audit."""
    args = parse_args(argv)
    result = audit(args.effect_report, args.checkpoint_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"release_audit_passed": result["release_audit_passed"], "checks": len(result["checks"])}, sort_keys=True
        )
    )
    if not result["release_audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
