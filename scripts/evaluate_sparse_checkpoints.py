"""Evaluate frozen sparse state dictionaries against matched original-weight controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.suite import collect_environment
from scripts.evaluate_router_choices import RouterChoice, prepare_subset, supported_layers
from ultralytics import YOLO


def load_candidate_state(path, base_hash, layer):
    """Require compatible base-weight provenance, declared Top-2 policy and tensor-only state."""
    provenance = json.loads((path.parent / "report.json").read_text())
    if provenance.get("weights_sha256") != base_hash:
        raise ValueError("candidate base-checkpoint identity mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("candidate payload must be a dictionary")
    if payload.get("layer") != layer or payload.get("policy") != "native_top2":
        raise ValueError("candidate layer/policy mismatch")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("candidate state_dict must be nonempty")
    if any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise TypeError("candidate state_dict must contain named tensors only")
    if any(value.is_floating_point() and not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("candidate state contains non-finite tensors")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--layer", default="model.12")
    parser.add_argument("--max-images", type=int, default=256)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or not args.weights.is_file() or min(args.max_images, args.imgsz) < 1:
        parser.error("fresh output, local checkpoint and positive sizes required")
    base_hash = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    candidates = []
    for path in args.candidates:
        name = f"{path.parent.name}__{path.stem}"
        state = load_candidate_state(path, base_hash, args.layer)
        candidates.append((name, path, state))
    if len({name for name, _, _ in candidates}) != len(candidates):
        parser.error("candidate names must be unique")
    args.output.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    (args.output / "runner.py").write_bytes(source)
    dataset, identities = prepare_subset(args.data, "val", args.max_images, args.output)
    report = {
        "settings": vars(args),
        "weights_sha256": base_hash,
        "script_sha256": hashlib.sha256(source).hexdigest(),
        "environment": collect_environment(),
        "data": identities,
        "candidates": [
            {
                "name": name,
                "path": str(path.resolve()),
                "state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "training_report_sha256": hashlib.sha256((path.parent / "report.json").read_bytes()).hexdigest(),
            }
            for name, path, _ in candidates
        ],
        "cases": [],
        "scope": "Frozen candidate verification only; no selection or tuning on these scores; pretraining overlap unknown",
    }

    def save():
        temporary = args.output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2, default=str, allow_nan=False))
        temporary.replace(args.output / "report.json")

    cases = [
        ("native", None),
        ("top2", None),
        *[(name, state) for name, _, state in candidates],
        ("native_repeat", None),
    ]
    for name, state in cases:
        detector = YOLO(str(args.weights))
        if state is not None:
            detector.model.load_state_dict(state, strict=True)
        detector.model.eval()
        block = supported_layers(detector.model)[args.layer]
        policy = "native" if name in {"native", "native_repeat"} else "native_top2"
        with RouterChoice(block, policy) as routing:
            detector.add_callback("on_val_start", lambda validator: setattr(routing, "active", True))
            detector.add_callback("on_val_end", lambda validator: setattr(routing, "active", False))
            metrics = detector.val(
                data=str(dataset),
                device=args.device,
                imgsz=args.imgsz,
                batch=1,
                workers=0,
                rect=False,
                conf=0.001,
                iou=0.7,
                max_det=300,
                plots=False,
                verbose=False,
                quantize=None,
                project=str(args.output),
                name=name,
            )
            values = {key: float(value) for key, value in metrics.results_dict.items()}
            if not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError("non-finite evaluation metrics")
            if set(metrics.box.image_metrics) != {Path(row["image"]).name for row in identities}:
                raise RuntimeError("evaluation image identity mismatch")
            execution = routing.summary()
            expected = block.num_experts if policy == "native" else 2
            if (
                execution["router_samples"] != len(identities)
                or abs(execution["executed_experts_per_sample"] - expected) > 1e-6
            ):
                raise RuntimeError("evaluation sample/expert execution mismatch")
            report["cases"].append({"name": name, "metrics": values, "execution": execution})
            save()
            print(name, values["metrics/mAP50-95(B)"] * 100, flush=True)
    first, last = report["cases"][0]["metrics"], report["cases"][-1]["metrics"]
    report["baseline_repeat_max_abs_delta"] = max(abs(first[key] - last[key]) for key in first)
    report["baseline_stable"] = report["baseline_repeat_max_abs_delta"] <= 1e-6
    save()
    if not report["baseline_stable"]:
        raise RuntimeError("native baseline drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
