"""Run a small dense/MoE train-save-reload-val-predict hardware preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data", default="coco8.yaml")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/hardware_preflight/mps")
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch) < 1:
        parser.error("epochs/imgsz/batch must be positive")
    import numpy as np

    from benchmarks.suite import collect_environment, resolve_device
    from ultralytics import YOLO

    device = resolve_device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "scope": "workflow smoke only; random initialization, tiny data, no accuracy comparison",
        "environment": collect_environment(),
        "settings": {**vars(args), "device": str(device)},
        "cases": [],
    }
    for name, filename in (("dense", "yolo-master-n-no-moe.yaml"), ("moe", "yolo-master-n.yaml")):
        case = {"id": name, "status": "running"}
        try:
            model_path = ROOT / "ultralytics/cfg/models/master/v0_10/det" / filename
            model = YOLO(str(model_path))
            model.train(
                data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=str(device),
                workers=0,
                seed=args.seed,
                amp=False,
                pretrained=False,
                project=str(output),
                name=name,
                exist_ok=False,
                plots=False,
                verbose=False,
            )
            checkpoint = Path(model.trainer.last)
            results_path = Path(model.trainer.save_dir) / "results.csv"
            with results_path.open(encoding="utf-8", newline="") as handle:
                rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]
            losses = [float(value) for row in rows for key, value in row.items() if "loss" in key]
            if not losses or not all(np.isfinite(value) for value in losses):
                raise RuntimeError("training losses missing or non-finite")
            restored = YOLO(str(checkpoint))
            metrics = restored.val(
                data=args.data,
                imgsz=args.imgsz,
                batch=args.batch,
                device=str(device),
                workers=0,
                plots=False,
                verbose=False,
                project=str(output),
                name=f"{name}_val",
            )
            predictions = restored.predict(
                np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8),
                imgsz=args.imgsz,
                device=str(device),
                verbose=False,
            )
            values = {key: float(value) for key, value in metrics.results_dict.items()}
            if not all(np.isfinite(value) for value in values.values()):
                raise RuntimeError("non-finite validation metrics")
            if len(predictions) != 1:
                raise RuntimeError("checkpoint prediction did not return one result")
            case.update(
                status="ok",
                checkpoint=str(checkpoint),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                metrics=values,
                prediction_count=len(predictions),
                training_results=str(results_path),
                finite_training_losses=True,
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-case failure evidence and try the next model
            case.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        report["cases"].append(case)
        (output / "preflight.json").write_text(
            json.dumps(report, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(case, default=str))
    return int(any(case["status"] != "ok" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
