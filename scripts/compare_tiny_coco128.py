#!/usr/bin/env python3
"""Train and compare YOLO-Master v0.15 nano vs tiny on local COCO128.

Reports accuracy (mAP), efficiency (params / GFLOPs / latency) and per-epoch
convergence into a summary.csv under the project directory.

Examples:
    python3.11 scripts/compare_tiny_coco128.py --dry-run
    python3.11 scripts/compare_tiny_coco128.py --check-build
    python3.11 scripts/compare_tiny_coco128.py --models tiny --epochs 5 --imgsz 320 --device mps
    python3.11 scripts/compare_tiny_coco128.py --resume-existing --epochs 5 --imgsz 320 --device mps
    python3.11 scripts/compare_tiny_coco128.py --summary-only --bench --bench-device mps
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import get_flops_with_torch_profiler  # noqa: E402

MODEL_CONFIGS = {
    "tiny": ROOT / "ultralytics/cfg/models/master/v0_15/det/yolo-master-tiny.yaml",
    "nano": ROOT / "ultralytics/cfg/models/master/v0_15/det/yolo-master-n.yaml",
}

METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "val/moe_loss",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    cfg: Path
    run_name: str


def default_data_yaml() -> Path:
    local = ROOT / "datasets/coco128/dataset.yaml"
    has_local_images = any(
        (ROOT / rel).exists()
        for rel in (
            "datasets/coco128/images/train",
            "datasets/coco128/images/val",
            "datasets/coco128/images/train2017",
        )
    )
    if local.exists() and has_local_images:
        return local
    return ROOT / "ultralytics/cfg/datasets/coco128.yaml"


def read_results_csv(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.exists():
        return []
    with results_csv.open(newline="") as f:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def completed_epoch(run_dir: Path) -> int | None:
    rows = read_results_csv(run_dir / "results.csv")
    if not rows:
        return None
    value = rows[-1].get("epoch")
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def bench_latency(cfg: Path, imgsz: int, device: str, warmup: int = 10, iters: int = 30) -> float:
    """Median single-image inference latency (ms) on the requested device."""
    import torch

    model = DetectionModel(str(cfg), ch=3, nc=80, verbose=False).eval()
    if device:
        model = model.to(device)
    x = torch.zeros(1, 3, imgsz, imgsz, device=model.parameters().__next__().device)
    times = []
    with torch.no_grad():
        for i in range(warmup + iters):
            if x.device.type == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            model(x)
            if x.device.type == "mps":
                torch.mps.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return times[len(times) // 2]


def check_build(specs: list[ModelSpec]) -> None:
    for spec in specs:
        model = DetectionModel(str(spec.cfg), ch=3, nc=80, verbose=False)
        params = sum(p.numel() for p in model.parameters())
        flops = get_flops_with_torch_profiler(model)
        print(
            f"[build-ok] {spec.name:<5} params={params / 1e6:.3f}M gflops={flops:.2f} cfg={spec.cfg.relative_to(ROOT)}"
        )


def train_one(args: argparse.Namespace, spec: ModelSpec, data_yaml: Path, project: Path) -> dict[str, str]:
    start = time.time()
    run_dir = project / spec.run_name
    completed = completed_epoch(run_dir)
    last_pt = run_dir / "weights/last.pt"

    if completed is not None and completed >= args.epochs:
        print(f"[skip] {spec.name}: already at epoch {completed}, target={args.epochs}")
        return {"model": spec.name, "status": "skipped", "duration_s": "0"}

    if args.resume_existing and last_pt.exists() and completed is not None:
        print(f"[resume] {spec.name}: {last_pt} epoch={completed} -> target={args.epochs}")
        model = YOLO(str(last_pt))
        resume = True
    else:
        print(f"[train] {spec.name}: cfg={spec.cfg.relative_to(ROOT)} data={data_yaml}")
        model = YOLO(str(spec.cfg))
        resume = False

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(project),
        name=spec.run_name,
        exist_ok=True,
        pretrained=False,
        val=True,
        plots=args.plots,
        cache=args.cache,
        patience=0,
        amp=args.amp,
        resume=resume,
        verbose=args.verbose,
    )
    duration = time.time() - start
    return {"model": spec.name, "status": "resumed" if resume else "ok", "duration_s": f"{duration:.2f}"}


def float_or_blank(value: str | None) -> str:
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.6g}"
    except ValueError:
        return value


def write_summary(project: Path, specs: list[ModelSpec], args: argparse.Namespace) -> Path:
    rows = []
    for spec in specs:
        run_dir = project / spec.run_name
        rows_csv = read_results_csv(run_dir / "results.csv")
        last = rows_csv[-1] if rows_csv else {}
        row: dict[str, str] = {
            "model": spec.name,
            "cfg": str(spec.cfg.relative_to(ROOT)),
            "epoch": last.get("epoch", ""),
        }
        for key in METRIC_KEYS:
            row[key] = float_or_blank(last.get(key))
        # per-epoch mAP50 trajectory for convergence analysis
        trajectory = [float_or_blank(r.get("metrics/mAP50(B)")) for r in rows_csv]
        row["mAP50_trajectory"] = " ".join(trajectory)

        model = DetectionModel(str(spec.cfg), ch=3, nc=80, verbose=False)
        row["params_M"] = f"{sum(p.numel() for p in model.parameters()) / 1e6:.3f}"
        row["gflops_640"] = f"{get_flops_with_torch_profiler(model):.2f}"
        if args.bench:
            row["latency_ms"] = f"{bench_latency(spec.cfg, args.bench_imgsz, args.bench_device):.2f}"
        rows.append(row)

    project.mkdir(parents=True, exist_ok=True)
    out = project / "summary.csv"
    fieldnames = list(rows[0].keys()) if rows else ["model"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["tiny", "nano"], choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--data", type=Path, default=default_data_yaml())
    parser.add_argument("--project", type=Path, default=ROOT / "runs/tiny_coco128_compare")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="'', cpu, mps, 0, ...")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--bench", action="store_true", help="Add median inference latency to the summary.")
    parser.add_argument("--bench-device", default="mps")
    parser.add_argument("--bench-imgsz", type=int, default=640)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-build", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = [ModelSpec(name=m, cfg=MODEL_CONFIGS[m], run_name=m) for m in args.models]
    data_yaml = args.data if args.data.is_absolute() else ROOT / args.data
    project = args.project if args.project.is_absolute() else ROOT / args.project

    print("[compare] models:", ", ".join(s.name for s in specs))
    print("[compare] data:", data_yaml)
    print("[compare] project:", project)
    for spec in specs:
        print(f"  - {spec.name:<5} -> {spec.cfg.relative_to(ROOT)}")

    if args.dry_run:
        return 0
    if args.check_build:
        check_build(specs)
        return 0
    if args.summary_only:
        out = write_summary(project, specs, args)
        print(f"[summary] wrote {out}")
        return 0

    statuses = []
    project.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        try:
            statuses.append(train_one(args, spec, data_yaml, project))
        except Exception as exc:
            print(f"[fail] {spec.name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            statuses.append({"model": spec.name, "status": "failed", "error": str(exc)})
            if args.stop_on_failure:
                break
        finally:
            try:
                out = write_summary(project, specs, args)
                print(f"[summary] updated {out}")
            except OSError as exc:
                print(f"[summary-warning] failed to write summary: {exc}")

    with (project / "status.json").open("w") as f:
        json.dump(statuses, f, indent=2, ensure_ascii=False)
    success_states = {"ok", "skipped", "resumed"}
    return 0 if all(s.get("status") in success_states for s in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
