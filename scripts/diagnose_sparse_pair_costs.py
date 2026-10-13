"""Inspect teacher pair-cost diversity on training images without fitting or using final labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_router_choices import supported_layers
from scripts.finetune_sparse_detector import verify_manifest
from scripts.sparse_kd_losses import expert_pairs, oracle_pair_costs
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset


def summarize_costs(costs):
    """Describe normalized costs and margins without mistaking constant labels for useful routing."""
    if costs.ndim != 2 or costs.shape[0] == 0 or costs.shape[1] < 2 or not torch.isfinite(costs).all():
        raise ValueError("finite nonempty image-by-pair costs with at least two pairs required")
    ordered = costs.sort(dim=1).values
    return {
        "images": costs.shape[0],
        "best_pair_counts": costs.argmin(1).bincount(minlength=costs.shape[1]).tolist(),
        "mean_normalized_cost": costs.mean(0).tolist(),
        "std_normalized_cost": costs.std(0, unbiased=False).tolist(),
        "best_to_second_margin_quantiles": torch.quantile(
            ordered[:, 1] - ordered[:, 0], torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--layer", default="model.12")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh output required")
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest)
    data = check_det_dataset(str(args.dataset / "dev.yaml"), autodownload=False)
    cfg = get_cfg(overrides={"imgsz": args.imgsz, "task": "detect", "rect": False})
    dataset = build_yolo_dataset(cfg, str(args.dataset / "train.txt"), args.batch, data, mode="val", rect=False)
    if set(dataset.im_files) != {row["image"] for row in manifest["splits"]["train"]}:
        raise ValueError("training identity mismatch")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, collate_fn=dataset.collate_fn, num_workers=0)
    teacher = YOLO(str(args.weights)).model.to(args.device).float().eval().requires_grad_(False)
    block = supported_layers(teacher)[args.layer]
    captured, handles, rows = {}, [], []

    def capture(key):
        def hook(module, inputs, output):
            captured[key] = output.detach()

        return hook

    handles.append(block.register_forward_hook(capture("target")))
    handles.append(block.routing.register_forward_hook(capture("routing")))
    for index, expert in enumerate(block.experts):
        handles.append(expert.register_forward_hook(capture(index)))
    try:
        with torch.no_grad():
            for batch in loader:
                teacher(batch["img"].to(args.device).float() / 255)
                costs = oracle_pair_costs(
                    block, [captured[i] for i in range(block.num_experts)], captured["routing"], captured["target"]
                )
                energy = captured["target"].square().flatten(1).mean(1).clamp_min(1e-8)
                normalized = (costs / energy[:, None]).cpu().tolist()
                routing = captured["routing"].mean((2, 3)).cpu().tolist()
                rows.extend(
                    {"image": Path(path).name, "normalized_costs": cost, "native_routing": weights}
                    for path, cost, weights in zip(batch["im_file"], normalized, routing)
                )
                captured.clear()
    finally:
        for handle in handles:
            handle.remove()
    if {row["image"] for row in rows} != {Path(row["image"]).name for row in manifest["splits"]["train"]}:
        raise ValueError("diagnostic image identity mismatch")
    result = {
        "settings": vars(args),
        "scope": "training-only teacher feature cost proxy, not detection AP or final-set selection",
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
            for name in (Path(__file__).name, "sparse_kd_losses.py")
        },
        "pairs": expert_pairs(block.num_experts),
        "summary": summarize_costs(torch.tensor([row["normalized_costs"] for row in rows])),
        "images": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str, allow_nan=False))
    print(result["summary"])


if __name__ == "__main__":
    main()
