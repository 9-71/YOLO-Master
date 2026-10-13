"""Evaluate native and fixed expert choices at one layer of a local detector checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from contextlib import AbstractContextManager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from benchmarks.suite import collect_environment, resolve_device
from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.mot import MoTBlock


def supported_layers(model):
    """Return explicit supported blocks, excluding their nested router modules."""
    return {name: module for name, module in model.named_modules() if isinstance(module, (ES_MOE, MoTBlock))}


class RouterChoice(AbstractContextManager):
    """Temporarily intervene at one block; restore flags and remove hooks even on failure."""

    def __init__(self, block, policy):
        if not isinstance(block, (ES_MOE, MoTBlock)):
            raise TypeError("only ES_MOE and MoTBlock are supported")
        if block.training:
            raise ValueError("router intervention requires eval mode")
        self.block, self.policy = block, policy
        self.expert = int(policy.split(":", 1)[1]) if policy.startswith("fixed:") else None
        if self.expert is not None and not 0 <= self.expert < block.num_experts:
            raise ValueError("fixed expert is outside the block's expert range")
        if self.expert is None and policy not in {"native", "native_top1", "native_top2", "uniform"}:
            raise ValueError(f"unknown routing policy: {policy}")
        if policy == "native_top2" and block.num_experts < 2:
            raise ValueError("native_top2 requires at least two experts")
        self.handles, self.saved = [], {}
        self.active = False
        self.observed = {
            "router_samples": 0,
            "mean_weight_sum": [0.0] * block.num_experts,
            "expert_calls": [0] * block.num_experts,
            "sample_expert_pairs": [0] * block.num_experts,
        }

    def __enter__(self):
        if self.policy != "native":
            uniform = self.policy == "uniform"
            overrides = {"top_k": self.block.num_experts if uniform else (2 if self.policy == "native_top2" else 1)}
            if isinstance(self.block, ES_MOE):
                overrides.update(use_top_k=not uniform, use_sparse_inference=not uniform)
                if self.policy == "native_top2":
                    overrides["dynamic_threshold"] = 0.0
            for attr, value in overrides.items():
                self.saved[attr] = (hasattr(self.block, attr), getattr(self.block, attr, None))
                setattr(self.block, attr, value)
        router = self.block.routing if isinstance(self.block, ES_MOE) else self.block.router
        self.handles.append(router.register_forward_hook(self._route))
        for index, expert in enumerate(self.block.experts):
            self.handles.append(expert.register_forward_pre_hook(self._observe_expert(index)))
        return self

    def _route(self, module, inputs, output):
        weights = output if isinstance(output, torch.Tensor) else output[0]
        if self.policy == "uniform":
            weights = torch.full_like(weights, 1.0 / self.block.num_experts)
            output = weights if isinstance(output, torch.Tensor) else (weights, *output[1:])
        elif self.policy != "native":
            if self.policy == "native_top2":
                values, indices = weights.topk(2, dim=1)
                values = values / values.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
            else:
                indices = weights.argmax(dim=1, keepdim=True)
                values = torch.ones_like(indices, dtype=weights.dtype)
            if self.expert is not None:
                indices = torch.full_like(indices, self.expert)
            weights = torch.zeros_like(weights).scatter_(1, indices, values)
            output = weights if isinstance(output, torch.Tensor) else (weights, indices, *output[2:])
        if self.active:
            self.observed["router_samples"] += weights.shape[0]
            means = weights.detach().float().mean(dim=(2, 3)).sum(dim=0).cpu().tolist()
            self.observed["mean_weight_sum"] = [a + b for a, b in zip(self.observed["mean_weight_sum"], means)]
        return output

    def _observe_expert(self, index):
        def observe(module, inputs):
            if self.active:
                self.observed["expert_calls"][index] += 1
                self.observed["sample_expert_pairs"][index] += int(inputs[0].shape[0])

        return observe

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        for attr, (existed, value) in self.saved.items():
            if existed:
                setattr(self.block, attr, value)
            else:
                delattr(self.block, attr)
        return False

    def summary(self):
        samples = self.observed["router_samples"]
        return {
            **self.observed,
            "mean_router_weight": [value / max(samples, 1) for value in self.observed["mean_weight_sum"]],
            "executed_experts_per_sample": sum(self.observed["sample_expert_pairs"]) / max(samples, 1),
        }


def prepare_subset(data, split, limit, output):
    """Freeze sorted image/label identities without automatically downloading data."""
    from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths

    resolved = check_det_dataset(data, autodownload=False)
    sources = resolved[split]
    files = []
    for source in sources if isinstance(sources, list) else [sources]:
        source = Path(source)
        if source.is_dir():
            files.extend(p.resolve() for p in source.rglob("*") if p.suffix.lower().lstrip(".") in IMG_FORMATS)
        else:
            for line in source.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    path = Path(line.strip())
                    files.append((source.parent / path).resolve() if not path.is_absolute() else path.resolve())
    files = sorted(set(files))[:limit]
    if not files or len({p.name for p in files}) != len(files):
        raise ValueError("subset must be nonempty with unique filenames for per-image metric keys")
    labels = [Path(p) for p in img2label_paths([str(p) for p in files])]
    identities = [
        {
            "image": str(image),
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "label": str(label),
            "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest() if label.exists() else None,
        }
        for image, label in zip(files, labels)
    ]
    image_list = output / "images.txt"
    image_list.write_text("\n".join(str(p) for p in files) + "\n", encoding="utf-8")
    dataset = output / "subset.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {"path": str(output), "train": str(image_list), "val": str(image_list), "names": resolved["names"]}
        ),
        encoding="utf-8",
    )
    return dataset, identities


def summarize_choices(cases):
    """Describe fixed-checkpoint differences; never select a deployable policy from test data."""
    by_policy = {case["policy"]: case for case in cases if case["status"] == "ok"}
    if "native" not in by_policy:
        return {}
    native = by_policy["native"]
    key = "metrics/mAP50-95(B)"
    result = {
        "map50_95_delta_points_vs_native": {
            policy: 100 * (case["metrics"][key] - native["metrics"][key]) for policy, case in by_policy.items()
        },
        "interpretation": "Single-checkpoint diagnostic. No statistical, generalization or router-training claim.",
    }
    if "native_repeat" in by_policy:
        repeat = by_policy["native_repeat"]
        difference = max(abs(repeat["metrics"][key] - value) for key, value in native["metrics"].items())
        result.update(
            baseline_repeat_max_metric_abs_delta=difference, baseline_repeat_within_tolerance=difference <= 1e-6
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--layer", help="exact block name; defaults to first supported block")
    parser.add_argument("--list-layers", action="store_true")
    parser.add_argument("--policies", nargs="+", help="optional subset; native and native_repeat are always included")
    parser.add_argument("--data", default="coco128.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.is_file() or min(args.max_images, args.imgsz, args.batch) < 1:
        parser.error("local checkpoint and positive max-images/imgsz/batch are required")
    from ultralytics import YOLO

    device = resolve_device(args.device)
    probe = YOLO(str(args.weights)).model.eval()
    layers = supported_layers(probe)
    if args.list_layers:
        print(
            json.dumps(
                {name: {"type": type(block).__name__, "experts": block.num_experts} for name, block in layers.items()},
                indent=2,
            )
        )
        return 0
    layer = args.layer or next(iter(layers), None)
    if layer not in layers:
        parser.error(f"unsupported layer {layer!r}; use --list-layers")
    count = layers[layer].num_experts
    del probe
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "choices.json").exists():
        parser.error("output already contains choices.json; use a fresh directory")
    source = Path(__file__).read_bytes()
    (output / "runner.py").write_bytes(source)
    dataset, identities = prepare_subset(args.data, args.split, args.max_images, output)
    report = {
        "schema_version": 1,
        "scope": "single-layer fixed-checkpoint expert-choice diagnostic",
        "weights": str(args.weights.resolve()),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(source).hexdigest(),
        "environment": collect_environment(),
        "settings": {**vars(args), "layer": layer, "device": str(device)},
        "data": {
            "images": identities,
            "subset_sha256": hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest(),
        },
        "limitations": [
            "Data may overlap pretraining. No held-out generalization claim.",
            "Fixed expert interventions change both choice and mixture weights.",
            "Telemetry hooks add overhead; validation speed is not a performance benchmark.",
            "Expert indices are local to this layer and checkpoint.",
        ],
        "cases": [],
    }
    requested = args.policies or (
        "uniform",
        "native_top1",
        "native_top2",
        *(f"fixed:{index}" for index in range(count)),
    )
    policies = tuple(dict.fromkeys(("native", *requested, "native_repeat")))
    for policy in policies:
        case = {"policy": policy, "status": "running"}
        try:
            model = YOLO(str(args.weights))
            model.model.eval()
            block = supported_layers(model.model)[layer]
            with RouterChoice(block, "native" if policy == "native_repeat" else policy) as intervention:
                model.add_callback("on_val_start", lambda validator: setattr(intervention, "active", True))
                model.add_callback("on_val_end", lambda validator: setattr(intervention, "active", False))
                metrics = model.val(
                    data=str(dataset),
                    split="val",
                    device=str(device),
                    imgsz=args.imgsz,
                    batch=args.batch,
                    workers=0,
                    plots=False,
                    verbose=False,
                    rect=False,
                    project=str(output),
                    name=policy.replace(":", "_"),
                    conf=0.001,
                    iou=0.7,
                    max_det=300,
                    augment=False,
                    quantize=None,
                )
                values = {key: float(value) for key, value in metrics.results_dict.items()}
                if not all(math.isfinite(value) for value in values.values()):
                    raise RuntimeError("non-finite validation metrics")
                if intervention.observed["router_samples"] != len(identities):
                    raise RuntimeError("router sample count does not match frozen subset")
                expected_images = {Path(identity["image"]).name for identity in identities}
                if set(metrics.box.image_metrics) != expected_images:
                    raise RuntimeError("per-image metric identities do not match frozen subset")
                case.update(
                    status="ok", metrics=values, execution=intervention.summary(), per_image=metrics.box.image_metrics
                )
        except Exception as exc:  # noqa: BLE001 - preserve failed cases without claiming a complete comparison
            case.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        report["cases"].append(case)
        report["summary"] = summarize_choices(report["cases"])
        pending = output / "choices.json.tmp"
        pending.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")
        pending.replace(output / "choices.json")
        print(json.dumps({key: value for key, value in case.items() if key != "per_image"}), flush=True)
    return int(
        any(case["status"] != "ok" for case in report["cases"])
        or not report["summary"].get("baseline_repeat_within_tolerance", False)
    )


if __name__ == "__main__":
    raise SystemExit(main())
