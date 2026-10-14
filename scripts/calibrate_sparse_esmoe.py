"""Controlled ES-MoE Top-2 BN calibration and block-local feature distillation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_router_choices import RouterChoice, prepare_subset, supported_layers
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules.moe.modules import ES_MOE


def assert_disjoint(calibration, evaluation):
    """Reject content duplicates and COCO image-ID overlap across the two sets."""
    names = {Path(row["image"]).stem for row in evaluation}
    hashes = {row["image_sha256"] for row in evaluation}
    if any(Path(row["image"]).stem in names or row["image_sha256"] in hashes for row in calibration):
        raise ValueError("calibration and evaluation images overlap")


def image_tensor(path, size, device):
    """Match square validation letterboxing, RGB ordering and input scaling."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"unreadable image: {path}")
    image = LetterBox((size, size), auto=False)(image=image)
    image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(image).to(device).float().unsqueeze(0) / 255


def recalibrate_bn(model, images, size, device, scope=None):
    """Update only running BN statistics, preserving affine parameters and module modes."""
    modes = {module: module.training for module in model.modules()}
    norms = [module for module in (scope or model).modules() if isinstance(module, torch.nn.BatchNorm2d)]
    momenta = {module: module.momentum for module in norms}
    try:
        model.eval()
        for module in norms:
            module.reset_running_stats()
            module.momentum = None
            module.train()
        with torch.no_grad():
            for path in images:
                model(image_tensor(path, size, device))
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum
        for module, training in modes.items():
            module.training = training


def distill_block(teacher, student, layer, images, size, device, epochs, lr):
    """Fit the sparse block to frozen native features; keep every BN statistic frozen."""
    block = supported_layers(student)[layer]
    student.eval().requires_grad_(False)
    block.requires_grad_(True)
    optimizer = torch.optim.Adam(block.parameters(), lr=lr)
    captured = {}

    def capture(module, inputs, output):
        captured["input"], captured["target"] = inputs[0].detach(), output.detach()

    handle = supported_layers(teacher)[layer].register_forward_hook(capture)

    def capture_student(module, inputs):
        captured["student_input"] = inputs[0].detach()

    student_handle = block.register_forward_pre_hook(capture_student)
    losses = []
    try:
        for _ in range(epochs):
            epoch_losses = []
            for path in images:
                with torch.no_grad():
                    image = image_tensor(path, size, device)
                    teacher(image)
                    student(image)
                optimizer.zero_grad(set_to_none=True)
                prediction = block(captured["student_input"])
                target = captured["target"]
                loss = (prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite distillation loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in block.parameters()):
                    raise RuntimeError("non-finite distillation gradients")
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            losses.append(sum(epoch_losses) / len(epoch_losses))
    finally:
        handle.remove()
        student_handle.remove()
        student.requires_grad_(False)
    return losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--data", default="coco128.yaml")
    parser.add_argument("--layer", default="model.12")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--bn-scope", choices=("target", "full"), default="target")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    images = sorted(args.calibration.resolve().glob("*.jpg"))
    if not args.weights.is_file() or not images or args.epochs < 1 or args.lr <= 0:
        parser.error("local weights, nonempty calibration images and positive training budget required")
    if args.output.exists():
        parser.error("use a fresh output directory")
    args.output.mkdir(parents=True)
    data, evaluation = prepare_subset(args.data, "val", 128, args.output)
    calibration = [{"image": str(p), "image_sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in images]
    assert_disjoint(calibration, evaluation)
    torch.manual_seed(42)
    source = Path(__file__).read_bytes()
    (args.output / "runner.py").write_bytes(source)
    report = {
        "settings": vars(args),
        "calibration": calibration,
        "evaluation": evaluation,
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(source).hexdigest(),
        "scope": "block-local feature distillation; coco128 regression, not held-out generalization or latency",
        "cases": [],
    }

    def save_report():
        temporary = args.output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2, default=str))
        temporary.replace(args.output / "report.json")

    teacher = YOLO(str(args.weights)).model.to(args.device).float().eval().requires_grad_(False)
    if not isinstance(supported_layers(teacher)[args.layer], ES_MOE):
        raise TypeError("this experiment requires an ES_MOE block")
    bn_state = None
    for name in ("native", "top2", "top2_bn", "top2_kd", "top2_bn_kd", "native_repeat"):
        detector = YOLO(str(args.weights))
        detector.model.to(args.device).float().eval()
        block = supported_layers(detector.model)[args.layer]
        policy = "native" if name.startswith("native") else "native_top2"
        with RouterChoice(block, policy) as routing:
            losses = []
            if name == "top2_bn":
                recalibrate_bn(
                    detector.model,
                    images,
                    args.imgsz,
                    args.device,
                    scope=block.norm if args.bn_scope == "target" else None,
                )
                bn_state = copy.deepcopy(detector.model.state_dict())
            elif name in {"top2_kd", "top2_bn_kd"}:
                if name == "top2_bn_kd":
                    detector.model.load_state_dict(bn_state)
                losses = distill_block(
                    teacher, detector.model, args.layer, images, args.imgsz, args.device, args.epochs, args.lr
                )
            # State dict deliberately omits temporary hooks. Replay the declared Top-2 policy when loading.
            if name in {"top2_bn", "top2_kd", "top2_bn_kd"}:
                torch.save(
                    {"state_dict": detector.model.state_dict(), "policy": policy, "layer": args.layer},
                    args.output / f"{name}.state.pt",
                )
            detector.add_callback("on_val_start", lambda validator: setattr(routing, "active", True))
            detector.add_callback("on_val_end", lambda validator: setattr(routing, "active", False))
            metrics = detector.val(
                data=str(data),
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
                project=str(args.output),
                name=name,
                augment=False,
                quantize=None,
            )
            if routing.observed["router_samples"] != len(evaluation):
                raise RuntimeError("evaluation router sample count mismatch")
            if not all(math.isfinite(float(v)) for v in metrics.results_dict.values()):
                raise RuntimeError("non-finite evaluation metrics")
            report["cases"].append(
                {
                    "name": name,
                    "metrics": metrics.results_dict,
                    "routing": routing.summary(),
                    "distillation_loss": losses,
                }
            )
            save_report()
    first, last = report["cases"][0]["metrics"], report["cases"][-1]["metrics"]
    report["native_repeat_max_abs_delta"] = max(abs(first[key] - last[key]) for key in first)
    report["baseline_stable"] = report["native_repeat_max_abs_delta"] <= 1e-6
    save_report()
    if not report["baseline_stable"]:
        raise RuntimeError("native baseline drift; candidate comparisons require investigation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
