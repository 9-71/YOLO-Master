"""Fixed-budget supervised sparse ES-MoE pilot with development-only candidate selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.suite import collect_environment
from scripts.calibrate_sparse_esmoe import assert_disjoint
from scripts.evaluate_router_choices import RouterChoice, supported_layers
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules.moe.modules import ES_MOE


def verify_manifest(manifest):
    """Verify every frozen image and label before allowing training or selection."""
    splits = manifest["splits"]
    if set(splits) != {"train", "dev", "test"}:
        raise ValueError("manifest must contain exactly train/dev/test")
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"empty split: {name}")
        if len({Path(row["image"]).stem for row in rows}) != len(rows):
            raise ValueError(f"duplicate image IDs in {name}")
        for row in rows:
            for kind in ("image", "label"):
                if hashlib.sha256(Path(row[kind]).read_bytes()).hexdigest() != row[f"{kind}_sha256"]:
                    raise ValueError(f"{kind} identity changed: {row[kind]}")
    assert_disjoint(splits["train"], splits["dev"])
    assert_disjoint(splits["train"], splits["test"])
    assert_disjoint(splits["dev"], splits["test"])


def choose_candidate(cases, candidate_names=None):
    """Use only development AP, with a stable tie break; never inspect test metrics."""
    expected = set(candidate_names or ("sparse_det", "sparse_det_kd"))
    eligible = [case for case in cases if case["name"] in expected]
    if (
        len(eligible) != len(expected)
        or {case["name"] for case in eligible} != expected
        or any(not math.isfinite(case["dev_ap"]) for case in eligible)
    ):
        raise ValueError("all declared sparse development scores must be present and finite")
    return max(eligible, key=lambda case: case["dev_ap"])["name"]


def train_target(model, teacher, layer, loader, epochs, lr, kd_weight, detection_weight=1.0):
    """Optimize one eval-mode block using detection loss and optional native-feature KD."""
    if min(detection_weight, kd_weight) < 0 or detection_weight + kd_weight <= 0:
        raise ValueError("loss weights must be nonnegative with at least one positive term")
    model.eval().requires_grad_(False)
    block = supported_layers(model)[layer]
    block.requires_grad_(True)
    optimizer = torch.optim.Adam(block.parameters(), lr=lr)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    bn = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name
    }
    captured, handles = {}, []
    if kd_weight:
        if teacher is None:
            raise ValueError("KD requires a frozen teacher")
        teacher.eval().requires_grad_(False)

        def capture_teacher(module, inputs, output):
            captured["teacher"] = output.detach()

        def capture_student(module, inputs, output):
            captured["student"] = output

        handles = [
            supported_layers(teacher)[layer].register_forward_hook(capture_teacher),
            block.register_forward_hook(capture_student),
        ]
    history = []
    try:
        for epoch in range(epochs):
            values = []
            image_order = hashlib.sha256()
            ordered_images, examples = 0, 0
            for raw_batch in loader:
                for path in raw_batch.get("im_file", ()):
                    image_order.update((Path(path).name + "\n").encode())
                    ordered_images += 1
                examples += int(raw_batch["img"].shape[0])
                device = next(model.parameters()).device
                batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in raw_batch.items()
                }
                batch["img"] = batch["img"].float() / 255
                if kd_weight:
                    with torch.no_grad():
                        teacher(batch["img"])
                optimizer.zero_grad(set_to_none=True)
                predictions = model(batch["img"])
                detection = (
                    model.loss(batch, predictions)[0].sum() / batch["img"].shape[0]
                    if detection_weight
                    else batch["img"].new_zeros(())
                )
                kd = detection.new_zeros(())
                if kd_weight:
                    target = captured["teacher"]
                    kd = (captured["student"] - target).square().mean() / target.square().mean().clamp_min(1e-8)
                loss = detection_weight * detection + kd_weight * kd
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in block.parameters()):
                    raise RuntimeError("non-finite target gradients")
                torch.nn.utils.clip_grad_norm_(block.parameters(), 10.0)
                optimizer.step()
                values.append((float(detection.detach()), float(kd.detach())))
                captured.clear()
            means = np.asarray(values).mean(axis=0).tolist()
            epoch_record = {
                "epoch": epoch + 1,
                "detection_loss": means[0],
                "feature_kd_loss": means[1],
                "optimizer_steps": len(values),
                "examples": examples,
            }
            if ordered_images:
                epoch_record["image_order_sha256"] = image_order.hexdigest()
            history.append(epoch_record)
            print(history[-1], flush=True)
    finally:
        for handle in handles:
            handle.remove()
        model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name in frozen and not torch.equal(parameter, frozen[name]):
            raise RuntimeError(f"frozen parameter changed: {name}")
    for name, value in bn.items():
        if not torch.equal(model.state_dict()[name], value):
            raise RuntimeError(f"frozen BN statistic changed: {name}")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--layer", default="model.12")
    parser.add_argument("--batch", type=int, default=1, help="training batch; evaluation remains batch=1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--include-kd-only", action="store_true", help="add a matched pure-KD arm to selection")
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output, args.dataset = args.output.resolve(), args.dataset.resolve()
    if args.output.exists() or not args.weights.is_file() or min(args.epochs, args.imgsz, args.lr, args.batch) <= 0:
        parser.error("fresh output, local weights and positive budget required")
    if args.kd_weight <= 0:
        parser.error("KD arm requires positive kd-weight")
    manifest_file = args.dataset / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    verify_manifest(manifest)
    data = check_det_dataset(str(args.dataset / "dev.yaml"), autodownload=False)
    args.output.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    (args.output / "runner.py").write_bytes(source)
    report = {
        "settings": vars(args),
        "environment": collect_environment(),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(source).hexdigest(),
        "dev": [],
        "test": [],
        "scope": "single-seed small COCO pilot; pretrained overlap unknown; no deployment latency claim",
    }
    states = {}

    def save():
        temporary = args.output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2, default=str))
        temporary.replace(args.output / "report.json")

    def evaluate(state, policy, split, name):
        detector = YOLO(str(args.weights))
        detector.model.load_state_dict(state)
        detector.model.eval()
        with RouterChoice(supported_layers(detector.model)[args.layer], policy) as routing:
            detector.add_callback("on_val_start", lambda validator: setattr(routing, "active", True))
            detector.add_callback("on_val_end", lambda validator: setattr(routing, "active", False))
            metrics = detector.val(
                data=str(args.dataset / f"{split}.yaml"),
                split="val",
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
                name=f"{split}_{name}",
            )
            if routing.observed["router_samples"] != len(manifest["splits"][split]):
                raise RuntimeError("evaluation image count mismatch")
            expected_images = {Path(row["image"]).name for row in manifest["splits"][split]}
            if set(metrics.box.image_metrics) != expected_images:
                raise RuntimeError("evaluation image identities do not match frozen split")
            if not all(math.isfinite(float(v)) for v in metrics.results_dict.values()):
                raise RuntimeError("non-finite evaluation metrics")
            return {"name": name, "metrics": metrics.results_dict, "routing": routing.summary()}

    sparse_names = (
        ("sparse_det", "sparse_kd", "sparse_det_kd") if args.include_kd_only else ("sparse_det", "sparse_det_kd")
    )
    arm_names = ("native", "top2", "dense_det", *sparse_names)
    report["declared_sparse_candidates"] = sparse_names
    for name in arm_names:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        model = YOLO(str(args.weights)).model.to(args.device).float().eval()
        block = supported_layers(model)[args.layer]
        if not isinstance(block, ES_MOE):
            raise TypeError("supervised pilot supports ES_MOE only")
        model.args = get_cfg(overrides={"imgsz": args.imgsz, "task": "detect", "rect": False})
        policy = "native" if name in {"native", "dense_det"} else "native_top2"
        history, counts = [], None
        if name in {"dense_det", *sparse_names}:
            cfg = get_cfg(overrides={"imgsz": args.imgsz, "task": "detect", "rect": False})
            dataset = build_yolo_dataset(cfg, str(args.dataset / "train.txt"), args.batch, data, mode="val", rect=False)
            expected = {row["image"] for row in manifest["splits"]["train"]}
            if set(dataset.im_files) != expected:
                raise RuntimeError("training loader does not match frozen split")
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch,
                shuffle=True,
                num_workers=0,
                collate_fn=dataset.collate_fn,
                generator=torch.Generator().manual_seed(args.seed),
            )
            teacher = (
                YOLO(str(args.weights)).model.to(args.device).float().eval()
                if name in {"sparse_kd", "sparse_det_kd"}
                else None
            )
            with RouterChoice(block, policy) as routing:
                routing.active = True
                history = train_target(
                    model,
                    teacher,
                    args.layer,
                    loader,
                    args.epochs,
                    args.lr,
                    args.kd_weight if teacher is not None else 0.0,
                    detection_weight=0.0 if name == "sparse_kd" else 1.0,
                )
                counts = routing.summary()
                if counts["router_samples"] != args.epochs * len(dataset):
                    raise RuntimeError("training sample count mismatch")
                expected_experts = block.num_experts if policy == "native" else 2
                if abs(counts["executed_experts_per_sample"] - expected_experts) > 1e-6:
                    raise RuntimeError("training expert execution does not match declared policy")
            del teacher
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        states[name] = state
        if history:
            torch.save({"state_dict": state, "layer": args.layer, "policy": policy}, args.output / f"{name}.state.pt")
        case = evaluate(state, policy, "dev", name)
        case.update(dev_ap=case["metrics"]["metrics/mAP50-95(B)"], history=history, training_routing=counts)
        report["dev"].append(case)
        save()
        del model
    orders = [
        tuple(epoch["image_order_sha256"] for epoch in case["history"]) for case in report["dev"] if case["history"]
    ]
    if any(order != orders[0] for order in orders[1:]):
        raise RuntimeError("training methods observed different image orders")
    report["matched_training_orders"] = True
    selected = choose_candidate(report["dev"], sparse_names)
    report["selected_from_dev"] = selected
    save()  # Candidate is frozen before any test metrics are obtained.
    for name in ("native", "top2", "dense_det", selected, "native_repeat"):
        source_name = "native" if name == "native_repeat" else name
        policy = "native" if source_name in {"native", "dense_det"} else "native_top2"
        report["test"].append(evaluate(states[source_name], policy, "test", name))
        save()
    first, last = report["test"][0]["metrics"], report["test"][-1]["metrics"]
    report["baseline_stable"] = all(abs(first[key] - last[key]) <= 1e-6 for key in first)
    save()
    if not report["baseline_stable"]:
        raise RuntimeError("native repeat drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
