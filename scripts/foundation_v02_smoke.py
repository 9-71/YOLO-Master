"""F09–F15 v0.2.x smoke: real foundation teachers on real/tiny data, engineering gates.

Usage: python3 scripts/foundation_v02_smoke.py {f09|f10|f11|f12|f13|f14|f15}

Verifies end-to-end on real data (not synthetic/contract level):
- f09: GT foreground-aware weighting path (interior 1.5 / boundary 1.0 / background 0.25)
- f10: multi-scale P3/P4/P5 distillation with per-level adapters
- f11: DINOv3-guided LatentMixture image-level routing KD
- f12: SigLIP2 as feature-KD teacher
- f13: positive-region SigLIP2 semantic distillation (feature KD off)
- f14: multi-foundation teacher router (DINOv3 spatial + SigLIP2 semantic)
- f15: MultiTask foundation transfer on a tiny COCO-format detect+segment fixture

Post-run assertions: all foundation_* telemetry finite; per-group engagement
signals present; saved checkpoint contains no teacher parameters.
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Running a script puts scripts/ (not the repo root) at sys.path[0], which would
# resolve `ultralytics` to a stale editable install in another workspace.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DINOV3 = "Tooony133/dinov3-vits16-pretrain-lvd1689m"
SIGLIP2 = "google/siglip2-base-patch16-512"

COMMON = dict(
    data=str(ROOT / "ultralytics/cfg/datasets/coco8.yaml"),
    epochs=2,
    imgsz=128,
    batch=4,
    device="mps",
    workers=0,
    amp=False,
    optimizer="SGD",
    lr0=0.01,
    seed=0,
    val=True,
    plots=False,
    project=str(ROOT / "runs/foundation"),
    exist_ok=True,
    verbose=False,
)

TEACHER_DINO = dict(
    foundation_enabled=True,
    foundation_teacher="dinov3",
    foundation_backend="transformers",
    foundation_model=DINOV3,
    foundation_teacher_device="mps",
    foundation_teacher_dtype="fp32",
    foundation_align_dim=32,
    foundation_loss="hybrid",
    foundation_relation_mode="sampled",
    foundation_relation_samples=16,
    foundation_loss_weight=0.05,
)

GROUPS = {
    "f09": dict(
        model="yolo26n.yaml",
        overrides=dict(
            foundation_target_levels=["p4"],
            foundation_foreground_weighting=True,
            foundation_foreground_weight=1.5,
            foundation_boundary_weight=1.0,
            foundation_background_weight=0.25,
        ),
    ),
    "f10": dict(
        model="yolo26n.yaml",
        overrides=dict(
            foundation_target_levels=["p3", "p4", "p5"],
            foundation_multiscale=True,
        ),
    ),
    "f11": dict(
        model="yolo26-master-latent-n.yaml",
        epochs=1,  # repo recipe is 1-epoch; latent model + teacher is ~2 min/epoch on MPS
        val=False,
        overrides=dict(
            foundation_target_levels=["p4"],
            foundation_router_distill=True,
            foundation_router_loss_weight=0.02,
            foundation_router_temperature=2.0,
        ),
    ),
    "f12": dict(
        model="yolo26n.yaml",
        teacher=dict(TEACHER_DINO, foundation_teacher="siglip2", foundation_model=SIGLIP2),
        overrides=dict(foundation_target_levels=["p4"]),
    ),
    "f13": dict(
        model="yolo11n.yaml",
        imgsz=256,
        teacher=dict(TEACHER_DINO, foundation_teacher="siglip2", foundation_model=SIGLIP2),
        overrides=dict(
            foundation_loss_weight=0.0,  # feature KD off; semantic branch only
            foundation_semantic_distill=True,
            foundation_semantic_loss_weight=0.1,
            foundation_semantic_text_weight=1.0,
            foundation_semantic_image_weight=1.0,
            foundation_semantic_temperature=0.07,
            foundation_semantic_prompt_template="a photo of a {class_name}",
        ),
    ),
    "f14": dict(
        model="yolo26-master-latent-n.yaml",
        epochs=1,  # same runtime constraint as f11
        val=False,
        teacher=dict(
            TEACHER_DINO,
            foundation_teacher="multi",
            foundation_dinov3_model=DINOV3,
            foundation_siglip2_model=SIGLIP2,
        ),
        overrides=dict(
            foundation_target_levels=["p4"],
            foundation_router_distill=True,
            foundation_router_loss_weight=0.02,
            foundation_router_temperature=2.0,
            foundation_router_teachers=["dinov3", "siglip2"],
            foundation_router_native_state=True,
        ),
    ),
    "f15": dict(
        model="yolo26-master-mt-n.yaml",
        imgsz=64,
        batch=2,
        val=False,
        overrides=dict(
            foundation_multitask=True,
            foundation_target_levels=["p4"],
            foundation_loss_weight=0.1,
            foundation_multitask_negative_transfer_threshold=4.0,
        ),
    ),
}


def build_mt_fixture(root: Path) -> str:
    """Create a minimal COCO-format detect+segment multi-task dataset (4 images, 1 box each)."""
    img_dir = root / "images" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for i in range(4):
        cv2.imwrite(str(img_dir / f"img{i}.jpg"), np.full((32, 32, 3), i * 40, dtype=np.uint8))
        images.append({"id": i, "file_name": f"img{i}.jpg", "width": 32, "height": 32})
        annotations.append(
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [4.0, 4.0, 16.0, 16.0],
                "area": 256.0,
                "iscrowd": 0,
                "segmentation": [[4.0, 4.0, 20.0, 4.0, 20.0, 20.0, 4.0, 20.0]],
            }
        )
    (root / "instances.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "person"}]})
    )
    yaml_file = root / "data.yaml"
    yaml_file.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/train",
                "names": {0: "person"},
                "multitask_format": "coco",
                "tasks": ["detect", "segment"],
                "train_instances": "instances.json",
                "val_instances": "instances.json",
                "kpt_shape": [17, 3],
            }
        )
    )
    return str(yaml_file)


def check_telemetry(run_dir: Path, group: str) -> dict:
    csv_path = run_dir / "results.csv"
    assert csv_path.exists(), f"missing {csv_path}"
    rows = list(csv.DictReader(csv_path.open()))
    assert rows, "empty results.csv"
    foundation_cols = [c for c in rows[0] if c.strip().startswith("train/foundation")]
    assert foundation_cols, "no foundation telemetry columns"
    bad = []
    for row in rows:
        for col in foundation_cols:
            value = float(row[col])
            if not math.isfinite(value):
                bad.append((col, value))
    assert not bad, f"non-finite telemetry: {bad}"
    last = {k.strip(): float(v) for k, v in rows[-1].items()}
    if group == "f09":
        assert last["train/foundation_foreground_enabled"] == 1.0, "foreground weighting not engaged"
        mean_w = last["train/foundation_foreground_mean_weight"]
        assert 0.25 < mean_w <= 1.5, f"foreground mean weight out of range: {mean_w}"
    if group == "f10":
        for level in ("p3", "p4", "p5"):
            col = f"train/foundation_{level}_loss"
            assert col in last, f"missing per-level column {col}"
            assert math.isfinite(last[col]), f"non-finite {col}"
    if group in ("f11", "f14"):
        modules = last.get("train/foundation_router_modules", 0.0)
        assert modules >= 1.0, f"no LatentMixture router engaged: {modules}"
        assert last["train/foundation_router_kl"] >= 0.0, "router KL negative"
    if group == "f13":
        assert last["train/foundation_semantic_enabled"] == 1.0, "semantic branch not engaged"
        assert last["train/foundation_semantic_regions"] > 0.0, "no positive regions distilled"
    if group == "f15":
        supervised = last.get("train/foundation_multitask_supervised_tasks", 0.0)
        assert supervised >= 2.0, f"F15 gate needs >=2 supervised tasks, got {supervised}"
    return last


def check_checkpoint(run_dir: Path) -> None:
    last_pt = run_dir / "weights/last.pt"
    assert last_pt.exists(), f"missing {last_pt}"
    ckpt = torch.load(last_pt, map_location="cpu", weights_only=False)
    # In-training saves keep model=None (weights live in `ema`); the final
    # strip_optimizer pass (val=True) writes a student-only `model`.
    model = ckpt.get("model") or ckpt.get("ema")
    assert model is not None, "checkpoint carries neither model nor ema weights"
    state = model.state_dict() if hasattr(model, "state_dict") else model
    # `_projector.teacher_proj` is the trainable alignment adapter (legit);
    # the actual teacher backbone/manager must never be serialized.
    banned_fragments = ("teacher_manager", "_route_teachers", "teacher_model", "dinov3", "siglip")
    teacher_keys = [k for k in state if any(fragment in k.lower() for fragment in banned_fragments)]
    assert not teacher_keys, f"teacher leaked into checkpoint: {teacher_keys[:5]}"


def main() -> None:
    group = sys.argv[1]
    assert group in GROUPS, group
    spec = GROUPS[group]
    name = f"v02-smoke-{group}-coco8"
    cfg = dict(COMMON)
    for key in ("imgsz", "batch", "val", "epochs"):
        if key in spec:
            cfg[key] = spec[key]
    cfg.update(spec.get("teacher", TEACHER_DINO))
    cfg.update(spec["overrides"])
    cfg["name"] = name
    if group == "f15":
        cfg["data"] = build_mt_fixture(ROOT / "datasets/coco4-mt-fixture")
        cfg["task"] = "multitask"
    print(f"[v02-smoke] {group}: training {cfg['epochs']} epochs, model={spec['model']}", flush=True)
    YOLO(spec["model"]).train(**cfg)
    run_dir = ROOT / "runs/foundation" / name
    last = check_telemetry(run_dir, group)
    check_checkpoint(run_dir)
    interesting = {k: round(v, 4) for k, v in last.items() if "foundation" in k}
    print(f"[v02-smoke] {group} PASS. final-epoch telemetry: {interesting}", flush=True)


if __name__ == "__main__":
    main()
