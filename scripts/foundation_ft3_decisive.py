"""F08 decisive experiment: 15-epoch fine-tune, baseline vs DINOv3 distillation (w=0.1).

Usage: python3 scripts/foundation_ft3_decisive.py {baseline|distill}

Designed for segmented execution: each invocation runs until interrupted
(foreground timeout), then re-invoking resumes from last.pt.

Notes:
- optimizer="SGD" is explicit so lr0=0.01 is honored (optimizer="auto" silently
  overrides lr0 with AdamW lr~1e-4, which stalled cosine alignment in F08).
- EPOCHS=15 chosen over 30 because the host is under heavy load (~5 min/epoch);
  gradient-injection probes showed cosine alignment responds within tens of
  steps at lr=0.01, so 15 epochs (120 steps) is sufficient to resolve the trend.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Running a script puts scripts/ (not the repo root) at sys.path[0], which would
# resolve `ultralytics` to the stale editable install in the v260720 workspace.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EPOCHS = 15
COMMON = dict(
    data=str(ROOT / "ultralytics/cfg/datasets/coco128.yaml"),
    epochs=EPOCHS,
    imgsz=256,
    batch=16,
    device="mps",
    workers=0,
    amp=False,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    seed=0,
    project=str(ROOT / "runs/foundation"),
    exist_ok=True,
    verbose=False,
)

DISTILL = dict(
    foundation_enabled=True,
    foundation_teacher="dinov3",
    foundation_backend="transformers",
    foundation_model="Tooony133/dinov3-vits16-pretrain-lvd1689m",
    foundation_teacher_device="mps",
    foundation_align_dim=32,
    foundation_loss="hybrid",
    foundation_relation_mode="sampled",
    foundation_relation_samples=16,
    foundation_loss_weight=0.1,
)

# "gated" tag: cosine-gated ramp-in + late decay schedule (defaults: gate at cosine EMA 1.0,
# width 0.05, warmup floor 0.2, linear decay to zero over the final 30% of epochs).
GATED = dict(
    foundation_weight_schedule="gate_decay",
)


def main() -> None:
    group = sys.argv[1]
    assert group in ("baseline", "distill"), group
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else EPOCHS
    tag = sys.argv[3] if len(sys.argv) > 3 else None
    name = f"coco128-ft3-{group}{'-' + tag if tag else ''}-e{epochs}"
    last = ROOT / "runs/foundation" / name / "weights/last.pt"
    if last.exists():
        print(f"[ft3] resuming {name} from {last}", flush=True)
        YOLO(str(last)).train(resume=True)
        return
    cfg = dict(COMMON, name=name, epochs=epochs)
    if group == "distill":
        cfg.update(DISTILL)
        if tag == "gated":
            cfg.update(GATED)
    print(f"[ft3] starting {name} (epochs={epochs}, SGD lr0=0.01)", flush=True)
    YOLO(str(ROOT / "weights/yolo26n.pt")).train(**cfg)


if __name__ == "__main__":
    main()
