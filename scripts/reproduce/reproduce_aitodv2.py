#!/usr/bin/env python3
"""Reproduce YOLO-Master-v0.1-N and YOLO-Master-EsMoE-N baselines on AI-TOD-v2.

AI-TOD-v2 (aerial tiny-object detection: 8 classes, ~800px crops, mean object ~12px),
built-in config AI-TOD-v2.yaml. Same two nano variants as VisDrone/SKU-110K. By default
the models are reproduced as-is (EsMoE-N keeps its sparse eval, which collapses mAP). Add
--no-sparse-eval to opt into the corrected dense evaluation for EsMoE-N (train==eval);
v0.1-N is unaffected.

AI-TOD is a tiny-object dataset -- train at the native crop size with --imgsz 800.

Examples:
    python scripts/reproduce/reproduce_aitodv2.py --check-build
    python scripts/reproduce/reproduce_aitodv2.py --imgsz 800 --epochs 300 --batch 64                  # as-is
    python scripts/reproduce/reproduce_aitodv2.py --imgsz 800 --model EsMoE-N --no-sparse-eval          # corrected
    python scripts/reproduce/reproduce_aitodv2.py --imgsz 800 --model v0.1-N --no-wandb
    python scripts/reproduce/reproduce_aitodv2.py --wandb-project my-proj --wandb-mode offline
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _reproduce_common import MODELS, VARIANTS, DatasetSpec, run_dataset  # noqa: E402

DATASET = DatasetSpec(
    name="AI-TOD-v2",
    data="AI-TOD-v2.yaml",
    project="runs/reproduce/aitodv2",
)


if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET, models=MODELS + VARIANTS))
