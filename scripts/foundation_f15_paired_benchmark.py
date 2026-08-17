"""Run a deterministic, paired F15 Foundation-vs-baseline effect benchmark.

This benchmark intentionally measures training signals rather than claiming
accuracy: both branches start from identical student weights and consume the
same synthetic three-task batch sequence.  Use a real COCO trainer run for
AP/mask/pose conclusions.

Example::

    python scripts/foundation_f15_paired_benchmark.py \
        --teacher-model Tooony133/dinov3-vits16-pretrain-lvd1689m \
        --steps 6 --output reports/foundation/v0.1/f15-paired-benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Make direct ``python scripts/...`` invocation resolve this checkout rather
# than an installed Ultralytics package from another environment.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.nn.foundation.teachers.dinov3 import DINOv3Teacher  # noqa: E402
from ultralytics.nn.foundation_distill_model import FoundationDistillationModel  # noqa: E402
from ultralytics.nn.tasks import MultiTaskModel  # noqa: E402


MODEL_CFG = "ultralytics/cfg/models/26/yolo26-master-mt-n.yaml"
DEFAULT_TEACHER = "Tooony133/dinov3-vits16-pretrain-lvd1689m"
TASKS = ["detect", "segment", "pose"]


def _batch(step: int, *, size: int = 64, batch_size: int = 2, seed: int = 20260814) -> dict[str, torch.Tensor]:
    """Create one deterministic, fully supervised synthetic detect/segment/pose batch."""
    generator = torch.Generator().manual_seed(int(seed) + int(step))
    return {
        "img": torch.rand(batch_size, 3, size, size, generator=generator),
        "batch_idx": torch.arange(batch_size, dtype=torch.long),
        "cls": torch.arange(batch_size, dtype=torch.float32).remainder(3).reshape(-1, 1),
        "bboxes": torch.tensor([[0.2, 0.2, 0.5, 0.5], [0.5, 0.5, 0.3, 0.3]], dtype=torch.float32)[:batch_size],
        "masks": torch.randint(0, 2, (batch_size, size, size), generator=generator, dtype=torch.uint8),
        "sem_masks": torch.randint(0, 3, (batch_size, size, size), generator=generator),
        "keypoints": torch.tensor([[[0.2, 0.2, 2.0]] * 17, [[0.5, 0.5, 2.0]] * 17], dtype=torch.float32)[:batch_size],
    }


def _config(*, foundation: bool, teacher_model: str, align_dim: int, foundation_loss_weight: float = 0.05) -> object:
    """Build the same training config for both paired branches."""
    overrides = {
        "task": "multitask",
        "mode": "train",
        "imgsz": 64,
        "batch": 2,
        "device": "cpu",
        "foundation_enabled": foundation,
        "foundation_multitask": foundation,
        "foundation_multitask_tasks": TASKS,
        "foundation_teacher": "dinov3",
        "foundation_backend": "local",
        "foundation_model": teacher_model,
        "foundation_teacher_dtype": "fp32",
        "foundation_teacher_device": "cpu",
        "foundation_target_levels": ["p4"],
        "foundation_align_dim": align_dim,
        "foundation_loss": "hybrid",
        "foundation_relation_mode": "sampled",
        "foundation_relation_samples": 16,
        "foundation_loss_weight": float(foundation_loss_weight),
    }
    return get_cfg(overrides=overrides)


def _build_student(config: object, state: dict[str, torch.Tensor] | None = None) -> MultiTaskModel:
    """Build a CPU student and optionally restore an identical initial state."""
    student = MultiTaskModel(MODEL_CFG, nc=3, verbose=False)
    if state is not None:
        student.load_state_dict(state, strict=True)
    student.args = config
    return student


def _p4_grad_norm(model: torch.nn.Module) -> float:
    """Measure gradient flow through the late shared feature path."""
    total = 0.0
    for name, parameter in model.named_parameters():
        if "model.6" in name and parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm())
    return total


def _run(
    model: torch.nn.Module,
    *,
    wrapped: bool,
    steps: int,
    seed: int,
    batch_seed: int,
) -> list[dict[str, object]]:
    """Run one branch over the paired batch sequence."""
    torch.manual_seed(seed)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    rows = []
    for step in range(steps):
        batch = _batch(step, seed=batch_seed)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        if wrapped:
            output, items = model(batch)
            loss = output.sum()
            task_loss = float(output[0].detach())
            foundation_loss = float(output[-1].detach())
            metrics = model.foundation_metrics()
        else:
            predictions = model(batch["img"])
            task_loss_tensor, items = model.loss(batch, predictions)
            loss = task_loss_tensor
            task_loss = float(task_loss_tensor.detach())
            foundation_loss = 0.0
            metrics = {}
        loss.backward()
        grad_norm = _p4_grad_norm(model)
        optimizer.step()
        rows.append(
            {
                "step": step,
                "task_loss": round(task_loss, 6),
                "foundation_loss": round(foundation_loss, 6),
                "loss_items": [round(float(value), 6) for value in items[:9].detach().cpu()],
                "supervised_tasks": float(metrics.get("foundation_multitask_supervised_tasks", 3.0)),
                "task_router_entropy": round(float(metrics.get("foundation_multitask_task_router_entropy", 0.0)), 6),
                "p4_grad_norm": round(grad_norm, 6),
                "wall_s": round(time.perf_counter() - started, 4),
            }
        )
    return rows


def run_benchmark(
    teacher_model: str,
    steps: int,
    align_dim: int,
    *,
    seed: int = 20260813,
    foundation_loss_weight: float = 0.05,
) -> dict[str, object]:
    """Run the paired baseline/Foundation benchmark and summarize deltas."""
    torch.manual_seed(int(seed))
    baseline_config = _config(
        foundation=False,
        teacher_model=teacher_model,
        align_dim=align_dim,
        foundation_loss_weight=foundation_loss_weight,
    )
    baseline = _build_student(baseline_config)
    initial_state = {key: value.detach().clone() for key, value in baseline.state_dict().items()}

    foundation_config = _config(
        foundation=True,
        teacher_model=teacher_model,
        align_dim=align_dim,
        foundation_loss_weight=foundation_loss_weight,
    )
    foundation_student = _build_student(foundation_config, initial_state)
    teacher = DINOv3Teacher(model_id=teacher_model, device="cpu", dtype="fp32", local_files_only=True)
    foundation = FoundationDistillationModel(foundation_student, teacher, foundation_config)

    batch_seed = int(seed) + 1
    baseline_rows = _run(baseline, wrapped=False, steps=steps, seed=int(seed) + 2, batch_seed=batch_seed)
    foundation_rows = _run(foundation, wrapped=True, steps=steps, seed=int(seed) + 2, batch_seed=batch_seed)
    return {
        "schema_version": 1,
        "benchmark": "f15_paired_training_signal",
        "teacher_model": teacher_model,
        "student_model": MODEL_CFG,
        "tasks": TASKS,
        "steps": steps,
        "seed": int(seed),
        "foundation_loss_weight": float(foundation_loss_weight),
        "batch_seed": batch_seed,
        "device": "cpu",
        "synthetic_batch": True,
        "real_accuracy_claim": False,
        "baseline": baseline_rows,
        "foundation": foundation_rows,
        "summary": {
            "baseline_task_delta_first_last": round(baseline_rows[-1]["task_loss"] - baseline_rows[0]["task_loss"], 6),
            "foundation_task_delta_first_last": round(
                foundation_rows[-1]["task_loss"] - foundation_rows[0]["task_loss"], 6
            ),
            "foundation_kd_delta_first_last": round(
                foundation_rows[-1]["foundation_loss"] - foundation_rows[0]["foundation_loss"], 6
            ),
            "baseline_mean_step_s": round(sum(row["wall_s"] for row in baseline_rows) / steps, 6),
            "foundation_mean_step_s": round(sum(row["wall_s"] for row in foundation_rows) / steps, 6),
            "foundation_step_overhead_ratio": round(
                sum(row["wall_s"] for row in foundation_rows) / max(sum(row["wall_s"] for row in baseline_rows), 1e-12)
                - 1.0,
                6,
            ),
            "foundation_supervised_task_gate": min(row["supervised_tasks"] for row in foundation_rows) >= 2,
            "foundation_nonzero_kd_gate": all(row["foundation_loss"] > 0 for row in foundation_rows),
            "foundation_max_task_router_entropy": round(max(row["task_router_entropy"] for row in foundation_rows), 6),
            "foundation_p4_grad_min": round(min(row["p4_grad_norm"] for row in foundation_rows), 6),
            "foundation_router_entropy_first_last": [
                foundation_rows[0]["task_router_entropy"],
                foundation_rows[-1]["task_router_entropy"],
            ],
        },
    }


def main() -> None:
    """Parse arguments and persist a JSON benchmark record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", default=os.environ.get("YOLO_MASTER_DINOV3_LOCAL", DEFAULT_TEACHER))
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--align-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--foundation-loss-weight", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("reports/foundation/v0.1/f15-paired-benchmark.json"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.foundation_loss_weight < 0:
        parser.error("--foundation-loss-weight must be non-negative")
    result = run_benchmark(
        args.teacher_model,
        args.steps,
        args.align_dim,
        seed=args.seed,
        foundation_loss_weight=args.foundation_loss_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
