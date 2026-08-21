# YOLO-Master-F v0.1.0-alpha — Release Notes

> **Status: ALPHA — research preview. No release-level accuracy claim is made.**
> Base: `YOLO-Master-v26.08` (Ultralytics 8.4.101) · Baseline record: `reports/foundation/v0.1/baseline.json` (kept local per repo policy)

## What this is

A **training-only Foundation Teacher subsystem** for YOLO-Master. Foundation models
(DINOv3, SigLIP2) act as representation/routing teachers during training; the deployed
student is architecturally identical to the v26.08 baseline and carries **zero teacher
dependency** at inference and export time.

## Scope delivered (F00–F08, alpha contract)

- Opt-in configuration boundary (`foundation_enabled: false` default is a global no-op;
  invalid combinations fail early with actionable errors).
- `FoundationTeacher` protocol + DINOv3 backend (transformers or local weights), frozen,
  with independent dtype/device resolution.
- Student P4 feature capture (`StudentFeatureTap`) and P4 alignment projector.
- Cosine and sampled-relational KD losses (hybrid mode supported).
- `FoundationDistillationModel` training wrapper wired into the trainer.
- Checkpoint contract: teacher parameters are never serialized into student checkpoints;
  student checkpoints load and predict in environments **without** `transformers`.
- Export contract: teachers never enter export graphs (student-only ONNX export verified).
- Loss-weight schedules: `constant` (default, unchanged behavior) and `gate_decay`
  (cosine-band gating with warmup floor + late linear decay to zero; gate EMA persisted
  across checkpoint resume).
- Raw telemetry columns (`foundation_cosine_raw`, `foundation_relational_raw`,
  `foundation_effective_weight`, `foundation_task_ratio`) for cross-configuration
  comparison.

## Beyond alpha (F09–F15, contract-tested)

Foreground-aware KD (GT box weighting: interior/boundary/background), multi-scale
P3/P4/P5 distillation with per-level adapters, foundation teacher router + routing KD,
SigLIP2 semantic teacher, region-semantic distillation (YOLOE), multi-foundation router,
and MultiTask foundation distillation — all contract-tested. F09/F10 additionally
verified end-to-end on real data (coco8 + real DINOv3, MPS) via
`scripts/foundation_v02_smoke.py`: finite losses, per-level telemetry, foreground
weighting engaged, teacher-free checkpoints.

## Validation evidence

| Gate | Result |
|------|--------|
| v26.08 focused release gate (11 test files) | 234 passed |
| Foundation suite (20 test files) | 169 passed |
| coco8/coco128 real-data smoke training (DINOv3, MPS) | end-to-end pass, finite telemetry |
| Teacher gradient isolation / checkpoint exclusion / export exclusion | contract tests pass |

## Honest research status

- The distillation pipeline is **functionally verified on real data** and the alignment
  signal is active (cosine_raw decreases monotonically under sufficient optimization
  strength).
- Accuracy gains observed at coco128 smoke scale (up to +3.6% relative mAP50-95 with the
  `gate_decay` schedule) are **within seed noise** (3-seed mean +0.0042 ± 0.0154) and do
  **not** constitute a validated improvement. Over-alignment is a real failure mode;
  keep `foundation_loss_weight ≤ 0.2` until cosine alignment is established.
- Paper-grade claims require a lower-variance testbed (VOC-scale or COCO subset) — see
  the experiment gates in the v0.1 implementation plan.

## Known limitations

- Teacher weights (DINOv3/SigLIP2) are **not** redistributed; users must obtain them
  under their own licenses. Offline environments need a local HF cache or stub teacher.
- `foundation_enabled` and `distill_model` (YOLO-to-YOLO KD) are mutually exclusive in
  this alpha.
- `compile=True` with Foundation KD follows the same restrictions as standard KD.
- AGPL-3.0 project boundaries apply.

## Upgrade & usage

See `docs/en/guides/foundation-distillation.md` for the user guide and
`ultralytics/cfg/experiments/foundation/f08–f15*.yaml` for recipe configs.
