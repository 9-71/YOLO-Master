# F13 — Region Semantic Distillation / YOLOE

## Scope

F13 adds an opt-in, training-only semantic objective on top of the F12 SigLIP2 teacher. The implementation uses the
native task-aligned assigner to select positive P4 locations, then maps each selected student region into the frozen
SigLIP2 semantic space:

\[
L_{sem}=L_{region-text}+L_{region-image}.
\]

Background locations are intentionally excluded from the first alpha implementation.

## Implementation

- `ultralytics/nn/foundation/semantic.py`
  - positive-region pooling from `fg_mask`/`target_gt_idx`;
  - `RegionSemanticProjector`;
  - region-text cross entropy and region-image cosine losses.
- `FoundationDistillationModel`
  - resolves class prompts from `foundation_semantic_prompts` or `model.names`;
  - caches text prototypes outside `state_dict`;
  - adds semantic metrics and checkpoint metadata;
  - restores the semantic projector on resume and strips it on export.
- Configuration is disabled by default. F13 currently requires `foundation_teacher=siglip2` and does not implement
  the F14 multi-foundation router.

## Validation

```text
pytest -q tests/test_foundation_semantic.py tests/test_foundation_distill_model.py tests/test_foundation_checkpoint.py
20 passed

pytest -q tests/test_foundation_*.py
134 passed
```

真实 `google/siglip2-base-patch16-512` 本地权重 smoke 也已通过：P4 dense `(1,768,32,32)`、semantic `(1,768)`、text prototype `(2,768)`；COCO8 单 epoch CPU 训练完成，日志中出现非零 `foundation` loss，且 stripped `best.pt` 恢复为纯 `DetectionModel`，无 semantic/projector 参数。

The semantic contracts cover positive-only pooling, empty-positive graph-connected zeros, projector gradients,
prototype shape, deterministic text caching, and the training-only teacher boundary. The existing Foundation F06–F12
regression suite remains green in the same run.

Recipe: `ultralytics/cfg/experiments/foundation/f13-foundation-semantic-coco8.yaml`.
