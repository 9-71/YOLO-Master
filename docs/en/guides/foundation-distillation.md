---
comments: true
description: Configure YOLO-Master-F training-only Foundation Teacher distillation with DINOv3 and SigLIP2.
keywords: YOLO-Master-F, Foundation Teacher, DINOv3, SigLIP2, knowledge distillation, deployment
---

# Foundation Teacher Distillation

YOLO-Master-F v0.1.0-alpha adds an opt-in, training-only Foundation Teacher plane. DINOv3 and SigLIP2 can provide
spatial, relational, routing, and region-semantic supervision while the deployable student remains a normal
YOLO-Master model. Foundation teachers are never required for prediction or export.

!!! warning

    This alpha release does not make an accuracy-improvement claim. The real COCO experiments currently validate the
    training path and checkpoint/export boundaries, but do not establish an AP gain. The exact budgets and limitations
    are recorded in `reports/foundation/v0.1/f15-foundation-multitask.md`.

## Installation

The default installation does not require `transformers` for inference. Install the optional Foundation dependency
only when running teacher-backed training:

```bash
pip install -e ".[foundation]"
```

Export dependencies are separate:

```bash
pip install -e ".[export-base]"
```

Teacher weights are resolved by the Hugging Face Transformers backend or an injected local backend. Model weights are
not included in this repository and are not redistributed by YOLO-Master-F.

## Configuration

Foundation training is disabled by default. The minimal DINOv3 P4 recipe is:

```yaml
foundation_enabled: true
foundation_teacher: dinov3
foundation_backend: transformers
foundation_model: facebook/dinov3-vits16-pretrain-lvd1689m
foundation_target_levels: [p4]
foundation_loss: hybrid
foundation_loss_weight: 0.01
foundation_cosine_weight: 1.0
foundation_relation_weight: 1.0
```

Useful optional switches include:

| Setting | Purpose |
|---|---|
| `foundation_multiscale` | Enable F10 P3/P4/P5 adapters. |
| `foundation_foreground_weighting` | Weight KD tokens using GT foreground boxes. |
| `foundation_router_distill` | Distill Foundation route targets into native LatentMixture routers. |
| `foundation_teacher: siglip2` | Enable F12/F13 semantic teacher paths. |
| `foundation_teacher: multi` | Enable the F14 DINOv3 + SigLIP2 teacher router. |
| `foundation_multitask` | Enable F15 detect/segment/pose representation-transfer diagnostics. |

`foundation_backend: local` does not construct a backend implicitly. Pass an explicit `teacher_manager` from Python so
local weight loading remains deliberate and testable. Invalid teacher, task, level, and loss combinations fail early.

## Training

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26-master-n.yaml")
model.train(
    data="coco8.yaml",
    epochs=1,
    imgsz=64,
    foundation_enabled=True,
    foundation_teacher="dinov3",
    foundation_model="facebook/dinov3-vits16-pretrain-lvd1689m",
    foundation_loss="hybrid",
    foundation_loss_weight=0.01,
)
```

Training logs expose `foundation_loss`, cosine/relational components, task ratio, and (when enabled) routing or
multi-task diagnostics. The teacher is frozen and kept outside the student optimizer, DDP graph, EMA, and state dict.

## Resume and export

Training checkpoints retain JSON-safe Foundation metadata and enough projector information to rebuild the training
wrapper on resume. The teacher itself is not serialized. Export strips the wrapper before graph construction:

```python
from ultralytics import YOLO

model = YOLO("runs/train/weights/best.pt")
model.export(format="onnx", imgsz=640, nms=False)
```

The exported graph contains only student inputs and task outputs. Prediction and export do not import or execute
DINOv3, SigLIP2, or `transformers`; a student checkpoint can be loaded in an environment where the optional
Foundation dependency is absent.

## Compatibility and limitations

- Foundation functionality is opt-in; disabled mode is an exact student-only no-op.
- Teacher weights must be obtained and used under their own model-card licenses. The project AGPL-3.0 license does not
  relicense third-party weights or text encoders.
- The current real COCO evidence uses CPU, short pilot/effect-gate budgets, and does not support AP, convergence, or
  latency claims. APs/APm/APl, multi-GPU training, GPU teacher smoke, and deployment latency benchmarking remain open
  release work.
- The teacher increases training memory and wall time. It is absent from the inference graph, so deployment cost must
  be measured from the stripped student/export artifact.
- SigLIP2 region-semantic distillation requires compatible text prototypes/prompts and is intended for training-time
  supervision, not automatic open-vocabulary deployment in this alpha.

## Verification artifacts

- `reports/foundation/v0.1/f07-real-dinov3-validation.md` — real DINOv3 validation
- `reports/foundation/v0.1/f14-foundation-multirouter.md` — multi-foundation router validation
- `reports/foundation/v0.1/f15-foundation-multitask.md` — multi-task report
- `reports/foundation/v0.1/foundation-alpha-completion.json` — Alpha completion audit
