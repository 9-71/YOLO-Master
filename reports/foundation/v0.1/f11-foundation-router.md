# F11 Foundation Teacher Router

日期：2026-08-13（Asia/Shanghai）

## Scope

F11 在 F10 的 DINOv3 训练期蒸馏之上，首选并只接入 `LatentMixture` image-level router：

- DINOv3 pooled embedding、P4 spatial mean/std 组成 teacher route summary。
- 每个 LatentMixture 使用冻结、未注册、确定性重建的 `FoundationTeacherRouter` 产生 teacher logits。
- 学生直接复用原生 router 的 graph-connected logits/latent summary。
- 路由损失为 `T² * KL(softmax(z_teacher/T) || softmax(z_student/T))`，通过统一 `publish_aux_loss(..., kind="foundation_route")` 发布。
- 不修改 MoE/MoA/MoT，不引入 SigLIP2、多 teacher router，也不改 YOLO model YAML 解析。

## 配置

```yaml
foundation_router_distill: true
foundation_router_loss_weight: 0.02
foundation_router_temperature: 2.0
```

默认 `foundation_router_distill=false`、权重为 `0.0`，F00–F10 的默认行为和指标键保持不变。

Recipe：[f11-foundation-router-coco8-dinov3.yaml](../../../ultralytics/cfg/experiments/foundation/f11-foundation-router-coco8-dinov3.yaml)

## Real DINOv3 smoke

运行：

```bash
PYTHONPATH=/Users/gatilin/PycharmProjects/YOLO-Master-v260810-latest \
HF_ENDPOINT=https://hf-mirror.com \
python - <<'PY'
from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/26/yolo26-master-latent-n.yaml")
model.train(cfg="ultralytics/cfg/experiments/foundation/f11-foundation-router-coco8-dinov3.yaml")
PY
```

输出目录：

`/Users/gatilin/PycharmProjects/YOLO-Master-v260720/runs/detect/runs/foundation/f11-foundation-router-coco8-dinov3`

结果：

- `train/foundation_loss=0.173859`
- `train/foundation_router_loss=0.000739797`
- `train/foundation_router_kl=0.0184949`
- `train/foundation_router_modules=3`
- teacher/student route entropy：`1.38167 / 1.38629`
- 1 epoch、COCO8、64px、CPU、真实 DINOv3 权重，loss 全部 finite。

## Checkpoint / export boundary

- `foundation` metadata 记录 `router_kind=latent_mixture_image_level`、temperature、loss weight 与 3 个静态 LatentMixture route specs（student dim 64/128/256，均 4 experts）。
- route teacher heads 不进入 `state_dict`、optimizer、EMA 或导出图；deepcopy/resume 时依据静态 spec 和稳定 seed 重建。
- 对 smoke checkpoint 执行 `strip_optimizer` 后得到纯 `DetectionModel`，3 个 LatentMixture 保留，route teacher keys=0，forward hooks=0。

## Verification

- F11 routing contract：7 passed
- Foundation/F07/F10/config/MoE routing 回归：通过
- `ruff check`、`ruff format --check`、`compileall`：通过
- 真实 DINOv3 F11 smoke：通过
