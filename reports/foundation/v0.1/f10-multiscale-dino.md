# F10 Multi-scale DINO pyramid

日期：2026-08-13（Asia/Shanghai）

## Scope

F10 在 F08/F09 的 P4 Foundation KD 上增加可选的多尺度蒸馏。采用方案 A：同一 DINOv3 final spatial map，分别通过独立的 P3/P4/P5 alignment adapter 对齐到 student 特征，不把单一 projector 粗暴复用到多个层级，也不进入 F11 routing。

开启方式：

```yaml
foundation_multiscale: true
foundation_target_levels: [p3, p4, p5]
```

默认 `foundation_multiscale=false`，单尺度 `p4` 路径保持兼容。

## Runtime design

- 每个目标层级拥有独立 `StudentFeatureTap`。
- 每个目标层级拥有独立 `P4AlignmentProjector` adapter，student-side 可训练，teacher-side frozen。
- 每个 batch 只调用一次 teacher `encode()`；同一 DINO spatial map 分别与 P3/P4/P5 做空间对齐。
- 各尺度 KD 分量取均值后乘 Foundation loss weight。
- 多尺度指标新增：`train/foundation_p3_loss`、`train/foundation_p4_loss`、`train/foundation_p5_loss`。
- checkpoint metadata 记录 `multiscale`、目标层级及每层 student/teacher channel；teacher 仍不进入 checkpoint module tree。

## Recipe

[f10-foundation-multiscale-coco8-dinov3.yaml](../../../ultralytics/cfg/experiments/foundation/f10-foundation-multiscale-coco8-dinov3.yaml)

使用 CPU、COCO8、64px、batch=2、1 epoch 的真实 DINOv3 smoke。

## Real smoke result

输出目录：

`/Users/gatilin/PycharmProjects/YOLO-Master-v260720/runs/detect/runs/foundation/f10-multiscale-smoke`

结果：

- `train/foundation_loss=0.177667`
- `train/foundation_p3_loss=0.184057`
- `train/foundation_p4_loss=0.171007`
- `train/foundation_p5_loss=0.177937`
- `train/foundation_foreground_enabled=0`
- loss 全部 finite
- 三个 teacher-side projector 参数全部 `requires_grad=False`
- metadata：student channels `{p3:64,p4:128,p5:256}`，teacher channels `{p3:384,p4:384,p5:384}`

## Verification

- Foundation 全套测试：111 passed
- F10 wrapper/config/recipe/checkpoint 定向测试：42 passed
- 真实 DINOv3 多尺度 smoke：通过
- F07 deployment strip/export contract：真实 F10 checkpoint strip 后为纯 `DetectionModel`，projector keys=0，hooks=0
- Ruff check、Ruff format、compileall、`git diff --check`：通过
- F11 teacher router 不在本阶段范围内
