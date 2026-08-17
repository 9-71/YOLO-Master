# F09 Foreground-aware DINO distillation

日期：2026-08-13（Asia/Shanghai）

## Scope

F09 在 F08 P4 Foundation KD 基础上增加可选的 GT foreground weighting。仅检测 batch 的归一化 `xywh` boxes 参与权重生成，不修改 YOLO task loss、student graph、teacher graph 或部署模型。

默认值保持关闭：`foundation_foreground_weighting=False`。

## Weight map

- GT box interior：`foundation_foreground_weight=1.5`
- one-cell dilated boundary：`foundation_boundary_weight=1.0`
- background：`foundation_background_weight=0.25`

权重按 student P4 网格 token center 计算，多个 box 取最大权重；无目标或缺失 target 字段时使用全背景权重。权重 detached，不参与反向传播。

## Loss integration

- cosine KD：按 token weight 加权平均
- relational KD：按 token-pair weight 加权平均
- l2 KD：按 token weight 加权平均
- hybrid KD：沿用 cosine/relational 分量和原有指标记录

新增训练指标：

- `foundation_foreground_enabled`
- `foundation_foreground_mean_weight`

checkpoint metadata 同步记录前景 weighting 开关和三个权重值，部署 strip 不包含该训练模块。

## Verification

- Foundation 全套测试：106 passed
- F09 定向测试（loss/config/wrapper/checkpoint/recipe）：57 passed
- Ruff check、Ruff format、compileall、`git diff --check`：通过
- 真实 DINOv3 + GT foreground weighting smoke：成功，CPU、64px、COCO8、1 epoch
- smoke `results.csv`：`train/foundation_foreground_enabled=1`，`train/foundation_foreground_mean_weight=1.33984`，Foundation loss `0.173706`
- smoke checkpoint metadata：记录 `foreground_weighting=true` 及三个权重值；teacher 仍不进入部署模型
- F10 多尺度、F11 routing distillation 不在本阶段范围内。
