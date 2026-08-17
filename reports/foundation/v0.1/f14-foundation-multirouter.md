# F14 Multi-Foundation Router

## 目标

让 Foundation Teacher 根据当前样本估计所需计算：DINOv3 提供空间复杂度，SigLIP2 提供语义不确定性，YOLO LatentMixture 提供 native visual state。冻结的 `FoundationTeacherRouter` 将这些摘要映射为每个学生路由层的 expert target，并以温度缩放 KL 蒸馏给学生路由器。

## 训练契约

- `foundation_teacher: multi` 严格要求同时提供 `dinov3` 与 `siglip2`，缺失任一教师或 SigLIP2 semantic 输出会显式报错。
- 两个 teacher、teacher router 和路由摘要均为 training-only，不进入 wrapper `state_dict`、optimizer、EMA、DDP 或 export graph。
- `foundation_router_native_state` 控制是否把 detached YOLO routing summary 作为 teacher router context；学生 logits 始终保留梯度。
- F11 单 DINO teacher 路径仍使用 `latent_mixture_image_level`，不改变既有行为；F13 semantic distillation 仍限定 `foundation_teacher: siglip2`。

## 验证记录

F14 定向测试覆盖摘要 shape/有限性、双 teacher 冻结与 state_dict 隔离、student route 梯度、缺失能力报错、metadata/resume、export strip，以及 F11 回归。recipe 使用 COCO8 单 epoch CPU smoke。

离线/真实权重结果：

- `pytest -q tests/test_foundation_*.py`：142 passed。
- Agent Skill quick suite：36 passed。
- 真实缓存权重（DINOv3 ViT-S/16 + SigLIP2 base）联合编码：DINO dense P4 `(1, 384, 4, 4)`，SigLIP2 semantic `(1, 768)`，F14 route summary `(1, 3458)`，全量 finite。
- F14 recipe COCO8 CPU smoke：1 epoch、2 batch steps 完成，训练日志中的 `foundation` 非零（约 `0.173`）；验证阶段正常完成。并修复了 Foundation 指标列为奇数时的结果绘图越界，`plot_results` 复测通过。
- `ruff check`、`ruff format --check`、`compileall`、`git diff --check` 通过。
