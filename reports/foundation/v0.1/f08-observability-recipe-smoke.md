# F08 训练可观测性、recipe 与真实 DINOv3 smoke

日期：2026-08-13（Asia/Shanghai）

## Recipe

- 配置：`ultralytics/cfg/experiments/foundation/f08-foundation-distill-coco8-dinov3.yaml`
- student：`yolo26n.yaml`
- 数据：`coco8.yaml`，CPU，`imgsz=64`，batch=2，1 epoch
- Foundation：DINOv3、Transformers backend、hybrid KD、P4、对齐维度 32、loss weight 0.05
- teacher：`Tooony133/dinov3-vits16-pretrain-lvd1689m`
- 权重通过 `HF_ENDPOINT=https://hf-mirror.com` 使用 Hugging Face 本地缓存加载；权重完整性和单独 teacher 验证记录见 `f07-real-dinov3-validation.md`。

## 真实训练 smoke

训练完成，输出目录：

`/Users/gatilin/PycharmProjects/YOLO-Master-v260720/runs/detect/runs/foundation/f08-smoke`

生成了 `weights/last.pt`、`weights/best.pt` 和 `results.csv`。训练日志确认真实 teacher 完成加载并参与两个 batch 的 Foundation KD，loss 全部 finite。

`results.csv` 新增并写入了以下 epoch 级指标：

- `train/foundation_loss`：0.173867
- `train/foundation_cosine_loss`：0.0998606
- `train/foundation_relational_loss`：0.0740062
- `train/foundation_task_ratio`：0.0285773
- `train/foundation_loss_weight`：0.05

Foundation hybrid 分量满足 `foundation_loss = cosine + relational`（浮点误差内），原有 detection loss 列保持不变。

## Checkpoint 与 resume

`last.pt` 保留顶层 `foundation` 以及 `mixture_checkpoint.foundation` 元数据，包含 teacher 标识、backend、model、loss 超参数、P4 通道数和 projector 对齐维度；teacher 本体不进入 checkpoint 的 module tree。

使用该 `last.pt` 执行 `resume=True, epochs=2` 成功恢复并完成第 2 个 epoch。日志确认 Foundation wrapper、真实 DINOv3 teacher 和 projector 被重新构造，`results.csv` 追加了第二个 epoch 的五项 Foundation 指标。

teacher-side projector 参数在训练与 resume 中保持 `requires_grad=False`，不会进入 optimizer 更新；student-side projector 仍可训练。

## Deployment strip

对真实训练 checkpoint 执行 `strip_optimizer()` 后：

- `model` 类型为 `ultralytics.nn.tasks.DetectionModel`；
- 不再包含 `projector` state-dict 参数；
- 所有 forward hook 数为 0；
- 不包含 teacher manager；
- checkpoint 的 Foundation 元数据仍保留，便于追溯训练来源。

因此 F08 训练可观测性、checkpoint resume 边界与 deployment strip 均通过真实权重 smoke。

另外修复了导出边界：从 checkpoint 加载的 student 在 wrapper 剥离后显式恢复 `task` 等部署元数据，避免 TorchScript/ONNX exporter 访问 `DetectionModel.task` 时丢失上下文。

真实 checkpoint 的 TorchScript 导出已复验成功：生成文件可由 `torch.jit.load` 加载，输入 `(1, 3, 64, 64)` 输出张量形状为 `(1, 84, 6)`。

## 自动化验证

- F08 与 Foundation 定向测试：15 passed
- Foundation 全套测试：99 passed
- F07 Foundation checkpoint / resume / strip 回归：包含在 Foundation 全套测试中并通过
- Ruff check：通过
- Ruff format check：通过
- compileall：通过
- `git diff --check`：通过
- `tests/test_engine.py -k 'distill or resume'`：9 passed；1 个既有 multitask 用例因仓库外部缺失模型配置而在测试前置构造阶段失败，与 Foundation 代码无关
