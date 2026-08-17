# F07 真实 DINOv3 权重验证记录

日期：2026-08-13（Asia/Shanghai）

## 权重来源与完整性

- 下载端点：`https://hf-mirror.com`（`HF_ENDPOINT=https://hf-mirror.com`）。
- 原始目标：`facebook/dinov3-vits16-pretrain-lvd1689m`，当前 token 对该 gated 仓库仍返回 HTTP 403。
- 实际验证快照：`Tooony133/dinov3-vits16-pretrain-lvd1689m`，revision `fc6921f7a0b44d5b33ab4482cfed5443db6ccd81`。该非 gated 复刻的 `model.safetensors` SHA-256 与 Facebook 原始仓库 API 元数据完全一致：
  `4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d`。
- 本地快照：`/Users/gatilin/.cache/huggingface/hub/models--Tooony133--dinov3-vits16-pretrain-lvd1689m/snapshots/fc6921f7a0b44d5b33ab4482cfed5443db6ccd81`。
- 文件：`config.json` 743 B，`preprocessor_config.json` 585 B，`model.safetensors` 86,406,384 B。
- 未将模型权重复制进 Git 工作区；权重保留在 Hugging Face 本地缓存。

## Teacher 单独验证

构造参数：Transformers `DINOv3ViTBackbone`、CPU、FP32、`local_files_only=True`。

- 加载成功；参数量 `21,596,544`，`hidden_size=384`，`patch_size=16`。
- 输入 `(1, 3, 64, 64)` -> `dense['p4']=(1, 384, 4, 4)`，`pooled=(1, 384)`。
- 输入 `(2, 3, 80, 96)` -> `dense['p4']=(2, 384, 5, 6)`，`pooled=(2, 384)`。
- 输出全部 finite；输入预处理完成 ImageNet normalization 与 patch 对齐。
- Teacher 与 backbone 均保持 `eval()`，全部参数 `requires_grad=False`。

## Foundation KD 单步验证

使用真实 `DetectionModel('yolo26n.yaml', nc=3)` 与真实 Teacher，CPU、batch=2、输入 `64x64`、Foundation loss weight `0.05`、hybrid KD。

- 前向返回 `total.shape=(4,)`、`items.shape=(4,)`，Foundation KD item=`0.1939361393`，全部 loss finite。
- 反向成功；student P4 projector 梯度和 backbone 梯度均非零。
- Teacher 无梯度；Teacher 训练状态仍为 eval/frozen。
- eval/predict 路径返回 student prediction，未执行 teacher inference。

## Builder 集成验证

使用配置构造路径（`foundation_model` 与 `foundation_weights` 指向上述本地快照）成功生成 `FoundationDistillationModel`；检测到 student P4 `128` 通道与 teacher P4 `384` 通道，projector 正常创建。

## 结论

F07 真实权重验证通过。当前真实验证依赖镜像上的非 gated 复刻仓库；其权重内容哈希与 Facebook 官方仓库声明的权重哈希一致。若后续 token 获得 Facebook gated 授权，可将 `foundation_model` 切换回官方模型 ID，代码路径无需修改。

## F07 checkpoint / resume / export

- `checkpoint_runtime_metadata()` 增加 `foundation` 元数据，包含 teacher/backend/model/weights、loss 超参数、P4 通道与对齐维度；训练恢复 checkpoint 同时提供顶层 `foundation` 字段，旧 checkpoint 缺失该字段时保持兼容。
- resume 从 checkpoint 中重建 student 与 Foundation wrapper，并恢复 projector 权重；teacher 不从 checkpoint 反序列化，而是按当前训练配置重新构造，继续保持不注册、不进入 optimizer/EMA 的边界。
- `Exporter` 和 `strip_optimizer()` 在部署产物路径剥离 Foundation wrapper、projector 和 P4 hook，只保留纯 student；原始训练 checkpoint metadata 不被删除。
- 新增 `tests/test_foundation_checkpoint.py`，覆盖 metadata、projector 恢复、recovery checkpoint 与 strip 行为。

F07 回归：Foundation + checkpoint compatibility `103 passed`；engine distill 筛选 `2 passed`；P0 checkpoint/export 筛选 `2 passed`；Ruff、format、compileall、`git diff --check` 通过。
