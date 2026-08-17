# F12 SigLIP2 Foundation Teacher

日期：2026-08-13（Asia/Shanghai）

## Scope

F12 新增可选的 SigLIP2 Foundation Teacher backend，保持 F00–F11 默认关闭和 DINOv3 路径兼容：

- `FoundationFeatures.pooled`：图像级 embedding。
- `FoundationFeatures.semantic`：归一化图像语义 embedding。
- `FoundationFeatures.dense['p4']`：可用时由 patch token 恢复的 BCHW dense feature。
- `encode_text(prompts)`：文本 prototype 编码，并按 prompt tuple 做确定性缓存。
- teacher 始终冻结、eval、training-only，不进入 student module tree、optimizer、EMA 或 export graph。

## 配置

```yaml
foundation_teacher: siglip2
foundation_model: google/siglip2-base-patch16-512
foundation_loss_weight: 0.05
```

Recipe：[f12-foundation-siglip2-coco8.yaml](../../../ultralytics/cfg/experiments/foundation/f12-foundation-siglip2-coco8.yaml)

## Verification

- SigLIP2 mock contracts：4 passed
- Foundation config/recipe/wrapper regression：47 passed
- 本地 Hugging Face cache 真实权重加载：通过
- 真实图像 smoke：`dense['p4']=(1,768,32,32)`，`pooled=(1,768)`，`semantic=(1,768)`
- 真实文本 prototype smoke：`encode_text([...]) -> (1,768)`，二次调用命中缓存
- SigLIP2 + FoundationDistillationModel wrapper smoke：通过，loss finite
- `ruff check`、`ruff format --check`、`compileall`：通过

注意：当前本地缓存的 `google/siglip2-base-patch16-512` 配置由 Transformers 识别为兼容的 `siglip` 架构；backend 使用 `AutoModel/AutoProcessor` 兼容加载，并在无文本输入时直接调用 vision encoder。后续若切换完整 SigLIP2 仓库，接口无需变化。
