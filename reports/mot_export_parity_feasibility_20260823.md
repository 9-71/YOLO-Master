# MoT 导出语义对齐可行性分析：eager-sparse vs export-dense

> 日期：2026-08-23
> 状态：**原型已验证，代码已落地（opt-in flag）**
> 关联：`reports/yolo_master_deep_analysis_20260821.md` §二十（问题根因）

---

## 一、问题回顾

MoT 在 eager eval 下执行 Top-K 稀疏分发（重归一化后选中专家权重=1.0），而导出图（ONNX/TorchScript）退化为全量 softmax dense 混合——两条路径计算的是不同函数，数值偏差 ~0.003（随机初始化路由器下），且部署语义不一致（导出模型精度未经 eager 路径验证）。

## 二、方案矩阵

| 方案 | 原理 | 数值对齐 | 计算节省 | 可 trace 性 | 结论 |
|:--|:--|:--|:--|:--|:--|
| **A. Masked-dense 导出** | 导出时用 TopK + 广播比较掩码重建稀疏等价权重，专家计算保持 dense | ✅ **位级精确** | ❌ 无（所有专家仍计算） | ✅ 纯静态算子 | **采纳，已实现** |
| B. Gumbel-hard / ST 导出 | 导出期以硬采样替代 softmax | ≈ 等价于 A 的另一种掩码来源 | ❌ 无 | ⚠️ Gumbel 噪声在 trace 中需固定 seed，引入额外复杂度 | 不优于 A，弃 |
| C. 路由器 LUT/蒸馏 | 把路由器决策蒸馏成查找表或轻量规则 | ❌ 有损 | ✅ 路由开销下降 | ✅ | 研究项，精度风险高，不推荐为对齐手段 |
| D. 真稀疏动态导出 | ONNX `NonZero` + 动态 shape 稀疏计算 | ✅ | ✅ 理论最优 | ❌ TensorRT/CoreML 对动态 shape 支持差，mainline 部署栈不可用 | 不可行（部署栈约束） |

**关键洞察**：对齐与加速是两个独立目标。方案 A 以零精度代价解决对齐；加速在部署栈不支持动态稀疏的前提下本就是 eager-only 能力——不应让"无法加速"阻碍"语义对齐"。

## 三、方案 A 实现（已落地）

**核心技巧**：用广播比较替代 `scatter_` 构建 Top-K 掩码（`scatter_` 的 bool 掩码正是当初触发 PyTorch 2.9 legacy exporter 别名分析失败的原因）：

```python
# ultralytics/nn/modules/mot/router.py（export 分支，export_masked=True 时）
_, topk_idx = weights.topk(self.top_k, dim=1)
expert_range = torch.arange(self.num_experts, device=weights.device).view(1, -1, 1, 1)
mask = torch.zeros_like(weights)
for k in range(self.top_k):  # K 为静态值，trace 时自动展开
    mask = mask + (expert_range == topk_idx[:, k : k + 1]).to(weights.dtype)
weights = stable_normalize(weights * mask.clamp(max=1.0), dim=1)
```

全部算子（TopK / Range / Equal / Cast / Add / Clamp / Sum / Div）均为 ONNX opset ≤17 原生支持，TorchScript trace 无动态控制流。

**API**：`MoTBlock(..., export_masked=True)`（默认 `False`，行为完全不变）；`export_capabilities()` 新增 `export_router_weights: "masked_topk" | "dense_softmax"` 遥测键。

## 四、原型验证证据（torch 2.9.1，CPU）

| 验证 | 结果 |
|:--|:--|
| 数值探针：masked-dense traced vs eager-sparse（top_k=1） | **max_abs_error = 0.000e+00**（位级精确；旧 dense 路径为 2.55e-03） |
| top_k ∈ {1, 2, 3} × TorchScript roundtrip | 全部 **0.000e+00** passed |
| `test_mot_export_masked_matches_sparse_eager`（torchscript + onnx） | ✅ passed（无需 reference 覆盖，直接与 eager-sparse 比较） |
| MoT/MoA 全范围回归 | ✅ 243 passed（含新增用例） |
| 默认行为（export_masked=False） | 不变，既有的 reference 机制继续守护 dense 语义 roundtrip |

## 五、开销与限制

1. **计算量**：导出图仍执行全部专家（dense compute）——对齐不带来部署加速；图级开销仅增加 TopK + E 路比较 + 归一化（相对专家计算可忽略）。
2. **精度语义变化**：`export_masked=True` 的导出产物输出等于 eager 稀疏输出——与既有 dense 导出产物数值不同（这正是目的）。对已部署 dense 导出的用户属行为变更，故保持 opt-in。
3. **真稀疏加速**仍是 eager-only；TensorRT 部署若要拿到稀疏收益需走条件执行/自定义 plugin，超出本方案范围。

## 六、建议

1. **v26.09 将 `export_masked` 默认置 True**（本周期 opt-in 浸泡后）——对齐应该是默认行为，dense-softmax 导出作为遗留兼容保留一个版本周期后在 release note 中标注弃用。
2. master 配置线中 MoT 变体（`yolo-master-mot-*.yaml`）的导出文档补充一句：导出产物与 eager 稀疏路径位级一致（当 export_masked 开启）。
3. MoA/MoE 家族若有同类 eager-sparse/export-dense 分叉，复用同一 `_export_semantics_reference` + masked 模式（当前它们的 roundtrip 已通过，无需动作）。

## 七、结论

**对齐问题已实质解决**：方案 A 以 ~15 行代码、零精度代价、全静态算子实现了导出产物与 eager 稀疏路径的位级一致，且在当初触发 exporter 失败的 torch 2.9.1 环境实证通过。剩余的"真稀疏部署加速"是部署栈能力问题（方案 D），不属于代码缺陷，建议作为 TensorRT plugin 方向的研究项单列。
