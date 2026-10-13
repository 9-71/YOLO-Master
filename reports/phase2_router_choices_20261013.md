# 第二阶段：预训练权重的专家选择受控诊断

日期：2026-10-13。MPS Apple M5 Pro；Python 3.11.16、torch 2.14.1。

## 本轮交付

新增 `scripts/evaluate_router_choices.py` 和 14 项有针对性的测试，将专家选择从随机模块计时推进到真实检测指标对照。支持 ES_MOE 与 MoTBlock；不把嵌套 router 当作独立专家块。每次只干预一个指定层，其余层保持原生策略；每个条件重新加载相同 checkpoint。

对照包括原生、均匀混合、原生 Top-1、原生 Top-2、逐个固定专家和原生基线复跑。记录逐图验证指标、路由平均权重、实际专家调用和样本—专家组合数；排除 warmup 的计数。Top-2 会临时关闭目标 ES_MOE 的动态阈值，防止第二个专家被阈值再次过滤；所有状态与 hook 在退出时恢复。

冻结所选图片与标签的哈希、checkpoint 哈希、环境和脚本身份，保存脚本快照；报告逐条件保存，最终代码使用临时文件原子替换 JSON。输入 checkpoint 必须在本地，数据自动下载关闭，输出目录中已有报告时拒绝覆盖。`--policies` 可只运行关注的策略，同时自动保留原生与复跑控制。

更新 `benchmarks/PROTOCOL.md`，加入 MPS/A800 命令、干预定义与解释边界。

## 权重与数据

使用项目公开发布的 `YOLO-Master-EsMoE-N.pt`，来自：

https://github.com/Tencent/YOLO-Master/releases/download/YOLO-Master-v26.02/YOLO-Master-EsMoE-N.pt

文件大小 5,943,935 字节，SHA256：`29e1b93f09b16c8cf7c402f36dcaafc19d4812155631ed45b769e941e4c88c32`。

该 checkpoint 含 model.3、model.6、model.9、model.12 四个 ES_MOE 块，每层三个专家。它与第一阶段 v0.10 的 VisualEnhancedAdaptiveGateMoE 架构不同，不能把本轮结论直接外推到 v0.10 或 MoT。

主要对照使用本地完整 coco128、320 输入、batch=1、MPS FP32、rect=False、conf=0.001、IoU NMS=0.7、max_det=300。数据来自 COCO train2017，可能与预训练数据重叠。**这是一组固定 checkpoint 的诊断，不是独立测试集泛化成绩，也不是训练种子实验。**

## 主要发现

下列 AP 均指 AP50-95×100，差值单位为 AP 点；不同层每次独立干预。

**原生基线：44.1963 AP。** 三组完整对照的原生前后复跑报告指标差均为 0，说明本轮测量未观察到基线漂移。

**早期层 model.3 对混合策略极其敏感：**

- 均匀混合：0.0075 AP；仍执行三个专家，因此下降不能解释为只减少了专家调用。
- 原生 Top-1：0.0114 AP；该条件在 128 张图中始终执行专家 0。
- 原生 Top-2：8.7922 AP，比基线低 35.4041 点；本次执行专家 0 和 2。
- 固定专家 0/1/2：0.0114 / 0.0000 / 0.0123 AP。

**后期层 model.12 相对更能承受权重替换，但直接减少专家仍有损失：**

- 均匀混合：43.6932 AP，比基线低 0.5031 点；仍执行三个专家。
- 原生 Top-1：31.6105 AP，比基线低 12.5858 点；本次始终执行专家 0。
- 原生 Top-2：41.0714 AP，比基线低 3.1249 点；本次执行专家 0 和 1。
- 固定专家 0/1/2：31.6105 / 20.3677 / 21.8710 AP。

原生每张图在目标层执行 3 个样本—专家组合；Top-1 和 Top-2 分别执行 1 个与 2 个。它们只代表该层专家执行量，不能解释为整个模型计算量下降 2/3 或 1/3。

主要证据由三组运行构成：model.3 完整控制 7 条件、model.12 完整控制 8 条件、model.3 Top-2 补充 3 条件，共 18 个完整 128 图条件。此前 32 图探针及 model.12 初始五条件用于流程验证，不混入上述主结果。

## 可以与不能得出的结论

可以确定：对这个 checkpoint 和数据，直接将早期层改为单专家或 Top-2 会破坏现有检测能力；后期层的 Top-2 损失较小，但仍明显超过路线图提出的 0.3 点 AP 降幅门槛。均匀混合在不同层表现差异很大，后续应按层设计策略，而非全模型统一降 k。

不能确定：这种敏感性的训练根因、在独立测试集上的程度、不同种子或不同架构是否一致。固定与 Top-k 干预同时改变了混合支持和重归一化后的权重；BatchNorm/下游特征分布变化、专家共同适配和其他机制尚未被分离。不能仅据此宣称专家没有互补性，或路由器已经学到可泛化的场景分工。

尤其需要注意：均匀混合控制保留了专家数量，但没有保留原生的加权特征分布；它是权重敏感性诊断，不是无混杂的因果机制证明。逐图指标采用验证器 IoU=0.5 和 conf=0.001，不能等同于标准 AP，也没有构造标注辅助部署选择器。

带 hook 的验证时延不作为正式性能结果。本轮没有修改模型训练默认值、保存裁剪模型或训练新的控制器。

## 下一步决策

1. **暂停对该已发布模型直接降 k 的交付尝试。** 当前没有通过精度门的减少专家候选；均匀混合也没有减少专家执行，不能当作稀疏部署收益。
2. 下一次 MPS 试验优先分离“分布校准”和“稀疏训练”：保留原模型；在独立校准样本上比较 Top-2 不校准、仅校准归一化统计、固定预算的稀疏微调，再在未参与校准/训练的样本上验证。避免使用当前 128 图选择并宣称测试收益。
3. 稀疏策略优先从后期层开始，早期层保持原生混合。只有当校准或训练恢复了精度，才投入预算路由器与真实硬件加速。
4. A800 先复现相同干预，再迁移到独立 COCO val2017 或业务验证集；冻结初始权重、数据划分、训练预算及独立种子。目标架构 v0.10 需另建自己的训练与选择诊断入口。
5. Studio macOS 监管问题沿用第一阶段的独立修复项，本轮未修改 runtime；当前实验为同步执行，没有依赖已知有问题的后台任务监管。

## 验证

- 新增路由选择测试、基准 runner、分发基准合计 **28 passed**；涵盖 ES_MOE/MoT 的实际专家执行、均匀控制、Top-2 阈值恢复、异常 hook 清理、旧 checkpoint 缺失字段恢复、标签内容哈希与基线漂移检测。
- 路由边界、动态调度、模型配置、RoutedModule 相关回归 **117 passed**。
- 新增 Python 文件 Ruff lint/format、所选文件 codespell 与 diff whitespace 检查通过。
- 模型代码没有改动；没有复跑完整 Studio 或清偿全仓 lint 债务。

## 本地产物与复用

- `runs/router_choices/mps_esmoe_model3_controls/choices.json`
- `runs/router_choices/mps_esmoe_model12_controls/choices.json`
- `runs/router_choices/mps_esmoe_model3_top2/choices.json`
- `runs/weights/YOLO-Master-EsMoE-N.provenance.json`
- `runs/verification/phase2/router-tools.xml`

各目录含冻结图片列表与 subset.yaml；model.12 控制组和 model.3 Top-2 补充组另外保存与报告脚本哈希匹配的 runner.py。早期 model.3 七条件组保留其运行时脚本哈希，后续新增 Top-2 等代码使当前脚本身份发生变化，未用当前代码冒充早期源码。

复用入口示例：

```bash
.venv/bin/python scripts/evaluate_router_choices.py \
  --weights runs/weights/YOLO-Master-EsMoE-N.pt \
  --layer model.12 --data coco128.yaml --max-images 128 \
  --imgsz 320 --batch 1 --device mps \
  --output runs/router_choices/new_model12
```

迁移 A800 时将 device 改为 cuda:0，使用新的输出目录。正式效果结论需要替换为独立数据协议；以上命令本身仍是诊断工具。
