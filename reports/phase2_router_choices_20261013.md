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

## 后续 MPS 校准与特征蒸馏验证（2026-10-13）

新增 `scripts/calibrate_sparse_esmoe.py`。用户授权下载额外 COCO 图片后，从本地 LVIS 图片清单提供候选 ID，仅接受 COCO 官方 S3 `val2017` 路径成功返回的图片；最终使用按 ID 排序的 32 张。下载来源及内容哈希保存在 `runs/calibration/coco_val32/provenance.json`。校准图片与 coco128 按 COCO ID 和内容 SHA256 双重排重，零重叠。该小样本不是随机抽取，不代表完整 COCO 分布，且不能保证与模型预训练数据无重叠。

首轮全模型 BN 重估作为负面证据保留：Top-2 + 全模型 BN 为 **21.1763 AP**，再做特征蒸馏为 **24.3279 AP**。不能将其下降归因于路由本身；全模型统计重置同时改变了上游特征分布。产物在 `runs/calibration/model12_top2_bn_kd_v2/`。此前相对路径错误导致的未完成启动目录保留在 `model12_top2_bn_kd/`，不计入完整实验。

随后将 BN 校准范围限制为 model.12 输出归一化层，其余 BN 保持冻结；完成六个完整 coco128 条件：

- 原生：**44.1963 AP**。
- 直接后期 Top-2：**41.0714 AP**。
- 仅目标输出 BN 校准：**40.9153 AP**。
- Top-2 + 特征蒸馏（不校准 BN）：**42.3069 AP**。
- Top-2 + 目标输出 BN 校准 + 特征蒸馏：**42.7580 AP**。
- 原生复跑：**44.1963 AP**，所有报告指标最大差为 0。

微调预算固定为 32 图 × 3 轮，batch=1，Adam lr=1e-4，torch seed=42，MPS FP32，320 输入。原生 teacher 冻结，仅目标 ES_MOE 块参数更新，所有 BN 统计在蒸馏中冻结。以 student 自身上游产生的输入拟合 teacher 原生层输出，损失为按 teacher 能量归一化的特征 MSE；没有使用 coco128 标签训练，也没有使用检测损失。目标层始终 eval 模式以实际走稀疏执行路径，同时允许参数梯度；其他层原生混合。验证中该层每图实际执行 2 个样本—专家组合，原生为 3。

最佳本轮候选相对直接 Top-2 恢复 **1.6866 AP 点**，距原生仍差 **1.4383 点**，未达到 ≤0.3 点门槛。不能宣称恢复完成，也不能把目标层执行量减少直接解释为全模型加速。两个蒸馏条件仅各一个训练种子；组合优于单独蒸馏尚无统计显著性结论。

coco128 继续作为已被观察过的固定回归集，因此这些分数不是独立测试泛化成绩。下一步应在独立带标注样本上开展固定预算的检测损失微调，并留出未用于模型/预算选择的验证集；保留本轮三轮蒸馏作为对照。达到精度门之后再安排 A800 无 hook 部署计时。

复用命令：

```bash
.venv/bin/python scripts/calibrate_sparse_esmoe.py \
  --weights runs/weights/YOLO-Master-EsMoE-N.pt \
  --calibration runs/calibration/coco_val32 \
  --layer model.12 --bn-scope target --epochs 3 --lr 0.0001 \
  --device mps --output runs/calibration/new_target_experiment
```

六条件证据、数据哈希、运行源码快照和策略化 state_dict 在 `runs/calibration/model12_target_bn_kd/`；state_dict 文件不是可直接交付的 YOLO checkpoint，加载时必须使用原始结构并重放文件中的 layer/policy。原始发布权重未修改，未上传图片或模型产物。

工具与相关路由回归 **74 passed**，覆盖内容/ID 排重、BN 参数与模式恢复、局部校准范围、真实稀疏反向更新、teacher 和 BN 冻结、hook 清理；Ruff lint/format 与 diff whitespace 通过。

## 带标注检测损失微调与独立划分试验（2026-10-13）

新增 `scripts/prepare_sparse_supervised_data.py` 与 `scripts/finetune_sparse_detector.py`，完成两种子、固定预算的 MPS 检测损失微调试验。数据从 COCO val2017 下载，标签来自项目 COCO 配置所引用的 Ultralytics `coco2017labels.zip`；共 160 张，按固定 seed=42 划分训练 64、开发 32、最终评估 64。排除上一轮 32 张校准图和 coco128，所有划分按 COCO ID、图片内容 SHA256 双重排重，标签也冻结哈希。它们在本轮训练/选型中互不重叠，**不保证与发布模型预训练数据无重叠**。

每个种子各跑五个开发集条件：原生、直接 Top-2、同预算原生稠密微调、Top-2 检测损失微调、Top-2 检测损失加原生特征蒸馏。仅在两个稀疏微调候选间按开发 AP 选型，并在首次最终评估前将候选写入报告。两个种子均选出检测损失加蒸馏。第二种子 43 在首轮最终成绩被读取前固定；两种子共享数据划分，不是两次独立数据抽样。

训练预算：各 64 图 × 3 轮，batch=1，320 输入，MPS FP32；Adam lr=1e-4，梯度范数上限 10，KD 系数 1，无额外 BN 重估、数据增强或优化器调参。仅 model.12 的参数可训练，其他参数和所有 BN 统计逐项验证不变。模型保持 eval 模式以走实际稀疏路径，检测头输出送入原生检测损失；本轮没有训练模式的 MoE 辅助损失。每个训练条件实际 192 个样本，稠密每图执行 3 个专家，两个稀疏条件均执行 2 个。

开发集（32 图）AP50-95×100：

- 原生：35.8162；直接 Top-2：33.8352。
- seed 42：稠密微调 36.2345；稀疏检测微调 34.3671；稀疏检测加 KD 34.8893。
- seed 43：稠密微调 36.5096；稀疏检测微调 34.5630；稀疏检测加 KD 35.2787。

最终评估（64 图，417 个实例）AP50-95×100：

- 原生：**49.0832**；直接 Top-2：**49.0186**，差 **−0.0646 AP 点**。
- seed 42：同预算稠密微调 **44.9681**；开发集选出的稀疏检测加 KD **47.9117**，相对原生 **−1.1715 点**。
- seed 43：同预算稠密微调 **47.8592**；开发集选出的稀疏检测加 KD **48.2986**，相对原生 **−0.7846 点**。
- 两个种子的原生前后复跑均一致。

本轮没有复现“微调恢复即可改善独立评估”的预期。两个稀疏候选都优于各自同预算稠密微调，但都低于未经微调的直接 Top-2；稠密微调也下降，因此不能把下降全部归因于稀疏路由。这里与 coco128 上 Top-2 损失 3.12 点的差异，说明效果对数据分布敏感；64 图上的 0.06 点差距不足以证明完整 COCO 或业务数据达到精度门。

决策：暂停扩大当前 64 图三轮微调预算，保留未经微调的 Top-2 作为下一步候选。优先在更大、冻结且不参与训练/预算选择的 COCO/业务评估集上复核原生与 Top-2；不同数据分布应分别报告。取得可信精度结论后再安排 A800 无 hook 部署计时。本轮没有修改默认模型配置或交付裁剪 checkpoint。

复现入口：

```bash
.venv/bin/python scripts/prepare_sparse_supervised_data.py \
  --archive runs/supervised_sparse/coco2017labels.zip \
  --exclude-report runs/calibration/model12_target_bn_kd/report.json \
  --counts 64 32 64 --seed 42 --output runs/supervised_sparse/new_coco160
.venv/bin/python scripts/finetune_sparse_detector.py \
  --weights runs/weights/YOLO-Master-EsMoE-N.pt \
  --dataset runs/supervised_sparse/new_coco160 --epochs 3 --seed 42 \
  --output runs/supervised_sparse/new_seed42
```

将训练 seed 改为 43，使用新输出目录复现第二种子。运行源码、完整报告、策略化 state_dict 保存在 `runs/supervised_sparse/mps_seed42/` 与 `mps_seed43/`。最终脚本在运行快照基础上补充了逐图评估身份和训练专家数的显式断言，训练行为与预算未变。可提交证据见 `reports/evidence/sparse_supervised_20261013/pilot.json`，含图片 ID/哈希、标签哈希、环境、训练轨迹、执行计数和开发/最终指标；图片和权重未上传。

新增 5 项测试覆盖分组排除、标签篡改、交叉内容重叠、只按开发集选型，以及真实 DetectionModel 检测损失的稀疏反向更新和 teacher/BN 冻结；相关测试合计 **87 passed**。最终防护更新后新增 5 项测试再次通过；Ruff lint/format 与 diff whitespace 通过。

## 256 张新增样本与冻结候选验证（2026-10-13）

继续下载额外 COCO val2017 图片，排除此前 32 张校准图、coco128 的 128 张图以及监督试验 160 张图，排除清单共 320 张。固定 seed=20261013，从剩余图片抽取 258 张，其中 256 张只用于本轮验证，另外 2 张因数据准备入口的 train/dev 正数约束而保留未使用；本轮没有训练或调参。图片/标签哈希冻结，按 COCO ID 和内容双重确认不重叠。256 图共 **1,921 个实例、77 个有标注类别**，仍不能保证与发布模型预训练数据无重叠。

先完成原生、直接后期 Top-2、原生复跑三个条件；随后对此前四份已保存稀疏权重做冻结验证，重复原生/Top-2/原生复跑控制。所有条件都是相同的 256 图、320 输入、MPS FP32、batch=1、conf=0.001、NMS IoU=0.7、max_det=300。新增 `scripts/evaluate_sparse_checkpoints.py` 校验基础权重来源、目标层/策略、张量 state_dict、逐图身份以及实际专家执行数；禁止静默换 checkpoint 或把 state_dict 当完整 YOLO 模型加载。

完整 AP50-95×100 结果：

- 原生：**36.7680**。
- 未微调 Top-2：**34.8181**，相对原生 **−1.9498 点**。
- 32 图特征蒸馏、不重估 BN：**35.9141**，相对原生 **−0.8539 点**。
- 32 图目标输出 BN 校准加特征蒸馏：**35.8919**，相对原生 **−0.8761 点**。
- 64 图监督检测损失加 KD，seed 42：**34.1299**，相对原生 **−2.6381 点**。
- 64 图监督检测损失加 KD，seed 43：**34.5928**，相对原生 **−2.1752 点**。
- 两组实验的原生复跑一致，最大指标绝对差为 0；两组原生/直接 Top-2 的结果也完全一致。

所有稀疏候选在验证中均每图执行 2 个目标层专家，原生为 3。特征蒸馏相对直接 Top-2 恢复约 1.07–1.10 点，但没有候选达到 ≤0.3 点参考精度门。32 图特征蒸馏与 64 图监督微调的数据、损失和训练预算不同，因此这里不能当作严格的训练方法优劣消融；本轮也不依据这批分数重新选定交付模型。

与之前 64 图上直接 Top-2 仅下降 0.0646 点相比，更大新增子集再次观察到明显损失。不同子集的绝对 AP 与有标注类别构成不同，不能直接横比绝对 AP；各自相对原生的降幅可以作为分布敏感性的证据，但尚无完整 COCO 或业务泛化保证。

当前结论：保持原生发布模型；继续研究特征蒸馏能否在更多训练样本上稳定恢复精度，暂停扩大现有 64 图监督微调预算及 A800 加速交付。下一阶段应把纯检测、纯 KD、检测加 KD 放在**同一更大训练集与固定预算**下比较，并另留未参与路线/预算选择的新验证集。已经观察过的 64 图和本轮 256 图今后只能作回归或开发数据，不能再次宣称为全新的最终测试。

复用入口：

```bash
.venv/bin/python scripts/evaluate_sparse_checkpoints.py \
  --weights runs/weights/YOLO-Master-EsMoE-N.pt \
  --data runs/supervised_sparse/verification256/test.yaml \
  --candidates runs/calibration/model12_target_bn_kd/top2_kd.state.pt \
    runs/calibration/model12_target_bn_kd/top2_bn_kd.state.pt \
    runs/supervised_sparse/mps_seed42/sparse_det_kd.state.pt \
    runs/supervised_sparse/mps_seed43/sparse_det_kd.state.pt \
  --output runs/supervised_sparse/new_frozen_candidate_verification
```

原始证据位于 `runs/router_choices/mps_model12_verification256/` 和 `runs/supervised_sparse/verification256_candidates/`。可提交证据在 `reports/evidence/sparse_verification256_20261013/results.json`，含 256 张图片 ID/哈希、标签哈希、候选权重和来源报告哈希、控制与全部候选指标、执行计数、环境和实际运行源码哈希。运行快照保留；最终入口另补充了 state_dict 非有限值/非法载荷拒绝检查，未改变合法候选的评估行为。

新增 6 项候选来源和状态契约测试；与此前路由、训练和模型配置验证合计 **93 passed**。Ruff lint/format、证据身份与运行快照哈希核对、diff whitespace 通过。没有上传图片或权重，也没有将带遥测的验证耗时作为部署性能结果。
