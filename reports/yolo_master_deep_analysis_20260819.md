# YOLO-Master 深度分析报告：稳定版本评估与下一阶段路线图

> 分析日期：2026-08-19
> 项目路径：`/Users/gatilin/PycharmProjects/YOLO-Master-v260819-latest`
> 当前版本：Ultralytics base `8.4.101` · 发布线 `YOLO-Master-v26.08`（2026-08-07）
> 分析方法：核心源码与历史报告交叉阅读（20260716 / 20260723 两份深度报告、v26.08 发布说明）+ 本轮实机验证（Ruff、CI 回归门禁、MoE/V-PEFT/引擎/Foundation/Agent Skill 测试）
> 前序报告：`reports/yolo_master_deep_analysis_20260716.md`、`reports/yolo_master_moe_moa_mot_peft_analysis_20260723.md`

---

## 一、项目总览

### 1.1 定位与愿景

YOLO-Master（Tencent Youtu Lab，CVPR 2026）是基于 Ultralytics 深度改造的实时目标检测框架，核心命题 **"compute-on-demand"**：以实例条件自适应计算（MoE/MoA/MoT 动态路由 + PEFT 低成本适配）取代静态稠密计算。主结果 YOLO-Master-N 在 MS COCO **42.4% AP @ 1.62ms**，比 YOLOv13-N +0.8 mAP 且快 17.8%。

### 1.2 当前阶段判断

项目刚完成 **v26.08 累积发布**（2026-08-07）：上游基线 8.3.240 → 8.4.101、622 commits / 96 PRs、官方 focused gate **377 passed / 1 xfailed**。发布后又合入一条明显的新主线：

| 发布后合入（git log 实证） | 内容 |
|:---|:---|
| PR #230 `foundation-alpha-f06-f15` | **Foundation 子系统 alpha**：teacher distillation 训练管线、SigLIP2/DINOv3 teacher、multirouter、semantic、F15 multitask，含 F06–F15 契约与 effect-gate 工具 |
| PR #228 `fix/molora-grouped-conv-merge` | 修复 MoLoRA 合并 grouped-conv 基础层时崩溃/组合错误 delta |
| PR #227 `examples/edge-v1.1.0` | 跨平台 edge runners v1.1.0 |
| PR #226 `fix/multitask-p0-hardening` | MultiTask P0 加固 |

**结论：项目处于"v26.08 稳定基线已立、Foundation alpha 刚并入主干"的节点。下一轮稳定版本（v26.09/v1.0 候选）的主题应是：Foundation 子系统从 alpha 收敛到 preview/stable + 历史 lint 债务清偿 + 真实硬件验证补证。**

### 1.3 规模盘点（本轮实测）

| 维度 | 数值 |
|:---|:---|
| `ultralytics/` Python 文件 | 338 个，**122,291 行** |
| `tests/` 测试文件 | **136 个**（7 月时为 99 个，+37%） |
| MoE 子系统 | 10,657 行（30 文件，STABLE/EXPERIMENTAL/LEGACY 三级 API） |
| MoA / MoT / MultiTask | 1,267 / 1,613 / 775 行 |
| PEFT 栈 | `nn/peft` 3,002 行 + `vpeft` 4,731 行 + `utils/lora` ~7.7K 行 |
| Foundation（新） | 2,126 行 + `foundation_distill_model.py` + 8 个 F08–F15 实验配置 |
| Agent 运行层 | 11,319 行 |
| 模型配置版本线 | `cfg/models/master/` v0 → v0_15 共 15 个版本目录 + exp 实验线 |

---

## 二、本轮实机验证结果（2026-08-19，Python 3.11 + torch 2.9.1）

### 2.1 测试验证矩阵

| 验证项 | 命令范围 | 结果 |
|:---|:---|:---|
| **CI P0/P1 回归门禁** | `test_default_config_integrity` + `test_master_model_configs` + `test_molora_dtype` + `test_molora_backend_roundtrip` + `test_molora_merge_semantics` + `test_adapter_backend_contract` | ✅ **24 passed** |
| **MoE 高风险区** | `test_moe_router_boundaries` + `test_moe_dynamic_schedule` + `test_vpeft` | ✅ **78 passed** |
| **引擎套件** | `test_engine.py`（-n 4 并行） | ⚠️ **32 passed / 1 failed**（详见 P1-1） |
| **Foundation 全套** | `-k foundation`（20 个测试文件） | ✅ **158 passed** |
| **Agent Skill 快速套件** | `validate_yolo_master_skill.py --suite quick` | ✅ 全部通过 |
| **Ruff lint** | `ultralytics/ tests/ scripts/ agent/` | ❌ **439 errors**（143 可自动修复） |
| **Ruff format** | 同上 | ❌ **159 文件待重排版** / 459 已合规 |

### 2.2 与 7 月报告的问题核销对照

| 7 月报告问题 | 当前状态 | 证据 |
|:---|:---|:---|
| 🔴 MoLoRA fp16 forward 失败 | ✅ **已修复** | `test_molora_dtype.py` 入 CI 门禁且通过 |
| 🔴 MoLoRA merge 均匀平均 ≠ 训练路由语义 | ✅ **已修复** | routing-aware merge + `test_molora_merge_semantics.py` 守护通过；PR #228 又修 grouped-conv 合并 |
| 🟡 `yolo26-master-n.yaml` SPPF 不兼容无法构建 | ✅ **已修复** | `test_master_model_configs` 中 yolo26-master-n 构建+前向通过（0.33s） |
| 🟢 `default.yaml` 4 个重复 key | ✅ **已修复** | `test_default_config_integrity.py` 通过 |
| 🔴 MoT/MoA 延迟高、增益未闭环 | ➡️ **维持原判**（见 4.2） | 无新精度证据合入 |
| 🟡 V-PEFT 未接主训练链路 | ➡️ **维持原判** | 仍为研究原型 |

**7 月报告的 4 个工程硬问题已全部收口**，MoLoRA "收敛冲刺"建议已被完整执行——这是本周期最大的质量进展。

---

## 三、架构现状（子系统成熟度评估）

### 3.1 成熟度矩阵

| 子系统 | 定位 | 成熟度 | 变化（vs 7 月） |
|:---|:---|:---|:---|
| **ES-MoE** | 论文核心，compute-on-demand 主力 | ★★★★★ | → 稳定，剪枝/诊断/恢复闭环完整 |
| **MoA** | 多尺度注意力软路由，即插即用件 | ★★★☆ | → 维持，精度增益仍未证 |
| **MoT** | 架构级路由（含 scene-aware router） | ★★★ | → 维持，性价比未闭环 |
| **LoRA 内核** | 纯配置激活的低成本微调 | ★★★★☆ | → 稳定 |
| **MoLoRA** | MoE 思想进 PEFT | ★★★☆ → ★★★★ | ↑ **fp16/merge/grouped-conv 三硬伤全修** |
| **V-PEFT Planner** | GATv2+PPO+MIP 的适配器规划器 | ★★★☆ | → 研究原型，未接主链路 |
| **MultiTask** | 统一 det/seg/pose/cls/depth/normal | ★★★（Preview） | ↑ P0 加固合入（#226） |
| **Foundation** | DINOv3/SigLIP2 蒸馏 + multirouter | ★★☆（**alpha**） | 🆕 158 测试全绿，但无真实硬件蒸馏证据 |
| **Agent Skill** | CLI 分发器 + 多模态管线 | ★★★★ | → quick 套件通过 |
| **统一路由基础设施** | routing_protocol + mixture_loss | ★★★★★ | → 五类 aux loss 收口，隐形骨架 |

### 3.2 Foundation 子系统（新主线，重点评估）

代码：`ultralytics/nn/foundation/`（losses / preprocessing / projectors / protocol / routing / semantic / taps / teachers）+ `foundation_distill_model.py` + `cfg/experiments/foundation/f08~f15` 8 个实验配置。

- **定位**：训练期 teacher 蒸馏管线（DINOv3-ViT-S16 / SigLIP2），把学生检测器对齐到基础模型表征，含 multirouter（多路由器）与 semantic 分支两条延伸线。
- **工程状态**：契约测试（F06–F15）+ effect-gate 工具 + checkpoint 元数据（JSON-safe、additive）+ hook 生命周期管理齐备，**158 个测试全绿**，工程质量与 MoE 主线同级。
- **关键缺口**：
  1. `test_foundation_f15_benchmark.py` 硬编码了本机 HF cache 路径（`~/.cache/huggingface/...dinov3-vits16`）——跨机器不可复现；
  2. 全部证据为契约/单元级，**无真实 COCO/VisDrone 蒸馏精度增益数据**，"alpha" 标定是诚实的；
  3. teacher 依赖外部权重下载，离线/CI 环境需注入 stub（`test_builder_accepts_offline_injected_teacher_manager` 表明已意识到此问题）。

---

## 四、问题分类与严重程度

### P0 — 阻塞性缺陷（稳定版发布前必须处理）

**本轮验证未发现新的 P0。** 所有 CI 门禁、MoE 高风险区、Foundation 契约、Agent Skill 套件均通过。7 月报告的 P0（MoLoRA fp16/merge）已核销。

### P1 — 显著问题（下周内修复）

| # | 问题 | 位置 | 影响 |
|:--|:---|:---|:---|
| P1-1 | **`test_resume_incomplete[multitask]` 失败**：`agent/assets/open-world-taxonomy/{lvis_1203_classes,sources}.json` 硬编码旧工作区绝对路径 `YOLO-Master-v260510-paper/...`，引擎测试 resume 时按此路径找 `lvis.yaml` 抛 FileNotFoundError | `agent/assets/open-world-taxonomy/` | 引擎套件 32/33 通过；换机/换目录即坏，阻塞"任意环境可跑全量测试"的稳定版承诺 |
| P1-2 | **Ruff 439 errors / 159 文件格式不合规**，含 **3 个 invalid-syntax**：`tests/test_validator_helpers.py` 使用带括号的 `with` 语句，与项目声明的 Python>=3.8 不兼容（3.9+ 语法） | 全仓，集中在 `scripts/`、`agent/`、legacy helpers | 发布说明已承认"历史 lint 债务"，但 py3.8 语法错误直接违反 setup 声明的最低版本 |
| P1-3 | **F821 undefined-name ×3**：`scripts/ablation_suite/full_ablation_multiscale.py`（LOGGER×2）、`scripts/reproduce/reproduce_visdrone_sparse.py`（spec）——运行即 NameError | `scripts/` | 消融/复现脚本可直接崩溃，损害论文可复现性叙事 |
| P1-4 | Foundation benchmark 测试硬编码本机 HF cache 路径 | `tests/test_foundation_f15_benchmark.py:11` | 跨机器不可复现，与 Foundation 走向 preview 的目标冲突 |

### P2 — 优化项（下一版本周期规划）

| # | 问题 | 说明 |
|:--|:---|:---|
| P2-1 | MoT/MoA 性价比未闭环 | 延迟 +52%~148% 换 ≤1.1% 相对增益；scene-aware router 假设待实证；建议冻结新投入、补 COCO128 多 seed 或转定位 |
| P2-2 | V-PEFT 未接主训练链路 | GATv2+PPO+MIP 是最有研究品位的资产，仍停留在原型；与 `utils/lora/planner.py`（2686 行工程版）职责重叠待切分 |
| P2-3 | 真实硬件证据缺口（v26.08 发布说明自承） | 无 CUDA/MPS AMP 多轮审计、无 NCCL 双卡训练记录、无 routed profile 的 TensorRT 延迟基准、Core ML 验证被 macOS coremltools/SciPy 二进制问题阻塞 |
| P2-4 | MultiTask 预测仍走 detection predictor | 无公开 `tasks=[...]` 多输出预测 API |
| P2-5 | 198 个 E402（模块级导入不在顶部） | 多为刻意延迟导入，建议用 per-file-ignores 显式声明而非裸挂 |
| P2-6 | `gated.py` 112KB 单文件 | MoE"活历史"集中在单文件，可观测性/评审成本高，建议按版本线拆分 |

---

## 五、跨模块交互风险

1. **硬编码路径的跨模块传染**：agent assets JSON → 引擎 multitask resume 测试（P1-1 已爆）、foundation benchmark → HF cache（P1-4 已爆）。模式相同：**资产文件/测试记录本机绝对路径**。建议全仓 grep `/Users/` 作为 CI 门禁。
2. **Foundation × Mixture 路由栈叠加**：foundation multirouter 复用统一路由协议，aux loss 已收口（`CompositeCriterion` EMA 归一化 + aux_budget + NaN 隔离），当前测试全绿；但蒸馏 loss + 五类 mixture aux loss 同时开启时的**量级相互作用尚无实证**——F15 multitask 配置上线前需一组"蒸馏开/关 × mixture 开/关"的 2×2 训练验证。
3. **版本共存**：master 配置线 v0–v0_15 + `cfg/models/26/` + `cfg/experiments/foundation/` 三套 YAML 体系并存，`test_master_model_configs` 覆盖了构建/前向，但导出（ONNX eager-sparse 回退 dense）对 foundation wrapper 的覆盖尚未在门禁中见到。

---

## 六、通往稳定版本的路线图

### Phase 1（本周，发布阻断项）

1. **修复 P1-1/P1-4 硬编码路径**：改为仓库相对路径 + 环境变量覆盖，并把 `grep -r "/Users/" agent/assets tests/` 加入 CI。
2. **修复 P1-3 三个 F821**：补 import 或删残留引用，消融脚本跑一次 dry-run。
3. **`ruff check --fix` 清偿 143 个可自动修复项**，`tests/test_validator_helpers.py` 的 `with()` 语法改为 3.8 兼容写法（或把 pyproject 声明提升到 >=3.9 并同步 CI 矩阵——二选一，必须显式决策）。
4. 全量 `pytest tests/ -n auto` 跑一次作为候选基线。

### Phase 2（下一版本 v26.09，主题：Foundation 收敛）

1. Foundation alpha → preview：补 COCO128/COCO 子集蒸馏增益数据（F08–F15 至少一条完整训练曲线），解除 HF cache 硬依赖。
2. 蒸馏 × mixture 2×2 交互验证（见风险 2）。
3. MoT/MoA 定位决策：补证 or 文档降级为 experimental 推荐配置（CPU 部署 top_k=1 / 仅 P5）。
4. 补真实硬件证据：NCCL 双卡、TRT routed profile 基准、Core ML 环境解锁。

### Phase 3（v1.0 候选）

1. V-PEFT 主链路接入（规划→训练→评测端到端闭环），与工程版 planner 职责切分。
2. MultiTask 公开预测 API。
3. MoE 剪枝做成一键 pipeline（20–30% 提速是当前最实际的部署卖点）。
4. Model Zoo L/X 权重补齐 + 全量精度/延迟表。

---

## 七、结论

**总体成熟度：8.2 / 10**（7 月评估口径下约 +0.6）

- **基本盘极稳**：MoE 论文级结果 + 统一路由基础设施 + 136 测试文件构成的门禁网，是实时检测框架中罕见的工程纵深。
- **本周期最大进展**：7 月报告的全部工程硬伤（MoLoRA fp16/merge、yaml 构建、配置重复 key）核销完毕，v26.08 上游升级落地，Foundation 新主线以 158 个全绿测试的高起点并入。
- **距稳定版本的差距集中在"纪律"而非"架构"**：硬编码绝对路径（2 处已爆雷）、439 个 lint 债务、3.8 语法承诺被破坏、真实硬件证据缺口。这些都是一周内可清偿的项。
- **建议**：按 Phase 1 清偿后即可打 **v26.08.1 补丁版**；Foundation 补证后打 **v26.09**。架构层面无需新动作，重点是证据链与卫生。

---

## 八、Phase 1 修复执行记录（2026-08-19 当日完成）

报告中的全部 P1 已在当日修复并复验通过，另发现并修复一个 **新 P0**：

### 8.1 新发现 P0（已修复）——MultiTask 训练端到端崩溃

`TaskRouter`（`ultralytics/nn/modules/multitask/router.py`）把 graph-connected 的 `last_affinity` 存为普通属性，导致 EMA 初始化 `deepcopy(model)` 必然崩溃（`RuntimeError: Only Tensors ... (graph leaves) support the deepcopy protocol`）。**即 v26.08 文档推荐的 `yolo train data=coco-multitask.yaml model=yolo26-master-mt-n.yaml` 在主干上无法运行**——此前无 multitask e2e 训练测试覆盖（单测均标注 "no full training"），官方门禁中该用例为 xfail。

**修复**：为 `TaskRouter` 增加 `__getstate__/__setstate__`，深拷贝/序列化时将路由缓存置空（运行时缓存，下次前向自动重建），保留 aux loss 梯度链路语义不变。

### 8.2 修复清单

| 项 | 修复内容 | 文件 |
|:--|:--|:--|
| P0 | TaskRouter deepcopy 崩溃（见上） | `ultralytics/nn/modules/multitask/router.py` |
| P1-1 | `TASK_MODEL_DATA` 对 YAML 配置不再拼 `WEIGHTS_DIR` 绝对路径，改由 `check_file` 仓内搜索；`TASK2DATA[multitask]` 指向契约合规的 `coco-multitask.yaml`；`test_resume_incomplete` 为 multitask 构建 4 图 COCO 格式临时夹具（detect+segment） | `tests/__init__.py`、`ultralytics/cfg/__init__.py`、`tests/test_engine.py` |
| P1-2 | `test_validator_helpers.py` 三处带括号 `with` 改为反斜杠续行（py3.8 兼容）；`ruff --fix` 清偿 **143 项**（54 文件）；恢复被误删的 `_MOE_FINITE_DIAGNOSTICS` 兼容再导出并加入 `__all__` + `noqa` | `tests/test_validator_helpers.py`、`ultralytics/nn/modules/moe/base.py` 等 |
| P1-3 | 补 `LOGGER` 导入；`spec` → 模块级 `SPARSE_MODEL` | `scripts/ablation_suite/full_ablation_multiscale.py`、`scripts/reproduce/reproduce_visdrone_sparse.py` |
| P1-4 | DINOv3 路径改为 `YOLO_MASTER_DINOV3_PATH` 环境变量优先 + 标准 HF cache 自动发现 + 缺失即 skip | `tests/test_foundation_f15_benchmark.py` |

### 8.3 修复后复验（全部通过）

| 验证项 | 结果 |
|:--|:--|
| 引擎套件 `test_engine.py`（含 multitask resume 真实训练+恢复，89.9s） | ✅ **33/33 passed**（修复前 32/33） |
| multitask 单元测试（router `__getstate__` 回归） | ✅ 89 passed |
| CI P0/P1 门禁 + MoE 高风险区 + V-PEFT + validator helpers | ✅ **107 passed** |
| Foundation 全套 | ✅ 158 passed |
| Agent Skill quick 套件 | ✅ 36/36 passed（score 1.0） |
| Ruff | 439 → **277**（剩余：E402×185 刻意延迟导入、E701/E702×67 风格、F841×23 全部位于 scripts/ 分析报告脚本，无核心代码） |

### 8.4 遗留事项（转入 Phase 2）

1. **环境侧（未动，需用户决策）**：全局 Ultralytics `settings.json` 的 `weights_dir`/`runs_dir`/`datasets_dir` 仍指向旧工作区（v260510-paper / v260720 / v260804-latest）。本次修复后测试不再依赖它们，但建议切换到当前工作区以绝后患。
2. `coco8-multitask.yaml` / `coco128-multitask.yaml` 两个数据集配置仍是硬化前格式（声明 obb、缺 `multitask_format: coco`），当前无法用于训练，建议更新或标注废弃。
3. 建议把 `TaskRouter` 缓存模式与 `grep -r "/Users/"` 纳入 CI 门禁，防止同类回归。
4. E402×185 建议在 `pyproject.toml` 用 per-file-ignores 显式声明，而非保持裸挂。

---

## 九、Phase 2 首组实证：F08 COCO128 蒸馏对照实验（2026-08-19）

> 运行环境：Apple Silicon MPS（`foundation_teacher_device=mps`，首次验证 teacher 可不回退 CPU）；本机同时有高负载训练任务，迭代速度受影响。数据集 coco128 从 `v260302` 工作区复制至本仓 `datasets/`（网络受限无法下载）。

### 9.1 实验设置

| 组 | 配置 |
|:--|:--|
| 基线 | `yolo26n.yaml` 从零训练，coco128，8 epochs，imgsz=256，batch=16，MPS，amp=False，seed 默认 |
| 蒸馏 | 同上 + DINOv3-ViT-S16 teacher（transformers 后端，本地 HF 缓存，`HF_HUB_OFFLINE=1`），target=P4，align_dim=32，hybrid loss（cosine 1.0 + relation 1.0，sampled 16），weight=0.05 |

产物：`runs/foundation/coco128-baseline-e8/`、`runs/foundation/coco128-foundation-e8/`（蒸馏组于第 6 epoch 末触及执行时限，按前 6 epoch 配对比较）。

### 9.2 结果

**逐 epoch 检测损失（前 6 epoch 配对）**：

| epoch | box 基线 | box 蒸馏 | cls 基线 | cls 蒸馏 | dfl 基线 | dfl 蒸馏 |
|:--|:--|:--|:--|:--|:--|:--|
| 1 | 3.5680 | 3.5680 | 5.6336 | 5.6335 | 0.0573 | 0.0573 |
| 3 | 3.6259 | 3.6248 | 5.6370 | 5.6373 | 0.0586 | 0.0586 |
| 6 | 3.6636 | 3.6830 | 5.5986 | 5.5957 | 0.0588 | 0.0592 |

- **mAP 两组均为 0**（8 epochs 从零训练 128 图的预期现象，本尺度不构成精度结论）。
- **蒸馏管线端到端真实数据首证通过**：teacher 前向稳定、无 NaN、完整 telemetry 落盘（`foundation_loss` / `cosine` / `relational` / `task_ratio` / `foreground_*` 八列）。
- **对齐信号活跃**：cosine 对齐损失 6 epoch 内 0.7613 → 0.7386（**-3.0%**），relational 稳定在 0.26–0.28——学生特征确实在向 DINOv3 表征靠拢，loss 不是平线。
- **检测损失差异 ±0.7% 以内**：weight=0.05 下蒸馏梯度占总损失仅 ~0.05，短程训练尚不足以扰动检测目标——符合预期，增益验证需要更长 schedule（或预训练初始化 + 微调范式）。

### 9.3 结论与下一步

Foundation alpha 的**真实数据训练通路已打通**（此前全部证据为合成数据/契约级），但**精度增益仍未证**——这是 alpha → preview 的剩余核心缺口。建议下一组实验（待本机高负载任务结束后）：

1. coco128 50–100 epochs 或 coco8-seg 预训练初始化微调，weight 扫 {0.05, 0.2}；
2. 蒸馏开/关 × mixture 开/关 2×2 交互验证（报告第五节风险 2）；
3. 记录 `foundation_task_ratio`（当前 ~0.02）随 weight 变化对收敛的影响。

---

## 十、微调范式对照矩阵与蒸馏损失诊断（2026-08-19）

> 动机：第九节从零训练尺度下蒸馏增益不可观测，切换为**预训练初始化 + 短程微调**范式（yolo26n.pt 零样本 mAP50-95=0.3527），并对蒸馏组发现的 cosine 对齐停滞与日志量级异常做根因诊断。

### 10.1 微调对照矩阵（coco128，train=val 128 图，单 seed，smoke 尺度）

| 组 | lr0 | epochs | mAP50-95 | 备注 |
|:--|:--|:--|:--|:--|
| 零样本参照 | — | — | 0.3527 | `weights/yolo26n.pt` 直接验证 |
| ft 基线 | 0.01 | 10 | 0.3340 | 蒸馏 w=0.2：0.3333（无差异） |
| ft 基线 | 0.001 | 8 | 0.3396 | **蒸馏 w=0.5：0.3267（受损 -0.013）** |

CKA（64 图，SPPF 输出 vs DINOv3 CLS token）：zero-shot 0.6507 / ft-基线 0.6460 / ft-蒸馏 0.6460——**特征相似度几乎不动**，与 cosine 停滞互为印证。

产物：`runs/foundation/{ft-baseline,ft-distill,ft2-baseline,ft2-distill}*/`。

### 10.2 根因诊断结论

**（1）日志 10× 量级异常——非 bug，是 telemetry 设计缺陷。** 日志列 `foundation_cosine_loss` = raw × `loss_weight` × `batch_size`（`foundation_distill_model.py:939-940`），导致跨 weight/batch 配置不可比。反推 raw 值：从零组 0.95→0.92（在降）；预训练组 ≈1.01（cos≈0，特征近正交且不动）。raw 上界 2.0（`cosine_kd_loss` = 1 − mean cos sim，归一化正确）。

**（2）cosine 停滞的主因是优化强度不足，不是梯度断裂。** 梯度注入探针（30 步 SGD，仅蒸馏损失回传）：lr=0.01 时 cosine 1.0048→0.9726（在降），lr=0.001 时仅 →1.0015。预训练组 8 epoch × 8 step/epoch = 64 优化步 + lr0=0.001，量级上不足以让 16k 参数 projector 把正交特征拉齐。

**（3）backbone 梯度通路完好（排查中一度误判为 P0）。** 探针初测显示 tap 捕获特征 `requires_grad=False`，疑似蒸馏梯度无法回传 backbone；复核证实这是探针加载路径的假象——`YOLO("*.pt").model` 预测路径会冻结全部 366 个参数。改用 `requires_grad_(True)` + `train()` 重测：梯度经 tap 通畅回传至 stem（model.0 |g|=0.157）、中层（model.4 |g|=0.497）、tap 源（model.16 |g|=0.095）。`taps.py:_hook_fn` 明确保留 autograd 图（不 detach），真实 trainer 路径无此问题；从零组 cosine 持续下降亦是旁证。

**（4）w=0.5 损伤检测精度**（0.3267 vs 基线 0.3396）：cos≈0 时蒸馏梯度方向近似随机，大 weight 下成为检测目标的噪声源。这界定了安全区间：**在 cosine 显著下降之前，weight 应保持 ≤0.2**。

### 10.3 修复：telemetry 增加 raw 指标（已落地，本地未提交）

- `foundation_distill_model.py`：`_last_foundation_metrics` 新增 `foundation_cosine_raw` / `foundation_relational_raw`（未经 weight×batch 缩放），保留原缩放列（二者之和仍恒等于 `foundation_loss`，兼容既有断言）。
- `tests/test_foundation_metrics_logging.py`：同步扩展键集合断言。
- 验证：foundation 全量 158 项测试通过；ruff check/format 通过。

### 10.4 对 alpha → preview 的影响

精度增益验证的阻塞项**不是代码缺陷**，而是实验设计：需要（a）足够优化强度（lr≥0.01 或 epoch≥30）让 cosine 先降下来，（b）weight 从 0.05–0.2 起步、随 cosine 下降再放大，（c）raw 指标落盘后可直接跨配置比较对齐进度。telemetry 修复已为此扫清观测障碍。

---

## 十一、决定性实验：15-epoch 微调蒸馏增益首证（2026-08-19）

> 配置：coco128（train=val 128 图），yolo26n.pt 初始化，**SGD lr0=0.01**（显式指定——`optimizer=auto` 会静默忽略 lr0 改用 AdamW lr≈1.2e-4，这正是此前 ft/ft2 组"lr0=0.01/0.001"标签失效、cosine 停滞的隐藏原因），imgsz=256，batch=16，seed=0，amp=False；蒸馏组 w=0.1，hybrid loss。脚本：`scripts/foundation_ft3_decisive.py`（支持断点续跑）。

### 11.1 结果（配对，单 seed，smoke 尺度）

| epoch | 基线 mAP50-95 | 蒸馏 mAP50-95 | Δ | cosine_raw |
|:--|:--|:--|:--|:--|
| 1 | 0.3355 | 0.3350 | -0.0005 | 1.0116 |
| 4 | 0.3034 | 0.2882 | -0.0152 | 1.0037 |
| 8 | 0.3483 | 0.3634 | +0.0151 | 0.9845 |
| 12 | 0.3944 | 0.4215 | **+0.0272** | 0.9613 |
| 15 | 0.4203 | **0.4350** | **+0.0147** | 0.9525 |

- **蒸馏增益首次在真实数据上证实**：终局 mAP50-95 +0.0147（相对 +3.5%），mAP50 +0.0025——增益集中在高 IoU 定位质量，与"特征对齐提升表征精度"的机理一致。
- **交叉点与对齐进度同步**：蒸馏组前 4 epoch 落后（cos≈0 阶段蒸馏梯度近似噪声），cosine_raw 跌破 1.0 后（epoch 5 起）反超并持续扩大，epoch 12 达峰 +0.027。完整验证了第十节的根因链：**优化强度 → cosine 下降 → 增益显现**。
- cosine_raw 15 epoch 内 1.0116 → 0.9525（-5.8%），relational 平稳 ~0.50；`foundation_task_ratio` ≈ 0.09–0.14，w=0.1 未干扰检测目标。
- 两曲线在 epoch 15 仍在上升（未收敛），epoch 12→15 差距从 +0.027 收窄至 +0.015——需更长 schedule 判断是噪声还是系统性回落。

### 11.2 执行备注（环境陷阱，已修复）

- 脚本方式启动时 `sys.path[0]` 是 `scripts/` 而非仓库根，会命中 v260720 工作区的陈旧 editable 安装——脚本内已显式 `sys.path.insert` 修正。
- 训练进程安装 SIGTERM 处理器（保存 `last_healthy.pt` 后不退出），后台 `&` + `kill` 会留下孤儿进程并发写同一 results.csv——改为前台运行由工具超时杀进程组，验证无残留。
- 结论：此前 ft/ft2 矩阵中"lr0.01 与 lr0.001 无差异"的现象由 `optimizer=auto` 覆盖 lr0 所致，两组实际都是 AdamW lr≈1.2e-4。

### 11.3 下一步

1. 50-epoch 长程复跑（判断 epoch 12→15 差距收窄的性质）；
2. 双 seed 复验排除单 seed 噪声；
3. weight 扫描 {0.05, 0.2}（cosine 已稳定下降，可试放大）。

---

## 十二、50-epoch 长程复跑：增益的调度依赖性（2026-08-19 深夜）

> 同配置延长至 50 epochs（`foundation_ft3_decisive.py {baseline|distill} 50`）。执行备注：蒸馏组分段续跑陷入"epoch 37 活锁"（每段重启重载 DINOv3 teacher ~2min + 训练 1min + 验证 1-6min > 290s 段长，last.pt 不前移），改为单进程长驻 + 轮询解决。

### 12.1 结果

| epoch | 基线 mAP50-95 | 蒸馏 mAP50-95 | Δ | cosine_raw |
|:--|:--|:--|:--|:--|
| 1 | 0.3355 | 0.3350 | -0.001 | 1.0116 |
| 15 | 0.3806 | 0.3617 | -0.019 | 0.9344 |
| 20 | 0.3856 | 0.3928 | +0.007 | 0.9106 |
| 30 | 0.4286 | 0.4345 | +0.006 | 0.8601 |
| 40 | 0.4460 | 0.4540 | +0.008 | 0.8163 |
| 50 | **0.4630** | 0.4553 | **-0.008** | 0.7886 |

BEST：基线 0.4630 vs 蒸馏 0.4558（Δ -0.007）。mAP50 终局 0.6332 vs 0.6216（Δ -0.012）。

### 12.2 结论：15e 的 +0.015 增益是调度位置相关的瞬态现象

- **差距轨迹三段论**（e20 后逐 epoch）：e20–32 蒸馏稳定领先 +0.005~+0.014 → e33–45 衰减至 ±0.005 → e46–50 稳定落后 -0.007~-0.008。15e 实验终点恰好落在"领先窗口"尾部，epoch 12→15 的收窄（+0.027→+0.015）正是此衰减的前兆，非噪声。
- **cosine_raw 持续深化至 0.7886（-22%）但增益消失**：对齐本身一直有效，但训练后期 DINOv3 CLS 语义特征与检测最优特征的拉扯开始损害检测目标——**恒定 weight=0.1 的蒸馏在三个阶段的角色是：早期噪声（cos≈0）→ 中期助力 → 后期干扰**。
- 对照第五节跨模块风险与第十节诊断：蒸馏管线功能正确、可观测性完备，剩余问题是**损失调度策略**而非实现缺陷。

### 12.3 对 alpha → preview 的修正结论与行动项

蒸馏增益真实但瞬态，恒定 weight 无法保住。下一步实验（按优先级）：

1. **weight 调度**：cosine 门控（cosine_raw < 1.0 后启用）+ 末段衰减（最后 20-30% epoch 线性降至 0），目标是把 e20–32 的 +0.01 窗口延续到收敛点；
2. 多 seed（×3）复验：当前 ±0.01 效应量接近单 seed 噪声，paper 级结论需要 coco128 多 seed 或更大数据集；
3. weight 扫描 {0.05, 0.2} 与调度策略叠加验证。

---

## 十三、gate_decay 调度实现与 50e 验证：增益保持到收敛（2026-08-20 凌晨）

### 13.1 实现（本地未提交）

新增 `foundation_weight_schedule=gate_decay`（默认 `constant`，行为完全不变）：

- **cosine 门控爬升**：`foundation_cosine_raw` 的 EMA（动量 `foundation_gate_ema=0.9`）跌破 `foundation_gate_cosine=1.0` 后，weight 在 `foundation_gate_width=0.05` 跨度内线性开满；门关闭期保留 `foundation_warmup_floor=0.2` 底线权重，让 projector 在特征近正交阶段仍能学习（避免"门等对齐、对齐等权重"死锁）。
- **末段衰减**：训练进度（trainer 每 epoch 经 `set_foundation_progress` 同步）超过 `foundation_decay_start=0.7` 后线性衰减，终局 epoch 归零。
- telemetry 新增 `foundation_effective_weight`；缩放列改用有效权重（恒等式 `foundation_loss = cosine + relational` 保持）。

改动：`foundation_distill_model.py`（调度器）、`trainer.py`（3 行进度同步，getattr 守卫）、`default.yaml`（5 个新键）、`tests/test_foundation_weight_schedule.py`（8 项新测试）。验证：foundation + 配置完整性 171 项通过，ruff 干净；`test_engine.py` 29/33 通过，4 个失败均为断网导致的截断权重下载（`PytorchStreamReader`），与本改动无关，损坏文件已清理。

### 13.2 三组 50e 对比（coco128，SGD lr0=0.01，单 seed）

| epoch | 基线 | 恒定 w=0.1 | gate_decay | Δ恒定 | Δ调度 |
|:--|:--|:--|:--|:--|:--|
| 10 | 0.3362 | 0.3209 | 0.3091 | -0.015 | -0.027 |
| 20 | 0.3856 | 0.3928 | 0.3940 | +0.007 | +0.009 |
| 35 | 0.4422 | 0.4269 | 0.4543 | -0.015 | +0.012 |
| 40 | 0.4460 | 0.4540 | 0.4637 | +0.008 | +0.018 |
| 50 | 0.4630 | 0.4553 | **0.4795** | -0.008 | **+0.017** |

- **终局 mAP50-95：调度 +0.0166（+3.6%），恒定 -0.0077**；晚窗口（e45–50）平均差距：调度 **+0.0172 稳定**，恒定 -0.0048。调度成功把中期增益保留到收敛点。
- mAP50 终局：调度 0.6517 vs 基线 0.6332（+0.019）——增益不再局限于高 IoU。
- 调度器行为符合设计：floor 期 eff_w=0.020 → e13 门开爬升 → e36 峰值 0.090 → 末段精确衰减至 0.000。
- 副发现：调度组 cosine_raw 仅降至 0.9154（恒定组 0.7886）——**更少的过对齐反而带来更高的检测精度**，佐证后期强对齐约束与检测目标存在结构性冲突。

### 13.3 剩余缺口

单 seed、coco128 smoke 尺度：+0.017 效应量需 seed×3 复验才能排除运气成分；其后是 weight 扫描与更大数据集（coco8-seg / VOC）验证。

### 13.4 三 seed 复验：效应量被种子噪声淹没（2026-08-20 上午）

复验：seed ∈ {0, 1, 2} 的基线/调度配对（seed1/2 两组并发同负载，配对公平）。

| seed | 基线终局 | 调度终局 | Δfinal | Δ晚窗口(e45-50) |
|:--|:--|:--|:--|:--|
| 0 | 0.4630 | 0.4795 | +0.0166 | +0.0172 |
| 1 | 0.4958 | 0.4828 | -0.0131 | -0.0084 |
| 2 | 0.4844 | 0.4936 | +0.0092 | +0.0018 |
| **均值±σ** | — | — | **+0.0042 ± 0.0154** | +0.0035 ± 0.0129 |

**结论修正**：基线本身的种子间散布 0.4630–0.4958（σ=0.0167），是调度效应均值的 ~4 倍——**coco128（train=val，128 图）在该效应量级上不具备统计分辨力**。seed0 的 +3.6% 增益未能复现（2/3 为正，均值弱正）。gate_decay 调度的定位修正为：**机制正确、遥测行为符合设计、均值上不损害精度（弱正向），但精度增益声明在 smoke 尺度不成立**——需要在种子方差更小的测试床（VOC 量级或 coco 子集）上验证，当前网络受限无法下载更大数据集，此为本轮实验的硬约束。

---

## 十四、机制修正：过对齐失败模式与 band 门控（2026-08-20 下午）

### 14.1 三 seed 遥测解剖发现的两个机制级问题

**（1）过对齐是真实失败模式。** 终局 Δ 与终局 cosine_raw 的跨运行相关（4 个数据点）：cos 0.79 → -0.008（恒定组）、0.87 → **-0.013**（seed1，对齐最深、唯一净负的调度组）、0.92 → +0.017、0.93 → +0.009——**对齐越深，检测越差**，最优区间在 cosine_raw ≈ 0.92–0.95（cos sim ≈ 0.05–0.08）。seed1 的失利路径：对齐最快（e5 cos 已 0.996）→ 门早开、权重 e30 即满 → 特征被拉向 DINOv3 过深（0.87）→ 末段衰减虽撤掉权重，但低 LR 余量不足以把检测特征调回。

**（2）EMA 状态不进 checkpoint（resume 缺陷）。** `_cosine_ema` 存于 wrapper `__dict__`，checkpoint 虽 pickle 整个 wrapper，但 `rebuild_foundation_distillation_wrapper` 只恢复 projector/semantic_projector——**resume 后 EMA 重置为 None，门控关闭、权重跌回 floor 约 2 epoch**（EMA 动量 0.9 下需 ~10-20 步重建）。本轮分段训练多次踩中。

### 14.2 修正（本地未提交）

- **band 门控**：新增 `foundation_gate_cosine_low=0.9`——EMA 低于下限后 ramp 反向关闭（tent 形调节器，峰值在 band 中心 0.95，两侧渐退至 floor），把"一次性爬升"改为"围绕目标对齐区间的调节"；`0` 可禁用回到单向门控。实证注释写入 `_gate_factor` docstring。
- **EMA 持久化**：`rebuild_foundation_distillation_wrapper` 恢复 `_cosine_ema`（旧 checkpoint 无此字段则保持 None 重新预热，向后兼容）。
- 测试：`test_foundation_weight_schedule.py` 新增 band 关闭/禁用/参数校验 3 项（单向爬升用例改为 low=0 隔离验证），`test_foundation_checkpoint.py` 新增 EMA 恢复 2 断言。验证：**174 项 foundation+配置测试通过**，ruff 干净；3-epoch 真实数据冒烟通过（新配置键管线无校验错误，无运行时异常）。

### 14.3 当前机制全貌与待验证项

gate_decay 最终形态：**floor 预热（防死锁）→ cosine band 调节（防噪声期干扰 + 防过对齐）→ 末段线性衰减归零（防末期拉扯）**。三个失败模式各有对策，机制自洽。待验证：band 下沿 0.9 取自 4 个噪声数据点的相关关系，属先验而非结论——需在低方差测试床（VOC 量级，待网络恢复）上做 band 开/关 × weight 扫描的因子实验确认。

---

## 附录 A：本轮验证命令与原始结果

```bash
# 环境：/Library/Frameworks/Python.framework/Versions/3.11/bin/python3（torch 2.9.1, ultralytics 8.4.101 editable）
ruff check ultralytics/ tests/ scripts/ agent/        # 439 errors（198 E402 / 107 F401 / 35 E702 / 32 E701 / 27 F541 / 23 F841 / 6 E401 / 4 F811 / 3 invalid-syntax / 3 F821 / 1 E741）
ruff format --check ...                                # 159 would reformat / 459 formatted
pytest tests/test_default_config_integrity.py tests/test_master_model_configs.py \
       tests/test_molora_dtype.py tests/test_molora_backend_roundtrip.py \
       tests/test_molora_merge_semantics.py tests/test_adapter_backend_contract.py   # 24 passed
pytest tests/test_moe_router_boundaries.py tests/test_moe_dynamic_schedule.py tests/test_vpeft.py  # 78 passed
pytest tests/test_engine.py -n 4                       # 32 passed, 1 failed（P1-1）
pytest tests/ -k foundation -n 4                       # 158 passed
python agent/scripts/validate_yolo_master_skill.py --suite quick --summary-only      # all passed
```

## 附录 B：关键文件索引

| 主题 | 路径 |
|:---|:---|
| MoE 核心 | `ultralytics/nn/modules/moe/{modules,gated,routers,experts,loss,pruning,analysis}.py` |
| MoA / MoT | `ultralytics/nn/modules/{moa,mot}/{block,router,wrappers}.py` 等 |
| 统一路由 | `ultralytics/nn/modules/routing_protocol.py`、`ultralytics/nn/mixture_loss.py` |
| MoLoRA / V-PEFT | `ultralytics/nn/peft/molora/`、`ultralytics/vpeft/{graph,constraints,policy,solver}.py` |
| Foundation | `ultralytics/nn/foundation/`、`ultralytics/nn/foundation_distill_model.py`、`ultralytics/cfg/experiments/foundation/` |
| 发布说明 | `docs/release-notes/v26.08.md`（含 Known Limitations 官方清单） |
| 历史报告 | `reports/yolo_master_deep_analysis_20260716.md`、`reports/yolo_master_moe_moa_mot_peft_analysis_20260723.md` |

---

## 十五、F09–F15 全阶段真实数据工程门（2026-08-20 晚）

> 工具：`scripts/foundation_v02_smoke.py`（新增，支持 f09–f15 七组）。真实 DINOv3 / SigLIP2 教师（本地 HF 缓存，离线模式），学生 MPS 训练，逐组断言遥测有限性 + 阶段激活信号 + checkpoint 无教师权重泄漏。产物：`runs/foundation/v02-smoke-{f09..f15}-coco8/`。

### 15.1 结果矩阵（全部 PASS）

| 阶段 | 内容 | 关键证据 |
|:--|:--|:--|
| F09 | GT 前景感知加权（1.5/1.0/0.25） | `foreground_enabled=1.0`，平均 token 权重 0.916 ∈ (0.25, 1.5] |
| F10 | 多尺度 P3/P4/P5 独立 adapter | 分级损失 p3=0.310 / p4=0.278 / p5=0.303，均有限 |
| F11 | DINOv3 → LatentMixture 路由蒸馏 | **3 个 image-level router 全部接入**，router_kl=0.0137，teacher/student 熵 1.383/1.386（4 专家近均匀，符合初始化预期） |
| F12 | SigLIP2 作为特征 KD 教师 | cosine_raw=0.980、relational_raw=0.587，管线与 DINOv3 同构可换 |
| F13 | 正区域语义蒸馏（特征 KD 关闭） | semantic_enabled=1.0，**237 个正区域**参与蒸馏，text_loss=0.422 / image_loss=0.099，特征分支正确归零 |
| F14 | 多基础模型教师路由（DINOv3 空间 + SigLIP2 语义） | 双教师加载成功，3 个 router 接入，router_kl=0.0106 |
| F15 | MultiTask 表征迁移（detect+segment 夹具） | **满足 F15 gate：同 run 两个监督任务损失非零**（detect 7.75 / segment 26.05），`representation_transfer_ready=1.0`，任务路由用量落盘（detect 0.497 / segment 0.072），负迁移监控 0.5 < 阈值 4.0 |

### 15.2 过程中发现的环境/工程问题（已记录，均未改核心代码）

1. **高系统负载导致 290s 窗口不足**：latent 模型组（f11/f14）单 epoch ~90–105s，叠加教师加载与终验会超窗。处置：这两组对齐仓库 recipe 改为 1 epoch + `val=False`。
2. **中断残留损坏 `.cache`**：被超时杀掉的进程在 `~/PycharmProjects/datasets/coco8/labels/` 留下截断缓存，后续运行报 `KeyError '_items'`。删除 `.cache` 后恢复。教训：轮询/分段执行时缓存写入非原子。
3. **checkpoint 序列化结构认知修正**：训练中保存的 `last.pt` 里 `model=None`、权重在 `ema`（recovery controller 的 `include_online_model=False` 设计）；`val=True` 收尾时 strip 才写入纯学生 `model`。smoke 脚本的教师泄漏检查相应改为 `model or ema` 兜底 + 精确黑名单（`teacher_manager`/`dinov3`/`siglip`），`_projector.teacher_proj` 为合法可训练 adapter，不算泄漏。

### 15.3 阶段状态总结

至此方案文档 F00–F15 **全部阶段均具备真实数据工程证据**（不再仅有契约/合成级测试）。剩余缺口不变：精度增益的统计有效性（需低方差测试床）与各研究分支的效果门（band 开/关 × weight 因子实验、F09 权重 ablation、F10 教师层选择 ablation），均受网络约束阻塞。
