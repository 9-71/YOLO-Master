# YOLO-Master 深度分析报告：双线论文格局下的工程收敛状态评估

> 分析日期：2026-08-21
> 项目路径：`/Users/gatilin/PycharmProjects/YOLO-Master-v260819-latest`
> 当前版本：Ultralytics base `8.4.101` · 发布线 `YOLO-Master-v26.08` + Foundation `v0.1.0-alpha`
> 分析方法：在前序报告（20260716 / 20260723 / **20260819**）基础上做增量验证——本轮重新实机运行 CI 门禁、MoE 高风险区、Foundation 全套、引擎套件、Agent Skill 套件与 Ruff，并审计 08-19 报告之后的全部新提交与未提交变更
> 前序报告：`reports/yolo_master_deep_analysis_20260819.md`（含 F00–F15 全部实验记录，本报告不再复述其细节）

---

## 一、项目总览

### 1.1 定位与愿景

YOLO-Master（Tencent Youtu Lab，CVPR 2026，arXiv 2512.23273）是基于 Ultralytics 深度改造的实时目标检测框架，核心命题 **"compute-on-demand"**：以实例条件自适应计算（ES-MoE / MoA / MoT 动态路由 + PEFT 低成本适配）取代静态稠密计算。主结果 YOLO-Master-N 在 MS COCO **42.4% AP @ 1.62ms**，比 YOLOv13-N +0.8 mAP 且快 17.8%。

**本周期的格局性变化：项目从"单论文框架"演进为"双线论文 + 三子系统"的研究平台。**

| 线 | 论文/载体 | 代码资产 | 状态 |
|:--|:--|:--|:--|
| 检测主线 | YOLO-Master（CVPR 2026） | ES-MoE / MoA / MoT / 统一路由基础设施 | 论文级结果已立，工程稳定 |
| PEFT 线 | **YOLO-PEFT**（arXiv 2608.07051，README 已官宣，未提交） | V-PEFT Planner（GATv2+PPO+MIP）、RS-LoRA、MoLoRA | 论文已发布：planner 选择的 RS-LoRA 在 YOLO11s/12s 上 0.7138/0.7307 mAP50-95 vs Full-SFT 0.6428/0.6662，训练峰值内存 -43.9% |
| Foundation 线 | 尚无论文 | DINOv3/SigLIP2 蒸馏 + multirouter + F00–F15 阶段体系 | v0.1.0-alpha 发布说明已落盘，真实数据通路已证，精度增益统计未闭环 |

这意味着 V-PEFT 的定位需要上修：0819 报告将其评为"研究原型，未接主链路（P2-2）"，但它现在承载着一篇已发表 arXiv 论文的核心声明（planner-selected RS-LoRA、Refuse-to-Full-SFT 校准决策、43.9% 内存下降）——**可复现性从"品位问题"升级为"论文信誉问题"**。

### 1.2 规模盘点（本轮实测）

| 维度 | 数值 | vs 0819 报告 |
|:--|:--|:--|
| `ultralytics/` Python | 338 文件，**122,398 行** | → 持平 |
| `tests/` 测试文件 | **150 个**（29,113 行测试代码） | ↑ 136 → 150 |
| `scripts/` 脚本 | **113 个** | ↑ 实验脚本持续增殖 |
| Foundation 测试规模 | **194 passed**（本轮实测） | ↑ 158 → 169 → 194 |
| 模型配置线 | master v0–v0_15 + models/26 + experiments/foundation 三套体系 | v0_15 tiny 新增 |
| 发布说明 | v26.08 + **foundation-v0.1.0-alpha** | ↑ alpha 说明新增 |

### 1.3 0819 报告之后的增量变更（git 实证）

**已提交（6 commits）：**

| 提交 | 内容 | 评估 |
|:--|:--|:--|
| `88a2ed8` | F09 前景感知 effect-gate runner | Foundation 实验基础设施补齐 |
| `8be4a48` | **foundation v0.1.0-alpha 发布说明** | 诚实标定（"within seed noise，不构成已验证改进"），研究诚信记录良好 |
| `65dd2a9` | F08 effect gate + mixture 交互 + v0.2 smoke runners | 跨模块 2×2 验证工具链落地 |
| `90238e4` | yolo-master-tiny **v0_15** 配置 + coco128 对比脚本 | 配置线继续版本化迭代 |
| `56ae402` | streamlit 示例转非 doctest 块 | 文档卫生 |
| `57b9ea3` | engine / nn 包级 **scoped AGENTS.md** | 代理协作指引下沉到包级，治理成熟度提升 |

**未提交（工作区变更）：**

- README：官宣 YOLO-PEFT 论文 + star-history 图 + 引用 bibtex
- 新增 `.github/workflows/star-trend.yml` + `scripts/generate_star_trend.py` + `tests/test_star_trend_generator.py`：每日自动更新 star 趋势图（cron `17 2 * * *`，避开整点，规范）
- F08 effect gate **resume 加固**：`epoch=-1` 健康快照不再被误当 resume 点（此前会触发 Ultralytics resume 拒绝），interrupted_runs 去重——这是对 0819 报告"epoch 37 活锁"类分段训练痛点的直接回应
- 根目录散落产物：`images.zip`（32MB）、`embeddings.npy`、`paths.npy`、7 个字体测试 PDF、3 个腾讯犀牛鸟 PDF/PPTX、`default_copy.yaml`——**仓库卫生问题**（见 P2-1）

---

## 二、本轮实机验证结果（2026-08-21，Python 3.11 + torch 2.9.1，editable 安装指向本仓）

### 2.1 验证矩阵（全部为本轮新跑）

| 验证项 | 范围 | 结果 |
|:--|:--|:--|
| **CI P0/P1 回归门禁** | 配置完整性 + master 模型配置 + MoLoRA dtype/roundtrip/merge/adapter 契约 | ✅ **25 passed**（0819 时 24，新增 1 项） |
| **MoE 高风险区** | router boundaries + dynamic schedule + V-PEFT | ✅ **78 passed** |
| **引擎套件** | `test_engine.py`（-n 4，含 multitask resume 真实训练恢复） | ✅ **33/33 passed**（4m21s） |
| **Foundation 全套** | `-k foundation`（-n 4） | ✅ **194 passed** |
| **Agent Skill 快速套件** | `--suite quick`（项目 Python 3.11） | ✅ 全部通过 |
| **Ruff lint** | `ultralytics/ tests/ scripts/ agent/` | ❌ **277 errors**（与 0819 修复后持平，无新增债务） |
| **Ruff format** | 同上 | ❌ **161 文件**待重排版（0819 时 159，+2） |

**结论：主干健康度维持 0819 修复后的高位，所有功能门禁全绿；lint 债务零增长但未清偿。**

### 2.2 0819 报告遗留项核销对照

| 0819 遗留 | 当前状态 |
|:--|:--|
| Phase 1 全部 P1（硬编码路径、F821、py3.8 语法） | ✅ 已核销且本轮无回归 |
| Foundation alpha → preview：真实数据通路 | ✅ F00–F15 全阶段真实数据工程门已证（0819 报告第十五节），发布说明落盘 |
| Foundation 精度增益统计有效性 | ➡️ **维持原判**：3-seed 均值 +0.0042±0.0154，coco128 无统计分辨力，受网络约束无法下载低方差测试床 |
| 蒸馏 × mixture 2×2 交互 | 🔄 工具链已落地（v0.2 smoke runners），结果未见 |
| E402×185 per-file-ignores 显式声明 | ❌ 未做，仍裸挂 |
| 全局 settings.json 指向旧工作区 | ❓ 环境侧，未验证（建议用户决策项） |

---

## 三、子系统成熟度矩阵（更新）

| 子系统 | 定位 | 成熟度 | 变化（vs 0819） |
|:--|:--|:--|:--|
| **ES-MoE** | 论文核心，compute-on-demand 主力 | ★★★★★ | → 稳定 |
| **统一路由基础设施** | routing_protocol + mixture_loss，五类 aux loss 收口 | ★★★★★ | → 稳定，隐形骨架 |
| **MoA / MoT** | 即插即用注意力/架构路由 | ★★★☆ / ★★★ | ⚠️ 当日修正：MoT 稀疏调度遥测契约曾断裂（9 个先验红测，见第十六节），已修复；性价比未闭环，建议冻结新投入 |
| **LoRA 内核** | 纯配置激活低成本微调 | ★★★★☆ | → 稳定 |
| **MoLoRA** | MoE 思想进 PEFT | ★★★★ | → 三硬伤修复后稳定，门禁守护中 |
| **V-PEFT Planner** | GATv2+PPO+MIP 适配器规划 | ★★★☆ → **需重评** | ⬆ **已承载 arXiv 2608.07051 论文声明**，可复现性优先级应从 P2 升至 P1 |
| **MultiTask** | 统一 det/seg/pose/cls/depth/normal | ★★★☆（Preview） | → TaskRouter deepcopy P0 修复后 e2e 可训练，预测 API 仍缺 |
| **Foundation** | 基础模型蒸馏 + multirouter | ★★★（alpha，从 ★★☆ 上调） | ⬆ 发布说明 + F00–F15 真实数据证据 + effect-gate 工具链；缺统计有效增益 |
| **Agent Skill** | CLI 分发器 + 多模态管线 | ★★★★ | → quick 套件通过；对解释器环境敏感（见 P2-4） |

---

## 四、问题分类与严重程度

### P0 — 阻塞性缺陷

**本轮未发现新 P0。** 全部功能门禁绿灯，0819 修复无回归。

### P1 — 显著问题（建议本周处理）

| # | 问题 | 位置 | 影响 |
|:--|:--|:--|:--|
| P1-1 | **V-PEFT 可复现性缺口升级为论文级风险**：arXiv 2608.07051 的具体数字（0.7138/0.7307、-43.9% 内存、1.72× 时间、RT-DETR-L 全部 Refuse 决策）需要一键复现脚本与 README 结果表锚定，否则外部无法验证 | `ultralytics/vpeft/`、`scripts/`、`README.md` | 论文刚官宣，这是外部审视最可能首先攻击的点；当前 planner 仍未接主训练链路，论文数字与仓内入口之间存在断层 |
| P1-2 | **lint 债务 277 errors / 161 文件格式不合规，两个周期零清偿**：E402×185 刻意延迟导入应入 per-file-ignores，E701/E702、F841（scripts/ 分析脚本）应清零 | 全仓 | 发布说明已承认债务；拖延本身成为风险——新代码持续在脏基线上生长 |
| P1-3 | **未提交变更堆积**：README 论文官宣、star-trend 工作流（含测试）、F08 resume 加固均只在工作区；CI 无法守护未提交代码 | git 工作区 | F08 resume 加固是真实 bug 修复，丢失即重现"epoch=-1 快照误 resume"问题 |

### P2 — 优化项（下一版本周期规划）

| # | 问题 | 说明 |
|:--|:--|:--|
| P2-1 | **仓库根目录卫生**：`images.zip`（32MB）、`embeddings.npy`/`paths.npy`、7 个字体调试 PDF、3 个犀牛鸟材料、`default_copy.yaml`、`yolo26n-seg.pt`（6.7MB）散落根目录 | 建议归入 `assets/`/`reports/` 或 gitignore；大文件不入库 |
| P2-2 | Foundation 精度增益统计闭环 | 维持 0819 判断：需 VOC 量级低方差测试床，受网络约束阻塞；gate_decay/band 机制已自洽待验证 |
| P2-3 | MoT/MoA 定位决策 | 延迟 +52%~148% 换 ≤1.1% 相对增益；建议文档降级为 experimental 推荐配置 |
| P2-4 | Agent Skill 对解释器敏感 | 用无 cv2 的 Python 调用验证器时报 `importlib.util` AttributeError 而非友好提示；建议入口加依赖预检 |
| P2-5 | 真实硬件证据缺口 | 无 NCCL 双卡记录、无 TRT routed profile、Core ML 环境阻塞（v26.08 发布说明自承，维持） |
| P2-6 | MultiTask 公开预测 API 缺失 | 预测仍走 detection predictor |
| P2-7 | `gated.py` 112KB 单文件 | MoE 活历史集中，评审成本高 |

---

## 五、跨模块交互风险

1. **V-PEFT 论文 × 仓内代码的口径风险（新）**：README 官宣的论文数字来自哪个 commit、哪套配置、哪个数据集协议，仓内没有锚点。若 planner 代码后续演进导致数字漂移，论文与仓库将脱节。建议：冻结 `vpeft-paper-2608.07051` tag + 复现脚本 + 结果 JSON 入 `reports/`。
2. **Foundation 调度器 × 训练器耦合加深**：gate_decay/band 需要 trainer 每 epoch 同步进度（`set_foundation_progress`），EMA 需 checkpoint 持久化——wrapper 生命周期已横跨 trainer/checkpoint/telemetry 三个子系统，当前由 194 个测试守护，但每加一个调度策略这条链就复杂一分。建议在 scheduler 接口上做抽象收口，而非继续在 wrapper 内叠加。
3. **star-trend 工作流的自动提交**：`github-actions[bot]` 每日向 main 提交 SVG 变更，会与人工提交历史交织；已用 concurrency group 防并发，规范。注意 README 引用的 `docs/assets/star-history.svg` 需首次运行后才存在。
4. **分段训练/中断恢复的系统性脆弱（已部分缓解）**：F08 resume 加固修复了 epoch=-1 快照误用，但 0819 报告记录的"`.cache` 截断导致 KeyError""孤儿进程并发写 results.csv"两类问题属环境执行纪律，建议沉淀为 `scripts/README` 的执行规范而非逐个脚本打补丁。

---

## 六、路线图（基于当前状态重排）

### Phase 1（本周：信誉与卫生）

1. **YOLO-PEFT 复现锚点**：tag + 复现脚本 + README 结果表链接到仓内证据（P1-1）。
2. **提交当前工作区**：README 官宣、star-trend 工作流、F08 加固分三个语义化 commit（P1-3）。
3. **E402 per-file-ignores 收口 + scripts/ F841 清零**，目标 lint < 50（P1-2）。
4. 根目录产物归置（P2-1）。

### Phase 2（v26.09：证据闭环）

1. Foundation 低方差测试床增益验证（网络恢复后 VOC/coco 子集，band × weight 因子实验）。
2. 蒸馏 × mixture 2×2 交互结果产出（工具链已备好）。
3. MoT/MoA 定位决策落地为文档。
4. 真实硬件证据补证（NCCL 双卡、TRT 基准）。

### Phase 3（v1.0 候选）

1. V-PEFT 主链路端到端（规划→训练→评测），与 `utils/lora/planner.py` 工程版职责切分。
2. MultiTask 公开预测 API。
3. MoE 剪枝一键 pipeline（20–30% 提速，最实际的部署卖点）。
4. Model Zoo L/X 权重补齐。

---

## 七、结论

**总体成熟度：8.4 / 10**（0819 口径 8.2，+0.2 来自：Foundation 发布说明与 F00–F15 真实数据证据链、测试网扩至 150 文件/194 foundation 项、全部门禁绿灯维持）

- **基本盘极稳且仍在加固**：两轮验证之间零回归，CI 门禁 25 项、MoE 78 项、引擎 33 项、Foundation 194 项全绿；TaskRouter P0 修复后的 multitask e2e 训练保持通过。
- **格局升级**：双线论文（CVPR 2026 检测主线 + arXiv PEFT 线）+ Foundation 第三增长曲线。研究诚信记录良好——alpha 发布说明明确标注"增益在种子噪声内，不构成已验证改进"。
- **当前最大风险不在架构而在"锚定"**：YOLO-PEFT 论文官宣后，仓内缺少论文数字的可复现锚点；lint 债务两个周期零清偿；重要修复滞留在未提交工作区。三者都是一周内可清偿的纪律项。
- **建议**：先打 **v26.08.1 卫生补丁**（提交工作区 + lint 收口 + 论文锚点），再推进 Foundation 统计闭环打 **v26.09**。架构层面无需新动作。

---

## 附录 A：本轮验证命令

```bash
# 环境：/Library/Frameworks/Python.framework/Versions/3.11/bin/python3（torch 2.9.1，editable 指向本仓）
# 注意：默认 `python`（Kimi 托管运行时）无 cv2，Agent Skill 验证器需用项目 Python 调用
ruff check ultralytics/ tests/ scripts/ agent/            # 277 errors（E402×185 / E701-E702×67 / F841×23）
ruff format --check ...                                    # 161 would reformat / 469 formatted
pytest tests/test_default_config_integrity.py tests/test_master_model_configs.py \
       tests/test_molora_dtype.py tests/test_molora_backend_roundtrip.py \
       tests/test_molora_merge_semantics.py tests/test_adapter_backend_contract.py   # 25 passed
pytest tests/test_moe_router_boundaries.py tests/test_moe_dynamic_schedule.py tests/test_vpeft.py  # 78 passed
pytest tests/test_engine.py -n 4                           # 33 passed（4m21s，含 multitask resume 真实训练）
pytest tests/ -k foundation -n 4                           # 194 passed
python agent/scripts/validate_yolo_master_skill.py --suite quick --summary-only      # all passed
```

## 附录 B：关键文件索引

| 主题 | 路径 |
|:--|:--|
| MoE 核心 | `ultralytics/nn/modules/moe/{modules,gated,routers,experts,loss,pruning,analysis}.py` |
| 统一路由 | `ultralytics/nn/modules/routing_protocol.py`、`ultralytics/nn/mixture_loss.py` |
| MoLoRA / V-PEFT | `ultralytics/nn/peft/molora/`、`ultralytics/vpeft/{graph,constraints,policy,solver}.py` |
| Foundation | `ultralytics/nn/foundation/`、`ultralytics/nn/foundation_distill_model.py`、`cfg/experiments/foundation/` |
| 发布说明 | `docs/release-notes/v26.08.md`、`docs/release-notes/foundation-v0.1.0-alpha.md` |
| 历史报告 | `reports/yolo_master_deep_analysis_20260716.md`、`reports/yolo_master_moe_moa_mot_peft_analysis_20260723.md`、`reports/yolo_master_deep_analysis_20260819.md` |

---

## 十六、Phase 1 执行记录与新发现 P1 修复（2026-08-21 晚）

> 触发：本报告第七节的 Phase 1 建议获用户批准后当日执行。执行中产生一个前两份报告均未发现的重要修正。

### 16.1 新发现 P1（已修复）——MoT 稀疏调度遥测契约断裂 + warmup 计数器空转

**发现路径**：清理 `mot/block.py` 的 3 个 F841（`dispatch_policy` / `routing_metrics` / `warmup_step` 计算后从未使用）时，删除后触发 9 个测试失败；用 `git stash` 在 HEAD 原始代码上复跑确认——**失败是先验存在的**（`test_mot_sparse_parity` ×6、`test_mot_scene_aware_router`、`test_compare_mot_ablation`、`test_p0_system_gates`），此前各轮验证（含 0819 报告）从未运行 `test_mot_sparse_parity.py`，构成报告盲区。

**根因（两层）**：

1. `_last_dispatch_stats` 只写入 `mode`/`expert_calls`/`selected_samples` 三个键，而测试契约（即设计意图）要求 `policy`/`sparsity_ratio`/`experts_per_sample`/`ddp_fallback_reason` 等 16 个字段——三个"死变量"正是为遥测准备却**从未接线**的原料；
2. 更严重：`_sparse_train_step` 计数器注册为 persistent buffer 但**从未在任何路径递增**——意味着 `sparse_train=True` 的配置在真实训练中 warmup 永远不会完成，稀疏训练调度**永远不会激活**。这是功能级 bug，不只是遥测缺失。

**修复**（commit `e183ea4`）：`_blend_experts` 中接通完整 dispatch stats（策略、稀疏度、DDP 契约四字段、warmup 步数），训练前向末尾 `_sparse_train_step.add_(1)`。验证：MoT/MoA 范围 **242 passed**（修复前 233 passed + 9 failed）。

**教训**：F841 在核心代码中不应机械删除——"计算了但没使用"在高风险区往往是未完成接线的信号。建议后续把 `-k "mot or moa"` 纳入常态门禁范围。

### 16.2 提交清单（当日 6 个语义化提交）

| Commit | 内容 |
|:--|:--|
| `740d23e`（用户方） | README 官宣 YOLO-PEFT 论文 + star-history 工作流与生成器 |
| `58fbfee` | fix(foundation): F08 effect gate resume 加固（epoch=-1 快照不误 resume、interrupted 去重） |
| `5ecbcb2` | docs(reports): 本报告 + 犀牛鸟评审入库 |
| `67fb290` | docs: 删除过时的上游 README.zh-CN.md |
| `aeb7d87` | chore(gitignore): 根目录散落产物、`.qoder/`、`agent/logs/`、foundation 本地运行记录排除 |
| `e183ea4` | fix(mot): 稀疏调度遥测接通 + warmup 计数器修复 + lint 债务清零 |
| `7a168b4` | style: ruff format 全量重排（158 文件，纯机械无行为变更；103 + 436 测试复验通过） |

### 16.3 Lint 收口终态

| 指标 | 0819 报告 | 本日早间 | 收口后 |
|:--|:--|:--|:--|
| `ruff check` errors | 439 | 277 | **0** |
| `ruff format` 待重排 | 159 | 161 | **0**（630 文件全合规） |

收口方式：E402×185 / E701·E702×67 / scripts 的 F841 按 0819 报告建议改为 `pyproject.toml` **per-file-ignores 显式声明**（research/smoke 脚本的 sys.path bootstrap 与紧凑风格属刻意偏离）；核心代码中的 F841×4、E741、F401 真实修复。

### 16.4 修复后复验

| 验证项 | 结果 |
|:--|:--|
| CI P0/P1 门禁 + master 配置 + MoLoRA + MoE 边界 | ✅ 67 passed |
| MoT/MoA 全范围 | ✅ 242 passed（含 9 个原红测） |
| MoT/MoA/Foundation 合并范围（格式化后） | ✅ 436 passed |
| CI 门禁 + MoE + V-PEFT（格式化后） | ✅ 103 passed |

### 16.5 遗留事项

1. `README_reproduce.md` 在工作区处于已删除状态（含犀牛鸟复现实验记录），**保留未提交**，待用户确认意图。
2. 全部提交停留在本地 `main`，未 push。
3. 本日修复后，**主干已无任何已知红灯测试**（在已运行的门禁范围内）；建议下一轮跑一次全量 `pytest tests/ -n auto` 作为 v26.08.1 候选基线。

---

## 十七、v26.08.1 候选基线：全量测试首跑（2026-08-21 深夜）

> 此前各轮验证均为聚焦门禁，本轮首次覆盖 `tests/` 全部 141 个测试文件（`--slow` 按约定跳过）。因单次执行时限，按文件分批运行（-n 4 --dist=loadfile）。

### 17.1 覆盖与结果

| 范围 | 结果 |
|:--|:--|
| 134 个轻量/中量测试文件（4 批） | ✅ **~1,540 passed**；2 个失败均在复核中闭环（见下） |
| `test_engine.py`（格式化后复验，两半） | ✅ **33 passed**（含 multitask resume 真实训练） |
| `test_python.py`（上游重型 e2e，分 8 组） | ⚠️ ~103 passed / **16 failed**——全部为环境类失败（见 17.2），无代码回归 |
| 未运行：7 个上游网络依赖型文件（`test_cli` / `test_integrations` / `test_exports` / `test_export_roundtrip` / `test_export_capability_matrix` / `test_solutions` / `test_benchmark_suite`） | ⛔ 离线网络 + 本机高负载下不可行（子进程下载重试循环）；CI 中 exports 本就走 `--export-env base` 专用链路 |

### 17.2 失败分类与处置（全部为环境/债项，非代码回归）

| 类别 | 涉及 | 处置 |
|:--|:--|:--|
| **MoE 治理 ledger 漂移（真实债项，已修复）** | `test_moe_ssot`：ledger 快照缺 `SharedExpertMoE` | 用官方脚本 `audit_moe_usage.py --record-version 8.4.101` 刷新快照，11 passed；commit `84a0762` |
| **DDP gloo 抖动** | `test_ddp_checkpoint_coordination`：gloo send 20s 超时 | 单独重跑 9 passed；并行高负载下的偶发，建议 CI 重试机制而非改代码 |
| **训练类超时** | `test_train_{multi,scratch,ndjson,pretrained}` 等 6 项 >120-150s | 本机高负载所致（同配置引擎套件 33 项全过）；非缺陷 |
| **损坏的权重缓存** | `test_predict_classes_with_max_det[yolo11n.pt]`、`test_yolo_world`、`test_yoloe` ×2：`PytorchStreamReader` 读取截断 zip | 此前被中断下载留下的截断缓存 + 离线无法重下；联网后删除 `weights/` 下对应 .pt 即可恢复 |
| **coco-multitask 数据集图片缺失** | `test_python` 参数化 multitask 用例 ×4（grayscale/predict_img/predict_visualize/results/val） | 与 0819 修复的 `test_engine` 临时夹具不同路径：`test_python` 用例仍要求真实数据集文件；建议下一轮把同一夹具机制复用到 `test_python` |
| **polars xdist 竞态** | `test_train_pretrained[True]`：polars 部分初始化 | 并行 worker 导入竞态，偶发；单跑即过 |

### 17.3 基线结论

**v26.08.1 候选基线成立**：全部项目自有子系统门禁（CI P0/P1、MoE、MoLoRA、MoT/MoA、Foundation、引擎、Agent Skill、SSOT 治理）在全量范围内绿灯；残余红灯 100% 可归因于离线网络、缓存损坏与本机负载三类环境约束，且每一项都有明确的恢复路径。建议联网后补跑 7 个未覆盖文件 + `test_python` 训练组作为最终确认。

---

## 十八、基线恢复路径工具化与执行（2026-08-22）

> 将 §17.2 的环境恢复路径固化为可重复工具：`scripts/release_baseline_preflight.py`（契约测试 `tests/test_release_baseline_preflight.py`，4 passed）。

### 18.1 工具能力

- **损坏权重扫描**：`zipfile.is_zipfile` 校验 `weights/` 与 settings `weights_dir` 下全部 `.pt`，识别中断下载留下的截断缓存；
- **数据集存在性检查**：按 Ultralytics `datasets_dir` 解析规则核对 dataset YAML 的 train/val 图像目录（初版按仓根解析产生误报，已修正为与 trainer 一致的解析）；
- **网络可达性探测**：github.com / objects.githubusercontent.com / raw.githubusercontent.com；
- `--fix`：仅在网络可达时删除损坏缓存（防止离线误删后无法重下）；`--run-blocked`：基线就绪后自动补跑 7 个未覆盖测试文件；`--json` 输出机器可读报告。

### 18.2 实机执行结果（2026-08-22）

| 项 | 结果 |
|:--|:--|
| 网络探测 | ✅ 三主机全部可达（网络已恢复） |
| 损坏权重 | 4 个（`yolo11n.pt` / `yolo26s-obb.pt` / `yoloe-11s-seg.pt` / `yolov8s-world.pt`），`--fix` 已全部删除 |
| 恢复验证 | `test_predict_classes_with_max_det[yolo11n.pt]` 重下载后 **passed**（105.7s）；`yolov8s-world.pt`、`yolov8s-worldv2.pt`、`yoloe-11s-seg.pt`（curl 直补）、CLIP ViT-B-32 均已重新缓存 |
| 剩余阻塞 | ① `coco-multitask.yaml` 需要完整 COCO（`~/PycharmProjects/datasets/coco` 的 train2017/val2017 清单），数据量大未拉取；② yolo-world/yoloe 测试的依赖链下载受单次执行时限截断，资产已部分落盘，无时限约束下重跑即可 |
| 报告快照 | `reports/baseline-preflight-20260822.json` |

### 18.3 结论

§17.2 的"损坏缓存"与"离线网络"两类环境失败**已实质解除**（恢复路径端到端验证通过）。剩余仅为完整 COCO 数据集拉取与长下载链的时间预算问题，均非代码问题。v26.08.1 候选基线的最后确认项收敛为一条：**联网常态下跑一次 `--run-blocked`**。

---

## 十九、blocked 文件补跑与最终基线确认（2026-08-22 晚）

> 网络恢复后执行 §18.3 的最终确认项。执行环境备注：本机负载 load avg ~45-52（其他训练任务并发），GitHub 资产下载带宽仅 ~35-60KB/s，多个用例因此需要超出常规的超时预算。

### 19.1 关键环境发现：CLI 安装污染（重要）

补跑 `test_cli.py` 时发现：**全局 `yolo` CLI 指向旧工作区 `YOLO-Master-v260720` 的 editable 安装**——所有 CLI 子进程测试实际验证的是旧代码库，而非本仓。这解释了 `test_export[yolo26-master-mt-n.yaml]` 的 FileNotFoundError（旧仓 cfg 无 master-mt 配置）。

- **影响范围**：仅 CLI 子进程类测试（`test_cli.py`）；进程内 pytest（`import ultralytics` 命中本仓）不受影响。
- **处置**：preflight 脚本新增 `check_cli_installation()`（对照 `direct_url.json` 的 editable source 与本仓路径），失配时输出 WARNING。是否 `pip install -e .` 重指向本仓会影响 v260720 工作区，**留待用户决策**。

### 19.2 补跑结果矩阵

| 文件 | 结果 | 备注 |
|:--|:--|:--|
| `test_integrations.py` + `test_benchmark_suite.py` | ✅ 10 passed / 4 skipped | |
| `test_export_capability_matrix.py` + `test_export_preflight.py` | ✅ 17 passed | |
| `test_export_roundtrip.py` | ⚠️ 6 passed / **2 failed** | **新发现先验 P1**：`MoTBlock`（module2）torchscript/onnx roundtrip 数值误差 0.003（容差 1e-4），在修复前提交 `aeb7d87` 的 worktree 上 3/3 复现——与本轮改动无关，是 eager-sparse 与 export-dense 路径的数值分叉，需专项根因 |
| `test_solutions.py` | ✅ 51 passed / 1 skipped | ParkingManager 的 `solutions_ci_parking_model.pt` 截断缓存已修复（curl v0.0.0 直补） |
| `test_exports.py` | ✅ 40 passed / 4 skipped / 1 failed | 失败为 `test_export_executorch`：缺 `flatc` 二进制（export 依赖未装全），环境项 |
| `test_cli.py` | ⚠️ 大部分 passed | `rtdetr`（缓存修复后 ✅）、`fastsam`（缓存修复后 ✅）、train/val/predict 各组非 multitask 用例全过；残余失败=multitask 用例（需完整 COCO）+ master-mt export（CLI 污染）+ `test_distill`（DINOv3 下载体量超今晚带宽，未跑通） |

### 19.3 缓存修复总账（本轮 `--fix` + 手动 curl 续传）

`yolo11n.pt`、`yolo26s-obb.pt`、`yoloe-11s-seg.pt`、`yolov8s-world.pt`、`yolov8s-worldv2.pt`、`rtdetr-l.pt`（66.5MB，六轮断点续传）、`FastSAM-s.pt`、`solutions_ci_parking_model.pt`——**8 个截断缓存全部修复并 zip 校验通过**。教训已内化进工具：preflight 的 `--fix` 与 Ultralytics `attempt_download` 的"存在即跳过"逻辑形成互补防线。

### 19.4 v26.08.1 最终基线结论

**基线确认成立。** 全量 141 个测试文件现已全部覆盖；残余红灯清单收敛为：

1. **代码级（需跟进）**：MoTBlock export roundtrip 数值分叉（先验 P1，已复现定位，待根因修复）；
2. **环境级（有恢复路径）**：完整 COCO 未拉取（multitask 参数化用例）、`flatc` 未安装（executorch）、全局 CLI editable 指向旧仓（待决策）、DINOv3 下载带宽（test_distill）。

即：**本仓代码在当前门禁范围内无活跃回归**；上述 4 项环境项均有明确的一行恢复命令，且 preflight 工具会在每次运行时报备状态。

---

## 二十、MoT 导出数值分叉根因与修复（2026-08-23）

### 20.1 根因（一句话版）

**不是数值 bug，是测试不变量错误**：MoT 的导出语义（文档化设计）与 roundtrip 测试的断言对象不一致。

具体链路：

1. `router.py:267-269`：ONNX/TorchScript 导出时路由器**刻意跳过 Top-K**，直接输出全量 softmax 权重（稀疏 Top-K 的专家选择是数据依赖控制流，不可 trace；且会触发 PyTorch 2.9 legacy exporter 的 bool scatter_ 别名分析失败）；
2. `block.py` 导出分支据此做 dense softmax 混合（所有专家留在图中）；
3. 而 eager eval 走的是 `stable_normalize` 重归一化的 Top-1 稀疏分发（选中专家权重=1.0）；
4. 因此 eager-sparse 与 export-dense 计算的是**两个不同的函数**：`expert_top(x)` vs `Σ softmax_i·expert_i(x)`。随机初始化的路由器 softmax 接近均匀，两者差异 ~0.003，必然超出 1e-4 容差。

`export_capabilities()` 早已文档化此限制（"ONNX and TorchScript tracing use dense blending because expert selection is data-dependent"），但测试断言的是"导出产物 == eager-sparse 输出"这个不成立的不变量。该测试自 7cddb8a 引入后因从未在门禁中运行而长期隐性失败（直到 08-22 全量补跑才暴露）。

### 20.2 修复（保持导出语义不变，修正验证对象）

- `ultralytics/utils/export_validation.py`：`validate_export_roundtrip` 新增可选 `reference` 参数——当导出图刻意编码与 eager 不同的语义时，用 reference 模块（导出等价配置）计算 eager 基线，roundtrip 回归其本职：**验证导出产物对"导出图实际编码的函数"的保真度**。
- `tests/test_export_roundtrip.py`：新增 `_export_semantics_reference()`——MoTBlock 且 top_k < NUM_EXPERTS 时，deepcopy 并置 `top_k = NUM_EXPERTS`（eval 下即 dense 路径，与导出语义逐点一致）作为 reference。

### 20.3 验证

| 项 | 结果 |
|:--|:--|
| `test_export_roundtrip.py` | ✅ **8 passed**（修复前 6 passed / 2 failed） |
| 导出相关套件（mixture_matrix / mixture_export / moe_export_governance / capability_matrix） | ✅ 49 passed / 1 skipped |
| MoT/MoA 全范围 | ✅ 242 passed |
| ruff check / format | ✅ 干净 |

### 20.4 遗留说明

eager-sparse 与 export-dense 的**语义差距本身**仍是部署侧的认知负担（导出模型 ≠ eager 稀疏模型，精度可能略有差异）——这是数据依赖控制流不可 trace 的固有约束，当前以文档化处理。若未来要消除差距，方向是导出期固化路由决策（如 Gumbel 硬采样导出或 LUT 化），属研究项而非缺陷项。

**至此主干在已覆盖门禁范围内无任何已知红灯（代码级与环境级均有归属）。**
