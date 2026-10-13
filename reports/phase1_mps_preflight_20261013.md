# 第一阶段实施结果：MPS 基线与 A800 迁移准备

日期：2026-10-13。基于 `8b00422` 的未提交工作区。

## 已落地

- 建立 `.venv`，editable 安装本仓库；Python 3.11.16、torch 2.14.1、MPS Apple M5 Pro。
- 新增 `benchmarks/hardware_baselines.yaml`，分别提供 MPS 160 输入的结构预检和 A800 640 输入的 forward 预检；原快速 smoke 默认值保持不变。
- 新增 `benchmarks/PROTOCOL.md`：明确推荐实验角色、实际训练/部署证据门、MPS 与 A800 迁移命令、评测边界和候选准入指标。
- 修复 `benchmark_mot_dispatch.py`：旧程序两组均为 eval sparse，且分别随机初始化，无法构成 dense/sparse 对照。现在共用 block、输入和 masked 权重，使用显式 dense 参考与实际 sample-sparse 分发；同步计时、禁用梯度、校验输出，并通过 untimed hooks 记录实际调用与样本—专家数量。
- 新增可复用 `scripts/run_hardware_preflight.py`，执行 dense/MoE 训练、有限损失检查、checkpoint 哈希、重载、验证和预测。
- 中英文 README 的 LoRA 速度/显存宣传改为论文受控审计口径：峰值显存降低 43.9%，训练时间为 Full-SFT 的 1.72 倍，并说明适用条件。
- 修复既有动态调度回归测试的 step 设置：两个独立 forward 先后调用 `begin_aux_step()`。辅助损失协议会保留同一步首次发布，原测试误将两次 forward 当作同一步；没有改动模型或防重复发布协议。

## MPS 实测结果

**真实训练闭环：dense 与 MoE 均通过。** coco8，1 epoch，imgsz=64，batch=2，workers=0，AMP 关闭。两组均保存 last/best checkpoint、训练损失有限、重载后验证完成，并对一个合成输入返回一个预测结果。

两组 AP 均为 0。这是随机初始化、极小数据和一个 epoch 的工作流预检，不能用于精度比较。MPS scatter/index 操作有确定性实现警告，因此也不宣称位级训练复现。

**MoT 对照：四种路由模式输出最大绝对误差均为 0。** 在 `[3,32,16,16]`、top-1 下，三种受控模式的 token 掩码稀疏率均约 66.7%：

- 单一专家：dense 执行 9 个样本—专家组合，sparse 执行 3 个。
- 按样本拆分专家：dense 执行 9 个组合，sparse 执行 3 个，仍需调用三个专家。
- 空间覆盖全部专家：dense 与 sparse 均执行 9 个组合，实际样本—专家计算没有减少。

随机 native 路由在本种子下执行 3 个组合；这不是已训练路由器的表现。计时为本地非独占窗口的模块诊断，包含并发 CPU 验证影响，不能作为正式加速声明。

**完整模型 forward 预检：dense、MoE、MoT 三组均成功。** MPS、FP32、batch=1、160 输入、warmup=5、30 次同步采样。随机权重、本地窗口，未测 AP、导出或端到端服务延迟；不据此选择正式推荐模型。

## 验证状态

- 基准 runner 与新增诊断测试：14 passed。
- 配置、模型构建、MoLoRA dtype/roundtrip/merge/adapter、路由边界、动态调度与 RoutedModule 协议：135 passed。
- Agent quick：36/36 passed。
- Studio 限定子集：294 passed、1 deselected；排除了 worker_lifecycle、worker_reaping、worker_ipc 三个文件及 accepted_cancel_wins_during_cooperative_shutdown，不能表述为 Studio 全套通过。
- 论文声明锚点检查、改动 Python 文件 Ruff lint/format、所选新增文件 codespell 和 git diff whitespace 检查通过。
- 全仓检查仍不通过：本环境 Ruff check 报 2,932 项，format 报 47 文件；codespell 退出码 65。日志在 `runs/verification/phase1/`。未做批量格式化或无关修复，避免与基准建设混在一起；这些是当前环境测得的全仓基线，不直接等同于历史报告的工具版本与规则。

## 已发现但尚未解决的 Studio 问题

完整混合回归在 macOS 真实进程生命周期测试中出现清理超时与取消竞态；该运行在 380 passed、9 failed、6 teardown errors 时中断，不能作为完整测试结果。已清理本轮测试创建的存活进程。

具体机制：Linux 使用 subreaper guardian；macOS 无该分支。停止时等待 zombie 被回收可能阻止其仍持有的父进程退出；崩溃后脱离 session 的后代也可能在被扫描到之前失去父子关系。尝试放宽 zombie 等待后，真实 crash 测试揭示了存活后代泄漏，因此已经撤回该尝试，生产 runtime 保持原样。

该问题应独立设计 macOS 进程监管与可证明的后代登记/退出协议，并用正常完成、取消、超时、崩溃、父进程死亡和脱离 session 的真实进程测试验收。不要通过跳过测试或提前标记任务终态解决。当前 MPS 实验使用 CLI/Python 同步路径，不依赖 Studio 监管。

另有 `test_accepted_cancel_wins_during_cooperative_shutdown` 出现进程已退出时的 `psutil.NoSuchProcess`；尚未完成独立根因验证。

## 下一阶段顺序

1. 用当前预检配置，在 MPS 上增加充分规模的小数据训练与已训练 checkpoint 诊断；正式速度实验移至 A800，不把本地随机模型时延当作收益。
2. A800 清洁环境跑同一协议与 P0/P1 门；固定数据划分、初始权重、训练预算和至少三种子。现阶段尚未连接或运行 A800。
3. 对 trained checkpoint 开展专家互补性、固定专家和当前 router 的受控对照；确认可优化空间后再做预算约束/效用路由。
4. 按同样证据标准开展场景 PEFT 总成本实验，再决定 Foundation 2×2 投入。
5. Studio 的 macOS 监管修复先于接入自动化长任务，避免已知清理问题扩大。

## 本地产物

- `runs/hardware_preflight/mps/preflight.json`：真实训练闭环结果、权重哈希与 CSV 路径。
- `runs/hardware_preflight/mps/{dense,moe}/weights/last.pt`：预检权重，不用于正式准确率声明。
- `runs/benchmarks/mps_dispatch/results.json`：四种路由的共享权重对照。
- `runs/benchmarks/mps_preflight/{results.json,results.csv,report.md}`：三模型 forward 预检。
- `runs/verification/phase1/dependencies.txt`：本地环境快照；含 macOS/editable 路径，不能直接用作 A800 的跨平台安装锁。
- `runs/verification/phase1/studio-unit.xml`：限定 Studio 子集报告。

`runs/` 按仓库策略保留在本地；代码与协议可以提交，实际权重和原始实验日志无需加入 Git。
