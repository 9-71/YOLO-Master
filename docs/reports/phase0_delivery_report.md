# F1 |  YOLO-Master Studio 平台内核 Phase 0 复盘与交付报告

> 全部数据基于 `rhino-f1-dev` 分支 `11303ca..c44d0b6` 真实演进轨迹复核，核心结论均可复现。

| 元数据 | 值 |
|---|---|
| 交付分支 | `rhino-f1-dev`（准入终点 HEAD 状态代码全部入库） |
| 准入基准 Commit | `11303ca` test(f1): complete entry smoke verification and contract specification |
| 交付终点 Commit | `c44d0b6` feat(ui): format Recent Jobs timestamps to local YYYY-MM-DD HH:MM:SS |
| 演进足迹（`HEAD~10..HEAD`，含准入基线提交） | **32 files changed, +8508 / −154** |
| 净交付口径（`11303ca..HEAD`） | 28 files changed, +7744 / −154 |
| 复测证据（发布前重新执行，显式业务测试清单口径） | 全量业务用例 **143/143 passed in 20.80s**；UI 模块 **50/50 passed in 12.79s**；Ruff check 全绿 |

> 附注：本报告配套的工程清理（`smoke/` 与 `smoke/f1/` 包级 `__init__.py` 补全、`smoke/f1/README.md` P0 清单同步勾选）位于交付终点之后的工作区变更，未计入上述提交统计。

---

## 一、P0 交付目标与完成状态矩阵

### 1.1 官方目标对齐

P0 保底目标原文：**「保留现有推理，新增任务列表、状态、日志和产物页；接入 predict 与 system doctor」**。以下矩阵逐项核验达成状态，证据全部指向可检索的文件与提交：

| # | 官方目标 | 落地实现 | 关键证据（文件 / Commit） | 达成状态 |
|---|---|---|---|---|
| 1 | 输入校验与契约准入 | Pydantic `JobRequest`/`SecurityConstraints` 强类型契约 + 提交期白名单校验 + Handler `validate_params` 二次把关；非法输入按标准化错误码短路（`PARAM_VALIDATION_FAILED` / `SEC_ERR_001`） | `dispatcher.py:111-226`；`handlers/base.py:35-147` | ✅ 达成 |
| 2 | 保留现有推理（Inference Studio） | 顶层 `gr.Tabs` 挂载两个兄弟 Tab；Inference Studio 代码块零改动，Jobs 全部状态封装于 `create_jobs_tab` / `JobsManager` 单例内，互不引用 | `app.py:327-445` | ✅ 达成（零侵入隔离） |
| 3 | 新增 Jobs 任务中心（列表/状态/日志/产物） | 右侧四子页签：Status Monitor（`gr.JSON` 状态看板 + 取消 + 告警横幅）、Live Logs（实时流式日志）、Artifacts（产物 Dataframe）、Recent Jobs（最近任务列表）；左侧任务提交表单（任务类型 / 超参 / 安全域配置） | `ui/jobs_tab.py:543-619` | ✅ 达成 |
| 4 | 接入 predict 与 system doctor | Agent Skills 层：`PredictSkill`（`yolo_predict`）与 `SystemDoctorSkill`（`system_doctor`）→ 强类型 `JobRequest` → 调度器 → 注册表动态解析 → `PredictHandler` / `DiagnoseHandler`（运行时环境自检：OS/Python/PyTorch/CUDA/Ultralytics 逐项探活） | `skills.py:43-441`；`handlers/predict.py`；`handlers/diagnose.py:156-231` | ✅ 达成 |
| 5 | Gradio 布局规范与状态绑定 | 单挂载生命周期契约 + 语言状态 `gr.State` 穿透 + 28 个组件输出的原子化语言切换绑定 | `ui/jobs_tab.py:502-530, 768-803` | ✅ 达成 |

### 1.2 规模总览

| 维度 | 实测数据 |
|---|---|
| 代码增量 | 32 files, +8508 / −154（其中准入基线 `11303ca` 自身 +764 行 / 4 文件）；净交付口径 28 files, +7744 / −154 |
| 核心模块 | `dispatcher.py` 268 行；`handlers/` 目录 8 文件共 **1,708 行**（P0 四件套 `registry`/`base`/`predict`/`diagnose` 计 862 行，`train`/`export` 为 P1 协议前瞻原型 688 行）；`skills.py` 441 行；`ui/jobs_tab.py` 877 行；`ui/i18n.py` 207 行；`app.py` 变更 377 行 |
| 测试资产 | `smoke/f1/` 下 7 个核心业务测试文件（累计 8 个测试与契约文件共 **2,966 行**）、**143 个 pytest 用例**；全量业务用例复测 **143/143 全绿（20.80s）**；UI 模块 50/50 全绿（12.79s） |
| 静态规范 | `ruff check smoke/f1/ui/` 与 `smoke/` 均 **All checks passed**；`ruff format --check smoke/f1/ui/` **6/6 files already formatted** |
| 文档工程 | `REGISTRY_ISOLATION_FIX.md`（注册表隔离复盘）、`SKILLS_IMPLEMENTATION.md`（技能层设计）、`handlers/USAGE.md`（Handler 框架使用指南）、`handlers/README.md`、`handlers/IMPLEMENTATION_SUMMARY.md`、`README.md`（P0 清单已同步勾选，v1.1.0） |

**测试数据链自洽性核验**：注册表隔离修复文档记录的历史基线与交付终点实测之间存在精确对应——修复后基线 **101/101**，此后随 Jobs Tab 模块新增 UI 集成用例 23 例 + JobsTab 管理层用例 19 例，`101 + 42 = 143`，与终点实测 143/143 完全吻合，无任何虚增或口径漂移。

---

## 二、准入基线与架构演进设计

### 2.1 准入现状痛点

准入时点的 WebUI 存在四重结构性短板： **(a)** 单图同步推理阻塞主线程，无异步长任务通道； **(b)** 任务无生命周期状态机，"进行中/失败"无标准可观测表达； **(c)** 无实时日志流与产物落盘管理，调试靠终端回显； **(d)** 任务分派逻辑与具体推理逻辑耦合，新增任务类型需改动主流程。

### 2.2 核心调度架构：线程安全的状态机 + 注册表解耦

**调度状态机（`JobDispatcherStateMachine`, `dispatcher.py:29`）。** 一个**纯编排、零硬编码任务类型分支**的同步状态机（模块 docstring 自述 "pure orchestration layer"，`dispatcher.py:7-9`），职责边界刻意收窄：

- **四态生命周期**：`JobStatus ∈ {PENDING, RUNNING, COMPLETED, FAILED}`；跃迁表（`dispatcher.py:60-65`）为 `PENDING → RUNNING | FAILED`、`RUNNING → COMPLETED | FAILED`，终态不可再跃迁，非法跃迁抛出 `Illegal state transition`；
- **取消语义**：框架层取消不是第五种状态，而是执行前检查 `cancel_requested` 后的**短路终止**——`FAILED + error_code=USER_CANCELLED`（`dispatcher.py:184-192`），UI 判定层另行保留 `CANCELLED` 仅作前向兼容。该设计避免了取消态在跃迁表上的二义性；
- **错误码阶梯**：`execute()` 内按序分类——`allow_shell=True` → `SEC_ERR_001`；路径未过白名单 → `SEC_ERR_001`；未注册任务类型 → `TASK_TYPE_UNKNOWN`；参数校验失败 → `PARAM_VALIDATION_FAILED`；执行返回失败 → `HANDLER_EXEC_FAILED`；未捕获异常 → `EXEC_ERR_500`（`dispatcher.py:162-266`）。错误码成为前后端契约的稳定面，UI 据此做告警归类（`SECURITY_ALERT_CODES` 集，`jobs_tab.py:47`）；
- **线程模型的精确边界**：调度器自身**零 `threading` 依赖、完全同步**——非阻塞由 UI 管理层提供：`JobsManager.submit_job` 每次提交派生一枚 `daemon=True` 工作线程（`jobs_tab.py:135-137`）；任务字典、日志流、取消标记以 `threading.Lock` 守卫的进程内共享结构承载，提供 `get_job_status / get_job_logs / get_job_artifacts / cancel_job / list_recent_jobs` 五个只读/控制 API（`jobs_tab.py:57-62, 185-291`）。同步编排 + 线程边界上移，使状态机可用单测全路径覆盖（非法跃迁、安全拒绝、取消短路均脱离线程即可断言）。

**插件化注册表（`TaskHandlerRegistry`, `handlers/registry.py`）。** 装饰器驱动、按 `task_type` 字符串索引的动态注册：类必须继承 `BaseTaskHandler`（否则 `TypeError`）、重复注册直接 `ValueError` 拒绝（`registry.py:106-126`）；具体 Handler 在 `handlers/__init__.py` 导入期自动完成注册。生命周期契约由 `BaseTaskHandler(ABC)` 锚定：抽象方法 `validate_params(params, security_constraints)`（白名单/禁 shell/资源约束）与 `execute(job_id, params, output_dir)`（返回 `success/artifacts/metadata/error` 结构化结果并强制按 `output_dir/job_id` 隔离产物），辅以 `_is_path_safe` 路径包含性工具（`base.py:35-147`）。

**测试期注册表强隔离。** 早期测试在 `setup_method` 中仅执行 `clear()`，永久清空类级单例 `_handlers`，导致下游 8 个用例连锁失败（`ValueError: Task type ... is not registered`）。修复落为 **backup → clear → restore** 夹具模式：每个用例前备份 `_handlers` 副本、清空隔离，用例后恢复（`test_handlers_framework.py:78-84, 167-199`），全套由 8 失败回到 **101/101 全绿**。完整复盘沉淀于 `REGISTRY_ISOLATION_FIX.md`。

### 2.3 前端自适应轮询引擎：双定时器

- **快轮询**：`POLL_FAST_SECONDS = 1.0`（`jobs_tab.py:49`），**初始关闭**（`active=False`）；
- **慢轮询**：`POLL_SLOW_SECONDS = 30.0`（`jobs_tab.py:51`），**常开保活**；
- **自熔断切换**：轮询回调末槽返回 `gr.update(active=keep_polling)`——当前任务进入终态（`COMPLETED/FAILED`）即 `keep_polling=False`，快轮询自动停止；**终态 tick 仍执行最后一次全量刷新**后熄灭（`jobs_tab.py:642-653`），杜绝"终态面板缺最后一帧"的竞态；提交新任务经 `submit_job_handler` 以 `gr.update(active=True)` 重新点火（`jobs_tab.py:690`）。慢轮询永不休眠，保证跨任务空闲期的面板新鲜度。空选择/无效取消亦降级熄灭快轮询（`jobs_tab.py:700-712`）。
- 设计收益：状态感知延迟 ≤1s 与客户端空转开销、Gradio 事件负载之间取得可量化折中；该机制由 UI 集成测试在状态层断言（keep_polling 语义矩阵）而非仅靠手工验证。

---

## 三、核心攻坚与典型缺陷复盘（代码级）

### 攻坚 1：Gradio DOM 双重挂载陷阱（`770db6d`）

**现象**：Jobs Tab 出现两份 DOM、语言切换事件绑定错位、部分 i18n 文本不随语言变化。

**根因（对 Gradio 挂载语义的误用）**：`create_jobs_tab()` 内部自带嵌套 `with gr.Blocks()` 上下文；而调用方在外部 `with gr.TabItem("📋 Jobs"):` **仍处于激活状态时**实例化该子 Blocks——上下文退出时 Gradio **自动嵌入**其组件——随后又在返回值上**追加显式 `.render()`**，于是全部 Jobs 组件被**第二次挂载**：

```python
# 修复前（770db6d^）：
with gr.TabItem("📋 Jobs"):
    create_jobs_tab(self.jobs_manager).render()   # 显式 render → 全体组件二次入 DOM
```

双重 DOM 使语言切换事件只有一个 output 目标却存在两份组件副本，绑定错位随之而来；另发现页标题 Markdown 从未进入语言切换事件的 outputs 列表，属于绑定缺漏。

**修复：确立"单挂载原则"**——子 Blocks 在激活上下文中以**裸调用**挂载，永不追加 `.render()`：

```python
# 修复后（app.py:414-419，注释即契约）：
# Single mount: entering the child Blocks inside this active context
# auto-embeds it on context exit; an explicit .render() would mount
# every Jobs component a second time (duplicate tabs in the DOM).
with gr.TabItem("📋 Jobs"):
    create_jobs_tab(self.jobs_manager)
```

同一契约同时固化进 builder docstring（"Mounting contract"，`jobs_tab.py:502-511`），并在集成测试中以布局树断言锁死：**每个组件 ID 全局唯一、6 个关键文本标记各恰好出现 1 次、定时器恰为 2 个**（快慢轮询各一，无重复队列/线程/事件源），语言切换输出绑定 28 个且两两互异（`test_app_integration.py:286-287, 336-359`）。此后任何"重复挂载"回归都会在测试层直接红掉——单挂载从约定升级为**可执行不变量**。

### 攻坚 2：框架级 Warning 误用导致的运行时崩溃（`e5df3c0`）

**现象**：安全告警触发时事件崩溃——`raise gr.Warning(...)` 抛出 `TypeError: exceptions must derive from BaseException`。

**根因**：该 Gradio 代次中 `gr.Warning` **并非 `BaseException` 子类**，只应作为"toast 触发"的语句调用；`raise` 一个非异常对象必然崩溃。更深一层：即使异常类型合法，**`raise` 会终止整个事件回调**，而告警发生的 `poll_snapshot` 必须继续走到面板刷新——一次告警即令状态监控停摆。

**修复：Toast 触发 + Fallback 返回的优雅降级**。安全告警位改为裸 `gr.Warning(toast_text)` 后**落穿至正常快照返回**（`jobs_tab.py:639`），docstring 明示取舍："raising terminates the event, whereas the snapshot below must still reach the monitoring panels"；`cancel_job_handler` 的三处 `raise gr.Warning(...)` 同样改为 toast + `return message, gr.update(active=False)`（消息落面板、快轮询安全熄灭，`jobs_tab.py:702-712`）。全仓库复扫：**`raise gr.Warning` 现存 0 处**。

### 攻坚 3：Pydantic 类定义期默认实例导致的时间戳共享与排序失效（`e5df3c0`）

**现象**：Recent Jobs 列表逐任务提交后无法按真实提交顺序排布。

**根因（类级默认值的时间绑定陷阱）**：领域模型以**类定义期实例**作为默认值：

```python
class JobRequest(BaseModel):
    ...
    metadata: Metadata = Metadata()   # 类加载瞬间创建一次，created_at 被冻结
```

提交路径未显式传 `metadata`，于是**每个任务的 `created_at` 都指向同一次类定义时刻**。而 `list_recent_jobs` 恰以该字段为排序键（`sorted(..., key=created_at, reverse=True)`，`jobs_tab.py:282`）——键值全等使排序退化为"先提交可能排后"，表象是排序失效，根子是元数据共享。

**修复：提交期动态注入**。任务构造后立即以当前 UTC ISO 时间戳回填：

```python
# 后端默认实例仅作结构模板，提交期注入真实创建时间（jobs_tab.py:125-129）
job_request.metadata.created_at = datetime.now(timezone.utc).isoformat()
```

回归测试以 3 个任务、`time.sleep(0.1)` 级差提交，断言列表**严格逆序**且三个时间戳**互不相同**（`test_jobs_tab.py:195-225`）。这一修复同时规范了"**时间戳一律 UTC ISO 入库**"的底层契约，为攻坚 4 的展示层本地化奠定数据基础。

### 攻坚 4：前后端契约解耦、展示层本地化与术语规范化（`82b4e1b` / `e5df3c0` / `c44d0b6`）

**契约纯洁性。** 状态 JSON 曾混入 UI 临时字段（`status_label` 等本地化文案直入数据层），导致"数据契约随语言环境漂移"。治理后：状态载荷收敛为纯英文标准键 `{job_id, status, duration, error_code, error_message, artifact_count}`，`NO_SELECTION` 分支收缩为 `{"status": "NO_SELECTION"}` 单一键；模块级约束写入 docstring——"**localized display text is confined to banners, toasts and column headers**"（`jobs_tab.py:428-433`），数据契约与展示文案从此物理分层。

**展示层本地化。** 底层时间戳保持纯净 UTC ISO（提交期注入、原样存储、原样排序）；仅当 Recent Jobs 行数据进入 UI 层时才经 `format_created_at()` 转换为本地可读时间：`datetime.fromisoformat(...).astimezone().strftime("%Y-%m-%d %H:%M:%S")`（`jobs_tab.py:318-348`），全程标准库、无第三方依赖；容错路径完备——空值渲染 `"-"`，非法字符串（`ValueError/TypeError`）**原样透传**而非崩溃（测试覆盖 `""`、`None`、`"garbage"` 三态）。

**术语规范化。** 中文文案将生硬的"作业"全面校准为业界通用"任务"（如 `提交作业→提交任务`、`取消作业→取消任务`、`📋 作业管理→📋 任务管理`、`作业状态→任务状态`），同步修正轮询提示与告警文案，回归断言与 doctest 逐条跟进（`82b4e1b`，代表性键对见 i18n 差异）。全仓库复扫：**"作业"一词在 `smoke/f1/` 下 0 残留**。

---

## 四、质量保障与防御性设计

### 4.1 安全沙盒防御

| 防线 | 机制 | 证据 |
|---|---|---|
| 路径白名单 | 提交期 `path_whitelisted` 门禁 + Handler `validate_params` 内 `_is_path_safe`：`Path.resolve()` 归一后做**父链包含性判定**，拒绝 `../` 与符号链接逃逸；空白名单**fail-closed** | `base.py:111-147`；`dispatcher.py:172-179, 214-226` |
| 目录遍历攻击面 | `"../../etc/passwd"` 类探针在调度器/Handler/UI 三层均有断言拦截 | `test_dispatcher.py:167-194`；`test_phase1_handlers.py:137-145, 275-283`；`test_jobs_tab.py:233-253` |
| Shell 执行禁用 | `allow_shell=True` 请求一律 `SEC_ERR_001` 拒绝 | `dispatcher.py:162-169` |
| 导出格式白名单 | 15 种格式闭集 + fail-closed | `handlers/export.py:26-44, 132-136` |
| 技能层边界校验 | `parameters_schema`（JSON Schema，required + conf 区间 `(0,1]`）在进入调度器前短路非法输入 | `skills.py:246-290, 341-371` |
| 标准化错误码 | `SEC_ERR_001` / `PARAM_VALIDATION_FAILED` / `TASK_TYPE_UNKNOWN` / `HANDLER_EXEC_FAILED` / `EXEC_ERR_500` / `USER_CANCELLED` 全链路唯一语义 | `dispatcher.py:111-266` |

### 4.2 自动化测试矩阵

`smoke/f1/` 累计包含 8 个测试与契约文件（2,966 行），其中由 **7 个核心业务测试文件**承载全部 143 个 pytest 用例（UI 集成测试 23 + JobsTab 单测 19 构成交付前最新增量）：

| 覆盖维度 | 承载用例 | 状态 |
|---|---|---|
| 调度器状态跃迁（合法序 + 非法跃迁拒绝） | `test_dispatcher.py`（12） | ✅ |
| Handler 框架契约与注册表（重复注册拒斥、隔离夹具） | `test_handlers_framework.py`（10） | ✅ |
| predict / diagnose 双 Handler（注册/校验/产物/隔离） | `test_predict_diagnose.py`（26） | ✅ |
| 技能层（元数据/动态白名单/fail-closed/错误传播） | `test_skills.py`（15） | ✅ |
| P1 协议原型（train/export 校验矩阵、ONNX 真实导出） | `test_phase1_handlers.py`（38） | ✅ |
| UI 管理层（时间戳本地化容错、逆序排序、并发提交、取消） | `test_jobs_tab.py`（19） | ✅ |
| 应用集成（双定时器、单挂载断言、28 输出语言绑定、空选择防御、headless 启动） | `test_app_integration.py`（23） | ✅ |

**交付终点实测（显式业务测试文件清单，exit 0）**：

- 全量业务用例：**143 passed in 20.80s，0 失败**；
- UI 模块：**50 passed in 12.79s，0 失败**；
- 唯一告警为 Ultralytics 上游 ONNX 导出的 `TracerWarning`（上游库告警，非本仓库代码）。


### 4.3 静态规范合规

`ruff check smoke/f1/ui/` → **All checks passed**；`ruff format --check smoke/f1/ui/` → **All files already formatted**；扩展至 `smoke/` 全包同样 0 告警，符合项目 Ruff（line-length=120、Google docstring）规范。

### 4.4 文档工程

`REGISTRY_ISOLATION_FIX.md`（隔离缺陷复盘 + 修复模式）、`SKILLS_IMPLEMENTATION.md`（技能层架构与错误码清单）、`handlers/USAGE.md`（框架使用指南）、`handlers/IMPLEMENTATION_SUMMARY.md`，配合逐模块 doctest 与 docstring 契约，形成"代码即文档、文档可回归"的双保险。P0 清单在 `README.md` 中同步勾选并升级文档版本（v1.1.0）。

---

## 五、Phase 1（P1）演进蓝图与落地规划

对齐官方目标：**「P1｜统一 train/val/predict/export 协议，支持取消、超时、批量推理、路径白名单与错误回传」**。基于 P0 沉淀的现状差距，规划三条主线：

### 1. 任务协议全量泛化（协议层）

P0 已顺手验证了协议的可扩展性：`train.py`（429 行）与 `export.py`（259 行）原型连同 38 个用例已在分支内随调度器架构落地，证明"注册表 + 生命周期契约"对新任务类型是零侵入接入。P1 补齐矩阵：

| 差距项 | P0 现状 | P1 动作 |
|---|---|---|
| `val` Handler | 未注册 | 补 `ValHandler`，对齐同一生命周期契约 |
| 超时控制 | 无 | 状态机注入 deadline 监督，超时 → `FAILED + TIMEOUT` 标准化码 |
| 批量推理 | 单任务语义 | `predict` 参数面扩展批量数据源，产物按序归档 |
| 错误回传 | 结构化 error 已就绪 | 错误消息契约升级为可机器消费的错误码 + 详情结构 |
| 取消一致性 | 执行前短路取消 | 扩展为运行中协作式取消（handler 检查点） |

### 2. 全局 UI 统一中英广播（状态提升）

P0 的语言状态（`gr.State`）封装于 Jobs Tab 内，Inference Studio 仍为硬编码英文——这是 P0 零侵入隔离的刻意取舍。P1 将语言状态**提升至应用顶层**，`gr.Radio` 广播驱动全部 Tab 的 i18n 重绑定；同时把 Jobs Tab 验证成熟的 28 输出绑定模式抽象为**通用语言广播工具**，供 Studio 与后续 Tabs 复用。

### 3. 架构转正（生产化晋级）

`smoke/f1/` 试验模块经受住 P0 全链路验证后，P1 向主工程生产目录晋级：将验证成熟的调度器、注册表、Handler 框架与 Jobs UI 移入正式包命名空间；**领域模型（`JobRequest` 等）当前仍寄居于准入期的 `test_f1_smoke.py` 遗留模块**（历史债务），转正时需拆分为独立生产模型模块并统一导入源——这一层清理是 P1 的第一批技术债偿还。

P0 收尾已先行完成其中一项**包结构正规化**：补全 `smoke/` 与 `smoke/f1/` 的包级 `__init__.py`，将 namespace 包固化为正规 Python 包，消除了跨模块导入时的命名空间歧义，`ruff check smoke/` 保持全绿。晋级以“测试资产随迁 + 零功能回归”为验收口径。

---

## 附：验证口径与复现命令

```bash
# 全量业务用例（显式清单，143/143 实测 20.80s）
python -m pytest smoke/f1/test_dispatcher.py smoke/f1/test_handlers_framework.py \
  smoke/f1/test_predict_diagnose.py smoke/f1/test_phase1_handlers.py \
  smoke/f1/test_skills.py smoke/f1/ui/test_jobs_tab.py smoke/f1/ui/test_app_integration.py -q

# UI 模块（50/50 实测 12.79s）
python -m pytest smoke/f1/ui/ -q

# 静态规范
ruff check smoke/f1/ui/ && ruff format --check smoke/f1/ui/
```

*本报告全部数据基于 `rhino-f1-dev` 分支 `11303ca..c44d0b6` 真实演进轨迹复核，核心结论均可在提交差异与测试用例中复现。欢迎社区评审与共建，P1 蓝图细节与分支状态将持续在 Discussions 同步。*
