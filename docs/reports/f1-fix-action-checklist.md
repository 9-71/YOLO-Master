# F1 课题修复操作清单

> 基于《F1 课题完成度评估报告》生成，按 **"功能先行 → 验证 → 文档"** 的开发时序排序。执行前请确认当前处于 `rhino-f1-dev` 分支。

***

## 📋 P0 紧急（结项前必做，约 1 小时）

> 纯文档/历史清理类工作，不依赖任何开发，可以立刻动手。

### 1. 更新 README P1 Checklist（10 分钟）

**文件**：`smoke/f1/README.md`

**操作步骤**：

1. 打开 `smoke/f1/README.md`，定位到 P1 checklist 部分（约第 298-306 行）
2. 将以下 3 项的 `[ ]` 改为 `[x]`：

   - [x] Unify `train`, `val`, `predict`, `export` task contracts

   - [x] Implement job cancellation, watchdog timers, and timeout mechanisms

   - [x] Add batch image/video inference support
3. 保留第 4 项 `Enhance path whitelisting with regex patterns` 为未勾选状态
4. 在 P1 checklist 上方添加一行状态说明：

   ```
   **P1 进度：3/4 项已完成，regex 白名单增强待实现**
   ```
5. 更新文档顶部的 Status 行，将版本号递增（如 `v1.2.0`）

**验证**：`grep -c '\[x\]' smoke/f1/README.md` 应显示 P0 5 项 + P1 3 项 = 8 项勾选

***

### 2. 清理 Git 重复提交（30 分钟）

**背景**：存在 3 对提交消息重复的提交，影响 PR 评审体验。

**重复提交对**：

| 前提交       | 后提交       | 消息                                                                                            |
| --------- | --------- | --------------------------------------------------------------------------------------------- |
| `3d1cc7f` | `f85985a` | feat(f1): implement decoupled task dispatcher and core handlers (P0/P1)                       |
| `8ff2fd2` | `669c9d8` | feat(p1): implement ValHandler, timeout watchdog, cooperative cancellation, and batch predict |
| `df29492` | `c13eeba` | refactor(core,ui): decouple domain schema to core/schema and lift global i18n broadcast       |

**操作步骤（交互变基方案）**：

```bash
# 1. 确保工作区干净
git status

# 2. 启动交互式变基，从准入基线开始
git rebase -i 11303ca

# 3. 在变基编辑器中，将重复的提交标记为 squash（或 fixup）
#    找到每对重复提交，将后者的 pick 改为 squash（保留提交消息）或 fixup（丢弃消息）
#
#    示例：
#    pick 3d1cc7f feat(f1): implement decoupled task dispatcher and core handlers (P0/P1)
#    squash f85985a feat(f1): implement decoupled task dispatcher and core handlers (P0/P1)  ← 改为 squash
#    ...
#    pick 8ff2fd2 feat(p1): implement ValHandler, watchdog timeout, cooperative cancellation, and batch predict
#    squash 669c9d8 feat(p1): implement ValHandler, timeout watchdog, cooperative cancellation, and batch predict  ← 改为 squash
#    ...
#    pick df29492 refactor(core,ui): decouple domain schema to core/schema and lift global i18n broadcast
#    squash c13eeba refactor(core,ui): decouple domain schema to core/schema and lift global i18n broadcast  ← 改为 squash

# 4. 保存退出，变基会自动合并重复提交

# 5. 验证历史是否干净
git log --oneline 11303ca..HEAD

# 6. 强制推送（如果已推送到远程）
git push --force-with-lease origin rhino-f1-dev
```

**⚠️ 注意事项**：

- 变基会重写历史，如果团队其他成员基于旧提交开发，需提前同步

- 如果不确定，可先在临时分支操作：`git checkout -b rhino-f1-dev-clean`

- 每对重复提交需要确认两者的代码差异是否为增量，还是完全重复

**验证**：`git log --oneline 11303ca..HEAD | wc -l` 应从当前约 14 条减少到约 11 条

***

### 3. 修复 Phase 0 报告中的文件路径引用（5 分钟）

**文件**：`docs/reports/phase0_delivery_report.md`

**问题**：规模总览表格中提到 `IMPLEMENTATION_SUMMARY.md` 位于 `smoke/f1/` 顶层，但实际文件在 `smoke/f1/handlers/IMPLEMENTATION_SUMMARY.md`。

**操作步骤**：

1. 打开 `docs/reports/phase0_delivery_report.md`
2. 定位到"文档工程"行（约第 40 行）
3. 将 `IMPLEMENTATION_SUMMARY.md` 的路径修正为 `handlers/IMPLEMENTATION_SUMMARY.md`
4. 搜索全文是否还有其他 `smoke/f1/IMPLEMENTATION_SUMMARY.md` 引用，一并修正

**验证**：`grep -n "IMPLEMENTATION_SUMMARY" docs/reports/phase0_delivery_report.md` 确认路径正确

***

## 📋 P1 高优先级（功能闭环，约 4-5 小时）

> 先把 P1 最后一块功能拼图补上，复测通过，再写文档。避免功能还在变、文档反复改。

### 4. 实现 regex 路径白名单增强（3-4 小时）

**背景**：P1 checklist 第 4 项 `Enhance path whitelisting with regex patterns` 未实现。当前 `_is_path_safe()` 仅做 `Path.resolve()` 父链包含性判定，不支持 regex 模式匹配。

**涉及文件**：

- `smoke/f1/handlers/base.py` — `_is_path_safe()` 方法

- `core/schema.py` — `SecurityConstraints` 模型

- `smoke/f1/ui/jobs_tab.py` — 表单输入

- `smoke/f1/test_phase1_handlers.py` — 补充测试

**实现方案**：

1. **扩展** **`SecurityConstraints`**：新增 `path_patterns` 字段（`list[str]`，regex 模式列表），与 `allowed_paths`（目录白名单）并存
2. **扩展** **`_is_path_safe()`**：在现有目录包含性判定基础上，增加 regex 模式匹配逻辑——路径需满足"在目录白名单内 **且** 匹配任一 regex 模式"（或逻辑，取决于设计）
3. **扩展 UI 表单**：添加 regex 模式输入框（多值文本域，每行一个 pattern）
4. **补充测试用例**：

   - regex 匹配通过 / 不通过

   - 空 pattern 列表的行为

   - 非法 regex 表达式的异常处理

   - 目录白名单 + regex 联合判定

**验收标准**：

- 所有已有测试零回归

- 新增 regex 测试用例全部通过

- README P1 checklist 第 4 项勾选为完成

***

### 5. 执行当前 HEAD 复测并更新基线（30-60 分钟）

**时机**：regex 增强完成并提交后执行，确保 P1 功能全部闭环。

**操作步骤**：

```bash
# 1. 运行全量业务测试
python -m pytest smoke/f1/test_dispatcher.py \
  smoke/f1/test_handlers_framework.py \
  smoke/f1/test_predict_diagnose.py \
  smoke/f1/test_phase1_handlers.py \
  smoke/f1/test_skills.py \
  smoke/f1/test_val_batch_runtime.py \
  smoke/f1/ui/test_jobs_tab.py \
  smoke/f1/ui/test_app_integration.py -v

# 2. 运行 UI 模块测试
python -m pytest smoke/f1/ui/ -v

# 3. 静态规范检查
ruff check smoke/f1/
ruff format --check smoke/f1/

# 4. 记录结果
#    - 总用例数 / 通过数 / 失败数
#    - 执行时间
#    - 告警信息
```

**更新文件**：

- `smoke/f1/README.md` 顶部状态行（更新测试通过数和版本号）

- P1 checklist 第 4 项改为 `[x]`

- 删除 P1 进度行，改为 **"P1 进度：4/4 项全部完成"**

**预期结果**：

- 测试用例数约 175+（regex 增强新增约 4-6 个用例）

- 全部通过（0 失败）

- ruff check / format 全绿

***

## 📋 P2 中优先级（交付物补齐，约 4-6 小时）

> P1 功能全部闭环、复测通过后，再一次性补齐文档交付物。此时功能稳定，文档不需要反复修改。

### 6. 补齐操作手册交付物（2-3 小时）

**新建文件**：`smoke/f1/USER_GUIDE.md`（或 `docs/studio/USER_GUIDE.md`）

**内容大纲**：

```markdown
# YOLO-Master Studio 操作手册

## 1. 快速开始
- 环境要求
- 启动方式（`python app.py`）
- 浏览器访问地址

## 2. Inference Studio 使用指南
- 模型选择
- 图片上传 / 摄像头
- 推理参数（conf、imgsz）
- 结果查看与下载

## 3. Jobs 任务中心使用指南

### 3.1 提交任务
- 任务类型选择（predict / train / val / export / diagnose）
- 各任务类型参数说明
- 安全域配置（路径白名单 + regex 模式）
- 提交按钮与预设恢复

### 3.2 状态监控
- Status Monitor 面板说明
- 状态含义（PENDING / RUNNING / COMPLETED / FAILED）
- 错误码解读（SEC_ERR_001 / PARAM_VALIDATION_FAILED / TIMEOUT / USER_CANCELLED 等）
- 取消任务操作

### 3.3 日志查看
- Live Logs 面板
- 日志刷新机制（快慢轮询）

### 3.4 产物管理
- Artifacts 面板说明
- 产物下载操作
- 产物隔离规则（按 job_id 分目录）

### 3.5 历史任务
- Recent Jobs 列表
- 排序规则（按提交时间倒序）
- 任务状态查看

## 4. 超时控制
- 默认超时时间
- 如何配置 timeout
- 超时后的行为

## 5. 批量推理
- 支持的输入格式（文件列表 / 目录）
- 批量大小配置
- 产物组织方式

## 6. 语言切换
- 中英文切换入口
- 支持范围

## 7. 常见问题（FAQ）
- 任务提交后无反应怎么办
- 任务失败如何排查
- 产物找不到怎么办
- 安全错误如何处理
- 任务超时了怎么办
```

**验证**：文档结构完整，覆盖所有 UI 功能模块，截图可选

***

### 7. 补齐安全测试报告交付物（1-2 小时）

**新建文件**：`smoke/f1/SECURITY_TEST_REPORT.md`

**内容大纲**：

```markdown
# F1 平台内核安全测试报告

## 1. 测试范围
- 安全红线三项：禁 shell、路径白名单、敏感环境变量过滤
- 测试对象：Dispatcher 层、Handler 层、Skill 层、UI 层

## 2. 测试方法
- 单元测试断言
- 渗透测试探针
- 代码审计

## 3. 测试用例矩阵

### 3.1 Shell 执行禁用（4 层纵深）
| 层级 | 测试用例 | 文件位置 | 结果 |
|------|----------|----------|------|
| Dispatcher | allow_shell=True 拒绝 | test_dispatcher.py | ✅ 通过 |
| Handler | allow_shell=True 拒绝 | test_predict_diagnose.py | ✅ 通过 |
| Skill | 固定构造 allow_shell=False | test_skills.py | ✅ 通过 |
| UI | 表单硬编码 allow_shell=False | jobs_tab.py | ✅ 通过 |

### 3.2 路径白名单（3 层纵深）
| 层级 | 测试用例 | 文件位置 | 结果 |
|------|----------|----------|------|
| Dispatcher | path_whitelisted=False 拒绝 | test_dispatcher.py | ✅ 通过 |
| Handler | 路径超出白名单拒绝 | test_phase1_handlers.py | ✅ 通过 |
| UI | 路径遍历攻击拦截 | test_jobs_tab.py | ✅ 通过 |

### 3.3 regex 路径模式增强
| 场景 | 测试位置 | 结果 |
|------|----------|------|
| 合法 regex 匹配通过 | test_phase1_handlers.py | ✅ 通过 |
| 不匹配 regex 拒绝 | test_phase1_handlers.py | ✅ 通过 |
| 非法 regex 异常处理 | test_phase1_handlers.py | ✅ 通过 |

### 3.4 目录遍历攻击
| 攻击向量 | 测试位置 | 结果 |
|----------|----------|------|
| ../../etc/passwd | test_dispatcher.py | ✅ 拦截 |
| ../../etc/passwd | test_phase1_handlers.py | ✅ 拦截 |
| ../../etc/passwd | test_jobs_tab.py | ✅ 拦截 |

### 3.5 空白名单 fail-closed
| 场景 | 测试位置 | 结果 |
|------|----------|------|
| allowed_paths 为空拒绝 | test_predict_diagnose.py | ✅ 通过 |

### 3.6 导出格式白名单
| 场景 | 测试位置 | 结果 |
|------|----------|------|
| 15 种格式闭集校验 | test_phase1_handlers.py | ✅ 通过 |
| 不支持格式拒绝 | test_phase1_handlers.py | ✅ 通过 |

## 4. 代码审计发现
- _is_path_safe 实现：Path.resolve() 归一 + 父链包含性判定 + regex 模式匹配
- 标准化错误码 7 种，全链路唯一语义
- fail-closed 策略贯穿所有安全门禁
- 四层 shell 禁用纵深防御

## 5. 已知限制与改进建议
- 敏感环境变量日志过滤：当前未显式实现，建议补充日志脱敏过滤器
- 单进程内存态：任务状态进程内存储，不支持多 worker 部署

## 6. 结论
安全红线中，禁 shell 和路径白名单两项实现了纵深防御并通过全部测试；敏感环境变量过滤建议补充实现。整体安全姿态在课题级别已属优秀。
```

**验证**：覆盖所有安全测试用例，结论与代码实际状态一致

***

## 📋 P3 中优先级（生产化准备，约 2-3 天）

> 结项答辩通过后再做的工程化改进。不影响课题验收，但影响长期可维护性。

### 8. 架构转正：smoke/f1/ → 生产包命名空间（1-2 天）

**目标**：将 `smoke/f1/` 下的核心代码迁移到正式包命名空间，例如 `ultralytics/studio/` 或新建顶层包。

**迁移清单**：

| 模块         | 源路径                      | 目标路径（建议）                           | 注意事项      |
| ---------- | ------------------------ | ---------------------------------- | --------- |
| 领域模型       | `core/schema.py`         | `ultralytics/studio/schema.py`     | 统一导入源     |
| 调度器        | `smoke/f1/dispatcher.py` | `ultralytics/studio/dispatcher.py` | —         |
| Handler 框架 | `smoke/f1/handlers/`     | `ultralytics/studio/handlers/`     | 保留注册表隔离测试 |
| Skill 层    | `smoke/f1/skills.py`     | `ultralytics/studio/skills.py`     | —         |
| UI 模块      | `smoke/f1/ui/`           | `ultralytics/studio/ui/`           | —         |
| 测试         | `smoke/f1/test_*.py`     | `tests/test_studio_*.py`           | 保持测试资产随迁  |
| 文档         | `smoke/f1/*.md`          | `docs/studio/*.md`                 | —         |

**验收口径**：

- 零功能回归（全量测试通过）

- 导入路径更新完成，无 `from smoke.f1...` 残留

- `smoke/f1/` 仅保留历史归档或完全移除

***

### 9. CI 流水线集成（2-4 小时）

**目标**：将 F1 测试套件绑定到项目 CI 配置。

**操作步骤**：

1. 打开 `.github/workflows/` 下的 CI 配置
2. 添加 F1 测试套件的 job：

   ```yaml
   - name: Test F1 Studio Kernel
     run: |
       pip install -e ".[dev]"
       pytest smoke/f1/ -v --cov=smoke/f1/ --cov-report=xml
   ```
3. 添加 ruff 静态检查步骤
4. 验证 CI 运行通过

***

### 10. 补充日志脱敏过滤器（2-3 小时）

**目标**：实现敏感环境变量和密钥的日志过滤，满足安全红线第三项。

**实现方案**：

1. 在 `BaseTaskHandler` 或 Dispatcher 层添加 `_sanitize_log()` 方法
2. 维护敏感 key 列表（如 `API_KEY`, `SECRET`, `TOKEN`, `PASSWORD` 等）
3. 日志输出前扫描并替换为 `***`
4. 添加对应的单元测试

***

## 📋 P4 低优先级（未来迭代，约 1-2 天）

### 11. 任务状态持久化（1-2 天）

- 方案：SQLite 或 Redis 存储任务状态

- 目标：支持进程重启恢复、多 worker 部署

***

## ✅ 执行追踪表

| #  | 任务                     | 优先级 | 预计耗时      | 前置依赖 | 状态 | 执行人    | 完成日期   |
| -- | ---------------------- | --- | --------- | ---- | -- | ------ | ------ |
| 1  | 更新 README P1 Checklist | P0  | 10 min    | 无    | ⬜  | <br /> | <br /> |
| 2  | 清理 Git 重复提交            | P0  | 30 min    | 无    | ⬜  | <br /> | <br /> |
| 3  | 修复 Phase 0 报告路径引用      | P0  | 5 min     | 无    | ⬜  | <br /> | <br /> |
| 4  | regex 路径白名单增强          | P1  | 3-4 h     | 无    | ⬜  | <br /> | <br /> |
| 5  | 当前 HEAD 复测更新基线         | P1  | 30-60 min | #4   | ⬜  | <br /> | <br /> |
| 6  | 补齐操作手册                 | P2  | 2-3 h     | #5   | ⬜  | <br /> | <br /> |
| 7  | 补齐安全测试报告               | P2  | 1-2 h     | #5   | ⬜  | <br /> | <br /> |
| 8  | 架构转正到生产包               | P3  | 1-2 天     | #5   | ⬜  | <br /> | <br /> |
| 9  | CI 流水线集成               | P3  | 2-4 h     | #8   | ⬜  | <br /> | <br /> |
| 10 | 日志脱敏过滤器                | P3  | 2-3 h     | 无    | ⬜  | <br /> | <br /> |
| 11 | 任务状态持久化                | P4  | 1-2 天     | #8   | ⬜  | <br /> | <br /> |

***

## 🔄 执行顺序速记

```
P0（并行做）: README更新 ─┐
            Git清理    ──┼── 约1小时
            路径修正   ──┘

P1（串行做）: regex增强 → 复测通过
             (3-4h)    (30min)
                         ↓
P2（并行写）: 操作手册 ──┐
            安全报告 ──┴── 功能稳定后一次性写好
                         
P3/P4（结项后）: 架构转正 → CI集成 → 日志脱敏 → 持久化
```

*本清单基于《F1 课题完成度评估报告》生成，按"功能先行 → 验证 → 文档"的开发时序排列。P0 三项建议立刻动手（约 1 小时），P1 regex 增强是 P1 功能闭环的最后一块拼图，做完复测通过后再统一补齐 P2 文档交付物。*
