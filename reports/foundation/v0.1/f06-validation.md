# F06 验证记录

日期：2026-08-12（Asia/Shanghai）

本阶段实现训练专用 `FoundationDistillationModel` 与最小 Trainer 接入。teacher 通过非注册引用保留在 wrapper 外部，不进入 optimizer、DDP、EMA 或 `state_dict`；EMA/复制路径恢复为 student-only wrapper，部署入口 `deployment_model()` 返回纯 student。

## 验证结果

- `pytest -q tests/test_foundation_distill_model.py`：7 passed
- Foundation 定向套件（配置、teacher protocol/DINOv3、tap、projector、loss、wrapper）：91 passed
- `pytest tests/test_engine.py -k 'foundation or distill' --tb=short`：9 passed
- F00 focused regression gate：230 passed，25 warnings
- Foundation/F06 变更目录 Ruff、Ruff format、codespell、compileall、`git diff --check`：通过

## 当前边界

- F06 仅实现 DINOv3/Transformers 构造路径；`foundation_backend=local` 需通过 `teacher_manager` 注入，避免引入未定义的本地权重协议。
- 真实 DINOv3 权重未下载，CI 使用离线 dummy teacher；真实 Transformers 集成留给手工/夜间验证。
- checkpoint/export 的完整 metadata 与 strip 流程属于 F07，本阶段只提供明确的 `deployment_model()` 与 student-only deepcopy 行为。
