# F1 Task Handler Framework - Usage Guide

## Overview

这是 YOLO-Master F1 平台的任务执行器解耦骨架，提供了基于装饰器的处理器注册机制和工厂方法模式。

## Architecture

```
smoke/f1/handlers/
├── __init__.py          # 模块入口，导出 BaseTaskHandler 和 TaskHandlerRegistry
├── base.py              # BaseTaskHandler 抽象基类
└── registry.py          # TaskHandlerRegistry 注册器和工厂
```

## Core Components

### 1. BaseTaskHandler (抽象基类)

定义了任务处理器的核心契约：

```python
from smoke.f1.handlers import BaseTaskHandler


class YourTaskHandler(BaseTaskHandler):
    def validate_params(self, params, security_constraints):
        """验证任务参数和安全约束"""
        # 实现验证逻辑
        return True, None  # (is_valid, error_message)

    def execute(self, job_id, params, output_dir):
        """执行任务并返回结果"""
        # 实现任务执行逻辑
        return {"success": True, "artifacts": [...], "metadata": {...}, "error": None}
```

**内置安全工具方法**:

- `_is_path_safe(target_path, allowed_roots)`: 路径安全验证，支持相对路径解析和目录遍历防御

### 2. TaskHandlerRegistry (注册器)

提供装饰器注册和工厂方法检索：

```python
from smoke.f1.handlers import TaskHandlerRegistry, BaseTaskHandler


@TaskHandlerRegistry.register("predict")
class PredictHandler(BaseTaskHandler):
    def validate_params(self, params, security_constraints):
        # 校验 shell 执行权限
        if security_constraints.get("allow_shell"):
            return False, "Shell execution not permitted"

        # 校验路径白名单
        allowed_paths = security_constraints.get("allowed_paths", [])
        model_path = params.get("model_path", "")
        if model_path and not self._is_path_safe(model_path, allowed_paths):
            return False, f"Model path '{model_path}' not in whitelist"

        return True, None

    def execute(self, job_id, params, output_dir):
        from ultralytics import YOLO

        # 执行真实的 YOLO 推理
        model = YOLO(params["model_path"])
        results = model.predict(
            source=params["data_source"], device=params.get("device", "0"), project=output_dir, name=job_id
        )

        # 捕获输出工件
        artifacts = [str(p) for p in Path(results[0].save_dir).glob("*.jpg")]

        return {
            "success": True,
            "artifacts": artifacts,
            "metadata": {"detected_count": len(results[0].boxes)},
            "error": None,
        }
```

## Dispatcher Integration Pattern

调度器使用示例：

```python
from smoke.f1.handlers import TaskHandlerRegistry


def dispatch_job(job_request):
    """调度器核心逻辑"""
    # 1. 根据 task_type 获取处理器类
    handler_class = TaskHandlerRegistry.get(job_request.task_type)
    handler = handler_class()

    # 2. 验证参数和安全约束
    is_valid, err_msg = handler.validate_params(job_request.params, job_request.security_constraints)

    if not is_valid:
        job_request.status = "FAILED"
        job_request.error = {"code": "SEC_ERR_001", "message": err_msg}
        return job_request

    # 3. 执行任务
    job_request.status = "RUNNING"
    result = handler.execute(job_request.job_id, job_request.params, job_request.output.output_dir)

    # 4. 更新任务状态
    if result["success"]:
        job_request.status = "COMPLETED"
        job_request.output.artifacts = result["artifacts"]
    else:
        job_request.status = "FAILED"
        job_request.error = {"code": "EXEC_ERR_500", "message": result["error"]}

    return job_request
```

## Security Enforcement

框架强制执行以下安全策略：

### 1. 路径白名单验证

```python
def validate_params(self, params, security_constraints):
    allowed_paths = security_constraints.get("allowed_paths", [])
    
    # 验证所有输入路径
    for key in ["model_path", "data_source"]:
        path = params.get(key, "")
        if path and not self._is_path_safe(path, allowed_paths):
            return False, f"{key} path '{path}' not in whitelist"
    
    return True, None
```

**安全属性**:
- 自动解析符号链接和相对路径（`../../`）
- 空白名单默认拒绝所有路径（fail-closed）
- 防止路径遍历攻击

### 2. Shell 执行禁止

```python
if security_constraints.get("allow_shell"):
    return False, "Shell execution not permitted"
```

### 3. 资源隔离

- 每个 `job_id` 独立的输出目录
- GPU 内存管理（使用上下文管理器）
- 超时控制（由 `runtime_tracking.timeout_seconds` 指定）

## Registry API Reference

### Registration

```python
@TaskHandlerRegistry.register(task_type: str)
```

**行为**:
- 在类定义时注册（模块导入阶段）
- 阻止重复注册（抛出 `ValueError`）
- 类型检查：必须继承 `BaseTaskHandler`

### Factory Method

```python
TaskHandlerRegistry.get(task_type: str) -> type[BaseTaskHandler]
```

**返回**: 处理器类（非实例）  
**异常**: 未注册的 `task_type` 抛出 `ValueError` 并列出可用类型

### Utility Methods

```python
TaskHandlerRegistry.list_registered() -> list[str]
```

返回已注册的所有任务类型（排序）

```python
TaskHandlerRegistry.clear()
```

清空注册表（仅用于测试隔离）

## Error Handling

### 注册时错误

```python
# 重复注册
@TaskHandlerRegistry.register("predict")
class DuplicateHandler(BaseTaskHandler):
    pass


# ValueError: Task type 'predict' is already registered


# 非 BaseTaskHandler 子类
@TaskHandlerRegistry.register("invalid")
class InvalidHandler:
    pass


# TypeError: Handler class InvalidHandler must inherit from BaseTaskHandler
```

### 运行时错误

```python
# 未注册的任务类型
handler = TaskHandlerRegistry.get("unknown_task")
# ValueError: Task type 'unknown_task' is not registered.
#            Available types: ['diagnose', 'export', 'predict', 'train']
```

## Testing

运行单元测试：

```bash
cd smoke/f1
python -m pytest test_handlers_framework.py -v
```

**测试覆盖**:
- 抽象基类实例化阻止
- 装饰器注册和检索
- 路径安全验证
- 重复注册防御
- 端到端调度器集成

## Extension Example: Train Handler

```python
@TaskHandlerRegistry.register("train")
class TrainHandler(BaseTaskHandler):
    def validate_params(self, params, security_constraints):
        # 验证训练特定参数
        if "data_yaml" not in params:
            return False, "data_yaml is required for training"

        # 路径白名单检查
        allowed = security_constraints.get("allowed_paths", [])
        if not self._is_path_safe(params["data_yaml"], allowed):
            return False, "data_yaml path not in whitelist"

        return True, None

    def execute(self, job_id, params, output_dir):
        from ultralytics import YOLO

        model = YOLO(params.get("model_path", "yolov8n.yaml"))
        results = model.train(
            data=params["data_yaml"],
            epochs=params.get("epochs", 100),
            project=output_dir,
            name=job_id,
            device=params.get("device", "0"),
        )

        # 捕获训练工件
        artifacts = [f"{output_dir}/{job_id}/weights/best.pt", f"{output_dir}/{job_id}/results.csv"]

        return {
            "success": True,
            "artifacts": artifacts,
            "metadata": {
                "final_map50": results.results_dict["metrics/mAP50(B)"],
                "epochs_completed": params.get("epochs", 100),
            },
            "error": None,
        }
```

## Design Principles

1. **分离关注点**: 调度器不依赖具体处理器实现
2. **开闭原则**: 新增任务类型无需修改调度器代码
3. **安全优先**: 验证阶段强制执行安全策略
4. **失败封闭**: 空白名单默认拒绝，而非允许
5. **类型安全**: 完整的 Type Hints 和运行时检查

## Future Enhancements

- [ ] 异步执行支持（`async def execute`）
- [ ] 处理器生命周期钩子（`on_start`, `on_complete`, `on_error`）
- [ ] 资源池管理（GPU 分配、并发限制）
- [ ] 处理器版本控制（支持多版本共存）
- [ ] 动态重载（热更新处理器实现）

## References

- **F1 Smoke Test**: `smoke/f1/test_f1_smoke.py`
- **JobRequest Contract**: `smoke/f1/README.md` (Section 2.2)
- **Security Model**: `smoke/f1/README.md` (Section 4)
- **Ultralytics YOLO**: https://docs.ultralytics.com/

---

**Version**: 1.0.0  
**Last Updated**: 2026-09-01  
**Maintained By**: [@9-71](https://github.com/9-71)
