# F1 Task Handler Framework

基于装饰器的任务执行器解耦骨架，为 YOLO-Master F1 平台提供类型安全、安全优先的处理器注册机制。

## 快速开始

### 定义处理器

```python
from smoke.f1.handlers import BaseTaskHandler, TaskHandlerRegistry


@TaskHandlerRegistry.register("predict")
class PredictHandler(BaseTaskHandler):
    def validate_params(self, params, security_constraints):
        """验证参数和安全约束"""
        if security_constraints.get("allow_shell"):
            return False, "Shell execution not permitted"

        allowed_paths = security_constraints.get("allowed_paths", [])
        model_path = params.get("model_path", "")
        if model_path and not self._is_path_safe(model_path, allowed_paths):
            return False, f"Model path not in whitelist"

        return True, None

    def execute(self, job_id, params, output_dir):
        """执行任务"""
        from ultralytics import YOLO

        model = YOLO(params["model_path"])
        results = model.predict(
            source=params["data_source"], device=params.get("device", "0"), project=output_dir, name=job_id
        )

        artifacts = [str(p) for p in Path(results[0].save_dir).glob("*.jpg")]

        return {
            "success": True,
            "artifacts": artifacts,
            "metadata": {"detected_count": len(results[0].boxes)},
            "error": None,
        }
```

### 调度器集成

```python
# 获取处理器并执行
handler_class = TaskHandlerRegistry.get(job_request.task_type)
handler = handler_class()

# 验证
is_valid, err = handler.validate_params(params, security_constraints)
if not is_valid:
    # 处理验证失败
    pass

# 执行
result = handler.execute(job_id, params, output_dir)
```

## 核心特性

✅ **类型安全**: 完整的 Type Hints 和抽象基类强制  
✅ **装饰器注册**: `@TaskHandlerRegistry.register(task_type)` 自动注册  
✅ **工厂方法**: `TaskHandlerRegistry.get(task_type)` 动态检索  
✅ **安全优先**: 内置路径白名单验证和 shell 执行防护  
✅ **错误清晰**: 重复注册和未注册类型抛出明确异常  

## 文件结构

```
handlers/
├── __init__.py          # 模块入口
├── base.py              # BaseTaskHandler 抽象基类
├── registry.py          # TaskHandlerRegistry 注册器
├── USAGE.md             # 详细使用指南
└── README.md            # 本文件
```

## 测试

```bash
cd smoke/f1
python -m pytest test_handlers_framework.py -v
```

**测试覆盖**: 10 个测试用例，覆盖注册、检索、验证、安全防护和端到端集成。

## 设计原则

- **开闭原则**: 新增任务类型无需修改调度器
- **依赖倒置**: 调度器依赖抽象基类，不依赖具体实现
- **安全优先**: 空白名单默认拒绝（fail-closed）
- **类型安全**: 运行时检查 BaseTaskHandler 继承

## API 参考

### BaseTaskHandler

抽象基类，定义处理器契约：

- `validate_params(params, security_constraints)` → `(bool, str | None)`
- `execute(job_id, params, output_dir)` → `dict[str, Any]`
- `_is_path_safe(target_path, allowed_roots)` → `bool` (辅助方法)

### TaskHandlerRegistry

- `@register(task_type)` - 装饰器注册
- `get(task_type)` - 工厂方法检索
- `list_registered()` - 列出已注册类型
- `clear()` - 清空注册表（仅测试用）

## 详细文档

完整使用指南、安全模型和扩展示例见 [USAGE.md](USAGE.md)

---

**Version**: 1.0.0 | **Created**: 2026-09-01
