# F1 Task Handler Framework - Usage Guide

## Overview

This is the decoupled task handler framework for the YOLO-Master F1 platform, providing decorator-based handler registration and factory method pattern.

## Architecture

```
smoke/f1/handlers/
├── __init__.py          # Module entry point, exports BaseTaskHandler and TaskHandlerRegistry
├── base.py              # BaseTaskHandler abstract base class
└── registry.py          # TaskHandlerRegistry and factory
```

## Core Components

### 1. BaseTaskHandler (Abstract Base Class)

Defines the core contract for task handlers:

```python
from smoke.f1.handlers import BaseTaskHandler


class YourTaskHandler(BaseTaskHandler):
    def validate_params(self, params, security_constraints):
        """Validate task parameters and security constraints"""
        # Implement validation logic
        return True, None  # (is_valid, error_message)

    def execute(self, job_id, params, output_dir):
        """Execute task and return result"""
        # Implement task execution logic
        return {"success": True, "artifacts": [...], "metadata": {...}, "error": None}
```

**Built-in Security Utilities**:

- `_is_path_safe(target_path, allowed_roots)`: Path safety validation with relative path resolution and directory traversal defense

### 2. TaskHandlerRegistry (Registry)

Provides decorator registration and factory method retrieval:

```python
from smoke.f1.handlers import TaskHandlerRegistry, BaseTaskHandler


@TaskHandlerRegistry.register("predict")
class PredictHandler(BaseTaskHandler):
    def validate_params(self, params, security_constraints):
        # Validate shell execution permission
        if security_constraints.get("allow_shell"):
            return False, "Shell execution not permitted"

        # Validate path whitelist
        allowed_paths = security_constraints.get("allowed_paths", [])
        model_path = params.get("model_path", "")
        if model_path and not self._is_path_safe(model_path, allowed_paths):
            return False, f"Model path '{model_path}' not in whitelist"

        return True, None

    def execute(self, job_id, params, output_dir):
        from ultralytics import YOLO

        # Execute actual YOLO inference
        model = YOLO(params["model_path"])
        results = model.predict(
            source=params["data_source"], device=params.get("device", "0"), project=output_dir, name=job_id
        )

        # Capture output artifacts
        artifacts = [str(p) for p in Path(results[0].save_dir).glob("*.jpg")]

        return {
            "success": True,
            "artifacts": artifacts,
            "metadata": {"detected_count": len(results[0].boxes)},
            "error": None,
        }
```

## Dispatcher Integration Pattern

Dispatcher usage example:

```python
from smoke.f1.handlers import TaskHandlerRegistry


def dispatch_job(job_request):
    """Dispatcher core logic"""
    # 1. Retrieve handler class by task_type
    handler_class = TaskHandlerRegistry.get(job_request.task_type)
    handler = handler_class()

    # 2. Validate parameters and security constraints
    is_valid, err_msg = handler.validate_params(job_request.params, job_request.security_constraints)

    if not is_valid:
        job_request.status = "FAILED"
        job_request.error = {"code": "SEC_ERR_001", "message": err_msg}
        return job_request

    # 3. Execute task
    job_request.status = "RUNNING"
    result = handler.execute(job_request.job_id, job_request.params, job_request.output.output_dir)

    # 4. Update task status
    if result["success"]:
        job_request.status = "COMPLETED"
        job_request.output.artifacts = result["artifacts"]
    else:
        job_request.status = "FAILED"
        job_request.error = {"code": "EXEC_ERR_500", "message": result["error"]}

    return job_request
```

## Security Enforcement

The framework enforces the following security policies:

### 1. Path Whitelisting

```python
def validate_params(self, params, security_constraints):
    allowed_paths = security_constraints.get("allowed_paths", [])
    
    # Validate all input paths
    for key in ["model_path", "data_source"]:
        path = params.get(key, "")
        if path and not self._is_path_safe(path, allowed_paths):
            return False, f"{key} path '{path}' not in whitelist"
    
    return True, None
```

**Security Properties**:
- Automatically resolves symlinks and relative paths (`../../`)
- Empty whitelist defaults to deny-all (fail-closed)
- Prevents path traversal attacks

### 2. Shell Execution Prohibition

```python
if security_constraints.get("allow_shell"):
    return False, "Shell execution not permitted"
```

### 3. Resource Isolation

- Independent output directory per `job_id`
- GPU memory management (using context managers)
- Timeout control (specified by `runtime_tracking.timeout_seconds`)

## Registry API Reference

### Registration

```python
@TaskHandlerRegistry.register(task_type: str)
```

**Behavior**:
- Registers at class definition time (module import phase)
- Prevents duplicate registration (raises `ValueError`)
- Type checking: must inherit from `BaseTaskHandler`

### Factory Method

```python
TaskHandlerRegistry.get(task_type: str) -> type[BaseTaskHandler]
```

**Returns**: Handler class (not instance)  
**Raises**: `ValueError` for unregistered `task_type` with list of available types

### Utility Methods

```python
TaskHandlerRegistry.list_registered() -> list[str]
```

Returns all registered task types (sorted)

```python
TaskHandlerRegistry.clear()
```

Clears the registry (testing isolation only)

## Error Handling

### Registration-Time Errors

```python
# Duplicate registration
@TaskHandlerRegistry.register("predict")
class DuplicateHandler(BaseTaskHandler):
    pass


# ValueError: Task type 'predict' is already registered


# Non-BaseTaskHandler subclass
@TaskHandlerRegistry.register("invalid")
class InvalidHandler:
    pass


# TypeError: Handler class InvalidHandler must inherit from BaseTaskHandler
```

### Runtime Errors

```python
# Unregistered task type
handler = TaskHandlerRegistry.get("unknown_task")
# ValueError: Task type 'unknown_task' is not registered.
#            Available types: ['diagnose', 'export', 'predict', 'train']
```

## Testing

Run unit tests:

```bash
cd smoke/f1
python -m pytest test_handlers_framework.py -v
```

**Test Coverage**:
- Abstract base class instantiation prevention
- Decorator registration and retrieval
- Path safety validation
- Duplicate registration defense
- End-to-end dispatcher integration

## Extension Example: Train Handler

```python
@TaskHandlerRegistry.register("train")
class TrainHandler(BaseTaskHandler):
    def validate_params(self, params, security_constraints):
        # Validate training-specific parameters
        if "data_yaml" not in params:
            return False, "data_yaml is required for training"

        # Path whitelist check
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

        # Capture training artifacts
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

1. **Separation of Concerns**: Dispatcher does not depend on concrete handler implementations
2. **Open-Closed Principle**: Adding new task types requires no dispatcher modifications
3. **Security-First**: Validation phase enforces security policies
4. **Fail-Closed**: Empty whitelist defaults to deny, not allow
5. **Type-Safe**: Complete type hints and runtime checking

## Future Enhancements

- [ ] Async execution support (`async def execute`)
- [ ] Handler lifecycle hooks (`on_start`, `on_complete`, `on_error`)
- [ ] Resource pool management (GPU allocation, concurrency limits)
- [ ] Handler versioning (support for multiple versions)
- [ ] Dynamic reloading (hot-swap handler implementations)

## References

- **F1 Smoke Test**: `smoke/f1/test_f1_smoke.py`
- **JobRequest Contract**: `smoke/f1/README.md` (Section 2.2)
- **Security Model**: `smoke/f1/README.md` (Section 4)
- **Ultralytics YOLO**: https://docs.ultralytics.com/

---

**Version**: 1.0.0  
**Last Updated**: 2026-09-01  
**Maintained By**: [@9-71](https://github.com/9-71)
