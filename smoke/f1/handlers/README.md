# F1 Task Handler Framework

Decorator-based task handler decoupling framework providing type-safe, security-first handler registration mechanism for the YOLO-Master F1 platform.

## Quick Start

### Define a Handler

```python
from smoke.f1.handlers import BaseTaskHandler, TaskHandlerRegistry


@TaskHandlerRegistry.register("predict")
class PredictHandler(BaseTaskHandler):
    def validate_params(self, params, security_constraints):
        """Validate parameters and security constraints"""
        if security_constraints.get("allow_shell"):
            return False, "Shell execution not permitted"

        allowed_paths = security_constraints.get("allowed_paths", [])
        model_path = params.get("model_path", "")
        if model_path and not self._is_path_safe(model_path, allowed_paths):
            return False, f"Model path not in whitelist"

        return True, None

    def execute(self, job_id, params, output_dir):
        """Execute the task"""
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

### Dispatcher Integration

```python
# Retrieve handler and execute
handler_class = TaskHandlerRegistry.get(job_request.task_type)
handler = handler_class()

# Validate
is_valid, err = handler.validate_params(params, security_constraints)
if not is_valid:
    # Handle validation failure
    pass

# Execute
result = handler.execute(job_id, params, output_dir)
```

## Core Features

✅ **Type-Safe**: Complete type hints and abstract base class enforcement  
✅ **Decorator Registration**: `@TaskHandlerRegistry.register(task_type)` auto-registration  
✅ **Factory Method**: `TaskHandlerRegistry.get(task_type)` dynamic retrieval  
✅ **Security-First**: Built-in path whitelisting and shell execution protection  
✅ **Clear Errors**: Duplicate registration and unregistered types throw explicit exceptions  

## File Structure

```
handlers/
├── __init__.py          # Module entry point
├── base.py              # BaseTaskHandler abstract base class
├── registry.py          # TaskHandlerRegistry
├── USAGE.md             # Detailed usage guide
└── README.md            # This file
```

## Testing

```bash
cd smoke/f1
python -m pytest test_handlers_framework.py -v
```

**Test Coverage**: 10 test cases covering registration, retrieval, validation, security enforcement, and end-to-end integration.

## Design Principles

- **Open-Closed Principle**: Adding new task types requires no dispatcher modifications
- **Dependency Inversion**: Dispatcher depends on abstract base class, not concrete implementations
- **Security-First**: Empty whitelist defaults to deny (fail-closed)
- **Type-Safe**: Runtime enforcement of BaseTaskHandler inheritance

## API Reference

### BaseTaskHandler

Abstract base class defining handler contract:

- `validate_params(params, security_constraints)` → `(bool, str | None)`
- `execute(job_id, params, output_dir)` → `dict[str, Any]`
- `_is_path_safe(target_path, allowed_roots)` → `bool` (helper method)

### TaskHandlerRegistry

- `@register(task_type)` - Decorator registration
- `get(task_type)` - Factory method retrieval
- `list_registered()` - List registered types
- `clear()` - Clear registry (testing only)

## Detailed Documentation

Complete usage guide, security model, and extension examples available in [USAGE.md](USAGE.md)

---

**Version**: 1.0.0 | **Created**: 2026-09-01
