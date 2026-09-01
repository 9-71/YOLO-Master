"""F1 Task Handler Framework - Base abstractions and registry.

This package provides the core architecture for YOLO-Master F1 platform task execution:
- BaseTaskHandler: Abstract base class defining validation and execution contract
- TaskHandlerRegistry: Decorator-based registration and factory for handler lookup

Usage:
    from smoke.f1.handlers import BaseTaskHandler, TaskHandlerRegistry

    @TaskHandlerRegistry.register("predict")
    class PredictHandler(BaseTaskHandler):
        def validate_params(self, params, security_constraints):
            # Security validation logic
            return True, None

        def execute(self, job_id, params, output_dir):
            # Task execution logic
            return {"success": True, "artifacts": [...]}

    # Dispatcher usage
    handler_class = TaskHandlerRegistry.get("predict")
    handler = handler_class()
    result = handler.execute("job-001", params, "runs/predict")
"""

# Auto-import concrete handlers to trigger @register decorators
from smoke.f1.handlers import diagnose, export, predict, train  # noqa: F401
from smoke.f1.handlers.base import BaseTaskHandler
from smoke.f1.handlers.registry import TaskHandlerRegistry

__all__ = ["BaseTaskHandler", "TaskHandlerRegistry"]
