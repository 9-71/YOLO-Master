"""Task handler registry with decorator-based registration and factory method.

This module provides a centralized registry for F1 task handlers, enabling dynamic
handler lookup by task_type string. Handlers register themselves via the @register
decorator, and the dispatcher retrieves them via TaskHandlerRegistry.get().

Registry Pattern Benefits:
    - Decouples dispatcher from concrete handler implementations
    - Enables runtime handler discovery and dynamic task routing
    - Centralizes handler validation (prevents duplicate registrations)
    - Simplifies addition of new task types (no dispatcher modification required)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from smoke.f1.handlers.base import BaseTaskHandler

# Type variable for handler classes (must inherit BaseTaskHandler)
THandler = TypeVar("THandler", bound="BaseTaskHandler")


class TaskHandlerRegistry:
    """Centralized registry for F1 task execution handlers.

    The registry maintains a mapping from task_type strings (e.g., "predict", "train")
    to concrete handler classes. Handlers self-register via the @register decorator,
    and the dispatcher retrieves them via the get() factory method.

    Thread Safety:
        This implementation is NOT thread-safe. Handler registration MUST occur
        at module import time (before any concurrent dispatch operations).

    Example:
        >>> from smoke.f1.handlers.registry import TaskHandlerRegistry
        >>> from smoke.f1.handlers.base import BaseTaskHandler
        >>>
        >>> @TaskHandlerRegistry.register("predict")
        ... class PredictHandler(BaseTaskHandler):
        ...     def validate_params(self, params, security_constraints):
        ...         return True, None
        ...
        ...     def execute(self, job_id, params, output_dir):
        ...         return {"success": True, "artifacts": []}
        >>>
        >>> handler_cls = TaskHandlerRegistry.get("predict")
        >>> handler = handler_cls()
        >>> result = handler.execute("test-001", {}, "runs/predict")
    """

    # Class-level handler registry: task_type -> handler_class
    _handlers: ClassVar[dict[str, type[BaseTaskHandler]]] = {}

    @classmethod
    def register(cls, task_type: str) -> Callable[[type[THandler]], type[THandler]]:
        """Decorator to register a handler class for a specific task_type.

        This decorator associates a concrete handler implementation with a task_type
        string (e.g., "predict", "train", "export", "diagnose"). The decorated class
        MUST inherit from BaseTaskHandler and implement all abstract methods.

        Args:
            task_type: Task type identifier (must match JobRequest.task_type enum values)
                Valid values: "predict", "train", "export", "diagnose"

        Returns:
            Callable: Decorator function that registers the handler and returns the class unchanged

        Raises:
            ValueError: If task_type is already registered (prevents duplicate registrations)
            TypeError: If decorated class does not inherit from BaseTaskHandler

        Usage:
            @TaskHandlerRegistry.register("predict")
            class PredictHandler(BaseTaskHandler):
                def validate_params(self, params, security_constraints):
                    # Implementation
                    pass
                def execute(self, job_id, params, output_dir):
                    # Implementation
                    pass

        Design Notes:
            - Registration happens at class definition time (module import)
            - Duplicate registrations raise ValueError to prevent silent overwrites
            - Type checking is deferred to runtime (Python's ABC mechanism)

        Example:
            >>> @TaskHandlerRegistry.register("custom_task")
            ... class CustomHandler(BaseTaskHandler):
            ...     def validate_params(self, params, security_constraints):
            ...         return True, None
            ...
            ...     def execute(self, job_id, params, output_dir):
            ...         return {"success": True, "artifacts": []}
            >>> TaskHandlerRegistry.get("custom_task")
            <class 'CustomHandler'>
        """

        def decorator(handler_class: type[THandler]) -> type[THandler]:
            # Prevent duplicate registrations
            if task_type in cls._handlers:
                raise ValueError(
                    f"Task type '{task_type}' is already registered to {cls._handlers[task_type].__name__}. "
                    f"Cannot register {handler_class.__name__}."
                )

            # Type safety check (runtime verification of BaseTaskHandler inheritance)
            # Note: Abstract method implementation is checked by Python's ABC at instantiation
            from smoke.f1.handlers.base import BaseTaskHandler

            if not issubclass(handler_class, BaseTaskHandler):
                raise TypeError(
                    f"Handler class {handler_class.__name__} must inherit from BaseTaskHandler. "
                    f"Found bases: {handler_class.__bases__}"
                )

            # Register the handler
            cls._handlers[task_type] = handler_class
            return handler_class

        return decorator

    @classmethod
    def get(cls, task_type: str) -> type[BaseTaskHandler]:
        """Factory method to retrieve a registered handler class by task_type.

        This method looks up the handler class registered for the given task_type.
        The dispatcher uses this to instantiate the appropriate handler at runtime.

        Args:
            task_type: Task type identifier to look up

        Returns:
            type[BaseTaskHandler]: Registered handler class (NOT an instance)

        Raises:
            ValueError: If task_type is not registered

        Usage Pattern:
            handler_class = TaskHandlerRegistry.get("predict")
            handler_instance = handler_class()  # Instantiate the handler
            is_valid, err = handler_instance.validate_params(params, constraints)
            if is_valid:
                result = handler_instance.execute(job_id, params, output_dir)

        Error Message Design:
            The ValueError includes:
            - The unregistered task_type that was requested
            - List of all currently registered task types
            This helps developers diagnose typos and missing registrations.

        Example:
            >>> TaskHandlerRegistry.get("predict")
            <class 'PredictHandler'>
            >>> TaskHandlerRegistry.get("unknown_task")
            Traceback (most recent call last):
                ...
            ValueError: Task type 'unknown_task' is not registered. Available types: ['predict', 'train']
        """
        if task_type not in cls._handlers:
            available = sorted(cls._handlers.keys())
            raise ValueError(
                f"Task type '{task_type}' is not registered. "
                f"Available types: {available}. "
                f"Ensure the handler module is imported and decorated with @TaskHandlerRegistry.register()."
            )
        return cls._handlers[task_type]

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return a sorted list of all registered task types.

        This utility method is primarily for debugging, testing, and generating
        dynamic UI elements (e.g., task type dropdowns in Gradio).

        Returns:
            list[str]: Sorted list of registered task_type strings

        Example:
            >>> TaskHandlerRegistry.list_registered()
            ['diagnose', 'export', 'predict', 'train']
        """
        return sorted(cls._handlers.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered handlers.

        **WARNING**: This method is ONLY for testing purposes. Clearing the registry
        in production code will break task dispatch and cause ValueError on all get() calls.

        Use Case:
            Test isolation - each test can register mock handlers without polluting
            other tests' registry state.

        Example:
            >>> # In test setup
            >>> TaskHandlerRegistry.clear()
            >>> @TaskHandlerRegistry.register("test_task")
            ... class MockHandler(BaseTaskHandler):
            ...     pass
        """
        cls._handlers.clear()
