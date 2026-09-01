"""Demo script showcasing PredictHandler and DiagnoseHandler usage.

This script demonstrates the complete workflow:
1. Handler registration via TaskHandlerRegistry
2. Parameter validation with security constraints
3. Task execution with artifact generation
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from smoke.f1.handlers import TaskHandlerRegistry


def demo_diagnose_handler():
    """Demonstrate DiagnoseHandler usage."""
    print("=" * 80)
    print("DiagnoseHandler Demo")
    print("=" * 80)

    # Retrieve handler from registry
    handler_class = TaskHandlerRegistry.get("diagnose")
    handler = handler_class()
    print(f"✓ Handler retrieved: {handler_class.__name__}")

    # Validate parameters
    params = {}
    constraints = {"allow_shell": False}
    is_valid, err = handler.validate_params(params, constraints)
    print(f"✓ Validation: {'PASSED' if is_valid else 'FAILED'}")
    if err:
        print(f"  Error: {err}")

    # Execute diagnostics collection
    output_dir = "runs/diagnose"
    result = handler.execute("demo_diag_001", params, output_dir)
    print(f"✓ Execution: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"  Artifacts generated: {len(result['artifacts'])}")

    for artifact in result["artifacts"]:
        print(f"    - {Path(artifact).name}")

    print("\n  Metadata:")
    for key, value in result["metadata"].items():
        print(f"    {key}: {value}")

    print()


def demo_predict_handler():
    """Demonstrate PredictHandler usage (validation only)."""
    print("=" * 80)
    print("PredictHandler Demo (Validation)")
    print("=" * 80)

    # Retrieve handler from registry
    handler_class = TaskHandlerRegistry.get("predict")
    handler = handler_class()
    print(f"✓ Handler retrieved: {handler_class.__name__}")

    # Test Case 1: Valid parameters
    print("\nTest Case 1: Valid parameters")
    params = {
        "model_path": "yolov8n.pt",
        "data_source": "ultralytics/assets/bus.jpg",
        "device": "cpu",
        "conf": 0.25,
    }
    constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": [".", "ultralytics"]}
    is_valid, err = handler.validate_params(params, constraints)
    print(f"  Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    if err:
        print(f"  Error: {err}")

    # Test Case 2: Path outside whitelist (security violation)
    print("\nTest Case 2: Path outside whitelist (security violation)")
    params = {
        "model_path": "yolov8n.pt",
        "data_source": "../../etc/passwd",
        "device": "cpu",
    }
    constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["ultralytics/assets"]}
    is_valid, err = handler.validate_params(params, constraints)
    print(f"  Validation: {'✓ PASSED' if is_valid else '✗ FAILED (expected)'}")
    if err:
        print(f"  Error: {err}")

    # Test Case 3: Invalid confidence threshold
    print("\nTest Case 3: Invalid confidence threshold")
    params = {
        "model_path": "yolov8n.pt",
        "data_source": "bus.jpg",
        "conf": 1.5,  # Invalid: must be in (0.0, 1.0]
    }
    constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}
    is_valid, err = handler.validate_params(params, constraints)
    print(f"  Validation: {'✓ PASSED' if is_valid else '✗ FAILED (expected)'}")
    if err:
        print(f"  Error: {err}")

    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("F1 Task Handler Framework - Demo")
    print("=" * 80)
    print(f"Registered handlers: {TaskHandlerRegistry.list_registered()}")
    print()

    demo_diagnose_handler()
    demo_predict_handler()

    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
