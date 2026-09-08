"""Core production package for the YOLO-Master platform.

This package hosts shared production modules that any layer of the platform
(dispatcher runtime, handler framework, WebUI, agent skills and test
scaffolding) may import without creating cross-layer dependency cycles.

The canonical domain schema module is :mod:`core.schema`; it used to live in
the legacy entry contract module ``f1/test_f1_smoke.py`` and was moved
here so production code never imports data structures from a test file.
"""
