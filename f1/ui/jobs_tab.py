"""Legacy UI exports; production Studio uses :mod:`studio.ui.jobs_tab`."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from studio.ui import jobs_tab as _implementation

__all__ = _implementation.__all__ + ["JobsManager", "get_job_image_artifacts"]
JobsManager: Any
get_job_image_artifacts: Any


def __getattr__(name: str) -> Any:
    """Preserve historical imports without importing a backend during UI startup."""
    if name in {"JobsManager", "get_job_image_artifacts"}:
        return getattr(import_module("f1.jobs_manager"), name)
    return getattr(_implementation, name)
