"""Compatibility exports for the canonical base MoE implementations.

The implementations live in :mod:`ultralytics.nn.modules.moe.modules`.
Keeping these aliases preserves historical imports and checkpoint pickle paths
without maintaining a second, drifting copy of each class.
"""

from .modules import _MOE_FINITE_DIAGNOSTICS, _MOE_FINITE_DIAGNOSTIC_MAX_EVENTS  # noqa: F401  # re-export
from .modules import (
    A2C2fMoE,
    ABlockMoE,
    AdaptiveCapacityMoE,
    ES_MOE,
    OptimizedMOE,
    OptimizedMOEImproved,
    UltraOptimizedMoE,
)

__all__ = (
    "UltraOptimizedMoE",
    "AdaptiveCapacityMoE",
    "ES_MOE",
    "OptimizedMOE",
    "OptimizedMOEImproved",
    "ABlockMoE",
    "A2C2fMoE",
    "_MOE_FINITE_DIAGNOSTICS",
    "_MOE_FINITE_DIAGNOSTIC_MAX_EVENTS",
)
