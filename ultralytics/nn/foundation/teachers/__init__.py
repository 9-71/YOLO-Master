"""Foundation Teacher backend implementations."""

from .dinov3 import DEFAULT_DINOV3_MODEL, DINOv3Teacher
from .multi import MultiFoundationTeacher
from .siglip2 import DEFAULT_SIGLIP2_MODEL, SigLIP2Teacher

__all__ = [
    "DEFAULT_DINOV3_MODEL",
    "DINOv3Teacher",
    "DEFAULT_SIGLIP2_MODEL",
    "SigLIP2Teacher",
    "MultiFoundationTeacher",
]
