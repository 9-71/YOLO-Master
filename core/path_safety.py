"""Filesystem path resolution helpers for security boundaries."""

from __future__ import annotations

import stat
from pathlib import Path

__all__ = ["resolve_path_allow_missing"]


def resolve_path_allow_missing(path_value: str | Path) -> Path:
    """Resolve a path while rejecting broken symlinks and allowing missing plain paths.

    ``Path.resolve(strict=False)`` intentionally permits a missing leaf, which is
    required for output directories that will be created later.  It also permits
    a broken symlink, however, and can therefore turn an unresolvable link into an
    apparently contained path.  Strictly resolve only components that are actual
    symlinks before performing the ordinary non-strict normalization.

    Args:
        path_value: Local filesystem path to resolve.

    Returns:
        Path: Absolute, normalized path with all resolvable symlinks expanded.

    Raises:
        OSError: If an existing component cannot be inspected or a symlink is
            broken or otherwise unresolvable.
        RuntimeError: If symlink resolution encounters a loop.
    """
    path = Path(path_value)
    lexical_path = path if path.is_absolute() else Path.cwd() / path
    current = Path(lexical_path.anchor)

    for part in lexical_path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            # A missing ordinary component is valid for a path that will be
            # created later. No child beneath it can be an existing symlink.
            continue
        if stat.S_ISLNK(mode):
            current.resolve(strict=True)

    return path.resolve(strict=False)
