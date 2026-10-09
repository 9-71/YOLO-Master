"""Manifest normalization and exact, read-time containment checks; no scanning."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from core.path_safety import resolve_path_allow_missing
from studio.admission import resolve_contained

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


class ArtifactResolver:
    """Resolve candidates and manifest IDs without committing lifecycle or manifest."""

    def __init__(self, output_root):
        self.output_root = Path(output_root)

    def root(self, job):
        """Return a trusted per-job root, or None when containment cannot be proved."""
        try:
            base = resolve_contained(job.output.output_dir, [self.output_root], "output_dir")
            root = resolve_path_allow_missing(base / job.job_id)
            root.relative_to(self.output_root)
            return root
        except (OSError, RuntimeError, ValueError):
            return None

    def normalize(self, job, candidates):
        """Convert handler candidates into unique job-relative IDs; caller commits them."""
        root = self.root(job)
        if root is None:
            return []
        identifiers = set()
        for entry in candidates:
            try:
                path = Path(entry)
                resolved = resolve_path_allow_missing(path if path.is_absolute() else root / path)
                identifier = resolved.relative_to(root).as_posix()
                if resolved.is_file():
                    identifiers.add(identifier)
            except (OSError, RuntimeError, ValueError):
                continue
        return sorted(identifiers)

    def resolve(self, job, identifier):
        """Require an exact manifest ID and recheck all symlinks on every read."""
        if not isinstance(identifier, str) or identifier not in job.output.artifacts:
            return None
        parts = PurePosixPath(identifier)
        if (
            not identifier
            or "\\" in identifier
            or ":" in identifier
            or parts.is_absolute()
            or any(part in {".", ".."} for part in identifier.split("/"))
        ):
            return None
        root = self.root(job)
        if root is None:
            return None
        try:
            resolved = resolve_path_allow_missing(root / identifier)
            resolved.relative_to(root)
            return resolved if resolved.is_file() else None
        except (OSError, RuntimeError, ValueError):
            return None

    def list(self, job):
        """List only manifest entries that still pass read-time authorization."""
        return [
            (identifier, str(path))
            for identifier in sorted(set(job.output.artifacts))
            if (path := self.resolve(job, identifier)) is not None
        ]
