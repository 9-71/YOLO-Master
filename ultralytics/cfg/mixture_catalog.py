"""Read-only discovery and filtering of packaged mixture model profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from ultralytics.mixture_metadata import MIXTURE_KINDS, MIXTURE_MODULE_KINDS
from ultralytics.utils import ROOT, YAML


DEFAULT_MIXTURE_MODEL_ROOT = ROOT / "cfg" / "models"
PROFILE_TASKS = ("classify", "detect", "obb", "pose", "segment", "semantic", "unknown")


class MixtureCatalogError(ValueError):
    """Raised when model files cannot form a deterministic, safe catalog."""


@dataclass(frozen=True)
class MixtureProfile:
    """Immutable metadata for one runnable mixture model YAML profile."""

    profile_id: str
    path: str
    task: str
    family: str
    scales: tuple[str, ...]
    mixture_kinds: tuple[str, ...]
    mixture_modules: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        return {
            "profile_id": self.profile_id,
            "path": self.path,
            "task": self.task,
            "family": self.family,
            "scales": list(self.scales),
            "mixture_kinds": list(self.mixture_kinds),
            "mixture_modules": list(self.mixture_modules),
        }


def _catalog_root(root: Optional[Union[str, Path]]) -> Path:
    """Resolve and validate the selected model catalog root."""
    path = Path(root) if root is not None else DEFAULT_MIXTURE_MODEL_ROOT
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise MixtureCatalogError(f"mixture catalog root does not exist: {path}") from exc
    if not resolved.is_dir():
        raise MixtureCatalogError(f"mixture catalog root is not a directory: {path}")
    return resolved


def _candidate_paths(root: Path) -> tuple[tuple[Path, Path], ...]:
    """Return safe candidate paths paired with their root-relative paths."""
    candidates = sorted((*root.rglob("*.yaml"), *root.rglob("*.yml")), key=lambda path: path.as_posix())
    resolved_candidates = []
    identifiers = {}
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise MixtureCatalogError(f"model profile '{candidate}' escapes catalog root '{root}'") from exc
        except OSError as exc:
            raise MixtureCatalogError(f"unable to resolve model profile '{candidate}': {exc}") from exc
        profile_id = relative.with_suffix("").as_posix()
        if profile_id in identifiers:
            first = identifiers[profile_id]
            raise MixtureCatalogError(
                f"duplicate mixture profile id '{profile_id}' from '{first.as_posix()}' and '{relative.as_posix()}'"
            )
        identifiers[profile_id] = relative
        resolved_candidates.append((resolved, relative))
    return tuple(resolved_candidates)


def _layer_modules(config: dict[str, Any], path: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate model layer declarations and return backbone and head module names."""
    sections = []
    for section_name in ("backbone", "head"):
        layers = config.get(section_name, [])
        if not isinstance(layers, list):
            raise MixtureCatalogError(f"invalid model profile '{path}': '{section_name}' must be a list")
        modules = []
        for index, layer in enumerate(layers):
            if not isinstance(layer, (list, tuple)) or len(layer) < 3:
                raise MixtureCatalogError(
                    f"invalid model profile '{path}': {section_name}[{index}] expected at least 3 items"
                )
            module = layer[2]
            if not isinstance(module, str):
                raise MixtureCatalogError(
                    f"invalid model profile '{path}': {section_name}[{index}] module name must be a string"
                )
            modules.append(module)
        sections.append(tuple(modules))
    return sections[0], sections[1]


def _infer_task(config: dict[str, Any], head_modules: tuple[str, ...], path: str) -> str:
    """Infer task from explicit metadata or the final head module."""
    explicit = config.get("task")
    if explicit is not None:
        if not isinstance(explicit, str) or explicit.casefold() not in PROFILE_TASKS[:-1]:
            raise MixtureCatalogError(f"invalid model profile '{path}': unknown explicit task {explicit!r}")
        return explicit.casefold()
    if not head_modules:
        return "unknown"
    module = head_modules[-1].casefold()
    if "semanticsegment" in module:
        return "semantic"
    if "classify" in module or module in {"classifier", "cls", "fc"}:
        return "classify"
    if "detect" in module:
        return "detect"
    if "segment" in module:
        return "segment"
    if "pose" in module:
        return "pose"
    if "obb" in module:
        return "obb"
    return "unknown"


def _family(relative: Path) -> str:
    """Derive a stable family label from the model directory layout."""
    parts = relative.parts
    if len(parts) > 2 and parts[0] == "master":
        return f"master/{parts[1]}"
    return parts[0] if len(parts) > 1 else "root"


def _scales(config: dict[str, Any], path: str) -> tuple[str, ...]:
    """Return the sorted scale names declared by a profile."""
    scales = config.get("scales")
    if scales is None:
        scale = config.get("scale")
        return (str(scale),) if scale is not None else ()
    if not isinstance(scales, dict):
        raise MixtureCatalogError(f"invalid model profile '{path}': 'scales' must be a mapping")
    return tuple(sorted(str(scale) for scale in scales))


def discover_mixture_profiles(
    root: Optional[Union[str, Path]] = None,
) -> tuple[MixtureProfile, ...]:
    """Discover mixture profiles below a model YAML root without constructing models."""
    catalog_root = _catalog_root(root)
    profiles = []
    for candidate, relative in _candidate_paths(catalog_root):
        relative_path = relative.as_posix()
        try:
            config = YAML.load(candidate)
        except (AssertionError, OSError, ValueError) as exc:
            raise MixtureCatalogError(f"invalid model profile '{relative_path}': {exc}") from exc
        backbone_modules, head_modules = _layer_modules(config, relative_path)
        module_names = tuple(dict.fromkeys((*backbone_modules, *head_modules)))
        mixture_modules = tuple(module for module in module_names if module in MIXTURE_MODULE_KINDS)
        if not mixture_modules:
            continue
        used_kinds = {MIXTURE_MODULE_KINDS[module] for module in mixture_modules}
        mixture_kinds = tuple(kind for kind in MIXTURE_KINDS if kind in used_kinds)
        profiles.append(
            MixtureProfile(
                profile_id=relative.with_suffix("").as_posix(),
                path=relative_path,
                task=_infer_task(config, head_modules, relative_path),
                family=_family(relative),
                scales=_scales(config, relative_path),
                mixture_kinds=mixture_kinds,
                mixture_modules=mixture_modules,
            )
        )
    return tuple(sorted(profiles, key=lambda profile: profile.profile_id))


def list_mixture_profiles(
    kind: Optional[str] = None,
    task: Optional[str] = None,
    family: Optional[str] = None,
    *,
    root: Optional[Union[str, Path]] = None,
) -> tuple[MixtureProfile, ...]:
    """List profiles matching exact case-insensitive metadata filters."""
    normalized_kind = kind.casefold() if isinstance(kind, str) else kind
    normalized_task = task.casefold() if isinstance(task, str) else task
    normalized_family = family.casefold() if isinstance(family, str) else family
    if normalized_kind is not None and normalized_kind not in MIXTURE_KINDS:
        raise ValueError(f"unknown mixture kind {kind!r}; expected one of {MIXTURE_KINDS}")
    if normalized_task is not None and normalized_task not in PROFILE_TASKS:
        raise ValueError(f"unknown mixture task {task!r}; expected one of {PROFILE_TASKS}")
    if family is not None and (not isinstance(family, str) or not family.strip()):
        raise ValueError("mixture family filter must be a non-empty string")

    return tuple(
        profile
        for profile in discover_mixture_profiles(root)
        if (normalized_kind is None or normalized_kind in profile.mixture_kinds)
        and (normalized_task is None or normalized_task == profile.task.casefold())
        and (normalized_family is None or normalized_family == profile.family.casefold())
    )


def get_mixture_profile(profile_id: str, *, root: Optional[Union[str, Path]] = None) -> MixtureProfile:
    """Return one profile by its exact root-relative identifier."""
    for profile in discover_mixture_profiles(root):
        if profile.profile_id == profile_id:
            return profile
    raise KeyError(f"unknown mixture profile {profile_id!r}")


__all__ = (
    "DEFAULT_MIXTURE_MODEL_ROOT",
    "MIXTURE_KINDS",
    "MixtureCatalogError",
    "MixtureProfile",
    "discover_mixture_profiles",
    "get_mixture_profile",
    "list_mixture_profiles",
)
