"""Tests for read-only discovery of packaged mixture model profiles."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest

from ultralytics.cfg.mixture_catalog import (
    MixtureCatalogError,
    discover_mixture_profiles,
    get_mixture_profile,
    list_mixture_profiles,
)
from ultralytics.mixture_metadata import MIXTURE_MODULE_KINDS
from ultralytics.utils import ROOT, YAML


def _write_profile(
    path: Path,
    modules: tuple[str, ...],
    *,
    head: str = "Detect",
    scales: tuple[str, ...] = ("n",),
    task: Optional[str] = None,
) -> None:
    """Write a minimal model profile for catalog tests."""
    data = {
        "nc": 1,
        "scales": {scale: [0.5, 0.25, 1024] for scale in scales},
        "backbone": [[-1, 1, module, [64]] for module in modules],
        "head": [[-1, 1, head, ["nc"]]],
    }
    if task is not None:
        data["task"] = task
    YAML.save(path, data)


def test_discovery_classifies_hybrid_profiles_and_metadata(tmp_path: Path) -> None:
    """Catalog profiles by registered modules while preserving deterministic metadata."""
    root = tmp_path / "models"
    _write_profile(root / "26" / "z-moe.yaml", ("A2C2fMoE",), scales=("s", "n"))
    _write_profile(
        root / "master" / "v0_10" / "seg" / "a-hybrid.yaml",
        ("VisualEnhancedAdaptiveGateMoE", "C2fMoA", "C2fMoT"),
        head="Segment",
    )
    _write_profile(root / "master" / "v0_10" / "obb" / "latent.yaml", ("LatentMixture",), head="OBB")
    _write_profile(root / "26" / "native.yaml", ("C3k2",))

    profiles = discover_mixture_profiles(root)

    assert [profile.profile_id for profile in profiles] == [
        "26/z-moe",
        "master/v0_10/obb/latent",
        "master/v0_10/seg/a-hybrid",
    ]
    moe, latent, hybrid = profiles
    assert moe.path == "26/z-moe.yaml"
    assert moe.task == "detect"
    assert moe.family == "26"
    assert moe.scales == ("n", "s")
    assert moe.mixture_kinds == ("moe",)
    assert moe.mixture_modules == ("A2C2fMoE",)
    assert latent.mixture_kinds == ("latent",)
    assert hybrid.task == "segment"
    assert hybrid.family == "master/v0_10"
    assert hybrid.mixture_kinds == ("moe", "moa", "mot")
    assert hybrid.mixture_modules == ("VisualEnhancedAdaptiveGateMoE", "C2fMoA", "C2fMoT")

    with pytest.raises(AttributeError):
        hybrid.task = "detect"


def test_filters_and_lookup_are_exact_case_insensitive(tmp_path: Path) -> None:
    """Filter profiles without fuzzy matches and resolve exact stable identifiers."""
    root = tmp_path / "models"
    _write_profile(root / "master" / "v0_10" / "det" / "mot.yaml", ("C2fMoT",))
    _write_profile(root / "master" / "v0_10" / "seg" / "moa.yaml", ("C2fMoA",), head="Segment")

    profiles = list_mixture_profiles(kind="MOT", task="DETECT", family="MASTER/V0_10", root=root)

    assert [profile.profile_id for profile in profiles] == ["master/v0_10/det/mot"]
    assert get_mixture_profile("master/v0_10/det/mot", root=root) == profiles[0]
    with pytest.raises(KeyError, match="unknown mixture profile"):
        get_mixture_profile("../mot", root=root)
    with pytest.raises(ValueError, match="unknown mixture kind"):
        list_mixture_profiles(kind="router", root=root)


def test_explicit_task_overrides_head_inference(tmp_path: Path) -> None:
    """Honor an explicit valid task while representing an unknown head honestly."""
    root = tmp_path / "models"
    _write_profile(root / "custom" / "explicit.yaml", ("C2fMoA",), head="CustomHead", task="pose")
    _write_profile(root / "custom" / "unknown.yaml", ("C2fMoT",), head="CustomHead")

    explicit, unknown = discover_mixture_profiles(root)

    assert explicit.task == "pose"
    assert unknown.task == "unknown"


@pytest.mark.parametrize(
    "payload, match",
    [
        ("backbone: [", "YAML syntax error"),
        ("backbone:\n  - [-1, 1]\nhead: []\n", "expected at least 3 items"),
        ("backbone:\n  - [-1, 1, 42, []]\nhead: []\n", "module name must be a string"),
    ],
)
def test_invalid_yaml_and_layer_records_have_path_context(tmp_path: Path, payload: str, match: str) -> None:
    """Reject invalid candidates with an actionable profile path."""
    root = tmp_path / "models"
    path = root / "broken.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(MixtureCatalogError, match=match) as error:
        discover_mixture_profiles(root)

    assert "broken.yaml" in str(error.value)


def test_duplicate_profile_ids_are_rejected(tmp_path: Path) -> None:
    """Reject a YAML/YML stem collision instead of choosing one by scan order."""
    root = tmp_path / "models"
    _write_profile(root / "duplicate.yaml", ("C2fMoA",))
    _write_profile(root / "duplicate.yml", ("C2fMoT",))

    with pytest.raises(MixtureCatalogError, match="duplicate mixture profile id 'duplicate'"):
        discover_mixture_profiles(root)


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    """Do not read a candidate symlink that resolves outside the selected root."""
    root = tmp_path / "models"
    outside = tmp_path / "outside.yaml"
    _write_profile(outside, ("C2fMoA",))
    root.mkdir()
    try:
        (root / "escaped.yaml").symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks is not permitted on this platform")

    with pytest.raises(MixtureCatalogError, match="escapes catalog root"):
        discover_mixture_profiles(root)


def test_metadata_covers_runtime_registry() -> None:
    """Keep lightweight catalog classification in lockstep with runtime registration."""
    from ultralytics.nn.mixture_registry import MIXTURE_MODULES

    assert set(MIXTURE_MODULE_KINDS) == set(MIXTURE_MODULES)


def test_catalog_import_does_not_load_nn_runtime() -> None:
    """Keep discovery imports independent of model tasks and runtime module registration."""
    code = (
        "import sys; "
        "from ultralytics.cfg.mixture_catalog import discover_mixture_profiles; "
        "assert 'ultralytics.nn.tasks' not in sys.modules; "
        "assert 'ultralytics.nn.mixture_registry' not in sys.modules"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


def test_packaged_catalog_contains_representative_profiles() -> None:
    """Discover representative packaged profiles without constructing their models."""
    profiles = {profile.profile_id: profile for profile in discover_mixture_profiles(ROOT / "cfg/models")}

    assert len(profiles) >= 332
    assert profiles["26/yolo26-master-n"].mixture_kinds == ("moe",)
    assert profiles["26/yolo26-master-moa-n"].mixture_kinds == ("moa",)
    assert profiles["26/yolo26-master-mot-n"].mixture_kinds == ("mot",)
    assert profiles["26/yolo26-master-moa-mot-n"].mixture_kinds == ("moa", "mot")
    assert profiles["26/yolo26-master-latent-n"].mixture_kinds == ("moe", "latent")
