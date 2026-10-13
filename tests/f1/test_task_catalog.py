"""Conformance tests for the shared aspirational task catalog (Handler Convergence Phase 1B).

``core.task_catalog`` is a DESCRIPTIVE, immutable catalog that records
candidate convergence links between the Studio task surface
(``core.schema.TaskType`` / ``f1.handlers.TaskHandlerRegistry``) and the Agent
skill surface (``agent.runtime.cli.dispatcher.HANDLERS``). These tests verify:

    A. the catalog data is immutable (frozen records in an immutable container);
    B. ``studio_task_id`` values are unique;
    C. ``agent_skill_id`` values are unique;
    D. every left-side id exists on the CURRENT Studio task surface (read from
       the production enum/registry, never redefined here);
    E. every right-side id exists in the CURRENT Agent ``HANDLERS`` mapping;
    F. the catalog module stays lightweight (stdlib-only, no f1/agent/web/pydantic
       dependency, verified both statically and in a clean interpreter);
    G. the catalog is not usable as a runtime registry (no register/dispatch/
       lookup API, no callables, no production importer).

The aspirational pairs -- notably ``diagnose -> yolo.system`` -- are NOT
production dispatch wiring. Nothing here asserts that an f1 diagnose job
invokes ``yolo.system``; Phase 0 (``test_handler_inventory.py``) pins the
current f1 tool name as ``system_doctor``.

Import discipline:
    Importing ``agent.runtime.cli.dispatcher`` has two process-wide side
    effects by design (``sys.path`` bootstrap and ``os.chdir`` to the repo
    root). The ``agent_handlers`` fixture reverts both immediately after the
    import and returns a plain snapshot dict.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import core.task_catalog as catalog
from core.schema import TaskType
from f1.handlers import TaskHandlerRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "core" / "task_catalog.py"

LINKS = catalog.ASPIRATIONAL_TASK_LINKS

# Static import whitelist for the catalog module (proof of stdlib-only by design).
ALLOWED_CATALOG_IMPORTS = frozenset({"__future__", "dataclasses"})

# Heavy/forbidden dependency roots for the lightweight-module guarantee.
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {"f1", "agent", "runtime", "fastapi", "starlette", "gradio", "pydantic", "ultralytics"}
)

# Names that would turn the catalog into a runtime registry/dispatcher.
FORBIDDEN_REGISTRY_NAMES = frozenset({"register", "dispatch", "get", "get_handler", "lookup", "handlers", "registry"})

# Directories skipped by the production-importer scan. The catalog is consumed
# only by tests in Phase 1B; vendored upstream code is scanned, not excluded.
SCAN_EXCLUDED_DIRS = frozenset(
    {".git", ".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__", "node_modules", "tests"}
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def agent_handlers() -> dict:
    """Snapshot Agent HANDLERS, reverting the dispatcher's import-time side effects."""
    cwd_before = os.getcwd()
    path_before = list(sys.path)
    try:
        dispatcher = importlib.import_module("agent.runtime.cli.dispatcher")
        return dict(dispatcher.HANDLERS)
    finally:
        os.chdir(cwd_before)
        sys.path[:] = path_before


def _absolute_imports(path: Path) -> set[str]:
    """Collect root module names of all absolute imports in a source file."""
    tree = ast.parse(path.read_bytes(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module)
    return roots


def _iter_production_python_files():
    """Yield repo Python files outside tests/, caches and the catalog itself."""
    for path in REPO_ROOT.rglob("*.py"):
        rel_parts = path.relative_to(REPO_ROOT).parts
        if any(part in SCAN_EXCLUDED_DIRS or part.startswith(".") for part in rel_parts):
            continue
        if path.resolve() == CATALOG_PATH.resolve():
            continue
        yield path


def _imports_core_task_catalog(tree: ast.AST) -> bool:
    """Return True if an AST imports ``core.task_catalog`` in any form."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "core.task_catalog" or alias.name.startswith("core.task_catalog.") for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "core.task_catalog" or node.module.startswith("core.task_catalog."):
                return True
            if node.module == "core" and any(alias.name == "task_catalog" for alias in node.names):
                return True
    return False


# ---------------------------------------------------------------------------
# Frozen-content pin: exactly the five aspirational pairs.
# ---------------------------------------------------------------------------
def test_catalog_pins_the_five_aspirational_pairs():
    assert {(link.studio_task_id, link.agent_skill_id) for link in LINKS} == {
        ("predict", "yolo.predict"),
        ("train", "yolo.train"),
        ("val", "yolo.val"),
        ("export", "yolo.export"),
        ("diagnose", "yolo.system"),
    }


def test_diagnose_to_yolo_system_is_aspirational_not_wired(agent_handlers):
    """The diagnose candidate exists, but only as an aspirational target.

    ``yolo.system`` is a real Agent skill, yet no f1 diagnose dispatch path
    invokes it today (active f1 tool name: ``system_doctor`` -- pinned by the
    Phase 0 inventory). The no-production-importer test below is what keeps
    this entry from becoming wiring by the back door.
    """
    candidates = {(link.studio_task_id, link.agent_skill_id) for link in LINKS}
    assert ("diagnose", "yolo.system") in candidates
    assert callable(agent_handlers["yolo.system"])
    # The catalog itself stores only strings; it cannot reach the handler.
    diagnose_link = next(link for link in LINKS if link.studio_task_id == "diagnose")
    assert diagnose_link.agent_skill_id == "yolo.system"
    assert not callable(diagnose_link.agent_skill_id)


# ---------------------------------------------------------------------------
# A. Immutability.
# ---------------------------------------------------------------------------
def test_catalog_container_is_an_immutable_tuple():
    assert isinstance(LINKS, tuple)
    assert not isinstance(LINKS, (list, dict, set, frozenset))
    with pytest.raises(TypeError):
        LINKS[0] = LINKS[0]  # type: ignore[index]
    assert not hasattr(LINKS, "append")
    assert not hasattr(LINKS, "add")
    assert not hasattr(LINKS, "clear")


def test_catalog_records_are_frozen_dataclasses():
    assert LINKS, "catalog must not be empty"
    for link in LINKS:
        assert dataclasses.is_dataclass(link)
        assert link.__dataclass_params__.frozen is True
        with pytest.raises(dataclasses.FrozenInstanceError):
            link.studio_task_id = "mutated"  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            link.agent_skill_id = "mutated"  # type: ignore[misc]
    # Frozen records are value-identical and hashable.
    assert len(set(LINKS)) == len(LINKS)


def test_catalog_records_have_exactly_two_string_fields():
    for link in LINKS:
        assert type(link) is catalog.TaskLinkCandidate
        assert [field.name for field in dataclasses.fields(link)] == ["studio_task_id", "agent_skill_id"]
        assert isinstance(link.studio_task_id, str) and link.studio_task_id
        assert isinstance(link.agent_skill_id, str) and link.agent_skill_id


# ---------------------------------------------------------------------------
# B/C. Uniqueness on both sides.
# ---------------------------------------------------------------------------
def test_studio_task_ids_are_unique():
    studio_ids = [link.studio_task_id for link in LINKS]
    duplicates = sorted({task_id for task_id in studio_ids if studio_ids.count(task_id) > 1})
    assert not duplicates, f"duplicate studio_task_id entries: {duplicates}"


def test_agent_skill_ids_are_unique():
    skill_ids = [link.agent_skill_id for link in LINKS]
    duplicates = sorted({skill_id for skill_id in skill_ids if skill_ids.count(skill_id) > 1})
    assert not duplicates, f"duplicate agent_skill_id entries: {duplicates}"


# ---------------------------------------------------------------------------
# D. Left side matches the CURRENT Studio task surface (production-defined).
# ---------------------------------------------------------------------------
def test_studio_side_ids_exist_on_live_studio_surface():
    # Production source of truth: the TaskType enum...
    enum_ids = {task.value for task in TaskType}
    # ...and the production handler registry, populated by f1.handlers imports.
    registered_ids = set(TaskHandlerRegistry.list_registered())
    assert registered_ids == enum_ids, "TaskType and TaskHandlerRegistry surfaces must agree"

    missing = sorted({link.studio_task_id for link in LINKS} - enum_ids)
    assert not missing, f"catalog references Studio tasks absent from TaskType: {missing}"


# ---------------------------------------------------------------------------
# E. Right side matches the CURRENT Agent HANDLERS mapping.
# ---------------------------------------------------------------------------
def test_agent_side_ids_exist_in_live_agent_handlers(agent_handlers):
    assert agent_handlers, "Agent HANDLERS snapshot must not be empty"
    missing = sorted({link.agent_skill_id for link in LINKS} - set(agent_handlers))
    assert not missing, f"catalog references Agent skills absent from HANDLERS: {missing}"
    for link in LINKS:
        assert callable(agent_handlers[link.agent_skill_id]), f"{link.agent_skill_id!r} value is not callable"


# ---------------------------------------------------------------------------
# F. The catalog stays lightweight: static import whitelist + clean-interpreter.
# ---------------------------------------------------------------------------
def test_catalog_imports_are_static_stdlib_only():
    imports = _absolute_imports(CATALOG_PATH)
    forbidden = {name.split(".")[0] for name in imports} & FORBIDDEN_IMPORT_ROOTS
    assert not forbidden, f"catalog must not import {sorted(forbidden)}"
    assert imports <= ALLOWED_CATALOG_IMPORTS, (
        f"unexpected catalog imports: {sorted(imports - ALLOWED_CATALOG_IMPORTS)}"
    )


def test_catalog_imports_no_forbidden_dependency_in_clean_interpreter():
    """Importing the catalog alone must not pull f1/agent/web/pydantic into sys.modules."""
    code = (
        "import sys\n"
        "import core.task_catalog as catalog\n"
        "forbidden = " + repr(tuple(sorted(FORBIDDEN_IMPORT_ROOTS))) + "\n"
        "leaked = sorted(name for name in sys.modules if name.split('.')[0] in forbidden)\n"
        "assert not leaked, leaked\n"
        "assert len(catalog.ASPIRATIONAL_TASK_LINKS) == 5\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"clean-interpreter import failed:\n{result.stderr}"


# ---------------------------------------------------------------------------
# G. The catalog is not a runtime registry and has no production consumer.
# ---------------------------------------------------------------------------
def test_catalog_exposes_no_registry_or_dispatch_api():
    # Only functions DEFINED in the module count; stdlib decorators imported by
    # it (``dataclasses.dataclass``) are part of the whitelisted import surface.
    module_functions = {
        name: func
        for name, func in inspect.getmembers(catalog, inspect.isfunction)
        if func.__module__ == catalog.__name__
    }
    assert module_functions == {}, f"catalog must define no functions: {sorted(module_functions)}"

    for forbidden_name in FORBIDDEN_REGISTRY_NAMES:
        assert not hasattr(catalog, forbidden_name), f"catalog must not expose a {forbidden_name!r} API"

    public_callables = {
        name
        for name, value in vars(catalog).items()
        if not name.startswith("_") and callable(value) and getattr(value, "__module__", None) == catalog.__name__
    }
    assert public_callables == {"TaskLinkCandidate"}, f"unexpected callable surface: {sorted(public_callables)}"

    # Records carry identifiers only -- never handler callables/deps.
    for link in LINKS:
        assert not callable(link.studio_task_id)
        assert not callable(link.agent_skill_id)


def test_no_production_module_imports_the_catalog():
    offenders: dict[str, str] = {}
    for path in _iter_production_python_files():
        try:
            # Some pre-existing scripts contain invalid escape sequences; only
            # their import graph matters here, so silence parse SyntaxWarnings.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_bytes(), filename=str(path))
        except SyntaxError:
            continue  # generated/scratch files must not mask genuine importers
        if _imports_core_task_catalog(tree):
            offenders[path.relative_to(REPO_ROOT).as_posix()] = "imports core.task_catalog"
    assert not offenders, f"Phase 1B allows no production consumer of the catalog: {offenders}"
