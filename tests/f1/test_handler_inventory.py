"""Characterization tests freezing the YOLO-Master handler inventory (Handler Convergence Phase 0).

These tests pin down the CURRENT contract surface of the Agent skill
dispatcher (``agent/runtime/cli/dispatcher.py``) and the Studio job platform
(``core`` + ``f1``) so that later convergence refactors can rely on a stable
baseline. They read static surfaces (dict keys, function signatures,
source-level imports) and verify dispatch binding with
``inspect.Signature.bind``; no handler body is ever executed. The sole
exception is ``test_agent_cli_bootstrap_imports_under_real_script_paths``,
which performs a real import of the runtime CLI inside a fully isolated
recreation of the ``run_yolo_master_skill.py`` startup environment.

The Studio->Agent table below is an ASPIRATIONAL convergence mapping only:
it is derived (never restated as literals) from the descriptive catalog
``core.task_catalog.ASPIRATIONAL_TASK_LINKS`` and is not wired into any
production dispatch path today.

Import discipline:
    Importing ``agent.runtime.cli.dispatcher`` has two process-wide side
    effects by design: it prepends ``agent/`` and the repo root to
    ``sys.path`` and calls ``os.chdir(REPO_ROOT)``. The ``agent_cli_modules``
    fixture reverts both immediately after the import so no state leaks into
    other tests of the same pytest process; the imported module objects
    remain usable from ``sys.modules`` afterwards. The CLI bootstrap test
    additionally snapshots/restores ``sys.modules`` itself.

Dual module identity:
    The agent runtime is importable as both ``agent.runtime.*`` and
    ``runtime.*`` (the dispatcher bootstraps ``agent/`` onto ``sys.path`` and
    then imports its siblings under the ``runtime.*`` identity). These tests
    read ``HANDLERS`` from the ``agent.runtime.cli.dispatcher`` identity and
    the Tier-B implementations from the ``runtime.cli.*`` identities that the
    dispatcher itself wired up. No module-level mutable state (e.g.
    ``MODULE_CACHE``) is ever asserted.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.schema import TaskType
from core.task_catalog import ASPIRATIONAL_TASK_LINKS
from f1.handlers import TaskHandlerRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Frozen inventory: the 24 skill ids currently exposed by the Agent dispatcher.
# ---------------------------------------------------------------------------
EXPECTED_AGENT_SKILLS = frozenset(
    {
        "yolo.system",
        "yolo.model.inspect",
        "yolo.train",
        "yolo.lora.train",
        "yolo.val",
        "yolo.predict",
        "yolo.track",
        "yolo.multimodal.infer",
        "yolo.multimodal.evaluate",
        "yolo.export",
        "yolo.benchmark",
        "yolo.tune",
        "yolo.lora.adapters",
        "yolo.lora.diagnose",
        "yolo.eval.peft_compare",
        "yolo.eval.sparse_sahi_compare",
        "yolo.moe.diagnose",
        "yolo.moe.prune",
        "yolo.solutions.run",
        "yolo.ui.launch",
        "yolo.pipeline.experiment",
        "yolo.release.audit",
        "yolo.job.status",
        "yolo.job.cancel",
    }
)

# Phase 0 characterization snapshot of the CURRENT Studio task surface
# (``core.schema.TaskType`` enum + ``f1.handlers.TaskHandlerRegistry``).
# SEMANTICS: this freezes what Studio ships today so unintended surface drift
# is visible. It is NOT a convergence mapping and NOT a runtime authority --
# the enum and the registry remain the production authorities. Studio may
# legitimately grow new tasks later; this snapshot is then updated on its own,
# independently of the aspirational convergence catalog.
EXPECTED_STUDIO_TASK_IDS = frozenset({"predict", "train", "val", "export", "diagnose"})

# Aspirational Studio (f1) task -> Agent skill SEMANTIC convergence mapping,
# DERIVED from the single descriptive source of truth
# (``core.task_catalog.ASPIRATIONAL_TASK_LINKS``); the five pairs are never
# restated as literals here. THIS IS NOT A PRODUCTION CONTRACT: nothing in
# core/f1/agent dispatches via this table today. The two runtimes do not even
# share a tool-name vocabulary (f1 currently ships "system_doctor"/
# "yolo_predict", see F1_ACTIVE_TOOL_NAMES). It only records the intended
# alignment target so a future convergence phase (Phase 1+) knows the planned
# semantics; it must not be read as evidence that the mapping already exists
# at runtime. Its keys are convergence CANDIDATES (a subset of the Studio
# surface), never the expected complete Studio task set.
ASPIRATIONAL_STUDIO_TO_AGENT_SKILL = {link.studio_task_id: link.agent_skill_id for link in ASPIRATIONAL_TASK_LINKS}

# f1's CURRENTLY ACTIVE agent-facing tool names (production surface:
# ``f1/skills.py``). ``system_doctor`` wraps the diagnose task and
# ``yolo_predict`` wraps the predict task. These are the real names in use
# now; Phase 1 convergence must not invent or silently rename them.
F1_ACTIVE_TOOL_NAMES = frozenset({"system_doctor", "yolo_predict"})

# Agent skills with no Studio counterpart in the aspirational mapping: 24 - 5 = 19.
EXPECTED_AGENT_ONLY_SKILLS = frozenset(
    {
        "yolo.model.inspect",
        "yolo.lora.train",
        "yolo.track",
        "yolo.multimodal.infer",
        "yolo.multimodal.evaluate",
        "yolo.benchmark",
        "yolo.tune",
        "yolo.lora.adapters",
        "yolo.lora.diagnose",
        "yolo.eval.peft_compare",
        "yolo.eval.sparse_sahi_compare",
        "yolo.moe.diagnose",
        "yolo.moe.prune",
        "yolo.solutions.run",
        "yolo.ui.launch",
        "yolo.pipeline.experiment",
        "yolo.release.audit",
        "yolo.job.status",
        "yolo.job.cancel",
    }
)

# Tier-B handlers: deps-injected implementations addressed as ``(request, deps)``.
# The train/predict variants carry one extra routing argument between the two.
# Each entry freezes only (parameter name, has_default) per parameter, in
# declaration order: parameter NAMES, ORDER and the presence of a default are
# part of the call contract, but annotation TEXT is deliberately not frozen
# (re-typing hints must not break this baseline). Dispatcher wrapper functions
# in ``agent/runtime/cli/dispatcher.py`` are intentionally NOT frozen here.
TIER_B_HANDLER_SIGNATURES = {
    "runtime.cli.core_handlers": {
        "run_train_like": (
            ("request", False),
            ("skill_name", False),
            ("deps", False),
        ),
        "run_val": (
            ("request", False),
            ("deps", False),
        ),
        "run_predict_like": (
            ("request", False),
            ("mode", False),
            ("deps", False),
        ),
        "run_export": (
            ("request", False),
            ("deps", False),
        ),
        "run_benchmark": (
            ("request", False),
            ("deps", False),
        ),
        "run_tune": (
            ("request", False),
            ("deps", False),
        ),
        "run_lora_adapters": (
            ("request", False),
            ("deps", False),
        ),
    },
    "runtime.cli.system_handlers": {
        "run_system": (
            ("request", False),
            ("deps", False),
        ),
    },
    "runtime.cli.model_handlers": {
        "run_model_inspect": (
            ("request", False),
            ("deps", False),
        ),
    },
    "runtime.cli.multimodal_handlers": {
        "run_multimodal_infer": (
            ("request", False),
            ("deps", False),
        ),
        "run_multimodal_evaluate": (
            ("request", False),
            ("deps", False),
        ),
    },
    "runtime.cli.lora_tools": {
        "run_lora_diagnose": (
            ("request", False),
            ("deps", False),
        ),
    },
    "runtime.cli.peft_compare": {
        "run_peft_compare": (
            ("request", False),
            ("deps", False),
        ),
    },
    "runtime.cli.launcher_handlers": {
        "run_solutions": (
            ("request", False),
            ("deps", False),
        ),
        "run_ui_launch": (
            ("request", False),
            ("deps", False),
        ),
    },
    "runtime.cli.job_handlers": {
        "run_job_status": (
            ("request", False),
            ("deps", True),
        ),
        "run_job_cancel": (
            ("request", False),
            ("deps", True),
        ),
    },
    "runtime.cli.pipeline": {
        "run_experiment_pipeline": (
            ("request", False),
            ("deps", False),
        ),
    },
}

# The 4 skills whose HANDLERS value IS the Tier-B impl itself, wired with no
# dispatcher wrapper around it. Their single-positional-``request`` signature
# is therefore the live CLI dispatch contract: changing it breaks CLI
# dispatch at runtime rather than at wrapper construction time.
BARE_IMPL_HANDLERS = {
    "yolo.eval.sparse_sahi_compare": ("runtime.cli.sahi_compare", "run_sahi_compare"),
    "yolo.moe.diagnose": ("runtime.cli.moe_tools", "run_moe_diagnose"),
    "yolo.moe.prune": ("runtime.cli.moe_tools", "run_moe_prune"),
    "yolo.release.audit": ("runtime.cli.release", "run_release_audit"),
}

# ``runtime.cli.*`` modules this test file reads (all are imported by the
# dispatcher itself, so they are grabbed from ``sys.modules`` post-import
# rather than re-imported under a second identity).
RUNTIME_CLI_MODULES = tuple(TIER_B_HANDLER_SIGNATURES) + tuple(
    {module_name for module_name, _ in BARE_IMPL_HANDLERS.values()}
)

# pipeline.execute_stage stage -> PipelineDeps method routing table.
STAGE_TO_DEPS_METHOD = {
    "system": "run_system",
    "inspect": "run_model_inspect",
    "train": "run_train_like",
    "val": "run_val",
    "lora_diagnose": "run_lora_diagnose",
    "moe_diagnose": "run_moe_diagnose",
    "export": "run_export",
    "benchmark": "run_benchmark",
    "peft_compare": "run_peft_compare",
}

# All callable fields of the frozen ``PipelineDeps`` dataclass, so tests can
# build a fully stubbed instance without touching real implementations.
PIPELINE_DEPS_FIELDS = (
    "normalize_request",
    "is_dry_run",
    "response",
    "plan_response",
    "write_manifest",
    "best_checkpoint",
    "run_system",
    "run_model_inspect",
    "run_train_like",
    "run_val",
    "run_export",
    "run_benchmark",
    "run_lora_diagnose",
    "run_moe_diagnose",
    "run_peft_compare",
)

# Import-direction guards (static, source level).
CORE_FORBIDDEN_IMPORT_ROOTS = frozenset({"f1", "agent", "runtime", "fastapi", "gradio"})
AGENT_FORBIDDEN_IMPORT_ROOTS = frozenset({"f1"})


@pytest.fixture(scope="module")
def agent_cli_modules():
    """Import the agent CLI dispatcher once, reverting its process-wide side effects.

    ``agent/runtime/cli/dispatcher.py`` prepends ``agent/`` and the repo root
    to ``sys.path`` and calls ``os.chdir(REPO_ROOT)`` at import time. Both
    mutations are reverted immediately after the import so they cannot
    pollute other tests; the imported module objects stay valid in
    ``sys.modules`` and are safe to read afterwards.
    """
    cwd_before = os.getcwd()
    path_before = list(sys.path)
    try:
        dispatcher = importlib.import_module("agent.runtime.cli.dispatcher")
        modules = {name: sys.modules[name] for name in RUNTIME_CLI_MODULES}
    finally:
        os.chdir(cwd_before)
        sys.path[:] = path_before
    return SimpleNamespace(dispatcher=dispatcher, modules=modules)


def _signature_surface(func) -> tuple:
    """Return a frozen ``(name, has_default)`` surface for a callable.

    Only parameter names, declaration order and default-presence are frozen;
    annotation text is intentionally ignored.
    """
    return tuple(
        (param.name, param.default is not inspect.Parameter.empty)
        for param in inspect.signature(func).parameters.values()
    )


def _absolute_import_roots(path: Path) -> set[str]:
    """Collect top-level package roots of all absolute imports in a source file."""
    tree = ast.parse(path.read_bytes(), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _build_stub_deps(pipeline, calls: list):
    """Build a ``PipelineDeps`` instance whose fields all record invocations."""

    def _recorder(field_name):
        def _stub(*args):
            calls.append((field_name, args))
            return {"status": "ok", "stage_handler": field_name}

        return _stub

    return pipeline.PipelineDeps(**{field: _recorder(field) for field in PIPELINE_DEPS_FIELDS})


# ---------------------------------------------------------------------------
# 1. Agent HANDLERS inventory: exactly the 24 frozen skill ids, all callable.
# ---------------------------------------------------------------------------
def test_agent_handlers_inventory_is_frozen(agent_cli_modules):
    handlers = agent_cli_modules.dispatcher.HANDLERS
    assert len(handlers) == 24
    assert set(handlers) == EXPECTED_AGENT_SKILLS


def test_agent_handler_values_are_callable(agent_cli_modules):
    handlers = agent_cli_modules.dispatcher.HANDLERS
    non_callable = sorted(skill for skill, handler in handlers.items() if not callable(handler))
    assert not non_callable, f"HANDLERS values must be callable: {non_callable}"


# ---------------------------------------------------------------------------
# 2. Real dispatch invariant: every HANDLERS value binds a single positional
#    ``request`` (the exact way ``main()`` invokes it). This binds only — no
#    handler body runs — and explicitly covers the 4 directly-wired bare impls.
# ---------------------------------------------------------------------------
def test_every_handler_binds_a_single_positional_request(agent_cli_modules):
    handlers = agent_cli_modules.dispatcher.HANDLERS
    for skill, handler in handlers.items():
        signature = inspect.signature(handler)
        request = {"skill": skill}
        try:
            signature.bind(request)
        except TypeError as exc:
            pytest.fail(f"HANDLERS[{skill!r}] cannot be dispatched with one positional request: {exc}")
        # The request argument is required: zero positional args must not bind.
        with pytest.raises(TypeError):
            signature.bind()


def test_bare_impl_handlers_are_directly_wired_and_request_callable(agent_cli_modules):
    handlers = agent_cli_modules.dispatcher.HANDLERS
    dispatcher_name = agent_cli_modules.dispatcher.__name__
    bare_skills = set(BARE_IMPL_HANDLERS)
    assert bare_skills <= set(handlers)
    # The 4 bare impls are the ONLY HANDLERS values not defined in the dispatcher module.
    non_dispatcher = sorted(skill for skill, handler in handlers.items() if handler.__module__ != dispatcher_name)
    assert non_dispatcher == sorted(bare_skills)
    for skill, (module_name, func_name) in BARE_IMPL_HANDLERS.items():
        impl = getattr(agent_cli_modules.modules[module_name], func_name)
        assert handlers[skill] is impl, f"{skill!r} must dispatch the bare impl {module_name}.{func_name}"
        inspect.signature(impl).bind({"skill": skill})  # single positional request, no execution


# ---------------------------------------------------------------------------
# 3a. Studio characterization snapshot (Phase 0): the CURRENT Studio surface
#     is pinned against EXPECTED_STUDIO_TASK_IDS -- an independent snapshot,
#     not the aspirational mapping. A future legitimate Studio task addition
#     updates that snapshot and must not fail merely because the convergence
#     catalog does not list the new task.
# ---------------------------------------------------------------------------
def test_studio_task_type_enum_matches_five_tasks():
    assert {task.value for task in TaskType} == set(EXPECTED_STUDIO_TASK_IDS)


def test_studio_registry_registers_exactly_the_five_tasks():
    assert TaskHandlerRegistry.list_registered() == sorted(EXPECTED_STUDIO_TASK_IDS)


def test_f1_active_tool_names_are_frozen():
    # Production surface today: f1/skills.py exposes exactly these two tool names.
    from f1.skills import PredictSkill, SystemDoctorSkill

    active = {SystemDoctorSkill().name, PredictSkill().name}
    assert active == set(F1_ACTIVE_TOOL_NAMES)


# ---------------------------------------------------------------------------
# 3b. Aspirational catalog coverage (subset-only): the catalog constrains
#     only its own candidates -- every left side must currently exist on the
#     Studio surface and every right side in Agent HANDLERS. It does NOT
#     define the complete Studio surface (see EXPECTED_STUDIO_TASK_IDS) or
#     the complete Agent inventory (see EXPECTED_AGENT_SKILLS).
# ---------------------------------------------------------------------------
def test_aspirational_catalog_left_sides_exist_on_studio_surface():
    studio_surface = {task.value for task in TaskType}
    # The two production Studio authorities must agree with each other...
    assert set(TaskHandlerRegistry.list_registered()) == studio_surface
    # ...and every catalog candidate must live on that surface (subset only).
    missing = sorted(set(ASPIRATIONAL_STUDIO_TO_AGENT_SKILL) - studio_surface)
    assert not missing, f"aspirational catalog candidates absent from the Studio surface: {missing}"


def test_aspirational_target_skills_exist_in_agent_inventory(agent_cli_modules):
    handlers = agent_cli_modules.dispatcher.HANDLERS
    for task, skill in ASPIRATIONAL_STUDIO_TO_AGENT_SKILL.items():
        assert skill in handlers, f"aspirational target for Studio task '{task}' is missing: '{skill}'"
        assert callable(handlers[skill])


def test_agent_only_skills_are_the_19_unmapped(agent_cli_modules):
    handlers = agent_cli_modules.dispatcher.HANDLERS
    agent_only = set(handlers) - set(ASPIRATIONAL_STUDIO_TO_AGENT_SKILL.values())
    assert len(agent_only) == 19
    assert agent_only == EXPECTED_AGENT_ONLY_SKILLS


# ---------------------------------------------------------------------------
# 4. Tier-B handler (request, deps) signatures (dispatcher wrappers excluded).
#    Names/order/default-presence only; annotation text is not frozen.
# ---------------------------------------------------------------------------
_TIER_B_CASES = [
    (module_name, func_name, expected)
    for module_name, functions in TIER_B_HANDLER_SIGNATURES.items()
    for func_name, expected in functions.items()
]


@pytest.mark.parametrize(
    ("module_name", "func_name", "expected"),
    _TIER_B_CASES,
    ids=[f"{module_name}.{func_name}" for module_name, func_name, _ in _TIER_B_CASES],
)
def test_tier_b_handler_signature_is_frozen(agent_cli_modules, module_name, func_name, expected):
    func = getattr(agent_cli_modules.modules[module_name], func_name)
    assert _signature_surface(func) == expected
    # The Tier-B contract: first parameter is the request, and a deps parameter exists.
    assert expected[0][0] == "request"
    assert any(name == "deps" for name, _ in expected)


# ---------------------------------------------------------------------------
# 5. Agent CLI bootstrap guard: ``python agent/scripts/run_yolo_master_skill.py``
#    must import the runtime CLI with only the script dir and ``agent/`` on
#    sys.path (plus stdlib/site-packages), from any cwd. A future module-level
#    ``from core...`` / ``from f1...`` in the runtime import chain would raise
#    ModuleNotFoundError here exactly as it would in the real CLI.
# ---------------------------------------------------------------------------
_BOOTSTRAP_PURGE_ROOTS = frozenset({"runtime", "agent", "core", "f1"})


def test_agent_cli_bootstrap_imports_under_real_script_paths(tmp_path):
    scripts_dir = REPO_ROOT / "agent" / "scripts"
    skill_root = REPO_ROOT / "agent"
    repo_root_str = os.path.normcase(str(REPO_ROOT))
    skill_root_str = os.path.normcase(str(skill_root))

    cwd_before = os.getcwd()
    path_before = list(sys.path)
    modules_before = dict(sys.modules)

    # What a plain script launch provides: sys.path[0] is the script directory;
    # the repo root (pytest's cwd insertion) and agent/ must NOT be present.
    minimal_path = [str(scripts_dir)]
    for path in path_before:
        if not path:
            continue  # "" means cwd; a script launch does not add cwd.
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized in {repo_root_str, skill_root_str}:
            continue
        minimal_path.append(path)

    def _is_bootstrapped_module(name: str) -> bool:
        return name.split(".", 1)[0] in _BOOTSTRAP_PURGE_ROOTS

    try:
        # Force a fresh import so finders are actually consulted.
        for name in [n for n in sys.modules if _is_bootstrapped_module(n)]:
            del sys.modules[name]
        os.chdir(tmp_path)
        sys.path[:] = minimal_path

        # Sanity pin: without the script's own bootstrap, upper layers are unreachable.
        for forbidden in ("core", "f1"):
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(forbidden)

        # Mirror the exact prelude of agent/scripts/run_yolo_master_skill.py.
        if str(skill_root) not in sys.path:
            sys.path.insert(0, str(skill_root))
        dispatcher = importlib.import_module("runtime.cli.dispatcher")

        # Under the real CLI the bare ``runtime.*`` identity is used, never ``agent.*``.
        assert dispatcher.__name__ == "runtime.cli.dispatcher"
        assert "agent.runtime.cli.dispatcher" not in sys.modules
        assert callable(dispatcher.main)
        assert len(dispatcher.HANDLERS) == 24
    finally:
        # Drop every module created during bootstrap, then restore the full snapshot.
        for name in [n for n in list(sys.modules) if n not in modules_before]:
            del sys.modules[name]
        sys.modules.clear()
        sys.modules.update(modules_before)
        sys.path[:] = path_before
        os.chdir(cwd_before)


# ---------------------------------------------------------------------------
# 6. Import-direction guards (static source scan, no imports executed).
# ---------------------------------------------------------------------------
def test_core_does_not_depend_on_upper_layers():
    offenders = {}
    for path in sorted((REPO_ROOT / "core").rglob("*.py")):
        forbidden = _absolute_import_roots(path) & CORE_FORBIDDEN_IMPORT_ROOTS
        if forbidden:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = sorted(forbidden)
    assert not offenders, f"core/ must not import {sorted(CORE_FORBIDDEN_IMPORT_ROOTS)}: {offenders}"


def test_agent_does_not_depend_on_f1():
    offenders = {}
    for path in sorted((REPO_ROOT / "agent").rglob("*.py")):
        forbidden = _absolute_import_roots(path) & AGENT_FORBIDDEN_IMPORT_ROOTS
        if forbidden:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = sorted(forbidden)
    assert not offenders, f"agent/ must not import {sorted(AGENT_FORBIDDEN_IMPORT_ROOTS)}: {offenders}"


# ---------------------------------------------------------------------------
# 7. Pipeline consistency: stages, skills and execute_stage routing agree.
# ---------------------------------------------------------------------------
def test_stage_skills_values_are_subset_of_handlers(agent_cli_modules):
    pipeline = agent_cli_modules.modules["runtime.cli.pipeline"]
    handlers = agent_cli_modules.dispatcher.HANDLERS
    missing = set(pipeline.STAGE_SKILLS.values()) - set(handlers)
    assert not missing, f"STAGE_SKILLS references skills missing from HANDLERS: {sorted(missing)}"


def test_stage_skills_keys_match_default_stage_order_as_a_set(agent_cli_modules):
    pipeline = agent_cli_modules.modules["runtime.cli.pipeline"]
    # Set consistency only: the dict's declaration order carries no semantics
    # (DEFAULT_STAGE_ORDER is the sequenced surface), so it must not be frozen.
    assert set(pipeline.STAGE_SKILLS) == set(pipeline.DEFAULT_STAGE_ORDER)
    assert len(pipeline.DEFAULT_STAGE_ORDER) == len(set(pipeline.DEFAULT_STAGE_ORDER))


def test_execute_stage_routes_every_known_stage(agent_cli_modules):
    pipeline = agent_cli_modules.modules["runtime.cli.pipeline"]
    # Guard: this test's routing table must cover exactly the frozen stages.
    assert set(STAGE_TO_DEPS_METHOD) == set(pipeline.STAGE_SKILLS)
    for stage, expected_method in STAGE_TO_DEPS_METHOD.items():
        calls = []
        deps = _build_stub_deps(pipeline, calls)
        skill = pipeline.STAGE_SKILLS[stage]
        stage_request = {"skill": skill}
        payload = pipeline.execute_stage(stage, skill, stage_request, deps)
        assert [call[0] for call in calls] == [expected_method], f"stage '{stage}' routed incorrectly"
        assert payload == {"status": "ok", "stage_handler": expected_method}
        if stage == "train":
            # The train branch forwards the resolved skill as second positional arg.
            assert calls[0][1] == (stage_request, skill)
        else:
            assert calls[0][1] == (stage_request,)


def test_execute_stage_rejects_unknown_stage(agent_cli_modules):
    pipeline = agent_cli_modules.modules["runtime.cli.pipeline"]
    deps = _build_stub_deps(pipeline, [])
    with pytest.raises(ValueError, match="Unsupported pipeline stage"):
        pipeline.execute_stage("not_a_stage", "yolo.unused", {}, deps)
