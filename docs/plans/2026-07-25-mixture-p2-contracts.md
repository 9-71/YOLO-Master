# Mixture P2 Contracts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining high-confidence Mixture P2 contract gaps without changing default routing outputs or enabling experimental algorithms.

**Architecture:** Extend the canonical routed-module capability contract with training-sparsity metadata, then let the trainer derive its DDP policy from that metadata. Make legacy gated MoE modules explicit protocol publishers, preserve capacity overflow forward behavior while restoring router gradients with a straight-through surrogate, and make the documented quality commands reproducible from the development extra.

**Tech Stack:** Python 3.8+, PyTorch, Ultralytics trainer/runtime extensions, pytest, Ruff, codespell.

---

### Task 1: Explicit AdaptiveGateMoE routed protocol

**Files:**
- Modify: `ultralytics/nn/modules/moe/gated.py`
- Modify: `tests/test_routed_module_protocol.py`

**Step 1: Write the failing test**

Add `AdaptiveGateMoE` to protocol compliance tests and assert that it exposes an initialized snapshot, publishes canonical auxiliary loss, returns a detached snapshot copy, and declares MoE training dispatch capabilities.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_routed_module_protocol.py -q`

Expected: failure because a newly constructed `AdaptiveGateMoE` lacks explicit snapshot/protocol methods.

**Step 3: Write minimal implementation**

Initialize `last_routing_snapshot` and add `publish_aux_loss()`, `routing_snapshot()`, and `export_capabilities()` to the canonical `AdaptiveGateMoE` base so descendants inherit the contract.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_routed_module_protocol.py tests/test_moe.py -q`

Expected: all selected tests pass.

### Task 2: Capability-driven DDP policy

**Files:**
- Modify: `ultralytics/nn/modules/routing_protocol.py`
- Modify: `ultralytics/nn/modules/moa/block.py`
- Modify: `ultralytics/nn/modules/moa/wrappers.py`
- Modify: `ultralytics/nn/modules/mot/block.py`
- Modify: `ultralytics/nn/modules/mot/wrappers.py`
- Modify: `ultralytics/nn/modules/latent_mixture.py`
- Modify: `ultralytics/nn/modules/moe/modules.py`
- Modify: `ultralytics/nn/peft/molora/layer.py`
- Modify: `ultralytics/engine/extensions/mixture.py`
- Modify: `ultralytics/engine/trainer.py`
- Modify: `tests/test_mot_ddp_contract.py`
- Modify: `tests/test_moe_ddp_fixes.py`

**Step 1: Write the failing tests**

Cover dense MoA, sparse-training MoT, sparse MoLoRA, and conservative unknown capability cases. Assert that compiled dense routed models may use `static_graph=True`, while any training-sparse routed module retains `find_unused_parameters=True`.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mot_ddp_contract.py tests/test_moe_ddp_fixes.py -q`

Expected: failures because the trainer currently treats every routed model as requiring unused-parameter discovery.

**Step 3: Write minimal implementation**

Add `training_sparse_dispatch` to routed capabilities, implement a conservative controller resolver, and use the resolved policy consistently for DDP preparation, `find_unused_parameters`, and `static_graph`.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mot_ddp_contract.py tests/test_moe_ddp_fixes.py tests/test_mot_sparse_parity.py -q`

Expected: all selected tests pass.

### Task 3: Capacity overflow gradient and diagnostics

**Files:**
- Modify: `ultralytics/nn/modules/moe/routers.py`
- Modify: `tests/test_moe_router_boundaries.py`

**Step 1: Write the failing tests**

Assert that overflow dispatch retains the exact deterministic one-hot forward weights, exposes mask/count/fraction diagnostics, and sends a finite non-zero gradient to overflow logits.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_moe_router_boundaries.py -q`

Expected: the gradient assertion fails under the current hard indexed assignment.

**Step 3: Write minimal implementation**

Apply the hard capacity assignment after normalizing ordinary Top-K weights and use a straight-through probability surrogate only for its backward path. Record detached overflow telemetry without changing inference or non-overflow behavior.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_moe_router_boundaries.py -q`

Expected: all selected tests pass.

### Task 4: Reproducible changed-file quality gate

**Files:**
- Modify: `pyproject.toml`
- Create: `scripts/check_changed_quality.py`
- Create: `tests/test_changed_quality.py`

**Step 1: Write the failing tests**

Cover path filtering, explicit-file mode, and command construction without invoking external tools.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_changed_quality.py -q`

Expected: failure because the changed-file quality entry point does not exist.

**Step 3: Write minimal implementation**

Add Ruff and codespell to the `dev` extra and implement a Python 3.8-compatible command that checks only changed supported files, safely passing file paths as subprocess arguments.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_changed_quality.py -q`

Expected: all selected tests pass.

### Task 5: Explicit scene-aware inference policy

**Files:**
- Modify: `ultralytics/nn/modules/mot/router.py`
- Modify: `ultralytics/nn/modules/mot/block.py`
- Modify: `ultralytics/nn/modules/mot/wrappers.py`
- Modify: `ultralytics/nn/modules/moe/config.py`
- Modify: `ultralytics/cfg/default.yaml`
- Modify: `ultralytics/cfg/__init__.py`
- Modify: `tests/test_mot_scene_aware_router.py`
- Modify: `tests/test_mixture_config_resolution.py`
- Modify: `tests/test_default_config_integrity.py`

**Step 1: Write the failing tests**

Assert that the default `dynamic` mode preserves scene-aware eval routing, explicit `bypass` mode skips scene statistics only during evaluation, training still updates the scene projector, invalid modes fail early, and runtime configuration reaches every nested MoT router.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mot_scene_aware_router.py tests/test_mixture_config_resolution.py tests/test_default_config_integrity.py -q`

Expected: failures because the explicit inference policy does not exist.

**Step 3: Write minimal implementation**

Add a validated `dynamic|bypass` router policy, plumb it through wrappers and unified configuration, and publish applied/bypass diagnostics without changing the default path.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mot_scene_aware_router.py tests/test_mixture_config_resolution.py tests/test_default_config_integrity.py -q`

Expected: all selected tests pass.

### Task 6: Integrated verification

**Files:**
- Verify all files changed by Tasks 1-4.

**Step 1: Run the focused regression gate**

Run: `pytest tests/test_routed_module_protocol.py tests/test_mot_ddp_contract.py tests/test_moe_ddp_fixes.py tests/test_mot_sparse_parity.py tests/test_moe_router_boundaries.py tests/test_changed_quality.py -q`

**Step 2: Run the broader Mixture gate**

Run: `pytest tests/test_moe.py tests/test_moa.py tests/test_mot.py tests/test_mixture_numeric.py tests/test_routing_diagnostics.py tests/test_export_capability_matrix.py -q`

**Step 3: Run scoped quality checks**

Run: `ruff check <touched Python files>`

Run: `ruff format --check <touched Python files>`

Run: `python -m compileall <touched implementation files>`

Run: `git diff --check`

Expected: all scoped checks pass. Repository-wide pre-existing Ruff and formatting debt remains outside this batch.

## Non-Goals

- Do not change the capacity overflow forward dispatch result.
- Do not disable scene-aware MoT routing in evaluation by default.
- Do not enable Expert Choice, Soft MoE, or loss-free balancing without benchmark evidence.
- Do not stage, commit, push, or blanket-format the existing dirty worktree.

## Execution Record (2026-07-25)

- The focused routed-protocol, DDP, sparse-dispatch, numerical, Latent Mixture, MoLoRA, and V-PEFT gate passed
  167 tests.
- The complete set of 23 modified or newly added test files passed 377 tests, including two-process Gloo DDP and
  blocked-Matplotlib import-resilience coverage.
- The broader MoE, MoA, MoT, routing-diagnostics, numerical, and export-capability gate passed 201 tests.
- Follow-up quality hardening removed all 98 Ruff lint violations from the 56 changed Python files while preserving
  legacy MoE and LoRA compatibility exports. The expanded MoE/LoRA regression gate passed 158 tests with 6 optional
  dependency cases skipped.
- `git diff --check` passed.
- The changed-file quality command now invokes the valid `codespell_lib` module entry point. Ruff lint passes for all
  56 changed Python files, and codespell passes for all 61 changed text files after registering the `MoT` project term.
  Its baseline-aware format policy makes the default gate pass while reporting 35 historical baseline-debt files;
  strict mode still rejects the 28 currently unformatted files. Two regressions in baseline-clean files were formatted
  directly, without blanket-formatting the active dirty worktree.
