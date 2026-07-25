# Mixture Maturity 9.0 Implementation Plan

**Goal:** Close the highest-impact maturity gaps identified in the MoE/MoA/MoT/PEFT/Latent analysis with behavior-level contracts and regression coverage.

**Architecture:** Keep the existing routing protocols and merge APIs, but make their defaults explicit and auditable. Optional diagnostics dependencies load lazily, MoT sparse dispatch becomes an intentional training policy, latent auxiliary loss is enabled at a conservative default gain, and V-PEFT plans are validated against the model before adapter injection.

**Tech Stack:** Python, PyTorch, Ultralytics configuration, pytest.

---

### Task 1: Restore import resilience

- Move the optional Matplotlib import in `ultralytics/nn/modules/moe/analysis.py` into the visualization method.
- Keep text diagnostics usable when plotting dependencies are unavailable or ABI-incompatible.
- Verify module imports and the focused routing tests collect.

### Task 2: Make latent routing regularization effective by default

- Change the conservative default latent auxiliary gain from `0.0` to `0.1` in the YAML and resolver fallback.
- Add a regression test proving a latent-only routed model contributes a non-zero composite auxiliary loss without an explicit override.
- Verify latent and mixture-loss tests.

### Task 3: Make MoT sparse training auditable

- Keep the explicit `sparse_train` opt-in API for backward compatibility, while add dispatch telemetry (`selected_experts`, `sparsity_ratio`, and policy) to the block snapshot.
- Expose the policy through `export_capabilities` and add tests for dense and sparse training modes.
- Verify MoT sparse parity and configuration tests.

### Task 4: Complete V-PEFT to adapter contract validation

- Add `PlacementPlan.validate_model(model, require_targets=True)` to validate model fingerprint, target existence, rank bounds, and supported layer types.
- Run the validation before applying a V-PEFT plan and fall back safely when the plan is stale or incompatible.
- Add tests for valid and stale plans.

### Task 5: Verify and report

- Run focused pytest suites, Ruff on touched files, and lightweight import/model smoke checks.
- Record any environment blockers separately from code failures.

### Task 6: Consume V-PEFT plans in MoLoRA

- Extend the MoLoRA entry point and wrapper to accept a `PlacementPlan` or serialized plan mapping.
- Validate plan status, model binding, target compatibility, variant semantics, and all per-target ranks before mutation.
- Construct each `MoLoRALayer` with its planned rank and attach auditable plan/rank metadata.
- Add regression tests for heterogeneous ranks and atomic rejection of invalid plans.

### Task 7: Audit latent cold starts

- Add routing snapshot fields for identity cold-start state, residual-gain magnitude, and local router auxiliary-gradient enablement.
- Add a backward regression proving the default zero-residual/default-router block receives router and residual-gain gradients through the composite criterion.

### Task 8: Re-run the mixture regression gate

- Run focused MoLoRA, V-PEFT, latent, mixture-loss, and routing protocol tests.
- Run Ruff on the newly added tests and narrowly touched implementation files, plus `git diff --check`.

### Task 9: Add controlled MoT sparse-training warmup

- Keep dense training as the backward-compatible default, but allow explicitly enabled sparse training to start after a configurable number of dense forwards.
- Persist warmup progress in `state_dict` so resume does not repeat exploration.
- Expose warmup phase, readiness, and policy through child and wrapper diagnostics.
- Route the setting through default config, CLI resolution, and YAML audit metadata.

### Task 10: Make Planner cold-start confidence auditable

- Add a stable evidence contract to every `PlacementDecision` and persisted `DecisionAudit`.
- Distinguish zero/insufficient observations (`cold_start`) from fitted-but-limited evidence and calibrated evidence.
- Count observations that have not yet reached the five-sample fit threshold without claiming learned-regression support.
- Record decision basis (`prior_prediction`, `learned_prediction`, guardrail, constraint, or runtime fallback) and named guardrails.
- Preserve backward-compatible prior-based ACCEPT behavior while marking it explicitly low confidence.

### Task 11: Define the MoT multi-GPU sparse-training contract

- Require an explicit DDP handshake confirming `find_unused_parameters=True` before sparse training may skip local experts.
- Keep the user's `sparse_train` request intact, but use `dense_ddp_fallback` when the DDP contract is absent or unsafe.
- Pass the exact Trainer DDP setting into routed modules before wrapping the model.
- Expose DDP activity, handshake source, safety result, and fallback reason in dispatch telemetry and export capabilities.
- Cover unconfigured, unsafe, and safe distributed states with monkeypatched regression tests.

### Task 12: Reduce gated MoE structural risk without checkpoint churn

- Add public import, pickle class-path, and `state_dict` key compatibility tests before moving implementation code.
- Extract only stateless visual routing helpers to an internal module; keep all public classes defined in `gated.py`.
- Preserve compatibility re-exports through `modules.py` and `hybrid.py`.
- Verify MoE SSOT, dynamic scheduling, AMP/index-add, and LoRA/DDP control paths after extraction.

### Task 13: Final regression and maturity gate

- Run Planner, MoT, MoE modularization, V-PEFT/MoLoRA, latent-mixture, configuration, and mixture-loss focused suites.
- Run Ruff lint/format checks on touched files and `git diff --check`.
- Run the broad mixture regression suite, excluding only tests blocked by the known local Matplotlib/NumPy ABI mismatch.
- Treat environment failures separately from product-code regressions and document any unavailable tools.

### Task 14: Make SoftRankAllocator cold starts deterministic and auditable

- Replace per-allocation random dummy embeddings with stable structural graph features.
- Pad or truncate encoder features explicitly when their width differs from the allocator input width.
- Record the feature source, cold-start state, dimensional adjustment, budget, and variant in allocation metadata.
- Verify repeated cold-start allocations are independent of global RNG activity.

### Task 15: Add an output-fidelity gate to MoLoRA merge

- Keep merge verification opt-in for backward compatibility.
- Reuse the exact calibration subset to capture dynamic and merged nested tensor outputs.
- Record normalized L2, mean absolute, and maximum absolute output error in the merge result and layer metadata.
- Atomically unmerge every layer changed by the operation when verification fails or any later merge step raises.
- Forward fidelity options through the unified adapter backend for wrapped and unwrapped MoLoRA models.

### Task 16: Restore top-level import resilience

- Move semantic-training Matplotlib imports into the plotting method.
- Convert missing or ABI-incompatible Matplotlib failures into a clear error only when a plot is requested.
- Verify top-level YOLO and semantic trainer imports while Matplotlib imports are intentionally blocked.

### Task 17: Verify the extended maturity gate

- Run V-PEFT allocator, MoLoRA merge/backend, semantic import, and engine collection regressions.
- Run Ruff lint/format checks and `git diff --check` on the new batch.
- Retry the broad test collection to confirm the former Matplotlib/NumPy collection blocker is removed.

## Execution Record (2026-07-24)

- Tasks 1-13 were completed in the preceding implementation batches; their focused mixture, planner, MoT, MoE, PEFT, latent, DDP, and configuration gates passed.
- Task 14 passed deterministic cold-start, feature-width adaptation, and allocation metadata regressions.
- Task 15 passed fidelity-gated merge, atomic rollback, metadata reset, and unified backend forwarding regressions.
- Task 16 passed blocked-Matplotlib top-level import and semantic trainer import regressions.
- Task 17 verification: the broad suite collected 1886 tests and completed with 1286 passed, 36 skipped, and 28 CLI failures caused by a stale editable install pointing at `YOLO-Master-v260130`; after reinstalling this repository with `pip install -e .`, `pytest tests/test_cli.py -q` completed with 44 passed.
- Agent Skill quick validation passed 34/34 cases; focused Ruff lint and format checks passed; `git diff --check` passed. Repository-wide Ruff still reports pre-existing issues outside this batch.
