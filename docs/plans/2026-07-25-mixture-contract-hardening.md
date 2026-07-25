# Mixture Contract Hardening Plan

**Date:** 2026-07-25

## Goal

Close the remaining high-confidence maturity gaps from the final Mixture-of-Everything audit without changing the default routing or adapter algorithms.

## Scope

1. Run MoA, MoT, and selected gated-MoE router projections and softmax operations in FP32 while returning routing weights in the activation dtype and retaining FP32 logits for auxiliary losses.
2. Synchronize Latent Mixture importance statistics during gradient-enabled DDP training, preserving the global forward value and local autograd Jacobian.
3. Preserve V-PEFT per-target ranks through standard LoRA injection with a `rank_pattern`, including the in-repo fallback backend, and prevent the legacy Planner from overwriting an accepted V-PEFT plan.
4. Include `top_k` and `router_type` in the per-layer MoLoRA checkpoint structure contract.

## Verification

- Add focused numerical, DDP, V-PEFT, and checkpoint tamper regression tests.
- Run the focused test files for every touched subsystem.
- Run the broad mixture/PEFT regression gate, Ruff checks, and `git diff --check`.

## Non-Goals

- Do not enable Expert Choice, DoRA, auxiliary-loss-free routing, or other unvalidated research variants by default.
- Do not change model YAML defaults or training recipes in this batch.
