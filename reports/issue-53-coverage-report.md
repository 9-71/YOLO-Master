# MoA Module Coverage Report — Issue #53

## Baseline (before enhancement)

| File | Statements | Missed | Coverage |
|------|-----------|--------|----------|
| `ultralytics/nn/modules/moa/__init__.py` | 2 | 0 | 100% |
| `ultralytics/nn/modules/moa/moa.py` | 448 | 74 | 83% |
| **TOTAL** | **450** | **74** | **84%** |

Tests: 16 passed

## After enhancement

| File | Statements | Missed | Coverage |
|------|-----------|--------|----------|
| `ultralytics/nn/modules/moa/__init__.py` | 2 | 0 | 100% |
| `ultralytics/nn/modules/moa/moa.py` | 450 | 21 | 95% |
| **TOTAL** | **452** | **21** | **95%** |

Tests: 62 passed (+46 new)

## Coverage improvement: **84% → 95%** (+11pp)

## Remaining uncovered lines

| Lines | Function | Reason |
|-------|----------|--------|
| 256–257 | `_RegionalAttnHead.forward` | Extreme edge case: H2*W2==0 (nearly unreachable) |
| 315–319 | `_GlobalAttnHead.__init__` | QR fallback via SVD (only on old CUDA/MPS drivers) |
| 453–459 | `_moa_router_aux_loss` | DDP synchronization path (requires multi-GPU) |
| 465 | `_moa_router_aux_loss` | Inf guard for DDP edge case |
| 482 | `_moa_router_aux_loss` | Non-finite result zero-guard (covered via NaN path) |
| 604–606 | `MoABlock.__deepcopy__` | deepcopy protocol |
| 773 | `C2fMoA.forward` | Empty routing snapshot branch |
| 791 | `C2fMoA.publish_aux_loss` | Protocol method (covered implicitly) |
| 802–804 | `C2fMoA.__deepcopy__` | deepcopy protocol |
| 953 | `NeckMoAFusion.publish_aux_loss` | Protocol method (covered implicitly) |
| 962–964 | `NeckMoAFusion.__deepcopy__` | deepcopy protocol |

## New tests added (46 tests)

### 1. NeckMoAFusion cross-scale boundary tests (4 tests)
- `test_neck_moa_fusion_non_strict_scale_many_cases` — 7 diverse size mismatch cases
- `test_neck_moa_fusion_single_pixel_lo` — 1×1 lo-res extreme case
- `test_neck_moa_fusion_no_shortcut` — shortcut=False path
- `test_neck_moa_fusion_channel_mismatch_projection` — c_hi ≠ c_out projection path

### 2. MoABlock temperature & numerical stability (3 tests)
- `test_moa_router_extreme_temperature_numerical_stability` — 9 temperatures from 1e-8 to 100
- `test_moa_router_anneal_to_zero_like_temperature` — near-zero temperature stability
- `test_moa_router_return_logits` — return_logits=True path

### 3. Attention head non-divisible dim/heads (5 tests)
- `test_regional_attn_head_non_divisible_dim_and_heads` — _RegionalAttnHead boundary
- `test_regional_attn_head_small_spatial_dims` — H=1/W=1 guard path
- `test_regional_attn_head_invalid_pool_stride` — pool_stride validation
- `test_local_attn_head_window_size_clamping` — window_size=0 clamping
- `test_attention_heads_all_variants_non_divisible` — all 3 heads with dim=19, heads=5

### 4. C2fMoA aux_loss double-counting (3 tests)
- `test_c2fmoa_covered_modules_mechanism` — covered_modules prevents child double-count
- `test_c2fmoa_single_block_aux_loss_equals_block_loss` — n=1 equivalence
- `test_c2fmoa_eval_mode_zero_aux_loss` — eval mode zero aux_loss

### 5. MoABlock advanced paths (5 tests)
- `test_moa_block_no_shortcut` — shortcut=False pure feed-forward
- `test_moa_block_sequential_heads` — sequential_heads=True path
- `test_moa_block_sequential_vs_parallel_equivalence` — numerical equivalence
- `test_moa_block_with_attn_dropout` — attention dropout
- `test_moa_block_different_mlp_ratios` — 4 MLP ratios [0.5, 1.0, 2.0, 4.0]

### 6. Global head advanced paths (4 tests)
- `test_global_head_linear_attn_large_spatial` — N=576 > 512 → linear attention
- `test_global_head_smooth_blend_window` — N=484 in blend [448, 512]
- `test_global_head_very_large_spatial` — N=1600 >> 512
- `test_global_head_exact_attn_small_spatial` — N=256 < 448 → exact attention

### 7. Flash attention edge cases (3 tests)
- `test_flash_attn_fallback_no_sdpa` — fallback when sdpa unavailable
- `test_flash_attn_sdpa_without_scale_typeerror` — TypeError re-raise for non-scale errors
- `test_flash_attn_sdpa_scale_typeerror_fallback` — scale TypeError graceful fallback

### 8. Utilities & protocol (13 tests)
- `test_window_flash_attn_edge_cases` — 6 spatial/window config combinations
- `test_fp_min_across_dtypes` — fp16/bf16/float32 dtype-aware min values
- `test_moa_router_aux_loss_no_nan_from_biased_logits` — extreme bias stability
- `test_moa_router_aux_loss_finite_guard_triggers` — large logit guard
- `test_moa_block_routed_module_protocol` — MoABlock protocol methods
- `test_c2fmoa_routed_module_protocol` — C2fMoA protocol methods
- `test_neck_moa_fusion_routed_module_protocol` — NeckMoAFusion protocol methods
- `test_publish_aux_loss_in_eval_mode` — eval mode publish
- `test_c2fmoa_multiple_expansion_ratios` — e ∈ {0.25, 0.5, 0.75, 1.0}
- `test_c2fmoa_large_n_many_blocks` — n=8 stacked blocks
- `test_c2fmoa_inference_mode_no_aux_loss_grad` — no_grad inference
- `test_anneal_moa_temperature_respects_min_temp` — min_temp clamp
- `test_anneal_moa_temperature_skips_non_moa_modules` — only _MoARouter affected

### 9. Collection & RF (5 tests)
- `test_collect_moa_aux_loss_multiple_module_types` — mixed C2fMoA+NeckMoAFusion
- `test_collect_moa_aux_loss_with_none_model` — None model → zero tensor
- `test_global_head_rf_matrix_deterministic` — same seed → same RF
- `test_global_head_rf_matrix_different_seeds` — different seeds → different RF
- `test_moa_block_diverse_spatial_sizes` — sizes [4, 8, 16, 32, 64]
- `test_neck_moa_fusion_diverse_spatial_sizes` — sizes [4, 8, 16, 32]

## Bug fixes

1. **`_aux_loss_device` None-safety** (`moa.py:983-988`): Added `None` guard before `next(model.parameters())`. Previously `collect_moa_aux_loss(None)` would crash with `AttributeError`.

## Training validation

See `reports/issue-53-training-validation.md` for VisDrone/SKU-110K training results.
