# Phase 1 (P1) Delivery Report — Protocol Unification, Runtime Supervision, Batch Inference

**Branch:** `rhino-f1-dev` · **Date:** 2026-09-03

## 1. Deliverables

### 1.1 `ValHandler` — new validation task handler

- **File:** `smoke/f1/handlers/val.py` (new) — registered as `"val"` via `@TaskHandlerRegistry.register("val")`, imported in `smoke/f1/handlers/__init__.py`.
- `validate_params` enforces the shared security contract (no shell, path whitelisting, non-empty `allowed_paths`) plus `model_path` (checkpoint) / `data_source` (data.yaml) containment, `imgsz` / `batch_size` > 0, and `conf ∈ (0, 1]`.
- `execute` wraps `ultralytics.YOLO.val()` with `plots=True`, isolates output under `output_dir/job_id`, collects artifacts deterministically (sorted absolute paths), and parses metrics (`mAP50`, `mAP75`, `mAP50-95`, per-class precision/recall, `speed_ms`) defensively via `_parse_val_metrics` with a `results_dict` fallback. Engine errors of any type become structured `success=False` results; `CooperativeCancellationError` is re-raised.
- `TaskType.VAL = "val"` added to the enum in `smoke/f1/test_f1_smoke.py`.

### 1.2 Timeout & cooperative cancellation — `smoke/f1/dispatcher.py`

- **Deadline supervision:** handler execution runs in a daemon worker thread joined against `deadline = now + runtime_tracking.timeout_seconds`. A job still running past the deadline transitions to `FAILED + TIMEOUT`; the worker is detached and its late result is discarded.
- **Cooperative cancellation:** `cancel_requested` is checked at dispatcher checkpoints (pre-execution short-circuit, post-validation, in-flight polling every 0.05 s, post-execution before the COMPLETED transition) and at handler checkpoints via the new `BaseTaskHandler._check_cancelled()` — the dispatcher injects the job's `RuntimeTracking` into `handler._runtime_tracking` before execution; handlers call `_check_cancelled()` between work items (e.g., between batch chunks) and it raises `CooperativeCancellationError` (`smoke/f1/handlers/base.py`), which the dispatcher maps to `FAILED + USER_CANCELLED`.
- The exception subclasses `Exception` directly so it is never swallowed by the engine-facing `except (OSError, ImportError, RuntimeError, ValueError)` guards in handler `execute()` methods.

### 1.3 Batch inference — `PredictHandler`

- `data_source` now accepts a single path, a directory, or a sliced list of paths; every entry is whitelist-validated. `batch_size` (int > 0, default 8) validated.
- `execute` normalizes inputs deterministically (`_normalize_sources`: directories expand to sorted media files, `SUPPORTED_MEDIA_EXTENSIONS`), splits into chunks of `batch_size`, runs one engine call per chunk saved under `output_dir/job_id/batch_NNN`, with a cooperative cancellation checkpoint between chunks, and returns a deterministically sorted artifact list plus batch metadata (`num_inputs`, `num_batches`, `batch_size`, `num_results`, `sources`).

### 1.4 Test suite — `smoke/f1/test_val_batch_runtime.py` (40 tests)

- **ValHandler:** registration/`TaskType.VAL`, full validation battery (valid params, missing model_path/data_source, path containment rejection, shell/whitelist/empty-paths rejection, imgsz/batch_size/conf bounds), mocked execution with artifact verification and metric parsing, job isolation, dispatcher E2E through the supervised worker path.
- **Timeout:** `TIMEOUT` code + terminal `FAILED` state (no illegal re-transition), zero-timeout immediate failure, prompt return (supervision returns at ~1 s, not the handler's sleep).
- **Cooperative cancellation:** `_check_cancelled()` no-op without token, raises with token, `CooperativeCancellationError` → `USER_CANCELLED` mapping, dispatcher in-flight polling of a non-cooperative handler, cooperative loop handler aborting mid-loop, PredictHandler checkpoint raise.
- **Batch prediction:** list/directory acceptance, empty list / non-string entry / out-of-whitelist entry / invalid batch_size rejection, chunk sizes `[2, 2, 1]` verified against the engine call log, deterministic sorted artifact list matching expected names, directory expansion ignoring non-media noise, controlled failure for media-less directories, and one real multi-image CPU inference E2E.

## 2. Verification Evidence

```bash
# Full business suite (explicit P0 file list + new P1 file)
NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost" python -m pytest \
  smoke/f1/test_dispatcher.py smoke/f1/test_handlers_framework.py \
  smoke/f1/test_predict_diagnose.py smoke/f1/test_phase1_handlers.py \
  smoke/f1/test_skills.py smoke/f1/ui/test_jobs_tab.py smoke/f1/ui/test_app_integration.py \
  smoke/f1/test_val_batch_runtime.py -q
# => 183 passed, 1 warning in 21.76s   (143 existing P0 cases + 40 new P1 cases)

# Static checks (changed files only)
ruff check  smoke/f1/dispatcher.py smoke/f1/handlers/base.py smoke/f1/handlers/val.py \
            smoke/f1/handlers/predict.py smoke/f1/handlers/__init__.py \
            smoke/f1/test_f1_smoke.py smoke/f1/test_phase1_handlers.py \
            smoke/f1/test_val_batch_runtime.py        # => All checks passed
ruff format --check <same files>                       # => 8 files already formatted
```

## 3. Notes & Caveats

1. **Verification scope** follows the P0 explicit file-list convention. Directory-level collection (`pytest smoke/f1`) also gathers legacy module doctests that were never part of the verified suite; the explicit list remains the canonical command.
2. **Environmental (not a code regression):** `test_build_app_launches_headless` fails with a 502 on Gradio's localhost startup health check when the OS-level proxy is followed (httpx `trust_env`); it passes with `NO_PROXY=127.0.0.1,localhost`. App launch imports only `smoke.f1.ui.jobs_tab` — no dispatcher/handler code — so P1 changes cannot affect it.
3. **Existing assertion updated:** `test_phase1_handlers.py::test_all_four_handlers_are_registered` → `test_all_five_handlers_are_registered` now expects `["diagnose", "export", "predict", "train", "val"]` — the only existing test touched, necessitated by the new registry entry.
4. **Timing hygiene:** the only >1 s sleep in the new suite is `SlowHandler`'s 2 s sleep inside the dispatcher's daemon worker (the 1 s `TIMEOUT` deadline under test fires first; `RuntimeTracking.timeout_seconds` is an `int` field, so the deadline granularity is whole seconds). All flip delays and poll intervals are sub-second; no test performs an unbounded thread join.
