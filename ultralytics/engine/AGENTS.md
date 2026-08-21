# AGENTS.md — `ultralytics/engine/`

Training, validation, prediction, export, and tuning engine for YOLO-Master.

> Cross-project commands, code style, and environment notes: see [root AGENTS.md](../../AGENTS.md).

## Responsibilities

- **`trainer.py`** — Training loop, optimizer setup, mixed-precision, multi-GPU, MoE-aware training
- **`validator.py`** — Validation/inference evaluation pipeline
- **`predictor.py`** — Prediction pipeline with post-processing (NMS, tracking)
- **`exporter.py`** — Model export to ONNX, TensorRT, CoreML, TFLite, OpenVINO, etc.
- **`model.py`** — High-level Model API (entry point for `yolo train/val/predict/export`)
- **`results.py`** — Result data structures (boxes, masks, keypoints, plots)
- **`tuner.py`** — Hyperparameter tuning (Ray Tune integration)
- **`telemetry.py`** — Training telemetry and metrics collection
- **`extensions/`** — Mixture training extensions, adapter integration, fault recovery

## High-Risk Files

Changes to these files require extra caution and targeted verification:

| File | Risk | Verification |
|------|------|--------------|
| `trainer.py` | Core training loop — affects all training workflows | `pytest tests/test_engine.py -v` |
| `validator.py` | Validation pipeline — affects metric computation | `pytest tests/test_engine.py -v` |
| `predictor.py` | Prediction pipeline — affects all inference | `pytest tests/test_engine.py -v` |
| `exporter.py` | Export pipeline — affects all deployment formats | `pytest tests/test_exports.py --export-env base -v` |
| `model.py` | Top-level API — entry point for all user operations | `pytest tests/test_engine.py -v` |
| `extensions/mixture.py` | Mixture training extensions — MoE/MoA/MoT training integration | `pytest tests/test_engine.py -v` + `test_master_model_configs.py` |
| `extensions/recovery.py` | Fault recovery — affects training resilience | `pytest tests/test_engine.py -v` |

## Verification

```bash
# Engine tests
pytest tests/test_engine.py -v

# Export tests
pytest tests/test_exports.py --export-env base -v

# Doctest
pytest --doctest-modules ultralytics/engine/

# Combined with config integrity (after engine + config changes)
pytest tests/test_engine.py tests/test_default_config_integrity.py -v
```

## Task Routing

| If you are changing… | Start here | Then verify |
|----------------------|------------|-------------|
| Training loop or optimizer | `trainer.py` | `test_engine.py` |
| Validation metrics | `validator.py` | `test_engine.py` |
| Prediction / post-processing | `predictor.py`, `results.py` | `test_engine.py` |
| Export formats | `exporter.py` | `test_exports.py` |
| Model API | `model.py` | `test_engine.py` |
| Mixture training extensions | `extensions/mixture.py` | `test_engine.py` + `test_master_model_configs.py` |
| Fault recovery | `extensions/recovery.py` | `test_engine.py` |
| Hyperparameter tuning | `tuner.py` | `test_engine.py` |
