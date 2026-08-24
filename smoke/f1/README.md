# F1 Entry Smoke Test: YOLO-Master Studio Platform Core

**Topic**: F1 - YOLO-Master Studio Platform Core
**Milestone**: Entry Check - 2026-08-24
**Status**: ✅ **ALL SMOKE SCENARIOS PASSED**

---

## 1. Baseline & System Matrix

### 1.1 Git Baseline Anchor

- **Commit SHA**: `acce839c7e895d6b179de7f7093fa879e237cc7b`
- **Commit Date**: 2026-08-21 23:59:59 UTC+8
- **Working Branch**: `rhino-f1-dev`
- **Remote**: `origin/main` (upstream tracking)

### 1.2 Runtime Environment

| Component                  | Version / Specification |
| -------------------------- | ----------------------- |
| **Operating System** | Windows 11 (x64)        |
| **Python**           | 3.12.10                 |
| **PyTorch**          | 2.5.1+cu121             |
| **Torchvision**      | 0.20.1+cu121            |
| **Ultralytics**      | 8.4.101                 |
| **CUDA Runtime**     | 12.1                    |

### 1.3 Hardware Acceleration

- **GPU**: 1x NVIDIA GeForce RTX 4050 Laptop GPU
- **VRAM**: 6GB
- **CUDA Capability**: Verified (device=0 operational)
- **Driver**: NVIDIA display driver compatible with CUDA 12.1

---

## 2. Interface Contract & Architecture

### 2.1 Core Architecture Principle

> **Studio does NOT rewrite training/inference engines; it bridges the product orchestration layer.**

The F1 Studio Platform follows a strict separation of concerns:

- **Gradio WebUI**: User-facing interface for job submission and monitoring
- **Job Dispatcher**: Unified task orchestration with state machine control
- **Ultralytics Engine**: Underlying YOLO training/inference execution
- **Artifact Management**: Persistent storage for results, logs, and intermediate outputs

### 2.2 JobRequest Contract Schema

The shared `JobRequest` contract serves as the inter-component interface:

```json
{
  "job_id": "string",                    // Unique job identifier
  "task_type": "predict|train|export|diagnose",
  "status": "pending|running|completed|failed",
  "metadata": {
    "created_at": "ISO8601 timestamp",
    "created_by": "string",
    "description": "string",
    "priority": "normal|high|low",
    "tags": ["array of strings"]
  },
  "params": {
    "model_path": "string",              // Model checkpoint path
    "data_source": "string",             // Input data path
    "conf_threshold": "float",           // Confidence threshold
    "device": "string"                   // CUDA device index or 'cpu'
  },
  "output": {
    "output_dir": "string",              // Results directory
    "save_images": "boolean",
    "save_labels": "boolean",
    "save_logs": "boolean",
    "artifacts": ["array of paths"]      // Generated output files
  },
  "security_constraints": {
    "path_whitelisted": "boolean",       // Path whitelist enforcement
    "allow_shell": "boolean",            // Shell execution permission
    "allowed_paths": ["array of paths"]  // Whitelisted directory roots
  },
  "runtime_tracking": {
    "stream_logs": "boolean",            // Real-time log streaming
    "timeout_seconds": "integer",        // Execution timeout
    "cancellable": "boolean",            // Cancel support flag
    "cancel_requested": "boolean"        // User cancel signal
  },
  "error": {
    "code": "string",                    // Error code (e.g., SEC_ERR_001)
    "message": "string",                 // Human-readable error message
    "timestamp": "ISO8601 timestamp"
  } | null
}
```

### 2.3 Task State Machine

The dispatcher enforces strict state transitions to prevent race conditions and illegal operations:

```
PENDING ──┬──> RUNNING ──┬──> COMPLETED (terminal)
          │              │
          │              └──> FAILED (terminal)
          │
          └──> FAILED (terminal)
```

**Transition Rules**:

- `PENDING → RUNNING`: Job execution started
- `PENDING → FAILED`: Pre-execution validation failure (e.g., security violation)
- `RUNNING → COMPLETED`: Successful task completion with artifacts
- `RUNNING → FAILED`: Runtime execution error
- `COMPLETED` and `FAILED` are **terminal states** with no outgoing transitions

**Illegal Transitions** (enforced by `JobDispatcherStateMachine`):

- `COMPLETED → RUNNING` (Cannot restart finished job)
- `COMPLETED → PENDING` (Cannot reset finished job)
- `FAILED → RUNNING` (Cannot resume failed job without resubmission)

---

## 3. Smoke Test Verification Results

### 3.1 Test Suite Overview

The F1 entry smoke test validates five critical dimensions:

| Case             | Verification Target                                 | Expected Outcome                              |
| ---------------- | --------------------------------------------------- | --------------------------------------------- |
| **Case 1** | E2E GPU inference with artifact persistence         | `status=COMPLETED`, `len(artifacts) > 0`  |
| **Case 2** | Security policy defense (shell execution)           | `status=FAILED`, `error.code=SEC_ERR_001` |
| **Case 3** | Security policy defense (path traversal)            | `status=FAILED`, `error.code=SEC_ERR_001` |
| **Case 4** | Illegal state transition guard (COMPLETED→RUNNING) | `ValueError` raised                         |
| **Case 5** | Illegal state transition guard (PENDING→COMPLETED) | `ValueError` raised                         |

### 3.2 Execution Evidence (from `smoke_run.log`)

#### Case 1: E2E Happy Path Execution

**Objective**: Validate contract parsing, GPU inference, and artifact capture.

```log
[Case 1] Contract Parsing & E2E Real Inference Execution...
  [StateMachine] Job job_20260824_f1_001 transitioned to: RUNNING
Results saved to D:\Projects\YOLO-Master\runs\detect\runs\predict\job_20260824_f1_001
  [Executor] Real inference executed on device=0. Detected count: 6
  [Artifacts] Verified artifacts (1 files) at: ['D:\\Projects\\YOLO-Master\\runs\\detect\\runs\\predict\\job_20260824_f1_001\\bus.jpg']
  [StateMachine] Job job_20260824_f1_001 transitioned to: COMPLETED
  [PASS] in 3.59s
```

**Evidence Analysis**:

- ✅ Contract parsed successfully from `job_request_draft.json`
- ✅ State transition `PENDING → RUNNING → COMPLETED` executed correctly
- ✅ Real GPU inference on `device=0` (RTX 4050) detected 6 objects in `bus.jpg`
- ✅ Output artifact `bus.jpg` persisted to `runs/detect/runs/predict/job_20260824_f1_001/`
- ✅ Total execution time: **3.59 seconds**

#### Case 2: Security Constraint Defense (Shell Execution)

**Objective**: Verify rejection of jobs attempting shell execution.

```log
[Case 2] Security Constraint Defense (Shell Attempt)...
  [StateMachine] Job smoke-sec-002 transitioned to: FAILED
  [PASS] (Shell execution blocked)
```

**Evidence Analysis**:

- ✅ Job with `allow_shell=true` immediately transitioned to `FAILED`
- ✅ Error code `SEC_ERR_001` correctly assigned
- ✅ No shell execution attempted; policy enforced at dispatch stage
- ✅ Error message: `"Security policy violation: Shell execution not permitted"`

#### Case 3: Security Constraint Defense (Path Traversal)

**Objective**: Verify rejection of jobs accessing non-whitelisted paths.

```log
[Case 3] Security Constraint Defense (Path Traversal)...
  [StateMachine] Job smoke-sec-003 transitioned to: FAILED
  [PASS] (Path traversal blocked)
```

**Evidence Analysis**:

- ✅ Job with `data_source="../../etc/passwd"` immediately failed validation
- ✅ Error code `SEC_ERR_001` correctly assigned
- ✅ Path whitelist enforcement prevented directory traversal attack
- ✅ Error message: `"Security policy violation: Data_source path '../../etc/passwd' not in whitelist"`

#### Case 4: State Machine Illegal Transition Guard (COMPLETED→RUNNING)

**Objective**: Ensure terminal states cannot transition back to active states.

```log
[Case 4] State Machine Guard (COMPLETED -> RUNNING)...
  [PASS] (Illegal transition prevented)
```

**Evidence Analysis**:

- ✅ Attempted transition `COMPLETED → RUNNING` raised `ValueError`
- ✅ State machine integrity preserved
- ✅ Terminal state immutability enforced

#### Case 5: State Machine Illegal Transition Guard (PENDING→COMPLETED)

**Objective**: Ensure jobs cannot skip mandatory execution stage.

```log
[Case 5] State Machine Guard (PENDING -> COMPLETED)...
  [PASS] (Skip prevented)
```

**Evidence Analysis**:

- ✅ Attempted transition `PENDING → COMPLETED` raised `ValueError`
- ✅ State machine prevents skipping `RUNNING` stage
- ✅ Execution integrity guaranteed

### 3.3 Final Test Suite Summary

```
=======================================================
Smoke Test Execution Summary:
  - E2E Happy Path (Real Predict)       [PASS] (Time: 3.59s, Artifacts: 1)
  - Security Defense (Shell)            [PASS] (ErrorCode: SEC_ERR_001)
  - Security Defense (Path Traversal)   [PASS] (ErrorCode: SEC_ERR_001)
  - State Guard (COMPLETED->RUNNING)    [PASS] (Prevented illegal transition)
  - State Guard (PENDING->COMPLETED)    [PASS] (Prevented skip to terminal)
=======================================================
>>> ALL F1 SMOKE SCENARIOS PASSED SUCCESSFULLY! <<<
```

**Exit Code**: `0` (all tests passed)

---

## 4. Security & Safety Declaration

### 4.1 Enforced Security Policies

The F1 Studio Platform implements the following mandatory security constraints:

1. **No Arbitrary Shell Execution**:

   - `allow_shell=false` enforced at dispatch layer
   - Jobs requesting shell access immediately fail with `SEC_ERR_001`
2. **Path Whitelisting**:

   - All `params` paths (`model_path`, `data_source`) and `output.output_dir` are resolved
     with `Path.resolve()` and checked for containment within an `allowed_paths` root
   - Contract whitelist: `[".", "ultralytics/assets", "runs/predict", "ckpts"]`
   - Enforcement is *positive*: an empty `allowed_paths` rejects every path rather than
     defaulting to permissive, so a malformed contract fails closed
   - Path traversal (`../`) is neutralized by resolution before the containment check
3. **Environment Variable Sanitization**:

   - Sensitive variables (API keys, credentials) excluded from logs
   - Runtime tracking logs redact `os.environ` contents
4. **Resource Isolation**:

   - Jobs execute with `timeout_seconds` limit (default: 300s)
   - GPU memory leaks mitigated via PyTorch context managers
   - Output directories use unique `job_id` to prevent collision

### 4.2 Threat Model Coverage

| Attack Vector       | Mitigation                                      | Validation                  |
| ------------------- | ----------------------------------------------- | --------------------------- |
| Shell execution     | `allow_shell=false` check                     | Case 2 smoke test           |
| Path traversal      | `_is_path_safe()` resolve + containment check | Case 3 smoke test           |
| Resource exhaustion | `timeout_seconds` contract field              | Declared; enforcement is P1 |
| State corruption    | State machine transition guards                 | Case 4 / Case 5 smoke tests |

---

## 5. Post-Entry Roadmap

### 5.1 Phase 0 (P0): Foundation Stabilization

**Target**: 2026-08-31
**Deliverables**:
- [X] Baseline entry validation & locked contract schema
- [ ] Preserve existing baseline inference functionality
- [ ] Add Jobs tab (submission, state monitoring, streaming logs, artifacts download)
- [ ] Integrate `predict` and `system doctor` Agent Skills
- [ ] Gradio UI layout optimization and state binding

### 5.2 Phase 1 (P1): Multi-Task Unification

**Target**: 2026-09-07
**Deliverables**:
- [ ] Unify `train`, `val`, `predict`, `export` task contracts
- [ ] Implement job cancellation, watchdog timers, and timeout mechanisms
- [ ] Add batch image/video inference support
- [ ] Enhance path whitelisting with regex patterns

### 5.3 Phase 2 (P2): Production Hardening

**Target**: 2026-09-12
**Deliverables**:
- [ ] Provide standalone FastAPI backend endpoints alongside Gradio WebUI
- [ ] Add real-time hardware telemetry (GPU/CPU utilization via NVML)
- [ ] Basic authentication and access control
- [ ] Comprehensive diagnostic and logging infrastructure

---

## 6. Reproduction Guide

### 6.1 Prerequisites

```bash
# Verify Python environment (Recommended 3.10 - 3.12, verified on 3.12.10)
python --version

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"   # Should print True

# Verify Ultralytics installation
python -c "from ultralytics import YOLO; print(YOLO.__version__)"
```

### 6.2 Execute Smoke Test

```bash
# Navigate to project root directory
cd path/to/YOLO-Master

# Execute multi-scenario smoke test suite
python smoke/f1/test_f1_smoke.py

# Optional: Capture and update run log
# Linux / macOS / Git Bash:
# python smoke/f1/test_f1_smoke.py > smoke/f1/smoke_run.log 2>&1
# Windows PowerShell:
# python smoke\f1\test_f1_smoke.py | Out-File -Encoding utf8 smoke\f1\smoke_run.log
```

### 6.3 Inspect Results

```bash
# View execution log
cat smoke/f1/smoke_run.log

# Verify artifacts
ls -lh runs/detect/runs/predict/job_20260824_f1_001/

# Validate contract schema
python -c "
import json
from smoke.f1.test_f1_smoke import JobRequest
with open('smoke/f1/job_request_draft.json') as f:
    JobRequest(**json.load(f))
print('Contract validation passed')
"
```

---

## 7. Known Limitations & Future Work

### 7.1 Current Constraints
- **Single-Task Execution**: No concurrent job execution
- **Limited Task Types**: Only `predict` implemented in smoke test
- **Local-Only Storage**: Artifacts stored on local filesystem
- **No Retry Mechanism**: Failed jobs require manual resubmission

### 7.2 Planned Enhancements
- Distributed task queue (Celery/Redis backend)
- Multi-GPU job scheduling and load balancing
- WebSocket-based real-time log streaming
- Cloud object storage support (COS/OSS/S3-compatible)

---

## 8. References

- **Gradio Baseline Entry**: `app.py`
- **Agent Skills Specification**: `agent/SKILL.md`
- **Async Runtime & Handlers**: `agent/runtime/cli/async_jobs.py`, `agent/runtime/cli/job_handlers.py`
- **Ultralytics YOLO Documentation**: https://docs.ultralytics.com/
---

**Document Version**: 1.0.0
**Last Updated**: 2026-08-24
**Maintained By**: [@9-71](https://github.com/9-71) (`rhino-f1-dev` branch)
