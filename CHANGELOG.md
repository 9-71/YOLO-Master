# Changelog

All notable changes to YOLO-Master will be documented in this file.

---

## [YOLO-Master-v26.08] — 2026-08-07

<div align="center">
  <img width="320" height="320" alt="YOLO-Master Logo" src="https://github.com/user-attachments/assets/847ce41b-7282-4e98-b8be-240a572dd87a" />

# 🎯 YOLO-Master v2026.08 Release Notes

[![License](https://img.shields.io/badge/License-Tencent-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![GitHub stars](https://img.shields.io/github/stars/Tencent/YOLO-Master)](https://github.com/Tencent/YOLO-Master)
[![GitHub forks](https://img.shields.io/github/forks/Tencent/YOLO-Master)](https://github.com/Tencent/YOLO-Master/network/members)
[![Technical Report](https://img.shields.io/badge/📄%20Technical%20Report-In%20Progress-orange?style=flat-square&logo=arxiv)](https://github.com/Tencent/YOLO-Master#technical-report)
</div>

---

## 🌟 Overview

We are thrilled to announce **YOLO-Master v2026.08**, a transformative release that expands the Mixture-of-Experts paradigm into a comprehensive **full-stack AI training & deployment ecosystem**. After 7.5 months of intensive development (620 commits, 96 merged PRs, 30+ contributors), this release delivers breakthroughs across 6 major dimensions.

### 🎯 Key Highlights

- **🏗️ MultiTask Learning**: Train a single model for detection, segmentation, pose estimation, and classification simultaneously — with dynamic TaskRouter dispatching
- **🧠 Shared Expert MoE**: Cross-scale expert pool sharing (方案 D) — experts are shared across detection scales, reducing parameters by 40% while improving mAP
- **⚡ MoA/MoT Optimization**: Mixture-of-Attention with regional window processing, Mixture-of-Transformers with GShard balancing — 2.5x sparse inference speedup
- **🎯 V-PEFT Planner System**: PPO-driven automatic rank allocation, LOVO (Leave-One-Variant-Out) cross-validation, FewShot-LoRA with scheduled DropConnect
- **🌡️ Latent Mixture**: Router initialization perturbation with temperature annealing for more diverse and stable expert assignments
- **🚀 Universal Edge Deployment**: Native Windows GUI runner (Dear ImGui + D3D11), Jetson Orin TensorRT, MNN backend, macOS Core ML — one codebase, all platforms
- **🛡️ Production-Grade Robustness**: End-to-end NaN self-healing, DDP checkpoint lifecycle hardening, EMA synchronization, AMP-safe MoE dispatch

---

## 🚀 New Features

### 1️⃣ MultiTask Learning — One Model, All Tasks

MultiTask Learning enables a single YOLO-Master model to jointly learn detection, instance segmentation, pose estimation, and classification — with a learned **TaskRouter** dynamically dispatching features to task-specific heads.

#### 🧩 Architecture

```
                    ┌──────────────────┐
                    │   Shared Backbone │
                    │  (MoE-Enhanced)   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │    TaskRouter     │
                    │  (Learned Dispatch)│
                    └──┬──────┬──────┬─┘
                       │      │      │
              ┌────────▼─┐ ┌──▼───┐ ┌▼────────┐
              │ Detection │ │ Pose │ │ Segment │
              │   Head    │ │ Head │ │  Head   │
              └──────────┘ └──────┘ └─────────┘
```

#### 🔧 Core Components

**📊 MultiTaskHead**: Task-specific detection heads with shared feature hierarchy
**🔀 TaskRouter**: Learns optimal feature routing per task, minimizing interference
**📈 MultiTask Loss**: Dynamic loss balancing across tasks with uncertainty weighting
**📦 Data Pipeline**: Unified dataset format supporting multi-task annotations

#### 💻 Quick Start

```bash
# Multi-task training (detection + pose + segmentation)
yolo multitask train \
  model=yolo-master-n-multitask.yaml \
  data=coco-multitask.yaml \
  epochs=300 \
  imgsz=640

# Multi-task inference
yolo multitask predict \
  model=runs/multitask/train/weights/best.pt \
  source=image.jpg \
  tasks=detect,pose,segment
```

```python
from ultralytics import YOLO

model = YOLO("yolo-master-n-multitask.yaml")
results = model.train(
    data="coco-multitask.yaml",
    epochs=300,
    imgsz=640,
    task_weights={"detect": 1.0, "pose": 0.8, "segment": 0.6}
)

# Run all tasks
outputs = model.predict("image.jpg", tasks=["detect", "pose", "segment"])
```

*Implementation*: `ultralytics/nn/modules/multitask/`

---

### 2️⃣ Shared Expert MoE — Cross-Scale Expert Pool (方案 D)

A breakthrough in MoE efficiency: instead of each detection scale maintaining independent expert pools, **Shared Expert MoE** creates a unified expert pool shared across all FPN/PAN scales. This eliminates expert redundancy across scales while maintaining or improving accuracy.

#### 📊 Architecture Comparison

| Aspect | Standard MoE (v26.02) | Shared Expert MoE (v26.08) |
|--------|----------------------|---------------------------|
| Expert Organization | Per-scale independent pools | Cross-scale unified pool |
| Parameter Efficiency | 1x baseline | **~40% parameter reduction** |
| Expert Utilization | Scale-isolated dispatch | Cross-scale load balancing |
| mAP Impact | Baseline | **+0.3-0.5% mAP50-95** |
| Implementation | `moe/routers.py` | `moe/shared_expert.py` |

#### 🔧 Key Innovations

**Shared Expert Pool**: All FPN scales (P3-P7) route to the same expert bank, eliminating duplicate capacity
**Cross-Scale Load Balancing**: Gini coefficient optimization across scales ensures no expert starvation
**Meta-Router**: Lightweight meta-network selects optimal routing strategy per input complexity

```python
# Enable Shared Expert MoE
model = YOLO("yolo-master-shared-expert.yaml")

# Training with cross-scale routing
results = model.train(
    data="coco.yaml",
    epochs=300,
    moe_shared_expert=True,
    moe_num_experts=8,
    moe_top_k=2,
    moe_cross_scale_balance=True
)
```

*Implementation*: `ultralytics/nn/modules/moe/shared_expert.py`, `ultralytics/nn/modules/moe/routers.py`

---

### 3️⃣ MoA/MoT — Attention-Level Mixture Optimization

#### 3a. Mixture of Attention (MoA)

Regional attention mechanism that partitions feature maps into spatial regions, each processed by a specialized attention expert. Achieves **2-3x attention speedup** without accuracy loss.

| Component | Description |
|-----------|-------------|
| **Regional Partition** | Splits feature maps into spatial windows with learned boundaries |
| **Window Attention** | Each region processed by a dedicated attention expert |
| **SVD Fallback** | Automatic rank reduction for memory-constrained scenarios |
| **Sparse Inference** | Opt-in sparse attention paths for latency-critical deployment |

```python
# Enable MoA regional attention
results = model.train(
    data="coco.yaml",
    moa_regional=True,
    moa_num_regions=4,
    moa_window_size=7
)
```

#### 3b. Mixture of Transformers (MoT)

GShard-style transformer routing with dynamic token-to-expert assignment. Replaces static backbone blocks with learned routing decisions.

| Feature | Benefit |
|---------|---------|
| **GShard Balance Loss** | Prevents expert collapse in transformer layers |
| **Dynamic Token Dispatch** | Routes each spatial token to optimal expert |
| **Scene-Aware Residual** | MOT-specific routing residual for crowded scenes |
| **Trace-Stable Routing** | Deterministic window shifts for reproducible results |

```python
# Enable MoT transformer routing
results = model.train(
    data="coco.yaml",
    mot_enabled=True,
    mot_num_experts=4,
    mot_top_k=2,
    mot_balance_loss_weight=0.01
)
```

#### 📊 Performance Benchmarks

| Config | mAP50-95 | GFLOPs | Latency (ms) | Speedup vs Baseline |
|--------|----------|--------|-------------|---------------------|
| Baseline (no mixture) | 0.427 | 8.7 | 1.56 | 1.0x |
| + MoE only | 0.433 | 8.7 | 1.62 | 0.96x |
| + MoA Regional | 0.435 | 7.2 | 1.21 | 1.29x |
| + MoT GShard | 0.438 | 9.1 | 1.85 | 0.84x |
| **+ Full Mixture** | **0.441** | **8.3** | **1.48** | **1.05x** |

*Implementation*: `ultralytics/nn/modules/moa/`, `ultralytics/nn/modules/mot/`

---

### 4️⃣ V-PEFT Planner — Intelligent Parameter-Efficient Fine-Tuning

The V-PEFT (Variational PEFT) Planner is an end-to-end automatic rank allocation system that replaces manual LoRA hyperparameter tuning with a learned optimization process.

#### 🧠 PPO Rank Allocation

Reinforcement learning-based rank allocation that maximizes downstream task performance under a parameter budget constraint:

```
State: Layer characteristics (GFLOPs, parameter count, gradient norms)
Action: Assign LoRA rank r ∈ {2, 4, 8, 16, 32, 64} per layer
Reward: mAP improvement per parameter
Policy: PPO with entropy bonus for exploration
```

```python
# Automatic rank allocation via PPO Planner
model = YOLO("yolov8n.pt")
results = model.train(
    data="custom_dataset.yaml",
    epochs=100,
    vpeft_planner="ppo",      # PPO-based auto rank
    vpeft_budget=500000,      # Max trainable params
    vpeft_semantic_targets=["backbone", "neck", "head"]
)
```

#### 📊 LOVO Cross-Validation

Leave-One-Variant-Out (LOVO) systematically evaluates PEFT configurations, fitting a **Scaling Law regression model** to predict optimal ranks without exhaustive search.

| LOVO Stage | Description |
|------------|-------------|
| **Data Collection** | Train variants with systematic rank combinations |
| **Scaling Law Fit** | Regress mAP vs. rank, params, GFLOPs |
| **Optimal Prediction** | Predict best configuration for unseen budgets |
| **Validation** | Hold-out validation on reserved variant |

```bash
# Run LOVO cross-validation
yolo lora lovo \
  model=yolov8n.pt \
  data=coco8.yaml \
  ranks="2,4,8,16,32,64" \
  variants=all \
  output_dir=reports/lovo_results/
```

#### 🎯 FewShot-LoRA

Scheduled DropConnect + hierarchical distillation for extreme low-data scenarios (5-50 samples):

```python
# FewShot-LoRA with 20 samples
results = model.train(
    data="fewshot_20.yaml",
    epochs=200,
    lora_fewshot=True,
    lora_dropconnect=0.3,         # Scheduled DropConnect
    lora_distillation_alpha=0.5,   # Hierarchical distillation weight
    lora_variational_rank=True     # Variational rank allocation
)
```

| Scenario | Full SFT mAP | LoRA mAP | **FewShot-LoRA mAP** |
|----------|-------------|----------|----------------------|
| 5 samples | 0.12 | 0.08 | **0.22** |
| 20 samples | 0.31 | 0.28 | **0.39** |
| 50 samples | 0.48 | 0.46 | **0.52** |
| Full dataset | 0.65 | 0.63 | **0.64** |

*Implementation*: `ultralytics/nn/peft/vpeft/`, `ultralytics/nn/peft/molora/`

---

### 5️⃣ MoLoRA — Routing-Aware Mixture-of-LoRA

MoLoRA extends standard LoRA by making adapter selection **routing-aware** — the router jointly decides which expert AND which LoRA adapter to activate.

#### 🔧 Key Innovations

**Routing-Aware Merge**: Adapter weights calibrated against router decisions for consistent inference
**Batched Einsum Kernels**: Vectorized same-rank Linear experts processed in a single batched operation
**Unified Contracts**: MoLoRA routing and rank contracts unified with V-PEFT planner

```python
# MoLoRA training with routing-aware merge
model = YOLO("yolo-master-esmoe-n.pt")
results = model.train(
    data="coco.yaml",
    epochs=100,
    molora=True,
    molora_rank=8,
    molora_routing_aware=True,     # Router-aware adapter selection
    molora_batched_einsum=True     # Batched expert computation
)
```

#### 📊 Performance

| Method | Trainable Params | mAP50-95 | Speedup vs Full FT |
|--------|-----------------|----------|-------------------|
| Full Fine-Tune | 2.68M (100%) | 0.427 | 1.0x |
| Standard LoRA | 0.53M (19.8%) | 0.418 | 1.6x |
| **MoLoRA (Ours)** | **0.48M (17.9%)** | **0.424** | **2.1x** |

*Implementation*: `ultralytics/nn/peft/molora/`

---

### 6️⃣ Latent Mixture — Training-Time Router Optimization

Latent Mixture introduces initialization perturbation and temperature annealing to the MoE router training process, dramatically improving expert diversity and routing stability.

#### 🌡️ Init Perturbation + Temperature Annealing

```
Phase 1 (Epochs 1-50):  High temperature (τ=5.0), large perturbation (σ=0.1)
                        → Maximum exploration, diverse expert assignments

Phase 2 (Epochs 50-200): Medium temperature (τ=2.0), moderate perturbation (σ=0.05)
                        → Balanced exploration-exploitation

Phase 3 (Epochs 200+):   Low temperature (τ=1.0), minimal perturbation (σ=0.01)
                        → Converged routing, stable assignments
```

```python
# Enable latent mixture with annealing
results = model.train(
    data="coco.yaml",
    epochs=300,
    latent_init_perturb=True,
    latent_perturb_sigma=0.1,
    latent_temperature_anneal=True,
    latent_temperature_start=5.0,
    latent_temperature_end=1.0
)
```

#### 📊 Expert Diversity Impact

| Config | Expert Utilization Gini | Expert Collapse Rate | mAP50-95 |
|--------|------------------------|---------------------|----------|
| No Latent Mixture | 0.42 | 12.5% | 0.427 |
| + Init Perturbation | 0.28 | 3.2% | 0.431 |
| + Temperature Anneal | 0.21 | 0.8% | **0.435** |

*Implementation*: `ultralytics/nn/modules/moe/latent.py`

---

### 7️⃣ Universal Edge Deployment

One codebase, all platforms. v26.08 brings production-ready edge deployment with native GUI, multiple backends, and CPU thread optimization.

#### 🖥️ Windows GUI Runner (Dear ImGui + D3D11)

Native Windows 10/11 desktop application with real-time detection visualization:

- **Zero-dependency**: Single .exe, no Python required
- **GPU Acceleration**: Direct3D 11 rendering pipeline
- **Real-time**: 30+ FPS on Intel iGPU with EsMoE-N
- **Interactive**: Drag-and-drop image/video, live parameter tuning

```bash
# Build Windows GUI runner
cd examples/YOLO-Master-Cross-Platform-Edge-Deployment
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

#### 📦 Multi-Backend Inference

| Backend | Platform | Status | Best For |
|---------|----------|--------|----------|
| **TensorRT** | Jetson Orin, NVIDIA GPU | ✅ Production | Max throughput |
| **ONNX Runtime** | Cross-platform | ✅ Production | Flexibility |
| **NCNN** | ARM, x86 | ✅ Production | Mobile/Embedded |
| **MNN** | Mobile, Edge | ✅ New in v26.08 | ARM optimization |
| **Core ML** | macOS, iOS | ✅ New in v26.08 | Apple ecosystem |
| **OpenVINO** | Intel CPU/GPU | ✅ Production | Intel hardware |

#### ⚙️ CPU Thread Controls

Fair CPU thread allocation for edge devices, preventing thread contention in multi-model scenarios:

```python
# CPU thread control for edge inference
from ultralytics import YOLO
model = YOLO("yolo-master-esmoe-n.pt", task="detect")
model.predict(
    source="video.mp4",
    device="cpu",
    cpu_threads=4,          # Fair allocation
    cpu_affinity=[0,1,2,3]  # Core pinning
)
```

*Implementation*: `examples/YOLO-Master-Cross-Platform-Edge-Deployment/`

---

### 8️⃣ Production-Grade Robustness

#### 🛡️ NaN Safety System

End-to-end NaN detection and self-healing pipeline across all mixture components:

| Layer | Protection |
|-------|-----------|
| **Pre-batch Detection** | NaN check before forward pass |
| **Per-component Isolation** | Router, expert, loss individually guarded |
| **Self-healing EMA** | Fallback to last valid EMA on NaN |
| **AMP Safety** | Dtype-aligned index_add_ for all sparse MoE paths |

#### 🔄 DDP Checkpoint Hardening

| Fix | Impact |
|-----|--------|
| Bootstrap before first optimizer step | Prevents cold-start divergence |
| Serialize checkpoint before epoch loop | Guarantees recovery point |
| NCCL buffer registration guard | Prevents ES_MOE stats crash |
| Cross-module DDP | Stable multi-architecture training |
| EMA buffer schema divergence | Synchronized across ranks |

#### 📊 Device Support

| Platform | Status |
|----------|--------|
| **NVIDIA CUDA (DDP)** | ✅ Production |
| **Apple MPS** | ✅ Stabilized (grid_sample fix) |
| **CPU** | ✅ Production (with thread controls) |
| **AMD ROCm** | ✅ Community verified |

---

## 📊 Model Zoo & Benchmarks

### 🏆 Official Models

### YOLO-Master-EsMoE Series

| Model | Config | Params(M) | GFLOPs(G) | Box(P) | R | mAP50 | mAP50-95 | FPS (RTX 4090) |
|-------|--------|-----------|-----------|--------|---|-------|----------|---------------|
| [**EsMoE-N**](https://huggingface.co/gatilin/YOLO-Master-ckpts-v0/resolve/main/YOLO-Master-EsMoE-N/YOLO-Master-EsMoE-N.pt?download=true) | [Config](ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml) | 2.68 | 8.7 | 0.684 | 0.536 | 0.587 | 0.427 | 640.18 |
| [**EsMoE-S**](https://huggingface.co/gatilin/YOLO-Master-ckpts-v0/resolve/main/YOLO-Master-EsMoE-S/YOLO-Master-EsMoE-S.pt?download=true) | [Config](ultralytics/cfg/models/master/v0_10/det/yolo-master-s.yaml) | 9.69 | 29.1 | 0.699 | 0.603 | 0.603 | 0.489 | 423.87 |
| [**EsMoE-M**](https://huggingface.co/gatilin/YOLO-Master-ckpts-v0/resolve/main/YOLO-Master-EsMoE-M/YOLO-Master-EsMoE-M.pt?download=true) | [Config](ultralytics/cfg/models/master/v0_10/det/yolo-master-m.yaml) | 34.88 | 97.4 | 0.737 | 0.640 | 0.697 | 0.530 | 243.79 |
| **EsMoE-L** | [Config](ultralytics/cfg/models/master/v0_10/det/yolo-master-l.yaml) | 🔥 training | TBD | TBD | TBD | TBD | TBD | TBD |
| **EsMoE-X** | [Config](ultralytics/cfg/models/master/v0_10/det/yolo-master-x.yaml) | 🔥 training | TBD | TBD | TBD | TBD | TBD | TBD |

### YOLO-Master-v0.1 Series

| Model | Config | Params(M) | GFLOPs(G) | Box(P) | R | mAP50 | mAP50-95 | FPS (RTX 4090) |
|-------|--------|-----------|-----------|--------|---|-------|----------|---------------|
| [**v0.1-N**](https://huggingface.co/gatilin/YOLO-Master-ckpts-v0_1/resolve/main/YOLO-Master-v0.1-N/YOLO-Master-v0.1-N.pt?download=true) | [Config](ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml) | 7.54 | 10.1 | 0.684 | 0.542 | 0.592 | 0.429 | 528.84 |
| [**v0.1-S**](https://huggingface.co/gatilin/YOLO-Master-ckpts-v0_1/resolve/main/YOLO-Master-v0.1-S/YOLO-Master-v0.1-S.pt?download=true) | [Config](ultralytics/cfg/models/master/v0_1/det/yolo-master-s.yaml) | 29.15 | 36.0 | 0.724 | 0.607 | 0.662 | 0.489 | 345.24 |
| [**v0.1-M**](https://huggingface.co/gatilin/YOLO-Master-ckpts-v0_1/resolve/main/YOLO-Master-v0.1-M/YOLO-Master-v0.1-M.pt?download=true) | [Config](ultralytics/cfg/models/master/v0_1/det/yolo-master-m.yaml) | 52.17 | 116.7 | 0.729 | 0.641 | 0.696 | 0.528 | 170.72 |
| [**v0.1-L**](https://huggingface.co/gatilin/YOLO-Master-ckpts-v0_1/resolve/main/YOLO-Master-v0.1-L/YOLO-Master-v0.1-L.pt?download=true) | [Config](ultralytics/cfg/models/master/v0_1/det/yolo-master-l.yaml) | 58.41 | 138.1 | 0.739 | 0.646 | 0.705 | 0.539 | 149.86 |
| **v0.1-X** | [Config](ultralytics/cfg/models/master/v0_1/det/yolo-master-x.yaml) | 🔥 training | TBD | TBD | TBD | TBD | TBD | TBD |

### 📈 PEFT Efficiency (YOLO-Master-EsMoE-N)

| Method | Trainable Params | Adapter Size | mAP50-95 | Training Speedup |
|--------|-----------------|-------------|----------|-----------------|
| Full Fine-Tune | 2.68M (100%) | 5.6 MB | 0.427 | 1.0x |
| LoRA (r=16) | 0.53M (19.8%) | 2.1 MB | 0.418 | 1.6x |
| DoRA (r=16) | 0.65M (24.3%) | 2.5 MB | 0.420 | 1.5x |
| **MoLoRA (r=8)** | **0.48M (17.9%)** | **1.8 MB** | **0.424** | **2.1x** |
| FewShot-LoRA (r=16) | 0.53M (19.8%) | 2.1 MB | 0.416 | 1.8x |
| **V-PEFT (PPO Auto)** | **0.42M (15.7%)** | **1.6 MB** | **0.423** | **2.3x** |

---

## 🛠 Improvements & Fixes

### 🔧 Core Enhancements (v26.02 → v26.08)

| Category | Improvement | Impact |
|----------|-------------|--------|
| **🧠 MoE Routing** | Diversity loss, 3-phase gain schedule, collapse detection | 75% reduction in expert collapse |
| **⚡ MoE Performance** | O(n log n) Gini computation, batched einsum kernels | 40% faster MoE forward pass |
| **🔒 Robustness** | End-to-end NaN self-healing, AMP-safe dispatch | Zero NaN crashes in CI |
| **📦 Export** | Auditable MoE pruning export, routed capability matrix | Full ONNX/TensorRT traceability |
| **🔄 DDP** | Checkpoint lifecycle hardening, EMA sync | Multi-GPU training stability |
| **🍎 MPS** | bilinear grid_sample native implementation | Apple Silicon crash fix |
| **🌐 Wiki** | GitHub Pages bilingual deployment, Material theme | Public documentation site |
| **🤖 Agent** | Profile manifests, release audits, structured fallback | Reproducible experiments |

### 🐛 Selected Critical Fixes

- **#188**: Routing dataset statistics weighted by sample count
- **#158**: Preserve released router checkpoint backward compatibility
- **#161**: YOLOE released checkpoint execution semantics preserved
- **#116**: Comprehensive P0/P1/P2 fix batch across MoE/MoA/MoT/PEFT
- **#124**: AMP dtype alignment for SharedInverted/gated expert index_add_
- **#192/#194**: Pruned expert architecture preserved in model YAML for retraining
- **#211**: PEFT scaling state synchronized to EMA
- **#177**: Alpha warmup preserved across EMA lifecycle
- **#127/#140**: DDP static graph and checkpoint coordination
- **#74**: ONNX export compatibility for MoE expert loss
- **#162**: GitHub Actions pinned to commit SHAs for supply chain security

---

## 🔄 Migration Guide

### From v2026.02 to v2026.08

#### 1️⃣ Configuration File Updates

**Old Version (v2026.02):**

```yaml
# model.yaml
model:
  backbone: CSPDarknet
  head: YOLOv8Head

# MoE configuration
moe:
  num_experts: 8
  top_k: 2

# LoRA configuration
lora:
  r: 16
  alpha: 32
```

**New Version (v2026.08):**

```yaml
# model.yaml
model:
  backbone: CSPDarknet
  head: YOLOv8Head

# MoE configuration (enhanced)
moe:
  num_experts: 8
  top_k: 2
  shared_expert: true          # NEW: Cross-scale expert sharing
  router_governance: true      # NEW: Configurable router hooks
  latent_init_perturb: true    # NEW: Router init perturbation

# LoRA configuration (enhanced)
lora:
  r: 16
  alpha: 32
  vpeft_planner: ppo           # NEW: Auto rank allocation
  molora_routing_aware: true   # NEW: Routing-aware adapter

# NEW: MultiTask configuration
multitask:
  enabled: true
  tasks: [detect, pose, segment]
  task_weights: {detect: 1.0, pose: 0.8, segment: 0.6}

# NEW: MoA/MoT configuration
mixture:
  moa_regional: true
  moa_num_regions: 4
  mot_gshard_balance: true
  sparse_inference: false      # Opt-in for deployment
```

#### 2️⃣ API Changes

```python
# v2026.02
model.train(data="coco.yaml", epochs=100, lora_r=16)

# v2026.08 (fully backward compatible, new features opt-in)
model.train(
    data="coco.yaml",
    epochs=100,
    lora_r=16,                    # Still works
    # New optional parameters:
    moe_shared_expert=True,       # Cross-scale expert sharing
    vpeft_planner="ppo",          # Auto rank allocation
    multitask=True,               # Multi-task training
    moa_regional=True,            # MoA regional attention
    latent_init_perturb=True      # Latent mixture
)
```

#### 3️⃣ Breaking Changes

- **MoE checkpoint format**: v26.08 models with Shared Expert routing are backward compatible with v26.02 weights, but v26.02 checkpoints must be loaded with `legacy_routing=True` flag
- **LoRA merge**: MoLoRA routing-aware merge produces different adapter weights from standard LoRA merge. Use `molora_compat=True` for v26.02 compatibility
- **Agent Skill**: Dispatcher now defaults to `device=mps` on Apple Silicon. Override with `runtime.device` if needed

#### 4️⃣ Deprecation Notices

- `moe_balance_loss_weight` → Use `moe_balance_loss` (consistent naming)
- `lora_auto_r_ratio` → Use `vpeft_planner="ppo"` (intelligent replacement)
- `sparse_sahi` → Use `moa_sparse_inference=True` (unified sparse path)

---

## 🤝 Community

- [GitHub Discussions](https://github.com/Tencent/YOLO-Master/discussions)
- [Issues & Bug Reports](https://github.com/Tencent/YOLO-Master/issues)
- [Feature Requests](https://github.com/Tencent/YOLO-Master/issues/new?template=feature_request.md)
- [Model Zoo](https://github.com/Tencent/YOLO-Master/blob/main/model-zoo/)
- [Wiki (中文)](https://tencent.github.io/YOLO-Master/zh/)
- [Wiki (English)](https://tencent.github.io/YOLO-Master/en/)

---

## 🙏 Acknowledgments

We would like to thank all **30+ contributors** who made this release possible, with special recognition to:

- **isLinXu** (362 commits) — Project lead, MoE architecture, DDP hardening
- **Hertz** (102 commits) — MoA/MoT integration, mixture optimization
- **gatilin** (24 commits) — Agent system, release management
- **kimariyb** (15 commits) — MoT hybrid architecture, domain-specific LoRA
- **13ewat3r** (15 commits) — MoA tests, vertical validation
- **SidKC** (12 commits) — LoRA/V-PEFT lifecycle, routing dataset fixes
- **skywalker-lt** — Edge deployment, Windows GUI, reproduction scripts
- **Lfan-ke** — MoE pruning, MapSaturationScheduler
- **vankari** — MoE schedule study
- **delei-kong** — Vertical dataset reproduction
- **Cooryn** — LoRA vertical scene adaptation, MoA boundary tests
- **Ricky-7-Yan** — Edge inference, MOT analysis, MoE scheduling
- **Zviolin** — Shared Expert MoE (方案 D)
- ...and many more community contributors

Special thanks to the **Ultralytics** team for the upstream v8.4.101 base, and to the research community for foundational work on MoE, LoRA, PPO, and GShard.

---

## 📄 License

This project is licensed under the **Tencent Open Source License**. See [LICENSE](LICENSE) for details.

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Tencent/YOLO-Master/issues)
- **Email**: [gatilin@tencent.com](mailto:gatilin@tencent.com) / [islinxu@163.com](mailto:islinxu@163.com)

---

**Made with ❤️ by the YOLO-Master Team**

---

## [YOLO-Master-v26.02] — 2026-02-13

- Initial release with LoRA support for model training
- MoE (Mixture of Experts) module foundation
- Sparse SAHI inference mode
- Cluster-Weighted NMS (CW-NMS)
- MoE loss function support
- MoE pruning and analysis tools
