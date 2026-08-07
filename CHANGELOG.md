# Changelog

All notable changes to YOLO-Master will be documented in this file.

---

## [YOLO-Master-v26.08] — 2026-08-07

> **Milestone**: First major release integrating the full Mixture-of-Experts (MoE) stack — 620 commits, 96 merged PRs, 30+ contributors across 7.5 months of development.

### 🏗️ MultiTask Learning

Multi-task joint training with dynamic task routing, enabling a single model to perform detection, segmentation, pose estimation, and classification simultaneously.

- `feat(multitask): add MultiTaskHead and TaskRouter core modules`
- `feat(multitask): register MultiTaskModel and wire into YOLO task routing`
- `feat(multitask): add trainer, loss function, and data pipeline`
- `feat(multitask): register multitask task type and add model/dataset configs`
- `test(multitask): add unit tests and data preparation tool`

### 🧠 Mixture of Experts (MoE)

Comprehensive MoE architecture with cross-scale expert sharing, routing governance, dynamic pruning, and diagnostic tooling.

**Cross-Scale Shared Expert (方案 D)**:
- `feat: 跨尺度专家池共享 MoE (方案 D) - Issue #54` (#219)
- `fix: register SharedExpertMoE and repair compare_routing_synthetic_vs_real`

**Routing Governance & Diagnostics**:
- `feat(moe): govern variants with configurable router hooks`
- `feat(moe): add routing diagnostics pruning and phase one gates`
- `feat(moe): routing stability with diversity loss, collapse detection, and 3-phase gain schedule`
- `feat(moe): add MoE diagnostic visualization with HTML dashboard`
- `feat(moe): add unified API compatibility layer for MoE info access`
- `feat(moe): complete issue 52 experiment workflow`
- `docs(moe): add issue 52 experiment report and figures`
- `fix(moe): rerun issue 52 ablations locally`
- `fix(moe): ground issue 52 report in local coco results`
- `fix(moe): make pruning CLI runnable and tolerate legacy router checkpoints`

**Dynamic Scheduling & Pruning**:
- `feat(moe): add dynamic scheduling and pruning sweep tools`
- `feat(moe/scheduler): add MapSaturationScheduler for mAP-driven balance annealing` (#104)
- `fix(moe): preserve pruned experts' kernel sizes in model.yaml` (#192)
- `fix(moe): sync pruned expert count into model.yaml so retraining keeps pruned architecture` (#194)
- `fix(moe): preserve pruned structure during lora recovery`
- `feat(moe): add weight verification tool for MoE checkpoint loading`

**Routing Correctness & Stability**:
- `fix(moe): P0/P1/P2 fixes for routers, loss, quantize, scheduler, integration` (#116)
- `fix(moe): routing grad flow, GShard balance, subclass re-init`
- `fix(moe): soft-balancing semantics, expert norm stability, batched dispatch (P2)`
- `fix(moe): harden AdvancedRoutingLayer and fix z-loss ordering (P1)`
- `fix(moe): eliminate runtime crashes in get_gflops and weight init (P0)`
- `fix(moe): restore balance-loss gradient to router and harden aux-loss registry`
- `fix(moe): restore router gradient in soft-balancing load loss (C-soft)`
- `fix(moe): DDP-aware balance loss, snapshot sampling, export guards`

**MoE Training Robustness**:
- `fix(moe): align SharedInverted/gated expert dtype for AMP index_add_` (#124)
- `fix(moe): propagate AMP dtype alignment across all sparse MoE paths`
- `fix(moe): add AMP-safe index_add_aligned_ helper`
- `fix(moe): set moe_balance_loss and moe_router_z_loss defaults to 1.0`
- `fix(moe-loss): align MoE loss magnitude with box/cls/dfl losses`
- `fix(trainer): gate MoE warmup/gain/collapse logic behind _has_moe`
- `fix(moe): add shared_expert compute + stop_gradient on routing weights`
- `fix(moe): prevent long-term moe_loss collapse`
- `fix(moe): add defensive checks to prevent moe-loss collapse`
- `fix(trainer): remove duplicate LoRA injection in _setup_train`

**MoE NaN Safety**:
- `fix: harden NaN handling end-to-end — pre-batch detection, per-component isolation, self-healing EMA`
- `fix(moa,mot,loss): comprehensive NaN safety for router aux losses and EMA scales`
- `fix: harden MoE nonfinite recovery`
- `fix: harden MoE NaN recovery and routing`
- `fix(validate): treat router nan as recovery signal`
- `fix(moe): skip scheduling updates for non-finite fitness`
- `fix(moe): update mAP scheduling only for accepted epochs`

**MoE Export & ONNX**:
- `fix(moe): ONNX export compatibility and missing aux_loss property` (#74)
- `fix: apply MoE routing exclusions during ONNX quantization`

**MoE Performance**:
- `perf(moe): compute usage_gini in O(n log n) via sorted cumulative form`

### ⚡ Mixture of Attention (MoA)

Regional attention mechanism for efficient multi-scale feature processing.

- `Add MoA attention modules` (#48)
- `Add YOLO-Master MoA model configs`
- `Add MoA tests and ablation script`
- `fix(moa): P0/P1/P2 fixes for SVD fallback, init weights, eff_heads` (#116)
- `fix(moa): make random-feature buffer persistent for reproducible runs`
- `fix(moa): declare sparse inference params missing from main merge`
- `fix: stabilize MoA layouts and MPS numerics`
- `fix(trainer): anneal MoA and MoT temperatures`
- `fix: prevent mixture EMA buffer from landing on CPU and detach buffers during NCCL sync`

### 🔀 Mixture of Transformers (MoT)

Transformer-based routing with GShard balance loss for improved feature representation.

- `Add MoT transformer routing modules` (#96)
- `Add MoT model configs and ablation tests`
- `Document MoT integration experiments`
- `feat: MoT/MoA ablation boundary tests and delivery documentation (#54)` (#146)
- `fix(mot): P0/P1/P2 fixes for torch.roll, grid_sample fp16, GPU sync` (#116)
- `fix(mot): deterministic window shift, remove GPU sync, trace-stable routing`
- `fix(mot): add boundary condition unit tests and handle edge case exceptions (#54)` (#189)
- `test(mixture): cover MoT DDP dispatch contracts`

### 🎯 PEFT / LoRA / MoLoRA

Parameter-efficient fine-tuning with advanced routing-aware capabilities.

**V-PEFT Planner System**:
- `feat(vpeft): add PlannerResult as stable external planner contract`
- `feat(vpeft): implement PPO rank allocation`
- `feat(vpeft): expose semantic targets and planner budget config`
- `feat(lora): persist planner result contract and structured V-PEFT fallback reasons`
- `feat: record V-PEFT solver diagnostics`
- `perf(peft): optimize planner graph and MoLoRA dispatch`

**MoLoRA (Mixture-of-LoRA)**:
- `feat: Add MoLoRA (Mixture-of-LoRA) PEFT extension for YOLO-Master`
- `feat(molora): add routing-aware merge calibration`
- `perf(molora): vectorize same-rank Linear experts in batched einsum kernels`
- `fix(peft): unify MoLoRA routing and rank contracts`
- `fix(peft/molora): P0/P1/P2 fixes for layer, model, moe_aware, utils`

**LOVO Cross-Validation**:
- `feat(lora): implement LOVO (Leave-One-Variant-Out) cross-validation engine`
- `feat(lora): add LOVO data collection and validation CLI tools`
- `feat(lora): extend PEFT API and config for LOVO integration`
- `test(planner): add grouped LOAO and variant LOVO validation`

**FewShot-LoRA**:
- `feat(lora): add FewShot-LoRA with scheduled DropConnect, hierarchical distillation, and variational rank`
- `fix(lora): preserve few-shot fallback settings`
- `fix: narrow fallback RS-LoRA scope`
- `fix: honor fallback LoRA effective configuration`

**LoRA EMA & Checkpoint Lifecycle**:
- `fix(lora): sync PEFT scaling state to EMA` (#211)
- `fix(lora): preserve alpha warmup across EMA lifecycle` (#177)
- `docs(lora): record PEFT EMA rank sweep results`

**LoRA Training & Stability**:
- `feat: LoRA vertical scene adaptation for YOLO-Master-EsMoE-N (#50)` (#135)
- `fix(lora): clarify result protocols and prevent CSV overwrite` (#178)
- `fix(lora): LoRAConfig object support, dedup mapping keys, YOLO12 attention safety`
- `fix(lora): prevent LoRA collapse on YOLO12 Area-Attention (A2C2f)`
- `perf(lora): better capacity allocation — bounded per-layer rank, stem skip, narrow-layer filter`
- `fix(lora): allow rankless PEFT types (BOFT/OFT/HRA/IA3) when r=0`
- `fix(boft): default block_size 4→2 and add auto-downgrade for YOLO Conv2d compatibility`
- `fix(unfreeze): unfreeze RT-DETR detection head during LoRA training`

**PEFT Backend Support**:
- `feat(lora): add HRA backend, training strategy params, and update MoE/LoRA defaults` (#166)
- `fix: PEFT adapter param stats miscount — OFT/BOFT/IA3/HRA/LoHa/LoKr shown as 0`
- `refactor: consolidate PEFT param stats & fix remaining hardcoded lora_ checks`
- `feat: add performance warnings for slow PEFT variants (HRA/OFT)`

**MoE-Aware PEFT**:
- `feat(peft): add MoE-aware PEFT core module` (#115)
- `feat(moe): add MoE v0.1-v0.10 block modules with dynamic scheduler`
- `feat(trainer): integrate MoE auxiliary losses, MoLoRA routing, and Planner hooks`

**V-PEFT & Contracts**:
- `feat(peft): integrate V-PEFT placement plans and adapter metadata` (#165)
- `feat(peft): strengthen MoLoRA and V-PEFT contracts`
- `fix(vpeft): restore per-node variant constraint semantics` (#163)

**Domain-Specific LoRA**:
- `fix(lora): repair 5 P0 + 4 P1 bugs uncovered by deep audit`
- `fix: reproject AO decisions onto adapter budget`
- `fix: harden router overflow and AO placement`
- `fix(planner): align architecture fingerprints and budget planning`

### 🌡️ Latent Mixture

Latent-space routing with initialization perturbation and temperature annealing.

- `feat(latent): register latent mixture models and export policy`
- `feat(latent): add router init perturbation support`
- `feat(latent): add init-perturb temperature configs`
- `feat(latent-mixture): enrich aux mixin and routing snapshot diagnostics`
- `test(latent): cover init-perturb anneal controls`
- `docs(latent): record init-perturb anneal plan`

### 🚀 Edge Deployment

Cross-platform edge deployment with native GUI and multiple inference backends.

**Windows GUI Runner**:
- `feat(examples): add native Windows 10/11 GUI runner (Dear ImGui + D3D11)` (#176)

**Edge Inference Backends**:
- `feat(edge): add unified edge deployment API`
- `feat(examples): add YOLO-Master-EsMoE-N ONNX/NCNN/MNN C++ inference example` (#97)
- `feat(examples): add Jetson Orin TensorRT deployment to the EsMoE-N edge example` (#105)
- `feat: Add MNN backend support for YOLO-Master Edge Deployment`
- `feat: Refactor YOLO-Master Edge Deployment C++ Benchmark` (#106)
- `feat(edge): add YOLO-Master deployment validation example` (#94)

**Cross-Platform Support**:
- `feat(edge): add fair CPU thread controls` (#209)
- `examples: rename to YOLO-Master-Cross-Platform-Edge-Deployment + add macOS Core ML Runner` (#134)
- `feat(examples): add native Windows 10/11 GUI runner (Dear ImGui + D3D11)`

### 🍎 Apple Silicon / MPS

Critical fixes for stable training on macOS with Metal Performance Shaders.

- `fix(mps): implement bilinear grid_sample with MPS-native ops to prevent crash`
- `fix: stabilize MoA layouts and MPS numerics`

### 🔧 Infrastructure & Governance

**DDP & Distributed Training**:
- `fix(ddp): disable static graph for hybrid moe` (#127)
- `fix(ddp): enable static graph for moe expert hooks`
- `fix: stabilize MoE DDP expert dispatch`
- `fix: harden DDP nonfinite recovery lifecycle`
- `fix: bootstrap DDP recovery before first optimizer step`
- `fix: serialize bootstrap checkpoint before epoch loop`
- `fix(ddp): synchronize epoch-end checkpoint coordination` (#140)
- `fix(ddp): prevent EMA buffer schema divergence`
- `fix(ddp): prevent NCCL crash by avoiding buffer registration for ES_MOE stats`
- `fix: harden cross-module DDP training`
- `fix: guard CPU training from CUDA DDP`
- `fix: propagate DDP worker root cause to parent process`
- `fix: detach DDP reductions from autograd graphs`
- `perf: pack MoE DDP statistics collectives`

**Checkpoint & Recovery**:
- `fix(checkpoint): harden DDP save and recovery lifecycle`
- `fix(trainer): restore upstream checkpoint saving`
- `fix(trainer): gate checkpoints on live finite state`
- `fix(trainer): preserve scaler backoff during recovery`
- `fix(trainer): tolerate missing EMA recovery buffers`
- `fix(trainer): recover before validating nonfinite ema`
- `fix(mixture): backport routed safety to main`

**Export Governance**:
- `feat(export): add auditable MoE pruning and routing export` (#182, #193)
- `feat(export): add routed capability matrix`
- `feat: add routed export preflight checks`
- `feat: add executable export and compile gates`
- `fix(export): add MoE-aware mixed-precision quantization`
- `fix: apply MoE routing exclusions during ONNX quantization`

**Mixture Governance**:
- `feat: govern mixture configuration and model registry` (#139)
- `feat: unify routed module runtime contracts`
- `feat(mixture): harden MoE MoA MoT and latent routing` (#175)
- `feat(mixture): add opt-in sparse inference paths`
- `feat: wire mixture attention runtime config`
- `feat: optimize MoA regional and MoT attention` (#195, #217)
- `perf(mixture): batch auxiliary loss and routing state updates`
- `docs(governance): document mixture profiles and release gates`

**CI & Tooling**:
- `ci: pin GitHub Actions to commit SHAs` (#162)
- `ci: add mixture regression and pytest gates`
- `ci: enforce routed governance evidence gates`
- `feat(config): add repository drift detector`

### 🌐 Wiki & Documentation

Bilingual documentation system with GitHub Pages deployment.

- `feat(wiki): add GitHub Pages deployment workflow and fix sync quality checks`
- `feat: launch GitHub Pages model zoo`
- `feat: publish repowiki to GitHub Pages (en + zh)`
- `feat: Redesign GitHub Pages with Material Indigo theme`
- `docs: refresh RepoWiki generated snapshot`
- `docs: add SECURITY.md`
- `docs: add AGENTS.md and CLAUDE.md for agent task routing and conventions`
- `docs: add phase 2/3 submission report`
- `docs(moe): record governance and export decisions`
- `docs(governance): document mixture profiles and release gates`
- `docs(vpeft): remove AAAI 2026 references`
- `docs: add YOLO-Master governance roadmap`

### 🤖 Agent System

Agent-based experiment orchestration with profile manifests and release audits.

- `feat(agent): add profile manifests and release audits`
- `refactor: YOLO agent into repo agent module` (#39)
- `feat: add multimodal batch evaluation to yolo-master-agent`
- `feat: add agent validation schemas and cases`
- `feat: add agent async evaluation capabilities`
- `feat: add multimodal open-world agent resources`

### 🧪 Testing & Quality

- `test(moe): cover multi-seed ablation aggregation`
- `test(mixture): cover MoT DDP dispatch contracts`
- `test(latent): cover init-perturb anneal controls`
- `test(config): cover PEFT runtime metadata propagation`
- `test(ddp): cover checkpoint coordination across ranks`
- `test(lora): cover MoE control-path exclusion and complexity detach`
- `test(moe): add AMP index_add_ Half/Float regression coverage`
- `test(moe): add 16 regressions for the 2026-06-25 deep-scan fixes`
- `test(moe): add MoE module and aux-loss regression suite`
- P0/P1/P2 fix batches with comprehensive regression coverage (#116)

### 📊 Reproduction & Benchmarks

- `feat(reproduce): add BCCD dataset reproduction for Issue #49` (#123)
- `feat(reproduce): add VisDrone & SKU-110K baseline reproduction scripts` (#174)
- `feat(reproduce): reproduce script for brain tumor dataset`
- `feat(reproduce): reproduce script for construction-ppe dataset`
- `feat: add COCO128 three-seed runner`
- `feat: add multi-seed ablation aggregator`
- `feat(benchmark): add standard mixture suite`
- `fix: add vertical dataset reproduction results` (#171)

### 🐛 Selected Critical Fixes

- `fix(routing): weight dataset statistics by sample` (#188)
- `fix(moe): preserve released router checkpoint compatibility` (#158)
- `fix(yoloe): preserve released checkpoint execution semantics` (#161)
- `fix(config): preserve requested PEFT audit metadata`
- `fix: harden mixture routing correctness`
- `fix: clarify sparse routing diagnostics`
- `fix(world/train): handle list-type img_path in set_text_embeddings`
- `fix: prevent fp16 CIoU gradient NaNs`
- `fix: recover from nonfinite gradients under AMP`
- `fix: re-align mixture EMA buffer device after model.to()`
- `fix: avoid MoE collectives during validation`

---

## [YOLO-Master-v26.02] — 2026-02-13

- Initial release with LoRA support for model training
- MoE (Mixture of Experts) module foundation
- Sparse SAHI inference mode
- Cluster-Weighted NMS (CW-NMS)
- MoE loss function support
- MoE pruning and analysis tools
