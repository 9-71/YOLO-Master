# MPS preflight and A800 experiment protocol

## Scope and candidate support

The initial comparison uses v0.10 no-MoE as the dense control, v0.10 MoE as
the candidate, and v0.10 MoT as an experimental diagnostic. These are experiment
roles, not a new stable-model certification. Keep historical configurations available.

Local MPS runs establish execution, finite loss, checkpoint reload and diagnostic
correctness. They cannot establish A800 speed, COCO accuracy or TensorRT performance.
The existing `suites.yaml` remains a fast CPU smoke catalog. The separate
`hardware_baselines.yaml` explicitly identifies synthetic forward measurements.

## Local MPS commands

Use the repository's editable Python environment. Keep CPU fallback disabled so
unsupported MPS operators fail visibly rather than silently entering CPU timing.

```bash
.venv/bin/python benchmarks/run.py --catalog benchmarks/hardware_baselines.yaml --suite mps_preflight --output runs/benchmarks/mps_preflight
.venv/bin/python benchmarks/benchmark_mot_dispatch.py --device mps --batch 3 --size 16 --warmup 5 --steps 30 --output runs/benchmarks/mps_dispatch/results.json
.venv/bin/python scripts/run_hardware_preflight.py --device mps --output runs/hardware_preflight/mps
```

Dispatch diagnostics use one block, one input and identical masked weights for both
paths. Untimed hooks count executed expert calls and sample-expert pairs. The
single, sample-split and spatial-union routes distinguish token masking from
sample-level work; native routes show random-initialized router behavior. Timers
synchronize the accelerator and disable autograd. Router, normalization/projection,
preprocessing and NMS costs are excluded. Samples execute sequentially; do not
interpret a local timing difference as a formal speedup claim.

The training smoke uses coco8, one epoch and small inputs. It validates a workflow,
not a learning-quality comparison. First use may download the dataset. Each dense
and MoE run saves a checkpoint, reloads it, validates and predicts a synthetic image.

## A800 migration

### Checkpoint expert-choice diagnostics

Use a local trained checkpoint to compare choices at one explicitly named ES_MOE
or MoTBlock layer. The runner rejects missing checkpoints and existing report
directories; dataset resolution disables automatic downloads. It freezes a sorted
image subset with image/label hashes and reloads the same checkpoint for every case.

```bash
.venv/bin/python scripts/evaluate_router_choices.py --weights runs/weights/YOLO-Master-EsMoE-N.pt --list-layers --output runs/router_choices/inventory
.venv/bin/python scripts/evaluate_router_choices.py --weights runs/weights/YOLO-Master-EsMoE-N.pt --layer model.12 --data coco128.yaml --max-images 128 --imgsz 320 --batch 1 --device mps --output runs/router_choices/model12
```

The diagnostic compares native routing, uniform all-expert mixing, native Top-1/Top-2,
each fixed one-hot expert, and a repeated native baseline. Other layers retain their
native policy. Uniform mixing controls for learned weights with all experts kept;
Top-1 and fixed choices change both support and mixture weights. Untimed warmup is
excluded from sample/expert counters. Hooks are restored even if evaluation fails.
The repeated baseline must agree within 1e-6 in reported metrics for a successful
run. Per-image metrics use the validator's IoU=0.5 matching and its low confidence
threshold of 0.001; they are not AP or a deployable oracle selection policy.

The published EsMoE-N checkpoint has four ES_MOE blocks with three experts each;
it differs from the v0.10 candidate architecture. Inspect checkpoint structure
before choosing a layer. COCO128 contains train2017 images and may overlap the
checkpoint's training data: results establish intervention sensitivity, not held-out
generalization. Validation timing contains hook overhead and is not a speedup claim.
On A800 reuse the command with `--device cuda:0` and a fresh output directory;
formal experiments require an independent data split and trained target architecture.

For ES_MOE Top-2, the diagnostic disables the target layer's dynamic threshold so
exactly the two retained experts can execute; otherwise the threshold could silently
drop the second expert. The previous threshold is restored when the context exits.
Use `--policies native_top2` for a focused three-case comparison including native
and its repeated baseline. The intervention hooks are an evaluation tool, not an
exportable budget controller or a saved production pruning operation.

## A800 forward preflight

Reinstall the package in a clean Linux CUDA environment, record the complete
dependency versions, and execute existing P0/P1 regression gates first. Reuse the
preflight command with `--device cuda:0`, then run:

```bash
python benchmarks/run.py --catalog benchmarks/hardware_baselines.yaml --suite a800_forward_preflight --output runs/benchmarks/a800_preflight
python benchmarks/benchmark_mot_dispatch.py --device cuda:0 --batch 3 --size 32 --warmup 50 --steps 300 --output runs/benchmarks/a800_dispatch/results.json
```

These remain forward preflights. Before a formal result, create a checkpoint-based
catalog with trained, matched weights and freeze:

- Commit and dirty diff, environment lock, GPU model/count, CUDA/runtime/driver,
  dtype, power/clock policy, thread settings and exporter/engine versions.
- Dataset identity, split/hash, class mapping, image size, initial weights/hash,
  epochs, seed, optimizer and model-selection rule. Use separate validation and test.
- At least three independent training seeds initially; report each seed and mean
  with dispersion. Increase repetitions when uncertainty prevents a decision.
- Isolated hardware, sufficient warmup, at least 300 synchronized timing samples
  per run, three independent measurement runs, batch 1 and the target production
  batch. These are starting sample counts, not a significance guarantee.
- AP50-95, AP_small, difficult-scene AP; forward and end-to-end p50/p95 separately;
  peak memory and actual sample-expert coverage. Include routing/gather/scatter,
  transfers and postprocessing in the end-to-end scope.
- Eager/export output alignment and separate export performance. Masked dense
  export may preserve values while still executing all experts.

Formal comparison: dense, existing MoE, fixed expert, physical pruning, then budget
controller. Calibrate policy on validation only. Proposed decision target: at most
0.3 AP-point loss with at least 15% lower end-to-end p95, or at least +0.5 AP points
at equal latency budget; freeze the business-specific target before experiments.

## Evidence and publication

Preserve resolved settings, checkpoint hashes, training/validation logs, prediction
artifacts, routing/dispatch observations and benchmark JSON in one indexed run
directory. Use existing release manifests and release audit when publishing a
candidate. A passing smoke, numeric anchor check, or random-weight benchmark is
not a substitute for a complete release evidence bundle.
