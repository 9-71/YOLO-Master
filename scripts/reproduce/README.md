# Reproduction Methodology for Training the Baseline YOLO-Master-v0.1-N & YOLO-Master-EsMoE-N on VisDrone, SKU-110K, and AI-TOD-v2
 

Reproducible training strategy for the two YOLO-Master nano variants on three vertical scenes, with per-epoch logging of the required metrics (mAP50, mAP50-95, box_loss, cls_loss, moe_loss)

📊 **Live training curves for all eight runs (Weights & Biases):** https://wandb.ai/1853979230-company-/yolo-master-reproduce

| Model | Config | # Params | MoE characteristics |
| --- | --- | --- | --- |
| `YOLO-Master-v0.1-N` | `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml` | 7.55 M | `ModularRouterExpertMoE` |
| `YOLO-Master-EsMoE-N` | `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml` | 2.69 M | `ES_MOE` |

Weights:

| Dataset | Model | mAP50 | mAP50-95 | Weights |
| --- | --- | --- | --- | --- |
| VisDrone | `YOLO-Master-v0.1-N` | 0.3443  | 0.2009 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-v01-n-visdrone.pt) |
| VisDrone | `YOLO-Master-EsMoE-N` | 0.3499 | 0.2029 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-esmoe-n-visdrone.pt) |
| SKU-110K | `YOLO-Master-v0.1-N` | 0.9059 | 0.5821 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-v01-n-sku110k.pt) |
| SKU-110K | `YOLO-Master-EsMoE-N` | 0.9041 | 0.5829  | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-esmoe-n-sku110k.pt) |
| AI-TOD-v2 | `YOLO-Master-v0.1-N` | 0.2822 | 0.1204 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-v01-n-aitodv2.pt) |
| AI-TOD-v2 | `YOLO-Master-EsMoE-N` | ≈0 (collapsed) | ≈0 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-esmoe-n-aitodv2.pt) |

Below is a comprehensive guide on how to reproduce the full training pipeline

### 🚀 Update: New dataset: AI-TOD-v2

[AI-TOD-v2](https://github.com/Chasel-Tsui/AI-TOD-v2) is a much harder aerial **tiny-object** benchmark: 8 classes, ~800px crops, **mean object size ≈ 12px**, and a heavy class imbalance (one class dominates ~88% of the boxes). It pushes the two nano variants well past VisDrone/SKU-110K, and cleanly exposes a difference between their two MoE designs.

## 1. Setup

Below is the physical server setup I used to train the two models:

| Category | My setup | Recommended |
| --- | --- | --- |
| OS | Ubuntu Server 22.04 LTS | Linux |
| CPU | Intel Xeon 8568Y 96C192T | 8C16T |
| Memory | 2,048GB | >32GB |
| GPU | Nvidia H200 SXM (only enabled 1) | Nvidia Ampere architecture (`sm_80`) or newer |
| GPU VRAM | 144GB / GPU | ≥16GB |
| Driver | `570.211.01` |  |
| Python | `3.14.6` | `3.13` or newer |
| CUDA | `12.8` | `12.8` |
| PyTorch | `2.11.0+cu128` | `2.11.0+cu128` |

Follow the official setup guides for environmental setup. Or directly install the exact conda environment I used: [download here](https://drive.google.com/file/d/1gskbzdVQ56pZBgungk9HcKtb2ft5WaVf/view?usp=share_link)

Download it (don't extract yet!), then run:

```bash
# 1) download the pack from Google Drive, then extract into your conda's envs dir
pip install gdown
gdown 1gskbzdVQ56pZBgungk9HcKtb2ft5WaVf -O yolo_master.tar.gz
ENV_DIR="$(conda info --base)/envs/yolo_master"
mkdir -p "$ENV_DIR"
tar -xzf yolo_master.tar.gz -C "$ENV_DIR"

# 2) activate, then rewrite the packed paths for THIS machine (conda-unpack ships inside the pack; run once)
conda activate yolo_master
conda-unpack

# 3) install this repo's ultralytics into the env (editable pkg was NOT bundled in the pack)
pip install -e .
```

## 2. Dataset Download

### VisDrone & SKU-110k:

Datasets shall download automatically the first time training initializes. To fetch them manually, execute:

```bash
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('VisDrone.yaml', autodownload=True)"
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('SKU-110K.yaml', autodownload=True)"
```
### AI-TOD-v2:

Since this dataset is constructed upon the [xVIEW](https://xviewdataset.org) dataset, please refer to the offical [AI-TOD repo](https://github.com/jwwangchn/AI-TOD) for raw image downloads and generation scripts. 

The dataset should be stored under the default Ultralytics `datasets_dir` , usually under `../datasets`. VisDrone is ~2.3GB, SKU-110K ~13.6GB and AI-TOD-v2 is ~27GB

## 3. Training

Recommended hyperparam settings: `--imgsz 640` (`--imgsz 800` is the native resolution of AI-TOD-v2) ,`--epochs 300` , and adjust batch size `--batch` based on your GPU memory. 

### Full commands

```bash
# Adjust the batch size and # of epochs based on your computer's capability.

# ------ VisDrone ------
# YOLO-Master-v0.1-N
python scripts/reproduce/reproduce_visdrone.py --model v0.1-N  --epochs <epoch> --batch <batch-size> 
# YOLO-Master-EsMoE-N
python scripts/reproduce/reproduce_visdrone.py --model EsMoE-N --epochs <epoch> --batch <batch-size>  --no-sparse-eval

# ------ SKU-110K ------
# YOLO-Master-v0.1-N
python scripts/reproduce/reproduce_sku110k.py  --model v0.1-N  --epochs <epoch> --batch <batch-size> 
# YOLO-Master-EsMoE-N (use --no-sparse-eval for correct dense evaluation)
python scripts/reproduce/reproduce_sku110k.py  --model EsMoE-N --epochs <epoch> --batch <batch-size>  --no-sparse-eval

# ------ AI-TOD-v2 ------
# YOLO-Master-v0.1-N
python scripts/reproduce/reproduce_aitodv2.py --model v0.1-N  --imgsz 800 --batch 64 --epochs 300
# YOLO-Master-EsMoE-N
python scripts/reproduce/reproduce_aitodv2.py --model EsMoE-N --imgsz 800 --batch 64 --epochs 300 --no-sparse-eval
```

### Key flags

| Flag | Default | Explanation |
| --- | --- | --- |
| `--model {v0.1-N,EsMoE-N,both}` | `both` | which model to train |
| `--no-sparse-eval` | **off** | **opt-in** correct evaluation for `EsMoE-N` **(see Known issue 1 below)**. Off = reproduce the model exactly as shipped. No-op for `v0.1-N`. |
| `--epochs / --imgsz / --batch` | `300 / 640 / 64` | training hyps |
| `--wandb / --no-wandb` | on | stream per-epoch metrics to Weights & Biases |
| `--wandb-entity <e>` | **default** | W&B entity/team to log under |
| `--wandb-mode {online,offline,disabled}` | `online` | W&B mode. To use `online` , you must login first. |

Tune batch size smaller if you encountered OOM (CUDA out of memory) errors.

A training of 100 epochs can already achieve a high mAP. **Only train for 300 or more epochs if you have enough GPU memory or want to challenge the SOTA.** 

---

### ⚡️(NEW!) DDP Training
The default scripts `reproduce_visdrone.py`, `reproduce_sku110k.py`, and `reproduce_aitodv2.py` only work with a single GPU. Here, we also provide a script dedicated to distributed training. It is compatible with both baseline models (`v0.1-N` and `EsMoE-N`) and the three datasets in the default scripts, and runs smoothly on NVIDIA's mainstream datacenter GPUs (we have tested A100, H100, H200, and B200).

> **Note:** it runs **within a single node** only, and will not work across multiple physical servers even if they are joined by an InfiniBand switch.

#### Commands

Multi-GPU DDP is triggered by a **comma-separated `--device`** with **≥ 2 GPUs**. `--batch` is the **total** batch, split evenly across the GPUs.

```bash
# VisDrone — EsMoE-N on 4 GPUs
python scripts/reproduce/reproduce_ddp.py --dataset VisDrone --model EsMoE-N \
    --device 0,1,2,3 --batch 128 --epochs 300 --no-sparse-eval --workers 0

# SKU-110K — v0.1-N on 2 GPUs
python scripts/reproduce/reproduce_ddp.py --dataset SKU-110K --model v0.1-N \
    --device 0,1 --batch 64 --epochs 300 --workers 0

# AI-TOD-v2 — v0.1-N on 8 GPUs (imgsz auto-set to 800)
python scripts/reproduce/reproduce_ddp.py --dataset AI-TOD-v2 --model v0.1-N \
    --device 0,1,2,3,4,5,6,7 --batch 256 --epochs 300 --workers 0

# Train both models on a dataset, sequentially
python scripts/reproduce/reproduce_ddp.py --dataset VisDrone --model both --device 0,1,2,3 --batch 128 --workers 0

# Preview the plan without launching (or validate the model builds with --check-build)
python scripts/reproduce/reproduce_ddp.py --dataset VisDrone --model EsMoE-N --device 0,1 --dry-run
```

**Key flags**

| Flag | Meaning |
| --- | --- |
| `--dataset` | `VisDrone` / `SKU-110K` / `AI-TOD-v2` |
| `--model` | `v0.1-N`, `EsMoE-N`, or `both` |
| `--device` | comma-separated GPU ids, **≥ 2** (single GPU / CPU is refused) |
| `--batch` | **total** batch across all GPUs (keep it divisible by the GPU count) |
| `--no-sparse-eval` | corrected dense evaluation for `EsMoE-N` (no-op for `v0.1-N`) |
| `--lr0` / `--optimizer` | LR override for large-batch scaling (e.g. `--lr0 0.04` at large batch; forces SGD) |
| `--cache ram \| disk` | cache decoded images in RAM or on disk; omit to read on the fly |
| `--workers` | dataloader workers **per GPU** (see note below) |

> **Tip — `--workers 0`:** on the Python 3.14 stack, dataloader worker processes can stall DDP setup on some hosts. `--workers 0` loads data in the main process (a little slower per step) and is the reliable default; you can increase it once a run is confirmed stable.

**Notes**
- Runs are **resumable** — re-issuing the same command continues from `last.pt`.
- Uses the single-node convention (GPUs `0..N-1`). To pin specific GPUs, set `CUDA_VISIBLE_DEVICES=4,5` and pass `--device 0,1`.
- For a large batch, scale the learning rate (linear rule, e.g. `--batch 512 --lr0 0.04`).

---

### Expected results

`v0.1-N` trains and validates cleanly all three datasets. `EsMoE-N` **trains correctly** on VisDrone & SKU-110k (its train losses track `v0.1-N`), ***but with the default sparse evaluation its validation mAP collapses***; `--no-sparse-eval` restores it to the `v0.1-N` level **(see Known issue 1 for the mechanism)** on VisDrone & SKU-110k. Moreover, `EsMoE-N`'s routering mechanism **completely collapsed** on AI-TOD-v2 even with `--no-sparse-eval` due to the scale and high complexity of the dataset **(see Known issue 4)**. 

| Model | Dataset | Eval Method | mAP50 | mAP50-95 | W&B Run | Raw Results |
| --- | --- | --- | --- | --- | --- | --- |
| v0.1-N | VisDrone | default | 0.344 | 0.201 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/rbmyjy6b) | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-v0.1n-visdrone.zip) |
| EsMoE-N | VisDrone | default (sparse) | 0.010 | 0.003 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/49bmlyp2) | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-esmoen-sparse-visdrone.zip) |
| EsMoE-N | VisDrone | `--no-sparse-eval` | 0.350 | 0.203 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/6rsdhsn9) | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-esmoen-visdrone.zip) |
| v0.1-N | SKU-110K | default | 0.906 | 0.582 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/rogiamt4) | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-v0.1n-sku110k.zip) |
| EsMoE-N | SKU-110K | default (sparse) | 0.305 | 0.136 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/7nofdfnb) | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-esmoen-sparse-sku110k.zip) |
| EsMoE-N | SKU-110K | `--no-sparse-eval` | 0.904 | 0.583 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/yiz22jp3) | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-esmoen-sku110k.zip) |
| v0.1-N | AI-TOD-v2 | default | 0.282 | 0.120 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/x8447xku) | N/A |
| EsMoE-N | AI-TOD-v2 | `--no-sparse-eval` | ≈0 (collapsed) | ≈0 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/x8447xku) | N/A | 

*The raw results of the AI-TOD-v2 training runs are unavailable. Please check the W&B runs instead.

### Visualization

| Model | VisDrone | SKU-110K | AI-TOD-v2 |
| --- | --- | --- | --- |
| **v0.1-N** | <img width="2234" height="882" alt="v0.1-visdrone" src="https://github.com/user-attachments/assets/7d076b6f-48aa-48a2-8d0c-31f55164d76b" /> | <img width="2234" height="882" alt="v0.1-sku110k" src="https://github.com/user-attachments/assets/15f98b56-0c47-4665-878f-0fc13e657381" /> | <img width="2234" height="882" alt="v0 1-aitodv2" src="https://github.com/user-attachments/assets/e10de077-22ac-4744-ab45-89e56a4cddd2" /> |
| **EsMoE-N (sparse eval)** | <img width="2234" height="882" alt="esmoe-sparse-visdrone" src="https://github.com/user-attachments/assets/e9a0dd9d-e760-4b41-8b06-c076f9793ad9" /> | <img width="2234" height="882" alt="esmoe-sparse-sku110k" src="https://github.com/user-attachments/assets/9fbf7230-fa11-4f55-9609-644a7b973762" /> | N/A (experiment not conducted) |
| **EsMoE-N (`--no-sparse-eval`)** | <img width="2234" height="882" alt="esmoe-visdrone" src="https://github.com/user-attachments/assets/1258edb0-bc03-4f50-84d3-b86507d663f6" /> | <img width="2234" height="882" alt="esmoe-sku110k" src="https://github.com/user-attachments/assets/cc50bd4c-1079-4abd-8b4d-9408d10d01a9" /> | <img width="2234" height="882" alt="esmoe-aitodv2" src="https://github.com/user-attachments/assets/8b9bb8d4-f32b-46d3-9176-0a9364889d8e" /> |

***Expected qualitative trend: `--no-sparse-eval` lifts `EsMoE-N` from collapsed (VisDrone) / far-below-baseline (SKU-110K) up to outperform the `v0.1-N` mAP, only with ~1/3 of its parameters. However, it may not help on AI-TOD-v2 since the rounting mechanism itself is incapable of handling it.***

## 4. Additional Reproduction Results

Below are the results from a reproduction run on RTX 4050 6GB GPU (Windows 11), training for 100 epochs with `batch=4`:

| Model | Dataset | Eval Mode | mAP50 | mAP50-95 | W&B Run |
| --- | --- | --- | --- | --- | --- |
| v0.1-N | VisDrone | default | 0.2916 | 0.1666 | [View](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/501cw1pq) |
| EsMoE-N | VisDrone | default (sparse) | 0.3008 | 0.1731 | [View](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/h9doimnp) |
| EsMoE-N | VisDrone | dense (`--no-sparse-eval`) | 0.3037 | 0.1758 | [View](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/h9doimnp) |
| v0.1-N | SKU-110K | default | 0.8877 | 0.5636 | [View](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/hkqclc61) |
| EsMoE-N | SKU-110K | default (sparse) | 0.2577 | 0.1106 | [View](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/fivhxn27) |
| EsMoE-N | SKU-110K | dense (`--no-sparse-eval`) | 0.8970 | 0.5728 | [View](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/l0cdknhd) |

**WandB Project**: [https://wandb.ai/1853979230-company-/yolo-master-reproduce](https://wandb.ai/1853979230-company-/yolo-master-reproduce)

**Key Observations**:
- v0.1-N reproduces closely with official reference (VisDrone: ~85% of official mAP50-95, SKU-110K: ~97% of official mAP50-95) at 100 epochs
- EsMoE-N shows depressed mAP on SKU-110K under default sparse inference mode, consistent with the known issue described below
- With `--no-sparse-eval` (dense eval), EsMoE-N mAP50 recovers dramatically on SKU-110K: 0.2577 → 0.8970, reaching parity with v0.1-N (0.8877)
- On VisDrone, dense eval provides marginal improvement (0.3008 → 0.3037), as sparse inference already performs comparably
- This confirms the ES_MOE sparse inference issue identified in PR #81

---

## 5. Known issues + solutions/takeaways ‼️Very important‼️

### 1. **`EsMoE-N` validation mAP collapses on VisDrone & SKU-110k (ES_MOE sparse inference)**

**Symptom.** `EsMoE-N` train losses (`box/cls/dfl`) descend normally — identical to `v0.1-N` — yet its **validation** mAP is near zero (VisDrone only ~0.01) or far below the `v0.1-N` (SKU-110K 0.31 vs 0.91). On VisDrone the mAP peaks mid-training then decays toward zero. 

**Why it happens? The machanism:**

the function `ES_MOE.forward` in `ultralytics/nn/modules/moe/modules.py` uses two different code paths: 

- **Training** → `_dense_forward`: it computes **all** experts and sums them weighted by the softmax routing weights, which sum to 1 → output at the correct magnitude.
- **Inference** → `_sparse_forward` (taken because `use_sparse_inference=True` by default). For these configs (`top_k=None`, i.e. dense softmax over all experts), it:
    1. **Prunes** every expert whose routing weight `< dynamic_threshold` (`0.4`), keeping only the top-ranked one. With ~4 experts whose softmax weights average ~0.25, this reduces the block to ≈ top-1.
    2. **Does not renormalize**: the surviving expert's output is scaled by its raw softmax weight (~0.3) and never rescaled to sum-1.

So at inference the block emits roughly **one** expert at **~1/N** the activation magnitude that the trained `BatchNorm` (`self.norm`) was fitted to during dense training. The downstream head then sees mis-scaled, wrong-expert features → degenerate detections. It gets worse as training sharpens the router.

**Proof.** Re-validating the **same** trained checkpoint with the two paths:

- sparse (default) → mAP50 0.06
- forced dense → mAP50 0.35 (≈ the `v0.1-N` result)

The weights are fine; but the inference path is wrongly configured.

**Solution.** `--no-sparse-eval` registers a callback that sets `ES_MOE.use_sparse_inference=False` on both the live model and its EMA at `on_pretrain_routine_end` / `on_train_start`, before any validation and before the EMA-derived checkpoints are written. Per-epoch validation, the saved `.pt`, and the final evaluation then all use the dense forward that matches training.

**Why `v0.1-N` is unaffected.** Its MoE block (`OptimizedMOEImproved`) runs the **same** top-k routing in train and eval (no dense→sparse switch), and adds an always-on shared expert plus a residual — a mode-invariant dense path that keeps the output scale stable.

**Be careful:** 

This is fixed at run time (a script flag), not in the library — `ES_MOE`'s default `use_sparse_inference=True` and `_sparse_forward` are unchanged. A plain `yolo val` or an exported `EsMoE-N` model will still exhibit the same collapse.

### 5. SKU-110K extraction error (`tar ... Operation not permitted`)

**The mechanism.** The dataset downloader extracts the SKU-110K archive with `tar xfz`, which tries to restore the files' archived ownership (`uid` / `gid`). On filesystems that disallow `chown` — for example, many networked, rootless, or container mounts — `tar` prints:

```bash
Cannot change ownership ... Operation not permitted
```

and exits non-zero, leaving the dataset unprepared.

**Solution.** Extract the archive while ignoring ownership and permissions:

```bash
tar -xzf SKU110K_fixed.tar.gz --no-same-owner --no-same-permissions -C <datasets_dir>
```

Then let Ultralytics re-run `check_det_dataset('SKU-110K.yaml')` to build the labels and the `train.txt`, `val.txt`, and `test.txt` split files.

### 4. `model.val()` hangs/crashes with dataloader workers on Python 3.14 (minor)

**Mechanism.** A standalone `model.val()` call with `workers >0` can hit a multiprocessing forkserver `ConnectionResetError` on Python 3.14.

Full training is unaffected because it validates each epoch through the training path.

**Solution.** Pass `workers=0` for standalone validation invocations:

```python
model.val(workers=0)
```
### 6. `ES_MOE` routing collapses on AI-TOD-v2

**Mechanism.** On AI-TOD-v2's homogeneous tiny objects, `EsMoE-N`'s pure top-k router collapses onto a single expert — one early MoE layer reaches `>0.8` max-usage — and validation mAP freezes at the noise floor (≈1e-5), even with the built-in routing-collapse recovery. `v0.1-N`'s `ModularRouterExpertMoE` keeps an always-on **shared expert** (a guaranteed signal path independent of the router), so on the *identical* data it trains cleanly to 0.28 mAP50.

**Takeaway.** On tiny-object / low-diversity data, prefer the shared-expert `v0.1-N`; `ES_MOE` needs a larger, more diverse distribution to keep its experts balanced. The AI-TOD-v2 `EsMoE-N` weights above are included for completeness — they are a collapsed model. 

## 7. Directory for Run logs

Per run, Ultralytics writes to:

```
runs/reproduce/<dataset>/<Dataset>_<model>/
```

Each run directory contains:

- `results.csv` — per-epoch `mAP50`, `mAP50-95`, and `box/cls/dfl/moe_loss` metrics for both training and validation.
- `results.png` — the corresponding metric curves.
- `weights/best.pt` and `weights/last.pt` — the best and latest model checkpoints.
- `args.yaml` — the exact resolved training configuration.

The dataset-level summary file:

```
runs/reproduce/<dataset>/summary.csv
```

aggregates the final metrics for both models, including a `dense_eval` column that records whether `--no-sparse-eval` was applied.

**Should you have any questions or doubts, feel free to make a comment or contact: rlici@connect.ust.hk**
