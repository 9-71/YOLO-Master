#!/usr/bin/env python3
"""Self-contained DDP-only trainer for YOLO-Master baselines — v0.1-N / EsMoE-N × {VisDrone, SKU-110K, AI-TOD-v2}.

MULTI-GPU ONLY, and STANDALONE: it needs no changes to the ultralytics library. It relaunches itself
under ``torchrun`` (one process per GPU) so this script — and the callbacks it attaches — execute in
*every* rank. (Ultralytics' built-in ``device=0,1`` auto-spawn regenerates the trainer from its args in a
fresh subprocess that never runs this script, so the callbacks would be lost there; torchrun is what keeps
this script library-free.) ``--device`` must list ≥ 2 GPUs; ``--batch`` is the TOTAL batch, split evenly.

Two things it handles in-script (both run in every rank):
  * **ES_MOE dense eval** (``--no-sparse-eval``) — flips ``use_sparse_inference=False`` on the model + EMA
    so evaluation matches training (ES_MOE's default sparse-eval path otherwise collapses mAP).
  * **ContiguousDistributedSampler guard** — when there are fewer whole batches than ranks (small val set
    / large batch / many GPUs, e.g. 548 val images ÷ a 256 val-batch across 4 GPUs = 3 batches for 4
    ranks), the sampler's per-batch chunking gives a trailing rank ``start_idx > end_idx`` → a negative
    ``__len__`` (``ValueError: __len__() should return >= 0``). We degenerate to per-sample distribution in
    that case, which is exactly the sampler's own ``batch_size >= total_size`` fallback.

Everything else DDP needs — ``broadcast_buffers=False``, gating the balance-loss collective out of eval
(``should_reduce_ddp``), and moving/skipping CPU EMA buffers at validation — is already handled by the
trainer and MoE modules, so this script does not touch it.

Recipe (matches the per-dataset scripts): from-scratch, ``lora_r=0``, ``optimizer=auto`` → SGD@0.01,
``deterministic``, ``seed 42``, ``patience 0``. Supports ``--lr0`` large-batch linear scaling, resume, and
per-dataset default imgsz (640 / 800). ES_MOE keeps the as-shipped sparse eval unless ``--no-sparse-eval``.

Usage (≥ 2 GPUs; ``--workers 0`` recommended on the Python-3.14 stack — see the reproduce README):
    python scripts/reproduce/reproduce_ddp.py --dataset VisDrone  --model EsMoE-N --device 0,1,2,3 --batch 128 --no-sparse-eval --workers 0
    python scripts/reproduce/reproduce_ddp.py --dataset SKU-110K  --model v0.1-N  --device 0,1 --batch 64 --workers 0
    python scripts/reproduce/reproduce_ddp.py --dataset AI-TOD-v2 --model v0.1-N  --device 0,1,2,3 --batch 128 --workers 0
    python scripts/reproduce/reproduce_ddp.py --dataset VisDrone  --model all     --device 0,1 --dry-run

Single-node convention (GPUs 0..N-1). To pin specific GPUs, set CUDA_VISIBLE_DEVICES and pass --device 0,1.
"""
from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# Registry — the two shared nano baselines × the three datasets               #
# --------------------------------------------------------------------------- #
DATASETS = {
    "VisDrone":  {"data": "VisDrone.yaml",  "imgsz": 640, "nc": 10, "project": "runs/reproduce/visdrone"},
    "SKU-110K":  {"data": "SKU-110K.yaml",  "imgsz": 640, "nc": 1,  "project": "runs/reproduce/sku110k"},
    "AI-TOD-v2": {"data": "AI-TOD-v2.yaml", "imgsz": 800, "nc": 8,  "project": "runs/reproduce/aitodv2"},
}
# esmoe=True -> contains ES_MOE blocks (sparse eval collapses mAP; --no-sparse-eval corrects it).
MODELS = {
    "v0.1-N":     {"cfg": "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml",         "esmoe": False},
    "EsMoE-N":    {"cfg": "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml",            "esmoe": True},
    "EsMoE-P2-N": {"cfg": "ultralytics/cfg/models/master/v0/det/yolo-master-n-p2.yaml",         "esmoe": True},
    "v0.1-P2-N":  {"cfg": "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-p2.yaml",       "esmoe": False},
    "UoMoE-N":    {"cfg": "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-uomoe.yaml",    "esmoe": False},
    "UoMoE-P2-N": {"cfg": "ultralytics/cfg/models/master/v0_1/det/yolo-master-n-uomoe-p2.yaml", "esmoe": False},
}

# --------------------------------------------------------------------------- #
# Distributed context (populated by torchrun before the process starts)       #
# --------------------------------------------------------------------------- #
RANK = int(os.environ.get("RANK", "-1"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
UNDER_TORCHRUN = "LOCAL_RANK" in os.environ


def is_main() -> bool:
    return RANK in (-1, 0)


def _gpu_count(device: str) -> int:
    if isinstance(device, str) and device.strip() not in ("", "cpu", "mps"):
        return len([d for d in device.split(",") if d.strip() != ""])
    return 0


# --------------------------------------------------------------------------- #
# In-script fixes — run in every torchrun rank (no library edits needed)      #
# --------------------------------------------------------------------------- #
def _cb_es_moe_dense_eval(trainer):
    """Force ES_MOE onto the dense forward for eval (``use_sparse_inference=False``) on model + EMA.

    ES_MOE's default eval path prunes to ~1 unnormalised expert while training blends all experts, which
    collapses validation mAP; flipping the flag makes eval match training. Registered only with
    ``--no-sparse-eval``; runs on ``on_pretrain_routine_end`` and ``on_train_start`` in every rank.
    """
    try:
        from ultralytics.nn.modules.moe.modules import ES_MOE
    except Exception:  # noqa: BLE001
        return
    for tgt in (getattr(trainer, "model", None), getattr(getattr(trainer, "ema", None), "ema", None)):
        if tgt is None:
            continue
        for mod in tgt.modules():
            if isinstance(mod, ES_MOE):
                mod.use_sparse_inference = False


def patch_contiguous_sampler() -> None:
    """Fix ContiguousDistributedSampler when there are fewer whole batches than ranks.

    The rect-mode DDP sampler hands whole BATCHES to ranks. When ``ceil(len(dataset) / batch) <
    num_replicas`` — e.g. the 548-image VisDrone val set with a val batch of 256 (= train_batch/rank × 2)
    across 4 GPUs gives 3 batches for 4 ranks — the trailing rank(s) get ``start_idx > end_idx``, so
    ``__len__`` returns negative → ``ValueError: __len__() should return >= 0`` at the first-epoch eval.
    Degenerate to per-sample distribution (batch_size=1), which is exactly the class's own
    ``batch_size >= total_size`` fallback; every rank then gets an even, non-negative share. The
    ``__init__`` wrapper is signature-agnostic (passes *args/**kwargs through) and runs in every rank.
    """
    try:
        from ultralytics.data.build import ContiguousDistributedSampler as _S

        orig_init = _S.__init__
        if getattr(orig_init, "_smallset_guarded", False):
            return

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            if getattr(self, "num_batches", 1) < getattr(self, "num_replicas", 1):
                self.batch_size = 1  # round-robin: num_batches == total_size >= num_replicas
                self.num_batches = self.total_size

        __init__._smallset_guarded = True
        _S.__init__ = __init__
    except Exception:  # noqa: BLE001  -- absent/renamed in this build -> leave stock behaviour
        pass


# --------------------------------------------------------------------------- #
# torchrun self-relaunch + dataset pre-stage + group teardown                 #
# --------------------------------------------------------------------------- #
def _prestage_dataset(data: str) -> None:
    """Download/prepare the dataset once in this parent process, before spawning the torchrun ranks, so
    the ranks don't race on a first-time download/convert (which can corrupt the label build)."""
    try:
        from ultralytics.data.utils import check_det_dataset

        print(f"[reproduce_ddp] pre-staging dataset '{data}' once before DDP launch...", flush=True)
        check_det_dataset(data, autodownload=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[reproduce_ddp][WARN] dataset pre-stage failed ({type(exc).__name__}: {exc}); "
              f"ranks will each run their own check.", flush=True)


def _free_port() -> int:
    """Pick a free localhost TCP port (avoids torchrun's default 29500 colliding with a stale/other run)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _reexec_under_torchrun(args: argparse.Namespace, data: str, n: int) -> None:
    """Pre-stage the dataset, then replace this process with a torchrun launch of the same command."""
    _prestage_dataset(data)
    cmd = [sys.executable, "-m", "torch.distributed.run",
           f"--nproc_per_node={n}", f"--master_port={_free_port()}",  # dynamic port -> no EADDRINUSE
           os.path.abspath(sys.argv[0]), *sys.argv[1:]]
    print(f"[reproduce_ddp] launching {n}-way DDP via torchrun:\n    {' '.join(cmd)}", flush=True)
    os.execv(sys.executable, cmd)  # replaces this process; does not return


def _teardown_ddp() -> None:
    """Destroy the process group between models (--model all) so the next model can re-init it."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Metrics / resume helpers                                                     #
# --------------------------------------------------------------------------- #
def _last_epoch(results_csv: Path) -> int | None:
    if not results_csv.exists():
        return None
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    try:
        return int(float({k.strip(): v for k, v in rows[-1].items()}["epoch"]))
    except (KeyError, ValueError):
        return None


def _final_metrics(results_csv: Path) -> str:
    if not results_csv.exists():
        return "(no results.csv)"
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "(empty results.csv)"
    r = {k.strip(): v for k, v in rows[-1].items()}
    return f"epoch={r.get('epoch','?')} mAP50={r.get('metrics/mAP50(B)','?')} mAP50-95={r.get('metrics/mAP50-95(B)','?')}"


def build_opt(args: argparse.Namespace) -> dict:
    """Default 'auto' -> SGD@0.01 (auto IGNORES lr0). --lr0 forces SGD and pins the auto recipe's
    momentum/warmup so only the LR differs (for large-batch linear scaling)."""
    opt = {"optimizer": args.optimizer}
    if args.lr0 is not None:
        opt["lr0"] = args.lr0
        if args.optimizer == "auto":
            opt["optimizer"] = "SGD"
        opt["momentum"] = 0.9
        opt["warmup_bias_lr"] = 0.0
    return opt


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train_one(args: argparse.Namespace, ds: dict, model_name: str, project: Path) -> dict:
    from ultralytics import YOLO

    ms = MODELS[model_name]
    run_name = f"{args.dataset}_{model_name}"
    run_dir = project / run_name
    last_pt = run_dir / "weights" / "last.pt"
    best_pt = run_dir / "weights" / "best.pt"
    done = _last_epoch(run_dir / "results.csv")

    if best_pt.exists() and done is not None and done + 1 >= args.epochs:
        if is_main():
            print(f"[skip] {run_name}: already complete at epoch {done}", flush=True)
        return {"model": model_name, "status": "skipped"}

    dense_eval = ms["esmoe"] and not args.sparse_eval
    imgsz = args.imgsz or ds["imgsz"]
    opt = build_opt(args)
    # torchrun single-node convention: ultralytics derives world_size from device, so span 0..N-1.
    device = ",".join(str(i) for i in range(WORLD_SIZE)) if (UNDER_TORCHRUN and WORLD_SIZE > 1) else args.device

    if last_pt.exists() and done is not None:
        if is_main():
            print(f"[resume] {run_name}: {last_pt} epoch={done} -> {args.epochs}", flush=True)
        model = YOLO(str(last_pt))
        resume = True
    else:
        if is_main():
            print(f"[train] {run_name}: cfg={ms['cfg']} data={ds['data']} imgsz={imgsz} batch={args.batch} "
                  f"ddp={WORLD_SIZE}x dense_eval={dense_eval} optimizer={opt['optimizer']}"
                  + (f" lr0={args.lr0}" if args.lr0 is not None else ""), flush=True)
        model = YOLO(str(ROOT / ms["cfg"]))
        resume = False

    if dense_eval:  # ES_MOE --no-sparse-eval; runs in every rank
        model.add_callback("on_pretrain_routine_end", _cb_es_moe_dense_eval)
        model.add_callback("on_train_start", _cb_es_moe_dense_eval)

    start = time.time()
    model.train(
        data=ds["data"],
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=str(project),
        name=run_name,
        exist_ok=True,
        pretrained=False,
        lora_r=0,          # disable default.yaml lora_r (would silently LoRA-fy the run)
        **opt,             # optimizer (auto->SGD@0.01) + optional --lr0 override
        val=True,
        plots=True,
        cache=args.cache,
        patience=args.patience,
        amp=args.amp,
        resume=resume,
        verbose=args.verbose,
    )
    if is_main():
        print(f"[done] {run_name}: {_final_metrics(run_dir / 'results.csv')}  ({time.time() - start:.1f}s)", flush=True)
    return {"model": model_name, "status": "resumed" if resume else "ok"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--model", required=True, choices=list(MODELS) + ["all"],
                   help="A model name, or 'all' to train every model in sequence.")
    p.add_argument("--device", required=True,
                   help="Comma-separated GPU ids, >=2 (e.g. '0,1'). REQUIRED: this script is DDP-only.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=64, help="TOTAL batch, split evenly across GPUs.")
    p.add_argument("--imgsz", type=int, default=None, help="Override the per-dataset default (640 / 800).")
    p.add_argument("--optimizer", default="auto",
                   help="Default 'auto' -> SGD@0.01 (IGNORES --lr0). --lr0 forces SGD.")
    p.add_argument("--lr0", type=float, default=None,
                   help="LR override for large-batch scaling (e.g. --lr0 0.04 at batch 256); forces SGD "
                        "and pins momentum=0.9 / warmup_bias_lr=0.")
    p.add_argument("--workers", type=int, default=16, help="Dataloader workers PER GPU (see README note).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cache", nargs="?", const="ram", default=False,
                   help="'--cache'/'--cache ram' = RAM, '--cache disk' = on-disk .npy, omit to disable.")
    p.add_argument("--sparse-eval", action=argparse.BooleanOptionalAction, default=True,
                   help="ES_MOE sparse eval. Default True = as-shipped (collapses mAP). --no-sparse-eval "
                        "opts into corrected dense eval. No-op for v0.1-N.")
    p.add_argument("--project", default=None, help="Override the per-dataset run project directory.")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled",
                   help="Sets WANDB_MODE (inherited by workers) for Ultralytics' native W&B. Default off; "
                        "per-epoch metrics always go to results.csv.")
    p.add_argument("--check-build", action="store_true", help="Instantiate the selected model(s) and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    ds = DATASETS[args.dataset]
    models = list(MODELS) if args.model == "all" else [args.model]
    project = Path(args.project) if args.project else (ROOT / ds["project"])
    os.environ["WANDB_MODE"] = args.wandb_mode  # inherited by torchrun ranks

    # --- utility modes (no DDP) ---
    if args.check_build:
        from ultralytics.nn.tasks import DetectionModel
        for m in models:
            mdl = DetectionModel(str(ROOT / MODELS[m]["cfg"]), ch=3, nc=ds["nc"], verbose=False)
            print(f"[build-ok] {m}: {sum(p.numel() for p in mdl.parameters()) / 1e6:.3f}M  ({MODELS[m]['cfg']})")
        return 0

    # --- DDP-only enforcement (single GPU / CPU refused) ---
    n_gpu = _gpu_count(args.device)
    if n_gpu < 2:
        print(f"[error] This script is DDP-ONLY. --device={args.device!r} implies {n_gpu} GPU(s). Pass at "
              f"least two (e.g. --device 0,1). For single-GPU use scripts/reproduce/reproduce_*.py.",
              file=sys.stderr)
        return 2

    if is_main():
        imgsz = args.imgsz or ds["imgsz"]
        print(f"[reproduce_ddp] dataset={args.dataset} data={ds['data']} imgsz={imgsz} project={project}\n"
              f"                models={models} device={args.device} ({n_gpu} GPUs) batch={args.batch}(total) "
              f"epochs={args.epochs} wandb={args.wandb_mode}", flush=True)
        for m in models:
            note = "ES_MOE dense-eval" if (MODELS[m]["esmoe"] and not args.sparse_eval) else \
                   ("ES_MOE sparse-eval (as-shipped)" if MODELS[m]["esmoe"] else "no ES_MOE")
            print(f"  - {m:<8} {note}")
        if args.batch % n_gpu:
            print(f"[warn] batch={args.batch} not divisible by {n_gpu} GPUs; each rank gets "
                  f"{args.batch // n_gpu} (remainder dropped).", flush=True)

    if args.dry_run:
        return 0

    # --- relaunch under torchrun so this script (and its callbacks) run in every rank ---
    if not UNDER_TORCHRUN:
        _reexec_under_torchrun(args, ds["data"], n_gpu)  # pre-stages, then os.execv -> does not return

    # --- now running as a torchrun rank ---
    patch_contiguous_sampler()   # small val set / large batch / many ranks -> negative sampler __len__
    project.mkdir(parents=True, exist_ok=True)
    statuses = []
    for m in models:
        try:
            statuses.append(train_one(args, ds, m, project))
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {m} (rank {RANK}): {type(exc).__name__}: {exc}", flush=True)
            if is_main():
                import traceback
                traceback.print_exc()
            statuses.append({"model": m, "status": "failed", "error": str(exc)})
        finally:
            _teardown_ddp()  # let the next model re-init the process group (all ranks call this)

    if is_main():
        print(f"\n[reproduce_ddp] DONE — {args.dataset}")
        for st in statuses:
            print("  ", st)
    ok = {"ok", "resumed", "skipped"}
    return 0 if all(s.get("status") in ok for s in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
