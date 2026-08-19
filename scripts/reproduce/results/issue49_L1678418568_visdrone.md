# Issue #49 VisDrone reproduction results

Git commit: `L1678418568-reproduce`

Configuration: `epochs=100`, `imgsz=640`, `batch=16`, `workers=4`, `seed=42`, `cache=disk`

GPU: NVIDIA GeForce RTX 5070 (12GB, consumer)

W&B project: `yolo-master-reproduce-rtx5070`


| Model | Evaluation | Best epoch | Precision | Recall | mAP50 | mAP50-95 | W&B |
|---|---:|---:|---:|---:|---:|---:|---|
| v0.1-N | standard | 88 | 0.43052 | 0.33626 | 0.31118 | 0.17468 | https://wandb.ai/lwb060819-guangdong-university-of-technology/yolo-master-reproduce-rtx5070/runs/wgtgtjs4 |
| EsMoE-N | dense | 96 | 0.42954 | 0.34056 | 0.31312 | 0.17660 | https://wandb.ai/lwb060819-guangdong-university-of-technology/yolo-master-reproduce-rtx5070/runs/tzl7ud7j |

Notes: EsMoE-N uses corrected dense validation with `--no-sparse-eval`.

The routed auxiliary loss is also logged under the canonical name `moe_loss`.
