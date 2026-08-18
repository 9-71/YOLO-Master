# Issue #49: VisDrone & SKU-110K Reproduction on RTX 5070 (Consumer GPU)

Related to [#49](https://github.com/Tencent/YOLO-Master/issues/49).

本目录记录在 VisDrone2019-DET 与 SKU-110K 上从零训练 YOLO-Master-v0.1-N 与
YOLO-Master-EsMoE-N 的个人复现实验。四次运行均完成 100 epochs，并通过公开 W&B
项目和本地日志保留逐 epoch 指标（mAP50 / mAP50-95 / box_loss / cls_loss / moe_loss）。

> 本复现的差异化价值：在**消费级显卡 NVIDIA GeForce RTX 5070 (12GB)** 上完成，
> 与仓库中其他贡献者使用的 A100 / H200 / A40 / RTX 4090 D 形成对比，
> 验证了该训练流水线在消费级硬件上的可行性。

## 交付文件

| 文件 | 内容 |
| --- | --- |
| `README.md` | 本文件：配置、命令、结果对比、已知问题 |
| `issue49_L1678418568_visdrone.md` / `.csv` | VisDrone 双模型结果 |
| `issue49_L1678418568_sku110k.md` / `.csv` | SKU-110K 双模型结果 |
| `training_logs/` | 四次运行的完整原始日志 |

公开 W&B 项目：https://wandb.ai/lwb060819-guangdong-university-of-technology/yolo-master-reproduce-rtx5070

## 实验配置

| 项目 | 配置 |
| --- | --- |
| 数据集 | VisDrone2019-DET（train 6,471 / val 548）、SKU-110K（train 8,219 / val 588） |
| 输入尺寸 | 640 |
| Epochs | 100 |
| Batch size | 16 |
| Workers | 4（Windows 平台） |
| 缓存 | `--cache disk`（缓解 Windows 无 workers 时的数据加载瓶颈） |
| 初始化 | `pretrained=False`，`lora_r=0` |
| 随机种子 | 42，`deterministic=True` |
| 设备 | 单卡 NVIDIA GeForce RTX 5070，12,227 MiB（sm_120 / Blackwell） |
| 软件 | Python 3.11.0，PyTorch 2.11.0+cu128，仓库内 ultralytics 8.4.101 |
| AMP | 开启（默认） |
| 优化器 | `optimizer=auto`（实际 AdamW） |
| W&B project | `yolo-master-reproduce-rtx5070` |

## 复现命令

```bash
# VisDrone
python scripts/reproduce/reproduce_visdrone.py --model v0.1-N  --epochs 100 --imgsz 640 --batch 16 --workers 4 --cache disk --wandb --wandb-project yolo-master-reproduce-rtx5070
python scripts/reproduce/reproduce_visdrone.py --model EsMoE-N --epochs 100 --imgsz 640 --batch 16 --workers 4 --cache disk --no-sparse-eval --wandb --wandb-project yolo-master-reproduce-rtx5070

# SKU-110K
python scripts/reproduce/reproduce_sku110k.py --model v0.1-N  --epochs 100 --imgsz 640 --batch 16 --workers 4 --cache disk --wandb --wandb-project yolo-master-reproduce-rtx5070
python scripts/reproduce/reproduce_sku110k.py --model EsMoE-N --epochs 100 --imgsz 640 --batch 16 --workers 4 --cache disk --no-sparse-eval --wandb --wandb-project yolo-master-reproduce-rtx5070
```

## 结果对比

### VisDrone

| 模型 | 评估 | Best epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v0.1-N | standard | 88 | 0.43052 | 0.33626 | 0.31118 | 0.17468 |
| EsMoE-N | dense | 96 | 0.42954 | 0.34056 | 0.31312 | 0.17660 |

### SKU-110K

| 模型 | 评估 | Best epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v0.1-N | standard | 90 | 0.90561 | 0.83158 | 0.88171 | 0.54141 |
| EsMoE-N | dense | 86 | 0.90601 | 0.83560 | 0.88434 | 0.54535 |

**要点**：EsMoE-N 以 2.8M 参数（v0.1-N 的 37%），在两个数据集上均取得更高 mAP50。

## 已知问题与解决方案（Windows 特有）

1. **Windows 数据加载瓶颈**：`workers=0`（脚本为防 Windows multiprocessing 死锁的默认值）
   导致 VisDrone 大图（2000x1500）resize 极慢（~8s/it）。解决方案：`--cache disk`
   将处理后的图缓存到磁盘，配合 `--workers 4` 提速至 ~0.24s/it（约 30 倍）。
   `--cache ram` 在 32GB 内存机器上会因 spawn worker 复制缓存导致 MemoryError，
   故优先使用 `--cache disk`。

2. **EsMoE-N 验证 mAP 塌缩**：与仓库已知问题一致——ES_MOE 默认
   `use_sparse_inference=True` 使验证走稀疏前向，输出幅度被压扁导致 mAP 塌缩。
   使用 `--no-sparse-eval` 启用密集验证（train==eval）。

3. **cu128 PyTorch 安装**：RTX 5070 (sm_120) 需要 cu128 构建；PyTorch 2.13.0 在
   cu128 源中尚无 GPU 版本，需显式安装 `torch==2.11.0+cu128`（Blackwell 支持完好）。
   且 pip 需加 `--extra-index-url https://pypi.org/simple` 否则构建依赖（flit_core）
   解析失败。

4. **单随机种子**：仅 seed 42，严谨比较应多种子取均值。

## Checklist

- [ ] VisDrone v0.1-N 完成 100 epochs
- [ ] VisDrone EsMoE-N 完成 100 epochs（dense eval）
- [ ] SKU-110K v0.1-N 完成 100 epochs
- [ ] SKU-110K EsMoE-N 完成 100 epochs（dense eval）
- [ ] 每 epoch 记录 mAP50、mAP50-95、box_loss、cls_loss、moe_loss（W&B + 本地日志）
- [ ] 提供公开 W&B URL、结果 CSV、完整训练日志
- [ ] 使用仓库标准 `reproduce_visdrone.py` / `reproduce_sku110k.py`
