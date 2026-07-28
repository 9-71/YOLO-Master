# 犀牛鸟任务：VisDrone & SKU-110K 复现结果

本次复现使用 YOLO-Master-v26.02 发布的 `YOLO-Master-v0.1-N.pt` 与
`YOLO-Master-EsMoE-N.pt`，以及项目内置的数据集配置：

- `ultralytics/cfg/datasets/VisDrone.yaml`
- `ultralytics/cfg/datasets/SKU-110K.yaml`

四组训练均已完成 120 轮。SKU-110K EsMoE-N 的第 1--103 轮在旧节点完成，因 GPU 2
驱动故障中断后迁移到 RTX 5090 节点恢复第 104--120 轮。恢复完成后已将两段原始
`results.csv` 合并为完整 120 轮记录，并对 epoch 120 的 `last.pt` 做了独立验证。
所有运行均保存了每轮 `mAP50`、`mAP50-95`、`box_loss`、`cls_loss` 和 `moe_loss`。
本报告只陈述实际日志结果。

## 新增文件

- `scripts/reproduce/reproduce_visdrone.py`：支持 `--model v01` 和 `--model moe`。
- `scripts/reproduce/reproduce_sku110k.py`：支持 `--model v01` 和 `--model moe`。
- `scripts/reproduce/_reproduce_common.py`：共享的训练、离线 W&B、逐 epoch 汇总与 Windows `workers=0` 兼容逻辑。
- `README_reproduce.md`：本报告。

## 环境

- 远端硬件：6 x NVIDIA GeForce RTX 2080 Ti，单卡显存 11 GB；本次使用 GPU 0、1、2。
- NVIDIA Driver：580.126.09。
- Python：3.8.10。
- PyTorch：2.1.2+cu121。
- Ultralytics：8.3.240。
- 训练设置：`batch=4`、`workers=4`、`patience=0`、W&B offline。

SKU-110K EsMoE-N 的恢复阶段在新节点执行：NVIDIA GeForce RTX 5090（32 GB）、
Driver 580.76.05、Python 3.12.3、PyTorch 2.11.0+cu128，训练参数保持
`imgsz=640`、`batch=4`、`workers=4`、`patience=0`。

官方发布权重的参数量为：YOLO-Master-v0.1-N `7.555M`，
YOLO-Master-EsMoE-N `2.694M`。数据集类别数会改变检测头，因此训练时模型摘要的参数量可能略有差异。

## 数据集下载

在仓库根目录运行。YAML 内置下载与标注转换逻辑：

```bash
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('VisDrone.yaml', autodownload=True)"
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('SKU-110K.yaml', autodownload=True)"
```

数据集根目录不在默认位置时，先配置：

```bash
yolo settings datasets_dir=/path/to/datasets
```

## 训练命令

以下是本次使用的等价训练命令。脚本默认从 YOLO-Master-v26.02 Release 获取对应官方权重；离线服务器可通过 `--v01-model` 或 `--esmoe-model` 指向已下载的 `.pt` 文件。

```bash
# VisDrone 基线，imgsz=800
python scripts/reproduce/reproduce_visdrone.py --model v01 --epochs 120 --imgsz 800 --batch 4 --device 0 --workers 4 --patience 0 --wandb-mode offline

# VisDrone EsMoE-N，imgsz=800
python scripts/reproduce/reproduce_visdrone.py --model moe --epochs 120 --imgsz 800 --batch 4 --device 1 --workers 4 --patience 0 --wandb-mode offline

# SKU-110K 基线，imgsz=640
python scripts/reproduce/reproduce_sku110k.py --model v01 --epochs 120 --imgsz 640 --batch 4 --device 0 --workers 4 --patience 0 --wandb-mode offline

# SKU-110K EsMoE-N，imgsz=640
python scripts/reproduce/reproduce_sku110k.py --model moe --epochs 120 --imgsz 640 --batch 4 --device 2 --workers 4 --patience 0 --wandb-mode offline
```

## 复现结果

原 EsMoE-N 的近零 mAP 已定位为推理路径错误：当 `top_k` 等于专家数时，
验证仍错误执行了专家裁剪。修复后 VisDrone EsMoE-N 已完成 120 轮，验证
指标恢复正常。下表使用每个 `results.csv` 最后一行的实际记录。

| 数据集 | 模型 | 输入分辨率 | 训练轮数 | mAP50 | mAP50-95 | Release 参数量 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| VisDrone | YOLO-Master v0.1-N | 800 | 120 | 0.42966 | 0.26027 | 7.555M | 有效 |
| VisDrone | YOLO-Master EsMoE-N | 800 | 120 | 0.42482 | 0.25704 | 2.694M | 有效，已修复推理路径 |
| SKU-110K | YOLO-Master v0.1-N | 640 | 120 | 0.91243 | 0.59063 | 7.555M | 有效 |
| SKU-110K | YOLO-Master EsMoE-N | 640 | 120 | 0.91344 | 0.59013 | 2.694M | 有效；103 轮断点恢复后完成 |

为便于审计，各运行最后一个已完成 epoch 的原始 `results.csv` 记录如下。

| 数据集 | 模型 | mAP50 | mAP50-95 | box_loss | cls_loss | moe_loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VisDrone | v0.1-N | 0.42966 | 0.26027 | 1.17997 | 0.76615 | 0.00000 |
| VisDrone | EsMoE-N | 0.42482 | 0.25704 | 1.17473 | 0.74958 | 0.00000 |
| SKU-110K | v0.1-N | 0.91243 | 0.59063 | 1.27920 | 0.53971 | 0.00000 |
| SKU-110K | EsMoE-N | 0.91344 | 0.59013 | 1.27205 | 0.53164 | 0.00000 |

SKU-110K EsMoE-N epoch 120 `last.pt` 的独立验证结果为 mAP50 `0.91362`、
mAP50-95 `0.59125`，与训练末轮记录一致。

## 训练日志

完整训练日志、逐 epoch CSV、训练参数、状态文件、图表及离线 W&B 运行包已从远端传回：

- `artifacts/remote_runs/completed_20260724/m26_yolo_master_results_20260724.tar.gz`
  - SHA-256：`4c3ab7d9b02261c190b70493c6092186ad90f76813b30a2904bd53724f72a182`
- `artifacts/remote_runs/completed_20260724/m26_yolo_master_final_validations_20260724.tar.gz`
  - SHA-256：`fe839e236583a192410d38cc9626861cbf87c6046ea454fb1d8088ab99cc3a2d`
- `artifacts/remote_runs/corrected_20260724/visdrone_esmoe_corrected_20260724.tar.gz`
  - SHA-256：`e62c2e5d7769aca83b595fc77164eebcae51d232c508d6a5017131d06fd6bb1f`
- `artifacts/remote_runs/new_server_20260725/yolo_master_sku110k_migration_status_20260725.tar.gz`
  - 新 RTX 5090 节点的恢复点、前 103 epoch `results.csv`、迁移/恢复脚本及环境安装日志。
  - SHA-256：`5496d9cef7680b826d7e207352ceee9c713271ddf207737b90e9dfdd24827ff5`
- `artifacts/remote_runs/final_20260725/yolo_master_sku110k_esmoe_epoch120_20260725.tar.gz`
  - SKU-110K EsMoE-N 完整 120 epoch CSV、epoch 120 权重、恢复日志、独立验证和离线 W&B 运行。
  - SHA-256：`8e2f9d9d9cd71af92eb72fe81440b16efa22e6e67a9eab827c9e76e1c52573d1`

本机的 `离线日志/` 含可用的 `offline-run-*` 目录。解压或直接进入对应目录后执行：

```bash
wandb sync /path/to/offline-run-*/
```

第二个归档中的 `validation_metrics.json` 为上表的独立验证原始输出。离线 W&B 包可解压后执行：

```bash
wandb sync /path/to/offline-run-*/
```

本次未创建公开 W&B 项目，因此没有可公开访问的 W&B URL，也没有虚构网盘链接。

## 已知问题与解决方案

- EsMoE-N 在 `top_k == num_experts` 时会错误进入带阈值的稀疏裁剪分支，造成验证近零 mAP。已在 `ES_MOE._sparse_forward()` 修复为与训练一致的稠密计算；VisDrone 复跑已完成并验证恢复正常。
- SKU-110K EsMoE-N 在第 103 轮触发 GPU 2 `CUDA unspecified launch failure`。旧节点随后使 PyTorch 无法初始化任何 CUDA 设备。恢复点迁移到 RTX 5090（32 GB，Driver 580.76.05，PyTorch 2.11.0+cu128）后，使用 `rsync --partial --append-verify` 完整校验数据集，并以 `--resume --device 0` 成功完成剩余 17 轮。
- 原始验证器缺少 `LOCAL_RANK`、`torch_distributed_zero_first` 导入，并错误调用不存在的 `convert_ndjson_to_yolo_if_needed`。已在 `ultralytics/engine/validator.py` 修复：仅对 `.ndjson` 调用现有转换器。四个 `best.pt` 已独立验证成功。
- SKU-110K 部分 JPEG 有截断或损坏警告。Ultralytics 会修复或跳过无法读取的样本，日志中保留了这些记录。
- 两个训练实例首次同时建立 `labels.cache` 时会竞争同一缓存文件。先完成一次缓存建立，或顺序启动首轮训练后再并行，可避免 `UnpicklingError`。
- Windows 环境应使用 `--workers 0`，否则多进程数据加载可能发生 I/O 死锁。
- 显存不足时先降低 `--batch`；本次 11 GB 显存使用 `batch=4`。
