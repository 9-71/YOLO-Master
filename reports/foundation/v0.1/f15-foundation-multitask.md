# F15 — MultiTask Foundation representation transfer

## 实现范围

- 新增显式 `foundation_multitask`（及兼容别名 `foundation_multitask_enabled`）配置。
- 仅允许 `task=multitask`，并要求模型至少有两个 active tasks；未启用时保持 Foundation exact no-op。
- 复用 F10/F13/F14 的共享 P4、语义和 routing KD，不把 TaskRouter 改名或改造成全局 selector。
- 记录固定九项 `MultiTaskLoss` 的任务级快照、任务损失不均衡、Foundation/Task 比值、TaskRouter entropy/usage，以及负迁移风险标记。
- checkpoint metadata、resume projector 重建和 export strip 均保留 F15 任务契约；teacher 仍 training-only。

## 验证

离线 synthetic contract：

```text
pytest -q tests/test_foundation_multitask.py
5 passed
```

覆盖共享 P4 projector 梯度、两个任务同时有正 loss、teacher 不进入 checkpoint、resume/export 边界、TaskRouter 语义保留和 disabled no-op。

完整 COCO 多任务 benchmark 需要本地 COCO 2017 detection/instance/pose annotations；本阶段不将 loss/routing 观测误报为 accuracy 提升。只有至少两个任务在同一运行中同时获得正监督证据时，才满足 representation-transfer gate。

## 配对效果验证

已加入 `scripts/foundation_f15_paired_benchmark.py`。它用相同初始化、相同 synthetic detect/segment/pose batch 和相同 SGD 设置，对比 baseline 与真实本地 DINOv3 Foundation 分支，输出 JSON 中的 task loss、Foundation KD loss、P4 梯度、TaskRouter entropy 和每步耗时。

最新一次 6-step CPU 运行结果（DINOv3 `Tooony133/dinov3-vits16-pretrain-lvd1689m`，本地缓存权重）：

| 指标 | Baseline | Foundation |
|---|---:|---:|
| 首步 task loss | 50.118668 | 50.118668 |
| 末步 task loss | 48.148949 | 49.715363 |
| 首末 task loss 变化 | -1.969719 | -0.403305 |
| 平均 step 时间 | 0.1031s | 0.1327s |
| Foundation KD（首步→末步） | 0 | 0.188721 → 0.188096 |
| 同步正监督任务数 | — | 3 |
| TaskRouter entropy | — | 约 0.7286 |

结论：Foundation 分支满足“链路有效”证据（每步 KD 非零、P4 最小梯度范数约 968.92、三个任务同步有监督、TaskRouter entropy 稳定），但训练开销约增加 28.8%。该次 synthetic run 的 task-loss 下降幅度明显小于 baseline（-0.403305 对 -1.969719），因此不能声称当前 Foundation 配置改善了多任务效果；这只证明训练信号和 representation-transfer gate 工作正常，并提示后续真实消融应覆盖更低 KD 权重与 teacher 计算优化。真实 AP/mask AP/pose AP 仍需使用 [coco-multitask-unified.yaml](/Users/gatilin/PycharmProjects/YOLO-Master-v260810-latest/ultralytics/cfg/datasets/coco-multitask-unified.yaml) 完成正式对照实验。


## 真实 COCO 训练信号矩阵

在完成数据预检和受控 smoke 后，使用同一统一 manifest、`fraction=0.0005`、`imgsz=64`、`batch=2`、`epochs=1`、CPU，运行了 3 个 seed（`20260813/20260814/20260815`）与两个 Foundation 权重（`0.01/0.05`）的 baseline/Foundation 配对矩阵。每个权重和 seed 都使用相同的数据顺序与初始化；完整逐运行记录见 [f15-real-coco-matrix.json](/Users/gatilin/PycharmProjects/YOLO-Master-v260810-latest/reports/foundation/v0.1/f15-real-coco-matrix.json)。

| Foundation 权重 | 配对数 | Foundation loss 均值 | Foundation/Task 比值均值 | 同步监督任务均值 | representation-transfer gate | TaskRouter entropy |
|---:|---:|---:|---:|---:|:---:|---:|
| 0.01 | 3 | 0.03449 | 0.000662 | 2.84444 | 3/3 通过 | 0.728575 |
| 0.05 | 3 | 0.17244 | 0.003312 | 2.84444 | 3/3 通过 | 0.728575 |

矩阵中 Foundation 分支的检测、分割、姿态监督损失与对应 baseline 配对保持一致，同时每个运行均产生非零 DINOv3 KD、`representation_transfer_ready=1` 和稳定的 TaskRouter entropy。该实验只验证真实 COCO 的多任务训练信号、teacher 路径和 gate；训练量仅为 1 epoch、`fraction=0.0005`，未运行完整 `val2017`，因此不能据此声称 AP、mask AP、pose AP 或收敛速度提升。矩阵训练时长受 CPU、缓存和系统负载影响，未作为性能结论使用。

从 `foundation-s20260814-w005/weights/last.pt` 读取 checkpoint 时，按 Ultralytics 保存约定从 `ema` 槽位恢复 Foundation wrapper（`model` 槽位为空）。验证结果：metadata 的 `training_only=true`、teacher=`dinov3`、active tasks 为 detect/segment/pose；wrapper state 中没有 teacher model 参数；`strip_foundation_distillation_model()` 返回纯 `MultiTaskModel`，其 state dict 不含 projector/foundation keys。

## Gate 状态

- F15 mechanism gate：通过。Foundation KD 每步非零、共享 P4 梯度非零、三个任务同步有正监督，checkpoint/resume/export 边界测试通过。
- F15 effect gate：受控版本完成。`weight=0.01` 和 `weight=0.05` 均完成 3-seed、3-epoch、完整 val2017 配对；两组权重的 box/mask/pose mAP50-95 配对差值均为 0，未观察到可报告的 AP 改善，`accuracy_claim=false` 保持不变。
- COCO 2017 gate：数据就绪。`/Users/gatilin/MyWork/datasets/coco2017` 的 train/val 图像、instances、polygon segmentation 和 person keypoints 均通过预检。
- Ready for next phase：NO。原因是受控 effect-gate 已完成但训练量仍不足以代表正式 COCO 收敛效果；下一阶段需要更高 fraction、分辨率和 epoch 的资源充足对照实验。

为进入正式 effect gate，已新增 [foundation_f15_real_effect_gate.py](/Users/gatilin/PycharmProjects/YOLO-Master-v260810-latest/scripts/foundation_f15_real_effect_gate.py)。该 runner 只编排现有 Ultralytics 训练入口，不改变模型或训练逻辑，默认生成 3 seed × 2 KD weight 的 baseline/Foundation 配对计划，支持 `--dry-run`、`--resume`、多 epoch、fraction、完整验证和逐运行 JSON 落盘；结果汇总只使用实际存在的 `results.csv` 指标，不对缺失验证数据做插值，也不自动设置 accuracy claim。推荐正式入口：

```bash
python scripts/foundation_f15_real_effect_gate.py \
  --dataset /Users/gatilin/MyWork/datasets/coco2017/unified_multitask_f15/coco2017_mot_multitask.yaml \
  --teacher-model /Users/gatilin/.cache/huggingface/hub/models--Tooony133--dinov3-vits16-pretrain-lvd1689m/snapshots/fc6921f7a0b44d5b33ab4482cfed5443db6ccd81 \
  --seeds 20260813,20260814,20260815 \
  --foundation-loss-weights 0.01,0.05 \
  --epochs 10 --fraction 0.1 --imgsz 256 --batch 2 \
  --output reports/foundation/v0.1/f15-real-coco-effect-gate.json
```

上述命令是资源充足时的正式多 seed 配置；本次先执行了下方受控单 seed smoke，以验证完整训练和 validation 链路。

## 受控真实 COCO effect-gate smoke（已完成）

为获得真实 validation 信号，在单个固定 seed `20260813` 下运行了 baseline/Foundation 配对：统一 COCO manifest、相同模型配置和初始化、`foundation_loss_weight=0.01`、`epochs=3`、`fraction=0.005`、`imgsz=128`、`batch=2`、CPU，并对每个 epoch 执行完整 val2017。原始记录见 [f15-real-coco-effect-gate-smoke.json](/Users/gatilin/PycharmProjects/YOLO-Master-v260810-latest/reports/foundation/v0.1/f15-real-coco-effect-gate-smoke.json)。

| 指标（最后 epoch） | Baseline | Foundation |
|---|---:|---:|
| train box loss | 3.79502 | 3.81190 |
| train segment loss | 23.13260 | 23.15910 |
| train pose loss | 22.19840 | 21.04220 |
| Foundation loss | 0 | 0.0319322 |
| Foundation/Task ratio | — | 0.000622005 |
| 同步监督任务数 | — | 2.88851 |
| TaskRouter entropy | — | 0.728565 |
| val box mAP50-95 | 0 | 0 |
| val mask mAP50-95 | 0 | 0 |
| val pose mAP50-95 | 0 | 0 |
| 训练耗时 | 1578.50 s | 1350.74 s |

该 smoke 确认真实 COCO 三任务训练、DINOv3 KD、TaskRouter 统计和完整 validation 链路均可运行；Foundation 分支的 `foundation_loss` 在训练中持续非零，`representation_transfer_ready=1`，未触发负迁移风险。当前 fraction、分辨率、epoch 和单 seed 仍不足以支持 AP/收敛优势结论；本报告保持 `accuracy_claim=false`。配对验证差异为 box/mask/pose mAP50-95 均为 0，box precision 变化 `-5e-05`，不能解释为效果提升。

### Checkpoint / resume / export 边界

Foundation checkpoint 的 `foundation` 与 `mixture_checkpoint.foundation` 元数据均记录 `training_only=true`、teacher=`dinov3`、active tasks=`detect/segment/pose`、target level=`p4`。加载 `best.pt` 后保存的 student 为纯 `MultiTaskModel`，其 state dict 不含 `teacher`、`projector` 或 `foundation` 参数；训练期 projector/teacher 的恢复由 F07 `rebuild_foundation_distillation_wrapper()` 契约负责，部署和 export 通过 `strip_foundation_distillation_model()` 返回纯 student。

在真实 `last_healthy.pt` 上进一步执行了端到端 resume smoke：追加到第 4 epoch、`fraction=0.0001`、`imgsz=128`、`batch=2`、CPU、关闭 validation，完成 6 个 train batches，最终 Foundation loss=`0.0318401`，无 checkpoint、optimizer、EMA 或 teacher 重建异常。验证过程中发现并修复了 Foundation wrapper 代理 student 的 `_mixture_loss_ema_buf` 未被 resume helper 正确识别的问题：EMA buffer 现在按注册 owner（`student_model`）解析，并兼容嵌套的 `student_model._mixture_loss_ema_buf` checkpoint key；新增回归测试后 checkpoint/DDP/mixture 相关测试为 `19 passed`。

本次 resume 遵循 Ultralytics 原有 run 目录约定，因此把第 4 epoch 追加到了 `runs/multitask/f15-effect-gate-smoke/foundation-s20260813-w0.01/results.csv`；`f15-real-coco-effect-gate-smoke.json` 和上方 effect-gate 表格仍明确对应原始 3 epoch baseline/Foundation 配对，不能把 resume smoke 当成新的 paired effect 结果。

## 三 seed 真实 COCO effect-gate（weight=0.01）

为补齐 seed 稳定性，新增运行了 `20260814` 和 `20260815`，并与已有 `20260813` 的同配置结果合并分析。三组均使用 `epochs=3`、`fraction=0.005`、`imgsz=128`、`batch=2`、CPU、完整 val2017，原始新增记录见 [f15-real-coco-effect-gate-seeds.json](/Users/gatilin/PycharmProjects/YOLO-Master-v260810-latest/reports/foundation/v0.1/f15-real-coco-effect-gate-seeds.json)。

| 指标 | seed 20260813 | seed 20260814 | seed 20260815 | 均值 | 标准差 |
|---|---:|---:|---:|---:|---:|
| Foundation loss | 0.0319322 | 0.0314017 | 0.0318358 | 0.0317232 | 0.0002826 |
| Foundation/Task ratio | 0.0006220 | 0.0006121 | 0.0006146 | 0.0006162 | 0.0000051 |
| 同步监督任务数 | 2.88851 | 2.89189 | 2.90541 | 2.89527 | 0.00894 |
| TaskRouter entropy | 0.728565 | 0.728570 | 0.728567 | 0.728567 | 0.0000025 |

三 seed 的 box/mask/pose `mAP50-95` 配对差值均为 `0.0`（均值和标准差均为 `0.0`）；因此当前训练量下没有观察到可报告的 AP 改善，也没有将其解释为效果提升。Foundation 平均运行时间为 `1441.66 s`，baseline 为 `1543.35 s`，该差异受 CPU、缓存和系统负载影响，不作为效率结论。三 seed 均满足非零 Foundation KD、至少两个同步监督任务和稳定 TaskRouter entropy 的机制证据。

## 三 seed 真实 COCO effect-gate（weight=0.05）

在同一 COCO manifest、模型初始化契约和完整 `val2017` 条件下，补齐了 `foundation_loss_weight=0.05` 的三 seed 配对实验。设置仍为 `epochs=3`、`fraction=0.005`、`imgsz=128`、`batch=2`、CPU；原始逐运行记录见 [f15-real-coco-effect-gate-w005.json](/Users/gatilin/PycharmProjects/YOLO-Master-v260810-latest/reports/foundation/v0.1/f15-real-coco-effect-gate-w005.json)。

| 指标 | seed 20260813 | seed 20260814 | seed 20260815 | 均值 | 标准差 |
|---|---:|---:|---:|---:|---:|
| Foundation loss | 0.159238 | 0.156607 | 0.158976 | 0.158274 | 0.001449 |
| Foundation/Task ratio | 0.0030988 | 0.0030506 | 0.0030696 | 0.0030730 | 0.0000242 |
| 同步监督任务数 | 2.88851 | 2.89189 | 2.90541 | 2.89527 | 0.00894 |
| TaskRouter entropy | 0.728568 | 0.728569 | 0.728564 | 0.728567 | 0.0000026 |
| representation-transfer ready | 1 | 1 | 1 | 1 | 0 |
| negative-transfer risk | 0 | 0 | 0 | 0 | 0 |

三 seed 的 box/mask/pose `mAP50-95` 配对差值全部为 `0.0`；只有 seed `20260813` 的 box `mAP50` 差值为 `+1e-05`，不具有可解释的效果意义。Foundation/Task 比值随权重从 `0.01` 的约 `0.000616` 增至约 `0.003073`，但同步监督任务数和 TaskRouter entropy 保持稳定，三个运行均通过 representation-transfer gate。该结果验证了较高 KD 权重的真实训练信号和边界契约，仍不能宣称 AP、收敛或效率提升。

### 受控 effect-gate 汇总

| Foundation 权重 | 配对数 | Foundation loss 均值±标准差 | Foundation/Task ratio 均值±标准差 | box/mask/pose mAP50-95 配对差值 | accuracy claim |
|---:|---:|---:|---:|:---:|:---:|
| 0.01 | 3 | 0.031723±0.000283 | 0.000616±0.000005 | 全部 0.0 | false |
| 0.05 | 3 | 0.158274±0.001449 | 0.003073±0.000024 | 全部 0.0 | false |

两组权重合计完成 6 个 baseline/Foundation 配对、每个运行 3 个 epoch 和完整 `val2017`。因此 F15 的受控 effect-gate 已完成，但它仍是低 fraction、低分辨率、短训练的工程验证，不等同于正式 COCO 收敛基准；正式 AP 结论仍需更高 `fraction`、`imgsz` 和 `epochs` 的资源充足实验。

## 高预算正式 pilot（weight=0.01）

为在低预算 effect-gate 之后增加一组更接近正式训练的对照，使用固定 seed `20260813`、`foundation_loss_weight=0.01`、`epochs=5`、`fraction=0.02`、`imgsz=192`、`batch=2`、CPU，并对完整 `val2017` 执行 baseline/Foundation 配对。两支 run 使用相同模型配置、初始化、数据切分和训练顺序；Foundation 使用本地 DINOv3 权重。原始记录见 [f15-formal-pilot-w001.json](/Users/gatilin/PycharmProjects/YOLO-Master-v260810-latest/reports/foundation/v0.1/f15-formal-pilot-w001.json)，checkpoint 位于 `runs/multitask/f15-formal-pilot-w001`。

Foundation 分支最后一个训练 epoch 的机制统计如下：

| 指标 | 最终值 |
|---|---:|
| Foundation loss | 0.0236958 |
| Foundation cosine loss | 0.0162245 |
| Foundation relational loss | 0.00747131 |
| Foundation/Task ratio | 0.000485084 |
| 同步监督任务数 | 2.90617 |
| TaskRouter entropy | 0.728525 |
| negative-transfer risk | 0 |
| representation-transfer-ready | 1 |

最后 epoch 的验证对照为：

| 指标 | Baseline | Foundation | Foundation - Baseline |
|---|---:|---:|---:|
| box mAP50-95 | 0 | 0 | 0 |
| mask mAP50-95 | 0 | 0 | 0 |
| pose mAP50-95 | 0 | 0 | 0 |
| box mAP50 | 0.00002 | 0.00001 | -0.00001 |
| mask mAP50 | 0 | 0 | 0 |
| pose mAP50 | 0 | 0 | 0 |
| val box loss | 3.61629 | 3.61759 | +0.00130 |
| val cls loss | 46.8693 | 85.8367 | +38.9674 |
| val dfl loss | 0.06673 | 0.06763 | +0.00090 |
| val segment loss | 22.2328 | 23.4100 | +1.1772 |
| val pose loss | 42.4943 | 42.5073 | +0.0130 |

该 pilot 证明提高预算后 Foundation teacher、共享表征蒸馏、三任务监督和 TaskRouter 统计仍稳定工作，且未触发负迁移风险；但三项 `mAP50-95` 仍为 0，box `mAP50` 反而低 `1e-05`，验证分类损失明显上升。因此本组结果不支持 AP、收敛或泛化优势结论，`accuracy_claim=false` 保持不变。5 epoch、`fraction=0.02` 仍不足以替代资源充足的多 seed COCO 收敛实验，`ready_for_next_phase=false` 继续保持。

## MPS 真实 teacher 训练信号验证

在 Apple Silicon M1 Pro 上直接使用 PyTorch MPS 执行了 F15 的真实 COCO 多任务训练信号验证。运行环境确认 `torch.backends.mps.is_available() == True`，配置为 `device=mps`、`workers=0`、`epochs=1`、`fraction=0.0005`、`imgsz=128`、`batch=2`、固定 seed `20260817`；baseline 与 Foundation 使用同一模型配置、数据切分和初始化契约。本次信号验证关闭 `val`，因此不产生可用于 AP 对比的 validation 结果。

原始记录见 [f15-mps-signal.json](/Users/gatilin/PycharmProjects/YOLO-Master-v260810-latest/reports/foundation/v0.1/f15-mps-signal.json)，运行目录为 `runs/multitask/f15-mps-signal`。

| 指标（第 1 epoch） | Baseline | Foundation |
|---|---:|---:|
| 训练完成 | 30/30 batches | 30/30 batches |
| 训练耗时 | 221.7902 s | 114.5831 s |
| train box loss | 3.68312 | 3.66393 |
| train segment loss | 26.7758 | 26.3168 |
| train pose loss | 20.4674 | 19.6444 |
| Foundation loss | 0 | 0.0317479 |
| Foundation cosine loss | 0 | 0.0196864 |
| Foundation relational loss | 0 | 0.0120616 |
| Foundation/Task ratio | — | 0.000632537 |
| active/supervised tasks | — | 3 / 2.83333 |
| TaskRouter entropy | — | 0.728574 |
| representation-transfer-ready | — | 1 |
| negative-transfer risk | — | 0 |

训练配置明确记录了 `foundation_enabled=true`、`foundation_teacher=dinov3`、`foundation_teacher_device=mps`，Foundation 分支持续产生非零 KD、共享多任务监督和 router 统计；未观察到 NaN、MPS 算子错误或训练崩溃。两支均生成了 `best.pt` checkpoint。该结果证明真实 DINOv3 teacher 在 MPS 上参与训练并完成 Foundation 链路，但它只是训练信号证据，不能解释为 AP、收敛或泛化提升。

另有一次开启完整 COCO `val2017` 的 MPS pilot 在约 `48/1250` validation batches 后因耗时手动中断；该中断运行不纳入任何 effect-gate 或 accuracy 结论。CUDA GPU real-teacher smoke 仍未验证，MPS 通过不等同于 CUDA 通过。

## F15 release audit

已新增只读审计入口 [foundation_f15_release_audit.py](/Users/gatilin/PycharmProjects/YOLO-Master-v260810-latest/scripts/foundation_f15_release_audit.py)，并对两份低预算真实 effect-gate 报告、正式 pilot 报告及其 Foundation checkpoint 执行审计。结果见 [f15-release-audit.json](/Users/gatilin/PycharmProjects/YOLO-Master-v260810-latest/reports/foundation/v0.1/f15-release-audit.json)：`46/46` 项检查通过，覆盖报告 schema/配对完整性、`accuracy_claim=false`、checkpoint `training_only=true`、DINOv3 teacher 不注册进 state、以及 export strip 后纯 `MultiTaskModel` 边界。

审计通过只表示 F15 发布产物的 provenance 和部署边界一致；它不等同于 AP 结论。审计结果明确保持 `ready_for_accuracy_claim=false` 与 `ready_for_next_phase=false`：低预算 effect-gate 为 3 seeds、3 epochs、`fraction=0.005`、`imgsz=128`，正式 pilot 也只有单 seed、5 epochs、`fraction=0.02`、`imgsz=192`，仍不足以支持多 seed COCO 收敛或 AP 优势结论。

## 效果矩阵与数据预检

已新增可复现入口：

- `scripts/foundation_f15_effect_matrix.py`：固定 student 初始化和 synthetic batch，遍历 `seed × foundation_loss_weight`，汇总 task-loss delta、均值/标准差、KD/P4 梯度 Gate 和训练开销。
- `scripts/check_coco2017_multitask.py`：只读检查 COCO 2017 train/val 的 images、instances、polygon segmentation 和 person keypoints，不下载、不改写数据。

一次本地 DINOv3 小矩阵 smoke（`2 seeds × 2 weights × 2 steps`）结果：

| KD weight | Foundation task delta mean | std | 平均开销 | 机制 Gate |
|---:|---:|---:|---:|:---:|
| 0.01 | 0.038034 | 1.226687 | 7.8% | 通过 |
| 0.05 | 0.079507 | 1.349770 | 19.6% | 通过 |

该矩阵仍是 synthetic mechanism validation，不是准确率实验。COCO 预检报告见 [coco2017-preflight.json](/Users/gatilin/PycharmProjects/YOLO-Master-v260810-latest/reports/foundation/v0.1/coco2017-preflight.json)，当前状态为 `ready=true`。

## 真实 COCO 链路 smoke

用户提供完整数据后，已使用 `/Users/gatilin/MyWork/datasets/coco2017` 完成真实数据预检和统一 manifest 生成。为保护已有产物，新 manifest 写入 `/Users/gatilin/MyWork/datasets/coco2017/unified_multitask_f15`，当前模型只启用 `detect + segment + pose` 三个已构建分支。

受控真实 smoke 设置：`fraction=0.0001`、`imgsz=64`、`batch=2`、`epochs=1`、CPU、6 个 train batches、固定 seed `20260813`；baseline 与 Foundation 使用相同初始化和数据顺序。完整记录见 [f15-real-coco-smoke.json](/Users/gatilin/PycharmProjects/YOLO-Master-v260810-latest/reports/foundation/v0.1/f15-real-coco-smoke.json)。

| 指标 | Baseline | Foundation |
|---|---:|---:|
| train time | 2.49995s | 2.70962s |
| box loss | 3.71018 | 3.71018 |
| segment loss | 27.3403 | 27.3403 |
| pose loss | 8.17838 | 8.17838 |
| Foundation loss | 0 | 0.0353075 |
| 同步监督任务均值 | — | 2.333333 |
| representation-transfer gate | — | 通过 |
| val32 box/mask/pose mAP50-95 | 0 / 0 / 0 | 0 / 0 / 0 |

结论：真实 COCO 的数据加载、三任务 loss、DINOv3 teacher、Foundation KD、checkpoint metadata 和 export strip 均已贯通；但该 smoke 训练量极小且未充分收敛，val32 指标均为 0，不能作为效果提升证据。正式 F15 effect gate 仍需多 epoch、多个 KD weight、至少 3 个 seed 和完整 val2017 评估。
