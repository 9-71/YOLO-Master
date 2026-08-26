# A Technical Analysis of Cross-Platform Edge Deployment of YOLO-Master

End-to-end deployment of **YOLO-Master** to the edge, spanning export formats (ONNX / NCNN / MNN / Core ML / TensorRT), mixed-precision INT8, runtime with GPU acceleration on all backends, two native GUI runners, cross-platform builds for Linux / Windows / Jetson / macOS, and accuracy-latency validation against the PyTorch original.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skywalker-lt/yolo-master-edge/main/assets/edge_deployment_architecture_dark.png">
  <img src="https://raw.githubusercontent.com/skywalker-lt/yolo-master-edge/main/assets/edge_deployment_architecture_light.png" alt="Edge Deployment Bundle — Architecture">
</picture>

- Source code: **[Edge Deployment Repo](https://github.com/skywalker-lt/yolo-master-edge)**
- Download pre-built bundles & GUI Apps: **[Releases](https://github.com/skywalker-lt/yolo-master-edge/releases)**

The evaluation focuses on **YOLO-Master-EsMoE-N** on VisDrone, as specified by Issue #51.

## 🖥️ System Configuration

This table lists the system configurations used for the deployment measurements in this report.

| Host | Linux x86_64 (Docker) | Windows x86_64 | Nvidia Jetson | Mac |
|:---:|:---:|:---:|:---:|:---:|
| Device | Datacenter Server | ROG Strix SCAR 18 | Jetson Orin Nano Super (4GB) DevKit | MacBook Pro 16" (2024) |
| CPU | 2x Intel Xeon 8568Y+ | Intel Core U9-275HX | Arm Cortex-A78AE | Apple **M4 Max CPU** |
| System RAM | 2,048 GB | 32 GB | 4 GB (unified) | 48 GB (unified) |
| GPU | **H200 SXM** | **RTX 5070 Ti Laptop** | Orin GPU, (GA10B, **`sm87`**) | Apple **M4 Max GPU** |
| VRAM | 141 GB / GPU | 12 GB | 4 GB (unified) | 48 GB (unified) |
| OS / platform | Ubuntu Server 22.04 LTS | Windows 11 x64 | JetPack 7 / Ubuntu 24.04 | macOS 26.3 (Tahoe) |

---
## 📋 Table of Contents

- [1. How and why the MoE internals dictate the deployment strategy](#-1-how-and-why-the-moe-internals-dictate-the-deployment-strategy)
- [2. Exports](#-2-exports)
- [3. INT8 quantization (the substantive part)](#-3-int8-quantization-the-substantive-part)
- [4. The inference runtime](#%EF%B8%8F-4-the-inference-runtime)
- [5. Accuracy validation](#-5-accuracy-validation)
- [6. GPU acceleration across all three backends](#-6-gpu-acceleration-across-all-three-backends)
- [7. Latency and throughput](#%EF%B8%8F-7-latency-and-throughput)
- [8. Cross-platform builds and distribution](#-8-cross-platform-builds-and-distribution)
- [9. Embedded GPU deployment: Jetson Orin](#-9-embedded-gpu-deployment-jetson-orin)
- [10. GUI runners for Windows & macOS](#-10-gui-runners-for-windows-and-macos)
- [11. Future work](#-11-future-work)

---

## 🔎 1. How and why the MoE internals dictate the deployment strategy

EsMoE-N is not a CNN YOLO like the older v8/v9 variants. Three structural properties drove the downstream deployment decisions:

1. **Mixture-of-Experts `ES_MOE`:** During training and inference, the router sparsely selects experts. This path is export-hostile because it contains data-dependent control flow, but `ES_MOE.forward` switches to a **dense** unroll under `torch.onnx.is_in_onnx_export()`: a static loop over the full expert list using `Conv/Pool/Softmax/Mul/Add`, with no dynamic dispatch. The `--no-sparse-eval` dense path is the **numerically faithful** one. It avoids sparse-inference collapse and improves export determinism.
2. **Area-attention `A2C2f`:** The backbone contains transformer-style attention blocks. These reshape activations to `[1, 1600, 192]` internally, which is where the static-shape assumptions of downstream quantizers and tracers break (as discussed later).
3. **A stride-8/16/32 detection head** (P3/P4/P5) where the classification branch produces raw logits fed through a terminal sigmoid. At 640×640 that is `80^2 + 40^2 + 20^2 = 8400` anchors, matching the exported `[1, 14, 8400]` output (4 box + 10 VisDrone classes). This branch is the most quantization-sensitive component in the network, for the reason explained in Section 3.

Accordingly, the model exports cleanly because the dense MoE path uses standard operators, while the attention blocks and detection head remain sensitive to INT8 quantization and third-party conversion.

## 📦 2. Exports

### 2.1 ONNX via onnxsim (opset 12)

Exported to a fully **static** graph, input `images [1,3,640,640]`, output `output0 [1,14,8400]`, 628 nodes, IR 7, and simplified with onnxsim. Opset 12 was chosen carefully for compatibility. It loads unchanged under ORT 1.18 / 1.20 / 1.27 and converts cleanly to *both* NCNN and MNN.

The export emitted a shape-inference warning on the attention transpose (`.../attn/Transpose_output_0 source:{1,1600,192} target:{}`, later resolved by ONNX Runtime's lenient merge). ORT resolves the shapes at runtime, but the warning is a **leading indicator**: tools that require fully static shape propagation may reject this graph (Section 3.5, MNN quantization, and Section 2.4, coremltools).

Ultralytics metadata (class names, `imgsz`, `stride`, `task`) is also embedded into the ONNX `metadata_props`, which the C++ runtime reads to auto-configure itself.

### 2.2 NCNN via pnnx

Exported via **pnnx** (PyTorch/ONNX → pnnx IR → ncnn), instead of the legacy `onnx2ncnn`. pnnx preserves higher-level operator semantics and emits a clean graph. The param file was validated: magic `7767517`, **561 layers / 665 blobs**, input blob `in0`, sigmoid-terminated head. A `metadata.yaml` sidecar carries the same names/imgsz so the ncnn path is self-contained and self-describing like the ONNX one.

### 2.3 MNN via mnnconvert

Converted with `mnnconvert` (ONNX → MNN), which emits the same graph structure as ONNX/ncnn and permits direct tensor comparison against the source graph (Section 5.3).

### 2.4 Core ML

Core ML export (`coreml_export/export_coreml.py`) exports a **mlprogram** `.mlpackage` carrying the metadata used by the Apple device application (`names`, `imgsz`, `output`, `task`, plus `proto`/`nm` for seg models). Conversion can run on **Linux**; only prediction requires macOS, keeping export in the same environment as the other formats.

Three failure modes had to be addressed; each reflects one of the structures described in Section 1:

- **Dynamic shapes → `aten::Int`:** coremltools' Torch frontend cannot lower the data-dependent integer extraction produced by the attention reshapes. Constant-folding first resolves the shapes to literals: `jit.freeze` + `run_frozen_optimizations` run before conversion.
- **MoE telemetry → `aten::copy_`:** The trace failed with "No matching select or slice" **not** in the expert computation, but in `ES_MOE`'s in-place auxiliary-loss bookkeeping used only during training. Routing was valid; the telemetry was the root cause.
- **Area-attention static shapes:** An eager warmup pass bakes each layer's concrete spatial dimensions so the area reshapes fold to static shapes. For `sunsmarterjie/yolov12` checkpoints (for testing and comparison), `--yolov12-aattn` additionally swaps in the split qk+v attention variant those weights expect.

The script also handles **segmentation** (detects the two-output signature and writes `task=segment` with `proto`/`nm`) and **LoRA fine-tunes** (`--merge-lora-dir` merges adapters before export, since a merged LoRA is a static graph whereas routed MoLoRA cannot be traced).

**Validation:** The Core ML path currently has **no mAP number**. The macOS app does not bundle a metric harness, and the `eval_map.py` pipeline used for the other formats consumes `--save-txt` output from the C++ runtime, which the Swift app does not produce. A validator is planned for a future Core ML Runner update.

## 🔢 3. INT8 quantization findings

The requirement was ≤ 1.0% mAP gap under INT8 with ≥ 300 images for calibration. The naive pipeline *fails*.

### 3.1 The collapse: full INT8 emits nothing

Static per-channel INT8 over the whole graph produces a model that runs, returns the correct output tensor shape, contains **no NaNs**, **and detects nothing**, mAP=0.0000.

Isolating the output tensor shows why: The box-regression channels are intact (`min 0, max 644, mean 210`, matching FP32); however the **classification channels are zero** (`max = 0.0000`, zero scores above 0.001). This indicates that the failure is concentrated in the class head.

The mechanism is as follows: the class branch emits wide-dynamic-range *logits* consumed by a sigmoid. Per-tensor/per-channel MinMax calibration maps that range to 256 INT8 levels; the small positive logits corresponding to real detections fall *below one quantization step* and round to a value whose sigmoid is approximately 0. The non-linearity turns this quantization error into a complete signal cutoff. Box regression, by contrast, is a smooth linear readout with no saturating nonlinearity downstream and therefore tolerates INT8. This asymmetry of **robust regression but degraded classification** is the key diagnostic.

### 3.2 Localizing the sensitivity

Retaining the detection head (`/model.25/`, 85 nodes) in FP32 while quantizing the remaining nodes recovers the model to **mAP50-95=0.1924, ∆ −1.12 pp** versus PyTorch. The model is functional but remains outside the stated budget; the residual loss is concentrated in two quantization-sensitive structures:

- **MoE router:** Expert mixing is a softmax over routing logits. INT8 trims the precision of routing weights.
- **Area-attn:** Attention scores pass through a softmax whose output is sensitive to input scale; INT8 on QK path shifts the attention distribution.

Both are the same failure class as the head: **a softmax/sigmoid amplifying a quantization perturbation.** This aligns with established LLM quantization strategies.

### 3.3 The mixed-precision outcome

The fix is node-level precision: keep the three softmax/sigmoid-bearing blocks, head (`/model.25/`), attention (`/attn/`), router (`routing`), **289 nodes** in FP32, INT8 everything else. The progression is diagnostic:

| Configuration | mAP50-95 | Δ vs PyTorch (percentage points) |
|---|---|---|
| Full INT8 | 0.0000 | collapse |
| head FP32 | 0.1924 | −1.12 pp |
| head + attention + router FP32 | **0.1952** | **−0.84 pp ✅** |

Final model: **10.9 → 5.4 MB (2.0×)**, with an mAP50-95 difference of **−0.84 percentage points**, within the 1.0-point requirement. The improvement follows from retaining the operators that violate the smooth, non-saturating assumption of PTQ in higher precision.

### 3.4 Calibration engineering

Three implementation details were decisive:

- **Letterbox-matched calibration.** Calibrators default to a plain resize; the model is trained on **letterboxed** input. Calibrating on the wrong preprocessing biases every activation range. We pre-letterbox 300+ VisDrone *train* images (no val leakage) to 640×640 and calibrate on those, so the calibration distribution matches inference exactly.
- **QOperator, and the opset floor.** Per-channel INT8 emits `DequantizeLinear` with an `axis` attribute, which is **only valid at opset ≥ 13**; the opset-12 export must be lifted (we upgrade to 17 in-line) or the quantized model is an invalid graph. QOperator (`QLinearConv`/`QLinearMatMul`) is chosen over QDQ for CPU execution.
- **MinMax over Percentile.** Percentile/entropy calibration builds a histogram per activation tensor; on a graph with hundreds of attention/MoE intermediates and hundreds of images, that becomes computationally expensive without an observed accuracy benefit. The exclusions remove the outlier-sensitive layers, so MinMax on the remaining convolutions is faster and sufficient.

### 3.5 Third-party INT8 toolchains and attention-shape handling

MNN's offline quantizer (`mnnquant`) aborts immediately on this model -- `std::length_error: cannot create std::vector larger than max_size()` -- before any calibration runs. The cause is precisely the `[1,1600,192]` attention reshapes flagged in Section 2.1: the quantizer allocates buffers from statically-inferred tensor dimensions, and the dynamically-shaped attention intermediate reads back as a garbage size. MNN executes this graph fine at *inference* (it resolves shapes lazily); its *quantizer* assumes static shapes. This is a limitation of the tool's static-shape contract, not of the model, and it is not configurable. The ONNXRuntime quantizer, which tolerates dynamic intermediates, is the correct vehicle for this architecture.

### 3.6 Where INT8 provides a benefit, and where it does not

On x86 CPU, INT8 is *slower* than FP32, measured at **137 ms/frame vs 49 ms for FP32 on the same host, ~2.8× slower** (7.2 vs 19.5 FPS; the 40 ms in Section 7 is the canonical 4-thread benchmark on the reference host, so use the paired figures here for the ratio). The QDQ/QOperator kernels don't engage INT8 SIMD paths that beat the well-tuned FP32 convolutions, and the FP32 $\leftrightarrow$ INT8 boundaries around the excluded blocks add conversion overhead. This is expected, not a defect: INT8's throughput win is a property of **INT8 tensor-core hardware**, not of desktop CPUs.

The natural next hypothesis was that tensor-core hardware would invert the result. **It does not for this model**: Section 9 shows that the calibrated TensorRT INT8 engine on Orin is both slower and less accurate than FP16. The mixed-precision assignment keeps the compute-dominant area-attention path out of INT8, so INT8 does not accelerate the dominant computation. More generally, **PTQ throughput is bounded by the fraction of computation that can be quantized**; for attention-heavy architectures, that fraction may be small. The ONNX INT8 result is therefore treated as **accuracy evidence** (−0.84 percentage points, within the stated budget), while FP16 GPU execution provides the measured throughput benefit (Section 6).

## ⚙️ 4. The inference runtime

### 4.1 Universal binary

One executable (`yolomaster_edge`) with **four backends** (ONNXRuntime, NCNN, MNN, TensorRT) behind a common interface (`backend_factory.hpp`). Backend, class names, and input size are **auto-detected** from the model (`.onnx` → ORT, a directory or `.param` → ncnn, `.mnn` → MNN, `.engine` → TensorRT; metadata read from ONNX `metadata_props` or the ncnn `metadata.yaml` sidecar), so the same binary serves any exported YOLO-Master variant with no recompilation.

```text
--backend  auto | onnx | ncnn | mnn | trt
--device   backend-dependent (cpu, cuda, vulkan, opencl, coreml, trt)
```

Source can be an image, a directory, a video, or a `dataset.yaml`. Eighteen robustness tests (corrupt images, missing files, image-size mismatch, backend inference, and output-collision handling) pass on all platforms (`cpp/run_tests.sh`).

### 4.2 Preprocessing

Aspect-ratio-preserving **letterbox** (min-side scale, 114 padding) → RGB `/255` NCHW, matching training. The letterbox metadata (scale & pad) is threaded through decode so boxes map back to original-image pixel coordinates in float with no intermediate rounding.

### 4.3 Decode, NMS, and the mAP-parity subtlety

An early version of the C++ pipeline read **1.19 mAP points low** despite bit-accurate inference. We found the cause in the decode: ultralytics `val` uses **`multi_label=True`**, one detection per class scoring at or above the threshold per anchor, not a single argmax. Reproducing that (`--multi-label` mode) recovered the gap exactly (0.3375 → 0.3494 mAP50). NMS is **per-class** (`agnostic=False`), implemented with a class-offset trick (shift each box by `class_id × (2·max(image dimension) + 8192)` so cross-class boxes never suppress each other), and capped at 300 detections. Default `conf` is low, appropriate to VisDrone's small/dense objects; `--conf`/`--iou` are tunable per deployment.

### 4.4 Instance Segmentation

The runtime also decodes instance segmentation models. A segmenter emits a second output, the prototype tensor, alongside the detection tensor; the exported metadata carries `task=segment` plus `proto`/`nm` so the runtime dispatches on the model. Masks are reconstructed by combining the per-instance mask coefficients with the prototypes cropped to each box and thresholded; the GUI runners composite them with anti-aliasing to avoid serrated edges.

The shipped default for both GUI runners is **`YOLO-Master-v0.1-seg-N`** (`task: segment`, imgsz 640, stride 32, **COCO-80** classes), exported to ONNX / MNN / ncnn. Note the domain change: the detection results throughout this report are VisDrone 10-class, while the bundled segmenter is COCO-80. **No segmentation mAP is reported** -- the metric harness in Section 5 is detection-only, and extending it to mask AP is future work (Section 11).

### 4.5 Dependency cut for a portable bundle

The first self-contained Linux bundle was **231 shared libraries, 129 MB** since Ubuntu's `libopencv_imgcodecs` links **GDAL**, which transitively pulls in PostgreSQL (`libpq`), MySQL, `libpoppler` (PDF), HDF5, and the GIS stack, and `libopencv_dnn` pulls protobuf. An object detector does not need a Postgres client. We removed both by replacing `cv::imread`/`imwrite` with **stb_image** and `cv::dnn::NMSBoxes`/`blobFromImage` with a hand-written NMS and a manual NCHW pack. That drops the OpenCV surface to **core + imgproc only**: 231 → 10 libraries, **129 → 35 MB**, at a cost of a **0.087%** detection-count difference (stb vs OpenCV JPEG decoders diverge by sub-LSB pixel values on a handful of borderline boxes), inside tolerance. On Linux the binary is `$ORIGIN`-rpath'd and verified to run with no `LD_LIBRARY_PATH`; on Windows the MSVC runtime is bundled so targets need no VC++ Redistributable.

## 📊 5. Accuracy validation

### 5.1 Methodology

Every model (PyTorch, ONNX, NCNN, MNN, INT8, CUDA, Vulkan, OpenCL, TensorRT) is scored through a single path: predictions at **conf 0.001, NMS iou 0.7, multi-label, cap 300** (ultralytics `val` settings), fed to ultralytics' own `DetMetrics` + `box_iou` + `match_predictions` (`eval_map.py`). This common procedure makes the numbers comparable across formats and directly comparable to the ultralytics reference. ONNX/ncnn/MNN/GPU predictions are produced by the C++ runtime; the Jetson uses a dependency-free reimplementation of the same harness `eval_map_standalone.py`. Core ML is the one path outside this harness and will be included later (Section 2.4).

### 5.2 Results (548 VisDrone val images)

| Model | Device | mAP50-95 | Δ mAP50-95 vs PyTorch (percentage points) |
|---|---|---|---|
| **PyTorch (reference)** | -- | 0.2036 | -- |
| ONNX | CPU | 0.2034 | **−0.02 pp** |
| NCNN | CPU | 0.2034 | **−0.02 pp** |
| MNN | CPU | 0.2034 | **−0.02 pp** |
| ONNX (CUDA) | H200 SXM | 0.2033 | **−0.03 pp** |
| ONNX (CUDA) | RTX 5070Ti Laptop | 0.2033 | **−0.03 pp** |
| NCNN (Vulkan, FP16) | RTX 5070Ti Laptop | 0.2034 | **−0.02 pp** |
| MNN (OpenCL, FP16) | RTX 5070Ti Laptop | 0.2034 | **−0.02 pp** |
| TensorRT FP16 | Jetson Orin Nano 4GB | 0.2029 | **−0.07 pp** |
| INT8 (mixed) | CPU | 0.1952 | **−0.84 pp** |

All three FP32 CPU export formats land on **identical** mAP (0.2034), as expected for the same graph, at **−0.02 percentage points** from PyTorch, 25× inside the 0.5-point target. INT8 is **−0.84 percentage points**, inside the 1.0-point target. (The INT8 mAP50 drop is larger, −1.27 percentage points, reflecting slightly softer classification confidences at INT8; the budget is defined on mAP50-95, which passes.)

The GPU rows extend the CPU measurements and expose a backend-dependent performance pattern (see Section 6).

### 5.3 Numerical parity, isolating format from pipeline

Because the FP32 formats share a graph, we verify fidelity directly rather than only through mAP. Feeding **identical letterboxed inputs** to MNN and the source ONNX across 100 val images yields **max|Δ| = 0.096, mean|Δ| = 9.7e-05** on the raw `[1,14,8400]` output. The max is a single box-coordinate least-significant bit (coordinates run to ~640; 0.096 px is nothing); the mean is negligible. Detection counts over the full set are effectively equal (ONNX 157,464 vs ncnn 157,465 at conf 0.001). This distinguishes *format equivalence* from *coincidentally similar mAP*.

The same analysis identified a **false alarm** on the CUDA path: a raw `max|Δ| = 2.31` was traced to FP32 box-coordinate variance in a single anchor, while functional mAP remained identical. A scalar max-absolute-difference gate would therefore reject a functionally equivalent model; separating box and class channels provides the relevant diagnostic.

## 🚀 6. GPU acceleration across all three backends

Each backend has a different native accelerator, and the runtime maps a single **Device: CPU / GPU** switch onto all of them: **ONNX → CUDA**, **ncnn → Vulkan**, **MNN → OpenCL**, with ncnn and MNN running **FP16** on the GPU. Every backend falls back to CPU cleanly and surfaces the reason when a provider is unavailable, which matters in a GUI where the user cannot read a stderr log.

Measured on the same 548 VisDrone images, one consumer laptop GPU (Win11):

| Backend | CPU | GPU | Speedup |
|---|---|---|---|
| ONNX → CUDA | 40 ms | **9.0 ms** | 4.4× |
| MNN → OpenCL (FP16) | 74 ms | **19.1 ms** | 3.9× |
| NCNN → Vulkan (FP16) | 80 ms | **20.2 ms** | 4.0× |

Three findings:

**FP16 on the GPU is accuracy-neutral.** The ncnn-Vulkan and MNN-OpenCL FP16 paths both score **0.2034 mAP50-95, identical to their FP32 CPU counterparts**, and CUDA scores 0.2033. This contrasts with Section 3: half-precision is a *uniform* reduction in mantissa that the softmax/sigmoid structures tolerate, whereas INT8 is a *range mapping* that the same structures amplify. For this architecture, FP16 is therefore the preferred accelerator precision and INT8 offers no corresponding benefit.

**The x86 backend ranking is preserved on GPU.** ORT retains an approximately 2× lead over MNN and ncnn on the GPU (9.0 versus 19.1/20.2 ms), similar to the CPU results (40 versus 74/80 ms). The ARM comparison remains untested because the Jetson path uses TensorRT (Section 9) rather than ncnn or MNN.

**The model does not saturate a datacenter GPU.** An H200 SXM (7.8 ms) is only ~15% faster than an RTX 5070Ti Laptop (9.0 ms). At nano scale with a 640×640 input, per-launch overhead and memory traffic dominate, not FLOPs -- so for this model class a consumer GPU is the sensible deployment target and the datacenter part buys almost nothing.

## ⏱️ 7. Latency and throughput

Per-frame inference, VisDrone val:

| Platform | Backend | Device | infer (ms) | FPS |
|---|---|---|---|---|
| Linux CPU (4-thread) | ONNX (ORT) | CPU | 40.0 | 25.0 |
| Windows 11 CPU | ONNX (ORT) | CPU | 37.6 | 25.4 |
| Linux CPU (4-thread) | MNN | CPU | 74.0 | 13.5 |
| Linux CPU (4-thread) | NCNN | CPU | 80.0 | 12.5 |
| Windows 11 CPU | NCNN | CPU | 80.1 | 12.2 |
| Linux CPU (4-thread) | ONNX INT8 (mixed) | CPU | 137 | 7.2 |
| Linux | ONNX (ORT) | **CUDA / H200 SXM** | 7.8 | **128** |
| Windows | ONNX (ORT) | **CUDA / RTX 5070Ti Laptop** | 9.0 | **111** |
| Windows | MNN | **OpenCL / RTX 5070Ti Laptop** | 19.1 | 52.4 |
| Windows | NCNN | **Vulkan / RTX 5070Ti Laptop** | 20.2 | 49.5 |
| Jetson Orin Nano 4GB | **TensorRT FP16** | Orin iGPU | 27.8 | 35.7 |
| macOS (M4 Max) | **Core ML** | MPS / ANE | 17.4 | 57.4 |

CPU latencies are measured on one x86 host with four threads. The ordering is consistent with the respective optimization targets: **ORT is approximately 2× faster than MNN and NCNN on x86**, while the latter runtimes prioritize mobile and ARM deployments; the ordering persists on the GPU (Section 6). **INT8 is the slowest row** for the reasons in Section 3.6. The Core ML row is latency-only (no mAP, Section 2.4); its 17.4 ms result places the M4 Max between the consumer GPU and CPU measurements at a lower power envelope.

No single format fits every deployment target; the results are therefore reported for four deployment distributions below.

## 🌐 8. Cross-platform builds and distribution

A single CMake tree now targets **four platforms**:

| Platform | Toolchain | Backends | Distribution |
|---|---|---|---|
| Linux x86_64 | GCC / CMake | ONNX (+CUDA EP), NCNN, MNN | 35 MB `$ORIGIN`-rpath'd tarball, 10 libs, verified isolated |
| Windows 10/11 x64 | VS 2022 / 2026, MSVC 19.5x | ONNX (+CUDA), NCNN (+Vulkan), MNN (+OpenCL) | Self-contained zip, MSVC runtime bundled; lean and CUDA variants |
| macOS 14+ | Swift 5.9+, Xcode CLT | Core ML | Universal (Apple Silicon + Intel) `.app`, Developer-ID signed and **notarized** |
| Jetson Orin (JetPack 7) | aarch64 GCC / CMake | TensorRT (priority), ONNX, NCNN, MNN | aarch64 tarball, OpenCV bundled, TensorRT/CUDA from JetPack |

The Windows port surfaced three concrete portability issues, each fixed in the build system rather than worked around: `Ort::Session` takes `const wchar_t*` on Windows (a platform `ORTCHAR_T` shim); the prebuilt OpenCV config doesn't recognize the VS 2026 toolset and reports an empty runtime (point `OpenCV_DIR` at the concrete `vc16/lib` config); and the exe needs the MSVC runtime on clean targets (bundled via `InstallRequiredSystemLibraries`). SDK locations live in a gitignored `sdk-paths.cmd` rather than in the build scripts, and `.cmd` entry points are provided alongside `.ps1` because PowerShell execution policy blocks the latter on default Windows installs.

The Windows GUI ships in two flavours: a lean bundle that gets GPU inference through ncnn-Vulkan and MNN-OpenCL using only the graphics driver, and a considerably larger CUDA bundle that additionally carries the cuDNN and CUDA runtime libraries for the fastest ONNX path. Neither requires a CUDA toolkit installation on the target. The executable is **not code-signed** (that needs a paid certificate), so SmartScreen warns on first launch; the macOS app, by contrast, is signed and notarized and installs by double-click.

## 🤖 9. Embedded GPU deployment: Jetson Orin

The runtime was taken to a **Jetson Orin Nano 4 GB**. The same CMake produces a native aarch64 binary with a `trt_backend` that deserializes a prebuilt engine and runs it via `enqueueV3`, joining the other backends behind the same interface. The engine is built on-device with `trtexec` from the exported ONNX.

**Result.** The FP16 engine runs at **27.8 ms/frame GPU compute (35.7 FPS)**, and the on-device accuracy over the full 548 VisDrone val set is **mAP50-95 0.2029, ∆ −0.07 pp versus the PyTorch FP32 baseline** (0.2036), matching the x86 ONNX result to within 0.2 mAP points.

**FP16, not INT8.** Section 3.6 reserved the INT8 *throughput* evaluation for this path, based on the expectation that tensor-core INT8 would invert the CPU result. For this model it does not. The mixed-precision assignment from Section 3 keeps the attention blocks, head, and router in higher precision, so INT8 leaves the compute-heavy area-attention on FP32/FP16 kernels; combined with TensorRT's lower INT8 accuracy relative to the ONNX Runtime path, the calibrated INT8 engine measures **0.3202 mAP50 at 21.7 FPS, slower and less accurate than FP16**. When attention dominates computation and is not quantized, **FP16 is the appropriate embedded target**. Taken together with Section 6, FP16 is the preferred target on every tested accelerator: it is accuracy-neutral on desktop GPUs and incurs minimal accuracy cost on Jetson Orin, whereas INT8 does not provide a benefit for this model.

**Build notes.** Two toolchain specifics are worth recording. On sm87 with TRT 10.16.2 a pure-FP16 build fails at low builder-optimization levels (the timing model references an sm80 shader that has no sm87 base); `--builderOptimizationLevel=3` selects tactics by on-device profiling instead and builds cleanly. And an ONNXRuntime-quantized QDQ model must use symmetric activations and non-quantized bias to be accepted by TensorRT's parser `quantize_int8.py --symmetric`. The Nano 4 GB version also needs swap for the engine *build* (inference itself uses only ~20 MB).

## 💻 10. GUI runners for Windows and macOS

The CLI is efficient for benchmarking and batch work but is less accessible for interactive use. Two native GUI runners use the same pipeline.

**macOS -- YOLO-Master CoreML Runner** (v1.0.0-macos). A SwiftUI frontend over Core ML: `YOLOMasterKit` carries the pipeline (preprocess, detect, annotate, image I/O) and the app layer adds camera and UI. Universal binary, Developer-ID signed & notarized, macOS 14+ (the SwiftUI API floor -- `onKeyPress`, zero-parameter `onChange`).

<img width="1799" height="1196" alt="Screen1" src="https://github.com/user-attachments/assets/c4573254-bada-4495-85a8-3164a976ab93" />

**Windows -- YOLO-Master Windows Runner GUI** (v1.0.0-windows). Native Win32 + Direct3D 11 + Dear ImGui. The important architectural property is that it **compiles the CLI's own runtime sources** (`cpp/common.cpp`, the backend implementations, `stb_impl.cpp`) rather than reimplementing them, so letterbox, decode, per-class NMS, and the class palette are identical to the validated CLI path. Sharing the translation unit is what lets Section 5's mAP table apply to what the user actually runs. It carries all three backends in one executable, switchable at runtime, which makes backend comparison on the same image a single-click operation.

<img width="1277" height="764" alt="49 2" src="https://github.com/user-attachments/assets/fb31fdd5-84c4-40d4-9294-13d7bc433a42" />

The shared interaction model is based on two design decisions:

- **Forward once, tune efficiently.** Confidence, IoU, box style, labels, and letterbox-vs-stretch all redraw from a cached forward pass. Inference never re-runs when a threshold moves, so the controls remain usable at interactive rates even where a frame costs 80 ms. Candidates are cached down to a 0.05 confidence floor so lowering the threshold remains immediate.
- **Two-phase media handling.** Folders and videos are inferred once with a progress bar, then browsed or scrubbed from cache. A 30 fps clip therefore plays back at 30 fps regardless of model speed, because inference is off the playback path entirely. The webcam path instead infers on a background thread with drop-late-frames, trading completeness for latency.

## 🔖 11. Future work
- **Core ML accuracy validation.** The one backend without an mAP number (Section 2.4). A `--save-txt`-compatible dump from the macOS app would feed directly into `eval_map.py` and make the Core ML row comparable with the other entries in Section 5.2.
- **Segmentation metrics.** The harness is detection-only; the bundled `v0.1-seg-N` is validated visually. Mask AP against the COCO protocol would close it.
- **ARM-native backend comparison.** Section 6 leaves the original hypothesis that ncnn and MNN close the gap on ARM -- untested since the Jetson work went through TensorRT. Running the ncnn-Vulkan and MNN-OpenCL paths on the Orin would settle it on the hardware they were designed for.
- **Non-NVIDIA GPUs.** The Vulkan and OpenCL paths should already work on AMD, Intel, and Ascend hardware.
- **Production drone platform (DJI Manifold 3).** VisDrone is aerial/drone imagery, so the natural production target is an onboard drone computer. [DJI Manifold 3](https://enterprise.dji.com/manifold-3) is an **NVIDIA Orin NX-based** enterprise edge computer purpose-built for drones where the exact aarch64 + TensorRT path in Section 9 deploys onto it directly. Validating this pipeline on the Manifold 3 exercises **real-time on-drone inference in operational conditions** (aerial surveillance, infrastructure inspection, search-and-rescue), closing the loop from VisDrone training to production drone edge deployment.

---

## Reproducibility

The C++ runtime, GUI frontends (`gui/`, `mac/`), Core ML exporter (`coreml_export/`), validation scripts (`quantize_int8.py`, `eval_map.py`, `eval_map_standalone.py`, `mnn_val.py`, `mnn_parity.py`, `package_linux.sh`), and Jetson tooling (`jetson/`, including `DEPLOYMENT_LOG.md`) are included in this repository. Exported models and prebuilt bundles for Linux, Windows, macOS, and Jetson Orin are available from the [release page](https://github.com/skywalker-lt/yolo-master-edge/releases).
