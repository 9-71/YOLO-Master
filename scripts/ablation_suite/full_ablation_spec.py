"""
================================================================================
YOLO-PEFT Full Ablation Experiment — Unified Interface Specification
================================================================================
本文件定义全量消融实验的统一数据结构规范，所有子脚本必须遵循。
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class DatasetConfig:
    """数据集配置规范。"""

    name: str  # 数据集标识名
    yaml: str  # ultralytics 数据 YAML 路径
    num_classes: int  # 类别数
    train_images: int  # 训练图数（用于参考）
    val_images: int  # 验证图数
    description: str = ""  # 人类可读描述
    download_cmd: Optional[str] = None  # 自动下载命令（若需要）
    domain: Optional[str] = None  # 用于持续学习的域标签


@dataclass
class VariantConfig:
    """PEFT 变体配置规范。"""

    name: str  # 变体标识名
    peft_type: str  # "full" / "peft" / "molora" / "molora_aware" / "molora_calib"
    description: str = ""  # 人类可读描述
    train_kwargs: Dict[str, Any] = field(default_factory=dict)  # 传给 model.train() 的额外参数
    molora_config: Dict[str, Any] = field(default_factory=dict)  # MoLoRA 专属配置
    epochs: int = 50  # 该变体的训练 epoch 数
    batch: int = 8  # 该变体的 batch size
    imgsz: int = 640  # 该变体的输入分辨率


@dataclass
class ExperimentResult:
    """单实验结果规范。所有子脚本必须按此结构写入 JSON。"""

    dataset: str
    variant: str
    seed: int
    ok: bool
    error: Optional[str] = None
    elapsed_sec: float = 0.0
    params_total: int = 0
    params_trainable: int = 0
    trainable_pct: float = 0.0
    final_metrics: Dict[str, float] = field(default_factory=dict)
    adapter_sig: Dict[str, Any] = field(default_factory=dict)
    molora_diagnostics: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None  # 推理延迟（若测量）
    latency_backend: Optional[str] = None  # 延迟测量后端
    map_small: Optional[float] = None  # mAP for small objects
    map_medium: Optional[float] = None  # mAP for medium objects
    map_large: Optional[float] = None  # mAP for large objects

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── 预定义数据集配置 ──
DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "coco": DatasetConfig(
        name="coco",
        yaml="coco.yaml",
        num_classes=80,
        train_images=118287,
        val_images=5000,
        description="COCO2017: 80-class general object detection benchmark",
    ),
    "coco128": DatasetConfig(
        name="coco128",
        yaml="coco128.yaml",
        num_classes=80,
        train_images=128,
        val_images=128,
        description="COCO128: 128-image subset for fast prototyping",
    ),
    "visdrone": DatasetConfig(
        name="visdrone",
        yaml="VisDrone.yaml",
        num_classes=10,
        train_images=6471,
        val_images=548,
        description="VisDrone: aerial small-object detection (10 classes)",
        download_cmd=None,  # 用户需手动放置或配置
    ),
    "sku110k": DatasetConfig(
        name="sku110k",
        yaml="SKU110K.yaml",
        num_classes=1,
        train_images=8237,
        val_images=2941,
        description="SKU110K: extreme dense retail detection (1 class, ~100 instances/img)",
        download_cmd=None,
    ),
    "cityscapes": DatasetConfig(
        name="cityscapes",
        yaml="cityscapes.yaml",
        num_classes=30,
        train_images=2975,
        val_images=500,
        description="Cityscapes: urban street scene detection (30 classes, fine-grained)",
        domain="day",
    ),
    "foggy_cityscapes": DatasetConfig(
        name="foggy_cityscapes",
        yaml="foggy_cityscapes.yaml",
        num_classes=30,
        train_images=2975,
        val_images=500,
        description="FoggyCityscapes: domain-shifted variant of Cityscapes",
        domain="fog",
    ),
    "voc0712": DatasetConfig(
        name="voc0712",
        yaml="/Users/gatilin/MyWork/datasets/voc/VOC0712/VOCdevkit/VOC2007/voc0712.yaml",
        num_classes=20,
        train_images=5011,
        val_images=4952,
        description="VOC0712: PASCAL VOC 2007 (trainval=5011, test=4952, 20 classes)",
    ),
}


# ── 预定义 PEFT 变体配置（COCO 全量实验） ──
FULL_ABLATION_VARIANTS: List[VariantConfig] = [
    # 0. 全量微调 Baseline
    VariantConfig(
        name="full",
        peft_type="full",
        description="Full fine-tuning: all parameters trainable",
        epochs=50,
        batch=16,
        imgsz=640,
    ),
    # 1. 标准 LoRA
    VariantConfig(
        name="lora_r16",
        peft_type="peft",
        description="Standard LoRA (r=16, alpha=32, RS-LoRA)",
        train_kwargs={
            "lora_type": "lora",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_backend": "peft",
            "lora_dropout": 0.05,
        },
        epochs=50,
        batch=16,
        imgsz=640,
    ),
    # 2. DoRA
    VariantConfig(
        name="dora_r16",
        peft_type="peft",
        description="DoRA (Weight-Decomposed LoRA, r=16, alpha=32, RS-LoRA)",
        train_kwargs={
            "lora_type": "lora",
            "lora_use_dora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_backend": "peft",
            "lora_dropout": 0.05,
        },
        epochs=50,
        batch=16,
        imgsz=640,
    ),
    # 3. LoHa
    VariantConfig(
        name="loha_r16",
        peft_type="peft",
        description="LoHA (Hadamard product, r=16, alpha=32, RS-LoRA)",
        train_kwargs={
            "lora_type": "loha",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_backend": "peft",
            "lora_dropout": 0.05,
        },
        epochs=50,
        batch=16,
        imgsz=640,
    ),
    # 4. IA3
    VariantConfig(
        name="ia3",
        peft_type="peft",
        description="IA3 (Infused Adapter, no rank)",
        train_kwargs={
            "lora_type": "ia3",
            "lora_backend": "peft",
        },
        epochs=50,
        batch=16,
        imgsz=640,
    ),
    # 5. HRA
    VariantConfig(
        name="hra_r16",
        peft_type="peft",
        description="HRA (Householder Reflection, r=16, alpha=32, RS-LoRA)",
        train_kwargs={
            "lora_type": "hra",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_backend": "peft",
            "lora_dropout": 0.05,
        },
        epochs=50,
        batch=16,
        imgsz=640,
    ),
    # 6. 标准 MoLoRA (E=4, K=2, r=8)
    VariantConfig(
        name="molora_4e2k",
        peft_type="molora",
        description="MoLoRA (4 experts, top-2, r=8, linear router)",
        molora_config={
            "r": 8,
            "alpha": 16,
            "num_experts": 4,
            "top_k": 2,
            "router_type": "linear",
            "dropout": 0.05,
            "use_rslora": True,
            "balance_loss_coef": 0.01,
            "z_loss_coef": 0.001,
        },
        epochs=50,
        batch=16,
        imgsz=640,
    ),
    # 7. MoLoRA + Spatial Router
    VariantConfig(
        name="molora_4e2k_spatial",
        peft_type="molora",
        description="MoLoRA (4 experts, top-2, r=8, spatial router)",
        molora_config={
            "r": 8,
            "alpha": 16,
            "num_experts": 4,
            "top_k": 2,
            "router_type": "spatial",
            "dropout": 0.05,
            "use_rslora": True,
            "balance_loss_coef": 0.01,
            "z_loss_coef": 0.001,
        },
        epochs=50,
        batch=16,
        imgsz=640,
    ),
    # 8. MoLoRA + Hybrid Router
    VariantConfig(
        name="molora_4e2k_hybrid",
        peft_type="molora",
        description="MoLoRA (4 experts, top-2, r=8, hybrid router)",
        molora_config={
            "r": 8,
            "alpha": 16,
            "num_experts": 4,
            "top_k": 2,
            "router_type": "hybrid",
            "dropout": 0.05,
            "use_rslora": True,
            "balance_loss_coef": 0.01,
            "z_loss_coef": 0.001,
        },
        epochs=50,
        batch=16,
        imgsz=640,
    ),
    # 9. MoLoRA + MoE-aware (per-expert rank)
    VariantConfig(
        name="molora_aware",
        peft_type="molora_aware",
        description="MoLoRA + MoE-aware (frequency-based per-expert rank)",
        molora_config={
            "r": 8,
            "alpha": 16,
            "num_experts": 4,
            "top_k": 2,
            "router_type": "linear",
            "dropout": 0.05,
            "use_rslora": True,
            "balance_loss_coef": 0.01,
            "z_loss_coef": 0.001,
            "per_expert_rank": True,
            "rank_allocator_mode": "frequency",
            "rank_budget_total": 32,
            "rank_min": 2,
        },
        epochs=50,
        batch=16,
        imgsz=640,
    ),
    # 10. MoLoRA + Router Calibration
    VariantConfig(
        name="molora_calib",
        peft_type="molora_calib",
        description="MoLoRA + Router Calibration (ΔW_r, calib_rank=4)",
        molora_config={
            "r": 8,
            "alpha": 16,
            "num_experts": 4,
            "top_k": 2,
            "router_type": "linear",
            "dropout": 0.05,
            "use_rslora": True,
            "balance_loss_coef": 0.01,
            "z_loss_coef": 0.001,
            "router_calibration": True,
            "router_calib_rank": 4,
        },
        epochs=50,
        batch=16,
        imgsz=640,
    ),
]


# ── 快速消融变体（COCO128 冒烟测试） ──
QUICK_ABLATION_VARIANTS: List[VariantConfig] = [
    VariantConfig(name="full", peft_type="full", epochs=3, batch=8, imgsz=320),
    VariantConfig(
        name="lora_r8",
        peft_type="peft",
        epochs=3,
        batch=8,
        imgsz=320,
        train_kwargs={"lora_type": "lora", "lora_r": 8, "lora_alpha": 16, "lora_backend": "peft", "lora_dropout": 0.05},
    ),
    VariantConfig(
        name="molora_4e2k",
        peft_type="molora",
        epochs=3,
        batch=8,
        imgsz=320,
        molora_config={
            "r": 8,
            "alpha": 16,
            "num_experts": 4,
            "top_k": 2,
            "router_type": "linear",
            "dropout": 0.05,
            "use_rslora": True,
            "balance_loss_coef": 0.01,
            "z_loss_coef": 0.001,
        },
    ),
]


# ── 多分辨率消融变体 ──
MULTIRES_VARIANTS: List[VariantConfig] = [
    VariantConfig(
        name="lora_r16",
        peft_type="peft",
        epochs=50,
        batch=16,
        imgsz=320,
        train_kwargs={
            "lora_type": "lora",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_backend": "peft",
            "lora_dropout": 0.05,
        },
    ),
    VariantConfig(
        name="lora_r16",
        peft_type="peft",
        epochs=50,
        batch=16,
        imgsz=640,
        train_kwargs={
            "lora_type": "lora",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_backend": "peft",
            "lora_dropout": 0.05,
        },
    ),
    VariantConfig(
        name="lora_r16",
        peft_type="peft",
        epochs=50,
        batch=8,
        imgsz=1280,
        train_kwargs={
            "lora_type": "lora",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_backend": "peft",
            "lora_dropout": 0.05,
        },
    ),
]
