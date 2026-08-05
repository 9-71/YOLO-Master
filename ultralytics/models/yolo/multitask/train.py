# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO Multi-Task Trainer.

Extends DetectionTrainer for unified multi-task training with
TaskRouter-contextualized features and combined multi-task loss.
"""

from copy import copy

import torch

from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import MultiTaskModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr

# ── per-task loss name kits ──────────────────────────────────────────────
_TASK_LOSS_NAMES = {
    "detect": ("box_loss", "cls_loss", "dfl_loss"),
    "segment": ("box_loss", "cls_loss", "dfl_loss", "seg_loss"),
    "pose": ("box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"),
    "classify": ("cls_global_loss",),
    "depth": ("depth_loss",),
    "obb": ("box_loss", "cls_loss", "dfl_loss", "angle_loss"),
}


class MultiTaskTrainer(DetectionTrainer):
    """Trainer for YOLO Multi-Task Vision Model.

    Builds a MultiTaskModel instead of DetectionModel, and configures
    the combined multi-task loss with per-task weighting and TaskRouter
    balance loss.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize MultiTaskTrainer."""
        if overrides is None:
            overrides = {}
        # Ensure multitask-specific overrides
        overrides.setdefault("task", "multitask")
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a MultiTaskModel instance."""
        model = self.set_model_names_for_load(
            MultiTaskModel(cfg, nc=self.data["nc"], ch=self.data.get("channels", 3), verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        from ultralytics.models import yolo

        # Match the 6-element tensor returned by MultiTaskLoss.forward():
        # [box_loss, cls_loss, dfl_loss, seg_loss, pose_loss, cls_global_loss]
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "seg_loss", "pose_loss", "cls_global_loss")
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def set_model_attributes(self):
        """Set MultiTaskModel attributes from dataset configuration."""
        super().set_model_attributes()
        # Propagate active_tasks from dataset YAML to model
        model = unwrap_model(self.model)
        if hasattr(model, "active_tasks"):
            dataset_tasks = self.data.get("tasks", None)
            if dataset_tasks:
                model.active_tasks = set(dataset_tasks)
                LOGGER.info(f"{colorstr('multitask:')} active_tasks={model.active_tasks}")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build YOLO Dataset for training or validation (fallback: detection)."""
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Label multi-task loss items with readable names."""
        keys = list(self.loss_names)
        if loss_items is not None:
            # loss_items may be a dict (from MultiTaskLoss) or list/tensor
            if isinstance(loss_items, dict):
                return {f"{prefix}/{k}": round(float(v), 5) for k, v in loss_items.items()}
            if isinstance(loss_items, torch.Tensor):
                loss_items = loss_items.tolist()
            return {f"{prefix}/{k}": round(float(v), 5) for k, v in zip(keys, loss_items)}
        return keys

    def progress_string(self):
        """Return a formatted string of training progress with wider columns for multi-task losses."""
        # Use %17s to fit longest loss names: cls_global_loss (14) / mixture_aux_loss (15)
        n_loss = len(self.loss_names)
        fmt = "\n" + "%12s" + "%10s" + "%17s" * n_loss + "%11s" + "%10s"
        return fmt % ("Epoch", "GPU_mem", *self.loss_names, "Instances", "Size")


def unwrap_model(model):
    """Unwrap DDP/DistributedDataParallel wrappers."""
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP

    if isinstance(model, DDP):
        return model.module
    return model
