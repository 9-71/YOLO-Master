"""Unified Multi-Task Detection Head.

Extends the YOLO Detect head to support detection, instance segmentation,
pose estimation, classification, depth estimation, and oriented bounding
box detection simultaneously.

MOT-inspired design:
- Two-stage task assignment: primary (high-conf) + secondary (ambiguous)
- Cross-task feature sharing through a shared bottleneck
- Task-aware dynamic weighting via learned task importance scores
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.block import Proto, Proto26
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.utils import get_safe_groups as _safe_groups
from ultralytics.utils.tal import dist2bbox, make_anchors


class MultiTaskHead(Detect):
    """Unified multi-task head: detection + segmentation + pose + classification + depth + OBB.

    Each task branch is independently configurable. The head can be used with
    or without a TaskRouter — without one it defaults to parallel prediction
    for all configured tasks.

    Args:
        nc: Number of classes (shared across detection/seg/pose/obb).
        ch: Tuple of input channel sizes from FPN (e.g. (256, 512, 1024)).
        tasks: List of active task names. Supported: "detect", "segment", "pose",
               "classify", "depth", "obb".
        nm: Number of mask coefficients (segment).
        npr: Number of mask prototypes (segment).
        kpt_shape: (num_keypoints, dims) for pose estimation.
        depth_bins: Number of depth discretization bins.
        reg_max: DFL channels.
        end2end: Use one-to-one + one-to-many heads.
        use_task_router: Integrate TaskRouter for adaptive task routing.
        task_router_dim: TaskRouter hidden dimension.
    """

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (),
        tasks: Optional[list[str]] = None,
        nm: int = 32,
        npr: int = 256,
        kpt_shape: tuple = (17, 3),
        depth_bins: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        use_task_router: bool = False,
        task_router_dim: Optional[int] = None,
    ):
        # Default: all tasks
        if tasks is None:
            tasks = ["detect", "segment", "pose", "classify", "depth", "obb"]
        self._active_tasks = set(tasks)
        self._use_task_router = use_task_router

        # Initialize Detect base (detection always present)
        super().__init__(nc, reg_max, end2end, ch)

        # ── Task-specific heads ──────────────────────────────────────────
        self._build_segment_head(ch, nm, npr, end2end)
        self._build_pose_head(ch, kpt_shape, end2end)
        self._build_classify_head(ch, end2end)
        self._build_depth_head(ch, depth_bins, end2end)
        self._build_obb_head(ch, end2end)

        # ── TaskRouter (optional) ────────────────────────────────────────
        if use_task_router:
            from .router import TaskRouter

            router_dim = task_router_dim or ch[0]
            self.task_router = TaskRouter(
                dim=router_dim,
                num_tasks=len(self._active_tasks),
                top_k=2,
                shared_expert_ratio=0.2,
            )
            # Per-scale projectors: shared_feats → each FPN scale channel
            shared_c = int(router_dim * 0.2)
            self.task_router_proj = nn.ModuleList([
                nn.Conv2d(shared_c, c, 1, bias=False) for c in ch
            ])
        else:
            self.task_router = None
            self.task_router_proj = None

        # ── Task importance scores (learnable) ───────────────────────────
        num_active = len(self._active_tasks)
        self.task_importance = nn.Parameter(torch.zeros(num_active))

    # ── Task builders ────────────────────────────────────────────────────
    def _build_segment_head(self, ch, nm, npr, end2end):
        if "segment" not in self._active_tasks:
            self.nm = 0
            self.proto = None
            self.cv4_seg = None
            return
        self.nm = nm
        self.npr = npr
        self.proto = Proto26(ch, npr, nm, self.nc)
        c4 = max(ch[0] // 4, nm)
        self.cv4_seg = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, nm, 1)) for x in ch
        )
        if end2end:
            self.one2one_cv4_seg = copy.deepcopy(self.cv4_seg)

    def _build_pose_head(self, ch, kpt_shape, end2end):
        if "pose" not in self._active_tasks:
            self.kpt_shape = (0, 0)
            self.nk = 0
            self.cv4_pose = None
            return
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        c4 = max(ch[0] // 4, self.nk)
        self.cv4_pose = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch
        )
        if end2end:
            self.one2one_cv4_pose = copy.deepcopy(self.cv4_pose)

    def _build_classify_head(self, ch, end2end):
        if "classify" not in self._active_tasks:
            self.cv4_cls = None
            self.global_pool = None
            return
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        cls_in = ch[-1]  # deepest feature
        self.cv4_cls = nn.Sequential(
            Conv(cls_in, cls_in // 2, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cls_in // 2, self.nc),
        )
        # Classification head doesn't use end2end

    def _build_depth_head(self, ch, depth_bins, end2end):
        if "depth" not in self._active_tasks:
            self.depth_bins = 0
            self.cv4_depth = None
            return
        self.depth_bins = depth_bins
        c4 = max(ch[0] // 4, depth_bins)
        self.cv4_depth = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, depth_bins, 1)) for x in ch
        )
        if end2end:
            self.one2one_cv4_depth = copy.deepcopy(self.cv4_depth)

    def _build_obb_head(self, ch, end2end):
        if "obb" not in self._active_tasks:
            self.ne = 0
            self.cv4_obb = None
            return
        self.ne = 1  # angle
        c4 = max(ch[0] // 4, self.ne)
        self.cv4_obb = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch
        )
        if end2end:
            self.one2one_cv4_obb = copy.deepcopy(self.cv4_obb)

    # ── Property helpers ─────────────────────────────────────────────────
    @property
    def active_tasks(self) -> list[str]:
        return sorted(self._active_tasks)

    def has_task(self, task: str) -> bool:
        return task in self._active_tasks

    @property
    def one2many(self):
        d = dict(box_head=self.cv2, cls_head=self.cv3)
        if self.has_task("segment"):
            d["mask_head"] = self.cv4_seg
        if self.has_task("pose"):
            d["pose_head"] = self.cv4_pose
        if self.has_task("depth"):
            d["depth_head"] = self.cv4_depth
        if self.has_task("obb"):
            d["obb_head"] = self.cv4_obb
        return d

    @property
    def one2one(self):
        d = dict()
        if self.has_task("segment") and hasattr(self, "one2one_cv4_seg"):
            d["mask_head"] = self.one2one_cv4_seg
        if self.has_task("pose") and hasattr(self, "one2one_cv4_pose"):
            d["pose_head"] = self.one2one_cv4_pose
        if self.has_task("depth") and hasattr(self, "one2one_cv4_depth"):
            d["depth_head"] = self.one2one_cv4_depth
        if self.has_task("obb") and hasattr(self, "one2one_cv4_obb"):
            d["obb_head"] = self.one2one_cv4_obb
        if hasattr(self, "one2one_cv2"):
            d["box_head"] = self.one2one_cv2
            d["cls_head"] = self.one2one_cv3
        return d

    # ── Core forward ─────────────────────────────────────────────────────
    def forward_head(
        self, x: list[torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Forward through task-specific heads.

        Args:
            x: List of feature maps from FPN [P3, P4, P5].
            **kwargs: head component overrides (box_head, cls_head, mask_head, etc.)

        Returns:
            dict with keys: boxes, scores, feats, and optionally:
                mask_coefficient, proto, kpts, depth, angle, cls_logits
        """
        box_head = kwargs.get("box_head", self.cv2)
        cls_head = kwargs.get("cls_head", self.cv3)

        # Base detection forward
        preds = super().forward_head(x, box_head=box_head, cls_head=cls_head)
        bs = x[0].shape[0]

        # ── TaskRouter integration ───────────────────────────────────────
        if self.task_router is not None and self.training:
            # Use P3 features (highest resolution) for routing
            task_feats, shared_feats, routing_stats = self.task_router(x[0])
            preds["routing_stats"] = routing_stats
            # Fuse shared features back into each scale (with per-scale projection)
            if shared_feats is not None:
                x_mod = [
                    xi + self.task_router_proj[i](
                        F.interpolate(shared_feats, size=xi.shape[2:], mode="bilinear", align_corners=False)
                    )
                    for i, xi in enumerate(x)
                ]
            else:
                x_mod = x
        else:
            x_mod = x

        # ── Segmentation ─────────────────────────────────────────────────
        if self.has_task("segment"):
            mask_head = kwargs.get("mask_head", self.cv4_seg)
            if mask_head is not None:
                preds["mask_coefficient"] = torch.cat(
                    [mask_head[i](x_mod[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2
                )
                if self.training:
                    preds["proto"] = self.proto(x_mod) if self.proto is not None else None

        # ── Pose ─────────────────────────────────────────────────────────
        if self.has_task("pose"):
            pose_head = kwargs.get("pose_head", self.cv4_pose)
            if pose_head is not None:
                preds["kpts"] = torch.cat(
                    [pose_head[i](x_mod[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2
                )

        # ── Depth ────────────────────────────────────────────────────────
        if self.has_task("depth"):
            depth_head = kwargs.get("depth_head", self.cv4_depth)
            if depth_head is not None:
                preds["depth"] = torch.cat(
                    [depth_head[i](x_mod[i]).view(bs, self.depth_bins, -1) for i in range(self.nl)], 2
                )

        # ── OBB ──────────────────────────────────────────────────────────
        if self.has_task("obb"):
            obb_head = kwargs.get("obb_head", self.cv4_obb)
            if obb_head is not None:
                preds["angle"] = torch.cat(
                    [obb_head[i](x_mod[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
                )

        # ── Classification ───────────────────────────────────────────────
        if self.has_task("classify") and self.cv4_cls is not None:
            preds["cls_logits"] = self.cv4_cls(x_mod[-1])  # global image-level classification

        return preds

    def forward(self, x: list[torch.Tensor]):
        """Unified multi-task forward pass.

        Training: returns dict with one2many + one2one predictions.
        Inference: returns (detections, task_outputs) tuple.
        """
        # Detection forward (base)
        preds = self.forward_head(x, **self.one2many)

        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one_kwargs = {k: v for k, v in self.one2one.items() if v is not None}
            one2one = self.forward_head(x_detach, **one2one_kwargs)
            preds = {"one2many": preds, "one2one": one2one}

        if self.training:
            return preds

        # Inference: decode detections (standard format)
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode boxes + scores. In eval mode, return standard Detection format (no extra dims)."""
        preds = super()._inference(x)

        # Training-only: append task-specific predictions
        if not self.training:
            return preds

        extra_parts = []
        if "mask_coefficient" in x:
            extra_parts.append(x["mask_coefficient"])
        if "kpts" in x:
            extra_parts.append(x["kpts"])
        if "angle" in x:
            extra_parts.append(x["angle"])
        if "depth" in x:
            extra_parts.append(x["depth"])

        if extra_parts:
            return torch.cat([preds] + extra_parts, dim=1)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Standard end2end postprocess: delegate to Detect."""
        return super().postprocess(preds)

    def bias_init(self):
        """Initialize biases for all head branches."""
        super().bias_init()
        # Additional init for task branches can be added here

    def fuse(self) -> None:
        """Fuse for inference: remove one2many heads."""
        self.cv2 = self.cv3 = None
        self.cv4_seg = None
        self.cv4_pose = None
        self.cv4_depth = None
        self.cv4_obb = None
