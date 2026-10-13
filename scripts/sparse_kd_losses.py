"""Explicit experimental losses for foreground, localization and teacher-oracle routing KD."""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F


def region_weights(batch, shape, foreground=4.0, small_multiplier=2.0, small_area=32.0**2):
    """Mark cells intersecting training GT boxes; small means transformed input-pixel area."""
    if foreground < 1 or small_multiplier < 1 or small_area <= 0:
        raise ValueError("region weights require foreground/small weights >= 1 and positive area")
    height, width = shape
    images = batch["img"]
    weights = images.new_ones((images.shape[0], 1, height, width))
    image_height, image_width = images.shape[-2:]
    boxes = batch["bboxes"].detach().cpu().tolist()
    indices = batch["batch_idx"].detach().cpu().tolist()
    for index, (x, y, box_width, box_height) in zip(indices, boxes):
        if box_width <= 0 or box_height <= 0:
            continue
        if x + box_width / 2 <= 0 or y + box_height / 2 <= 0 or x - box_width / 2 >= 1 or y - box_height / 2 >= 1:
            continue
        left = max(0, min(width - 1, math.floor((x - box_width / 2) * width)))
        top = max(0, min(height - 1, math.floor((y - box_height / 2) * height)))
        right = max(left + 1, min(width, math.ceil((x + box_width / 2) * width)))
        bottom = max(top + 1, min(height, math.ceil((y + box_height / 2) * height)))
        small = box_width * image_width * box_height * image_height < small_area
        value = foreground * (small_multiplier if small else 1.0)
        region = weights[int(index), :, top:bottom, left:right]
        region.copy_(torch.maximum(region, region.new_tensor(value)))
    return weights


def feature_distillation(student, teacher, weights=None):
    """Normalize weighted feature error by teacher energy; detach all teacher supervision."""
    if student.shape != teacher.shape:
        raise ValueError("teacher/student feature shapes must match")
    target = teacher.detach()
    if weights is None:
        weights = torch.ones_like(target[:, :1])
    return ((student - target).square() * weights).mean() / (target.square() * weights).mean().clamp_min(1e-8)


def raw_detection(output):
    """Require the same dense raw detection representation for aligned localization KD."""
    raw = output[1] if isinstance(output, tuple) else output
    if not isinstance(raw, dict) or not {"boxes", "scores", "feats"} <= set(raw):
        raise TypeError("localization KD requires aligned dense detection logits and feature maps")
    return raw


def localization_distillation(student_output, teacher_output, batch):
    """Distill pre-NMS box-bin distributions on GT/teacher-confident aligned anchors."""
    student, teacher = raw_detection(student_output), raw_detection(teacher_output)
    boxes, target_boxes = student["boxes"], teacher["boxes"].detach()
    if boxes.shape != target_boxes.shape or boxes.shape[1] % 4 or boxes.shape[1] <= 4:
        raise ValueError("localization KD requires matching four-sided distribution logits")
    if len(student["feats"]) != len(teacher["feats"]):
        raise ValueError("teacher/student detection scales must match")
    masks = []
    for student_feature, teacher_feature in zip(student["feats"], teacher["feats"]):
        if student_feature.shape != teacher_feature.shape:
            raise ValueError("teacher/student detection grids must align")
        masks.append((region_weights(batch, teacher_feature.shape[-2:]) > 1).flatten(1))
    foreground = torch.cat(masks, dim=1)
    confidence = teacher["scores"].detach().sigmoid().amax(dim=1)
    if foreground.shape != confidence.shape or confidence.shape[-1] != boxes.shape[-1]:
        raise ValueError("localization anchor counts must align")
    weights = (foreground | (confidence >= 0.25)).to(boxes.dtype) * confidence.clamp_min(0.05)
    batch_size, channels, anchors = boxes.shape
    prediction = boxes.reshape(batch_size, 4, channels // 4, anchors)
    target = target_boxes.reshape_as(prediction)
    kl = F.kl_div(prediction.log_softmax(dim=2), target.softmax(dim=2), reduction="none").sum(2).mean(1)
    return (kl * weights).sum() / weights.sum().clamp_min(1e-8)


def expert_pairs(count):
    """Enumerate local Top-2 supports in a fixed expert-index order."""
    if count < 2:
        raise ValueError("Top-2 oracle requires at least two experts")
    return tuple(itertools.combinations(range(count), 2))


@torch.no_grad()
def oracle_pair_costs(block, expert_outputs, routing, target):
    """Return per-image feature errors for every pair before collapsing to hard labels."""
    if block.training:
        raise ValueError("teacher oracle requires eval mode with frozen normalization")
    pairs = expert_pairs(block.num_experts)
    if len(expert_outputs) != block.num_experts:
        raise ValueError("oracle needs every frozen teacher expert output")
    errors = []
    for left, right in pairs:
        left_weight, right_weight = routing[:, left : left + 1], routing[:, right : right + 1]
        normalizer = (left_weight + right_weight).clamp_min(1e-8)
        mixture = (expert_outputs[left] * left_weight + expert_outputs[right] * right_weight) / normalizer
        output = block.norm(mixture)
        errors.append((output - target).square().flatten(1).mean(1))
    return torch.stack(errors, dim=1)


def oracle_pair_targets(block, expert_outputs, routing, target):
    """Choose teacher-expert pairs minimizing native layer-feature error, not detection AP."""
    return oracle_pair_costs(block, expert_outputs, routing, target).argmin(dim=1)


def router_distillation(routing, targets):
    """Fit raw student routing probabilities to frozen teacher feature-oracle pair labels."""
    probabilities = routing.mean(dim=(2, 3)).clamp_min(1e-8)
    pair_logits = torch.stack(
        [
            probabilities[:, left].log() + probabilities[:, right].log()
            for left, right in expert_pairs(probabilities.shape[1])
        ],
        dim=1,
    )
    return F.cross_entropy(pair_logits, targets.detach()), pair_logits.argmax(dim=1)
