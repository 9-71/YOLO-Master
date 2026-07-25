"""Regression coverage for released YOLOE segmentation checkpoint reconstruction."""

import os
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.tasks import YOLOEModel, YOLOESegModel


ROOT = Path(os.environ.get("YOLO_MASTER_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
DETECT_CFG = ROOT / "ultralytics/cfg/models/11/yoloe-11.yaml"
SEGMENT_CFG = ROOT / "ultralytics/cfg/models/11/yoloe-11-seg.yaml"
SPPF_INDEX = 9


def _checkpoint_like_segmentation_source_and_detection_target():
    """Build the legacy execution metadata pattern without downloading checkpoint assets."""
    source = YOLOESegModel(str(SEGMENT_CFG), verbose=False).eval()
    target = YOLOEModel(str(DETECT_CFG), verbose=False).eval()
    source.model[SPPF_INDEX].cv1.act = nn.SiLU(inplace=True)
    return source, target


def _assert_target_state_was_fully_shared(source, target):
    source_state, target_state = source.state_dict(), target.state_dict()
    assert all(name in source_state and torch.equal(value, source_state[name]) for name, value in target_state.items())


def test_released_segmentation_load_restores_shared_sppf_execution_semantics():
    """A released segmentation source must preserve SiLU when reconstructing the shared detection path."""
    torch.manual_seed(0)
    source, target = _checkpoint_like_segmentation_source_and_detection_target()
    source_sppf, target_sppf = source.model[SPPF_INDEX], target.model[SPPF_INDEX]
    x = torch.randn(1, source_sppf.cv1.conv.in_channels, 8, 8)

    assert isinstance(source_sppf.cv1.act, nn.SiLU)
    assert isinstance(target_sppf.cv1.act, nn.Identity)
    with torch.inference_mode():
        target.load({"model": source}, verbose=False)

        _assert_target_state_was_fully_shared(source, target)
        assert isinstance(target_sppf.cv1.act, nn.SiLU)
        assert torch.equal(source_sppf(x), target_sppf(x))


def test_detection_constructor_and_detection_source_keep_current_sppf_identity_semantics():
    """The migration is not a global SPPF default change and does not apply to detection sources."""
    source = YOLOEModel(str(DETECT_CFG), verbose=False).eval()
    target = YOLOEModel(str(DETECT_CFG), verbose=False).eval()
    source.model[SPPF_INDEX].cv1.act = nn.SiLU(inplace=True)

    target.load({"model": source}, verbose=False)

    assert isinstance(YOLOEModel(str(DETECT_CFG), verbose=False).model[SPPF_INDEX].cv1.act, nn.Identity)
    assert isinstance(target.model[SPPF_INDEX].cv1.act, nn.Identity)


def test_released_segmentation_activation_migration_fails_closed_on_sppf_state_layout_mismatch():
    """A source with a different SPPF state layout must not transfer non-state execution metadata."""
    source, target = _checkpoint_like_segmentation_source_and_detection_target()
    source_sppf = source.model[SPPF_INDEX]
    source_sppf.cv1.conv = nn.Conv2d(
        source_sppf.cv1.conv.in_channels,
        source_sppf.cv1.conv.out_channels + 1,
        kernel_size=1,
        bias=False,
    )

    migrated = target._migrate_released_segmentation_execution_semantics(source)

    assert migrated == ()
    assert isinstance(target.model[SPPF_INDEX].cv1.act, nn.Identity)
