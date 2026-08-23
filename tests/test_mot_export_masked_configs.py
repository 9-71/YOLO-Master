"""Config-level regression: master MoT YAMLs must wire ``export_masked=True``.

The module-level ``export_masked`` path (masked Top-K router weights) makes
traced/exported MoT graphs bit-exact with eager sparse dispatch. These tests
pin the master configs so the bit-exact export semantics stay the product
default rather than an opt-in module capability.
"""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.mot import C2fMoT

ROOT = Path(__file__).resolve().parents[1]

MOT_MASTER_CONFIGS = [
    "ultralytics/cfg/models/26/yolo26-master-mot-n.yaml",
    "ultralytics/cfg/models/26/yolo26-master-moa-mot-n.yaml",
    "ultralytics/cfg/models/26/yolo26-master-mt-n.yaml",
    "ultralytics/cfg/models/master/v0_8/det/yolo-master-mot-n.yaml",
    "ultralytics/cfg/models/master/v0_8/det/yolo-master-moa-mot-n.yaml",
    "ultralytics/cfg/models/master/v0_8/det/yolo-master-moe-mot-shared-n.yaml",
    "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml",
    "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-mot-n.yaml",
    "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-scene-n.yaml",
]


def _c2f_mot_modules(model):
    return [module for module in model.modules() if isinstance(module, C2fMoT)]


@pytest.mark.parametrize("relative_path", MOT_MASTER_CONFIGS)
def test_mot_master_config_enables_export_masked(relative_path):
    model = YOLO(ROOT / relative_path).model
    wrappers = _c2f_mot_modules(model)
    assert wrappers, f"{relative_path} should contain C2fMoT modules"
    for wrapper in wrappers:
        assert wrapper.m, f"{relative_path}: C2fMoT has no MoTBlock children"
        for block in wrapper.m:
            assert block.router.export_masked is True, f"{relative_path}: MoTBlock router missing export_masked=True"
        capabilities = wrapper.export_capabilities()
        assert capabilities["export_router_weights"] == "masked_topk"


def test_config_built_c2f_mot_trace_matches_sparse_eager():
    """A C2fMoT built from a master config must trace to the sparse-eager output."""
    torch.manual_seed(0)
    model = YOLO(ROOT / "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml").model.eval()
    wrapper = _c2f_mot_modules(model)[0].eval()
    in_channels = wrapper.cv1.conv.in_channels
    sample = torch.randn(1, in_channels, 16, 16)
    with torch.no_grad():
        eager = wrapper(sample)
        traced = torch.jit.trace(wrapper, sample)
        exported = traced(sample)
    torch.testing.assert_close(exported, eager, rtol=0.0, atol=1e-6)
