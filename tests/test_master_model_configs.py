"""Build and minimal-forward regression tests for Master configurations."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    [
        "ultralytics/cfg/models/26/yolo26-master-n.yaml",
        "ultralytics/cfg/models/26/yolo26.yaml",
        "ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-n.yaml",
        "ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-n.yaml",
    ],
)
def test_master_config_builds_and_forwards(relative_path):
    model = YOLO(ROOT / relative_path).model
    with torch.no_grad():
        result = model(torch.zeros(1, 3, 64, 64))
    assert result is not None


def test_yolo26_master_uses_current_sppf_signature():
    text = (ROOT / "ultralytics/cfg/models/26/yolo26-master-n.yaml").read_text()
    assert "SPPF, [1024, 5]" in text
