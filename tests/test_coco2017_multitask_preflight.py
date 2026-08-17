"""Tests for the read-only COCO 2017 multi-task preflight."""

import json
from pathlib import Path

from scripts.check_coco2017_multitask import preflight
from scripts.prepare_coco2017_unified_multitask import write_training_yaml
from ultralytics.utils import YAML


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _dataset(tmp_path: Path, *, complete: bool) -> Path:
    root = tmp_path / "coco2017"
    annotations = root / "annotations"
    (root / "images" / "train2017").mkdir(parents=True)
    (root / "images" / "val2017").mkdir(parents=True)
    (root / "images" / "train2017" / "000000000001.jpg").write_bytes(b"jpeg")
    (root / "images" / "val2017" / "000000000001.jpg").write_bytes(b"jpeg")
    categories = [{"id": index + 1, "name": f"class-{index + 1}"} for index in range(80)]
    instances = {
        "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 8, "height": 8}],
        "annotations": [
            {
                "id": 7,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 4, 4],
                "iscrowd": 0,
                "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]],
            }
        ],
        "categories": categories,
    }
    keypoints = {
        "images": instances["images"],
        "annotations": [{"id": 7, "image_id": 1, "num_keypoints": 1, "keypoints": [1, 1, 2]}],
        "categories": [{"id": 1, "name": "person", "keypoints": ["nose"]}],
    }
    for split in ("train2017", "val2017"):
        _write_json(annotations / f"instances_{split}.json", instances)
        if complete or split == "train2017":
            _write_json(annotations / f"person_keypoints_{split}.json", keypoints)
    return root


def test_complete_dataset_passes_all_three_task_checks(tmp_path):
    report = preflight(_dataset(tmp_path, complete=True))

    assert report["ready"] is True
    assert report["errors"] == []
    assert all(split["ready"] for split in report["splits"])
    assert report["splits"][0]["counts"]["polygon_instances"] == 1


def test_missing_pose_annotation_fails_closed_without_mutating_dataset(tmp_path):
    root = _dataset(tmp_path, complete=False)
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))

    report = preflight(root)

    assert report["ready"] is False
    assert any("person_keypoints_val2017.json" in error for error in report["errors"])
    after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    assert before == after


def test_generated_f15_training_yaml_declares_only_built_heads(tmp_path):
    root = _dataset(tmp_path, complete=True)
    output_dir = root / "unified_multitask_f15"
    output_dir.mkdir()

    config_path = write_training_yaml(root, output_dir, {index: f"class-{index}" for index in range(80)})
    config = YAML.load(config_path)

    assert config["tasks"] == ["detect", "segment", "pose"]
    assert "classify" not in config["tasks"]
    assert "depth" not in config["tasks"]
    assert "normal" not in config["tasks"]
