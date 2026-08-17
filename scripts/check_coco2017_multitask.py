"""Preflight official COCO 2017 detection, instance-segmentation, and pose labels.

This command is read-only. It reports exactly which files and split-level
contracts are available for the F15 real-data gate and exits with status 2 when
the gate is not ready. It does not download data or generate manifests.

Example::

    python scripts/check_coco2017_multitask.py \
        --dataset-root /path/to/coco2017 \
        --output reports/foundation/v0.1/coco2017-preflight.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ROOT = Path(os.environ.get("YOLO_MASTER_COCO_ROOT", Path.home() / "datasets" / "coco2017"))
SPLITS = ("train2017", "val2017")
REQUIRED_FILES = (
    "instances_{split}.json",
    "person_keypoints_{split}.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one COCO annotation file with a bounded diagnostic."""
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"COCO annotation root must be an object: {path}")
    return payload


def _polygon_count(annotations: list[dict[str, Any]]) -> int:
    """Count non-crowd annotations containing at least one valid polygon."""
    return sum(
        not bool(annotation.get("iscrowd", 0))
        and isinstance(annotation.get("segmentation"), list)
        and any(isinstance(poly, list) and len(poly) >= 6 for poly in annotation["segmentation"])
        for annotation in annotations
    )


def check_split(root: Path, split: str) -> dict[str, Any]:
    """Check one COCO split and return counts plus actionable failures."""
    annotations_dir = root / "annotations"
    image_dir = root / "images" / split
    result: dict[str, Any] = {
        "split": split,
        "image_dir": str(image_dir),
        "annotation_files": {},
        "ready": False,
        "errors": [],
    }
    for pattern in REQUIRED_FILES:
        filename = pattern.format(split=split)
        path = annotations_dir / filename
        result["annotation_files"][filename] = {"path": str(path), "exists": path.is_file()}
        if not path.is_file():
            result["errors"].append(f"missing required annotation: {path}")
    if not image_dir.is_dir():
        result["errors"].append(f"missing image directory: {image_dir}")
    if result["errors"]:
        return result

    instances = _read_json(annotations_dir / f"instances_{split}.json")
    keypoints = _read_json(annotations_dir / f"person_keypoints_{split}.json")
    images = instances.get("images", [])
    instance_annotations = instances.get("annotations", [])
    categories = instances.get("categories", [])
    keypoint_annotations = keypoints.get("annotations", [])
    if not isinstance(images, list) or not isinstance(instance_annotations, list) or not isinstance(categories, list):
        result["errors"].append("instances annotation lists are malformed")
        return result
    if not isinstance(keypoint_annotations, list):
        result["errors"].append("keypoint annotation list is malformed")
        return result

    image_ids = {int(item["id"]) for item in images if isinstance(item, dict) and "id" in item}
    instance_image_ids = {
        int(item["image_id"]) for item in instance_annotations if isinstance(item, dict) and "image_id" in item
    }
    keypoint_image_ids = {
        int(item["image_id"]) for item in keypoint_annotations if isinstance(item, dict) and "image_id" in item
    }
    image_names = {str(item["file_name"]) for item in images if isinstance(item, dict) and item.get("file_name")}
    present_names = {path.name for path in image_dir.glob("*.jpg")}
    missing_images = sorted(image_names - present_names)
    non_crowd = [item for item in instance_annotations if isinstance(item, dict) and not item.get("iscrowd", 0)]
    pose_with_keypoints = sum(
        int(item.get("num_keypoints", 0)) > 0 for item in keypoint_annotations if isinstance(item, dict)
    )
    result["counts"] = {
        "images_declared": len(image_ids),
        "images_present": len(image_names & present_names),
        "images_missing": len(missing_images),
        "instance_annotations": len(instance_annotations),
        "non_crowd_instances": len(non_crowd),
        "polygon_instances": _polygon_count([item for item in non_crowd if isinstance(item, dict)]),
        "keypoint_annotations": len(keypoint_annotations),
        "keypoint_images": len(keypoint_image_ids),
        "keypoints_with_visible_points": pose_with_keypoints,
        "categories": len(categories),
    }
    if missing_images:
        result["errors"].append(f"{len(missing_images)} declared images are missing from {image_dir}")
    if not image_ids or not instance_image_ids.intersection(image_ids):
        result["errors"].append("instances annotations contain no image-linked records")
    if not keypoint_image_ids.intersection(image_ids) or pose_with_keypoints == 0:
        result["errors"].append("person keypoints contain no visible pose supervision")
    if result["counts"]["polygon_instances"] == 0:
        result["errors"].append("instances annotations contain no polygon segmentation supervision")
    if result["counts"]["categories"] != 80:
        result["errors"].append(f"expected 80 COCO categories, found {result['counts']['categories']}")
    result["ready"] = not result["errors"]
    return result


def preflight(dataset_root: Path) -> dict[str, Any]:
    """Return a JSON-safe F15 COCO readiness report without modifying the dataset."""
    root = dataset_root.expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "check": "f15_coco2017_multitask_preflight",
        "dataset_root": str(root),
        "required_tasks": ["detect", "segment", "pose"],
        "read_only": True,
        "ready": False,
        "splits": [],
        "errors": [],
    }
    if not root.is_dir():
        report["errors"].append(f"dataset root does not exist: {root}")
        return report
    report["splits"] = [check_split(root, split) for split in SPLITS]
    report["errors"] = [error for split in report["splits"] for error in split["errors"]]
    report["ready"] = not report["errors"] and all(split["ready"] for split in report["splits"])
    return report


def main() -> int:
    """Run the preflight and return 0 only when both splits are ready."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    report = preflight(args.dataset_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
