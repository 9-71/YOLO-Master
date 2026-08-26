#!/usr/bin/env python3
"""Compute VisDrone mAP50-95 from per-image pixel-xyxy predictions.

The evaluator delegates matching and AP integration to Ultralytics' own
``DetMetrics`` so the result is comparable to ``model.val``. Predictions must
use the runner's ``class conf x1 y1 x2 y2`` format. Native VisDrone rows are
accepted for diagnostics; for the formal acceptance gate, use labels produced
by the official ``visdrone2yolo`` conversion so ignored-region matching is
defined by the dataset conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

NAMES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor",
}
CLASS_TABLES = {
    "visdrone": NAMES,
    "sku110k": {0: "object"},
}


def manifest_name(path: Path, root: Path) -> str:
    """Return a mount-independent, POSIX-formatted image path."""
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError:
        # Preserve a relative relationship for symlinked files outside root;
        # absolute mount paths would make equivalent runs hash differently.
        try:
            relative = Path(os.path.relpath(resolved_path, resolved_root))
        except ValueError:
            # Different Windows drives have no common relative spelling. Use
            # the basename rather than failing after metrics have been computed.
            relative = Path(resolved_path.name)
    return relative.as_posix()

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_gt(
    path: Path,
    width: int,
    height: int,
    torch,
    label_format: str = "auto",
    num_classes: Optional[int] = None,
):
    boxes, classes = [], []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            # Accept both YOLO whitespace rows and native VisDrone CSV rows.
            # Native ignored regions are dropped here; formal acceptance runs
            # should use visdrone2yolo-converted labels (see the module docstring).
            fields = line.replace(",", " ").split()
            if len(fields) < 5:
                continue
            try:
                values = [float(value) for value in fields]
            except ValueError:
                continue
            # Check the complete row before converting class/category IDs with
            # ``int``; ``int(float('inf'))`` raises OverflowError and should be
            # treated as a malformed annotation, not abort the whole report.
            if not np.isfinite(values).all():
                continue
            # Native VisDrone rows always carry at least eight fields
            # (x/y/w/h/score/category plus truncation/occlusion).  Use that
            # structural marker instead of a value-magnitude heuristic, which
            # misclassifies a tiny 1x1 raw box as a normalized YOLO row.
            use_visdrone = label_format == "visdrone" or (label_format == "auto" and len(values) >= 8)
            if use_visdrone:
                if len(values) < 6:
                    continue
                left, top, bw, bh = values[0:4]
                score = values[4]
                if values[5] != np.floor(values[5]):
                    continue
                category = int(values[5])
                if score <= 0 or category <= 0:
                    continue
                cls = category - 1
                box = [left, top, left + bw, top + bh]
            else:
                if values[0] != np.floor(values[0]):
                    continue
                cls = int(values[0])
                cx, cy, bw, bh = values[1:5]
                box = [(cx - bw / 2) * width, (cy - bh / 2) * height,
                       (cx + bw / 2) * width, (cy + bh / 2) * height]
            if num_classes is not None and not 0 <= cls < num_classes:
                # VisDrone category 11 ("others") and ignored/out-of-profile
                # classes are not part of the ten-class detection task.
                continue
            if not np.isfinite([cls, *box]).all() or bw <= 0 or bh <= 0 or cls < 0:
                continue
            classes.append(cls)
            boxes.append(box)
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(classes, dtype=torch.int64)


def load_predictions(path: Path, torch, num_classes: Optional[int] = None):
    boxes, scores, classes = [], [], []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 6:
                continue
            try:
                cls_value = float(fields[0]); score = float(fields[1])
                box = [float(value) for value in fields[2:6]]
            except (ValueError, OverflowError):
                continue
            if not np.isfinite([cls_value, score, *box]).all():
                continue
            if cls_value != np.floor(cls_value):
                continue
            cls = int(cls_value)
            if cls < 0 or (num_classes is not None and cls >= num_classes):
                raise ValueError(f"prediction class {cls} outside [0, {num_classes}) in {path}")
            # Keep the evaluator's candidate contract aligned with the C++ and
            # MNN decoders: degenerate pixel boxes are discarded before IoU.
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            classes.append(cls); scores.append(score); boxes.append(box)
    return (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(scores, dtype=torch.float32), torch.tensor(classes, dtype=torch.int64))


def match_predictions(pred_cls, true_cls, iou, torch, iou_thresholds):
    """Ultralytics-compatible greedy class-aware matching for 0.50:0.95 IoU."""
    correct = np.zeros((pred_cls.shape[0], len(iou_thresholds)), dtype=bool)
    correct_class = true_cls[:, None] == pred_cls
    iou_np = (iou * correct_class).cpu().numpy()
    # Convert Torch scalars to Python floats before comparing with the NumPy
    # IoU matrix.  This avoids version-dependent ndarray/Tensor dispatch while
    # preserving the 0.50:0.95 thresholds exactly.
    for column, threshold in enumerate(iou_thresholds.tolist()):
        matches = np.array(np.nonzero(iou_np >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), column] = True
    return torch.tensor(correct)


def nonnegative_finite_float(value: str) -> float:
    """Parse a finite, non-negative percentage budget for the CLI."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite non-negative number") from exc
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def delta_gate_passes(abs_delta_pct: float, max_abs_delta_pct: Optional[float]) -> bool:
    """Return whether an observed relative mAP delta is within the budget.

    ``None`` means that no gate was requested.  Budgets are expressed as
    percentage points of the reference mAP (for example, ``0.5`` means 0.5%).
    The comparison is inclusive so a result exactly on the declared maximum
    is accepted.
    """
    if max_abs_delta_pct is None:
        return True
    return float(abs_delta_pct) <= float(max_abs_delta_pct)


def validate_delta_budget(
    max_abs_delta_pct: Optional[float], reference_json: Optional[Path]
) -> None:
    """Reject a requested gate that cannot be compared with a reference."""
    if max_abs_delta_pct is not None and (
        not np.isfinite(float(max_abs_delta_pct)) or float(max_abs_delta_pct) < 0
    ):
        raise ValueError("--max-abs-delta-pct must be a finite non-negative number")
    if max_abs_delta_pct is not None and reference_json is None:
        raise ValueError("--max-abs-delta-pct requires --reference-json")


def validate_smoke_gate(smoke: bool, max_abs_delta_pct: Optional[float]) -> None:
    """Keep an acceptance delta gate from being attached to a smoke subset."""
    if smoke and max_abs_delta_pct is not None:
        raise ValueError(
            "--max-abs-delta-pct cannot be used with --smoke; smoke runs are not acceptance evidence"
        )


def validate_acceptance_image_floor(args: argparse.Namespace) -> None:
    """Keep the 500-image acceptance floor from being weakened by a CLI flag."""
    if not args.smoke and args.min_images < 500:
        raise ValueError(
            "Issue #51 acceptance requires --min-images >= 500; use --smoke for a smaller diagnostic run"
        )


def apply_delta_gate(result, max_abs_delta_pct: Optional[float]) -> int:
    """Annotate ``result`` and return the process code for the optional gate."""
    if max_abs_delta_pct is None:
        return 0
    observed = result.get("abs_delta_mAP50-95_pct")
    if observed is None:
        raise ValueError("--max-abs-delta-pct requires a computed reference delta")
    passed = delta_gate_passes(observed, max_abs_delta_pct)
    result["max_abs_delta_mAP50-95_pct"] = max_abs_delta_pct
    result["mAP50-95_delta_gate_passed"] = passed
    return 0 if passed else 2


def extract_reference_map(payload: dict) -> float:
    """Extract mAP50-95 from common Ultralytics result JSON layouts."""
    if not isinstance(payload, dict):
        raise ValueError("reference JSON must contain an object")
    direct_keys = (
        "mAP50-95",
        "map50-95",
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95",
    )
    mappings = [payload]
    for key in ("metrics", "results", "results_dict", "metrics_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            mappings.append(value)
    for mapping in mappings:
        for key in direct_keys:
            if key in mapping:
                try:
                    return float(mapping[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"reference mAP50-95 is not numeric: {mapping[key]!r}") from exc
        # Be tolerant of case differences while retaining the metric suffix
        # used by Ultralytics for box results.
        for key, value in mapping.items():
            normalized = str(key).lower().replace(" ", "")
            if normalized in {"map50-95", "metrics/map50-95(b)", "metrics/map50-95"}:
                try:
                    return float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"reference mAP50-95 is not numeric: {value!r}") from exc
    raise ValueError(
        "reference JSON must contain mAP50-95, map50-95, or metrics/mAP50-95(B)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("/data/datasets/VisDrone/images/val"))
    parser.add_argument("--labels", type=Path, default=Path("/data/datasets/VisDrone/labels/val"))
    parser.add_argument("--classes", choices=tuple(CLASS_TABLES), default="visdrone")
    parser.add_argument(
        "--label-format", choices=("auto", "yolo", "visdrone"), default="auto",
        help=(
            "YOLO normalized labels or raw VisDrone x/y/w/h/score/category rows "
            "(native rows are diagnostic only)"
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--smoke", action="store_true",
        help="allow fewer than --min-images for a dependency/data smoke check (not an Issue #51 acceptance run)",
    )
    parser.add_argument("--min-images", type=int, default=500, help="minimum images outside --smoke (default: 500)")
    parser.add_argument("--json", type=Path, help="optional JSON result path")
    parser.add_argument(
        "--reference-json", type=Path,
        help="optional PyTorch/reference JSON containing mAP50-95 for a relative delta",
    )
    parser.add_argument(
        "--max-abs-delta-pct",
        type=nonnegative_finite_float,
        default=None,
        metavar="PERCENT",
        help=(
            "fail with a non-zero status when the absolute relative mAP50-95 "
            "delta exceeds PERCENT (for example, 0.5 for a 0.5%% budget); "
            "requires --reference-json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_delta_budget(args.max_abs_delta_pct, args.reference_json)
    validate_smoke_gate(args.smoke, args.max_abs_delta_pct)
    if not args.images.is_dir():
        raise FileNotFoundError(f"image directory not found: {args.images}")
    if not args.preds.is_dir():
        raise FileNotFoundError(f"prediction directory not found: {args.preds}")
    if args.limit < 0 or args.min_images < 1:
        raise ValueError("limit must be non-negative and min-images must be positive")
    validate_acceptance_image_floor(args)

    import torch
    from PIL import Image
    from ultralytics.utils.metrics import DetMetrics, box_iou

    iou_thresholds = torch.linspace(0.5, 0.95, 10)
    # Match the portable C++ runner's stb decoder so every backend evaluates
    # the same ordered image set.
    images = sorted(
        path for path in args.images.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"no validation images found under {args.images}")
    stems = [image_path.stem for image_path in images]
    if len(stems) != len(set(stems)):
        raise RuntimeError("validation image stems are not unique; use a flattened/renamed validation split")
    if not args.smoke and len(images) < args.min_images:
        raise RuntimeError(
            f"Issue #51 acceptance requires at least {args.min_images} images; found {len(images)}. "
            "Use --smoke only for a non-acceptance check."
        )

    names = CLASS_TABLES[args.classes]
    metrics = DetMetrics()
    metrics.names = names
    for image_index, image_path in enumerate(images):
        label_path = args.labels / f"{image_path.stem}.txt"
        prediction_path = args.preds / f"{image_path.stem}.txt"
        if not args.smoke:
            missing = [str(path) for path in (label_path, prediction_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "acceptance requires one label and prediction file per image; missing: "
                    + ", ".join(missing)
                )
        with Image.open(image_path) as image:
            width, height = image.size
        gt_boxes, gt_classes = load_gt(
            label_path, width, height, torch, args.label_format, len(names)
        )
        pred_boxes, pred_scores, pred_classes = load_predictions(prediction_path, torch, len(names))
        n_pred, n_true = pred_boxes.shape[0], gt_boxes.shape[0]
        if n_pred and n_true:
            true_positive = match_predictions(
                pred_classes, gt_classes, box_iou(gt_boxes, pred_boxes), torch, iou_thresholds
            ).cpu().numpy()
        else:
            true_positive = np.zeros((n_pred, len(iou_thresholds)), dtype=bool)
        metrics.update_stats({
            "tp": true_positive,
            "target_cls": gt_classes.numpy(),
            # One image index per GT instance is required by DetMetrics. Using
            # class IDs here under-counts/over-counts images with mixed classes.
            "target_img": np.full(n_true, image_index, dtype=np.int64),
            "conf": pred_scores.numpy(),
            "pred_cls": pred_classes.numpy(),
            # Newer Ultralytics versions use this key for per-image metrics;
            # older versions ignore unknown dictionary entries.
            "im_name": image_path.name,
        })

    metrics.process()
    manifest_names = [manifest_name(path, args.images) for path in images]
    image_manifest = "\n".join(manifest_names) + "\n"
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    if not np.isfinite([map50, map5095]).all():
        raise RuntimeError("mAP computation returned NaN or Inf; check labels and predictions")
    result = {
        "images": len(images),
        "classes": len(names),
        "class_profile": args.classes,
        "label_format": args.label_format,
        "image_manifest_sha256": hashlib.sha256(image_manifest.encode("utf-8")).hexdigest(),
        "mAP50": map50,
        "mAP50-95": map5095,
    }
    if args.reference_json:
        if not args.reference_json.is_file():
            raise FileNotFoundError(f"reference JSON not found: {args.reference_json}")
        reference = json.loads(args.reference_json.read_text(encoding="utf-8"))
        reference_map = extract_reference_map(reference)
        if not np.isfinite(reference_map) or reference_map <= 0:
            raise ValueError("reference mAP50-95 must be a finite positive value for a relative delta")
        result["reference_mAP50-95"] = reference_map
        delta_pct = float((result["mAP50-95"] - reference_map) / reference_map * 100.0)
        result["delta_mAP50-95_pct"] = delta_pct
        result["abs_delta_mAP50-95_pct"] = abs(delta_pct)
    gate_exit_code = apply_delta_gate(result, args.max_abs_delta_pct)
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if gate_exit_code:
        print(
            "mAP50-95 delta gate failed: "
            f"observed {result['abs_delta_mAP50-95_pct']:.6g}% > "
            f"maximum {args.max_abs_delta_pct:.6g}%",
            file=sys.stderr,
        )
    return gate_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
