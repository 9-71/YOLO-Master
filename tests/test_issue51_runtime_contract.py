"""Contract tests for the issue #51 export and validation protocol.

The tests exercise dependency-light helpers only. Optional runtime bindings
such as MNN and NCNN are imported by the command-line entry points after input
validation, so the contract suite remains runnable on a standard CI worker.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EDGE = ROOT / "examples" / "YOLO-Master-Cross-Platform-Edge-Deployment"


def load_module(name: str, path: Path):
    """Load a script module without importing optional runtime packages."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_selection_enforces_issue_floor(tmp_path):
    """Calibration selection is deterministic and requires at least 300 images."""
    quant = load_module("issue51_quant", EDGE / "scripts" / "quantize_int8.py")
    with pytest.raises(ValueError, match="300"):
        quant.select_calibration_images(tmp_path, 299)

    for index in range(300):
        (tmp_path / f"{index:04d}.jpg").touch()
    selected = quant.select_calibration_images(tmp_path, 300)
    assert len(selected) == 300
    assert selected[0].name == "0000.jpg"
    assert selected[-1].name == "0299.jpg"


def test_export_layout_normalizers_accept_common_shapes():
    """MNN and ONNX exporters may transpose the feature/anchor dimensions."""
    parity = load_module("issue51_parity", EDGE / "scripts" / "mnn_parity.py")
    values = np.arange(14 * 5, dtype=np.float32).reshape(14, 5)
    assert parity.normalize_output(values, 14).shape == (14, 5)
    assert parity.normalize_output(values.T[None], 14).shape == (14, 5)
    assert np.array_equal(parity.normalize_output(values.T, 14), values)
    assert parity.select_detection_output(
        [np.zeros((1, 2, 2)), values[None]], 14
    ).shape == (1, 14, 5)

    mnn_val = load_module("issue51_mnn_val", EDGE / "scripts" / "mnn_val.py")
    assert mnn_val.normalize_output(values[None, :, :], 14).shape == (14, 5)
    with pytest.raises(ValueError, match="batch"):
        parity.normalize_output(np.zeros((2, 5, 14), dtype=np.float32), 14)


def test_nms_is_class_offset_friendly_and_handles_empty():
    """The MNN decoder handles empty candidates and suppresses overlaps."""
    mnn_val = load_module("issue51_mnn_val_nms", EDGE / "scripts" / "mnn_val.py")
    assert mnn_val.class_nms_offset(1920, 1080) == 2.0 * 1920 + 8192.0
    assert mnn_val.nms(np.empty((0, 4)), np.empty((0,)), 0.5) == []
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 9, 9], [30, 30, 40, 40]], dtype=np.float32
    )
    keep = mnn_val.nms(boxes, np.array([0.9, 0.8, 0.7], dtype=np.float32), 0.5)
    assert keep == [0, 2]


def test_parity_image_list_rejects_duplicate_stems(tmp_path):
    """A repeated stem would overwrite per-image debug snapshots."""
    parity = load_module("issue51_parity_images", EDGE / "scripts" / "mnn_parity.py")
    (tmp_path / "a.jpg").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.png").touch()
    with pytest.raises(RuntimeError, match="stems are not unique"):
        parity.image_list(tmp_path, 0)


def test_map_delta_gate_is_optional_and_inclusive():
    """A declared mAP budget is inclusive and requires a reference value."""
    evaluator = load_module("issue51_eval_map_gate", EDGE / "scripts" / "eval_map.py")
    assert evaluator.delta_gate_passes(25.0, None) is True
    assert evaluator.delta_gate_passes(0.5, 0.5) is True
    assert evaluator.delta_gate_passes(0.500001, 0.5) is False

    passing = {"abs_delta_mAP50-95_pct": 0.5}
    assert evaluator.apply_delta_gate(passing, 0.5) == 0
    assert passing["mAP50-95_delta_gate_passed"] is True
    failing = {"abs_delta_mAP50-95_pct": 0.500001}
    assert evaluator.apply_delta_gate(failing, 0.5) != 0
    assert failing["mAP50-95_delta_gate_passed"] is False

    assert evaluator.nonnegative_finite_float("0.5") == pytest.approx(0.5)
    with pytest.raises(argparse.ArgumentTypeError):
        evaluator.nonnegative_finite_float("nan")
    with pytest.raises(argparse.ArgumentTypeError):
        evaluator.nonnegative_finite_float("-0.1")
    assert evaluator.extract_reference_map({"metrics/mAP50-95(B)": 0.2036}) == pytest.approx(0.2036)
    assert evaluator.extract_reference_map({"results_dict": {"mAP50-95": 0.2036}}) == pytest.approx(0.2036)


def test_map_gate_rejects_ambiguous_cli_combinations(monkeypatch):
    """Smoke subsets cannot claim the full image-count or delta gates."""
    evaluator = load_module("issue51_eval_map_args", EDGE / "scripts" / "eval_map.py")
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_map.py", "--preds", "preds", "--max-abs-delta-pct", "0.5"],
    )
    args = evaluator.parse_args()
    assert args.max_abs_delta_pct == pytest.approx(0.5)
    with pytest.raises(ValueError, match="reference-json"):
        evaluator.validate_delta_budget(args.max_abs_delta_pct, args.reference_json)

    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_map.py", "--preds", "preds", "--min-images", "1"],
    )
    args = evaluator.parse_args()
    assert args.smoke is False and args.min_images == 1
    with pytest.raises(ValueError, match="500"):
        evaluator.validate_acceptance_image_floor(args)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_map.py",
            "--preds",
            "preds",
            "--smoke",
            "--reference-json",
            "reference.json",
            "--max-abs-delta-pct",
            "0.5",
        ],
    )
    args = evaluator.parse_args()
    with pytest.raises(ValueError, match="smoke"):
        evaluator.validate_smoke_gate(args.smoke, args.max_abs_delta_pct)


def test_mnn_cli_rejects_invalid_numeric_options_before_import(tmp_path, monkeypatch):
    """Invalid MNN options fail before optional bindings are imported."""
    mnn_val = load_module("issue51_mnn_val_args", EDGE / "scripts" / "mnn_val.py")
    model = tmp_path / "model.mnn"
    image = tmp_path / "image.jpg"
    model.touch()
    image.touch()

    monkeypatch.setattr(
        sys,
        "argv",
        ["mnn_val.py", "--mnn", str(model), "--images", str(image), "--conf", "nan"],
    )
    with pytest.raises(ValueError, match="finite"):
        mnn_val.main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["mnn_val.py", "--mnn", str(model), "--images", str(image), "--limit", "-1"],
    )
    with pytest.raises(ValueError, match="non-negative"):
        mnn_val.main()


def test_exporter_reports_mnn_as_pending_parity():
    """MNN serialization alone is not reported as runtime acceptance evidence."""
    export = (EDGE / "scripts" / "export_models.py").read_text(encoding="utf-8")
    assert '"checked_scope": "converter_output"' in export
    assert '"runtime_smoke_checked": False' in export
    assert '"acceptance_ready": False' in export
    assert '"parity_required": True' in export


def test_repository_contains_no_legacy_issue51_path():
    """The final branch uses the current cross-platform directory only."""
    for path in (EDGE / "scripts", EDGE / "cpp"):
        assert "YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP" not in str(path)
    validation = (ROOT / "examples" / "YOLO-Master-Edge-Deployment" / "VALIDATION.md").read_text(
        encoding="utf-8"
    )
    assert "YOLO-Master-EsMoE-N-ONNX-NCNN-MNN-CPP" not in validation
