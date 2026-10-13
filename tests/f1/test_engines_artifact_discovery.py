"""Cross-engine artifact discovery contract tests.

All five task handlers (train / val / predict / export / diagnose) must collect
artifacts with the same full-tree scan (``rglob("*")``) so no file type can be
silently dropped (e.g. an ``.onnx`` export hiding its ``.pt`` weights, or a
``.yaml`` metadata sidecar nested in a subdirectory).  Each test pre-places a
rich multi-type file tree in the job output directory, runs the handler with a
mocked Ultralytics engine, and asserts the returned artifact list equals the
full sorted tree exactly.  No real model work is performed.

Run:
    pytest f1/test_engines_artifact_discovery.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from f1.handlers.diagnose import DiagnoseHandler
from f1.handlers.export import ExportHandler
from f1.handlers.predict import PredictHandler
from f1.handlers.train import TrainHandler
from f1.handlers.val import ValHandler


def _preplace_full_tree(job_dir: Path) -> None:
    """Create a rich multi-type tree (nested dirs + 5 suffixes) in ``job_dir``."""
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "weights").mkdir()
    (job_dir / "weights" / "best.pt").write_bytes(b"weights")
    (job_dir / "batch0.jpg").write_bytes(b"jpg")
    (job_dir / "curve.png").write_bytes(b"png")
    (job_dir / "args.yaml").write_text("lr0: 0.01\n", encoding="utf-8")
    (job_dir / "export.onnx").write_bytes(b"onnx")


def _expected_artifacts(job_dir: Path) -> list[str]:
    """The canonical full-tree contract every handler must reproduce."""
    return sorted(str(p.resolve()) for p in job_dir.rglob("*") if p.is_file())


class TestFullTreeArtifactDiscovery:
    """Every task engine must capture the complete output tree without filters."""

    def test_train_collects_full_tree(self, tmp_path: Path, monkeypatch: Any) -> None:
        """TrainHandler returns the whole job tree (nested weights/, jpg, png, yaml)."""

        class FakeTrainModel:
            def __init__(self, path: str) -> None:
                self.path = path
                self.trainer = None

            def train(self, **kwargs: Any) -> None:
                return None

        monkeypatch.setattr("ultralytics.YOLO", FakeTrainModel)

        job_dir = tmp_path / "train-full-001"
        _preplace_full_tree(job_dir)
        params: dict[str, Any] = {
            "model_path": str(tmp_path / "model.pt"),
            "data_source": "coco8.yaml",
            "epochs": 1,
            "device": "cpu",
        }
        (tmp_path / "model.pt").write_bytes(b"pt")

        result = TrainHandler().execute("train-full-001", params, str(tmp_path))

        assert result["success"] is True
        assert result["artifacts"] == _expected_artifacts(job_dir)
        assert any(a.endswith("best.pt") for a in result["artifacts"])
        assert any(a.endswith("batch0.jpg") for a in result["artifacts"])
        assert any(a.endswith("curve.png") for a in result["artifacts"])
        assert any(a.endswith("args.yaml") for a in result["artifacts"])
        assert any(a.endswith("export.onnx") for a in result["artifacts"])

    def test_val_collects_full_tree(self, tmp_path: Path, monkeypatch: Any) -> None:
        """ValHandler merges engine-written reports with pre-placed side files."""

        class FakeValModel:
            def __init__(self, path: str) -> None:
                self.path = path

            def val(self, **kwargs: Any) -> Any:
                job_dir = Path(kwargs["project"]) / kwargs["name"]
                job_dir.mkdir(parents=True, exist_ok=True)
                (job_dir / "results.csv").write_text("epoch\n", encoding="utf-8")
                (job_dir / "confusion_matrix.png").write_bytes(b"png")
                box = SimpleNamespace(map50=0.5, map75=0.4, map=0.3, p=[0.9], r=[0.8])
                return SimpleNamespace(box=box, speed={"inference": 1.0})

        monkeypatch.setattr("ultralytics.YOLO", FakeValModel)

        job_dir = tmp_path / "val-full-001"
        _preplace_full_tree(job_dir)
        params: dict[str, Any] = {
            "model_path": str(tmp_path / "model.pt"),
            "data_source": "coco8.yaml",
            "device": "cpu",
        }
        (tmp_path / "model.pt").write_bytes(b"pt")

        result = ValHandler().execute("val-full-001", params, str(tmp_path))

        assert result["success"] is True
        # 7 files: 5 pre-placed + results.csv + confusion_matrix.png
        assert result["artifacts"] == _expected_artifacts(job_dir)
        assert len(result["artifacts"]) == 7
        assert any(a.endswith("results.csv") for a in result["artifacts"])

    def test_predict_collects_full_tree(self, tmp_path: Path, monkeypatch: Any) -> None:
        """PredictHandler keeps annotated images and pre-placed label sidecars."""

        class FakePredictModel:
            def __init__(self, path: str) -> None:
                self.path = path

            def predict(self, **kwargs: Any) -> list[Any]:
                batch_dir = Path(kwargs["project"]) / kwargs["name"]
                batch_dir.mkdir(parents=True, exist_ok=True)
                results = []
                for source in kwargs["source"]:
                    annotated = batch_dir / f"{Path(source).stem}.annotated.jpg"
                    annotated.write_bytes(b"annotated")
                    results.append(SimpleNamespace(save_dir=str(batch_dir)))
                return results

        monkeypatch.setattr("ultralytics.YOLO", FakePredictModel)

        source = tmp_path / "bus.jpg"
        source.write_bytes(b"jpg")
        job_dir = tmp_path / "predict-full-001"
        job_dir.mkdir()
        (job_dir / "labels").mkdir()
        (job_dir / "labels" / "bus.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (job_dir / "args.yaml").write_text("conf: 0.25\n", encoding="utf-8")
        params: dict[str, Any] = {
            "model_path": str(tmp_path / "model.pt"),
            "data_source": [str(source)],
            "batch_size": 2,
            "device": "cpu",
        }
        (tmp_path / "model.pt").write_bytes(b"pt")

        result = PredictHandler().execute("predict-full-001", params, str(tmp_path))

        assert result["success"] is True
        # 3 files: annotated.jpg (batch_000/) + labels/bus.txt + args.yaml
        assert result["artifacts"] == _expected_artifacts(job_dir)
        assert len(result["artifacts"]) == 3
        assert any(a.endswith("bus.annotated.jpg") for a in result["artifacts"])

    def test_export_collects_full_tree(self, tmp_path: Path, monkeypatch: Any) -> None:
        """ExportHandler captures the model copy, the .onnx and sidecar metadata."""

        class FakeExportModel:
            def __init__(self, path: str) -> None:
                self.path = Path(path)

            def export(self, **kwargs: Any) -> str:
                artifact = self.path.with_suffix(f".{kwargs['format']}")
                artifact.write_bytes(b"export")
                return str(artifact)

        monkeypatch.setattr("ultralytics.YOLO", FakeExportModel)

        model_file = tmp_path / "dummy_model.pt"
        model_file.write_bytes(b"pt")
        job_dir = tmp_path / "export-full-001"
        job_dir.mkdir()
        (job_dir / "meta").mkdir()
        (job_dir / "meta" / "info.json").write_text("{}", encoding="utf-8")
        (job_dir / "args.yaml").write_text("imgsz: 320\n", encoding="utf-8")
        params: dict[str, Any] = {"model_path": str(model_file), "format": "onnx", "imgsz": 320, "device": "cpu"}

        result = ExportHandler().execute("export-full-001", params, str(tmp_path))

        assert result["success"] is True
        # 4 files: model copy (.pt) + .onnx + args.yaml + meta/info.json
        assert result["artifacts"] == _expected_artifacts(job_dir)
        assert len(result["artifacts"]) == 4
        assert any(a.endswith("dummy_model.pt") for a in result["artifacts"])
        assert any(a.endswith("dummy_model.onnx") for a in result["artifacts"])

    def test_diagnose_collects_full_tree(self, tmp_path: Path) -> None:
        """DiagnoseHandler no longer hardcodes its two report filenames."""
        job_dir = tmp_path / "diag-full-001"
        job_dir.mkdir()
        (job_dir / "sysinfo").mkdir()
        (job_dir / "sysinfo" / "cpu.txt").write_text("model name\n", encoding="utf-8")
        (job_dir / "notes.txt").write_text("ad hoc note\n", encoding="utf-8")

        result = DiagnoseHandler().execute("diag-full-001", {}, str(tmp_path))

        assert result["success"] is True
        # 4 files: system_diagnostics.json + system_diagnostics.txt + 2 pre-placed
        assert result["artifacts"] == _expected_artifacts(job_dir)
        assert len(result["artifacts"]) == 4
        assert any(a.endswith("system_diagnostics.json") for a in result["artifacts"])
        assert any(a.endswith("cpu.txt") for a in result["artifacts"])
