"""Opt-in (--slow --studio-integration) real Core engine checks; require local weights, never download assets."""

import os
from pathlib import Path

import pytest
from PIL import Image

from studio.handlers.export import ExportHandler
from studio.handlers.predict import PredictHandler
from studio.handlers.train import TrainHandler

pytestmark = [pytest.mark.slow, pytest.mark.studio_integration]


@pytest.fixture
def local_model():
    value = os.environ.get("STUDIO_TEST_MODEL")
    assert value, "Set STUDIO_TEST_MODEL to existing local YOLO weights"
    model = Path(value).resolve(strict=True)
    assert model.is_file() and model.suffix == ".pt"
    return model


def test_real_predict(local_model, tmp_path):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (64, 64), "white").save(image)
    result = PredictHandler().execute(
        "predict", {"model_path": str(local_model), "data_source": str(image), "device": "cpu"}, str(tmp_path / "runs")
    )
    assert result["success"], result["error"]
    assert result["artifacts"] and all(Path(p).is_file() for p in result["artifacts"])


def test_real_tiny_train(local_model, tmp_path, monkeypatch):
    import yaml

    from ultralytics import YOLO

    for split in ("train", "val"):
        images = tmp_path / "images" / split
        labels = tmp_path / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (64, 64), "white").save(images / f"{index}.jpg")
            (labels / f"{index}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    data = tmp_path / "data.yaml"
    data.write_text(
        yaml.safe_dump({"path": str(tmp_path), "train": "images/train", "val": "images/val", "names": {0: "object"}})
    )
    seen = []

    def bounded_model(path):
        model = YOLO(path)
        train = model.train

        def bounded_train(**kwargs):
            seen.append(kwargs["imgsz"])
            return train(**kwargs, workers=0, amp=False, plots=False)

        monkeypatch.setattr(model, "train", bounded_train)
        return model

    monkeypatch.setattr("ultralytics.YOLO", bounded_model)
    result = TrainHandler().execute(
        "train",
        {
            "model_path": str(local_model),
            "data_source": str(data),
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "imgsz": 64,
        },
        str(tmp_path / "runs"),
    )
    assert result["success"], result["error"]
    assert seen == [64]
    checkpoint = next(Path(p) for p in result["artifacts"] if p.endswith("last.pt"))
    assert checkpoint.is_file()
    assert result["metadata"]["optimizer_audit"]["audited"]


def test_real_export(local_model, tmp_path):
    result = ExportHandler().execute(
        "export",
        {"model_path": str(local_model), "format": "torchscript", "imgsz": 64, "device": "cpu"},
        str(tmp_path / "runs"),
    )
    assert result["success"], result["error"]
    assert any(Path(p).is_file() and p.endswith(".torchscript") for p in result["artifacts"])
