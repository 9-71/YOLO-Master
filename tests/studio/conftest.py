"""Offline fixtures and explicit local-asset requirements for Core tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest


def pytest_ignore_collect(collection_path, config):
    """Keep Studio out of broad suites; explicitly select its directory, file, or node to run it."""
    studio_dir = Path(__file__).resolve().parent
    if studio_dir not in collection_path.parents:
        return None
    for arg in config.args:
        selected = (config.invocation_params.dir / str(arg).split("::", 1)[0]).resolve()
        if selected != studio_dir and studio_dir not in selected.parents:
            continue
        if selected == collection_path or selected in collection_path.parents or collection_path in selected.parents:
            return None
    return True


@pytest.fixture(autouse=True)
def no_implicit_download(monkeypatch):
    """Fail if a unit or integration test tries to download model/data assets."""

    def reject(*args, **kwargs):
        raise AssertionError("Studio tests require local assets; downloads are prohibited")

    monkeypatch.setattr("ultralytics.utils.downloads.safe_download", reject)
    monkeypatch.setattr("ultralytics.utils.downloads.download", reject)
    monkeypatch.setattr("ultralytics.data.utils.download", reject)
    monkeypatch.setattr("ultralytics.utils.checks.check_font", lambda *a, **kw: None)


@pytest.fixture
def fake_yolo(monkeypatch):
    """Provide the explicit offline engine for legacy bare-filename unit cases."""

    class FakeYOLO:
        def __init__(self, model):
            if "nonexistent" in str(model):
                raise FileNotFoundError(model)

        def predict(self, **kwargs):
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "result.jpg").write_bytes(b"fake-image")
            return [SimpleNamespace(save_dir=save_dir)]

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)
    return FakeYOLO


@pytest.fixture
def fake_checkpoint(tmp_path):
    """Provide a local file for the export handler's copy-before-export step."""
    path = tmp_path / "model.pt"
    path.write_bytes(b"fake-weights")
    return path


def pytest_addoption(parser):
    """Require an explicit opt-in separate from upstream's general --slow flag."""
    parser.addoption(
        "--studio-integration",
        action="store_true",
        default=False,
        help="Run local-asset Studio engine tests (also requires --slow and STUDIO_TEST_MODEL)",
    )


def pytest_collection_modifyitems(config, items):
    """Keep local-model checks out of upstream full-suite slow CI."""
    if not config.getoption("--studio-integration"):
        items[:] = [item for item in items if "studio_integration" not in item.keywords]
