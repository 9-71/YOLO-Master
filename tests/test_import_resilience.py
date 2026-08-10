"""Import resilience tests for optional visualization dependencies."""

import subprocess
import sys

import pytest

from ultralytics.utils import plt_settings


def test_top_level_yolo_import_does_not_require_matplotlib():
    code = """
import builtins

original_import = builtins.__import__

def reject_matplotlib(name, *args, **kwargs):
    if name == "matplotlib" or name.startswith("matplotlib."):
        raise ModuleNotFoundError("matplotlib intentionally unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_matplotlib
from ultralytics import YOLO
from ultralytics.models.yolo.semantic.train import SemanticSegmentationTrainer
assert YOLO is not None
assert SemanticSegmentationTrainer is not None
"""

    subprocess.run([sys.executable, "-c", code], check=True)


def test_plot_settings_reports_missing_matplotlib_when_plotting(monkeypatch):
    original_import = __import__

    def reject_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("matplotlib intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_matplotlib)

    @plt_settings()
    def plot():
        return None

    with pytest.raises(ImportError, match="Matplotlib is required for plotting"):
        plot()
