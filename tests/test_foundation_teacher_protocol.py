"""Contracts for the dependency-free Foundation Teacher boundary."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ultralytics.nn.foundation import FoundationFeatures, FoundationTeacher
from ultralytics.nn.foundation.preprocessing import prepare_image_tensor


ROOT = Path(__file__).resolve().parents[1]


def test_foundation_features_accepts_dense_and_optional_representations():
    features = FoundationFeatures(
        dense={"p4": torch.zeros(2, 8, 4, 5)},
        pooled=torch.zeros(2, 8),
        semantic=torch.zeros(2, 8),
        metadata={"source": "test"},
    )

    assert features.dense["p4"].shape == (2, 8, 4, 5)
    assert features.metadata["source"] == "test"


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"dense": []}, TypeError, "dense must be a dict"),
        ({"dense": {"p4": torch.zeros(2, 8, 4)}}, ValueError, r"shape \(B, C, H, W\)"),
        ({"dense": {"": torch.zeros(2, 8, 4, 4)}}, TypeError, "non-empty strings"),
        ({"dense": {"p4": "feature"}}, TypeError, "torch.Tensor"),
        ({"dense": {}, "pooled": torch.zeros(2, 8, 1)}, ValueError, r"shape \(B, C\)"),
        ({"dense": {}, "semantic": "feature"}, TypeError, "torch.Tensor or None"),
        ({"dense": {}, "metadata": []}, TypeError, "metadata must be a dict"),
    ],
)
def test_foundation_features_rejects_invalid_shapes_and_types(kwargs, error, message):
    with pytest.raises(error, match=message):
        FoundationFeatures(**kwargs)


def test_foundation_teacher_protocol_is_runtime_checkable():
    class StubTeacher:
        name = "stub"

        def freeze(self):
            return None

        def preprocess(self, images):
            return images

        def encode(self, images):
            return FoundationFeatures(dense={})

        def to(self, device=None, dtype=None):
            return self

    assert isinstance(StubTeacher(), FoundationTeacher)
    assert not isinstance(object(), FoundationTeacher)


def test_default_foundation_import_does_not_import_transformers():
    script = "import sys; import ultralytics.nn.foundation; print('transformers' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "False"


def test_prepare_image_tensor_normalizes_and_pads_only_bottom_right():
    images = torch.zeros(1, 3, 5, 6)
    images[:, :, -1, -1] = 1.0
    result = prepare_image_tensor(images, patch_size=4, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))

    assert result.shape == (1, 3, 8, 8)
    assert torch.equal(result[:, :, :5, :6], images)
    assert torch.count_nonzero(result[:, :, 5:, :]).item() == 0
    assert torch.count_nonzero(result[:, :, :, 6:]).item() == 0


def test_prepare_image_tensor_converts_uint8_and_rejects_invalid_values():
    images = torch.tensor([[[[0]], [[128]], [[255]]]], dtype=torch.uint8)
    result = prepare_image_tensor(images, patch_size=1, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    assert torch.allclose(result.flatten(), torch.tensor([0.0, 128 / 255, 1.0]))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prepare_image_tensor(torch.full((1, 3, 2, 2), 2.0), patch_size=2)
    with pytest.raises(ValueError, match="NaN or Inf"):
        prepare_image_tensor(torch.full((1, 3, 2, 2), float("nan")), patch_size=2)
