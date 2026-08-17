"""Offline tests for the DINOv3 Foundation Teacher adapter."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.foundation import DINOv3Teacher


class DummyBackbone(nn.Module):
    def __init__(self, *, output="feature_maps", register_tokens=2, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(patch_size=4, hidden_size=hidden_size, num_register_tokens=register_tokens)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.output = output
        self.calls = []

    def forward(self, pixel_values):
        self.calls.append((pixel_values.shape, pixel_values.dtype, self.training))
        batch, _, height, width = pixel_values.shape
        grid_h, grid_w = height // 4, width // 4
        dense = self.scale * torch.ones(batch, self.config.hidden_size, grid_h, grid_w, device=pixel_values.device)
        if self.output == "feature_maps":
            return SimpleNamespace(feature_maps=(dense / 2, dense), pooler_output=dense.mean(dim=(2, 3)))
        tokens = self.scale * torch.arange(
            1 + self.config.num_register_tokens + grid_h * grid_w,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).view(1, -1, 1).expand(batch, -1, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=tokens)


class BadBackbone(DummyBackbone):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, pixel_values):
        output = super().forward(pixel_values)
        if self.kind == "nan":
            output.feature_maps = (torch.full_like(output.feature_maps[-1], float("nan")),)
        elif self.kind == "batch":
            output.feature_maps = (output.feature_maps[-1][:1],)
        elif self.kind == "grid":
            output.feature_maps = (output.feature_maps[-1][:, :, :-1, :],)
        elif self.kind == "pooled":
            output.pooler_output = torch.zeros(pixel_values.shape[0], 2)
        elif self.kind == "missing":
            return SimpleNamespace()
        return output


def test_dummy_teacher_is_frozen_and_always_eval():
    model = DummyBackbone()
    teacher = DINOv3Teacher(model=model, device="cpu")

    assert teacher.training is False
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    teacher.train(True)
    assert teacher.training is False
    assert model.training is False


def test_teacher_to_updates_device_and_dtype_without_unfreezing():
    model = DummyBackbone()
    teacher = DINOv3Teacher(model=model, device="cpu")

    moved = teacher.to(dtype=torch.float64)

    assert moved is teacher
    assert teacher.device == torch.device("cpu")
    assert teacher.dtype == torch.float64
    assert next(teacher.parameters()).dtype == torch.float64
    assert teacher.preprocess(torch.zeros(1, 3, 4, 4)).dtype == torch.float64
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_feature_maps_output_is_normalized_to_p4_with_metadata():
    model = DummyBackbone()
    teacher = DINOv3Teacher(model=model, device="cpu")
    result = teacher.encode(torch.zeros(2, 3, 5, 6))

    assert result.dense["p4"].shape == (2, 8, 2, 2)
    assert result.pooled.shape == (2, 8)
    assert result.metadata["input_size"] == (5, 6)
    assert result.metadata["padded_size"] == (8, 8)
    assert result.metadata["grid_size"] == (2, 2)
    assert result.metadata["num_register_tokens"] == 2
    assert torch.isfinite(result.dense["p4"]).all()
    assert model.calls[-1][0] == torch.Size((2, 3, 8, 8))


def test_token_sequence_output_respects_register_token_count():
    model = DummyBackbone(output="tokens", register_tokens=3)
    teacher = DINOv3Teacher(model=model, device="cpu")
    result = teacher.encode(torch.zeros(1, 3, 8, 12))

    assert result.dense["p4"].shape == (1, 8, 2, 3)
    assert result.metadata["prefix_tokens"] == 4
    assert result.pooled.shape == (1, 8)


def test_token_sequence_without_prefix_is_supported():
    model = DummyBackbone(output="tokens", register_tokens=2)
    teacher = DINOv3Teacher(model=model, device="cpu")
    tokens = torch.randn(1, 2, 8)
    result = teacher._parse_output(SimpleNamespace(feature_maps=tokens), batch_size=1, spatial_size=(4, 8))
    assert result.dense["p4"].shape == (1, 8, 1, 2)


def test_model_loader_is_injected_without_transformers_or_network_access():
    model = DummyBackbone()
    calls = []

    def loader(model_id, weights_path):
        calls.append((model_id, weights_path))
        return model

    teacher = DINOv3Teacher(model_id="local-dummy", weights_path="dummy.bin", model_loader=loader, device="cpu")
    assert calls == [("local-dummy", "dummy.bin")]
    assert teacher.model is model


def test_model_loader_with_single_argument_is_supported():
    model = DummyBackbone()
    calls = []

    def loader(model_id):
        calls.append(model_id)
        return model

    teacher = DINOv3Teacher(model_id="single-arg", model_loader=loader, device="cpu")
    assert calls == ["single-arg"]
    assert teacher.model is model


@pytest.mark.parametrize(
    "kind, message",
    [
        ("nan", "NaN or Inf"),
        ("batch", "batch"),
        ("grid", "grid"),
        ("pooled", "pooled"),
        ("missing", "does not contain"),
    ],
)
def test_invalid_backbone_outputs_fail_fast(kind, message):
    teacher = DINOv3Teacher(model=BadBackbone(kind), device="cpu")
    with pytest.raises(ValueError, match=message):
        teacher.encode(torch.zeros(2, 3, 8, 8))


def test_loader_internal_type_error_is_not_retried():
    calls = []

    def loader(model_id, weights_path):
        calls.append((model_id, weights_path))
        raise TypeError("loader failure")

    with pytest.raises(TypeError, match="loader failure"):
        DINOv3Teacher(model_loader=loader, device="cpu")
    assert len(calls) == 1
