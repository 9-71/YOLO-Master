"""Offline contracts for the F12 SigLIP2 Foundation Teacher adapter."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.foundation import SigLIP2Teacher


class DummySigLIP2(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(patch_size=2, hidden_size=hidden_size, projection_size=hidden_size)
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.image_calls = 0
        self.text_calls = 0

    def forward(self, pixel_values, pixel_attention_mask=None, spatial_shapes=None, **kwargs):
        self.image_calls += 1
        batch, tokens, _ = pixel_values.shape
        hidden = self.scale * torch.ones(batch, tokens, self.config.vision_config.hidden_size)
        return SimpleNamespace(
            vision_model_output=SimpleNamespace(last_hidden_state=hidden, pooler_output=hidden.mean(dim=1))
        )

    def get_text_features(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        self.text_calls += 1
        return SimpleNamespace(pooler_output=self.scale * torch.ones(input_ids.shape[0], 8))


class DummyProcessor:
    def __call__(self, *, images=None, text=None, return_tensors="pt", **kwargs):
        if images is not None:
            batch = images.shape[0]
            return {
                "pixel_values": torch.zeros(batch, 6, 8),
                "pixel_attention_mask": torch.ones(batch, 6, dtype=torch.int32),
                "spatial_shapes": torch.tensor([[2, 3]] * batch),
            }
        prompts = list(text)
        return {
            "input_ids": torch.ones(len(prompts), 4, dtype=torch.long),
            "attention_mask": torch.ones(len(prompts), 4, dtype=torch.long),
        }


def test_siglip2_teacher_is_frozen_and_returns_semantic_dense_features():
    model = DummySigLIP2()
    teacher = SigLIP2Teacher(model=model, processor=DummyProcessor(), device="cpu")
    assert teacher.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    features = teacher.encode(torch.rand(2, 3, 8, 10))
    assert features.dense["p4"].shape == (2, 8, 2, 3)
    assert features.pooled.shape == (2, 8)
    assert features.semantic.shape == (2, 8)
    assert torch.allclose(features.semantic.norm(dim=-1), torch.ones(2))
    assert model.image_calls == 1


def test_siglip2_text_prototypes_are_normalized_and_cached():
    model = DummySigLIP2()
    teacher = SigLIP2Teacher(model=model, processor=DummyProcessor(), device="cpu")
    first = teacher.encode_text(["a cat", "a dog"])
    second = teacher.encode_text(["a cat", "a dog"])
    assert torch.equal(first, second)
    assert first.shape == (2, 8)
    assert torch.allclose(first.norm(dim=-1), torch.ones(2))
    assert model.text_calls == 1
    teacher.clear_text_cache()
    teacher.encode_text(["a cat", "a dog"])
    assert model.text_calls == 2


def test_siglip2_teacher_train_and_preprocess_boundaries():
    teacher = SigLIP2Teacher(model=DummySigLIP2(), processor=DummyProcessor(), device="cpu")
    teacher.train(True)
    assert teacher.training is False
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        teacher.preprocess(torch.full((1, 3, 4, 4), 2.0))
    with pytest.raises(ValueError, match="non-empty"):
        teacher.encode_text([""])


def test_siglip2_text_encoder_is_not_needed_for_cached_retrieval():
    model = DummySigLIP2()
    teacher = SigLIP2Teacher(model=model, processor=DummyProcessor(), device="cpu")
    teacher.encode_text(["cached"])
    model.get_text_features = lambda **kwargs: (_ for _ in ()).throw(AssertionError("text encoder called"))
    cached = teacher.encode_text(["cached"])
    assert cached.shape == (1, 8)
