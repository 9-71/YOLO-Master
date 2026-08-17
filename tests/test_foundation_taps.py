"""Offline contracts for dynamic student P-level feature capture."""

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.foundation import StudentFeatureTap
from ultralytics.nn.modules.head import Detect


class TinyGraph(nn.Module):
    def __init__(self, sources=(1, 2, 3), *, relative=False, head_cls=Detect):
        super().__init__()
        self.model = nn.ModuleList(
            [
                nn.Conv2d(3, 4, 1),
                nn.Conv2d(4, 8, 1),
                nn.Conv2d(8, 16, 1),
                nn.Conv2d(16, 32, 1),
            ]
        )
        self.head = head_cls(nc=2, ch=(8, 16, 32))
        self.head.f = [source - len(self.model) if relative else source for source in sources]
        self.head.i = len(self.model)
        self.model.append(self.head)

    def forward(self, x):
        outputs = []
        for index, layer in enumerate(self.model):
            if index == self.head.i:
                return layer([outputs[source] for source in self.head.f])
            x = layer(x)
            outputs.append(x)
        raise AssertionError("unreachable")


class TupleSourceGraph(TinyGraph):
    def __init__(self):
        super().__init__()
        self.model[2] = TupleOutput(self.model[2])


class TupleOutput(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, x):
        value = self.wrapped(x)
        return value, value.mean()


class BadHead(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x


def test_p4_tap_resolves_detect_sources_and_preserves_gradients():
    model = TinyGraph()
    tap = StudentFeatureTap(model, target="p4")
    assert tap.source_indices == (1, 2, 3)
    assert tap.source_index == 2
    assert tap.head_index == 4
    assert not tap.has_feature
    with pytest.raises(RuntimeError, match="run the student forward"):
        _ = tap.feature

    result = model(torch.randn(2, 3, 8, 8))
    feature = tap.feature
    assert feature.shape == (2, 16, 8, 8)
    assert feature.requires_grad
    assert tap.has_feature
    (feature.square().mean()).backward()
    assert model.model[2].weight.grad is not None
    assert result is not None


def test_relative_negative_detect_indices_are_resolved_against_head_index():
    model = TinyGraph(sources=(1, 2, 3), relative=True)
    tap = StudentFeatureTap(model, target="p3")
    assert tap.source_indices == (1, 2, 3)


def test_clear_prevents_stale_feature_reuse_and_close_removes_hook():
    model = TinyGraph()
    tap = StudentFeatureTap(model)
    model(torch.randn(1, 3, 8, 8))
    first = tap.feature
    tap.clear()
    assert not tap.has_feature
    with pytest.raises(RuntimeError):
        _ = tap.feature
    model(torch.randn(1, 3, 8, 8))
    assert tap.feature is not first
    tap.close()
    assert not tap.has_feature
    model(torch.randn(1, 3, 8, 8))
    with pytest.raises(RuntimeError):
        _ = tap.feature


def test_context_manager_removes_hook():
    model = TinyGraph()
    with StudentFeatureTap(model) as tap:
        model(torch.randn(1, 3, 8, 8))
        assert tap.has_feature
    assert not tap.has_feature
    model(torch.randn(1, 3, 8, 8))
    assert not tap.has_feature


def test_ambiguous_output_is_rejected_by_the_hook_contract():
    model = TinyGraph()
    tap = StudentFeatureTap(model)
    with pytest.raises(TypeError, match="exactly one tensor"):
        tap._hook_fn(model.model[2], (), (torch.zeros(1), torch.zeros(1)))


def test_single_tensor_tuple_output_is_supported():
    model = TinyGraph()
    tap = StudentFeatureTap(model)
    tap._hook_fn(model.model[2], (), (torch.zeros(1, 16, 4, 4),))
    assert tap.feature.shape == (1, 16, 4, 4)


@pytest.mark.parametrize(
    "target, error, message",
    [
        ("p2", ValueError, "target must be one"),
        ("P6", ValueError, "target must be one"),
        (1, ValueError, "target must be one"),
    ],
)
def test_invalid_target_is_rejected(target, error, message):
    with pytest.raises(error, match=message):
        StudentFeatureTap(TinyGraph(), target=target)


def test_no_detect_head_is_rejected():
    model = nn.Module()
    model.model = nn.ModuleList([nn.Conv2d(3, 4, 1)])
    with pytest.raises(ValueError, match="No Detect head"):
        StudentFeatureTap(model)


def test_invalid_detect_source_metadata_is_rejected():
    for sources, error, message in [
        ((1, 2), ValueError, "at least three"),
        ((1, "2", 3), TypeError, "must be integers"),
        ((1, 2, 8), ValueError, "outside the student layers"),
    ]:
        with pytest.raises(error, match=message):
            TinyGraph(sources=sources)
            StudentFeatureTap(TinyGraph(sources=sources))


def test_bare_sequential_container_is_supported():
    graph = TinyGraph()
    tap = StudentFeatureTap(graph.model)
    assert tap.source_index == 2
