"""Numerical and gradient contracts for Foundation KD losses."""

import pytest
import torch

from ultralytics.nn.foundation import cosine_kd_loss, foreground_token_weights, hybrid_kd_loss, relational_kd_loss


def test_cosine_loss_is_zero_for_identical_features_and_backpropagates_student_only():
    student = torch.randn(2, 4, 3, 3, requires_grad=True)
    teacher = student.detach().clone().requires_grad_(True)

    loss = cosine_kd_loss(student, teacher)
    loss.backward()

    assert loss.ndim == 0
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert student.grad is not None
    assert teacher.grad is None


def test_cosine_loss_penalizes_opposite_vectors_and_is_fp16_stable():
    student = torch.ones(1, 4, 2, 2, dtype=torch.float16, requires_grad=True)
    teacher = -torch.ones(1, 4, 2, 2, dtype=torch.float16)

    loss = cosine_kd_loss(student, teacher)
    loss.backward()

    assert loss.item() == pytest.approx(2.0, abs=1e-3)
    assert torch.isfinite(student.grad).all()


def test_relational_loss_is_zero_for_identical_features_and_supports_deterministic_indices():
    student = torch.randn(2, 5, 3, 4, requires_grad=True)
    teacher = student.detach().clone().requires_grad_(True)
    indices = torch.tensor([0, 3, 7, 11])

    sampled = relational_kd_loss(student, teacher, mode="sampled", sample_indices=indices)
    full = relational_kd_loss(student, teacher, mode="full")
    sampled.backward()

    assert sampled.item() == pytest.approx(0.0, abs=1e-6)
    assert full.item() == pytest.approx(0.0, abs=1e-6)
    assert student.grad is not None
    assert teacher.grad is None


def test_sampled_relational_loss_matches_full_when_sample_count_covers_grid():
    student = torch.randn(1, 3, 2, 3)
    teacher = torch.randn(1, 3, 2, 3)

    sampled = relational_kd_loss(student, teacher, mode="sampled", num_samples=6)
    full = relational_kd_loss(student, teacher, mode="full")
    assert sampled == full


def test_hybrid_loss_respects_weights_and_zero_weights_keep_graph_connected():
    student = torch.randn(1, 4, 2, 2, requires_grad=True)
    teacher = torch.randn(1, 4, 2, 2)

    cosine_only = hybrid_kd_loss(student, teacher, cosine_weight=1.0, relation_weight=0.0)
    direct_cosine = cosine_kd_loss(student, teacher)
    assert cosine_only == direct_cosine

    zero = hybrid_kd_loss(student, teacher, cosine_weight=0.0, relation_weight=0.0)
    zero.backward()
    assert zero.item() == 0.0
    assert student.grad is not None


def test_foreground_token_weights_assign_inside_boundary_and_background():
    batch = {
        "img": torch.zeros(1, 3, 8, 8),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        "batch_idx": torch.tensor([0]),
    }
    weights = foreground_token_weights(
        batch,
        height=8,
        width=8,
        image_height=8,
        image_width=8,
        foreground_weight=1.5,
        boundary_weight=1.0,
        background_weight=0.25,
    )
    assert weights.shape == (1, 64)
    assert weights.max().item() == pytest.approx(1.5)
    assert weights.min().item() == pytest.approx(0.25)
    assert (weights == 1.0).any()


def test_foreground_token_weights_empty_targets_are_background_only():
    batch = {"img": torch.zeros(2, 3, 8, 8), "bboxes": torch.empty(0, 4), "batch_idx": torch.empty(0)}
    weights = foreground_token_weights(batch, height=2, width=2, image_height=8, image_width=8)
    assert torch.allclose(weights, torch.full((2, 4), 0.25))


def test_weighted_cosine_loss_changes_token_emphasis():
    student = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    teacher = torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]]]])
    uniform = cosine_kd_loss(student, teacher)
    weighted = cosine_kd_loss(student, teacher, token_weights=torch.tensor([[10.0, 1.0]]))
    assert weighted < uniform


@pytest.mark.parametrize(
    "fn, kwargs",
    [
        (cosine_kd_loss, {"eps": 0}),
        (relational_kd_loss, {"mode": "invalid"}),
        (relational_kd_loss, {"mode": "full", "sample_indices": torch.tensor([0])}),
        (relational_kd_loss, {"mode": "sampled", "num_samples": 0}),
        (relational_kd_loss, {"mode": "sampled", "sample_indices": torch.tensor([99])}),
        (hybrid_kd_loss, {"cosine_weight": -1.0}),
    ],
)
def test_invalid_loss_arguments_fail_fast(fn, kwargs):
    student = torch.randn(1, 3, 2, 2)
    teacher = torch.randn(1, 3, 2, 2)
    with pytest.raises((TypeError, ValueError)):
        fn(student, teacher, **kwargs)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_features_fail_fast(bad):
    student = torch.randn(1, 3, 2, 2)
    teacher = torch.randn(1, 3, 2, 2)
    student[0, 0, 0, 0] = bad
    with pytest.raises(ValueError, match="NaN or Inf"):
        cosine_kd_loss(student, teacher)


def test_shape_and_device_contracts_fail_fast():
    with pytest.raises(ValueError, match="identical aligned shapes"):
        cosine_kd_loss(torch.randn(1, 3, 2, 2), torch.randn(1, 3, 1, 2))
    with pytest.raises(TypeError, match="torch.Tensor"):
        relational_kd_loss("student", torch.randn(1, 3, 2, 2))
