"""Offline contracts for P4 feature alignment projectors."""

import pytest
import torch

from ultralytics.nn.foundation import P4AlignmentProjector


def test_projector_aligns_channels_without_resizing_and_keeps_teacher_frozen():
    projector = P4AlignmentProjector(student_channels=16, teacher_channels=32, align_dim=8)
    student = torch.randn(2, 16, 4, 4, requires_grad=True)
    teacher = torch.randn(2, 32, 4, 4, requires_grad=True)

    student_aligned, teacher_aligned = projector(student, teacher)

    assert student_aligned.shape == (2, 8, 4, 4)
    assert teacher_aligned.shape == (2, 8, 4, 4)
    assert student_aligned.requires_grad
    assert not teacher_aligned.requires_grad
    assert projector.alignment == {
        "student_size": (4, 4),
        "teacher_size": (4, 4),
        "target_size": (4, 4),
        "teacher_resized": False,
        "resize_ratio": None,
    }
    assert projector.teacher_projection_frozen
    assert all(not parameter.requires_grad for parameter in projector.teacher_proj.parameters())

    student_aligned.square().mean().backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_projector_resizes_only_teacher_to_student_grid_and_records_ratio():
    projector = P4AlignmentProjector(student_channels=4, teacher_channels=8, align_dim=4, use_norm=False)
    student = torch.randn(1, 4, 8, 12)
    teacher = torch.randn(1, 8, 4, 6, requires_grad=True)

    student_aligned, teacher_aligned = projector(student, teacher)

    assert student_aligned.shape == teacher_aligned.shape == (1, 4, 8, 12)
    assert projector.alignment["teacher_resized"] is True
    assert projector.alignment["resize_ratio"] == (2.0, 2.0)
    assert projector.student_proj[1].__class__.__name__ == "Identity"
    assert not teacher_aligned.requires_grad
    assert teacher.grad is None


def test_equal_teacher_channels_use_frozen_identity_projection():
    projector = P4AlignmentProjector(student_channels=4, teacher_channels=4, align_dim=4)
    assert projector.teacher_proj.__class__.__name__ == "Identity"
    assert projector.teacher_projection_frozen
    teacher = torch.randn(1, 4, 2, 2, requires_grad=True)
    _, aligned = projector(torch.randn(1, 4, 2, 2), teacher)
    assert torch.equal(aligned, teacher.detach())


def test_train_mode_never_unfreezes_teacher_projection():
    projector = P4AlignmentProjector(4, 8, 4).train()
    projector.eval()
    projector.train()
    assert projector.training
    assert projector.student_proj.training
    assert projector.teacher_projection_frozen


@pytest.mark.parametrize(
    "student, teacher, error, message",
    [
        (torch.randn(1, 4, 2), torch.randn(1, 8, 2, 2), ValueError, "student_feat must have shape"),
        (torch.randn(1, 3, 2, 2), torch.randn(1, 8, 2, 2), ValueError, "student_feat has 3 channels"),
        (torch.randn(1, 4, 2, 2), torch.randn(2, 8, 2, 2), ValueError, "batch sizes must match"),
        (torch.randn(1, 4, 2, 2), torch.randn(1, 8, 0, 2), ValueError, "positive spatial"),
    ],
)
def test_invalid_feature_inputs_fail_fast(student, teacher, error, message):
    projector = P4AlignmentProjector(4, 8, 4)
    with pytest.raises(error, match=message):
        projector(student, teacher)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"student_channels": 0, "teacher_channels": 8},
        {"student_channels": 4, "teacher_channels": 0},
        {"student_channels": 4, "teacher_channels": 8, "align_dim": 0},
        {"student_channels": 4, "teacher_channels": 8, "use_norm": 1},
    ],
)
def test_invalid_projector_configuration_is_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        P4AlignmentProjector(**kwargs)
