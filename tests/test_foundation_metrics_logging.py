"""F08 Foundation metric cache and trainer aggregation contracts."""

import torch
import pytest

from tests.test_foundation_distill_model import DummyTeacher, TinyStudent, config
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.foundation_distill_model import FoundationDistillationModel


def test_wrapper_exposes_scalar_foundation_metrics_and_clears_eval_cache():
    wrapper = FoundationDistillationModel(TinyStudent(), DummyTeacher(), config())
    wrapper.train()

    wrapper({"img": torch.rand(2, 3, 64, 64)})
    metrics = wrapper.foundation_metrics()

    assert wrapper.has_foundation_metrics()
    assert set(metrics) == {
        "foundation_loss",
        "foundation_cosine_loss",
        "foundation_relational_loss",
        "foundation_cosine_raw",
        "foundation_relational_raw",
        "foundation_task_ratio",
        "foundation_loss_weight",
        "foundation_effective_weight",
        "foundation_foreground_enabled",
        "foundation_foreground_mean_weight",
    }
    assert all(isinstance(value, float) and value >= 0 for value in metrics.values())
    assert metrics["foundation_loss"] == pytest.approx(
        metrics["foundation_cosine_loss"] + metrics["foundation_relational_loss"], rel=1e-6
    )

    wrapper.eval()
    wrapper(torch.rand(1, 3, 64, 64))
    assert wrapper.foundation_metrics() == {}
    assert not wrapper.has_foundation_metrics()


def test_base_trainer_aggregates_foundation_metrics_without_changing_loss_columns():
    wrapper = FoundationDistillationModel(TinyStudent(), DummyTeacher(), config())
    wrapper.train()
    wrapper({"img": torch.rand(2, 3, 64, 64)})

    trainer = object.__new__(BaseTrainer)
    trainer.model = wrapper
    trainer._reset_foundation_metric_state()
    trainer._collect_foundation_metrics()
    wrapper.__dict__["_last_foundation_metrics"] = {
        "foundation_loss": 2.0,
        "foundation_cosine_loss": 1.0,
        "foundation_relational_loss": 1.0,
        "foundation_task_ratio": 0.5,
        "foundation_loss_weight": 1.0,
    }
    trainer._collect_foundation_metrics()

    mean = trainer._mean_foundation_metrics(prefix="train/")
    assert trainer.foundation_metric_steps == 2
    assert mean["train/foundation_loss"] == (trainer.foundation_metric_totals["foundation_loss"] / 2)
    assert mean["train/foundation_loss_weight"] == 1.0


def test_base_trainer_collection_is_noop_for_plain_student():
    trainer = object.__new__(BaseTrainer)
    trainer.model = TinyStudent()
    trainer._reset_foundation_metric_state()
    trainer._collect_foundation_metrics()

    assert trainer.foundation_metric_steps == 0
    assert trainer._mean_foundation_metrics() == {}
