"""Contracts for the Foundation gate_decay loss-weight schedule."""

import pytest
import torch

from tests.test_foundation_distill_model import DummyTeacher, TinyStudent, config
from ultralytics.nn.foundation_distill_model import FoundationDistillationModel


def _wrapper(**overrides):
    cfg = config(foundation_weight_schedule="gate_decay", **overrides)
    wrapper = FoundationDistillationModel(TinyStudent(), DummyTeacher(), cfg)
    wrapper.train()
    return wrapper


def test_constant_schedule_is_legacy_default():
    wrapper = FoundationDistillationModel(TinyStudent(), DummyTeacher(), config())
    assert wrapper.weight_schedule == "constant"
    assert wrapper.effective_loss_weight() == wrapper.loss_weight


def test_gate_decay_starts_at_warmup_floor():
    wrapper = _wrapper()
    # No cosine EMA yet: gate closed, decay at 1.0 -> base * floor.
    assert wrapper._gate_factor() == 0.0
    assert wrapper.effective_loss_weight() == pytest.approx(wrapper.loss_weight * wrapper.warmup_floor)


def test_gate_opens_as_cosine_ema_drops():
    wrapper = _wrapper(
        foundation_gate_cosine=1.0,
        foundation_gate_width=0.1,
        foundation_warmup_floor=0.2,
        foundation_gate_cosine_low=0,  # disable the lower band edge for a pure ramp-up check
    )
    wrapper.__dict__["_cosine_ema"] = 0.97  # 30% through the gate span
    assert wrapper._gate_factor() == pytest.approx(0.3)
    assert wrapper.effective_loss_weight() == pytest.approx(wrapper.loss_weight * (0.2 + 0.8 * 0.3))
    wrapper.__dict__["_cosine_ema"] = 0.80  # fully open
    assert wrapper._gate_factor() == 1.0
    assert wrapper.effective_loss_weight() == pytest.approx(wrapper.loss_weight)


def test_gate_band_closes_on_over_alignment():
    wrapper = _wrapper(foundation_gate_cosine=1.0, foundation_gate_cosine_low=0.9, foundation_gate_width=0.05)
    wrapper.__dict__["_cosine_ema"] = 0.95  # band centre: both ramps fully open
    assert wrapper._gate_factor() == pytest.approx(1.0)
    wrapper.__dict__["_cosine_ema"] = 0.93  # 40% into the closing edge
    assert wrapper._gate_factor() == pytest.approx(0.6)
    wrapper.__dict__["_cosine_ema"] = 0.88  # over-aligned: gate closed, back to floor
    assert wrapper._gate_factor() == 0.0
    assert wrapper.effective_loss_weight() == pytest.approx(wrapper.loss_weight * wrapper.warmup_floor)


def test_gate_band_disabled_with_zero_low():
    wrapper = _wrapper(foundation_gate_cosine_low=0)
    assert wrapper.gate_cosine_low is None
    wrapper.__dict__["_cosine_ema"] = 0.5  # deeply aligned: no close-out when disabled
    assert wrapper._gate_factor() == 1.0


def test_late_decay_reaches_zero_at_final_epoch():
    wrapper = _wrapper(foundation_decay_start=0.5)
    wrapper.__dict__["_cosine_ema"] = 0.0  # gate fully open
    wrapper.set_foundation_progress(0, 11)
    assert wrapper._decay_factor() == 1.0
    wrapper.set_foundation_progress(5, 11)  # progress 0.5 == decay_start boundary
    assert wrapper._decay_factor() == 1.0
    wrapper.set_foundation_progress(7, 11)  # progress 0.7 -> (1 - 0.7) / 0.5
    assert wrapper._decay_factor() == pytest.approx(0.6)
    wrapper.set_foundation_progress(10, 11)  # final epoch -> 0
    assert wrapper._decay_factor() == 0.0
    assert wrapper.effective_loss_weight() == 0.0


def test_forward_updates_cosine_ema_and_reports_effective_weight():
    wrapper = _wrapper()
    before = wrapper.effective_loss_weight()
    wrapper({"img": torch.rand(2, 3, 64, 64)})
    assert wrapper.__dict__["_cosine_ema"] is not None
    metrics = wrapper.foundation_metrics()
    assert "foundation_effective_weight" in metrics
    # The metric records the weight applied for this step (EMA updates only affect later steps).
    assert metrics["foundation_effective_weight"] == pytest.approx(before)


def test_repeated_forwards_converge_ema_to_observed_cosine():
    wrapper = _wrapper()
    img = torch.rand(2, 3, 64, 64)  # fixed input -> deterministic raw cosine each step
    for _ in range(40):
        wrapper({"img": img})
    ema = wrapper.__dict__["_cosine_ema"]
    last_raw = wrapper.foundation_metrics()["foundation_cosine_raw"]
    assert ema == pytest.approx(last_raw, rel=0.05)


def test_scaled_metrics_track_effective_weight():
    wrapper = _wrapper()
    wrapper({"img": torch.rand(2, 3, 64, 64)})
    metrics = wrapper.foundation_metrics()
    assert metrics["foundation_loss"] == pytest.approx(
        metrics["foundation_cosine_loss"] + metrics["foundation_relational_loss"], rel=1e-6
    )
    ratio = metrics["foundation_effective_weight"] / metrics["foundation_loss_weight"]
    assert metrics["foundation_cosine_loss"] == pytest.approx(
        metrics["foundation_cosine_raw"] * metrics["foundation_loss_weight"] * ratio * 2, rel=1e-6
    )


def test_invalid_schedule_config_rejected():
    with pytest.raises(ValueError, match="foundation_weight_schedule"):
        FoundationDistillationModel(TinyStudent(), DummyTeacher(), config(foundation_weight_schedule="bogus"))
    with pytest.raises(ValueError, match="foundation_warmup_floor"):
        _wrapper(foundation_warmup_floor=1.5)
    with pytest.raises(ValueError, match="foundation_decay_start"):
        _wrapper(foundation_decay_start=1.0)
    with pytest.raises(ValueError, match="foundation_gate_width"):
        _wrapper(foundation_gate_width=0.0)
    with pytest.raises(ValueError, match="foundation_gate_cosine_low"):
        _wrapper(foundation_gate_cosine_low=1.5)  # above gate_cosine
