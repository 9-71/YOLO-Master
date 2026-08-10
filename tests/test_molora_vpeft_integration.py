"""Integration contracts between V-PEFT placement plans and MoLoRA injection."""

import pytest
import torch.nn as nn

from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRALayer, MoLoRAModel, get_peft_molora_model
from ultralytics.utils.lora.api import _vpeft_model_fingerprint
from ultralytics.vpeft import PlacementPlan, PlacementTarget


def _model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))


def _plan(model: nn.Module, *, status: str = "ACCEPT") -> PlacementPlan:
    return PlacementPlan(
        model_fingerprint=_vpeft_model_fingerprint(model),
        planner_backend="vpeft",
        solver="ao",
        budget={"max_adapter_params": 1_024},
        targets=(PlacementTarget("0", "lora", 2), PlacementTarget("2", "molora", 3)),
        status=status,
    )


def _config() -> MoLoRAConfig:
    return MoLoRAConfig(r=1, alpha=8, num_experts=2, top_k=1, target_modules=["0"])


def test_molora_wrapper_consumes_per_target_plan_ranks_and_metadata():
    model = _model()
    plan = _plan(model)

    wrapper = MoLoRAModel(model, _config(), placement_plan=plan)

    assert isinstance(wrapper.model[0], MoLoRALayer)
    assert isinstance(wrapper.model[2], MoLoRALayer)
    assert wrapper.model[0].r == 2
    assert wrapper.model[2].r == 3
    assert wrapper.model.molora_rank_map == {"0": 2, "2": 3}
    assert wrapper.model.molora_placement_fingerprint == plan.fingerprint
    assert wrapper.model.molora_placement_plan == plan.to_dict()
    assert wrapper.model[0].molora_placement == {
        "target": "0",
        "variant": "lora",
        "rank": 2,
        "plan_fingerprint": plan.fingerprint,
    }


def test_molora_entry_point_accepts_serialized_placement_plan():
    model = _model()
    plan = _plan(model)

    wrapped = get_peft_molora_model(model, _config(), placement_plan=plan.to_dict())

    assert wrapped.molora_rank_map == {"0": 2, "2": 3}
    assert [wrapped[index].r for index in (0, 2)] == [2, 3]


@pytest.mark.parametrize("failure", ["fingerprint", "target", "variant", "status"])
def test_invalid_molora_placement_plan_is_rejected_before_any_replacement(failure):
    model = _model()
    fingerprint = _vpeft_model_fingerprint(model)
    target = PlacementTarget("0", "lora", 2)
    status = "ACCEPT"
    match = ""
    if failure == "fingerprint":
        fingerprint = "stale-model"
        match = "fingerprint"
    elif failure == "target":
        target = PlacementTarget("1", "lora", 1)
        match = "unsupported"
    elif failure == "variant":
        target = PlacementTarget("0", "dora", 2)
        match = "variant"
    else:
        status = "REFUSE"
        match = "status"
    plan = PlacementPlan(
        model_fingerprint=fingerprint,
        planner_backend="vpeft",
        solver="ao",
        budget={"max_adapter_params": 1_024},
        targets=(target,),
        status=status,
    )
    original_children = tuple(model.children())

    with pytest.raises(ValueError, match=match):
        get_peft_molora_model(model, _config(), placement_plan=plan)

    assert tuple(model.children()) == original_children
    assert not any(isinstance(module, MoLoRALayer) for module in model.modules())
    assert not hasattr(model, "molora_enabled")
