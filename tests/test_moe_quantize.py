from types import SimpleNamespace

import torch.nn as nn


def test_quantization_plan_marks_router_parameters_fp16():
    from ultralytics.nn.modules.moe.quantize import get_quantization_plan

    model = nn.Module()
    model.router = nn.Linear(4, 4)
    model.experts = nn.ModuleList([nn.Linear(4, 4)])
    plan = get_quantization_plan(model)
    assert plan["router.weight"] == "fp16"
    assert plan["experts.0.weight"] == "int8"


def test_onnx_routing_nodes_are_excluded_from_quantization(monkeypatch):
    import onnx
    from ultralytics.nn.modules.moe import quantize as module

    graph = SimpleNamespace(
        node=[
            SimpleNamespace(name="router/Conv", op_type="Conv"),
            SimpleNamespace(name="expert/Conv", op_type="Conv"),
            SimpleNamespace(name="", op_type="Conv"),
        ]
    )
    monkeypatch.setattr(onnx, "load", lambda _: SimpleNamespace(graph=graph))
    assert module._onnx_routing_nodes("model.onnx") == ["router/Conv"]

