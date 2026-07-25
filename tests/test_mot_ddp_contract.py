from types import SimpleNamespace

import torch.nn as nn

from ultralytics.engine.extensions.mixture import MixtureRuntimeController
from ultralytics.nn.modules.moa import C2fMoA
from ultralytics.nn.modules.moe.gated import AdaptiveGateMoE
from ultralytics.nn.modules.moe.modules import OptimizedMOE
from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.nn.peft.molora.layer import MoLoRALayer


def _controller(model):
    trainer = SimpleNamespace(model=nn.Sequential(model), args=SimpleNamespace(), world_size=2)
    return MixtureRuntimeController(trainer)


def test_mixture_controller_configures_safe_sparse_mot_contract_without_disabling_request():
    block = MoTBlock(24, num_heads=3, top_k=1, sparse_train=True)
    trainer = SimpleNamespace(model=nn.Sequential(block), args=SimpleNamespace(), world_size=2)
    controller = MixtureRuntimeController(trainer)

    disabled, frozen, dense = controller.prepare_ddp(find_unused_parameters=True)

    assert (disabled, frozen, dense) == (0, 0, 0)
    assert block.sparse_train is True
    assert block._ddp_find_unused_parameters is True
    assert block._ddp_contract_source == "trainer"


def test_mixture_controller_marks_unsafe_sparse_mot_for_dense_fallback():
    block = MoTBlock(24, num_heads=3, top_k=1, sparse_train=True)
    trainer = SimpleNamespace(model=nn.Sequential(block), args=SimpleNamespace(), world_size=2)
    controller = MixtureRuntimeController(trainer)

    disabled, frozen, dense = controller.prepare_ddp(find_unused_parameters=False)

    assert (disabled, frozen, dense) == (0, 0, 1)
    assert block.sparse_train is True
    assert block._ddp_find_unused_parameters is False


def test_compiled_dense_moa_uses_static_graph_without_unused_parameter_scan():
    controller = _controller(C2fMoA(64, 64, n=1, num_heads=3))

    assert controller.resolve_ddp_policy(compile_enabled=True) == (False, True)


def test_compiled_sparse_training_mot_keeps_unused_parameter_scan():
    controller = _controller(MoTBlock(24, num_heads=3, top_k=1, sparse_train=True))

    assert controller.resolve_ddp_policy(compile_enabled=True) == (True, False)


def test_compiled_sparse_molora_keeps_unused_parameter_scan():
    layer = MoLoRALayer(nn.Linear(16, 16), r=2, num_experts=4, top_k=2)
    controller = _controller(layer)

    assert controller.resolve_ddp_policy(compile_enabled=True) == (True, False)


def test_uncompiled_models_preserve_existing_unused_parameter_safety():
    controller = _controller(C2fMoA(64, 64, n=1, num_heads=3))

    assert controller.resolve_ddp_policy(compile_enabled=False) == (True, False)


def test_adaptive_gate_ddp_dense_fallback_allows_compiled_static_graph():
    module = AdaptiveGateMoE(16, 16, num_experts=4, top_k=2)
    controller = _controller(module)
    controller.prepare_ddp(find_unused_parameters=True)

    assert module.export_capabilities()["training_sparse_dispatch"] is False
    assert controller.resolve_ddp_policy(compile_enabled=True) == (False, True)


def test_unknown_training_dispatch_capability_falls_back_conservatively():
    class UnknownRoutedModule(nn.Module):
        def export_capabilities(self):
            return {"routing_kind": "custom", "sparse_dispatch": True}

    controller = _controller(UnknownRoutedModule())

    assert controller.resolve_ddp_policy(compile_enabled=True) == (True, False)


def test_invalid_dispatch_capability_falls_back_conservatively():
    class InvalidRoutedModule(nn.Module):
        def export_capabilities(self):
            return None

    controller = _controller(InvalidRoutedModule())

    assert controller.resolve_ddp_policy(compile_enabled=True) == (True, False)


def test_registered_legacy_moe_without_capabilities_falls_back_conservatively():
    controller = _controller(OptimizedMOE(16, 16, num_experts=2, top_k=1))

    assert controller.resolve_ddp_policy(compile_enabled=True) == (True, False)
