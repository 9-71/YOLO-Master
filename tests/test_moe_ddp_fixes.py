"""Regression contracts for trainer sparse-MoE DDP safety."""

import ast
from pathlib import Path

S = (Path(__file__).parents[1] / "ultralytics/engine/trainer.py").read_text(encoding="utf-8")
V06 = (Path(__file__).parents[1] / "ultralytics/cfg/models/master/v0_6/det/yolo-master-n.yaml").read_text(
    encoding="utf-8"
)


def test_syntax():
    ast.parse(S)


def test_contracts():
    assert "torch.cuda.set_device(index)" in S
    assert 'backend="nccl" if dist.is_nccl_available() else "gloo"' in S
    assert "device_ids=[self.device.index]" in S
    assert "resolve_ddp_policy(" in S
    assert "compile_enabled=bool(self.args.compile)" in S
    assert "find_unused_parameters=ddp_find_unused_parameters" in S
    assert "prepare_ddp(find_unused_parameters=ddp_find_unused_parameters)" in S
    assert "broadcast_buffers=False" in S
    assert "static_graph=ddp_static_graph" in S
    assert "amp_flag.item()" in S
    assert "self.batch_size % self.world_size" in S


def test_accumulation_and_collapse():
    assert "or i == nb - 1" in S
    assert "self.model.no_sync()" in S
    assert "if should_step:" in S


def test_v06_hybrid_gate_regression():
    assert "HybridAdaptiveGateMoE" in V06
    assert "static_graph=ddp_static_graph" in S
