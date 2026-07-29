# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Regression test: MoEPruner must keep ``model.yaml`` consistent with the pruned
expert count so a pruned model survives a YAML-based rebuild during retraining
(the prune -> LoRA / full fine-tune recovery workflow).

Without the fix, ``DetectionModel(pruned.yaml)`` rebuilds with ES_MOE's default
expert count and ``intersect_dicts`` silently drops the reduced expert/router
weights, re-inflating the model with randomly initialized experts.
"""
import copy

import yaml

from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.moe.pruning import MoEPruner
from ultralytics.nn.tasks import DetectionModel

CFG = "ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone.yaml"


def _experts(model):
    return [m.num_experts for _, m in model.named_modules() if isinstance(m, ES_MOE)]


def _build_reduced(num_experts):
    """Build a model whose ES_MOE blocks each have ``num_experts`` (stand-in for a pruned model)."""
    d = yaml.safe_load(open(CFG))
    for layer in d["backbone"]:
        if layer[2] == "ES_MOE":
            layer[3] = [layer[3][0], num_experts]
    return DetectionModel(d, ch=3, nc=10, verbose=False)


def test_reinflation_without_sync():
    """Reproduce the bug: reduced model + un-synced (full) yaml re-inflates on rebuild."""
    reduced = _build_reduced(2)
    assert _experts(reduced) == [2, 2, 2, 2]
    reduced.yaml = yaml.safe_load(open(CFG))  # bug state: yaml still describes default experts
    rebuilt = DetectionModel(reduced.yaml, ch=3, nc=10, verbose=False)
    assert _experts(rebuilt) != [2, 2, 2, 2], "expected re-inflation when yaml is not synced"


def test_sync_preserves_pruned_experts():
    """The fix: MoEPruner._sync_yaml_num_experts writes the reduced count into yaml,
    so a YAML rebuild preserves the pruned architecture."""
    reduced = _build_reduced(2)
    reduced.yaml = yaml.safe_load(open(CFG))  # start from the un-synced (buggy) yaml
    MoEPruner._sync_yaml_num_experts(type("D", (), {})(), reduced)  # method only uses self for logging
    es_args = [layer[3] for layer in reduced.yaml["backbone"] if layer[2] == "ES_MOE"]
    assert all(a[-1] == 2 for a in es_args), f"yaml not synced: {es_args}"
    rebuilt = DetectionModel(reduced.yaml, ch=3, nc=10, verbose=False)
    assert _experts(rebuilt) == [2, 2, 2, 2], "pruned expert count must survive the rebuild"
