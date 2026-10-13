"""Frozen candidate evaluations must preserve checkpoint and routing provenance."""

import json

import pytest
import torch

from scripts.evaluate_sparse_checkpoints import load_candidate_state


def write_candidate(tmp_path, **overrides):
    path = tmp_path / "candidate.state.pt"
    (tmp_path / "report.json").write_text(json.dumps({"weights_sha256": "base"}))
    payload = {"layer": "model.12", "policy": "native_top2", "state_dict": {"weight": torch.ones(2)}}
    payload.update(overrides)
    torch.save(payload, path)
    return path


def test_load_candidate_state_requires_matching_original_checkpoint(tmp_path):
    path = write_candidate(tmp_path)
    torch.testing.assert_close(load_candidate_state(path, "base", "model.12")["weight"], torch.ones(2))
    with pytest.raises(ValueError, match="base-checkpoint"):
        load_candidate_state(path, "other", "model.12")


@pytest.mark.parametrize("overrides", [{"layer": "model.3"}, {"policy": "native"}])
def test_load_candidate_rejects_incompatible_routing_metadata(tmp_path, overrides):
    path = write_candidate(tmp_path, **overrides)
    with pytest.raises(ValueError, match="layer/policy"):
        load_candidate_state(path, "base", "model.12")


@pytest.mark.parametrize("state, error", [({}, ValueError), ({"weight": "not-a-tensor"}, TypeError)])
def test_load_candidate_requires_nonempty_named_tensor_state(tmp_path, state, error):
    path = write_candidate(tmp_path, state_dict=state)
    with pytest.raises(error):
        load_candidate_state(path, "base", "model.12")


def test_load_candidate_rejects_invalid_payload_and_nonfinite_weights(tmp_path):
    path = write_candidate(tmp_path, state_dict={"weight": torch.tensor([float("nan")])})
    with pytest.raises(ValueError, match="non-finite"):
        load_candidate_state(path, "base", "model.12")
    torch.save([], path)
    with pytest.raises(TypeError, match="dictionary"):
        load_candidate_state(path, "base", "model.12")
