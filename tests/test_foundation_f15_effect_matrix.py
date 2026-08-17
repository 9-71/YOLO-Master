"""Tests for the reproducible F15 seed/KD-weight matrix wrapper."""

import json

from scripts import foundation_f15_effect_matrix as matrix


def _record(seed: int, weight: float, foundation_delta: float, baseline_delta: float) -> dict[str, object]:
    return {
        "student_model": "student.yaml",
        "tasks": ["detect", "segment", "pose"],
        "seed": seed,
        "foundation_loss_weight": weight,
        "summary": {
            "foundation_task_delta_first_last": foundation_delta,
            "baseline_task_delta_first_last": baseline_delta,
            "foundation_step_overhead_ratio": 0.1,
            "foundation_supervised_task_gate": True,
            "foundation_nonzero_kd_gate": True,
            "foundation_p4_grad_min": 1.0,
        },
    }


def test_matrix_summary_aggregates_mean_std_and_gates():
    payload = matrix._summary(
        [
            _record(1, 0.01, -0.2, -0.3),
            _record(2, 0.01, -0.4, -0.5),
            _record(1, 0.05, -0.1, -0.3),
        ]
    )

    assert payload["weights"][0]["foundation_loss_weight"] == 0.01
    assert payload["weights"][0]["runs"] == 2
    assert payload["weights"][0]["foundation_task_delta_mean"] == -0.3
    assert payload["weights"][0]["all_mechanism_gates"] is True
    assert payload["weights"][1]["runs"] == 1


def test_run_matrix_preserves_seed_weight_provenance(monkeypatch):
    calls = []

    def fake_run(teacher_model, steps, align_dim, *, seed, foundation_loss_weight):
        calls.append((teacher_model, steps, align_dim, seed, foundation_loss_weight))
        return _record(seed, foundation_loss_weight, -0.1, -0.2)

    monkeypatch.setattr(matrix, "run_benchmark", fake_run)
    result = matrix.run_matrix("teacher", [3, 4], [0.01, 0.05], steps=2, align_dim=8)

    assert calls == [
        ("teacher", 2, 8, 3, 0.01),
        ("teacher", 2, 8, 4, 0.01),
        ("teacher", 2, 8, 3, 0.05),
        ("teacher", 2, 8, 4, 0.05),
    ]
    assert result["synthetic_batch"] is True
    assert result["real_accuracy_claim"] is False
    json.dumps(result)
