"""Contracts for the real-data F15 effect-gate runner."""

import json

import pytest

from scripts.foundation_f15_real_effect_gate import (
    TASKS,
    build_run_plan,
    parse_args,
    read_last_results,
    run_matrix,
    summarize_run,
)


def _plan():
    return build_run_plan(
        dataset="/tmp/coco.yaml",
        model="/tmp/model.yaml",
        teacher_model="/tmp/dino",
        project="/tmp/runs",
        seeds=[1, 2],
        foundation_loss_weights=[0.01, 0.05],
        epochs=3,
        fraction=0.25,
        imgsz=128,
        batch=2,
        device="cpu",
        workers=0,
        val=True,
    )


def test_plan_has_paired_baseline_and_foundation_runs():
    plan = _plan()

    assert len(plan) == 8
    assert [(item["foundation"], item["seed"], item["foundation_loss_weight"]) for item in plan[:4]] == [
        (False, 1, 0.01),
        (True, 1, 0.01),
        (False, 2, 0.01),
        (True, 2, 0.01),
    ]
    for baseline, foundation in zip(plan[::2], plan[1::2]):
        assert baseline["seed"] == foundation["seed"]
        assert baseline["foundation_loss_weight"] == foundation["foundation_loss_weight"]
        assert baseline["overrides"]["pretrained"] is foundation["overrides"]["pretrained"] is False
        assert baseline["overrides"]["data"] == foundation["overrides"]["data"]
        assert baseline["overrides"]["foundation_enabled"] is False
        assert foundation["overrides"]["foundation_enabled"] is True
        assert foundation["overrides"]["foundation_multitask_tasks"] == TASKS


def test_parse_args_enforces_effect_gate_defaults_and_boundaries():
    args = parse_args(["--epochs", "5", "--fraction", "0.1", "--no-val"])
    assert args.epochs == 5
    assert args.fraction == 0.1
    assert args.val is False
    with pytest.raises(SystemExit):
        parse_args(["--epochs", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--fraction", "0"])


def test_run_matrix_writes_progress_and_never_claims_accuracy(tmp_path):
    output = tmp_path / "effect.json"
    calls = []

    def fake_runner(spec):
        calls.append(spec["name"])
        return {
            "name": spec["name"],
            "seed": spec["seed"],
            "foundation": spec["foundation"],
            "foundation_loss_weight": spec["foundation_loss_weight"],
            "results_available": False,
        }

    result = run_matrix(_plan()[:2], output, runner=fake_runner)

    assert calls == ["baseline-s1-w0.01", "foundation-s1-w0.01"]
    assert result["completed_runs"] == 2
    assert result["total_runs"] == 2
    assert result["accuracy_claim"] is False
    assert result["real_data"] is True
    assert result["summary"]["paired_runs"] == 1
    assert json.loads(output.read_text()) == result


def test_run_matrix_resume_skips_completed_names(tmp_path):
    output = tmp_path / "effect.json"
    calls = []

    def fake_runner(spec):
        calls.append(spec["name"])
        return {
            "name": spec["name"],
            "seed": spec["seed"],
            "foundation": spec["foundation"],
            "foundation_loss_weight": spec["foundation_loss_weight"],
            "validation_metrics": {},
            "results_available": False,
        }

    plan = _plan()[:2]
    run_matrix(plan[:1], output, runner=fake_runner)
    resumed = run_matrix(plan, output, runner=fake_runner, resume=True)

    assert calls == ["baseline-s1-w0.01", "foundation-s1-w0.01"]
    assert resumed["completed_runs"] == 2


def test_results_summary_is_missing_safe(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert read_last_results(run_dir / "results.csv") == {}
    summary = summarize_run(run_dir)
    assert summary["results_available"] is False
    assert summary["checkpoint"] is None
