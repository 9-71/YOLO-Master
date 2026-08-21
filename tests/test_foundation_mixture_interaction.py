"""Offline contracts for the Foundation x Mixture 2x2 evidence runner."""

import json

import pytest

from scripts.foundation_mixture_interaction import (
    ARCHITECTURES,
    _paired_summary,
    build_run_plan,
    parse_args,
    run_matrix,
    summarize_run,
)


def _plan():
    return build_run_plan(
        dataset="/tmp/coco.yaml",
        project="/tmp/runs",
        teacher_model="/tmp/dino",
        seeds=[7],
        foundation_loss_weights=[0.05],
        epochs=2,
        fraction=0.25,
        imgsz=128,
        batch=2,
        device="cpu",
        workers=0,
        val=True,
    )


def _record(architecture, foundation, observed, *, seed=7, weight=0.05):
    return {
        "name": f"{architecture}-{foundation}",
        "architecture": architecture,
        "foundation": foundation,
        "seed": seed,
        "foundation_loss_weight": weight,
        "observed": observed,
    }


def test_plan_has_four_paired_cells_and_shared_initialization_contract():
    plan = _plan()

    assert len(plan) == 4
    assert [(item["architecture"], item["foundation"]) for item in plan] == [
        ("dense", False),
        ("dense", True),
        ("routed", False),
        ("routed", True),
    ]
    for baseline, foundation in zip(plan[::2], plan[1::2]):
        assert baseline["model"] == foundation["model"]
        assert baseline["initialization_contract"] == foundation["initialization_contract"]
        assert baseline["overrides"]["pretrained"] is foundation["overrides"]["pretrained"] is False
        assert baseline["overrides"]["foundation_enabled"] is False
        assert foundation["overrides"]["foundation_enabled"] is True
        assert foundation["overrides"]["foundation_model"] == "/tmp/dino"


def test_plan_rejects_invalid_architecture_map():
    with pytest.raises(ValueError, match="exactly"):
        build_run_plan(
            dataset="data",
            project="runs",
            teacher_model="teacher",
            architectures={"dense": ARCHITECTURES["dense"]},
            seeds=[0],
            foundation_loss_weights=[0.1],
            epochs=1,
            fraction=1.0,
            imgsz=64,
            batch=1,
            device="cpu",
            workers=0,
            val=False,
        )


def test_paired_summary_computes_interaction_without_imputing_metrics():
    records = [
        _record("dense", False, {"metrics/mAP50-95(B)": 0.20, "train/mixture_aux_loss": 0.0}),
        _record("dense", True, {"metrics/mAP50-95(B)": 0.23, "train/mixture_aux_loss": 0.0}),
        _record("routed", False, {"metrics/mAP50-95(B)": 0.30, "train/mixture_aux_loss": 0.4}),
        _record("routed", True, {"metrics/mAP50-95(B)": 0.34, "train/mixture_aux_loss": 0.5}),
    ]

    summary = _paired_summary(records)

    assert summary["accuracy_claim"] is False
    assert summary["pairs"][0]["observed_deltas"]["metrics/mAP50-95(B)"] == pytest.approx(0.03)
    assert summary["pairs"][1]["observed_deltas"]["metrics/mAP50-95(B)"] == pytest.approx(0.04)
    interaction = summary["interactions"][0]["observed_metric_interactions"]
    assert interaction["metrics/mAP50-95(B)"] == pytest.approx(0.01)
    assert interaction["train/mixture_aux_loss"] == pytest.approx(0.1)


def test_incomplete_pairs_do_not_report_complete_interactions():
    records = [
        _record("dense", False, {"metrics/mAP50-95(B)": 0.20}),
        _record("dense", True, {"metrics/mAP50-95(B)": 0.23}),
        _record("routed", False, {"metrics/mAP50-95(B)": 0.30}),
    ]

    interaction = _paired_summary(records)["interactions"][0]

    assert interaction["complete"] is False
    assert interaction["observed_metric_interactions"] == {}


def test_paired_summary_aggregates_observed_deltas_by_seed_without_imputation():
    records = [
        _record("dense", False, {"metrics/mAP50-95(B)": 0.20}, seed=7),
        _record("dense", True, {"metrics/mAP50-95(B)": 0.22}, seed=7),
        _record("routed", False, {"metrics/mAP50-95(B)": 0.30}, seed=7),
        _record("routed", True, {"metrics/mAP50-95(B)": 0.33}, seed=7),
        _record("dense", False, {"metrics/mAP50-95(B)": 0.10}, seed=8),
        _record("dense", True, {"metrics/mAP50-95(B)": 0.14}, seed=8),
        _record("routed", False, {"metrics/mAP50-95(B)": 0.40}, seed=8),
        _record("routed", True, {"metrics/mAP50-95(B)": 0.48}, seed=8),
    ]

    summary = _paired_summary(records)

    dense = summary["foundation_delta_aggregates"][0]
    routed = summary["foundation_delta_aggregates"][1]
    interaction = summary["interaction_aggregates"][0]
    assert dense["complete_pairs"] == routed["complete_pairs"] == 2
    assert dense["observed_delta_summary"]["metrics/mAP50-95(B)"] == {
        "n": 2,
        "mean": pytest.approx(0.03),
        "sample_std": pytest.approx(0.01414214),
    }
    assert routed["observed_delta_summary"]["metrics/mAP50-95(B)"]["mean"] == pytest.approx(0.055)
    assert interaction["complete_interactions"] == 2
    assert interaction["observed_interaction_summary"]["metrics/mAP50-95(B)"]["mean"] == pytest.approx(0.025)


def test_run_matrix_is_resumable_and_never_claims_accuracy(tmp_path):
    output = tmp_path / "interaction.json"
    calls = []

    def fake_runner(spec):
        calls.append(spec["name"])
        return _record(spec["architecture"], spec["foundation"], {"metrics/mAP50-95(B)": 0.1}) | {
            "name": spec["name"],
            "foundation_loss_weight": spec["foundation_loss_weight"],
            "seed": spec["seed"],
        }

    plan = _plan()
    first = run_matrix(plan[:1], output, runner=fake_runner)
    assert first["completed_runs"] == 1
    resumed = run_matrix(plan, output, runner=fake_runner, resume=True)

    assert len(calls) == 4
    assert resumed["completed_runs"] == 4
    assert resumed["accuracy_claim"] is False
    assert json.loads(output.read_text()) == resumed

    legacy_payload = json.loads(output.read_text())
    legacy_payload["summary"].pop("foundation_delta_aggregates")
    legacy_payload["summary"].pop("interaction_aggregates")
    legacy_payload["records"][0]["epoch_time_s"] = 12.5
    legacy_payload["records"][0]["observed"]["train/epoch_time_s"] = 12.5
    output.write_text(json.dumps(legacy_payload), encoding="utf-8")
    refreshed = run_matrix(
        plan, output, runner=lambda spec: pytest.fail(f"unexpected rerun: {spec['name']}"), resume=True
    )
    assert refreshed["summary"]["foundation_delta_aggregates"]
    assert refreshed["summary"]["interaction_aggregates"]
    assert refreshed["records"][0]["train_elapsed_s"] == pytest.approx(12.5)
    assert refreshed["records"][0]["observed"]["train/elapsed_s"] == pytest.approx(12.5)


def test_resume_rejects_configuration_drift(tmp_path):
    output = tmp_path / "interaction.json"
    plan = _plan()

    run_matrix(
        plan[:1],
        output,
        runner=lambda spec: _record(spec["architecture"], spec["foundation"], {}) | {"name": spec["name"]},
    )
    changed_plan = build_run_plan(
        dataset="/tmp/coco.yaml",
        project="/tmp/runs",
        teacher_model="/tmp/dino",
        seeds=[7],
        foundation_loss_weights=[0.05],
        epochs=3,
        fraction=0.25,
        imgsz=128,
        batch=2,
        device="cpu",
        workers=0,
        val=True,
    )

    with pytest.raises(ValueError, match="changed or truncated plan"):
        run_matrix(
            changed_plan[:1],
            output,
            runner=lambda spec: _record(spec["architecture"], spec["foundation"], {}) | {"name": spec["name"]},
            resume=True,
        )


def test_summary_missing_safe_and_cli_boundaries(tmp_path):
    missing = summarize_run(tmp_path / "missing")
    assert missing["results_available"] is False

    run_dir = tmp_path / "observed"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text(
        "epoch,time,metrics/mAP_small(B),metrics/mAP_medium(B),metrics/mAP_large(B)\n1,12.5,0.1,0.2,0.3\n",
        encoding="utf-8",
    )
    observed = summarize_run(run_dir)
    assert observed["train_elapsed_s"] == pytest.approx(12.5)
    assert observed["observed"]["train/elapsed_s"] == pytest.approx(12.5)
    assert observed["observed"]["metrics/mAP_small(B)"] == pytest.approx(0.1)

    assert parse_args(["--epochs", "2", "--no-val"]).val is False
    with pytest.raises(SystemExit):
        parse_args(["--epochs", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--fraction", "0"])
