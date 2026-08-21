"""Admission and paired-run contracts for the F09 foreground effect gate."""

import json

from scripts import foundation_f08_effect_gate as f08
from scripts.foundation_f09_foreground_effect_gate import (
    _paired_summary,
    build_run_plan,
    f08_admission,
    main,
    run_effect_gate,
)


def _completed_f08_report(tmp_path):
    plan = f08.build_run_plan(
        dataset=str(tmp_path / "subset.yaml"),
        project=str(tmp_path / "f08-runs"),
        seeds=[0, 1, 2],
        imgsz=256,
        batch=4,
        device="mps",
    )
    records = []
    for spec in plan:
        observed = {metric: 0.1 + spec["seed"] * 0.01 for metric in f08.REQUIRED_VALIDATION_METRICS}
        if spec["arm"] == "D2":
            observed["train/foundation_loss"] = 0.2
        records.append(
            {
                "name": spec["name"],
                "arm": spec["arm"],
                "seed": spec["seed"],
                "foundation": spec["foundation"],
                "foundation_loss": spec["foundation_loss"],
                "observed": observed,
                "student_only": {"teacher_state_clean": True},
            }
        )
    return {
        "benchmark": f08.BENCHMARK,
        "real_data": True,
        "accuracy_claim": False,
        "data_contract": {"dataset_yaml": str(tmp_path / "subset.yaml")},
        "plan": plan,
        "records": records,
        "completed_runs": len(records),
        "total_runs": len(plan),
        "interrupted_runs": [],
    }


def test_f09_admission_requires_complete_analyzable_f08_evidence(tmp_path):
    admission = f08_admission(_completed_f08_report(tmp_path))

    assert admission["ready"] is True
    assert admission["seeds"] == [0, 1, 2]
    assert all(check["passed"] for check in admission["checks"])


def test_f09_admission_rejects_incomplete_f08_report(tmp_path):
    report = _completed_f08_report(tmp_path)
    report["records"] = report["records"][:-1]
    report["completed_runs"] -= 1
    admission = f08_admission(report)

    assert admission["ready"] is False
    assert any(check["name"] == "matrix_complete" and not check["passed"] for check in admission["checks"])


def test_f09_plan_only_changes_foreground_weighting(tmp_path):
    admission = f08_admission(_completed_f08_report(tmp_path))
    plan = build_run_plan(admission=admission, project=str(tmp_path / "f09-runs"))

    assert [item["arm"] for item in plan[:3]] == ["D2"] * 3
    assert [item["arm"] for item in plan[3:]] == ["D3"] * 3
    d2, d3 = plan[0], plan[3]
    assert d2["foreground_weighting"] is False
    assert d3["foreground_weighting"] is True
    assert d2["overrides"]["foundation_loss"] == d3["overrides"]["foundation_loss"] == "hybrid"
    assert d2["overrides"]["foundation_weight_schedule"] == d3["overrides"]["foundation_weight_schedule"]
    changed = {key for key in d2["overrides"] if d2["overrides"][key] != d3["overrides"][key]}
    assert changed == {"name", "foundation_foreground_weighting"}
    assert d3["foreground_weights"] == {"interior": 1.5, "boundary": 1.0, "background": 0.25}


def test_f09_pair_summary_does_not_impute_missing_metrics():
    summary = _paired_summary(
        [
            {"arm": "D2", "seed": 0, "observed": {"metrics/mAP50-95(B)": 0.2}},
            {"arm": "D3", "seed": 0, "observed": {"metrics/mAP50-95(B)": 0.21}},
        ],
        [0],
    )

    assert summary["pairs"][0]["observed_metric_deltas_D3_minus_D2"] == {"metrics/mAP50-95(B)": 0.01}
    assert summary["required_validation_metrics_present"] is False
    assert summary["accuracy_claim"] is False


def test_f09_resume_skips_completed_runs(tmp_path):
    admission = f08_admission(_completed_f08_report(tmp_path))
    plan = build_run_plan(admission=admission, project=str(tmp_path / "f09-runs"))[:2]
    output = tmp_path / "f09.json"
    calls = []

    def runner(spec):
        calls.append(spec["name"])
        return {"name": spec["name"], "arm": spec["arm"], "seed": spec["seed"], "observed": {}}

    run_effect_gate(plan, output, admission=admission, runner=runner)
    run_effect_gate(plan, output, admission=admission, runner=runner, resume=True)
    assert calls == ["d2-s0", "d2-s1"]


def test_f09_resume_clears_interrupted_run(tmp_path):
    admission = f08_admission(_completed_f08_report(tmp_path))
    plan = build_run_plan(admission=admission, project=str(tmp_path / "f09-runs"))[:1]
    output = tmp_path / "f09.json"

    def interrupt(_spec):
        raise KeyboardInterrupt

    try:
        run_effect_gate(plan, output, admission=admission, runner=interrupt)
    except KeyboardInterrupt:
        pass

    def complete(spec):
        return {"name": spec["name"], "arm": spec["arm"], "seed": spec["seed"], "observed": {}}

    resumed = run_effect_gate(plan, output, admission=admission, runner=complete, resume=True)
    assert resumed["interrupted_runs"] == []


def test_f09_dry_run_reports_f08_blocker(tmp_path, capsys):
    path = tmp_path / "f08.json"
    payload = _completed_f08_report(tmp_path)
    payload["completed_runs"] = 0
    payload["records"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")

    main(["--f08-report", str(path), "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["ready"] is False
    assert result["plan"] if "plan" in result else True
