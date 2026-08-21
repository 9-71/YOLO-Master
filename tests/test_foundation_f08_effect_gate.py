"""Contracts for the finite F08 MPS effect gate runner."""

import csv
import json
from pathlib import Path

from torch import nn

import scripts.foundation_f08_effect_gate as f08
from scripts.foundation_f08_effect_gate import (
    ARMS,
    REQUIRED_VALIDATION_METRICS,
    _paired_summary,
    build_run_plan,
    prepare_fixed_subsets,
    run_effect_gate,
    student_only_contract,
    summarize_run,
)


def test_f08_plan_has_four_arms_and_fixed_kd_contract(tmp_path):
    plan = build_run_plan(
        dataset="subset.yaml",
        project=str(tmp_path / "runs"),
        seeds=[0, 1, 2],
        epochs=10,
        imgsz=256,
        batch=2,
        device="mps",
    )

    assert len(plan) == 12
    assert [item["arm"] for item in plan[:3]] == ["B0"] * 3
    assert [item["arm"] for item in plan[3:6]] == ["D0"] * 3
    assert [item["arm"] for item in plan[6:9]] == ["D1"] * 3
    assert [item["arm"] for item in plan[9:]] == ["D2"] * 3
    assert set(ARMS) == {"B0", "D0", "D1", "D2"}

    b0, d0, d1, d2 = (plan[index] for index in (0, 3, 6, 9))
    assert b0["overrides"]["foundation_enabled"] is False
    assert b0["overrides"]["foundation_loss_weight"] == 0.0
    assert d0["overrides"]["foundation_loss"] == "cosine"
    assert d1["overrides"]["foundation_loss"] == "relational"
    assert d1["overrides"]["foundation_relation_mode"] == "sampled"
    assert d2["overrides"]["foundation_loss"] == "hybrid"
    for item in plan:
        assert item["overrides"]["optimizer"] == "SGD"
        assert item["overrides"]["lr0"] == 0.01
        assert item["overrides"]["amp"] is False
        assert item["overrides"]["foundation_weight_schedule"] == "gate_decay"


def test_prepare_fixed_subsets_is_deterministic(tmp_path):
    root = tmp_path / "coco"
    for split, count in (("train2017", 8), ("val2017", 5)):
        image_dir = root / "images" / split
        image_dir.mkdir(parents=True)
        for index in range(count):
            (image_dir / f"{index:012d}.jpg").write_bytes(b"")
    (root / "coco2017.yaml").write_text("path: old\ntrain: old\nval: old\nnames: {0: thing}\n", encoding="utf-8")

    first = prepare_fixed_subsets(root, tmp_path / "out", train_size=3, val_size=2, seed=7)
    second = prepare_fixed_subsets(root, tmp_path / "out2", train_size=3, val_size=2, seed=7)
    assert Path(first["train_list"]).read_text() == Path(second["train_list"]).read_text()
    assert Path(first["val_list"]).read_text() == Path(second["val_list"]).read_text()
    data = json.loads(json.dumps(__import__("yaml").safe_load(Path(first["dataset_yaml"]).read_text())))
    assert data["path"] == str(root.resolve())
    assert data["subset"]["train_size"] == 3
    assert data["subset"]["val_size"] == 2


def test_run_effect_gate_resume_skips_completed_records(tmp_path):
    plan = build_run_plan(dataset="subset.yaml", project=str(tmp_path / "runs"), seeds=[0], imgsz=256)
    output = tmp_path / "report.json"
    calls = []

    def runner(spec):
        calls.append(spec["name"])
        return {"name": spec["name"], "arm": spec["arm"], "seed": spec["seed"], "observed": {}}

    first = run_effect_gate(plan[:2], output, data_contract={"dataset_yaml": "subset.yaml"}, runner=runner)
    assert first["completed_runs"] == 2
    second = run_effect_gate(
        plan[:2], output, data_contract={"dataset_yaml": "subset.yaml"}, runner=runner, resume=True
    )
    assert second["completed_runs"] == 2
    assert calls == ["b0-s0", "d0-s0"]


def test_run_effect_gate_persists_interruption_for_resume(tmp_path):
    plan = build_run_plan(dataset="subset.yaml", project=str(tmp_path / "runs"), seeds=[0], imgsz=128)
    output = tmp_path / "report.json"

    def interrupt(_spec):
        raise KeyboardInterrupt

    try:
        run_effect_gate(plan[:1], output, data_contract={}, runner=interrupt)
    except KeyboardInterrupt:
        pass
    payload = json.loads(output.read_text())
    assert payload["completed_runs"] == 0
    assert payload["interrupted_runs"][0]["status"] == "interrupted"

    def complete(spec):
        return {"name": spec["name"], "arm": spec["arm"], "seed": spec["seed"], "observed": {}}

    resumed = run_effect_gate(plan[:1], output, data_contract={}, runner=complete, resume=True)
    assert resumed["completed_runs"] == 1
    assert resumed["interrupted_runs"] == []


def test_train_one_resumes_from_last_healthy_checkpoint(tmp_path, monkeypatch):
    import ultralytics

    plan = build_run_plan(dataset="subset.yaml", project=str(tmp_path / "runs"), seeds=[0], imgsz=128)
    spec = plan[0]
    healthy = Path(spec["project"]) / spec["name"] / "weights" / "last_healthy.pt"
    healthy.parent.mkdir(parents=True)
    healthy.write_bytes(b"checkpoint")
    captured = {}

    class DummyYOLO:
        def __init__(self, source):
            captured["source"] = source
            self.trainer = type("Trainer", (), {"save_dir": Path(spec["project"]) / spec["name"]})()

        def train(self, **overrides):
            captured["resume"] = overrides.get("resume")

    monkeypatch.setattr(ultralytics, "YOLO", DummyYOLO)
    monkeypatch.setattr(f08, "summarize_run", lambda *args, **kwargs: {"observed": {}})
    result = f08._train_one(spec)
    assert captured["source"] == str(healthy)
    assert captured["resume"] == str(healthy)
    assert result["resumed_from_last"] is True
    assert result["resume_checkpoint"] == str(healthy)


def test_missing_metrics_are_not_interpolated():
    records = [
        {"arm": "B0", "seed": 0, "observed": {"metrics/mAP50-95(B)": 0.2}},
        {"arm": "D0", "seed": 0, "observed": {"metrics/mAP50-95(B)": 0.21}},
    ]
    summary = _paired_summary(records, [0])
    assert summary["pairs"][0]["observed_metric_deltas_vs_B0"] == {"metrics/mAP50-95(B)": 0.01}
    assert "metrics/mAP50(B)" not in summary["pairs"][0]["observed_metric_deltas_vs_B0"]
    assert summary["accuracy_claim"] is False


def test_summarize_run_reads_small_medium_large_and_telemetry(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", *REQUIRED_VALIDATION_METRICS, "time"))
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 9,
                "metrics/mAP50-95(B)": 0.3,
                "metrics/mAP50(B)": 0.5,
                "metrics/mAP_small(B)": 0.1,
                "metrics/mAP_medium(B)": 0.2,
                "metrics/mAP_large(B)": 0.4,
                "time": 12.0,
            }
        )
    (run_dir / "telemetry.json").write_text(
        json.dumps(
            {
                "aggregation": {"world_size": 1},
                "ranks": [
                    {
                        "metadata": {"rank": 0},
                        "steps": {"p50_milliseconds": 10},
                        "memory": {"measurement": "mps_sampled_current_allocated_memory", "is_true_peak": False},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_run(run_dir, imgsz=256, device="cpu")
    assert summary["observed"]["metrics/mAP_small(B)"] == 0.1
    assert summary["observed"]["metrics/mAP_medium(B)"] == 0.2
    assert summary["observed"]["metrics/mAP_large(B)"] == 0.4
    assert summary["telemetry"]["memory_measurement"] == "mps_sampled_current_allocated_memory"


def test_student_only_checkpoint_scan_rejects_teacher_keys(tmp_path, monkeypatch):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.teacher_model = nn.Linear(2, 2)

        def forward(self, image):
            return self.teacher_model(image[:, :, :1, :1].flatten(1))

    model = Tiny()
    path = tmp_path / "last.pt"
    path.write_bytes(b"placeholder")
    import ultralytics.nn.foundation_distill_model as foundation_module
    import ultralytics.nn.tasks as tasks_module
    import ultralytics.utils.torch_utils as torch_utils

    monkeypatch.setattr(tasks_module, "load_checkpoint", lambda *args, **kwargs: (model, {}))
    monkeypatch.setattr(foundation_module, "strip_foundation_distillation_model", lambda value: value)
    monkeypatch.setattr(torch_utils, "get_num_params", lambda value: sum(item.numel() for item in value.parameters()))
    monkeypatch.setattr(torch_utils, "get_flops", lambda value, imgsz=640: 1.25)
    result = student_only_contract(path, imgsz=32, device="cpu", latency_runs=1)
    assert result["teacher_state_clean"] is False
    assert any("teacher_model" in key for key in result["teacher_state_keys"])
    assert result["params"] == 6
