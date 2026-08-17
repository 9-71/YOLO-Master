"""Tests for the read-only F15 release audit contracts."""

import json
from pathlib import Path

from scripts.foundation_f15_release_audit import audit_effect_report, audit


def _report(path: Path, *, accuracy_claim: bool = False):
    payload = {
        "schema_version": 1,
        "benchmark": "f15_real_coco_effect_gate",
        "real_data": True,
        "accuracy_claim": accuracy_claim,
        "records": [],
        "completed_runs": 0,
        "total_runs": 0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_effect_report_rejects_accuracy_claim(tmp_path):
    path = tmp_path / "effect.json"
    _report(path, accuracy_claim=True)
    checks = audit_effect_report(path)
    assert any(check["name"] == "accuracy_claim_false" and not check["passed"] for check in checks)


def test_audit_is_explicitly_not_an_accuracy_gate(tmp_path):
    report = tmp_path / "effect.json"
    _report(report)
    result = audit([report], [tmp_path / "missing-checkpoints"])

    assert result["accuracy_claim"] is False
    assert result["ready_for_accuracy_claim"] is False
    assert result["ready_for_next_phase"] is False
    assert result["release_audit_passed"] is False
    assert any(check["name"] == "foundation_checkpoints_present" for check in result["checks"])
