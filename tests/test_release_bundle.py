"""Focused tests for manifest-rooted release bundle audits."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from agent.runtime.cli.contract import manifest_checksum
from agent.runtime.cli.release import run_release_audit
from agent.runtime.release import audit_manifest, write_release_bundle


def _write(path: Path, value: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")
    return path


def _fixture(tmp_path: Path, *, status: str = "ok", complete: bool = True) -> tuple[Path, dict[str, Path]]:
    model = _write(tmp_path / "model.yaml", "nc: 1\nbackbone: []\nhead: []\n")
    checkpoint = _write(tmp_path / "best.pt", b"checkpoint")
    args = _write(tmp_path / "args.yaml", "task: detect\nmodel: model.yaml\n")
    routing = _write(tmp_path / "routing.json", json.dumps({"layers": {"model.1": {"entropy": 0.9}}}))
    benchmark = _write(tmp_path / "results.json", json.dumps({"schema_version": 1, "cases": []}))
    planner = _write(
        tmp_path / "placement.json",
        json.dumps(
            {
                "schema_version": 1,
                "model_fingerprint": "model-fingerprint",
                "planner_backend": "legacy",
                "solver": "none",
                "budget": {"max_adapter_params": 0},
                "targets": [],
                "constraints": {"hard": [], "soft": []},
                "status": "FALLBACK",
            }
        ),
    )
    merge = _write(tmp_path / "runtime_metadata.json", json.dumps({"backend": "lora", "merge_mode": "exact"}))
    prune = _write(
        tmp_path / "pruned.pt.prune.json",
        json.dumps({"schema_version": 1, "source_model": str(checkpoint), "output_model": str(checkpoint)}),
    )
    governance = _write(
        tmp_path / "model-registry.yaml",
        "schema_version: 1\nmodels:\n  - name: fixture\n    path: model.yaml\n    task: detect\n    status: stable\n    export: {onnx: full_model_roundtrip}\n",
    )
    export_matrix = _write(
        tmp_path / "export-capability-matrix.yaml",
        "schema_version: 1\nformats:\n  onnx: {supported: true, default: dense_fallback}\n",
    )
    artifacts = [
        {"kind": "checkpoint", "label": "best", "path": str(checkpoint)},
        {"kind": "model_yaml", "path": str(model)},
        {"kind": "config", "path": str(args)},
        {"kind": "routing_summary", "path": str(routing)},
        {"kind": "benchmark", "path": str(benchmark)},
        {"kind": "placement_plan", "path": str(planner)},
        {"kind": "merge_manifest", "path": str(merge)},
        {"kind": "prune_manifest", "path": str(prune)},
    ]
    if not complete:
        artifacts = [item for item in artifacts if item["kind"] not in {"routing_summary", "benchmark"}]
    manifest = {
        "schema_version": 1,
        "created_at": "2026-07-25T00:00:00Z",
        "skill": "yolo.pipeline.experiment",
        "status": status,
        "request": {
            "skill": "yolo.pipeline.experiment",
            "inputs": {"model": str(model), "task": "detect"},
            "params": {"stages": ["train", "val", "benchmark"]},
        },
        "result": {
            "best_checkpoint": str(checkpoint),
            "stages": {"train": {"status": status}},
            "export": {"format": "onnx", "status": "ok"},
        },
        "artifacts": artifacts,
    }
    manifest["manifest_sha256"] = manifest_checksum(manifest)
    manifest_path = _write(tmp_path / "skill_manifest.json", json.dumps(manifest, indent=2) + "\n")
    paths = {
        "model": model,
        "checkpoint": checkpoint,
        "args": args,
        "routing": routing,
        "benchmark": benchmark,
        "planner": planner,
        "merge": merge,
        "prune": prune,
        "governance": governance,
        "export_matrix": export_matrix,
    }
    return manifest_path, paths


def _params(paths: dict[str, Path]) -> dict[str, str]:
    return {
        "governance_registry": str(paths["governance"]),
        "export_matrix": str(paths["export_matrix"]),
    }


def test_complete_stable_manifest_is_publishable_and_checksum_addressed(tmp_path: Path) -> None:
    manifest, paths = _fixture(tmp_path)
    bundle = audit_manifest(manifest, params=_params(paths))

    assert bundle.decision.status == "publishable"
    assert bundle.identity["model_yaml"]["sha256"]
    assert bundle.identity["checkpoint"]["sha256"]
    assert all(item.status == "valid" for item in bundle.evidence if item.required)

    output = write_release_bundle(bundle, tmp_path / "release_bundle.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["decision"]["status"] == "publishable"
    expected_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert payload["source_manifest"]["sha256"] == expected_digest


def test_release_handler_writes_bundle_for_fixture(tmp_path: Path) -> None:
    manifest, paths = _fixture(tmp_path)
    request = {
        "skill": "yolo.release.audit",
        "inputs": {"manifest": str(manifest)},
        "params": {
            **_params(paths),
            "artifact_root": str(tmp_path),
            "output": str(tmp_path / "handler-bundle.json"),
        },
        "policy": {"dry_run": False},
    }

    payload = run_release_audit(request)

    assert payload["decision"]["status"] == "publishable"
    assert Path(payload["artifacts"][0]["path"]).exists()


def test_missing_evidence_is_experimental_with_reasons(tmp_path: Path) -> None:
    manifest, paths = _fixture(tmp_path, complete=False)
    bundle = audit_manifest(manifest, params=_params(paths))

    assert bundle.decision.status == "experimental"
    assert {"routing_summary", "benchmark_result"} <= set(bundle.decision.missing)
    assert {item.kind for item in bundle.evidence if item.status == "missing"} >= {
        "routing_summary",
        "benchmark_result",
    }


def test_tampered_manifest_is_refused(tmp_path: Path) -> None:
    manifest, paths = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    bundle = audit_manifest(manifest, params=_params(paths))
    assert bundle.decision.status == "refused"
    assert any("manifest checksum" in reason for reason in bundle.decision.hard_failures)


def test_malformed_manifest_schema_is_refused_without_raising(tmp_path: Path) -> None:
    manifest, paths = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = "not-an-int"
    payload["manifest_sha256"] = manifest_checksum({**payload, "manifest_sha256": None})
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    bundle = audit_manifest(manifest, params=_params(paths))

    assert bundle.decision.status == "refused"
    assert any("schema_version" in reason for reason in bundle.decision.hard_failures)


def test_path_escape_is_refused(tmp_path: Path) -> None:
    manifest, paths = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"].append({"kind": "benchmark", "path": "/etc/passwd"})
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = manifest_checksum(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    bundle = audit_manifest(manifest, params=_params(paths), artifact_root=tmp_path)
    assert bundle.decision.status == "refused"
    assert any("outside artifact root" in reason for reason in bundle.decision.hard_failures)


@pytest.mark.parametrize("source_status", ["failed", "running", "partial"])
def test_non_success_source_status_cannot_be_published(tmp_path: Path, source_status: str) -> None:
    manifest, paths = _fixture(tmp_path, status=source_status)
    bundle = audit_manifest(manifest, params=_params(paths))

    assert bundle.decision.status == "refused"
    assert any("source manifest status" in reason for reason in bundle.decision.hard_failures)


def test_legacy_manifest_is_migration_report_and_uses_explicit_references(tmp_path: Path) -> None:
    save_dir = tmp_path / "legacy-run"
    model = _write(tmp_path / "legacy-model.yaml", "nc: 1\nbackbone: []\nhead: []\n")
    checkpoint = _write(save_dir / "weights" / "best.pt", b"legacy-checkpoint")
    args = _write(save_dir / "args.yaml", "task: detect\n")
    manifest = _write(
        tmp_path / "skill_manifest.json",
        json.dumps(
            {
                "skill": "yolo.train",
                "status": "ok",
                "artifacts": [
                    {"kind": "checkpoint", "path": "weights/best.pt"},
                    {"kind": "config", "path": "args.yaml"},
                ],
                "environment": {"references": {"model": {"resolved": str(model)}}},
                "job": {"save_dir": str(save_dir)},
            }
        ),
    )

    bundle = audit_manifest(manifest, params={"governance_registry": str(tmp_path / "missing.yaml")})

    assert bundle.compatibility["kind"] == "legacy_unversioned"
    assert bundle.compatibility["checksum"]["present"] is False
    assert bundle.decision.status == "refused"
    assert any("migration required" in reason for reason in bundle.decision.hard_failures)
    assert next(item for item in bundle.evidence if item.kind == "checkpoint").path == str(checkpoint.resolve())
    assert next(item for item in bundle.evidence if item.kind == "resolved_args").path == str(args.resolve())
    assert bundle.identity["model"] == str(model.resolve())
    assert args.exists()


def test_legacy_relative_path_escape_is_refused(tmp_path: Path) -> None:
    save_dir = tmp_path / "legacy-run"
    model = _write(tmp_path / "legacy-model.yaml", "nc: 1\nbackbone: []\nhead: []\n")
    manifest = _write(
        tmp_path / "skill_manifest.json",
        json.dumps(
            {
                "skill": "yolo.train",
                "status": "ok",
                "artifacts": [{"kind": "checkpoint", "path": "../../etc/passwd"}],
                "environment": {"references": {"model": {"resolved": str(model)}}},
                "job": {"save_dir": str(save_dir)},
            }
        ),
    )

    bundle = audit_manifest(manifest, artifact_root=tmp_path)

    assert bundle.decision.status == "refused"
    assert any("outside artifact root" in reason for reason in bundle.decision.hard_failures)


def test_release_script_fail_on_threshold(tmp_path: Path) -> None:
    manifest, paths = _fixture(tmp_path, complete=False)
    command = [
        sys.executable,
        "scripts/audit_release_manifest.py",
        str(manifest),
        "--artifact-root",
        str(tmp_path),
        "--governance-registry",
        str(paths["governance"]),
        "--export-matrix",
        str(paths["export_matrix"]),
        "--output",
        str(tmp_path / "threshold-bundle.json"),
        "--fail-on",
        "refused",
    ]
    refused_threshold = subprocess.run(command, check=False, capture_output=True, text=True)
    assert refused_threshold.returncode == 0
    assert json.loads(refused_threshold.stdout)["decision"]["status"] == "experimental"

    experimental_threshold = subprocess.run(
        [*command[:-1], "experimental"], check=False, capture_output=True, text=True
    )
    assert experimental_threshold.returncode == 1
