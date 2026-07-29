"""Agent handler for read-only release bundle audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..release import audit_manifest, write_release_bundle

from .contract import plan_response, response
from .normalize import is_dry_run, resolved_path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOVERNANCE_REGISTRY = REPO_ROOT / "docs" / "governance" / "model-registry.yaml"
DEFAULT_EXPORT_MATRIX = REPO_ROOT / "ultralytics" / "cfg" / "export-capability-matrix.yaml"


def run_release_audit(request: dict[str, Any]) -> dict[str, Any]:
    """Audit a versioned Agent manifest and optionally write its bundle."""
    inputs = request.get("inputs", {})
    params = request.get("params", {})
    manifest = inputs.get("manifest") or params.get("manifest")
    if not manifest:
        raise ValueError("`inputs.manifest` or `params.manifest` is required.")
    manifest_path = resolved_path(str(manifest))
    artifact_root = params.get("artifact_root")
    governance = params.get("governance_registry", str(DEFAULT_GOVERNANCE_REGISTRY))
    export_matrix = params.get("export_matrix", str(DEFAULT_EXPORT_MATRIX))
    audit_params = {
        "governance_registry": str(resolved_path(str(governance))),
        "export_matrix": str(resolved_path(str(export_matrix))),
    }
    if is_dry_run(request):
        return plan_response(
            request,
            "release audit dry run prepared",
            "module",
            "audit_manifest",
            params={
                "manifest": str(manifest_path),
                "artifact_root": str(resolved_path(str(artifact_root))) if artifact_root else None,
                **audit_params,
            },
            next_actions=["run with policy.dry_run=false to write release_bundle.json"],
        )

    bundle = audit_manifest(
        manifest_path,
        params=audit_params,
        artifact_root=resolved_path(str(artifact_root)) if artifact_root else None,
    )
    output = params.get("output") or str(manifest_path.parent / "release_bundle.json")
    output_path = write_release_bundle(bundle, resolved_path(str(output)))
    return response(
        request["skill"],
        "ok" if bundle.decision.status != "refused" else "partial",
        f"release audit completed: {bundle.decision.status}",
        release=bundle.to_dict(),
        decision=bundle.decision.to_dict(),
        artifacts=[{"kind": "release_bundle", "path": str(output_path)}],
        next_actions=[]
        if bundle.decision.status == "publishable"
        else ["resolve release decision reasons before publishing"],
    )


__all__ = ["run_release_audit"]
