"""Read-only release bundle collection and readiness decisions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cli.contract import manifest_checksum


RELEASE_BUNDLE_SCHEMA_VERSION = 1
_REQUIRED_KINDS = {
    "checkpoint",
    "model_yaml",
    "resolved_args",
    "routing_summary",
    "benchmark_result",
    "export_capability",
}
_CONDITIONAL_KINDS = {"placement_plan", "merge_manifest", "prune_manifest"}
_KIND_ALIASES = {
    "checkpoint": "checkpoint",
    "weights": "checkpoint",
    "model": "model_yaml",
    "model_yaml": "model_yaml",
    "yaml": "model_yaml",
    "config": "resolved_args",
    "args": "resolved_args",
    "args_yaml": "resolved_args",
    "resolved_args": "resolved_args",
    "placement": "placement_plan",
    "placement_plan": "placement_plan",
    "planner": "placement_plan",
    "routing": "routing_summary",
    "routing_summary": "routing_summary",
    "merge": "merge_manifest",
    "merge_manifest": "merge_manifest",
    "prune": "prune_manifest",
    "prune_manifest": "prune_manifest",
    "export": "export_capability",
    "export_capability": "export_capability",
    "benchmark": "benchmark_result",
    "benchmark_result": "benchmark_result",
    "results": "benchmark_result",
    "governance_registry": "governance_registry",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml_load(path: Path) -> Any:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency is part of the repository runtime
        raise ValueError(f"PyYAML is required to parse {path}") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _path_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _iter_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _iter_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_mappings(child)


def _safe_roots(manifest_path: Path, artifact_root: str | Path | None) -> tuple[Path, ...]:
    repository = Path(__file__).resolve().parents[2]
    if artifact_root is not None:
        return (Path(artifact_root).expanduser().resolve(), repository)
    return (repository, manifest_path.parent.resolve())


def _resolve_path(
    raw: Any,
    *,
    manifest_path: Path,
    roots: tuple[Path, ...],
    relative_bases: tuple[Path, ...] = (),
) -> tuple[Path | None, str | None]:
    value = _path_text(raw)
    if not value:
        return None, "path is empty"
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidates = [base / candidate for base in relative_bases]
        candidates.extend([manifest_path.parent / candidate, *[root / candidate for root in roots]])
        candidate = next((item for item in candidates if item.exists()), candidates[0])
    resolved = candidate.resolve()
    if not any(resolved == root or root in resolved.parents for root in roots):
        return None, f"path outside artifact root: {value}"
    return resolved, None


def _compatibility_diagnostic(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Classify the source manifest and describe the bounded migration path."""
    has_schema = "schema_version" in manifest
    raw_schema = manifest.get("schema_version")
    try:
        schema = int(raw_schema) if has_schema else None
    except (TypeError, ValueError):
        schema = None
    expected = manifest.get("manifest_sha256")
    checksum_valid = bool(expected) and str(expected) == manifest_checksum(dict(manifest))
    if not has_schema:
        kind = "legacy_unversioned"
        actions = [
            "emit a new schema_version=1 manifest before publishing",
            "preserve this bundle as a migration report only",
        ]
    elif schema != 1 or not expected or not checksum_valid:
        kind = "invalid"
        actions = ["repair the source manifest schema/checksum and rerun the audit"]
    else:
        kind = "versioned"
        actions = []
    return {
        "kind": kind,
        "source_schema_version": schema,
        "checksum": {"present": bool(expected), "valid": checksum_valid},
        "migration_actions": actions,
    }


@dataclass(frozen=True)
class EvidenceRecord:
    """One checksum-addressed release evidence item."""

    kind: str
    path: str | None
    sha256: str | None
    size_bytes: int | None
    status: str
    required: bool = False
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseDecision:
    """Deterministic readiness result."""

    status: str
    reasons: tuple[str, ...] = ()
    hard_failures: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "hard_failures": list(self.hard_failures),
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class ReleaseBundle:
    """Serializable release audit projection."""

    source_manifest: dict[str, Any]
    identity: dict[str, Any]
    evidence: tuple[EvidenceRecord, ...]
    governance: dict[str, Any]
    decision: ReleaseDecision
    export: dict[str, Any] = field(default_factory=dict)
    compatibility: dict[str, Any] = field(default_factory=dict)
    schema_version: int = RELEASE_BUNDLE_SCHEMA_VERSION
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "source_manifest": dict(self.source_manifest),
            "identity": dict(self.identity),
            "evidence": [item.to_dict() for item in self.evidence],
            "governance": dict(self.governance),
            "compatibility": dict(self.compatibility),
            "export": dict(self.export),
            "decision": self.decision.to_dict(),
        }


def _artifact_entries(manifest: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    entries: list[tuple[str, Mapping[str, Any]]] = []
    for item in manifest.get("artifacts", []) if isinstance(manifest.get("artifacts"), list) else []:
        if isinstance(item, Mapping) and item.get("path"):
            entries.append((str(item.get("kind", "artifact")), item))
    # Preserve handler artifacts nested in result/stages without relying on directory names.
    result = manifest.get("result", {})
    for item in _iter_mappings(result):
        if item is result:
            continue
        if item.get("path") and (item.get("kind") or item.get("label")):
            entries.append((str(item.get("kind", item.get("label", "artifact"))), item))
    return entries


def _record_file(
    kind: str,
    raw: Any,
    *,
    manifest_path: Path,
    roots: tuple[Path, ...],
    relative_bases: tuple[Path, ...] = (),
    required: bool,
    declared_sha256: str | None = None,
) -> tuple[EvidenceRecord, str | None]:
    path, error = _resolve_path(raw, manifest_path=manifest_path, roots=roots, relative_bases=relative_bases)
    if error:
        return EvidenceRecord(kind, None, None, None, "invalid", required, error), error
    assert path is not None
    if not path.exists() or not path.is_file():
        reason = f"missing evidence file: {path}"
        return EvidenceRecord(kind, str(path), None, None, "missing", required, reason), reason if required else None
    digest = _sha256(path)
    if declared_sha256 and str(declared_sha256) != digest:
        reason = f"checksum mismatch for {kind}: {path}"
        return EvidenceRecord(kind, str(path), digest, path.stat().st_size, "invalid", required, reason), reason
    return EvidenceRecord(kind, str(path), digest, path.stat().st_size, "valid", required), None


def _validate_payload(kind: str, record: EvidenceRecord, hard_failures: list[str]) -> None:
    if record.status != "valid" or not record.path:
        return
    path = Path(record.path)
    try:
        if kind in {"model_yaml", "resolved_args", "export_capability", "governance_registry"}:
            value = _yaml_load(path)
            if not isinstance(value, Mapping):
                raise ValueError("expected a YAML mapping")
            if kind == "export_capability" and int(value.get("schema_version", 0)) != 1:
                raise ValueError("unsupported export capability schema_version")
        elif kind in {"placement_plan", "merge_manifest", "prune_manifest", "routing_summary", "benchmark_result"}:
            value = _json_load(path)
            if not isinstance(value, Mapping):
                raise ValueError("expected a JSON object")
            if kind == "placement_plan" and int(value.get("schema_version", 0)) != 1:
                raise ValueError("unsupported PlacementPlan schema_version")
            if kind == "benchmark_result" and int(value.get("schema_version", 0)) != 1:
                raise ValueError("unsupported benchmark schema_version")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        hard_failures.append(f"invalid {kind} evidence {path}: {exc}")


def _governance(
    model_path: str | None,
    registry_path: str | Path | None,
    *,
    manifest_path: Path,
    roots: tuple[Path, ...],
    hard_failures: list[str],
) -> dict[str, Any]:
    if not registry_path:
        return {"status": "unknown", "matched": False, "reason": "governance registry not provided"}
    registry, error = _resolve_path(registry_path, manifest_path=manifest_path, roots=roots)
    if error or registry is None or not registry.exists():
        reason = error or f"missing governance registry: {registry_path}"
        hard_failures.append(reason)
        return {"status": "invalid", "matched": False, "reason": reason}
    try:
        payload = _yaml_load(registry)
        models = payload.get("models", []) if isinstance(payload, Mapping) else []
        if not isinstance(models, list):
            raise ValueError("models must be a list")
        model_resolved = Path(model_path).expanduser().resolve() if model_path else None
        for entry in models:
            if not isinstance(entry, Mapping) or not entry.get("path"):
                continue
            candidate = Path(str(entry["path"]))
            if not candidate.is_absolute():
                candidate_options = [registry.parent / candidate, *[root / candidate for root in roots]]
            else:
                candidate_options = [candidate]
            matched_candidate = next(
                (
                    option.resolve()
                    for option in candidate_options
                    if model_resolved and option.resolve() == model_resolved
                ),
                None,
            )
            if matched_candidate is not None:
                return {
                    "status": str(entry.get("status", "unknown")),
                    "matched": True,
                    "name": entry.get("name"),
                    "path": str(matched_candidate),
                    "entry": dict(entry),
                }
        return {"status": "unknown", "matched": False, "reason": "model is not registered"}
    except Exception as exc:
        reason = f"invalid governance registry {registry}: {exc}"
        hard_failures.append(reason)
        return {"status": "invalid", "matched": False, "reason": reason}


def audit_manifest(
    manifest_path: str | Path,
    *,
    params: Mapping[str, Any] | None = None,
    artifact_root: str | Path | None = None,
) -> ReleaseBundle:
    """Audit one Agent manifest without executing or modifying experiment code."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    params = dict(params or {})
    hard_failures: list[str] = []
    reasons: list[str] = []
    missing: list[str] = []
    roots = _safe_roots(manifest_path, artifact_root)
    source_digest = _sha256(manifest_path) if manifest_path.exists() else None
    source = {"path": str(manifest_path), "sha256": source_digest, "status": "invalid"}
    if not manifest_path.exists():
        hard_failures.append(f"missing source manifest: {manifest_path}")
        return ReleaseBundle(
            source,
            {},
            (),
            {},
            ReleaseDecision("refused", (), tuple(hard_failures), ()),
            compatibility={
                "kind": "invalid",
                "source_schema_version": None,
                "checksum": {"present": False, "valid": False},
                "migration_actions": ["provide a readable source manifest"],
            },
            created_at=_utc_now(),
        )
    try:
        manifest = _json_load(manifest_path)
        if not isinstance(manifest, Mapping):
            raise ValueError("manifest must be a JSON object")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        hard_failures.append(f"invalid source manifest {manifest_path}: {exc}")
        return ReleaseBundle(
            source,
            {},
            (),
            {},
            ReleaseDecision("refused", (), tuple(hard_failures), ()),
            compatibility={
                "kind": "invalid",
                "source_schema_version": None,
                "checksum": {"present": False, "valid": False},
                "migration_actions": ["repair the source JSON object"],
            },
            created_at=_utc_now(),
        )
    source["status"] = str(manifest.get("status", "unknown"))
    try:
        schema_version = int(manifest.get("schema_version", 0))
    except (TypeError, ValueError):
        schema_version = 0
    compatibility = _compatibility_diagnostic(manifest)
    if compatibility["kind"] == "legacy_unversioned":
        hard_failures.append("legacy source manifest is unversioned; migration required before publishing")
    elif schema_version != 1:
        hard_failures.append(f"unsupported source manifest schema_version: {manifest.get('schema_version')!r}")
    expected = manifest.get("manifest_sha256")
    if not expected:
        hard_failures.append("source manifest has no verifiable checksum")
    elif str(expected) != manifest_checksum(dict(manifest)):
        hard_failures.append("source manifest checksum mismatch")
    if source["status"] != "ok":
        hard_failures.append(f"source manifest status is not ok: {source['status']}")

    request = manifest.get("request", {}) if isinstance(manifest.get("request"), Mapping) else {}
    inputs = request.get("inputs", {}) if isinstance(request.get("inputs"), Mapping) else {}
    result = manifest.get("result", {}) if isinstance(manifest.get("result"), Mapping) else {}
    model_raw = inputs.get("model")
    environment = manifest.get("environment", {}) if isinstance(manifest.get("environment"), Mapping) else {}
    references = environment.get("references", {}) if isinstance(environment.get("references"), Mapping) else {}
    model_reference = references.get("model", {}) if isinstance(references.get("model"), Mapping) else {}
    if compatibility["kind"] == "legacy_unversioned" and not model_raw:
        model_raw = model_reference.get("resolved")
    legacy_bases: tuple[Path, ...] = ()
    if compatibility["kind"] == "legacy_unversioned":
        job = manifest.get("job", {}) if isinstance(manifest.get("job"), Mapping) else {}
        save_dir_raw = job.get("save_dir")
        if save_dir_raw:
            save_dir, save_dir_error = _resolve_path(save_dir_raw, manifest_path=manifest_path, roots=roots)
            if save_dir_error:
                hard_failures.append(f"legacy job.save_dir: {save_dir_error}")
            elif save_dir is not None:
                legacy_bases = (save_dir,)
    model_path, model_error = _resolve_path(
        model_raw,
        manifest_path=manifest_path,
        roots=roots,
        relative_bases=legacy_bases,
    )
    if model_error:
        hard_failures.append(model_error)
    identity: dict[str, Any] = {"model": str(model_path) if model_path else _path_text(model_raw)}
    profile = request.get("profile") if isinstance(request.get("profile"), Mapping) else None
    if profile:
        identity["profile_id"] = profile.get("profile_id")
        profile_path = _path_text(profile.get("model_path"))
        if profile_path and model_path and Path(profile_path).resolve() != model_path:
            hard_failures.append("profile model path conflicts with request model")
        if profile_path and profile.get("sha256"):
            profile_file, profile_error = _resolve_path(profile_path, manifest_path=manifest_path, roots=roots)
            if profile_error or profile_file is None or not profile_file.exists():
                hard_failures.append(profile_error or f"missing profile YAML: {profile_path}")
            elif _sha256(profile_file) != str(profile["sha256"]):
                hard_failures.append("profile YAML checksum mismatch")

    records: list[EvidenceRecord] = []
    seen: set[str] = set()
    for raw_kind, item in _artifact_entries(manifest):
        kind = _KIND_ALIASES.get(raw_kind.casefold(), raw_kind.casefold())
        seen.add(kind)
        record, error = _record_file(
            kind,
            item.get("path"),
            manifest_path=manifest_path,
            roots=roots,
            relative_bases=legacy_bases,
            required=kind in _REQUIRED_KINDS,
            declared_sha256=item.get("sha256"),
        )
        records.append(record)
        if error and record.status == "invalid":
            hard_failures.append(error)
        elif record.status == "missing" and record.required:
            missing.append(kind)
        _validate_payload(kind, record, hard_failures)

    # Profile/model path and explicit best checkpoint are common result references.
    if model_path and "model_yaml" not in seen:
        record, error = _record_file("model_yaml", model_path, manifest_path=manifest_path, roots=roots, required=True)
        records.append(record)
        seen.add("model_yaml")
        if error:
            (hard_failures if record.status == "invalid" else missing).append(
                error if record.status == "invalid" else "model_yaml"
            )
        _validate_payload("model_yaml", record, hard_failures)
    checkpoint_raw = result.get("best_checkpoint") or inputs.get("checkpoint")
    if checkpoint_raw and "checkpoint" not in seen:
        record, error = _record_file(
            "checkpoint", checkpoint_raw, manifest_path=manifest_path, roots=roots, required=True
        )
        records.append(record)
        seen.add("checkpoint")
        if error:
            (hard_failures if record.status == "invalid" else missing).append(
                error if record.status == "invalid" else "checkpoint"
            )
    for required_kind in ("resolved_args", "routing_summary", "benchmark_result"):
        if required_kind not in seen:
            records.append(
                EvidenceRecord(
                    required_kind,
                    None,
                    None,
                    None,
                    "missing",
                    required=True,
                    reason="no explicit artifact reference in source manifest",
                )
            )
            missing.append(required_kind)

    registry = params.get("governance_registry")
    if registry:
        registry_record, registry_error = _record_file(
            "governance_registry",
            registry,
            manifest_path=manifest_path,
            roots=roots,
            required=True,
        )
        records.append(registry_record)
        if registry_error:
            (hard_failures if registry_record.status == "invalid" else missing).append(
                registry_error if registry_record.status == "invalid" else "governance_registry"
            )
        _validate_payload("governance_registry", registry_record, hard_failures)
    governance = _governance(
        identity.get("model"), registry, manifest_path=manifest_path, roots=roots, hard_failures=hard_failures
    )
    export_matrix = params.get("export_matrix")
    export_evidence: dict[str, Any] = {
        "format": result.get("export", {}).get("format") if isinstance(result.get("export"), Mapping) else None
    }
    if export_matrix:
        record, error = _record_file(
            "export_capability", export_matrix, manifest_path=manifest_path, roots=roots, required=True
        )
        records.append(record)
        if error:
            (hard_failures if record.status == "invalid" else missing).append(
                error if record.status == "invalid" else "export_capability"
            )
        _validate_payload("export_capability", record, hard_failures)
        if record.status == "valid":
            try:
                matrix = _yaml_load(Path(record.path))
                export_format = export_evidence.get("format") or "onnx"
                export_evidence.update(matrix.get("formats", {}).get(export_format, {}))
            except (OSError, TypeError, ValueError):
                pass
    else:
        missing.append("export_capability")

    record_kinds = {record.kind for record in records}
    for required_kind in _REQUIRED_KINDS:
        if required_kind not in record_kinds:
            records.append(
                EvidenceRecord(
                    required_kind,
                    None,
                    None,
                    None,
                    "missing",
                    required=True,
                    reason="no explicit artifact reference in source manifest",
                )
            )
            missing.append(required_kind)
    request_text = json.dumps(request, sort_keys=True, default=str).casefold()
    adapter_requested = any(token in request_text for token in ("lora", "molora", "peft", "adapter"))
    for conditional_kind in sorted(_CONDITIONAL_KINDS):
        if conditional_kind in record_kinds:
            continue
        applicable = adapter_requested or (conditional_kind == "prune_manifest" and "prune" in request_text)
        records.append(
            EvidenceRecord(
                conditional_kind,
                None,
                None,
                None,
                "missing" if applicable else "not_applicable",
                required=applicable,
                reason=(
                    "adapter or pruning evidence was requested"
                    if applicable
                    else "no adapter or pruning stage declared"
                ),
            )
        )
        if applicable:
            missing.append(conditional_kind)
    missing = sorted(set(missing))
    stable = governance.get("status") == "stable"
    if hard_failures:
        decision = ReleaseDecision("refused", tuple(reasons), tuple(dict.fromkeys(hard_failures)), tuple(missing))
    elif missing or not stable:
        if missing:
            reasons.append("required release evidence is missing")
        if not stable:
            reasons.append(f"governance status is not stable: {governance.get('status')}")
        decision = ReleaseDecision("experimental", tuple(reasons), (), tuple(missing))
    else:
        decision = ReleaseDecision("publishable", (), (), ())
    checkpoint = next((item for item in records if item.kind == "checkpoint" and item.status == "valid"), None)
    if checkpoint:
        identity["checkpoint"] = checkpoint.to_dict()
        identity["base_checkpoint_fingerprint"] = checkpoint.sha256
    identity["model_yaml"] = next((item.to_dict() for item in records if item.kind == "model_yaml"), None)
    return ReleaseBundle(
        source,
        identity,
        tuple(records),
        governance,
        decision,
        export=export_evidence,
        compatibility=compatibility,
        created_at=_utc_now(),
    )


def write_release_bundle(bundle: ReleaseBundle, path: str | Path) -> Path:
    """Atomically write a release bundle without changing referenced artifacts."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".release_bundle.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(bundle.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = ["EvidenceRecord", "ReleaseBundle", "ReleaseDecision", "audit_manifest", "write_release_bundle"]
