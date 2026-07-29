# Release Bundle and Readiness Gate

## Goal

Create a read-only, manifest-rooted release audit that turns existing YOLO-Master experiment evidence into a
checksum-addressed `release_bundle.json` and a deterministic `publishable` / `experimental` / `refused` decision.

## Requirements

### Functional

1. Accept a versioned Agent `skill_manifest.json` and optional artifact root.
2. Verify the manifest semantic checksum before consuming it.
3. Collect explicit references for checkpoint, model YAML, resolved args, planner, routing, merge/prune, export,
   benchmark, and governance evidence.
4. Hash existing files and retain size, kind, status, and reason for each record.
5. Join dynamic profile provenance with governance by normalized model path/profile ID.
6. Refuse integrity, path-safety, schema, source-stage, and identity failures; classify missing/experimental evidence
   as `experimental`.
7. Write an atomic, schema-versioned release bundle and expose it through `yolo.release.audit`.

### Non-functional

- Python >= 3.8; no new runtime dependency.
- No model construction, weight loading, network request, or mutation of source artifacts.
- Deterministic JSON ordering and bounded diagnostic messages.
- Focused tests complete quickly and cover success, missing evidence, tampering, path escape, schema errors, and
  governance joins.

## Design

`agent/runtime/release.py` owns immutable evidence records, path validation, checksum calculation, source manifest
loading, known-schema parsing, governance/profile joins, decision rules, and atomic output. The dispatcher exposes a
thin `yolo.release.audit` handler that normalizes `inputs.manifest` / `params.manifest`, resolves an output path, and
returns the bundle plus decision in the standard response envelope.

The source manifest is authoritative for request/result references. Evidence discovery is explicit and conservative:
known artifact entries are inspected first, then well-defined fields such as `best_checkpoint`, `profile.model_path`,
`result.pipeline`, `result.stages`, and `result.*.artifacts` are traversed. No glob scan is performed.

The bundle schema contains:

```json
{
  "schema_version": 1,
  "source_manifest": {"path": "...", "sha256": "..."},
  "identity": {"model": "...", "profile_id": null, "checkpoint": {"path": "...", "sha256": "..."}},
  "evidence": [{"kind": "model_yaml", "path": "...", "sha256": "...", "size_bytes": 1, "status": "valid"}],
  "governance": {"status": "experimental", "matched": true},
  "decision": {"status": "experimental", "reasons": [], "hard_failures": [], "missing": []}
}
```

## Implementation Steps

1. Add `agent/runtime/release.py` with `ReleaseBundle`, `EvidenceRecord`, `audit_manifest`, and `write_release_bundle`.
2. Add unit tests using temporary JSON/YAML/artifact fixtures; do not depend on existing runs or checkpoints.
3. Add dispatcher handler and register `yolo.release.audit`; add a contract case for dry-run/real fixture audit.
4. Update Agent README, SKILL, metadata, and Phase 6 governance documentation with CLI examples and decision rules.
5. Run focused tests, Agent quick suite, Ruff/codespell, changed-file quality gate, JSON/YAML parse, and diff check.

## Acceptance Criteria

- Valid stable fixture produces `publishable` and an atomic bundle with correct checksums.
- Valid experimental fixture produces `experimental` with explicit governance/missing reasons.
- Tampered manifest or artifact, malformed schema, escaped path, failed source stage, and profile mismatch produce
  `refused` without writing a misleading publishable bundle.
- Existing Agent quick suite remains green and no unrelated worktree changes are reverted.
