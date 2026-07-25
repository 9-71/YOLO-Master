# ADR-0004: Add Legacy Release-Audit Compatibility Diagnostics

## Status

Accepted

## Context

The repository contains two generations of Agent manifests. Current manifests are schema-versioned and carry a
semantic `manifest_sha256`, normalized request, and additive result fields. Older manifests contain the same useful
artifact list plus explicit environment model references and job save directories, but do not contain the versioned
contract. Auditing these files currently emits generic missing-path errors alongside checksum failures, making the
migration action unclear.

Phase 6 needs a practical release preflight for real historical runs without weakening the release gate. The preflight
must remain manifest-rooted: it may use only explicit fields already present in the legacy document, and must never
glob a run directory or infer an artifact from a filename convention.

## Decision

Extend the release bundle with a `compatibility` block. It reports `versioned`, `legacy_unversioned`, or `invalid`, the
source schema version, whether the source checksum is verifiable, and bounded migration actions. A legacy source is
always at least `experimental` and cannot be `publishable`; missing checksum/schema are hard failures for release.

For legacy manifests only, the collector may resolve:

- `environment.references.model.resolved` as the model YAML reference;
- `job.save_dir` as a safe base for explicit relative artifact paths;
- existing `artifacts[*].path` entries exactly as recorded.

These are explicit manifest fields, not directory discovery. The audit still hashes every referenced file and rejects
path escapes, malformed evidence, failed source status, and checksum/schema failures.

Add `scripts/audit_release_manifest.py` as a thin command wrapper. It accepts a manifest, optional artifact root,
governance registry, export matrix, output path, and `--fail-on {refused,experimental}`. The wrapper prints the full
JSON decision and returns a non-zero code only when the selected threshold is reached, making it usable in CI and local
release review. It does not create a model, download weights, or modify source artifacts except for the requested
bundle output.

## Consequences

### Positive

- Historical runs produce actionable migration diagnostics instead of opaque path errors.
- The integrity gate remains strict: legacy manifests cannot silently become publishable.
- CI and release managers get a stable, dependency-light command surface.
- Explicit model/save-dir provenance improves artifact completeness without a post-hoc scanner.

### Negative

- Legacy audits may hash large checkpoints and still end in `refused` because they lack a valid source checksum.
- The compatibility branch adds a second manifest interpretation that must remain frozen and tested.

### Neutral

- New manifests continue using the versioned path unchanged.
- A legacy bundle is useful as a migration report, not as a release approval.

## Failure Modes and Mitigations

- **Legacy path points outside roots:** reject it as path escape; do not fall back to basename matching.
- **Legacy job directory is stale:** only explicit artifact paths are consumed; missing files remain missing.
- **Schema value is malformed:** classify source as `invalid` and return structured refusal, never raise from the CLI.
- **Threshold misuse in CI:** validate `--fail-on` choices and document the exit code semantics.

## Alternatives Considered

### Reject all legacy manifests without diagnostics

This is safe but provides no migration value and makes historical experiments difficult to recover.

### Scan the run directory for args/checkpoints/results

Rejected because stale files and naming collisions can create false evidence. Explicit manifest references remain the
only accepted source.

### Auto-rewrite legacy manifests

Rejected because rewriting historical records changes provenance. The compatibility bundle is a read-only projection;
new experiments must emit the current schema.

## References

- `agent/runtime/release.py`
- `scripts/audit_release_manifest.py`
- `docs/adr/0003-use-manifest-rooted-release-bundles.md`
- `docs/plans/2026-07-23-yolo-master-next-optimization.md` Phase 6
