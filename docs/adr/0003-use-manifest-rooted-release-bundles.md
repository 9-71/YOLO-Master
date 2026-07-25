# ADR-0003: Use Manifest-Rooted Release Bundles

## Status

Accepted

## Context

Phase 6 requires every model release candidate to be traceable from its base checkpoint through configuration,
planning, routing, adapter/pruning, export, benchmark, governance, and artifact checksums. The repository already
emits versioned `skill_manifest.json` files, fully resolved training `args.yaml`, `PlacementPlan` records, MoLoRA
runtime metadata, MoE prune manifests, benchmark `results.json`, and governance/export YAML. These documents have
different owners and are not guaranteed to exist for every experiment.

The release gate must therefore distinguish missing evidence from invalid evidence, avoid copying large artifacts,
and remain safe when a run directory contains stale or forged files. It must also join dynamic mixture profile
discovery with manually reviewed governance without creating a second synchronized model catalog.

Functional requirements:

- collect explicit evidence references from one versioned Agent manifest;
- resolve paths safely within the repository or an explicitly declared artifact root;
- record SHA-256, byte size, and kind for every included file;
- validate manifest checksum, referenced file checksums, JSON/YAML schemas, and model/profile identity;
- return deterministic `publishable`, `experimental`, or `refused` decisions with machine-readable reasons;
- emit one portable release bundle JSON without duplicating checkpoints or secrets.

Non-functional requirements:

- read-only collection with no model construction, weight loading, network access, or mutation of source artifacts;
- Python >= 3.8 and existing standard-library/YAML dependencies only;
- bounded runtime for ordinary local artifacts and deterministic serialization apart from timestamps and run identity;
- explicit operational guidance for missing optional evidence and hard integrity failures.

## Decision

Implement a typed release bundle collector rooted at a caller-selected `skill_manifest.json`. The collector follows
only explicit paths from the manifest, its normalized request, and its handler result. It may resolve the following
known evidence kinds when present: checkpoint, model YAML, resolved `args.yaml`, `PlacementPlan`, routing summary,
merge metadata, prune manifest, export capability evidence, benchmark result, and governance registry entry.

Each evidence record contains `kind`, absolute normalized `path`, `sha256`, `size_bytes`, `status`, and an optional
reason. Missing optional records are represented with `status=missing`; malformed files, checksum mismatches,
manifest checksum failures, path escapes, model/profile conflicts, and explicit failed source stages are hard errors.

The collector writes a schema-versioned `release_bundle.json` atomically beside the source manifest. It records the
source manifest checksum, base checkpoint fingerprint, model/profile identity, evidence records, governance status,
and a decision. `publishable` requires all required evidence, valid integrity, a successful source run, and governance
status `stable`. A complete but experimental/insufficiently governed candidate is `experimental`; any integrity or
source failure is `refused`.

Governance remains in `docs/governance/model-registry.yaml`, dynamic profile metadata remains in the mixture catalog,
and the collector joins them by normalized model path/profile ID. The bundle is an audit projection, not a new source
of truth. Large checkpoints remain referenced by path and checksum only.

## Consequences

### Positive

- A release review has one portable, checksum-addressed index without copying model weights.
- Missing evidence is actionable rather than silently interpreted as success.
- Existing Agent manifests and artifact formats remain authoritative and evolve independently.
- Stable versus experimental maturity is enforced by governance instead of by filename conventions.
- Read-only validation reduces the risk that audits execute arbitrary model code or overwrite artifacts.

### Negative

- Producers must expose explicit artifact paths; an unstructured historical run may be experimental or refused.
- Governance metadata and export capability still require manual review and updates.
- A bundle can become stale when referenced files change; consumers must re-run the audit or compare checksums.

### Neutral

- A release bundle does not make an unverified export format publishable; it records the capability evidence and its
  current status.
- Artifact checksums add local I/O proportional to referenced file size, including checkpoints.

## Failure Modes and Mitigations

- **Manifest tampering:** recompute and compare `manifest_sha256`; refuse on mismatch.
- **Artifact replacement:** hash every referenced file and retain the digest in the bundle; consumers can re-audit.
- **Path traversal or symlink escape:** resolve paths and require them to stay under the repository/artifact roots.
- **Malformed JSON/YAML or unsupported schema:** record a typed invalid reason and refuse when required.
- **Missing optional diagnostics:** record `missing` and classify as experimental rather than inventing evidence.
- **Stale profile metadata:** compare profile YAML path and SHA-256 against normalized request provenance.
- **Failed training/export/prune stage:** propagate source status and refuse publication.
- **Secrets in source manifests:** consume already-redacted manifest fields and never read environment credentials.

## Alternatives Considered

### Post-hoc run-directory scanner

Rejected as the primary path because it cannot reliably recover stage order, source request, fallback decisions, or
explicit artifact intent. It may be added later as an opt-in migration helper that always produces `experimental`.

### Separate static release registry

Rejected because it would duplicate model identity and evidence already maintained by Agent manifests and governance.
The bundle is generated evidence, while governance remains the reviewed registry.

### Copy all artifacts into a release archive

Rejected for the first version because copying checkpoints is expensive and increases storage/security exposure.
The bundle records portable paths and checksums; archive packaging can consume it later.

## Security and Operations

The collector is read-only, uses safe JSON/YAML parsing, rejects path escapes, and does not import or instantiate
models. It should run in CI with a repository-local artifact root and in release tooling with an explicit immutable
artifact directory. Bundles should be retained with the source manifest and invalidated whenever a referenced checksum
changes.

## References

- `docs/plans/2026-07-23-yolo-master-next-optimization.md` Phase 6
- `agent/runtime/cli/contract.py`
- `ultralytics/vpeft/placement_plan.py`
- `benchmarks/suite.py`
- `docs/governance/model-registry.yaml`
- `ultralytics/cfg/export-capability-matrix.yaml`
