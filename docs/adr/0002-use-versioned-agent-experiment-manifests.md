# ADR-0002: Use Versioned Agent Experiment Manifests

## Status

Accepted

## Context

YOLO-Master exposes complete experiment orchestration through `yolo.pipeline.experiment`. A request already describes
model inputs, stage parameters, runtime policy, artifact locations, and the ordered train/val/export/benchmark flow.
The pipeline response also contains stage payloads, the current or best checkpoint, progress metadata, and artifacts.

The current `skill_manifest.json` writer retains only a fixed subset of response fields. In particular, it drops
`plan`, `pipeline`, `stages`, `best_checkpoint`, `results`, and other handler-specific evidence. A successful pipeline
therefore cannot be reconstructed or audited from its manifest. The newly added mixture profile catalog also remains
disconnected from Agent requests: users must copy a YAML path instead of selecting a stable profile identifier.

The next platform increment must:

- connect a catalog `profile_id` directly to every model-consuming Agent skill;
- record the resolved profile path, task, mixture metadata, and model YAML SHA-256;
- preserve normalized inputs, parameters, stage results, selected checkpoint, diagnostics, and execution evidence;
- keep existing manifest fields compatible for current consumers;
- prevent API keys, tokens, passwords, secrets, and authorization values from reaching disk;
- bound raw stdout/stderr growth and write the manifest atomically;
- avoid model construction or weight downloads during profile resolution and pipeline dry runs;
- remain deterministic except for explicit run identity and timestamps.

## Decision

Extend the existing Agent request and manifest contracts instead of adding a second recipe language.

`inputs.profile` accepts an exact mixture catalog identifier such as `26/yolo26-master-mot-n`. During request
normalization, the Agent resolves it through `ultralytics.cfg.mixture_catalog`, removes the non-YOLO `profile` input,
sets `inputs.model` to the packaged YAML path, validates any explicit task, and attaches immutable profile provenance
to the normalized request. `inputs.model` and `inputs.profile` are mutually exclusive so execution is unambiguous.

The profile provenance contains the stable identifier, catalog-relative path, task, family, scales, mixture kinds,
mixture modules, resolved model path, and SHA-256 of the YAML bytes. Resolution is read-only and does not instantiate a
network.

`skill_manifest.json` gains `schema_version`, `created_at`, `request`, `result`, and `manifest_sha256` fields. Existing
top-level fields such as `skill`, `status`, `artifacts`, `metrics`, `environment`, `usage`, and `progress` remain. The
`request` snapshot stores normalized inputs, parameters, runtime, artifact, policy, and profile provenance. The
`result` object stores every response field not already represented by the compatibility fields, including pipeline
stage payloads and best-checkpoint selection.

Before serialization, manifest content is converted to JSON-safe values and recursively redacted by key and by common
inline credential patterns. Raw stdout and stderr values are capped at 20,000 characters each, retaining their tail.
The semantic manifest checksum is SHA-256 over canonical JSON before the checksum field is added. The final JSON is
written to a unique temporary file in the target directory and atomically replaced.

## Consequences

### Positive

- A stable profile identifier can drive train, validation, export, benchmark, and full pipeline requests.
- Pipeline manifests become sufficient to inspect stage order, stage outcomes, chosen checkpoint, and requested
  configuration without consulting ephemeral stdout.
- Profile provenance binds an experiment to exact YAML content even if the same path changes later.
- Existing manifest consumers continue to find their current top-level fields.
- Credential redaction and bounded logs make richer manifests safe enough for routine artifact collection.

### Negative

- Manifests become larger because they preserve handler-specific result evidence.
- Request normalization incurs one catalog scan and one file hash when `inputs.profile` is used.
- The manifest records normalized requested parameters; framework defaults fully resolved during training remain in
  the emitted `args.yaml` artifact rather than being duplicated into the Agent request snapshot.

### Neutral

- Direct `inputs.model` requests continue to work and have no synthetic catalog provenance.
- Experiment recipes remain ordinary JSON requests, so existing dispatcher and validator tooling stays authoritative.

## Failure Modes and Mitigations

- **Unknown or malformed profile ID:** fail normalization before a model or subprocess is created.
- **Both model and profile supplied:** reject the ambiguous request rather than choosing precedence.
- **Task conflicts with catalog metadata:** reject before execution.
- **Profile YAML changes:** the stored SHA-256 exposes the difference even when the profile ID is unchanged.
- **Credential embedded in nested configuration or CLI text:** recursive key redaction and inline assignment/Bearer
  redaction replace values with `<redacted>`.
- **Large training logs:** stdout/stderr are tail-capped with an explicit truncation marker.
- **Interrupted write:** atomic replacement leaves either the prior complete manifest or the new complete manifest.
- **New response fields:** the additive `result` projection preserves them without another allow-list update.

## Security and Operations

The Agent never accepts a filesystem path through `profile`; it resolves exact identifiers already constrained by the
catalog root. Direct `model` inputs retain their existing behavior. Secret redaction is defense in depth, not a reason
to pass credentials in request JSON; environment variables remain preferred.

No database, background service, generated registry, or migration job is introduced. Manifest schema evolution stays
additive and versioned. Validator dry runs and focused unit tests cover profile resolution, secret removal, stage
retention, checksum verification, log bounds, and atomic output.

## Alternatives Considered

### Add a separate experiment recipe YAML DSL

A dedicated DSL could offer polished presets, but it would duplicate the existing `inputs`, `params`, `stages`,
`runtime`, and `policy` structures and require another parser and compatibility lifecycle.

### Build release manifests by scanning run directories after execution

Post-hoc scanning can find checkpoints and `args.yaml`, but cannot reliably recover the original stage order, fallback
decisions, failed stages, diagnostics, or Agent request. It also cannot safely recover credentials after they have
already leaked into logs.

### Store every response verbatim

This is simple but permits unbounded logs and credential persistence. The selected design preserves all semantic fields
while applying redaction and targeted log compaction.

## References

- `agent/runtime/cli/pipeline.py`
- `agent/runtime/cli/contract.py`
- `agent/runtime/cli/normalize.py`
- `ultralytics/cfg/mixture_catalog.py`
- `docs/plans/2026-07-23-yolo-master-next-optimization.md`
