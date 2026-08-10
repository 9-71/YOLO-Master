# ADR-0001: Use a Dynamic Mixture Profile Catalog

## Status

Accepted

## Context

YOLO-Master ships 405 model YAML files, including 332 profiles that use registered MoE, MoA, MoT, or Latent MoE
modules. The runtime registry in `ultralytics/nn/mixture_registry.py` resolves module classes, while
`docs/governance/model-registry.yaml` records maturity and verification evidence for a deliberately small subset.
Neither gives users or automation a queryable inventory of runnable mixture profiles.

The catalog must:

- discover packaged profiles by mixture kind, task, family, scale, module, and path;
- expose deterministic Python and CLI results without constructing models or downloading weights;
- keep model YAML files as the runnable configuration source of truth;
- keep governance status separate from profile discovery;
- support Python 3.8 and avoid adding dependencies;
- report malformed YAML, invalid layer records, duplicate identifiers, and unsafe paths clearly;
- complete a full packaged scan in under two seconds on a development workstation.

## Decision

Build a read-only catalog dynamically from `ultralytics/cfg/models/**/*.yaml`.

A lightweight metadata module defines the authoritative mapping from registered module names to the canonical
mixture kinds `moe`, `moa`, `mot`, and `latent`. Both the runtime registry and catalog consume this mapping. The
catalog parses YAML with the existing safe loader, inspects only `backbone` and `head` layer declarations, and returns
immutable records. It does not import model classes, instantiate networks, read checkpoints, or access the network.

The stable profile identifier is the POSIX path relative to the catalog root without its `.yaml` or `.yml` suffix.
The family is derived from the directory layout, such as `26`, `master/v0_10`, or `master/exp`. Task is inferred from
an explicit `task` key when present, otherwise from the final head module. Unknown heads are represented as
`unknown`, rather than silently claimed as detection models.

`yolo mixtures` exposes the same catalog with exact, case-insensitive `kind`, `task`, and `family` filters. Human
output is a deterministic table; `format=json` provides a stable machine-readable representation. The command scans
only the packaged root and accepts no arbitrary filesystem root.

Discovery is intentionally uncached in v1. A 405-file scan is cheap for an explicit command, and uncached reads avoid
stale results while contributors edit YAML files. Caching may be added later if measured latency exceeds the target.

## Consequences

### Positive

- New or removed YAML profiles appear automatically without updating a second manifest.
- Users can discover complete runnable configurations through Python or CLI filters.
- Catalog listing remains side-effect free and independent of PyTorch model construction.
- Module classification cannot drift independently from runtime registration.
- Governance evidence retains its stricter, manually reviewed role.

### Negative

- Each catalog call reads all candidate YAML files.
- Directory conventions become part of family naming and profile identifier stability.
- A malformed candidate file fails a strict scan, so one broken profile can block catalog output until fixed.

### Neutral

- Experimental profiles are discoverable but are not presented as stable; maturity stays in the governance registry.
- Joining discovery data with governance evidence remains a separate future capability.

## Failure Modes and Mitigations

- **Malformed YAML or layer shapes:** raise a path-specific `MixtureCatalogError`; tests cover syntax and structural
  failures.
- **Duplicate identifiers:** reject duplicate `.yaml`/`.yml` stems instead of selecting one nondeterministically.
- **Symlink escape:** resolve every candidate and reject files outside the selected catalog root.
- **Unknown task head:** retain the profile with `task=unknown`, making uncertainty explicit.
- **Unregistered routed module:** it is not classified as a mixture module; registry parity tests make intentional
  additions require metadata updates.
- **Output drift:** sort profiles by identifier and serialize fixed fields in a fixed order.

## Security and Operations

The existing safe YAML loader prevents arbitrary Python object construction. The public API may inspect an explicitly
provided local root for tests and tooling, but it validates all resolved files remain inside that root. The CLI never
accepts a root override, performs no writes, starts no training, loads no weights, and makes no network requests.

Operationally, the catalog requires no service, database, cache invalidation, or generated artifact. Focused tests and
the changed-file quality gate cover schema behavior and CLI integration.

## Alternatives Considered

### Maintain a static profile manifest

This could store richer descriptions and maturity labels, but duplicating hundreds of paths would create immediate
drift and contributor overhead. The existing governance registry remains the appropriate static evidence manifest.

### Extend only the runtime module registry

This would expose supported classes but not complete, runnable model profiles, tasks, scales, or configuration paths.
It does not solve configuration discovery.

### Generate and commit a catalog artifact

Generation could make reads faster, but it adds regeneration tooling and review noise for negligible runtime savings.
It can be revisited if measured scan time exceeds the non-functional target.

## References

- `ultralytics/nn/mixture_registry.py`
- `ultralytics/cfg/models/`
- `docs/governance/model-registry.yaml`
- `docs/plans/2026-07-23-yolo-master-next-optimization.md`
