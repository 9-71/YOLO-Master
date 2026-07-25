# Release-Audit Compatibility and CI Preflight

## Goal

Make `yolo.release.audit` useful against historical Agent manifests while preserving strict publication semantics and
providing a script suitable for CI/manual release review.

## Steps

1. Add compatibility metadata and explicit legacy reference resolution to `agent/runtime/release.py`.
2. Add a thin `scripts/audit_release_manifest.py` wrapper with thresholded exit codes.
3. Add tests for legacy diagnostics, explicit environment model references, save-directory relative paths, path escape,
   malformed schema, and CLI threshold behavior.
4. Document the migration command in Agent README/SKILL and add a manual GitHub workflow that accepts a manifest path,
   writes a bundle, uploads it, and fails according to the requested threshold.
5. Run real historical audits, focused tests, Agent quick suite, changed-file quality, YAML/JSON parsing, and diff check.

## Acceptance Criteria

- A legacy manifest returns `compatibility.kind=legacy_unversioned`, never `publishable`, and includes migration actions.
- Explicit `environment.references.model.resolved` is hashed as `model_yaml` when present.
- Explicit relative artifact references resolve against the manifest's `job.save_dir` only when that directory is under a
  permitted root.
- `audit_release_manifest.py --fail-on refused` exits non-zero for refused and zero for experimental; `--fail-on
  experimental` exits non-zero for both experimental and refused.
- No directory globbing, model construction, network access, or source-artifact mutation is introduced.
