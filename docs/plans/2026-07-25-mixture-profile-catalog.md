# Mixture Profile Catalog Implementation Plan

**Goal:** Make packaged MoE, MoA, MoT, and Latent MoE model configurations discoverable without loading a model.

**Architecture:** Treat model YAML files as the runnable source of truth. Share a lightweight module-kind mapping with
the runtime registry, parse model layer records into immutable profile metadata, and expose one query through Python
and `yolo mixtures`.

## Step 1: Lock the catalog contract with tests

- Add temporary-root tests for each mixture kind, hybrid profiles, task and family inference, scales, deterministic
  ordering, exact filters, duplicate identifiers, malformed YAML, invalid layers, and symlink escape.
- Add a parity test between lightweight module metadata and the runtime module registry.

Verification: `pytest tests/test_mixture_catalog.py -v`

## Step 2: Implement lightweight metadata and catalog discovery

- Add an immutable `MixtureProfile` value object and `MixtureCatalogError`.
- Add `discover_mixture_profiles`, `list_mixture_profiles`, and `get_mixture_profile`.
- Parse YAML only; do not import or construct model implementations in the catalog module.

Verification: `pytest tests/test_mixture_catalog.py tests/test_mixture_model_registry.py -v`

## Step 3: Add the CLI surface

- Add `yolo mixtures` with `kind`, `task`, `family`, and `format=table|json` options.
- Keep JSON deterministic and table output concise; report invalid filters before any model execution path.
- Add help text and direct entrypoint tests that monkeypatch catalog discovery.

Verification: `pytest tests/test_mixture_catalog.py tests/test_mixture_catalog_cli.py -v`

## Step 4: Verify packaged data and quality

- Assert representative packaged MoE, MoA, MoT, Latent, and hybrid profiles are discoverable.
- Measure a complete catalog scan against the two-second development target.
- Run focused model registry and configuration tests, changed-file lint/format/spelling checks, and `git diff --check`.

Verification:

```bash
pytest tests/test_mixture_catalog.py tests/test_mixture_catalog_cli.py tests/test_mixture_model_registry.py -v
python scripts/check_changed_quality.py
git diff --check
```
