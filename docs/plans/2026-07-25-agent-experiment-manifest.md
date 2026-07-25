# Agent Experiment Manifest Implementation Plan

**Goal:** Connect mixture profile discovery to Agent execution and make every Agent experiment manifest complete,
versioned, integrity-checkable, and safe to retain.

**Architecture:** Reuse the existing Agent JSON request as the experiment recipe. Resolve `inputs.profile` through the
read-only mixture catalog, then serialize a redacted request snapshot and all non-duplicated response evidence into an
atomic, versioned `skill_manifest.json`.

## Step 1: Lock request and manifest contracts with tests

- Test exact profile resolution, task propagation, SHA-256 provenance, unknown IDs, and model/profile conflicts.
- Test backward-compatible fields, pipeline stage retention, best-checkpoint retention, nested and inline secret
  redaction, log truncation, checksum verification, and atomic output.

Verification: `pytest tests/test_agent_experiment_manifest.py -v`

## Step 2: Implement profile-aware request normalization

- Add `inputs.profile` to the Agent metadata schema.
- Resolve the profile before generic path normalization and remove it from downstream YOLO CLI inputs.
- Attach the catalog record, absolute YAML path, and file checksum to the normalized request.

Verification: profile normalization tests plus a dispatcher pipeline dry run.

## Step 3: Implement the versioned manifest writer

- Preserve the existing top-level compatibility fields.
- Add redacted `request` and additive `result` snapshots.
- Add bounded logs, timestamp, schema version, semantic SHA-256, and atomic replacement.

Verification: manifest unit tests and existing contract cases.

## Step 4: Document and validate the real Agent workflow

- Add a profile-driven pipeline example and describe manifest contents and secret behavior.
- Add an AutoTrain dry-run case that uses a packaged MoT profile ID.
- Run the Agent quick suite, focused unit tests, changed-file quality gate, and whitespace checks.

Verification:

```bash
pytest tests/test_agent_experiment_manifest.py -v
python agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only
python scripts/check_changed_quality.py
git diff --check
```
