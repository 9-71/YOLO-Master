# F1 Task Dispatcher & Core Handlers — Security Test Report

| Field | Value |
| --- | --- |
| **Project Title** | YOLO-Master F1 Platform — Security Verification & Test Report |
| **Version** | 1.0 |
| **Author** | F1 Platform Security Engineering & Technical Writing |
| **Date** | 2026-09-13 |
| **Target Scope** | F1 Task Dispatcher (`f1/dispatcher.py`) & Core Handlers (`f1/handlers/`), including the credential sanitizer (`core/security.py`) and strongly-typed job contract (`core/schema.py`) |
| **Test Evidence** | `tests/f1/test_security_sanitizer.py`, `tests/f1/test_handlers_framework.py` |
| **Overall Result** | **Security-directed test result (sum of the two targeted commands in §4): 21 passed (9 sanitizer + 12 path-whitelist) · 5 skipped (Windows OS symlink privilege) · 0 failed** — this is not the full project test suite |

---

## 1. Executive Summary & Security Objectives

The F1 platform exposes a FastAPI task backend that is the sole owner of `JobsManager`, workers, lifecycle state, and persistence. Gradio reaches that owner through the stateless `StudioJobsApiClient`; it does not create a second lifecycle. Because the dispatcher is the security boundary between untrusted job submissions and the host filesystem/runtime, its enforcement logic was treated as a first-class security control rather than a functional convenience.

This report verifies three **official security red lines** and confirms compliance against each:

| # | Security Objective | Compliance Result |
| --- | --- | --- |
| SO-1 | **Shell injection defense** — prohibit arbitrary shell execution; task dispatch must never invoke an OS shell with attacker-influenced input. | **COMPLIANT** — `allow_shell=True` is rejected at dispatcher pre-execution guard (`SEC_ERR_001`); handlers delegate to the Python engine directly (no `shell=`, no `subprocess`, no `os.system`/`os.popen`). |
| SO-2 | **Directory traversal / unauthorized path blocking** — enforce strict path whitelisting (models, datasets, output artifacts) so `../` escapes, symlink escapes, and non-whitelisted patterns cannot read or write outside approved roots. | **COMPLIANT** — fail-closed containment + regex whitelist in `BaseTaskHandler._is_path_safe`; violations raise `PathWhitelistViolationError` mapped to `SEC_ERR_001`. |
| SO-3 | **Sensitive credential masking** — environment variables and tokens (`API_KEY`, `TOKEN`, `PASSWORD`, `SECRET`, etc.) must never reach terminal output, log files, error messages, or tracebacks. | **COMPLIANT** — `core/security.py` redacts keys, secret-shaped values, and `KEY=value` assignments; the canonical log append path (`JobRequest.append_log`) sanitizes before storage; dispatcher error messages are sanitized before `ErrorInfo` attachment. |

> **Note:** All three objectives are enforced under a **fail-closed** philosophy: empty whitelists reject everything, `allow_shell` defaults to `False`, and `path_whitelisted` defaults to `True`. A missing or malformed security control yields *deny*, never *allow*.

---

## 2. Security Test Scope & Test Environment

### 2.1 Scope & Exclusion

**In scope:**

- `core/security.py` — log & environment sanitizer (regex redaction patterns + `sanitize_env_dict` / `sanitize_log_text`).
- `f1/handlers/base.py` — `BaseTaskHandler`, `_is_path_safe` (directory containment + regex whitelist), `PathWhitelistViolationError`, `allow_shell` prohibition contract.
- `f1/dispatcher.py` — `JobDispatcherStateMachine` pre-execution security guard (`SEC_ERR_001`) and exception/log sanitization on the `FAILED` path.
- `core/schema.py` — `SecurityConstraints` (fail-closed defaults) and `JobRequest.append_log` (sanitizing log interface).

**Out of scope:** API authentication, role-based access control (RBAC), multi-tenant isolation, and cross-process dispatch hardening — documented in §5 Residual Risk.

### 2.2 Test Environment

| Dimension | Value |
| --- | --- |
| **Runtime** | `Python 3.12.10 (Project Virtual Environment)` — `D:\Projects\YOLO-Master-Recovered\.venv\Scripts\python.exe` |
| **Test harness** | `pytest 9.1.1`, `pluggy 1.6.0` |
| **Platform** | `Windows (x86_64)` |
| **pytest config** | Repository-root `pytest.ini` (`rootdir = D:\Projects\YOLO-Master-Recovered`) |
| **Isolation technique** | In-memory single-process execution; `tmp_path` fixture for filesystem-bound cases; `monkeypatch.setitem` for registry injection; mock `BaseTaskHandler` subclasses with no real model inference |
| **Mocked environment states** | `TaskHandlerRegistry._handlers` mutated via `monkeypatch`; secret-carrying handler raises a `RuntimeError` embedding a plaintext token to exercise the leak vector |
| **Test data** | Ephemeral string fixtures only — no real credentials, no network, no GPU; the selected tests run offline |

---

## 3. Detailed Test Suites & Verification Matrix

### 3.0 Requirement → Enforcement → Evidence Traceability

| Requirement (red line) | Enforcement point | Test evidence |
| --- | --- | --- |
| Prohibit arbitrary shell execution | Dispatcher guard `Rule 1` + per-handler `allow_shell` check → `SEC_ERR_001` | `tests/f1/test_handlers_framework.py::TestEndToEndIntegration` (happy path uses `allow_shell: False`; violation path rejected) |
| Strict path whitelisting | `SecurityConstraints.path_whitelisted` + `BaseTaskHandler._is_path_safe` | `tests/f1/test_handlers_framework.py::TestBaseTaskHandler::test_path_safety_validation_baseline`, `::TestPathSafetyLiteralWhitelist`, `::TestPathSafetyRegexWhitelist` |
| No credential leakage | `core.security.sanitize_log_text` / `sanitize_env_dict` + `JobRequest.append_log` | `tests/f1/test_security_sanitizer.py` (9 cases) |

### 3.1 Path Traversal & Regex Whitelist

Implemented by `BaseTaskHandler._is_path_safe`, which partitions `allowed_roots` into directory-containment roots and regex patterns (entries prefixed with `^`), then applies two independent checks:

1. **Directory containment** — `Path(target).resolve()` is compared against each resolved root via the parent-chain (`resolved in root.parents`), so `..` and symlinks are resolved *before* the containment decision.
2. **Regex whitelist** — entries in `allowed_patterns` (and `^`-prefixed `allowed_paths` entries) are matched with `re.match` against the **normalized resolved POSIX path**, and the match **must consume the entire string** (`match.end() == len(resolved_str)`). Partial / prefix matches never whitelist sibling paths.

| Case | Input | Result | Assertion |
| --- | --- | --- | --- |
| Valid within root | `ultralytics/assets/bus.jpg` ⊆ `["."]` | **ALLOW** | Returns `True` |
| Dot-segment neutralization | `./././ultralytics/../ultralytics/assets` ⊆ `["."]` | **ALLOW** | Returns `True` |
| Upward traversal | `../../etc/passwd` ⊆ `["ultralytics/assets"]` | **DENY** | Returns `False` |
| Empty whitelist (fail-closed) | `valid_file.txt` ⊆ `[]` | **DENY** | Returns `False` |
| Full regex match | pattern `^{root}/models/.*\.pt$` vs `…/models/yolov8n.pt` | **ALLOW** | Returns `True` |
| Textual match, resolved escape | `…/sub/../../escape.pt` vs `^{root}/sub/.*\.pt$` | **DENY** | Returns `False` |
| Substring/prefix bypass | sibling `…/models_evil/weight.pt` vs `^{root}/models` | **DENY** | Returns `False` |
| Symlink escape | link → `tmp_path.parent`, target `link/secret.pt` vs `^{root}/.*\.pt$` | **DENY** | Returns `False` (resolved path leaves the pattern boundary) |
| Malformed regex (fail-closed) | pattern `^[unclosed(` | **DENY, no crash** | Returns `False` without raising |

**Verified results:** 12 passed, 5 skipped. The four literal-root symlink matrix cases and `test_regex_symlink_escape_fails` require symlink creation, which this Windows host denied; each is conditionally skipped via `pytest.skip`.

> **Warning:** Five symlink cases are **SKIPPED** on the current Windows host (not *failing*). Their logic can run on hosts where the test process may create symlinks.

### 3.2 Command & Shell Injection Prevention

The dispatcher enforces two pre-execution guards *before* any handler executes:

```text
Rule 1: allow_shell == True  →  FAILED, SEC_ERR_001  ("Shell execution not permitted")
Rule 2: path_whitelisted == False → FAILED, SEC_ERR_001 ("Path whitelisting must be enabled")
```

Additionally, every concrete handler (`predict`, `train`, `val`, `export`, `diagnose`) independently re-checks `allow_shell` in `validate_params` and returns `(False, "Shell execution is not allowed for <task> tasks")` — a defense-in-depth check that fails closed even if the dispatcher guard were bypassed.

| Case | Setup | Result |
| --- | --- | --- |
| Shell allowed | `security_constraints.allow_shell = True` | Dispatcher rejects → `SEC_ERR_001` |
| Whitelist disabled | `security_constraints.path_whitelisted = False` | Dispatcher rejects → `SEC_ERR_001` |
| Happy path (direct param list) | `allow_shell: False`, valid path | Handler validates & executes without any OS shell |
| Traversal via params | `model_path = "../../etc/passwd"` | Handler `validate_params` returns `(False, "... not in whitelist")` |

The dispatcher and task handlers contain **no** `subprocess.run(..., shell=True)`, `os.system`, `os.popen`, or string-concatenated shell command construction. Handlers delegate execution to the Ultralytics Python engine using direct parameter lists — shell-interpreted metacharacters (`;`, `|`, `&&`, `` ` ``, `$()`) have no interpreter to act on.

### 3.3 Credential & Sensitive Variable Sanitization

`core/security.py` implements a layered redaction model (stdlib-only: `re` + `typing`, no third-party dependencies or import cycles):

| Pattern | Purpose | Examples redacted |
| --- | --- | --- |
| `SENSITIVE_KEY_PATTERN` (case-insensitive) | Sensitive **key names** in env mappings | `WANDB_API_KEY`, `AWS_SECRET_ACCESS_KEY`, `DB_PASSWORD`, `GITHUB_TOKEN` |
| `SENSITIVE_VALUE_PATTERNS` | Secret **value shapes** even under innocent names | `Bearer eyJ…`, `sk-…`, `AKIA…`, `ghp_…`, `github_pat_…`, `hf_…`, `api_key=…` |
| `SENSITIVE_ASSIGNMENT_PATTERN` | Fail-closed `KEY=value` / `KEY: value` assignments in free-form text | `DB_PASSWORD=hunter2` (short secrets that value-shape patterns would miss) |

Redaction behavior:
- `sanitize_env_dict` returns a **new** sanitized dict (input never mutated), preserving non-secret values and their types.
- `sanitize_log_text` is **idempotent** (re-sanitizing redacted text is a no-op) and passes empty input through unchanged.
- Prefix-preserving patterns keep a readable prefix (e.g. `Bearer ***REDACTED***`, `API_KEY=***REDACTED***`); whole-secret patterns (e.g. `ghp_…`) redact the entire match.
- `JobRequest.append_log` routes every log line through `sanitize_log_text` **before** entering `job.logs`, and the dispatcher sanitizes the exception message before attaching `ErrorInfo`.

---

## 4. Execution Results & Coverage

The full sanitizer suite was executed on 2026-09-13 under the project virtual environment (`python -m pytest`) and produced **9 passed in 0.22 s**. Below is the complete verification matrix for `tests/f1/test_security_sanitizer.py`.

| # | Test Case | Status | Key Assertions | Log / Output Snippet |
| --- | --- | --- | --- | --- |
| 1 | `TestSanitizeEnvDictKeys::test_sanitize_env_dict_keys` | **PASS** | Sensitive key names → `***REDACTED***`; `PATH`, `PYTHONPATH`, `USER`, `JOB_ID` preserved; input dict not mutated and a new dict returned. | `WANDB_API_KEY → ***REDACTED***`; `PATH → /usr/local/bin:/usr/bin` (unchanged) |
| 2 | `TestSanitizeEnvDictValues::test_sanitize_env_dict_values` | **PASS** | Secret-shaped values under innocent names (`MY_VAR`, `MY_OTHER_VAR`) redacted; `PLAIN_VAR` kept. | `MY_VAR="Bearer eyJhbGciOiJIUzI1NiIs…" → ***REDACTED***`; `MY_OTHER_VAR="sk-1234567890abcdef12345678" → ***REDACTED***` |
| 3 | `TestSanitizeLogText::test_sanitize_log_text_patterns` | **PASS** | Multi-line text: Bearer token, `API_KEY=`, `AKIA` ID, `ghp_` and `hf_` tokens all masked; prefixes and ordinary lines preserved. | `Connecting with Bearer ***REDACTED***`; `Exporting API_KEY=***REDACTED***`; `DEBUG: epoch=1 loss=0.4231` (unchanged) |
| 4 | `TestSanitizeLogText::test_sanitize_log_text_idempotent_and_empty` | **PASS** | Idempotency (`sanitize(once) == once`) and empty-input passthrough (`""` → `""`). | `token sk-secret998877665544332211 → ***REDACTED***`; re-run is a no-op |
| 5 | `TestKnownSecretConfigLiteralFilter::test_ambient_boolean_flag_does_not_redact_ordinary_log_literal` | **PASS** | Boolean/config literals from sensitive-named ambient flags do not over-redact ordinary metrics. | `epoch=1`, `ratio=0`, and `mode=True` remain readable. |
| 6 | `TestKnownSecretConfigLiteralFilter::test_short_real_secret_is_still_redacted` | **PASS** | A short real secret under a sensitive key remains protected without redacting larger ordinary words. | Known secret value redacted; `alphabet` preserved. |
| 7 | `TestKnownSecretConfigLiteralFilter::test_ordinary_credential_redaction_unaffected` | **PASS** | Token/password assignments and Bearer-shaped values remain redacted. | All three credential forms become `***REDACTED***`. |
| 8 | `TestDispatcherExceptionLogSanitization::test_dispatcher_exception_log_sanitization` | **PASS** | End-to-end: secret-carrying handler → job `FAILED` with `EXEC_ERR_500`; secret absent from both `error_message` and joined `logs`; `***REDACTED***` present in both. | `✓ Dispatcher exception path fully sanitized: <REDACTED message>` |
| 9 | `TestDispatcherExceptionLogSanitization::test_append_log_interface_sanitizes` | **PASS** | Canonical `job.append_log` redacts before storage; `len(job.logs)==1`; secret absent, placeholder present; no-failure job exposes empty `error_message`. | `uploading artifact with token ***REDACTED***` stored in `job.logs[0]` |

**Session command and output (2026-09-13):**

```text
.\.venv\Scripts\python.exe -m pytest tests/f1/test_security_sanitizer.py -q -o addopts= -p no:cacheprovider --basetemp=runs/doc-security-sanitizer
.........                                                                [100%]
9 passed in 0.22s
```

**Supplementary path-whitelist command and output (2026-09-13):**

```text
.\.venv\Scripts\python.exe -m pytest tests/f1/test_handlers_framework.py::TestBaseTaskHandler::test_path_safety_validation_baseline tests/f1/test_handlers_framework.py::TestPathSafetyLiteralWhitelist tests/f1/test_handlers_framework.py::TestPathSafetyRegexWhitelist -q -o addopts= -p no:cacheprovider --basetemp=runs/doc-path-whitelist
......ssss......s                                                        [100%]
12 passed, 5 skipped in 0.19s
```

---

## 5. Residual Risk & Mitigation Roadmap

### 5.1 Current Boundary

The FastAPI service is the sole lifecycle owner and holds one process-wide `JobsManager`; Gradio uses `StudioJobsApiClient` and owns no workers, lifecycle registry, or persistence. Jobs execute in managed worker processes under the service owner's supervision. The current protections are strong for this threat model:

- No OS shell is ever instantiated; no `subprocess`, `os.system`, or `os.popen` call sites exist.
- Path whitelisting is fail-closed, resolves symlinks/`..` before decision, and regex patterns must fully consume the resolved path (no prefix/substring bypass).
- Credential redaction covers key names, value shapes, and `KEY=value` assignments, and is enforced at the canonical log append boundary *and* the dispatcher error boundary.

### 5.2 Known Residual Gaps

| Risk | Severity | Notes |
| --- | --- | --- |
| Symlink tests skipped on Windows | Low | Five literal/regex symlink cases skip when symlink creation is non-permitted. Their assertions were not executed on this host. |
| Regex whitelist is policy-authored | Medium | Trusted backend or embedding policy can supply patterns to handlers; the Job API discards client path lists/patterns and injects server-configured roots. An overly broad trusted pattern could still whitelist too much. |
| Worker privilege model | Medium | Managed worker processes retain ambient host privileges; there is no per-job OS-level sandbox. Acceptable for trusted local operation, not for multi-tenant or remote untrusted submission. |
| No authentication / authorization | High | The task API has no caller authentication and no RBAC; any caller able to reach it can submit jobs. |
| Redaction is regex heuristics | Low | `core/security.py` is pattern-based, not a parser; novel secret formats (e.g. a new vendor token shape) are not covered until a pattern is added. |

### 5.3 Hardening Roadmap (P2)

1. **RBAC & API authentication** — introduce caller identity and role scopes (submit / cancel / view-all) at the dispatcher boundary; enforce least-privilege per task type.
2. **Pattern-policy validation** — validate `allowed_path_patterns` against a deny-of-broad-pattern rule (reject `.*`-to-root patterns) at schema/validation time.
3. **Per-job sandboxing** — move execution to an isolated subprocess or container with a read-only filesystem and restricted environment.
4. **Extensible secret catalog** — promote `SENSITIVE_VALUE_PATTERNS` to a configuration-driven registry so new credential formats can be added without code changes.
5. **Windows symlink CI leg** — add an admin-gated or Linux CI leg so the symlink-escape case is always executed, not skipped.
