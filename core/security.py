"""Sensitive environment-variable and log sanitization for the F1 platform.

This module closes the official security red line: sensitive environment
variables and credentials MUST NEVER reach job logs, error messages, tracebacks
or any UI-facing text. It is intentionally implemented with the Python standard
library only (``re`` + ``typing``) so it can be imported by any layer of the
platform (``core.schema``, dispatcher runtime, handlers, WebUI) without pulling
in third-party dependencies or creating import cycles.

Redaction model:
    - :data:`SENSITIVE_KEY_PATTERN` matches common sensitive KEY NAMES
      (case-insensitive): ``KEY``, ``TOKEN``, ``SECRET``, ``PASSWD``,
      ``PASSWORD``, ``AUTH``, ``CREDENTIAL``, ``PRIVATE``, ``ACCESS_KEY``.
    - :data:`SENSITIVE_VALUE_PATTERNS` matches common secret VALUE shapes:
      Bearer tokens, ``sk-`` API keys, AWS ``AKIA`` access key IDs and
      ``api_key=...`` style assignments.
    - :data:`SENSITIVE_ASSIGNMENT_PATTERN` matches ``SENSITIVE_KEY=value``
      assignments inside free-form log text (fail-closed hardening for short
      secrets that the value-shape patterns cannot detect).
    - Every detected credential fragment is replaced with
      :data:`REDACTED`. Patterns whose first capture group is a non-secret
      prefix (e.g. ``"Bearer "`` or ``"API_KEY="``) keep that prefix so logs
      stay readable; patterns that capture the whole secret redact the entire
      match.

Example:
    >>> sanitize_log_text("connecting with sk-1234567890abcdef12345678")
    'connecting with ***REDACTED***'
    >>> sanitize_env_dict({"GITHUB_TOKEN": "ghp_secret", "EDITOR": "vim"})
    {'GITHUB_TOKEN': '***REDACTED***', 'EDITOR': 'vim'}
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "REDACTED",
    "SENSITIVE_ASSIGNMENT_PATTERN",
    "SENSITIVE_KEY_PATTERN",
    "SENSITIVE_VALUE_PATTERNS",
    "sanitize_env_dict",
    "sanitize_for_persistence",
    "sanitize_log_text",
]

# Canonical replacement string for every redacted credential fragment.
REDACTED: str = "***REDACTED***"

# Sensitive key-name pattern: matches common credential key names as a
# case-insensitive substring, so compound names such as "WANDB_API_KEY",
# "AWS_SECRET_ACCESS_KEY" or "DB_PASSWORD" are all covered.
SENSITIVE_KEY_PATTERN: re.Pattern = re.compile(
    r"(?i)(key|token|secret|pass(?:word|wd)?|auth(?:orization)?|bearer|cred(?:ential)?|private|cookie|access_?key)"
)

# Sensitive value-shape patterns. Group semantics differ per pattern:
#   - Prefix patterns (Bearer, api_key=, KEY=/KEY=" assignments) capture a
#     NON-SECRET prefix as group 1; only the trailing secret is redacted.
#   - Whole-secret patterns (sk-, AKIA, ghp_, github_pat_, hf_) capture the
#     ENTIRE secret as group 1; the whole match is redacted. ``_redact_match``
#     handles both conventions generically.
SENSITIVE_VALUE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)(bearer\s+)[^\s,;\"']+"),
    re.compile(r"(?i)(sk-[A-Za-z0-9_\-]{20,})"),
    re.compile(r"(?i)(AKIA[0-9A-Z]{16})"),
    re.compile(r"(?i)(api[_\-]?key[\s:=]+)[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b(ghp_[A-Za-z0-9]{10,60})\b"),
    re.compile(r"\b(github_pat_[A-Za-z0-9_]{82})\b"),
    re.compile(r"\b(hf_[A-Za-z0-9]{34,40})\b"),
    re.compile(
        r"(?i)\b([A-Za-z_][A-Za-z0-9_]*(?:key|token|secret|pass(?:word|wd)?|auth|cred(?:ential)?|private)"
        r"[A-Za-z0-9_]*\s*[=:]\s*[\"']?)[^\s,;\"']+"
    ),
]

# Fail-closed hardening: a ``SENSITIVE_KEY=value`` (or ``KEY: value``)
# assignment inside log text is redacted regardless of the value's shape, so
# short secrets (e.g. "DB_PASSWORD=hunter2") cannot leak either. Group 1 keeps
# the "KEY=" prefix, group 2 is the secret value up to the next whitespace,
# comma, semicolon or quote.
SENSITIVE_ASSIGNMENT_PATTERN: re.Pattern = re.compile(
    r"(?i)\b([A-Za-z0-9_]*(?:api_?key|token|secret|pass(?:word|wd)?|auth(?:orization)?|bearer|"
    r"cred(?:ential)?|private|cookie)[A-Za-z0-9_]*\s*[=:]\s*)([^\s,;\"']+)"
)
SENSITIVE_QUOTED_ASSIGNMENT_PATTERN: re.Pattern = re.compile(
    r"(?i)([\"']?[A-Za-z0-9_]*(?:api_?key|token|secret|pass(?:word|wd)?|auth(?:orization)?|bearer|"
    r"cred(?:ential)?|private|cookie)[A-Za-z0-9_]*[\"']?\s*[:=]\s*)([\"'])(.*?)\2"
)


def _known_secret_values() -> set[str]:
    """Return sensitive process/.env values that must be scrubbed from persisted text."""
    values = {str(value) for key, value in os.environ.items() if value and SENSITIVE_KEY_PATTERN.search(key)}
    env_path = Path.cwd() / ".env"
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip("\"'")
            if value and SENSITIVE_KEY_PATTERN.search(key.strip()):
                values.add(value)
    except OSError:
        pass
    return values


def _redact_match(match: re.Match[str]) -> str:
    """Build the replacement text for a single sensitive-pattern match.

    Args:
        match: Regex match produced by one of the sensitive patterns.

    Returns:
        str: ``"<prefix>***REDACTED***"`` when group 1 is a non-secret strict
        prefix of the match (e.g. ``"Bearer "``), otherwise the bare
        :data:`REDACTED` placeholder for whole-secret matches.
    """
    if match.lastindex:
        prefix = match.group(1)
        if prefix and match.start(1) == match.start(0) and match.end(1) < match.end(0):
            return f"{prefix}{REDACTED}"
    return REDACTED


def _redact_quoted_assignment(match: re.Match[str]) -> str:
    """Preserve a quoted assignment's key and quote style while masking its value."""
    return f"{match.group(1)}{match.group(2)}{REDACTED}{match.group(2)}"


def _contains_sensitive_value(text: str) -> bool:
    """Check whether a string carries any sensitive value-shaped fragment.

    Args:
        text: Candidate string (typically an environment variable value).

    Returns:
        bool: True when any assignment or value-shape pattern matches.
    """
    if SENSITIVE_ASSIGNMENT_PATTERN.search(text):
        return True
    return any(pattern.search(text) for pattern in SENSITIVE_VALUE_PATTERNS)


def sanitize_env_dict(env_mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitize an environment-variable mapping, masking sensitive keys and values as '***REDACTED***'.

    A key is redacted when its NAME matches :data:`SENSITIVE_KEY_PATTERN`
    (case-insensitive). Otherwise its VALUE (stringified for inspection) is
    tested against :data:`SENSITIVE_VALUE_PATTERNS` and
    :data:`SENSITIVE_ASSIGNMENT_PATTERN`, so secrets stored under innocent
    names (e.g. ``MY_VAR="Bearer eyJ..."``) are redacted as well. Non-secret
    values are copied through unchanged, preserving their original types.

    Args:
        env_mapping: The original environment-variable mapping.

    Returns:
        dict[str, Any]: A new sanitized dict copy; the input mapping is never modified.

    Example:
        >>> sanitize_env_dict({"AWS_SECRET_ACCESS_KEY": "abc", "JOB_ID": "j-1"})
        {'AWS_SECRET_ACCESS_KEY': '***REDACTED***', 'JOB_ID': 'j-1'}
    """
    sanitized: dict[str, Any] = {}
    for key, value in env_mapping.items():
        if SENSITIVE_KEY_PATTERN.search(str(key)):
            sanitized[key] = REDACTED
            continue
        candidate = value if isinstance(value, str) else str(value)
        sanitized[key] = REDACTED if _contains_sensitive_value(candidate) else value
    return sanitized


def sanitize_log_text(text: str) -> str:
    """Sanitize single-line or multi-line log text, redacting sensitive patterns as '***REDACTED***'.

    The text is scanned first for ``SENSITIVE_KEY=value`` assignments
    (:data:`SENSITIVE_ASSIGNMENT_PATTERN`, catches short secrets) and then for
    the credential value shapes in :data:`SENSITIVE_VALUE_PATTERNS`. The
    function is idempotent: re-sanitizing already-redacted text is a no-op.

    Args:
        text: The raw log or traceback text to sanitize.

    Returns:
        str: The sanitized, safe log text; empty input is returned unchanged.

    Example:
        >>> sanitize_log_text("Exporting API_KEY=secret_key_12345678")
        'Exporting API_KEY=***REDACTED***'
    """
    if not text:
        return text
    sanitized = text
    for pattern in SENSITIVE_VALUE_PATTERNS:
        sanitized = pattern.sub(_redact_match, sanitized)
    sanitized = SENSITIVE_QUOTED_ASSIGNMENT_PATTERN.sub(_redact_quoted_assignment, sanitized)
    sanitized = SENSITIVE_ASSIGNMENT_PATTERN.sub(_redact_match, sanitized)
    for secret in sorted(_known_secret_values(), key=len, reverse=True):
        if len(secret) >= 4:
            sanitized = sanitized.replace(secret, REDACTED)
        else:
            short_secret = re.compile(rf"(?<![A-Za-z0-9]){re.escape(secret)}(?![A-Za-z0-9])")
            sanitized = short_secret.sub(REDACTED, sanitized)
    return sanitized


def sanitize_for_persistence(value: Any, key: str | None = None) -> Any:
    """Recursively redact sensitive request/state content before JSON persistence."""
    if key is not None and SENSITIVE_KEY_PATTERN.search(key):
        return REDACTED
    if isinstance(value, Mapping):
        return {str(k): sanitize_for_persistence(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_persistence(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_persistence(item) for item in value]
    if isinstance(value, str):
        return sanitize_log_text(value)
    return value
