"""Unit and end-to-end tests for the log & environment sanitization red line.

This suite closes the official security requirement: sensitive environment
variables and credentials must never reach job logs or error messages.

Coverage:
    1. ``sanitize_env_dict`` redacts values whose KEY NAME is sensitive while
       leaving ordinary variables (PATH, PYTHONPATH, USER, JOB_ID) untouched.
    2. ``sanitize_env_dict`` redacts values whose VALUE looks like a secret
       (Bearer token, ``sk-`` key) even under an innocent variable name.
    3. ``sanitize_log_text`` redacts credential fragments inside multi-line
       log text while preserving ordinary log lines.
    4. End-to-end: a handler raising a secret-carrying exception is dispatched
       through ``JobDispatcherStateMachine``; the job reaches FAILED and
       neither ``job.error_message`` nor ``job.logs`` contains the plaintext
       secret, while both contain the redaction placeholder.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.schema import JobRequest, JobStatus, RuntimeTracking, SecurityConstraints, TaskType
from core.security import REDACTED, sanitize_env_dict, sanitize_log_text
from f1.dispatcher import JobDispatcherStateMachine
from f1.handlers.base import BaseTaskHandler
from f1.handlers.registry import TaskHandlerRegistry

# Plaintext secret used by the end-to-end test. Defined as a constant (never
# inline in the raising statement) so the traceback's source line does not
# embed the literal; the exception MESSAGE line carries it, which is exactly
# the leak vector under test.
SECRET_TOKEN = "sk-secret998877665544332211"


class TestSanitizeEnvDictKeys:
    """Key-name based redaction for ``sanitize_env_dict``."""

    def test_sanitize_env_dict_keys(self) -> None:
        """Sensitive key names are redacted; ordinary variables pass through unchanged."""
        env = {
            "WANDB_API_KEY": "wandb-secret-value-001",
            "GITHUB_TOKEN": "ghp_1234567890abcdefghij",
            "AWS_SECRET_ACCESS_KEY": "aws-secret-value",
            "DB_PASSWORD": "super-secret-password",
            "PATH": "/usr/local/bin:/usr/bin",
            "PYTHONPATH": "/opt/python/lib",
            "USER": "tester",
            "JOB_ID": "job-20260905-001",
        }

        sanitized = sanitize_env_dict(env)

        for key in ("WANDB_API_KEY", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY", "DB_PASSWORD"):
            assert sanitized[key] == REDACTED, f"{key} must be redacted"
        for key in ("PATH", "PYTHONPATH", "USER", "JOB_ID"):
            assert sanitized[key] == env[key], f"{key} must keep its original value"

        # The input mapping is never mutated and a new dict is returned.
        assert sanitized is not env
        assert env["WANDB_API_KEY"] == "wandb-secret-value-001"


class TestSanitizeEnvDictValues:
    """Value-shape based redaction for ``sanitize_env_dict``."""

    def test_sanitize_env_dict_values(self) -> None:
        """Secret-shaped values are redacted even under innocent variable names."""
        env = {
            "MY_VAR": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "MY_OTHER_VAR": "sk-1234567890abcdef12345678",
            "PLAIN_VAR": "just-a-normal-value",
        }

        sanitized = sanitize_env_dict(env)

        assert sanitized["MY_VAR"] == REDACTED
        assert sanitized["MY_OTHER_VAR"] == REDACTED
        assert sanitized["PLAIN_VAR"] == "just-a-normal-value"


class TestSanitizeLogText:
    """Credential-fragment redaction for ``sanitize_log_text``."""

    def test_sanitize_log_text_patterns(self) -> None:
        """Multi-line log text has every credential fragment masked, ordinary lines kept."""
        log_text = (
            "INFO: Starting training run job-001\n"
            "Connecting with Bearer eyJhbGci1234567890abcdef\n"
            "Exporting API_KEY=secret_key_12345678\n"
            "AWS credential AKIAIOSFODNN7EXAMPLE detected\n"
            "Pushing results with GitHub token ghp_abcdefghijklmnopqrstuvwxyz0123456789\n"
            "Uploading weights with HuggingFace token hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ01234567\n"
            "DEBUG: epoch=1 loss=0.4231\n"
        )

        sanitized = sanitize_log_text(log_text)

        assert REDACTED in sanitized
        assert "eyJhbGci1234567890abcdef" not in sanitized
        assert "secret_key_12345678" not in sanitized
        assert "AKIAIOSFODNN7EXAMPLE" not in sanitized
        assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in sanitized
        assert "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ01234567" not in sanitized
        # Non-secret context is preserved: prefixes stay, ordinary lines pass through.
        assert "Bearer ***REDACTED***" in sanitized
        assert "API_KEY=***REDACTED***" in sanitized
        # Developer tokens are replaced wholesale by the redaction placeholder.
        assert "Pushing results with GitHub token ***REDACTED***" in sanitized
        assert "Uploading weights with HuggingFace token ***REDACTED***" in sanitized
        assert "DEBUG: epoch=1 loss=0.4231" in sanitized

    def test_sanitize_log_text_idempotent_and_empty(self) -> None:
        """Re-sanitizing redacted text is a no-op; empty input passes through."""
        once = sanitize_log_text(f"token {SECRET_TOKEN}")
        assert SECRET_TOKEN not in once
        assert sanitize_log_text(once) == once
        assert sanitize_log_text("") == ""


class _SecretLeakingHandler(BaseTaskHandler):
    """Mock handler that leaks a secret token through a runtime exception."""

    def validate_params(self, params: dict[str, Any], security_constraints: dict[str, Any]) -> tuple[bool, str | None]:
        """Accept all parameters; the leak happens at execution time.

        Args:
            params: Task parameters (unused).
            security_constraints: Security policy dict (unused).

        Returns:
            tuple[bool, str | None]: Always ``(True, None)``.
        """
        return True, None

    def execute(self, job_id: str, params: dict[str, Any], output_dir: str) -> dict[str, Any]:
        """Raise an exception whose message embeds a plaintext secret token.

        Args:
            params: Task parameters (unused).
            job_id: Job identifier (unused).
            output_dir: Output directory (unused).

        Raises:
            RuntimeError: Always, carrying the plaintext ``SECRET_TOKEN``.
        """
        raise RuntimeError(f"Database auth failed for token {SECRET_TOKEN}")


class TestDispatcherExceptionLogSanitization:
    """End-to-end red-line enforcement inside ``JobDispatcherStateMachine``."""

    def test_dispatcher_exception_log_sanitization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FAILED job carries no plaintext secret in error_message or logs."""
        monkeypatch.setitem(TaskHandlerRegistry._handlers, TaskType.PREDICT.value, _SecretLeakingHandler)

        dispatcher = JobDispatcherStateMachine()
        job = JobRequest(
            job_id="test-sec-sanitizer-001",
            task_type=TaskType.PREDICT,
            params={},
            security_constraints=SecurityConstraints(
                path_whitelisted=True,
                allow_shell=False,
                allowed_paths=["."],
            ),
            runtime_tracking=RuntimeTracking(timeout_seconds=30),
        )

        result = dispatcher.execute(job)

        assert result.status == JobStatus.FAILED
        assert result.error is not None
        assert result.error.code == "EXEC_ERR_500"
        # Error message (UI-facing) is sanitized.
        assert SECRET_TOKEN not in result.error_message
        assert REDACTED in result.error_message
        # Log buffer (in-memory / persisted) is sanitized, traceback included.
        logs_text = "\n".join(result.logs)
        assert SECRET_TOKEN not in logs_text
        assert REDACTED in logs_text
        print("✓ Dispatcher exception path fully sanitized:", result.error_message)

    def test_append_log_interface_sanitizes(self) -> None:
        """The canonical ``job.logs`` append interface redacts before storage."""
        job = JobRequest(job_id="test-log-iface-001", task_type=TaskType.PREDICT)

        job.append_log(f"uploading artifact with token {SECRET_TOKEN}")

        assert len(job.logs) == 1
        assert SECRET_TOKEN not in job.logs[0]
        assert REDACTED in job.logs[0]
        # A job without failure exposes an empty error message, never secrets.
        assert job.error_message == ""
