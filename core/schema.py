"""Canonical domain schema for the F1 job execution platform.

This module is the single production source of truth for the job domain
contract consumed by the dispatcher runtime, the handler framework, the Gradio
Jobs UI and the agent skill layer.

History: the models below originally lived in the entry contract module
``f1/test_f1_smoke.py`` (P0 legacy debt). P1 relocated them here so that
production modules stop importing data structures from a test file; the legacy
module now re-imports them from this module.

Semantics (unchanged from the original contract):
    - Every field default that carries mutable state (``Metadata``,
      ``OutputConfig``, ``SecurityConstraints``, ``RuntimeTracking``) is
      deep-copied by Pydantic at instance creation, so two jobs never share a
      single class-definition-time instance. ``JobRequest.metadata.created_at``
      is nevertheless stamped per submission by the Jobs UI so Recent Jobs
      ordering reflects real creation times.
    - ``allow_shell=False`` and ``path_whitelisted=True`` are the fail-closed
      security defaults enforced by the dispatcher before any execution.
    - ``TaskType`` lists every registered task handler, including the P1
      ``VAL`` entry.

Example:
    >>> from core.schema import JobRequest, TaskType
    >>> job = JobRequest(job_id="demo-001", task_type=TaskType.PREDICT)
    >>> job.status.value
    'pending'
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from core.security import sanitize_log_text

__version__ = "1.0.0"

__all__ = [
    "ArtifactManifest",
    "ErrorInfo",
    "JobRequest",
    "JobStatus",
    "Metadata",
    "OutputConfig",
    "RuntimeTracking",
    "SecurityConstraints",
    "TaskType",
    "__version__",
]


class TaskType(str, Enum):
    """Task identifier accepted by the job dispatcher and handler registry."""

    PREDICT = "predict"
    TRAIN = "train"
    EXPORT = "export"
    DIAGNOSE = "diagnose"
    VAL = "val"


class JobStatus(str, Enum):
    """Lifecycle state of a job, enforced by the dispatcher state machine."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Metadata(BaseModel):
    """Job bookkeeping metadata attached to every ``JobRequest``."""

    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: str = "anonymous"
    description: str | None = None
    priority: str = "normal"
    tags: list[str] = Field(default_factory=list)


class OutputConfig(BaseModel):
    """Output destination and artifact-capture policy for a job."""

    output_dir: str = "runs/predict/job_20260824_f1_001"
    save_images: bool = True
    save_labels: bool = False
    save_logs: bool = True
    artifacts: list[str] = Field(default_factory=list)


class SecurityConstraints(BaseModel):
    """Fail-closed security policy enforced before and during job execution.

    Regex-enhanced whitelisting (P1): ``allowed_path_patterns`` holds additional
    regex patterns accepted by ``BaseTaskHandler._is_path_safe``. Patterns are
    matched against the normalized resolved POSIX path and must consume the
    entire path string; malformed patterns fail closed. For backward
    compatibility, entries in ``allowed_paths`` starting with ``^`` are also
    interpreted as regex patterns by the handler layer.
    """

    path_whitelisted: bool = True
    allow_shell: bool = False
    allowed_paths: list[str] = Field(default_factory=list)
    allowed_path_patterns: list[str] = Field(default_factory=list)


class RuntimeTracking(BaseModel):
    """Phase 1 runtime supervision knobs: streaming, deadline and cancellation."""

    stream_logs: bool = True
    timeout_seconds: int = 300
    cancellable: bool = True
    cancel_requested: bool = False


class ErrorInfo(BaseModel):
    """Structured error attached to a ``FAILED`` job transition."""

    code: str
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class JobRequest(BaseModel):
    """Strongly typed job submission contract shared by UI, skills and dispatcher.

    Attributes:
        job_id: Unique job identifier (``{task_type}_{timestamp}_{uuid}`` from the UI).
        task_type: Registered task type resolved by the handler registry.
        status: Current lifecycle state; transitions are FSM-enforced.
        metadata: Bookkeeping metadata (author, priority, tags, timestamps).
        params: Task-specific parameters validated by the resolved handler.
        output: Output directory and artifact capture configuration.
        security_constraints: Fail-closed security policy for the job.
        runtime_tracking: Deadline, streaming and cooperative-cancellation flags.
        error: Structured failure detail; ``None`` while the job is not FAILED.
        logs: Sanitized in-memory execution log lines; only written through
            :meth:`append_log`, which redacts credentials before storage so
            UI-facing and persisted logs never carry plaintext secrets.
    """

    job_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    task_type: TaskType
    status: JobStatus = JobStatus.PENDING
    metadata: Metadata = Metadata()
    params: dict[str, Any] = Field(default_factory=dict)
    output: OutputConfig = OutputConfig()
    security_constraints: SecurityConstraints = SecurityConstraints()
    runtime_tracking: RuntimeTracking = RuntimeTracking()
    error: ErrorInfo | None = None
    logs: list[str] = Field(default_factory=list)

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        """Reject path-like reserved identifiers even though they contain allowed characters."""
        if value in {".", ".."}:
            raise ValueError("job_id must not be '.' or '..'")
        return value

    @property
    def error_message(self) -> str:
        """Return the failure message attached to this job, or an empty string.

        The dispatcher sanitizes error text before attaching ``ErrorInfo``, so
        this accessor never exposes plaintext credentials.

        Returns:
            str: ``error.message`` when an error is attached, otherwise ``""``.
        """
        return self.error.message if self.error else ""

    def append_log(self, text: str) -> None:
        """Append a log entry after routing it through the security sanitizer.

        This is the canonical interface for writing UI-facing or persisted job
        logs: every line is passed through
        :func:`core.security.sanitize_log_text` before entering memory, so
        Bearer tokens, ``sk-`` keys, ``KEY=value`` secrets and similar
        credential fragments can never reach ``job.logs``.

        Args:
            text: Raw log text (single- or multi-line) to sanitize and store.
        """
        self.logs.append(sanitize_log_text(text))


class ArtifactManifest(BaseModel):
    """Structured artifact manifest capturing the minimal fields from Appendix B."""

    experiment_id: str = Field(description="Unique experiment identifier, e.g. topic_seed_timestamp")
    git_ref: str = Field(description="Version control anchor: tag + commit SHA")
    config: dict[str, Any] = Field(default_factory=dict, description="Configuration file path and hash")
    dataset: str = Field(description="Dataset name, version, split, and sample count")
    hardware: str = Field(description="Hardware spec: GPU/CPU, VRAM, driver, CUDA/TRT version")
    budget: str = Field(description="Resource budget: epochs, batch size, imgsz, GPU-hours")
    seed: int = Field(description="Explicit random seed value")
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Primary metrics, aux metrics, and confidence intervals",
    )
    artifact: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of artifact roles to paths and checksums",
    )
    status: str = Field(description="Execution outcome: success, failed, or inconclusive")
    limitation: str | None = Field(
        default=None,
        description="Known constraints: single seed, scale, backend limitations",
    )
