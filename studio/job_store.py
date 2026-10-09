"""Validated JSON snapshots, atomic replacement and known-good backup recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path

from core.schema import JobRequest
from studio.job_logs import sanitize_log_text, sanitize_snapshot_value


class StateRecoveryError(RuntimeError):
    """Existing state cannot be safely restored from either snapshot."""


class JobStore:
    """Store records without interpreting or changing their lifecycle."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path is not None else None
        self.backup = self.path.with_name(self.path.name + ".bak") if self.path else None

    @staticmethod
    def validate(payload):
        """Validate every record, identity and log entry before accepting a snapshot."""
        if not isinstance(payload, dict) or payload.get("version", 1) != 1:
            raise ValueError("Unsupported state snapshot")
        jobs, logs = payload.get("jobs"), payload.get("job_logs", {})
        if not isinstance(jobs, dict) or not isinstance(logs, dict):
            raise TypeError("Invalid state snapshot shape")
        clean_jobs = {}
        for job_id, raw in jobs.items():
            job = JobRequest.model_validate(raw)
            if job_id != job.job_id:
                raise ValueError("Snapshot job identity mismatch")
            clean = sanitize_snapshot_value(raw)
            # Credential-shaped IDs cannot be safely published or restored.
            JobRequest.model_validate(clean)
            if clean["job_id"] != job_id:
                raise ValueError("Snapshot identifier cannot be safely sanitized")
            clean_jobs[job_id] = clean
        if any(
            not isinstance(key, str) or not isinstance(lines, list) or any(not isinstance(line, str) for line in lines)
            for key, lines in logs.items()
        ):
            raise ValueError("Invalid snapshot logs")
        return {
            "version": 1,
            "jobs": clean_jobs,
            "job_logs": {key: [sanitize_log_text(line) for line in lines] for key, lines in logs.items()},
        }

    def _read(self, path):
        return self.validate(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _atomic_write(path, serialized):
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except OSError:
                pass  # Some filesystems do not support fsync; replacement is still atomic.
        os.replace(temporary, path)

    @staticmethod
    def _serialize(payload):
        return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)

    def load(self):
        """Recover from backup or fail closed; never silently erase existing history."""
        if self.path is None:
            return {"version": 1, "jobs": {}, "job_logs": {}}
        for candidate in (self.path, self.backup):
            try:
                payload = self._read(candidate)
            except (OSError, ValueError, TypeError):
                continue
            if candidate == self.backup:
                self._atomic_write(self.path, self._serialize(payload))
            return payload
        for candidate in (self.path, self.backup):
            try:
                candidate.lstat()  # Broken symlinks are existing invalid state, not a new environment.
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise StateRecoveryError("Cannot establish whether job snapshots exist") from exc
            raise StateRecoveryError("Neither primary nor backup contains a valid job snapshot")
        return {"version": 1, "jobs": {}, "job_logs": {}}

    def save(self, jobs, logs):
        """Serialize first, rotate only valid primary, then atomically publish new state."""
        if self.path is None:
            return
        payload = self.validate(
            {"version": 1, "jobs": {key: job.model_dump(mode="json") for key, job in jobs.items()}, "job_logs": logs}
        )
        serialized = self._serialize(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            primary = self._read(self.path)
        except (OSError, ValueError, TypeError):
            primary = None
        if primary is not None:
            self._atomic_write(self.backup, self._serialize(primary))
        self._atomic_write(self.path, serialized)
