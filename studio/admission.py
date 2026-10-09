"""Server-owned admission policy; no registration or lifecycle decisions."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from core.path_safety import resolve_path_allow_missing
from core.schema import JobRequest, TaskType
from core.security import sanitize_log_text

NETWORK_INPUT_SCHEMES = frozenset({"http", "https", "rtmp", "rtsp", "tcp"})


def configured_roots(env_name: str, defaults: list[Path]) -> list[Path]:
    """Read trusted paths exclusively from server configuration."""
    raw = os.environ.get(env_name)
    entries = raw.split(os.pathsep) if raw is not None else [str(path) for path in defaults]
    roots = [resolve_path_allow_missing(entry) for entry in entries if entry.strip()]
    if not roots:
        raise ValueError(f"{env_name} must contain at least one trusted root")
    return roots


def resolve_contained(path_value: str, roots: list[Path], label: str) -> Path:
    """Resolve symlinks and missing tails, rejecting paths outside trusted roots."""
    if not path_value or any(ord(char) < 32 for char in path_value):
        raise ValueError(f"{label} is empty or contains control characters")
    try:
        resolved = resolve_path_allow_missing(path_value)
        for root in roots:
            if resolved == root or root in resolved.parents:
                return resolved
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{label} is not a valid local path") from exc
    raise ValueError(f"{label} is outside the server-configured trusted roots")


class AdmissionPolicy:
    """Normalize a detached request against immutable server-owned configuration."""

    def __init__(self, model_roots=None, data_roots=None, output_root=None, network_input_hosts=None):
        cwd = resolve_path_allow_missing(Path.cwd())
        self.model_roots = (
            tuple(resolve_path_allow_missing(path) for path in model_roots)
            if model_roots is not None
            else tuple(configured_roots("STUDIO_MODEL_ROOTS", [cwd]))
        )
        self.data_roots = (
            tuple(resolve_path_allow_missing(path) for path in data_roots)
            if data_roots is not None
            else tuple(configured_roots("STUDIO_DATA_ROOTS", [cwd]))
        )
        self.output_root = (
            resolve_path_allow_missing(output_root)
            if output_root is not None
            else configured_roots("STUDIO_OUTPUT_ROOT", [cwd / "runs"])[0]
        )
        hosts = (
            network_input_hosts
            if network_input_hosts is not None
            else os.environ.get("STUDIO_NETWORK_INPUT_HOSTS", "").split(",")
        )
        self.network_input_hosts = frozenset(host.strip().casefold() for host in hosts if host.strip())

    def prepare(self, request: JobRequest) -> JobRequest:
        """Validate paths and execution inputs without registering or deciding state."""
        job = JobRequest.model_validate(request.model_dump(mode="python"))
        if sanitize_log_text(job.job_id) != job.job_id:
            raise ValueError("Job identifier must not contain credential-shaped text")
        job.security_constraints.allow_shell = False
        job.security_constraints.path_whitelisted = True
        job.security_constraints.allowed_paths = sorted({str(path) for path in (*self.model_roots, *self.data_roots)})
        job.security_constraints.allowed_path_patterns = []
        job.output.output_dir = str(resolve_contained(job.output.output_dir, [self.output_root], "output_dir"))
        if job.params.get("model_path"):
            model = str(job.params["model_path"])
            scheme = urlsplit(model).scheme
            if scheme and not (os.name == "nt" and len(scheme) == 1):
                raise ValueError("model_path must be a local path")
            job.params["model_path"] = str(resolve_contained(model, self.model_roots, "model_path"))
        source = job.params.get("data_source")
        if source is not None:
            normalized_sources = []
            for value in source if isinstance(source, (list, tuple)) else [source]:
                parsed = urlsplit(str(value))
                if parsed.scheme.lower() in NETWORK_INPUT_SCHEMES and parsed.netloc:
                    if parsed.username is not None or parsed.password is not None:
                        raise ValueError("Network input URLs must not contain user credentials")
                    if parsed.hostname is None or parsed.hostname.casefold() not in self.network_input_hosts:
                        raise ValueError("data_source network host is not server-authorized")
                    normalized_sources.append(str(value))
                else:
                    if parsed.scheme and not (os.name == "nt" and len(parsed.scheme) == 1):
                        raise ValueError("data_source must be a local path or an authorized network URL")
                    normalized_sources.append(str(resolve_contained(str(value), self.data_roots, "data_source")))
            job.params["data_source"] = (
                normalized_sources if isinstance(source, (list, tuple)) else normalized_sources[0]
            )
        if job.task_type == TaskType.TRAIN:
            job.params.setdefault("epochs", 1)
            job.params.setdefault("imgsz", 640)
        elif job.task_type == TaskType.VAL:
            job.params.setdefault("imgsz", 640)
        return job
