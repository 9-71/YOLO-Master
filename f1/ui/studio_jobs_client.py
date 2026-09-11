"""Stateless Studio Job API adapter for the Gradio Jobs tab.

The Gradio UI historically received a local ``JobsManager`` instance.  This
adapter preserves that small, duck-typed presentation interface while routing
every background-job operation through the FastAPI service.  It owns no worker,
job registry, lifecycle state, or persistence file.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import quote

import requests

STUDIO_JOB_API_URL_ENV = "F1_STUDIO_API_URL"
DEFAULT_STUDIO_JOB_API_URL = "http://127.0.0.1:8000"


class ArtifactMetadata(TypedDict):
    """Dual-path artifact metadata consumed by the Gradio compatibility layer."""

    filename: str
    artifact_id: str
    preview_path: str
    source_path: str
    is_image: bool
    download_url: str


class StudioJobsApiError(RuntimeError):
    """Base class for structured Studio Job API failures."""

    code = "STUDIO_API_ERROR"
    i18n_key = "platform.api_error"

    def __init__(self, method: str, path: str, *, detail: str = "", status_code: int | None = None) -> None:
        self.method = method
        self.path = path
        self.detail = detail
        self.status_code = status_code
        super().__init__(f"{self.code}: {method} {path}")


class StudioBackendUnavailableError(StudioJobsApiError):
    """The Studio Job API could not be reached before a request was accepted."""

    code = "STUDIO_BACKEND_UNAVAILABLE"
    i18n_key = "platform.backend_unavailable"


class StudioJobsApiResponseError(StudioJobsApiError):
    """The Studio Job API returned an HTTP error or malformed response."""

    code = "STUDIO_API_RESPONSE_ERROR"
    i18n_key = "platform.api_response_error"


class StudioJobsApiClient:
    """Adapt the Studio Job API to the interface consumed by the Gradio Jobs tab."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 5.0,
        session: requests.Session | None = None,
    ) -> None:
        """Configure the API endpoint without creating or caching any job state.

        Args:
            base_url: Studio API origin. Defaults to ``F1_STUDIO_API_URL`` and
                then ``http://127.0.0.1:8000``.
            timeout: Per-request timeout in seconds.
            session: Optional requests-compatible session, primarily for tests.
        """
        configured_url = base_url or os.environ.get(STUDIO_JOB_API_URL_ENV, DEFAULT_STUDIO_JOB_API_URL)
        self.base_url = configured_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()
        self._artifact_cache = tempfile.TemporaryDirectory(prefix="yolo-master-gradio-artifacts-")
        self._artifact_cache_lock = threading.Lock()
        self._artifact_roots: dict[str, Path] = {}

    def _send(self, method: str, path: str, **kwargs: Any) -> Any:
        """Send one API request and convert transport/HTTP failures consistently."""
        url = f"{self.base_url}{path}"
        if method.upper() == "GET":
            headers = dict(kwargs.pop("headers", {}))
            headers.setdefault("Cache-Control", "no-cache, no-store")
            headers.setdefault("Pragma", "no-cache")
            kwargs["headers"] = headers
        try:
            response = self._session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise StudioBackendUnavailableError(method, path) from exc
        except requests.HTTPError as exc:
            detail = ""
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    detail = response.json().get("detail", "")
                except (AttributeError, ValueError):
                    detail = response.text
            raise StudioJobsApiResponseError(
                method,
                path,
                detail=str(detail),
                status_code=getattr(response, "status_code", None),
            ) from exc
        except requests.RequestException as exc:
            raise StudioJobsApiError(method, path) from exc
        return response

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Send one JSON request and validate the response shape."""
        response = self._send(method, path, **kwargs)

        try:
            payload = response.json()
        except ValueError as exc:
            raise StudioJobsApiResponseError(method, path, detail="invalid_json") from exc
        if not isinstance(payload, dict):
            raise StudioJobsApiResponseError(method, path, detail="invalid_payload")
        return payload

    @staticmethod
    def _status(value: Any) -> str:
        """Normalize REST enum values to the uppercase form expected by Gradio."""
        return str(value or "NOT_FOUND").upper()

    def _artifact_url(self, job_id: str, artifact_id: str) -> str:
        """Build the API's fail-closed artifact delivery URL."""
        safe_job_id = quote(job_id, safe="")
        safe_artifact_id = quote(artifact_id, safe="/")
        return f"{self.base_url}/static/artifacts/{safe_job_id}/{safe_artifact_id}"

    def _artifact_source_path(self, job_id: str, artifact_id: str) -> str:
        """Resolve a submitted job's local source path, falling back to its download URL."""
        root = self._artifact_roots.get(job_id)
        if root is None:
            return self._artifact_url(job_id, artifact_id)
        try:
            source_path = (root / artifact_id).resolve()
            source_path.relative_to(root)
        except (OSError, ValueError):
            return self._artifact_url(job_id, artifact_id)
        return str(source_path)

    def _materialize_artifact(self, job_id: str, entry: dict[str, Any]) -> str:
        """Download one manifest-approved artifact to a local Gradio-safe path."""
        artifact_id = str(entry.get("artifact_id") or "")
        filename = Path(str(entry.get("filename") or artifact_id)).name
        if not artifact_id or not filename:
            return ""

        cache_key = hashlib.sha256(f"{job_id}\0{artifact_id}".encode()).hexdigest()
        target_dir = Path(self._artifact_cache.name) / cache_key
        target = target_dir / filename
        with self._artifact_cache_lock:
            if target.is_file():
                return str(target)

            path = f"/static/artifacts/{quote(job_id, safe='')}/{quote(artifact_id, safe='/')}"
            response = self._send("GET", path, stream=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(f"{target.suffix}.part")
            with partial.open("wb") as file:
                iterator = (
                    response.iter_content(chunk_size=1024 * 1024)
                    if hasattr(response, "iter_content")
                    else response.iter_bytes(chunk_size=1024 * 1024)
                )
                for chunk in iterator:
                    if chunk:
                        file.write(chunk)
            partial.replace(target)
        return str(target)

    def submit_job(
        self,
        task_type: str,
        model_path: str,
        data_source: str,
        output_dir: str,
        conf: float,
        device: str,
        allowed_paths: list[str],
    ) -> tuple[str, str]:
        """Submit a background job exclusively through ``POST /api/v1/jobs``."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_id = f"{task_type}_{timestamp}_{uuid.uuid4().hex[:8]}"

        params: dict[str, Any] = {}
        if task_type in {"predict", "train", "val"}:
            params.update(model_path=model_path, data_source=data_source, conf=conf, device=device)
        elif task_type == "export":
            params.update(model_path=model_path, format="onnx")
        if task_type == "train":
            params.update(epochs=1, imgsz=640)
        elif task_type == "val":
            params["imgsz"] = 640

        payload = {
            "job_id": job_id,
            "task_type": task_type,
            "params": params,
            "output": {"output_dir": output_dir},
            "security_constraints": {
                "path_whitelisted": True,
                "allow_shell": False,
                "allowed_paths": allowed_paths,
            },
            "runtime_tracking": {
                "stream_logs": True,
                "timeout_seconds": 300,
                "cancellable": True,
                "cancel_requested": False,
            },
        }
        submitted = self._request("POST", "/api/v1/jobs", json=payload)
        accepted_job_id = str(submitted.get("job_id") or job_id)
        submitted_output = submitted.get("output") if isinstance(submitted.get("output"), dict) else {}
        source_output_dir = str(submitted_output.get("output_dir") or output_dir)
        self._artifact_roots[accepted_job_id] = (
            Path(source_output_dir).expanduser().resolve() / accepted_job_id
        ).resolve()
        return accepted_job_id, ""

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Return the selected job status in the Gradio-compatible shape."""
        try:
            payload = self._request("GET", f"/api/v1/jobs/{quote(job_id, safe='')}")
        except StudioJobsApiResponseError as exc:
            if exc.status_code == 404:
                return {"status": "NOT_FOUND", "message": ""}
            raise
        return {
            "status": self._status(payload.get("status")),
            "duration": payload.get("duration"),
            "error_code": payload.get("error_code"),
            "error_message": payload.get("error_message"),
            "artifact_count": payload.get("artifact_count", 0),
        }

    def get_job_logs(self, job_id: str) -> str:
        """Fetch all currently available sanitized log lines."""
        payload = self._request("GET", f"/api/v1/jobs/{quote(job_id, safe='')}/logs")
        logs = payload.get("logs", [])
        return "\n".join(str(line) for line in logs)

    def _artifact_entries(self, job_id: str) -> list[dict[str, Any]]:
        """Fetch one artifact manifest without retaining it locally."""
        payload = self._request("GET", f"/api/v1/jobs/{quote(job_id, safe='')}/artifacts")
        entries = payload.get("artifacts", [])
        return [entry for entry in entries if isinstance(entry, dict)]

    def get_job_artifact_metadata(self, job_id: str) -> list[ArtifactMetadata]:
        """Return source paths for file actions and local paths for Gradio previews."""
        metadata: list[ArtifactMetadata] = []
        for entry in self._artifact_entries(job_id):
            artifact_id = str(entry.get("artifact_id") or "")
            filename = Path(str(entry.get("filename") or artifact_id)).name
            if not artifact_id or not filename:
                continue
            is_image = bool(entry.get("is_image"))
            metadata.append(
                {
                    "filename": filename,
                    "artifact_id": artifact_id,
                    "preview_path": self._materialize_artifact(job_id, entry) if is_image else "",
                    "source_path": self._artifact_source_path(job_id, artifact_id),
                    "is_image": is_image,
                    "download_url": self._artifact_url(job_id, artifact_id),
                }
            )
        return metadata

    def get_job_artifacts(self, job_id: str) -> list[tuple[str, str]]:
        """Return artifact IDs paired with their original on-disk source paths."""
        return [
            (artifact_id, self._artifact_source_path(job_id, artifact_id))
            for entry in self._artifact_entries(job_id)
            if (artifact_id := str(entry.get("artifact_id") or ""))
        ]

    def get_job_image_artifacts(self, job_id: str) -> list[str]:
        """Return local paths for image artifacts suitable for Gradio previews."""
        return [
            local_path
            for entry in self._artifact_entries(job_id)
            if entry.get("is_image") and (local_path := self._materialize_artifact(job_id, entry))
        ]

    def cancel_job(self, job_id: str) -> str:
        """Request cancellation through the Studio Job API."""
        payload = self._request("POST", f"/api/v1/jobs/{quote(job_id, safe='')}/cancel")
        return str(payload.get("message") or "")

    def list_recent_jobs(self, limit: int = 10) -> list[dict[str, str]]:
        """Fetch recent jobs from the Studio Job API."""
        payload = self._request("GET", "/api/v1/jobs", params={"limit": limit, "offset": 0})
        return [
            {
                "job_id": str(job.get("job_id", "")),
                "task_type": str(job.get("task_type", "")),
                "status": self._status(job.get("status")),
                "created_at": str(job.get("created_at", "")),
            }
            for job in payload.get("jobs", [])
            if isinstance(job, dict)
        ]
