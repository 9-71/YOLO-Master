"""One-command launcher for the FastAPI Studio Job API and Gradio WebUI."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from f1.ui.studio_jobs_client import DEFAULT_STUDIO_JOB_API_URL, STUDIO_JOB_API_URL_ENV

PROJECT_ROOT = Path(__file__).resolve().parent
STARTUP_TIMEOUT_SECONDS = 20.0


def api_is_ready(api_url: str) -> bool:
    """Return whether a compatible Studio Job API answers its health probe."""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=0.5)
        payload = response.json()
    except (requests.RequestException, ValueError):
        return False
    return response.ok and payload.get("status") == "ok"


def start_api(api_url: str) -> subprocess.Popen:
    """Start the local FastAPI service and wait until it becomes healthy."""
    parsed = urlparse(api_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.path not in {"", "/"}:
        raise RuntimeError("Automatic API startup requires a local http://127.0.0.1 or http://localhost URL")

    port = parsed.port or 8000
    environment = os.environ.copy()
    environment["F1_ENGINE_HOST"] = parsed.hostname
    environment["F1_ENGINE_PORT"] = str(port)
    process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "main_engine.py")], cwd=PROJECT_ROOT, env=environment
    )

    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Studio Job API exited during startup with code {process.returncode}")
        if api_is_ready(api_url):
            return process
        time.sleep(0.1)

    stop_api(process)
    raise RuntimeError(f"Studio Job API did not become ready within {STARTUP_TIMEOUT_SECONDS:.0f} seconds")


def stop_api(process: subprocess.Popen | None) -> None:
    """Stop only the API process created by this launcher."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> None:
    """Ensure the API is ready, then run the Gradio WebUI until exit."""
    api_url = os.environ.get(STUDIO_JOB_API_URL_ENV, DEFAULT_STUDIO_JOB_API_URL).rstrip("/")
    os.environ[STUDIO_JOB_API_URL_ENV] = api_url
    api_process = None
    try:
        if not api_is_ready(api_url):
            api_process = start_api(api_url)

        from app import YOLO_Master_WebUI

        checkpoints_dir = PROJECT_ROOT / "ckpts"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        YOLO_Master_WebUI(str(checkpoints_dir)).launch()
    finally:
        stop_api(api_process)


if __name__ == "__main__":
    main()
