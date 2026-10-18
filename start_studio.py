"""One-command local Studio Service and Gradio launcher with owned-process cleanup."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STUDIO_JOB_API_URL = "http://127.0.0.1:8000"
STUDIO_JOB_API_URL_ENV = "STUDIO_API_URL"
LEGACY_STUDIO_JOB_API_URL_ENV = "F1_STUDIO_API_URL"
STARTUP_TIMEOUT_SECONDS = 20.0
POLL_SECONDS = 0.1
# JobsManager.shutdown_budget_seconds currently includes these bounded OS cleanup
# seconds. Keep this outer caller allowance aligned with that public contract.
RUNTIME_CLEANUP_SECONDS = 16.0
SERVICE_EXIT_SECONDS = 5.0
CONFIGURATION_ERROR = "Invalid Studio launcher configuration"
SERVICE_NAME = "YOLO-Master F1 Task Engine"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def configuration_value(suffix: str, default: str | None = None) -> str | None:
    """Honor modern and legacy trusted settings without concealing contradictions."""
    modern, legacy = os.environ.get("STUDIO_" + suffix), os.environ.get("F1_" + suffix)
    if modern is not None and legacy is not None and modern.strip() != legacy.strip():
        raise ValueError(CONFIGURATION_ERROR) from None
    return modern if modern is not None else legacy if legacy is not None else default


def local_api_url(value: str) -> tuple[str, str, int]:
    """Validate a trusted loopback HTTP base URL without echoing invalid input."""
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in LOCAL_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.port == 0
            or value != value.strip()
        ):
            raise ValueError
        port = parsed.port or 80
        return value.rstrip("/"), parsed.hostname, port
    except (TypeError, ValueError):
        raise ValueError(CONFIGURATION_ERROR) from None


@dataclass(frozen=True)
class LauncherSettings:
    """Snapshot the service's trusted shutdown settings before creating its process."""

    api_url: str
    host: str
    port: int
    shutdown_timeout_seconds: float

    @classmethod
    def from_environment(cls, api_url: str | None = None):
        """Validate URL aliases, bind aliases and finite Service/Runtime budgets."""
        modern, legacy = os.environ.get(STUDIO_JOB_API_URL_ENV), os.environ.get(LEGACY_STUDIO_JOB_API_URL_ENV)
        if modern is not None and legacy is not None and modern.strip() != legacy.strip():
            raise ValueError(CONFIGURATION_ERROR) from None
        resolved = modern if modern is not None else legacy if legacy is not None else DEFAULT_STUDIO_JOB_API_URL
        url, host, port = local_api_url(resolved if api_url is None else api_url)
        try:
            bind_host, bind_port = configuration_value("ENGINE_HOST"), configuration_value("ENGINE_PORT")
            if bind_host is not None and bind_host != host or bind_port is not None and int(bind_port) != port:
                raise ValueError
            budgets = []
            for suffix, default, maximum in (
                ("HTTP_DRAIN_SECONDS", "5", 30),
                ("SHUTDOWN_GRACE_SECONDS", "60", 3600),
                ("STOP_GRACE_SECONDS", "2", 30),
            ):
                value = float(configuration_value(suffix, default))
                if not math.isfinite(value) or not 0 <= value <= maximum:
                    raise ValueError
                budgets.append(value)
            return cls(url, host, port, sum(budgets) + RUNTIME_CLEANUP_SECONDS + SERVICE_EXIT_SECONDS)
        except (TypeError, ValueError):
            raise ValueError(CONFIGURATION_ERROR) from None


def api_is_ready(api_url: str) -> bool:
    """Require the existing compatible service identity and successful jobs access."""
    api_url, _, _ = local_api_url(api_url)
    try:
        # The local service must not be routed through ambient proxy settings or
        # follow an HTTP redirect to a different origin.
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(f"{api_url}/health", timeout=0.5, allow_redirects=False)
            payload = response.json()
            if response.status_code != 200 or not isinstance(payload, dict):
                return False
            if payload.get("status") != "ok" or payload.get("service") != SERVICE_NAME:
                return False
            jobs = session.get(f"{api_url}/api/v1/jobs", timeout=0.5, allow_redirects=False)
            listing = jobs.json()
            return jobs.status_code == 200 and isinstance(listing, dict) and isinstance(listing.get("jobs"), list)
    except (requests.RequestException, ValueError):
        return False


def start_api(api_url: str) -> subprocess.Popen:
    """Start one isolated local service, retaining its full outer shutdown budget."""
    settings = LauncherSettings.from_environment(api_url)
    environment = os.environ.copy()
    for prefix in ("STUDIO_", "F1_"):
        environment[prefix + "ENGINE_HOST"] = settings.host
        environment[prefix + "ENGINE_PORT"] = str(settings.port)
    process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "main_engine.py")],
        cwd=PROJECT_ROOT,
        env=environment,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )
    process.studio_shutdown_timeout_seconds = settings.shutdown_timeout_seconds
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Studio Service exited during startup with code {process.returncode}")
            if api_is_ready(settings.api_url) and process.poll() is None:
                return process
            time.sleep(POLL_SECONDS)
        raise RuntimeError(f"Studio Service did not become ready within {STARTUP_TIMEOUT_SECONDS:.0f} seconds")
    except BaseException:
        # Startup failure (including Ctrl+C) must not leak an owned child.
        stop_api(process)
        raise


def stop_api(process: subprocess.Popen | None) -> None:
    """Request cooperative shutdown and fail visibly on nonzero exit or forced kill."""
    if process is None:
        return  # An API discovered on the network is never an owned process.
    if process.poll() is None:
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
        except ProcessLookupError:
            pass  # It may have exited between poll and signal delivery.
        except OSError:
            process.kill()
            process.wait(timeout=SERVICE_EXIT_SECONDS)
            raise RuntimeError("Studio Service shutdown signal failed; forced termination was required") from None
        budget = getattr(process, "studio_shutdown_timeout_seconds", None)
        if budget is None:
            budget = LauncherSettings.from_environment().shutdown_timeout_seconds
        deadline = time.monotonic() + budget
        while True:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
                break
            except KeyboardInterrupt:
                # A repeated Ctrl+C must not abandon the owned service or shorten
                # the already captured cooperation and cleanup deadline.
                continue
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=SERVICE_EXIT_SECONDS)
                raise RuntimeError(
                    "Studio Service exceeded its shutdown budget; forced termination was required"
                ) from None
    if process.returncode != 0:
        raise RuntimeError(f"Studio Service exited with code {process.returncode}")


def run_webui(api_process: subprocess.Popen | None, *, inbrowser: bool = False) -> None:
    """Close Gradio on every exit path and notice a launcher-owned service dying."""
    from app import GlobalConfig, YOLO_Master_WebUI

    checkpoints_dir = PROJECT_ROOT / "ckpts"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    webui = YOLO_Master_WebUI(str(checkpoints_dir)).build_app()
    try:
        webui.launch(share=False, inbrowser=inbrowser, theme=GlobalConfig.THEME, prevent_thread_lock=True)
        print("Studio launcher ready", flush=True)
        while webui.is_running:
            if api_process is not None and api_process.poll() is not None:
                raise RuntimeError(f"Studio Service exited unexpectedly with code {api_process.returncode}")
            time.sleep(POLL_SECONDS)
    finally:
        webui.close()


@contextmanager
def loopback_proxy_bypass():
    """Keep Gradio's own local HTTP probes direct within this launcher process."""
    keys = ("NO_PROXY", "no_proxy")
    previous = {key: os.environ.get(key) for key in keys}
    entries = []
    for value in previous.values():
        for entry in (value or "").split(","):
            entry = entry.strip()
            if entry and entry not in entries:
                entries.append(entry)
    for host in ("127.0.0.1", "localhost", "::1"):
        if host not in {entry.lower() for entry in entries}:
            entries.append(host)
    for key in keys:
        os.environ[key] = ",".join(entries)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def launcher_signals():
    """Route terminal and service-manager signals through the same cleanup path."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signals = [signal.SIGINT, signal.SIGTERM]
    if os.name == "nt":
        signals.append(signal.SIGBREAK)
    previous = {sig: signal.signal(sig, interrupt) for sig in signals}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> int:
    """Reuse an existing local service or own one until Gradio closes or Ctrl+C."""
    with launcher_signals(), loopback_proxy_bypass():
        settings = LauncherSettings.from_environment()
        # The modern and compatibility clients now observe exactly the same URL.
        os.environ[STUDIO_JOB_API_URL_ENV] = settings.api_url
        os.environ[LEGACY_STUDIO_JOB_API_URL_ENV] = settings.api_url
        api_process = None
        try:
            if not api_is_ready(settings.api_url):
                api_process = start_api(settings.api_url)
            run_webui(api_process)
        except KeyboardInterrupt:
            pass
        finally:
            stop_api(api_process)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        # These launcher errors contain only fixed messages and exit codes.
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
