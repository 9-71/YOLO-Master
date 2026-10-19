"""Actual combined Gradio launcher checks, outside the service-only Core lane."""

from __future__ import annotations

import json
import os
import socket
import time

import pytest
import requests

import start_studio as launcher
from tests.studio import test_launcher_process as service_tests

# Register the same service-only fixture in the optional GUI test module.
local_service = service_tests.local_service


def test_combined_launcher_ignores_broken_proxy_for_gradio_loopback(local_service, monkeypatch):
    """Gradio's own ambient HTTP probes must reach the two local servers directly."""
    import signal
    import subprocess
    import sys

    import psutil

    tmp_path, api_url = local_service
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        ui_port = server.getsockname()[1]
    ui_url = f"http://127.0.0.1:{ui_port}"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(key, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("GRADIO_SERVER_PORT", str(ui_port))
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "false")
    log_path = tmp_path / "combined-proxy-launcher.log"
    facts = {
        "entry": "python start_studio.py",
        "broken_proxy": "http://127.0.0.1:1",
        "api_url": api_url,
        "ui_url": ui_url,
    }
    identities = []
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-X", "utf8", "start_studio.py"],
        cwd=launcher.PROJECT_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=dict(os.environ),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    try:
        with requests.Session() as session:
            session.trust_env = False
            deadline = time.monotonic() + 35
            while True:
                assert process.poll() is None, log_path.read_text(encoding="utf-8")
                try:
                    api = session.get(api_url + "/api/v1/jobs", timeout=0.3)
                    ui = session.get(ui_url + "/config", timeout=0.3)
                    if api.status_code == ui.status_code == 200 and "Studio launcher ready" in log_path.read_text(
                        encoding="utf-8"
                    ):
                        assert api.json()["jobs"] == []
                        assert ui.json()["components"]
                        break
                except requests.RequestException:
                    pass
                assert time.monotonic() < deadline, log_path.read_text(encoding="utf-8")
                time.sleep(0.1)
            assert process.poll() is None, log_path.read_text(encoding="utf-8")
            identities = [
                {"pid": p.pid, "created": p.create_time()} for p in psutil.Process(process.pid).children(recursive=True)
            ]
            assert identities
            facts.update(api_ready=True, gradio_config_ready=True, launch_returned=True, identities=identities)
            closing = time.monotonic()
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
            process.wait(timeout=35)
            facts.update(shutdown_seconds=time.monotonic() - closing, exit_code=process.returncode)
            assert process.returncode == 0, log_path.read_text(encoding="utf-8")
            for identity in identities:
                try:
                    descendant = psutil.Process(identity["pid"])
                    assert descendant.create_time() != identity["created"], identity
                except psutil.NoSuchProcess:
                    pass
            assert not launcher.api_is_ready(api_url)
            with pytest.raises(requests.RequestException):
                session.get(ui_url + "/config", timeout=0.3)
            facts["owned_descendants_gone"] = True
    finally:
        if process.poll() is None:
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
            try:
                process.wait(timeout=35)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        facts.setdefault("exit_code", process.returncode)
        log.close()
        (tmp_path / "combined-proxy-facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
