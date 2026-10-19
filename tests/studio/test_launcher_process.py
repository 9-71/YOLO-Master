"""Unmodified production service started and closed through the launcher helpers."""

from __future__ import annotations

import json
import os
import socket
import time

import pytest
import requests

import start_studio as launcher


@pytest.fixture
def local_service(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("STUDIO_", "F1_")):
            monkeypatch.delenv(name)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
    for suffix, value in {
        "API_URL": f"http://127.0.0.1:{port}",
        "JOBS_STATE_PATH": str(tmp_path / "state.json"),
        "MODEL_ROOTS": str(tmp_path),
        "DATA_ROOTS": str(tmp_path),
        "OUTPUT_ROOT": str(tmp_path),
        "HTTP_DRAIN_SECONDS": "1",
        "SHUTDOWN_GRACE_SECONDS": "2",
        "STOP_GRACE_SECONDS": "0.2",
    }.items():
        monkeypatch.setenv("STUDIO_" + suffix, value)
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    return tmp_path, f"http://127.0.0.1:{port}"


def test_launcher_owned_production_service_idle_shutdown_and_external_reuse(local_service, monkeypatch):
    tmp_path, api_url = local_service
    started = time.monotonic()
    process = launcher.start_api(api_url)
    facts = {
        "entry": "start_studio.start_api",
        "service_pid": process.pid,
        "startup_seconds": time.monotonic() - started,
    }
    try:
        assert launcher.api_is_ready(api_url)
        assert process.studio_shutdown_timeout_seconds == pytest.approx(24.2)
        with requests.Session() as session:
            session.trust_env = False
            assert session.get(api_url + "/api/v1/jobs", timeout=3).json()["jobs"] == []
        seen = []
        monkeypatch.setattr(launcher, "run_webui", lambda owned: seen.append(owned))
        # The process is external to this new main() invocation. Closing that UI
        # must leave the already running service alive.
        assert launcher.main() == 0
        assert seen == [None] and process.poll() is None
        facts["external_reuse_survived_ui_close"] = True
    finally:
        closing = time.monotonic()
        launcher.stop_api(process)
        facts.update(shutdown_seconds=time.monotonic() - closing, exit_code=process.returncode)
        (tmp_path / "launcher-process-facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    assert process.returncode == 0
    assert not launcher.api_is_ready(api_url)


def test_production_startup_failure_is_nonzero_and_has_no_owned_process_left(local_service, monkeypatch):
    tmp_path, api_url = local_service
    # An existing corrupt durable store is a real production startup failure;
    # the launcher does not replace it or invent a new owner.
    (tmp_path / "state.json").write_text("corrupt", encoding="utf-8")
    popen = launcher.subprocess.Popen
    processes = []

    def track(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(launcher.subprocess, "Popen", track)
    with pytest.raises(RuntimeError, match="exited.*code [1-9]"):
        launcher.start_api(api_url)
    assert len(processes) == 1 and processes[0].poll() not in {None, 0}
    assert not launcher.api_is_ready(api_url)
