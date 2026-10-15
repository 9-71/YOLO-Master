"""Real loopback HTTP, external service signals, budget and cleanup-failure propagation."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil
import pytest

from tests.studio.test_service_api import INVALID_SERVICE_SETTINGS, payload, wait_for


def free_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


def signal_service(process):
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.send_signal(signal.SIGTERM)


def identities_gone(identities):
    for identity in identities:
        try:
            process = psutil.Process(identity["pid"])
            if process.create_time() == identity["created"]:
                return False  # Includes zombies: wait/reap must finish before acceptance.
        except psutil.NoSuchProcess:
            continue
    return True


@pytest.mark.parametrize("mode", ["cooperative", "fallback", "cleanup_failure", "rest"])
def test_real_service_signal_budget_and_cleanup(tmp_path, mode):
    port = free_port()
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(Path.cwd()), os.environ.get("PYTHONPATH", ""))),
        "YOLO_AUTOINSTALL": "false",
    }
    log = (tmp_path / "service-process.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "tests.studio.service_process_runner", str(tmp_path), str(port), mode],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    facts = {"mode": mode, "service_pid": process.pid}
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=3, trust_env=False) as client:

            def ready():
                assert process.poll() is None, (tmp_path / "service-process.log").read_text()
                try:
                    response = client.get("/health")
                    if response.status_code == 200:
                        assert response.json()["service"] == "YOLO-Master F1 Task Engine"
                        return True
                    return False
                except httpx.TransportError:
                    return False

            wait_for(ready, timeout=20)
            assert (
                client.get(
                    "/health", params={"password": "access-query-secret", "token": "access-query-token"}
                ).status_code
                == 200
            )
            request = payload(tmp_path, "shutdown-train", hold=mode == "cleanup_failure")
            if mode != "rest":
                request["task_type"] = "train"
            if mode == "fallback":
                request["params"]["ignore"] = True
            response = client.post("/api/v1/jobs", json=request)
            assert response.status_code == 201, response.text
            marker = "shutdown-train.ready" if mode in {"rest", "cleanup_failure"} else "started"
            if mode == "cleanup_failure":
                # Use the deterministic real worker that cannot cooperate in this failure case.
                marker = "started"
            wait_for(lambda: (tmp_path / marker).exists(), timeout=20)
            if mode == "rest":
                wait_for(lambda: client.get("/api/v1/jobs/shutdown-train").json()["status"] == "completed")
                manifest = client.get("/api/v1/jobs/shutdown-train/artifacts").json()
                assert client.get(manifest["artifacts"][0]["download_url"]).text == "authorized-result"
                assert client.get("/api/v1/jobs/shutdown-train/logs").json()["logs"]
            identities = client.get("/_test/identities").json()
            facts["identities"] = identities
            assert identities or mode == "rest"
            start = time.monotonic()
            signal_service(process)
            process.wait(timeout=35)
            elapsed = time.monotonic() - start
            facts.update(signal_to_exit_seconds=elapsed, exit_code=process.returncode)
            measured = json.loads((tmp_path / "service-facts.json").read_text())
            facts["service"] = measured
            assert elapsed < measured["runtime_budget"] + measured["http_drain_budget"] + 5
            assert identities_gone(identities), identities
            expected_code = 1 if mode == "cleanup_failure" else 0
            service_log = (tmp_path / "service-process.log").read_text()
            assert process.returncode == expected_code, service_log
            assert "access-query-secret" not in service_log and "access-query-token" not in service_log
            if mode == "cleanup_failure":
                assert measured["service_failure"] == "SHUTDOWN_FAILED" and measured["owner_retained"]
                assert measured["jobs"][0]["status"] == "running"
                assert measured["events"][-1]["event"] == "service.failure"
            else:
                assert measured["service_failure"] is None and not measured["owner_retained"]
                assert measured["events"][-1]["event"] == "manager.shutdown.return"
                assert measured["events"][-1]["owners"] == 0
                job = json.loads((tmp_path / "state.json").read_text())["jobs"]["shutdown-train"]
                if mode == "rest":
                    assert job["status"] == "completed"
                else:
                    assert job["status"] == "failed" and job["error"]["code"] == "SERVICE_SHUTDOWN"
                    assert not job["runtime_tracking"]["cancel_requested"]
                    if mode == "cooperative":
                        assert "weights/shutdown.pt" in job["output"]["artifacts"]
                        assert not any("SHUTDOWN_FORCED" in line for line in measured["logs"])
                        assert (tmp_path / "acknowledged").exists() and (tmp_path / "finalized").exists()
                    elif mode == "fallback":
                        assert any("SHUTDOWN_FORCED" in line for line in measured["logs"])
    finally:
        if process.poll() is None:
            signal_service(process)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log.close()
        (tmp_path / "probe-facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")


@pytest.mark.slow
@pytest.mark.studio_integration
def test_real_model_http_signal_checkpoint_and_resume(tmp_path):
    import torch
    import yaml
    from PIL import Image

    from ultralytics import YOLO

    model = Path(os.environ["STUDIO_TEST_MODEL"]).resolve(strict=True)
    for split in ("train", "val"):
        images, labels = tmp_path / "images" / split, tmp_path / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (64, 64), "white").save(images / f"{index}.jpg")
            (labels / f"{index}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    data = tmp_path / "data.yaml"
    data.write_text(
        yaml.safe_dump({"path": str(tmp_path), "train": "images/train", "val": "images/val", "names": {0: "object"}})
    )
    port = free_port()
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(Path.cwd()), os.environ.get("PYTHONPATH", ""))),
        "YOLO_AUTOINSTALL": "false",
        "STUDIO_TEST_MODEL": str(model),
    }
    log = (tmp_path / "service-process.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "tests.studio.service_process_runner", str(tmp_path), str(port), "real"],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    facts = {"service_pid": process.pid, "model": str(model), "device": "cpu", "amp": False, "workers": 1}
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5, trust_env=False) as client:

            def ready():
                assert process.poll() is None, (tmp_path / "service-process.log").read_text()
                try:
                    response = client.get("/health")
                    if response.status_code == 200:
                        assert response.json()["service"] == "YOLO-Master F1 Task Engine"
                        return True
                    return False
                except httpx.TransportError:
                    return False

            wait_for(ready, timeout=30)
            request = payload(
                tmp_path,
                "shutdown-train",
                model_path=str(model),
                data_source=str(data),
                epochs=10,
                batch_size=2,
                imgsz=64,
            )
            request["task_type"] = "train"
            request["runtime_tracking"]["timeout_seconds"] = 240
            response = client.post("/api/v1/jobs", json=request)
            assert response.status_code == 201, response.text
            wait_for(lambda: (tmp_path / "training.started").exists(), timeout=120)
            identities = client.get("/_test/identities").json()
            facts["identities"] = identities
            started = time.monotonic()
            signal_service(process)
            process.wait(timeout=150)
            facts["signal_to_exit_seconds"] = time.monotonic() - started
            facts["exit_code"] = process.returncode
            assert process.returncode == 0, (tmp_path / "service-process.log").read_text()
            measured = json.loads((tmp_path / "service-facts.json").read_text())
            facts["service"] = measured
            assert facts["signal_to_exit_seconds"] < measured["runtime_budget"] + measured["http_drain_budget"] + 5
            assert not measured["owner_retained"] and identities_gone(identities)
            assert not any("SHUTDOWN_FORCED" in line for line in measured["logs"])
            job = json.loads((tmp_path / "state.json").read_text())["jobs"]["shutdown-train"]
            assert job["status"] == "failed" and job["error"]["code"] == "SERVICE_SHUTDOWN"
            assert "weights/shutdown.pt" in job["output"]["artifacts"]
            checkpoint = tmp_path / "shutdown-train" / "weights" / "shutdown.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert state["model"] is not None and state["optimizer"] is not None
            assert state["scaler"] is not None and state["ema"] is not None
            resumed = YOLO(str(checkpoint))
            epochs_seen = []
            resumed.add_callback("on_train_epoch_end", lambda trainer: epochs_seen.append(trainer.epoch))
            resumed.train(
                resume=True, epochs=state["epoch"] + 2, device="cpu", workers=0, amp=False, plots=False, val=False
            )
            assert epochs_seen == [state["epoch"] + 1]
            facts.update(checkpoint_epoch=state["epoch"], resumed_epochs=epochs_seen)
            # A new Service lifespan uses durable history, preserving shutdown terminal.
            from main_engine import create_app
            from tests.studio.test_service_api import client_for, manager_for

            with client_for(create_app(manager_factory=lambda: manager_for(tmp_path))) as restarted:
                response = restarted.get("/api/v1/jobs/shutdown-train").json()
                assert response["status"] == "failed" and response["error_code"] == "SERVICE_SHUTDOWN"
    finally:
        if process.poll() is None:
            signal_service(process)
            try:
                process.wait(timeout=145)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log.close()
        (tmp_path / "probe-facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")


@pytest.mark.parametrize("key,value", INVALID_SERVICE_SETTINGS)
def test_invalid_configuration_script_exits_without_input_or_traceback(key, value):
    env = dict(os.environ)
    for name in {item[0] for item in INVALID_SERVICE_SETTINGS}:
        env.pop(name, None)
    env[key] = value
    result = subprocess.run(
        [sys.executable, "-B", "main_engine.py"], env=env, capture_output=True, text=True, timeout=15, check=False
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip() == "Invalid Studio Service configuration"
