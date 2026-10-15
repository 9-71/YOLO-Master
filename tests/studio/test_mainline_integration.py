"""Offline real-model HTTP acceptance through the unmodified production factory."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil
import pytest

from f1.ui.studio_jobs_client import StudioJobsApiClient
from tests.studio.test_mainline_compatibility import SUFFIXES
from tests.studio.test_service_api import payload, wait_for
from tests.studio.test_service_process import free_port, identities_gone, signal_service


@pytest.mark.slow
@pytest.mark.studio_integration
def test_production_five_tasks_legacy_client_checkpoint_resume_restart(tmp_path):
    import torch
    import yaml
    from PIL import Image

    from ultralytics import YOLO

    source_model = Path(os.environ["STUDIO_TEST_MODEL"]).resolve(strict=True)
    model = tmp_path / source_model.name
    shutil.copy2(source_model, model)
    for split in ("train", "val"):
        images, labels = tmp_path / "images" / split, tmp_path / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (64, 64), "white").save(images / f"{index}.jpg")
            (labels / f"{index}.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    data.write_text(
        yaml.safe_dump({"path": str(tmp_path), "train": "images/train", "val": "images/val", "names": {0: "object"}}),
        encoding="utf-8",
    )
    image = tmp_path / "images" / "val" / "0.jpg"
    port = free_port()
    env = dict(os.environ)
    for suffix in SUFFIXES:
        env.pop("F1_" + suffix, None)
        env.pop("STUDIO_" + suffix, None)
    env.update(
        {
            "F1_ENGINE_PORT": str(port),
            "F1_JOBS_STATE_PATH": str(tmp_path / "state.json"),
            "F1_MODEL_ROOTS": str(tmp_path),
            "F1_DATA_ROOTS": str(tmp_path),
            "F1_OUTPUT_ROOT": str(tmp_path),
            "F1_CPU_CONCURRENCY": "1",
            "F1_GPU_CONCURRENCY": "1",
            "STUDIO_SHUTDOWN_GRACE_SECONDS": "120",
            "YOLO_AUTOINSTALL": "false",
            "YOLO_OFFLINE": "true",
        }
    )
    log = (tmp_path / "production.log").open("w", encoding="utf-8")

    def launch():
        return subprocess.Popen(
            [sys.executable, "main_engine.py"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )

    process = launch()
    facts = {
        "entry": "python main_engine.py",
        "factory": "production_manager",
        "model": str(source_model),
        "device": "cpu",
        "tasks": {},
    }
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5, trust_env=False) as client:

            def ready():
                assert process.poll() is None, (tmp_path / "production.log").read_text(encoding="utf-8")
                try:
                    return client.get("/health").status_code == 200
                except httpx.TransportError:
                    return False

            wait_for(ready, timeout=30)
            legacy_client = StudioJobsApiClient(f"http://127.0.0.1:{port}")
            predict_id, _ = legacy_client.submit_job(
                "predict", str(model), str(image), str(tmp_path), 0.25, "cpu", [str(tmp_path)]
            )

            def terminal(job_id):
                detail = client.get("/api/v1/jobs/" + job_id).json()
                return detail if detail.get("status") in {"completed", "failed", "cancelled"} else False

            detail = wait_for(lambda: terminal(predict_id), timeout=120)
            assert detail["status"] == "completed", detail
            assert legacy_client.get_job_status(predict_id)["status"] == "COMPLETED"
            assert legacy_client.get_job_logs(predict_id)
            metadata = legacy_client.get_job_artifact_metadata(predict_id)
            assert metadata and any(Path(item["preview_path"]).is_file() for item in metadata if item["is_image"])
            facts["tasks"]["predict"] = detail
            assert client.post(f"/api/v1/jobs/{predict_id}/cancel").status_code == 200
            for task, params in (
                ("diagnose", {}),
                ("val", {"model_path": str(model), "data_source": str(data), "imgsz": 64, "batch_size": 2}),
                ("export", {"model_path": str(model), "format": "torchscript", "imgsz": 64}),
                (
                    "train",
                    {"model_path": str(model), "data_source": str(data), "epochs": 1, "imgsz": 64, "batch_size": 2},
                ),
            ):
                request = payload(tmp_path, "production-" + task, **params)
                request["task_type"] = task
                request["runtime_tracking"]["timeout_seconds"] = 300
                assert client.post("/api/v1/jobs", json=request).status_code == 201
                detail = wait_for(lambda job_id=request["job_id"]: terminal(job_id), timeout=240)
                assert detail["status"] == "completed", detail
                manifest = client.get(f"/api/v1/jobs/{request['job_id']}/artifacts").json()
                assert manifest["artifacts"], (task, detail)
                artifact = client.get(manifest["artifacts"][0]["download_url"])
                assert artifact.status_code == 200 and artifact.content
                assert client.get(f"/api/v1/jobs/{request['job_id']}/logs").json()["logs"]
                facts["tasks"][task] = {"detail": detail, "manifest": manifest}
            trained = torch.load(
                tmp_path / "production-train" / "weights" / "last.pt", map_location="cpu", weights_only=False
            )
            assert trained["train_args"]["imgsz"] == 64
            facts["effective_training"] = {
                key: trained["train_args"].get(key) for key in ("imgsz", "workers", "amp", "device")
            }
            cancellation = payload(
                tmp_path,
                "production-cancel",
                model_path=str(model),
                data_source=str(data),
                epochs=1000,
                imgsz=64,
                batch_size=2,
            )
            cancellation["task_type"] = "train"
            cancellation["runtime_tracking"]["timeout_seconds"] = 300
            assert client.post("/api/v1/jobs", json=cancellation).status_code == 201
            wait_for(
                lambda: any(
                    "Starting training for" in line
                    for line in client.get("/api/v1/jobs/production-cancel/logs").json()["logs"]
                ),
                timeout=120,
            )
            cancel_children = psutil.Process(process.pid).children(recursive=True)
            cancel_identities = [{"pid": child.pid, "created": child.create_time()} for child in cancel_children]
            assert cancel_identities
            assert client.get("/api/v1/jobs/production-cancel").json()["status"] == "running"
            assert client.post("/api/v1/jobs/production-cancel/cancel").status_code == 202
            cancelled = wait_for(lambda: terminal("production-cancel"), timeout=30)
            assert cancelled["status"] == "cancelled" and cancelled["error_code"] == "USER_CANCELLED"
            assert identities_gone(cancel_identities)
            assert client.post("/api/v1/jobs/production-cancel/cancel").status_code == 200
            facts["cancellation"] = {"detail": cancelled, "identities": cancel_identities, "identities_gone": True}
            request = payload(
                tmp_path,
                "production-shutdown",
                model_path=str(model),
                data_source=str(data),
                epochs=1000,
                imgsz=64,
                batch_size=2,
            )
            request["task_type"] = "train"
            request["runtime_tracking"]["timeout_seconds"] = 300
            assert client.post("/api/v1/jobs", json=request).status_code == 201

            def training_started():
                logs = client.get("/api/v1/jobs/production-shutdown/logs").json()["logs"]
                return any("Starting training for" in line for line in logs)

            wait_for(training_started, timeout=120)
            root = psutil.Process(process.pid)
            identities = [{"pid": p.pid, "created": p.create_time()} for p in root.children(recursive=True)]
            assert identities
            started = time.monotonic()
            signal_service(process)
            process.wait(timeout=150)
            facts["shutdown"] = {
                "signal_to_exit_seconds": time.monotonic() - started,
                "exit_code": process.returncode,
                "identities": identities,
            }
            assert process.returncode == 0
            assert identities_gone(identities)
            snapshot = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["jobs"]["production-shutdown"]
            assert snapshot["status"] == "failed" and snapshot["error"]["code"] == "SERVICE_SHUTDOWN"
            assert not snapshot["runtime_tracking"]["cancel_requested"]
            assert "weights/shutdown.pt" in snapshot["output"]["artifacts"]
            checkpoint = tmp_path / "production-shutdown" / "weights" / "shutdown.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert state["model"] is not None and state["optimizer"] is not None and state["ema"] is not None
            resumed = YOLO(str(checkpoint))
            epochs = []
            resumed.add_callback("on_train_epoch_end", lambda trainer: epochs.append(trainer.epoch))
            resumed.train(
                resume=True, epochs=state["epoch"] + 2, device="cpu", workers=0, amp=False, plots=False, val=False
            )
            assert epochs == [state["epoch"] + 1]
            facts["shutdown"].update(checkpoint_epoch=state["epoch"], resumed_epochs=epochs, identities_gone=True)
            process = launch()
            wait_for(ready, timeout=30)
            detail = client.get("/api/v1/jobs/production-shutdown").json()
            assert detail["status"] == "failed" and detail["error_code"] == "SERVICE_SHUTDOWN"
            assert client.get("/api/v1/jobs").json()["total"] == 7
            facts["restart"] = detail
            signal_service(process)
            process.wait(timeout=30)
            assert process.returncode == 0
    finally:
        if process.poll() is None:
            signal_service(process)
            try:
                process.wait(timeout=150)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log.close()
        (tmp_path / "production-facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
