"""Windows console-backed Service regressions; local model acceptance remains explicit."""

from __future__ import annotations

import ctypes
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import httpx
import psutil
import pytest

from core.schema import JobStatus
from main_engine import ServiceSettings, create_app, run_service
from studio.jobs_manager import JobsManager
from tests.studio.test_service_api import payload, wait_for
from tests.studio.test_service_process import free_port, identities_gone

# Imported during spawn, before Worker detaches or redirects Python streams.
CONSOLE_LOGGER = logging.getLogger("studio.console-probe")
CONSOLE_LOGGER.addHandler(logging.StreamHandler(sys.stdout))
CONSOLE_LOGGER.setLevel(logging.INFO)
CONSOLE_LOGGER.propagate = False


def console_process(command, env):
    """Give Service a real, hidden Windows console that its spawned worker inherits."""
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    return subprocess.Popen(
        command,
        env=env,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        startupinfo=startup,
    )


def break_console(process):
    """Send actual CTRL_BREAK from a short-lived sender attached only to this Service."""
    sender = (
        "import ctypes,sys,time; from ctypes import wintypes; "
        "k=ctypes.WinDLL('kernel32',use_last_error=True); "
        "k.FreeConsole(); assert k.AttachConsole(int(sys.argv[1])); "
        "handler=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.DWORD)(lambda event: True); "
        "assert k.SetConsoleCtrlHandler(handler,True); "
        "assert k.GenerateConsoleCtrlEvent(1,0); time.sleep(0.2)"
    )
    subprocess.run(
        [sys.executable, "-c", sender, str(process.pid)],
        check=True,
        timeout=5,
        creationflags=subprocess.CREATE_NO_WINDOW,
        capture_output=True,
    )


def stdio_executor(job):
    """Exercise print, stderr, Unicode, partial flush and genuine failure over owned IPC."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    root = Path(job.output.output_dir)
    identity = {"pid": os.getpid(), "created": psutil.Process().create_time(), "console_cp": kernel32.GetConsoleCP()}
    (root / "worker.json").write_text(json.dumps(identity), encoding="utf-8")
    assert identity["console_cp"] == 0, "Worker must remain detached from Service CTRL_BREAK"
    print("Worker 输出正常 password=stdout-probe-secret", flush=True)
    print("Worker 错误流正常 token=stderr-probe-secret", file=sys.stderr, flush=True)
    CONSOLE_LOGGER.info("prebound-logger password=logger-probe-secret")
    sys.stdout.write("password=")
    sys.stdout.flush()
    sys.stdout.write("split-probe-secret\n")
    sys.stdout.write("password=")
    sys.stdout.flush()
    sys.stderr.write("interleaved-error正常\n")
    sys.stdout.write("interleaved-probe-secret\n")
    sys.stdout.write("http://url-probe-user:")
    sys.stdout.flush()
    sys.stdout.write("url-probe-secret@localhost/example\n")
    os.environ["STUDIO_CONSOLE_PRIVATE_SECRET"] = "password=multiline-alpha-probe\nmultiline-beta-probe"
    sys.stdout.write("password=multiline-alpha-probe\n")
    sys.stdout.flush()
    sys.stdout.write("multiline-beta-probe\n")
    sys.stdout.write("partial-output")
    sys.stdout.flush()
    sys.stderr.write("partial-error")
    sys.stderr.flush()
    sys.stdout.write("unflushed-tail")
    os.environ["STUDIO_CONSOLE_TAIL_SECRET"] = "tail-alpha-probe\ntail-beta-probe"
    sys.stdout.write("\ntail-alpha-probe\ntail-beta")
    sys.stdout.flush()
    # Keep the Worker observable until the HTTP test releases it.
    (root / "printed").touch()
    wait_for(lambda: (root / "release").exists())
    if job.params.get("failure"):
        raise OSError(6, f"句柄无效 password=exception-probe-secret path={root}")
    job.status = JobStatus.COMPLETED
    return job


def console_service(root, port):
    """Test-only Service with production owner/IPC and explicit stdout probe executor."""
    assert ctypes.WinDLL("kernel32").GetConsoleCP() != 0

    def factory():
        manager = JobsManager(
            storage_path=root / "state.json",
            output_root=root,
            model_roots=[root],
            data_roots=[root],
            stop_grace_seconds=0.1,
            shutdown_grace_seconds=1,
        )
        manager._worker_executor = stdio_executor
        return manager

    run_service(create_app(settings=ServiceSettings(port=port, http_drain_seconds=0.2), manager_factory=factory))


@pytest.mark.parametrize(
    ("secrets", "chunks"),
    [
        (["review-alpha\nreview-beta"], ["review-alpha\nreview-beta\n"]),
        (["review-alpha\nreview-beta"], ["review-alpha\n", "review-beta\n"]),
        (["password=review-alpha\nreview-beta"], ["password=review-alpha\n", "review-beta\n"]),
        (["ab\nab"], ["ab\nab"]),
        (["ab\nab"], ["ab\nab\nab\n"]),
        (["review-alpha\nreview-beta"], ["review-alpha\nreview-be"]),
        (
            ["review-alpha\nreview-beta", "review-alpha\nreview-beta\nreview-gamma"],
            ["review-alpha\nreview-beta\nreview-ga"],
        ),
    ],
)
def test_output_stream_known_multiline_context(tmp_path, monkeypatch, secrets, chunks):
    """Keep complete/partial overlapping known credentials out of persisted logs."""
    from core.schema import JobRequest
    from studio.worker_runtime import _JobLogStream

    for index, secret in enumerate(secrets):
        monkeypatch.setenv(f"STUDIO_PROBE_{index}_SECRET", secret)
    job = JobRequest.model_validate(payload(tmp_path))
    output = _JobLogStream(job)
    output.write("ordinary prefix\n")
    for chunk in chunks:
        output.write(chunk)
        output.flush()
    output.finish()
    before = list(job.logs)
    output.finish()
    assert job.logs == before
    text = "\n".join(job.logs)
    assert "ordinary prefix" in text and "***REDACTED***" in text
    assert not any(part in text for secret in secrets for part in secret.splitlines() if len(part) >= 4)
    assert "review-be" not in text and "review-ga" not in text


def test_output_stream_interleaved_credentials(tmp_path):
    """A stderr newline must not truncate stdout's split credential context."""
    from core.schema import JobRequest
    from studio.worker_runtime import _JobLogStream

    job = JobRequest.model_validate(payload(tmp_path))
    output = _JobLogStream(job)
    error_output = _JobLogStream(job, output._lock)
    output.write("password=")
    output.flush()
    error_output.write("unrelated diagnostic\n")
    output.write("review-interleaved-secret\n")
    output.finish()
    error_output.finish()
    text = "\n".join(job.logs)
    assert "unrelated diagnostic" in text
    assert "password=***REDACTED***" in text and "review-interleaved-secret" not in text


@pytest.mark.skipif(os.name != "nt", reason="Windows console handles and CTRL_BREAK")
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("stream_logs", [True, False])
def test_console_worker_output_and_failure(tmp_path, failure, stream_logs):
    port = free_port()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "YOLO_AUTOINSTALL": "false"}
    log_path = tmp_path / "service.log"
    with log_path.open("w", encoding="utf-8"):
        process = console_process(
            [sys.executable, "-m", "tests.studio.test_worker_console", str(tmp_path), str(port)],
            env,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5, trust_env=False) as client:

                def ready():
                    assert process.poll() is None, log_path.read_text(encoding="utf-8")
                    try:
                        return client.get("/health").status_code == 200
                    except httpx.TransportError:
                        return False

                wait_for(ready)
                request = payload(tmp_path, failure=failure)
                request["runtime_tracking"]["stream_logs"] = stream_logs
                response = client.post("/api/v1/jobs", json=request)
                assert response.status_code == 201, response.text

                def printed():
                    detail = client.get("/api/v1/jobs/service-job").json()
                    assert detail["status"] not in {"failed", "cancelled"}, detail
                    return (tmp_path / "printed").exists()

                wait_for(printed)
                identity = json.loads((tmp_path / "worker.json").read_text(encoding="utf-8"))
                assert identity["console_cp"] == 0
                (tmp_path / "release").touch()
                expected = "failed" if failure else "completed"
                wait_for(lambda: client.get("/api/v1/jobs/service-job").json()["status"] == expected)
                detail = client.get("/api/v1/jobs/service-job").json()
                text = "\n".join(client.get("/api/v1/jobs/service-job/logs").json()["logs"])
                assert "Worker 输出正常" in text and "Worker 错误流正常" in text
                assert "http://***REDACTED***@localhost/example" in text
                assert "prebound-logger password=***REDACTED***" in text
                assert "interleaved-error正常" in text
                assert "partial-output" in text and "partial-error" in text and "unflushed-tail" in text
                assert "\ufffd" not in text
                assert not any(
                    secret in text
                    for secret in (
                        "stdout-probe-secret",
                        "stderr-probe-secret",
                        "exception-probe-secret",
                        "logger-probe-secret",
                        "split-probe-secret",
                        "interleaved-probe-secret",
                        "multiline-alpha-probe",
                        "multiline-beta-probe",
                        "tail-alpha-probe",
                        "tail-beta",
                        "url-probe-user",
                        "url-probe-secret",
                    )
                )
                if failure:
                    assert detail["error_code"] == "EXECUTION_FAILED"
                    assert "Traceback (most recent call last)" in text and "OSError" in text and "句柄无效" in text
                    assert str(tmp_path) in text  # Existing contract masks credentials, not all local paths.
                assert identities_gone([identity])
                saved = (tmp_path / "state.json").read_text(encoding="utf-8")
                assert not any(
                    secret in saved
                    for secret in (
                        "stdout-probe-secret",
                        "stderr-probe-secret",
                        "exception-probe-secret",
                        "logger-probe-secret",
                        "split-probe-secret",
                        "interleaved-probe-secret",
                        "multiline-alpha-probe",
                        "multiline-beta-probe",
                        "tail-alpha-probe",
                        "tail-beta",
                        "url-probe-user",
                        "url-probe-secret",
                    )
                )
            break_console(process)
            process.wait(timeout=25)
            assert process.returncode == 0, log_path.read_text(encoding="utf-8")
        finally:
            (tmp_path / "release").touch()
            if process.poll() is None:
                break_console(process)
                process.wait(timeout=25)


@pytest.mark.skipif(os.name != "nt", reason="Windows console-backed production Service")
@pytest.mark.slow
@pytest.mark.studio_integration
def test_console_production_service_real_predict(tmp_path):
    from PIL import Image

    model = Path(os.environ["STUDIO_TEST_MODEL"]).resolve(strict=True)
    image = tmp_path / "input.png"
    Image.new("RGB", (64, 64), "white").save(image)
    port = free_port()
    env = {
        **os.environ,
        "STUDIO_ENGINE_PORT": str(port),
        "STUDIO_MODEL_ROOTS": str(model.parent),
        "STUDIO_DATA_ROOTS": str(tmp_path),
        "STUDIO_OUTPUT_ROOT": str(tmp_path),
        "STUDIO_JOBS_STATE_PATH": str(tmp_path / "state.json"),
        "STUDIO_CPU_CONCURRENCY": "1",
        "YOLO_AUTOINSTALL": "false",
        "YOLO_OFFLINE": "true",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    log_path = tmp_path / "service.log"
    identities, service_children = [], []
    with log_path.open("w", encoding="utf-8"):
        process = console_process([sys.executable, "-m", "main_engine"], env)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5, trust_env=False) as client:

                def ready():
                    assert process.poll() is None, log_path.read_text(encoding="utf-8")
                    try:
                        return client.get("/health").status_code == 200
                    except httpx.TransportError:
                        return False

                wait_for(ready)
                response = client.post(
                    "/api/v1/jobs",
                    json=payload(
                        tmp_path,
                        "real-predict",
                        model_path=str(model),
                        data_source=str(image),
                        imgsz=64,
                    ),
                )
                assert response.status_code == 201, response.text

                def finished():
                    for child in psutil.Process(process.pid).children(recursive=True):
                        try:
                            command = child.cmdline()
                            identity = {
                                "pid": child.pid,
                                "created": child.create_time(),
                                "name": child.name(),
                                "cmdline": command,
                            }
                        except psutil.NoSuchProcess:
                            continue  # The observed process exited between enumeration and identity sampling.
                        if identity not in service_children:
                            service_children.append(identity)
                        if "multiprocessing.spawn" in " ".join(command) and identity not in identities:
                            identities.append(identity)
                    detail = client.get("/api/v1/jobs/real-predict").json()
                    assert detail["status"] not in {"failed", "cancelled"}, detail
                    return detail["status"] == "completed"

                wait_for(finished, timeout=120)
                manifest = client.get("/api/v1/jobs/real-predict/artifacts").json()["artifacts"]
                assert manifest and client.get(manifest[0]["download_url"]).status_code == 200
                logs = client.get("/api/v1/jobs/real-predict/logs").json()["logs"]
                assert any("Resolved handler: PredictHandler" in line for line in logs)
                assert not any("WinError 6" in line or "\ufffd" in line for line in logs)
                assert identities and identities_gone(identities)
                (tmp_path / "predict-facts.json").write_text(
                    json.dumps(
                        {"identities": identities, "manifest": manifest, "logs": logs}, indent=2, ensure_ascii=False
                    ),
                    encoding="utf-8",
                )
            break_console(process)
            process.wait(timeout=25)
            assert process.returncode == 0, log_path.read_text(encoding="utf-8")
            # conhost is Service's console server, outside the owned Worker tree.
            # Its teardown is observed separately after Service process exit.
            wait_for(lambda: identities_gone(service_children), timeout=5)
        finally:
            if process.poll() is None:
                break_console(process)
                process.wait(timeout=25)


@pytest.mark.skipif(os.name != "nt", reason="Windows console CTRL_BREAK checkpoint isolation")
@pytest.mark.slow
@pytest.mark.studio_integration
def test_console_training_checkpoint_and_resume(tmp_path, monkeypatch):
    """Run the existing real model checkpoint/resume Gate with actual console output."""
    from tests.studio import test_service_process as service

    original_popen = subprocess.Popen

    def hidden_service(*args, **kwargs):
        if "tests.studio.service_process_runner" in args[0]:
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
            kwargs.update(creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup, stdout=None, stderr=None)
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", hidden_service)
    monkeypatch.setattr(service, "signal_service", break_console)
    service.test_real_model_http_signal_checkpoint_and_resume(tmp_path)
    measured = json.loads((tmp_path / "service-facts.json").read_text(encoding="utf-8"))
    assert not any("WinError 6" in line or "--- Logging error ---" in line for line in measured["logs"])


if __name__ == "__main__":
    console_service(Path(sys.argv[1]), int(sys.argv[2]))
