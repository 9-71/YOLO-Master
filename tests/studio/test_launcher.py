"""Launcher settings, process ownership and UI cleanup without backend imports."""

from __future__ import annotations

import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import start_studio as launcher


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    for name in list(launcher.os.environ):
        if name.startswith(("STUDIO_", "F1_")):
            monkeypatch.delenv(name)


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8000",
        "http://example.com:8000",
        "http://127.0.0.1:8000/path",
        "http://user:password-secret@localhost:8000",
        "http://localhost:8000?token=secret",
        "http://localhost:8000#secret",
        "http://localhost:0",
        "http://localhost:99999",
        "http://localhost:secret-port",
        " http://localhost:8000",
    ],
)
def test_local_url_rejects_unsafe_or_malformed_input_without_echo(url):
    with pytest.raises(ValueError, match="^Invalid Studio launcher configuration$") as caught:
        launcher.local_api_url(url)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
def test_local_url_and_actual_service_shutdown_budget(monkeypatch, host):
    url = f"http://{host}:8123/"
    monkeypatch.setenv("STUDIO_API_URL", url)
    monkeypatch.setenv("F1_STUDIO_API_URL", url)
    monkeypatch.setenv("STUDIO_HTTP_DRAIN_SECONDS", "7")
    monkeypatch.setenv("F1_SHUTDOWN_GRACE_SECONDS", "120")
    monkeypatch.setenv("STUDIO_STOP_GRACE_SECONDS", "3")
    settings = launcher.LauncherSettings.from_environment()
    assert settings.api_url == url.rstrip("/")
    assert settings.host == host.strip("[]") and settings.port == 8123
    assert settings.shutdown_timeout_seconds == 7 + 120 + 3 + 16 + 5
    assert launcher.local_api_url("http://localhost")[2] == 80


@pytest.mark.parametrize(
    "environment",
    [
        {"STUDIO_API_URL": "http://127.0.0.1:8000", "F1_STUDIO_API_URL": "http://127.0.0.1:8001"},
        {"STUDIO_ENGINE_HOST": "localhost", "F1_ENGINE_HOST": "127.0.0.1"},
        {"STUDIO_ENGINE_PORT": "8000", "F1_ENGINE_PORT": "8001"},
        {"STUDIO_HTTP_DRAIN_SECONDS": "5", "F1_HTTP_DRAIN_SECONDS": "6"},
        {"STUDIO_SHUTDOWN_GRACE_SECONDS": "60", "F1_SHUTDOWN_GRACE_SECONDS": "61"},
        {"STUDIO_STOP_GRACE_SECONDS": "2", "F1_STOP_GRACE_SECONDS": "3"},
        {"STUDIO_ENGINE_PORT": "8123"},
        {"STUDIO_ENGINE_HOST": "localhost"},
    ],
)
def test_conflicting_aliases_or_bind_target_fail_before_spawn(monkeypatch, environment):
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    popen = Mock()
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    with pytest.raises(ValueError, match="^Invalid Studio launcher configuration$"):
        launcher.start_api("http://127.0.0.1:8000")
    popen.assert_not_called()


@pytest.mark.parametrize(
    "suffix,maximum", [("HTTP_DRAIN_SECONDS", 30), ("STOP_GRACE_SECONDS", 30), ("SHUTDOWN_GRACE_SECONDS", 3600)]
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1", "password=invalid-secret", "too-high"])
def test_nonfinite_or_out_of_range_shutdown_budgets_fail_closed(monkeypatch, suffix, maximum, value):
    monkeypatch.setenv("STUDIO_" + suffix, str(maximum + 1) if value == "too-high" else value)
    with pytest.raises(ValueError, match="^Invalid Studio launcher configuration$") as caught:
        launcher.LauncherSettings.from_environment()
    assert caught.value.__suppress_context__


def session_for(responses):
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    session.get.side_effect = [
        SimpleNamespace(status_code=status, json=lambda body=body: body) for status, body in responses
    ]
    return session


@pytest.mark.parametrize(
    "responses,expected",
    [
        ([(200, {"status": "ok"})], False),
        ([(200, {"status": "ok", "service": "other"})], False),
        ([(200, [])], False),
        ([(302, {"status": "ok", "service": launcher.SERVICE_NAME})], False),
        ([(200, {"status": "ok", "service": launcher.SERVICE_NAME}), (503, {"jobs": []})], False),
        ([(200, {"status": "ok", "service": launcher.SERVICE_NAME}), (200, {"jobs": []})], True),
    ],
)
def test_readiness_requires_service_identity_and_ready_jobs_without_proxy_or_redirect(monkeypatch, responses, expected):
    session = session_for(responses)
    monkeypatch.setattr(launcher.requests, "Session", lambda: session)
    assert launcher.api_is_ready("http://127.0.0.1:8000") is expected
    assert session.trust_env is False
    assert all(call.kwargs["allow_redirects"] is False for call in session.get.call_args_list)


def process_double(code=None):
    process = Mock()
    process.returncode = code
    process.studio_shutdown_timeout_seconds = 88
    process.poll.side_effect = lambda: process.returncode

    def wait(timeout):
        process.returncode = 0
        return 0

    process.wait.side_effect = wait
    return process


def test_start_captures_budget_and_process_group_then_uses_cooperative_signal(monkeypatch):
    process = process_double()
    popen = Mock(return_value=process)
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher, "api_is_ready", lambda url: True)
    monkeypatch.setenv("STUDIO_SHUTDOWN_GRACE_SECONDS", "130")
    assert launcher.start_api("http://127.0.0.1:8123") is process
    options = popen.call_args.kwargs
    assert options["env"]["STUDIO_ENGINE_PORT"] == options["env"]["F1_ENGINE_PORT"] == "8123"
    assert options["env"]["STUDIO_ENGINE_HOST"] == options["env"]["F1_ENGINE_HOST"] == "127.0.0.1"
    assert options["creationflags"] == (subprocess.CREATE_NEW_PROCESS_GROUP if launcher.os.name == "nt" else 0)
    assert options["start_new_session"] is (launcher.os.name != "nt")
    assert process.studio_shutdown_timeout_seconds == 158
    # Later environment changes cannot shorten an already owned service budget.
    monkeypatch.setenv("STUDIO_SHUTDOWN_GRACE_SECONDS", "0")
    launcher.stop_api(process)
    process.send_signal.assert_called_once_with(signal.CTRL_BREAK_EVENT if launcher.os.name == "nt" else signal.SIGTERM)
    assert process.wait.call_args.kwargs["timeout"] > 157
    process.kill.assert_not_called()
    process.terminate.assert_not_called()


@pytest.mark.parametrize("failure", [RuntimeError("probe failed"), KeyboardInterrupt()])
def test_startup_failure_closes_owned_process(monkeypatch, failure):
    process = process_double()
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: process)

    def fail(url):
        raise failure

    monkeypatch.setattr(launcher, "api_is_ready", fail)
    with pytest.raises(type(failure)):
        launcher.start_api("http://127.0.0.1:8000")
    process.send_signal.assert_called_once()
    assert process.returncode == 0


def test_startup_timeout_closes_owned_process(monkeypatch):
    process = process_double()
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(launcher, "STARTUP_TIMEOUT_SECONDS", 0)
    with pytest.raises(RuntimeError, match="did not become ready"):
        launcher.start_api("http://127.0.0.1:8000")
    assert process.returncode == 0


@pytest.mark.parametrize("code", [1, 17, -15])
def test_preexisting_nonzero_service_exit_is_not_success(code):
    process = process_double(code)
    with pytest.raises(RuntimeError, match=f"exited with code {code}"):
        launcher.stop_api(process)
    process.send_signal.assert_not_called()


def test_forced_kill_is_failure_even_when_kill_wait_succeeds():
    process = process_double()
    process.studio_shutdown_timeout_seconds = 1
    process.wait.side_effect = [subprocess.TimeoutExpired("service", 1), 0]
    with pytest.raises(RuntimeError, match="exceeded its shutdown budget; forced termination"):
        launcher.stop_api(process)
    process.kill.assert_called_once()
    assert process.wait.call_args.kwargs["timeout"] == launcher.SERVICE_EXIT_SECONDS


def test_repeated_interrupt_does_not_abandon_cleanup():
    process = process_double()
    process.studio_shutdown_timeout_seconds = 4
    process.wait.side_effect = [KeyboardInterrupt(), KeyboardInterrupt(), 0]
    process.returncode = None
    process.poll.side_effect = [None]

    def wait(timeout):
        if process.wait.call_count <= 2:
            raise KeyboardInterrupt
        process.returncode = 0
        return 0

    process.wait.side_effect = wait
    launcher.stop_api(process)
    assert process.wait.call_count == 3
    process.kill.assert_not_called()


@pytest.mark.parametrize("existing", [True, False])
@pytest.mark.parametrize("failure", [None, KeyboardInterrupt(), RuntimeError("UI failed")])
def test_main_respects_external_service_ownership_and_cleans_on_ui_exit(monkeypatch, existing, failure):
    process = process_double(0)
    monkeypatch.setattr(launcher, "api_is_ready", lambda url: existing)
    start = Mock(return_value=process)
    stop = Mock()
    monkeypatch.setattr(launcher, "start_api", start)
    monkeypatch.setattr(launcher, "stop_api", stop)

    def ui(api_process):
        assert api_process is (None if existing else process)
        if failure is not None:
            raise failure

    monkeypatch.setattr(launcher, "run_webui", ui)
    if isinstance(failure, RuntimeError):
        with pytest.raises(RuntimeError, match="UI failed"):
            launcher.main()
    else:
        assert launcher.main() == 0
    assert start.call_count == (0 if existing else 1)
    stop.assert_called_once_with(None if existing else process)
    assert launcher.os.environ["STUDIO_API_URL"] == launcher.os.environ["F1_STUDIO_API_URL"]


@pytest.mark.parametrize("failure", [RuntimeError("launch failed"), KeyboardInterrupt(), None])
def test_webui_is_closed_on_launch_failure_interrupt_or_unexpected_service_exit(monkeypatch, tmp_path, failure):
    webui = Mock()
    webui.is_running = True
    webui.launch.side_effect = failure
    ui_class = Mock(return_value=SimpleNamespace(build_app=lambda: webui))
    monkeypatch.setitem(
        sys.modules, "app", SimpleNamespace(GlobalConfig=SimpleNamespace(THEME="theme"), YOLO_Master_WebUI=ui_class)
    )
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    expected = type(failure) if failure else RuntimeError
    with pytest.raises(expected):
        launcher.run_webui(process_double(1))
    assert webui.launch.call_args.kwargs["prevent_thread_lock"] is True
    assert webui.launch.call_args.kwargs["inbrowser"] is False
    webui.close.assert_called_once()


def test_launcher_signals_restore_original_handlers():
    signals = [signal.SIGINT, signal.SIGTERM]
    if launcher.os.name == "nt":
        signals.append(signal.SIGBREAK)
    originals = {sig: signal.getsignal(sig) for sig in signals}
    with pytest.raises(KeyboardInterrupt), launcher.launcher_signals():
        signal.getsignal(signals[-1])(signals[-1], None)
    assert {sig: signal.getsignal(sig) for sig in signals} == originals


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"NO_PROXY": "example.internal, 10.0.0.0/8", "no_proxy": "other.internal,example.internal"},
        {"NO_PROXY": "*", "no_proxy": "LOCALHOST"},
    ],
)
def test_loopback_proxy_bypass_preserves_exclusions_and_restores_process_environment(monkeypatch, environment):
    environment = {**environment, "HTTP_PROXY": "http://127.0.0.1:1", "UNRELATED": "unchanged"}
    original = dict(environment)
    monkeypatch.setattr(launcher.os, "environ", environment)
    with launcher.loopback_proxy_bypass():
        assert environment["NO_PROXY"] == environment["no_proxy"]
        exclusions = {entry.lower() for entry in environment["NO_PROXY"].split(",")}
        assert {"127.0.0.1", "localhost", "::1"} <= exclusions
        for key in ("NO_PROXY", "no_proxy"):
            assert {entry.strip().lower() for entry in original.get(key, "").split(",") if entry.strip()} <= exclusions
        assert environment["HTTP_PROXY"] == original["HTTP_PROXY"]
        assert environment["UNRELATED"] == "unchanged"
        first = dict(environment)
        with launcher.loopback_proxy_bypass():
            assert environment == first
        assert environment == first
    assert environment == original


def test_loopback_proxy_bypass_restores_even_on_startup_failure(monkeypatch):
    environment = {"NO_PROXY": "existing.internal"}
    monkeypatch.setattr(launcher.os, "environ", environment)
    with pytest.raises(RuntimeError), launcher.loopback_proxy_bypass():
        raise RuntimeError("startup failed")
    assert environment == {"NO_PROXY": "existing.internal"}
