"""Offline HTTP client contracts; backend and optional Gradio imports stay separate."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
import requests

from studio.ui.jobs_tab import compute_poll_state
from studio.ui.studio_jobs_client import (
    StudioBackendUnavailableError,
    StudioJobsApiClient,
    StudioJobsApiResponseError,
)


class Response:
    """Minimal HTTP response, including streamed download failure injection."""

    def __init__(self, payload=None, content=b"artifact", error=None):
        self.payload, self.content, self.error = payload, content, error
        self.closed = False

    def raise_for_status(self):
        """Supply the successful HTTP status for this response."""

    def json(self):
        """Return the configured response body."""
        return self.payload

    def iter_content(self, chunk_size):
        """Yield one chunk, then any injected streaming failure."""
        yield self.content
        if self.error:
            raise self.error

    def close(self):
        """Record release of this streamed response."""
        self.closed = True


class Session:
    """Queue HTTP responses and retain requests for contract assertions."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, **kwargs):
        """Return one scripted response."""
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, Response) else Response(response)


def test_client_and_presentation_import_without_backend_or_gradio():
    """Service-only installations can collect Studio doctests with GUI dependencies blocked."""
    script = """
import doctest
import sys
class BlockBackend:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('gradio', 'f1.jobs_manager', 'studio.jobs_manager', 'api', 'main_engine')):
            raise RuntimeError('forbidden dependency: ' + fullname)
sys.meta_path.insert(0, BlockBackend())
import studio.ui
import studio.ui.i18n
import studio.ui.jobs_tab
import studio.ui.studio_jobs_client
import f1.ui.jobs_tab
import f1.ui.studio_jobs_client
runner = doctest.DocTestRunner()
for module in (studio.ui, studio.ui.i18n, studio.ui.jobs_tab, studio.ui.studio_jobs_client):
    for test in doctest.DocTestFinder().find(module):
        runner.run(test)
summary = runner.summarize()
assert summary.failed == 0
assert summary.attempted > 0
assert 'gradio' not in sys.modules
assert 'studio.jobs_manager' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_legacy_client_is_the_same_modern_class(monkeypatch):
    """Modern endpoint configuration wins while historical imports remain aliases."""
    from f1.ui.studio_jobs_client import StudioJobsApiClient as LegacyClient

    monkeypatch.setenv("STUDIO_API_URL", "http://127.0.0.1:9000/")
    monkeypatch.setenv("F1_STUDIO_API_URL", "http://127.0.0.1:8001")
    assert LegacyClient is StudioJobsApiClient
    assert StudioJobsApiClient().base_url == "http://127.0.0.1:9000"


def test_submission_and_cancel_ack_leave_lifecycle_to_status():
    """An accepted cancellation is followed by a still-running owner snapshot."""
    session = Session(
        {"job_id": "j1", "status": "pending"},
        {"job_id": "j1", "status": "cancel_requested", "message": "Cancellation requested"},
        {"job_id": "j1", "status": "running", "artifact_count": 0},
    )
    client = StudioJobsApiClient("http://localhost", session=session)
    assert client.submit_job("predict", "model.pt", "image.jpg", "runs", 0.25, "cpu", ["."])[0] == "j1"
    assert session.calls[0][2]["json"]["security_constraints"]["allow_shell"] is False
    assert client.cancel_job("j1") == "Cancellation requested"
    assert client.get_job_status("j1")["status"] == "RUNNING"
    assert not hasattr(client, "jobs")


@pytest.mark.parametrize("payload", [None, [], {}, {"status": "invented"}])
def test_malformed_status_is_a_platform_error(payload):
    """Malformed responses cannot be rendered as a valid terminal job."""
    client = StudioJobsApiClient(session=Session(payload))
    with pytest.raises(StudioJobsApiResponseError):
        client.get_job_status("j1")


@pytest.mark.parametrize("payload", [{}, {"job_id": "j1"}, {"job_id": [], "status": "pending"}])
def test_malformed_submission_never_claims_acceptance(payload):
    """Only a backend job identifier and valid lifecycle constitute an acceptance."""
    with pytest.raises(StudioJobsApiResponseError):
        StudioJobsApiClient(session=Session(payload)).submit_job("diagnose", "", "", "runs", 0.25, "cpu", ["."])


def test_connection_failure_keeps_request_outcome_unknown():
    """A network timeout is represented as an API failure, never an invented job state."""
    client = StudioJobsApiClient(session=Session(requests.Timeout("unknown server outcome")))
    with pytest.raises(StudioBackendUnavailableError):
        client.submit_job("diagnose", "", "", "runs", 0.25, "cpu", ["."])


def test_logs_drain_pages_and_do_not_treat_current_tail_as_permanent():
    """The final page and a later terminal tail both remain visible."""
    session = Session(
        {"logs": ["one", "two"], "next_offset": 2},
        {"logs": ["three"], "next_offset": None},
        {"logs": ["one", "two", "three", "terminal tail"], "next_offset": None},
    )
    client = StudioJobsApiClient(session=session)
    assert client.get_job_logs("j1") == "one\ntwo\nthree"
    assert session.calls[1][2]["params"]["offset"] == 2
    assert client.get_job_logs("j1").endswith("terminal tail")


@pytest.mark.parametrize("cursor", [-1, 0, True, "1", 4])
def test_invalid_log_cursor_fails_without_looping(cursor):
    """A malformed or stalled page cursor cannot hang the GUI polling worker."""
    client = StudioJobsApiClient(session=Session({"logs": ["one"], "next_offset": cursor}))
    with pytest.raises(StudioJobsApiResponseError):
        client.get_job_logs("j1")


@pytest.mark.parametrize("logs", ["one", None, [1], [{}]])
def test_invalid_log_entries_are_rejected(logs):
    """The client consumes the line-array contract without inventing text from objects."""
    with pytest.raises(StudioJobsApiResponseError):
        StudioJobsApiClient(session=Session({"logs": logs})).get_job_logs("j1")


def entry(identifier, *, image=False):
    """Build a server manifest entry without any local server path."""
    return {"artifact_id": identifier, "filename": Path(identifier).name, "is_image": image}


def test_nested_artifact_ids_are_exact_and_downloads_come_from_http(tmp_path):
    """Equal basenames remain distinct and are never resolved from a submitted output root."""
    session = Session(
        {
            "artifacts": [
                entry("a/preview.png", image=True),
                entry("b/preview.png", image=True),
                entry("weights/last.pt"),
            ]
        },
        Response(content=b"first"),
        Response(content=b"second"),
        Response(content=b"weights"),
    )
    client = StudioJobsApiClient("http://localhost", session=session)
    metadata = client.get_job_artifact_metadata("j1")
    assert [item["artifact_id"] for item in metadata] == ["a/preview.png", "b/preview.png", "weights/last.pt"]
    assert [Path(item["source_path"]).read_bytes() for item in metadata] == [b"first", b"second", b"weights"]
    assert metadata[0]["preview_path"] != metadata[1]["preview_path"]
    assert session.calls[-1][1] == "http://localhost/static/artifacts/j1/weights/last.pt"
    assert client.get_artifact_download_directory("j1").is_dir()
    client.base_url = "http://localhost:9000"
    assert client.get_artifact_download_directory("j1") is None


@pytest.mark.parametrize(
    "identifier", ["../secret", "a/../secret", "/secret", "a//b", "a/./b", "C:/secret", "a\\secret", "a\nsecret"]
)
def test_malformed_manifest_ids_fail_closed(identifier):
    """Only safe exact relative identifiers can enter the local download cache."""
    session = Session({"artifacts": [entry(identifier)]})
    with pytest.raises(StudioJobsApiResponseError):
        StudioJobsApiClient(session=session).get_job_artifact_metadata("j1")
    assert len(session.calls) == 1


def test_partial_download_failure_closes_response_and_cannot_become_cached():
    """A failed stream leaves no successful artifact and releases the HTTP connection."""
    response = Response(error=requests.ConnectionError("lost stream"))
    client = StudioJobsApiClient(session=Session({"artifacts": [entry("result.csv")]}, response))
    from studio.ui.studio_jobs_client import StudioJobsApiError

    with pytest.raises(StudioJobsApiError):
        client.get_job_artifact_metadata("j1")
    assert response.closed
    assert list(Path(client._artifact_cache.name).rglob("*.part")) == []
    assert list(Path(client._artifact_cache.name).rglob("result.csv")) == []


class PresentationClient:
    """HTTP presentation stub; intentionally no manager state or helper methods."""

    def __init__(self, status="RUNNING"):
        self.status, self.log_reads = status, 0

    def get_job_status(self, job_id):
        """Supply a detached API status."""
        return {"status": self.status, "duration": None}

    def get_job_logs(self, job_id):
        """Record consumption of logs."""
        self.log_reads += 1
        return "terminal tail"

    def get_job_artifact_metadata(self, job_id):
        """Supply the exact empty manifest."""
        return []

    def list_recent_jobs(self, limit):
        """Supply the recent-jobs window."""
        return []


def test_paused_live_logs_preserve_text_and_terminal_poll_still_drains():
    """Disabling Live affects tailing alone, with final terminal output retained."""
    client = PresentationClient()
    active = compute_poll_state(client, "j1", live_logs=False, previous_logs="kept")
    assert active.logs == "kept"
    assert active.keep_polling and client.log_reads == 0
    client.status = "COMPLETED"
    terminal = compute_poll_state(client, "j1", live_logs=False, previous_logs="kept")
    assert terminal.logs == "terminal tail"
    assert not terminal.keep_polling and client.log_reads == 1


@pytest.mark.parametrize(
    "origin", ["http://localhost:8000", "http://127.0.0.1:8000", "http://127.0.0.2:8000", "http://[::1]:8000"]
)
def test_owned_loopback_session_bypasses_ambient_proxy_settings(origin):
    """The supported local API transport cannot be redirected through environment proxies."""
    client = StudioJobsApiClient(origin)
    assert client._session.trust_env is False
    client.close()


def test_remote_or_injected_session_environment_semantics_are_preserved():
    """Only this client's default loopback transport changes proxy configuration."""
    remote = StudioJobsApiClient("http://studio.test")
    assert remote._session.trust_env is True
    remote.close()
    session = Session({"status": "running"})
    session.trust_env = True
    client = StudioJobsApiClient("http://localhost", session=session)
    client.get_job_status("j1")
    assert session.trust_env is True
    assert session.calls[0][2]["allow_redirects"] is False


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_redirect_response_is_rejected_before_it_can_change_origin(method):
    """Mutations and reads share the same no-redirect HTTP boundary."""
    redirect = Response({"status": "completed"})
    redirect.status_code = 302
    session = Session(redirect)
    client = StudioJobsApiClient("http://localhost", session=session)
    with pytest.raises(StudioJobsApiResponseError) as caught:
        client._request(method, "/api/v1/jobs/j1")
    assert caught.value.status_code == 302
    assert caught.value.detail == "redirect_not_allowed"
    assert session.calls[0][2]["allow_redirects"] is False
    assert redirect.closed


@contextmanager
def local_http_server(*, redirect_to=None):
    """Serve a real loopback endpoint and retain every received path."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            """Serve status JSON or an explicit redirect to a different local origin."""
            seen.append(self.path)
            if redirect_to:
                self.send_response(302)
                self.send_header("Location", redirect_to)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            payload = json.dumps({"status": "running"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            """Keep expected local probe traffic out of test output."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def set_broken_ambient_proxy(monkeypatch):
    """Remove loopback exemptions while pointing all proxy schemes at an unavailable port."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")


def test_real_loopback_api_works_with_broken_proxy_and_no_no_proxy(monkeypatch):
    """Actual requests reaches the local API despite ambient proxy configuration."""
    set_broken_ambient_proxy(monkeypatch)
    with local_http_server() as (origin, seen):
        client = StudioJobsApiClient(origin, timeout=1)
        try:
            assert client.get_job_status("j1")["status"] == "RUNNING"
        finally:
            client.close()
        assert seen == ["/api/v1/jobs/j1"]


def test_real_cross_origin_redirect_is_not_followed_even_with_broken_proxy(monkeypatch):
    """A second server receives no request when the API sends a cross-origin Location."""
    set_broken_ambient_proxy(monkeypatch)
    with local_http_server() as (foreign_origin, foreign_seen):
        with local_http_server(redirect_to=f"{foreign_origin}/foreign-origin") as (origin, seen):
            client = StudioJobsApiClient(origin, timeout=1)
            try:
                with pytest.raises(StudioJobsApiResponseError) as caught:
                    client.get_job_status("j1")
                assert caught.value.status_code == 302
                assert caught.value.detail == "redirect_not_allowed"
            finally:
                client.close()
            assert seen == ["/api/v1/jobs/j1"]
        assert foreign_seen == []
