"""Optional Gradio integration checks, run by the dedicated Studio Frontend lane."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import gradio as gr
import pytest

from studio.ui.i18n import get_text
from studio.ui.jobs_tab import (
    _artifact_download_update,
    _artifact_selector_update,
    create_jobs_tab,
)
from studio.ui.studio_jobs_client import StudioBackendUnavailableError


class Client:
    """A presentation-only HTTP stub with no Runtime or backend state."""

    base_url = "http://127.0.0.1:8000"

    def __init__(self):
        self.status = "RUNNING"
        self.cancel_calls = []
        self.log_reads = 0

    def list_recent_jobs(self, limit):
        """Return a recent row for selection testing."""
        return [{"job_id": "j1", "task_type": "predict", "status": self.status, "created_at": ""}]

    def get_job_status(self, job_id):
        """Read the current owner snapshot."""
        return {"status": self.status, "duration": 1.0, "artifact_count": 0}

    def get_job_logs(self, job_id):
        """Return the available logs without changing lifecycle."""
        self.log_reads += 1
        return "current logs"

    def get_job_artifact_metadata(self, job_id):
        """Return only manifest-approved metadata."""
        return []

    def cancel_job(self, job_id):
        """Acknowledge the request while leaving the owner snapshot running."""
        self.cancel_calls.append(job_id)
        return "Cancellation requested"


def callback(tab, name):
    """Find the actual registered callback rather than reimplementing its behavior."""
    return next(value.fn for value in tab.fns.values() if value.fn and value.fn.__name__ == name)


def test_gradio_cancel_ack_keeps_timer_active_and_does_not_invent_terminal_state():
    """The UI continues polling until the backend confirms cleanup."""
    client = Client()
    tab = create_jobs_tab(client)
    message, timer, error = callback(tab, "cancel_job_handler")("j1", "en")
    assert "j1" in message and timer.active and not error
    assert client.cancel_calls == ["j1"]
    snapshot = callback(tab, "poll_handler")("j1", "en")
    assert snapshot[0]["status"] == "RUNNING" and snapshot[-1].active


def test_live_checkbox_pauses_tailing_and_preserves_lifecycle_polling():
    """Paused logs do not disable state monitoring; terminal refresh drains the tail."""
    client = Client()
    tab = create_jobs_tab(client)
    live = next(block for block in tab.blocks.values() if isinstance(block, gr.Checkbox))
    assert live.value is True
    poll = callback(tab, "poll_handler")
    paused = poll("j1", "en", False, "previous logs")
    assert paused[3] == "previous logs" and paused[-1].active
    assert client.log_reads == 0
    client.status = "COMPLETED"
    terminal = poll("j1", "en", False, "previous logs")
    assert terminal[3] == "current logs" and not terminal[-1].active


@pytest.mark.parametrize("change", ["job", "endpoint", "switch_back"])
def test_job_or_endpoint_switch_discards_inflight_poll_response(change):
    """An old HTTP response cannot overwrite panels after selection changes."""
    client = Client()
    entered, release = Event(), Event()
    original = client.get_job_status

    def blocked(job_id):
        entered.set()
        assert release.wait(5)
        return original(job_id)

    client.get_job_status = blocked
    tab = create_jobs_tab(client)
    request = gr.Request(session_hash="session-one")
    poll = callback(tab, "poll_handler")
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(poll, "j1", "en", True, "", request)
        assert entered.wait(5)
        if change in {"job", "switch_back"}:
            callback(tab, "select_job")("j2", request)
            if change == "switch_back":
                callback(tab, "select_job")("j1", request)
        else:
            client.base_url = "http://127.0.0.1:9000"
        release.set()
        updates = result.result(timeout=5)
    assert len(updates) == 13
    assert all(update == gr.skip() for update in updates)


def test_recent_table_selection_uses_row_job_id_and_manual_job_entry_is_available():
    """Clicking any recent-table column selects the same exact job ID."""
    tab = create_jobs_tab(Client())
    event = gr.SelectData(
        None, {"index": [0, 2], "value": "RUNNING", "row_value": ["exact-job", "predict", "RUNNING", ""]}
    )
    assert callback(tab, "select_recent")(event) == "exact-job"
    job_input = next(
        block
        for block in tab.blocks.values()
        if isinstance(block, gr.Textbox) and block.label == get_text("en", "field.job_id")
    )
    assert job_input.interactive


def test_nested_artifact_selector_and_table_keep_exact_identity():
    """Equal filenames in different directories resolve to the selected manifest entry."""
    metadata = [
        {
            "filename": "preview.png",
            "artifact_id": f"{part}/preview.png",
            "preview_path": f"/cache/{part}/preview.png",
            "source_path": f"/cache/{part}/preview.png",
            "is_image": True,
            "download_url": f"http://localhost/static/artifacts/j1/{part}/preview.png",
        }
        for part in ("a", "b")
    ]
    update = _artifact_selector_update(metadata, "en")
    assert update["choices"] == ["a/preview.png", "b/preview.png"]
    assert _artifact_download_update(metadata, "b/preview.png")["value"] == "/cache/b/preview.png"
    tab = create_jobs_tab(Client())
    image, download = callback(tab, "on_artifact_select")("b/preview.png", metadata, "en")
    assert image["value"] == download["value"] == "/cache/b/preview.png"
    event = gr.SelectData(None, {"index": [1, 0], "value": "b/preview.png"})
    selected, image, download = callback(tab, "on_artifact_table_select")(event, metadata, "en")
    assert selected["value"] == "b/preview.png"
    assert image["value"] == download["value"] == "/cache/b/preview.png"


def test_backend_failure_is_a_platform_error_not_a_completed_job():
    """Failed polling displays diagnostics and stops the fast timer."""
    client = Client()
    tab = create_jobs_tab(client)

    def unavailable(job_id):
        raise StudioBackendUnavailableError("GET", f"/api/v1/jobs/{job_id}")

    client.get_job_status = unavailable
    snapshot = callback(tab, "poll_handler")("j1", "en")
    assert snapshot[0] == {"scope": "platform", "error_code": "STUDIO_BACKEND_UNAVAILABLE"}
    assert snapshot[1] and not snapshot[-1].active


def test_submission_registers_selection_before_first_poll():
    """The returned job can immediately poll even before Gradio delivers a textbox change."""
    client = Client()
    client.submit_job = lambda **kwargs: ("accepted-job", "")
    tab = create_jobs_tab(client)
    submit = callback(tab, "submit_job_handler")
    assert submit("diagnose", "", "", "runs", 0.25, "cpu", ".", "en")[0] == "accepted-job"
    assert callback(tab, "poll_handler")("accepted-job", "en")[0]["status"] == "RUNNING"


@pytest.mark.parametrize("operation", ["submit", "cancel"])
def test_selection_change_discards_late_mutation_acknowledgments(operation):
    """Accepted HTTP mutations do not overwrite the newly selected job's controls."""
    client = Client()
    entered, release = Event(), Event()

    def mutate(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return ("accepted-job", "") if operation == "submit" else "Cancellation requested"

    client.submit_job = mutate
    client.cancel_job = mutate
    tab = create_jobs_tab(client)
    request = gr.Request(session_hash="mutation-session")
    if operation == "submit":
        fn = callback(tab, "submit_job_handler")
        args = ("diagnose", "", "", "runs", 0.25, "cpu", ".", "en", request)
    else:
        fn = callback(tab, "cancel_job_handler")
        args = ("j1", "en", request)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args)
        assert entered.wait(5)
        callback(tab, "select_job")("new-job", request)
        release.set()
        updates = future.result(timeout=5)
    assert all(update == gr.skip() for update in updates)


@pytest.mark.parametrize("callback_name", ["poll_handler", "sync_handler"])
def test_stale_security_failure_cannot_emit_toast_or_consume_warning_guard(callback_name, monkeypatch):
    """Discarded snapshots have no UI side effects, including one-shot warning bookkeeping."""
    client = Client()
    entered, release = Event(), Event()
    warnings = []
    monkeypatch.setattr(gr, "Warning", lambda message: warnings.append(message))

    def failed_status(job_id):
        entered.set()
        assert release.wait(5)
        return {"status": "FAILED", "error_code": "SEC_ERR_001", "error_message": "rejected"}

    client.get_job_status = failed_status
    tab = create_jobs_tab(client)
    request = gr.Request(session_hash="security-session")
    poll = callback(tab, callback_name)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(poll, "old-job", "en", True, "", request)
        assert entered.wait(5)
        callback(tab, "select_job")("new-job", request)
        release.set()
        updates = future.result(timeout=5)
    assert all(update == gr.skip() for update in updates)
    assert warnings == []
    callback(tab, "select_job")("old-job", request)
    assert poll("old-job", "en", True, "", request)[0]["status"] == "FAILED"
    assert len(warnings) == 1
    poll("old-job", "en", True, "", request)
    assert len(warnings) == 1


@pytest.mark.parametrize("operation", ["submit", "cancel"])
def test_late_mutation_errors_cannot_emit_toast_or_overwrite_new_selection(operation, monkeypatch):
    """A failed old HTTP mutation is silently discarded once the user selects another job."""
    client = Client()
    entered, release = Event(), Event()
    warnings = []
    monkeypatch.setattr(gr, "Warning", lambda message: warnings.append(message))

    def fail(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        raise StudioBackendUnavailableError("POST", "/api/v1/jobs")

    client.submit_job = fail
    client.cancel_job = fail
    tab = create_jobs_tab(client)
    request = gr.Request(session_hash="failed-mutation-session")
    if operation == "submit":
        fn = callback(tab, "submit_job_handler")
        args = ("diagnose", "", "", "runs", 0.25, "cpu", ".", "en", request)
    else:
        fn = callback(tab, "cancel_job_handler")
        args = ("old-job", "en", request)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args)
        assert entered.wait(5)
        callback(tab, "select_job")("new-job", request)
        release.set()
        updates = future.result(timeout=5)
    assert all(update == gr.skip() for update in updates)
    assert warnings == []


@pytest.fixture
def artifact_client():
    """Expose a mixed manifest, with identical IDs scoped to different jobs."""
    client = Client()
    client.status = "COMPLETED"
    client.manifests = {
        job: [
            {
                "filename": identifier.rsplit("/", 1)[-1],
                "artifact_id": identifier,
                "preview_path": f"/cache/{job}/{identifier}" if image else None,
                "source_path": f"/cache/{job}/{identifier}",
                "is_image": image,
                "download_url": f"http://localhost/static/artifacts/{job}/{identifier}",
            }
            for identifier, image in (
                ("nested/large_payload.bin", False),
                ("preview.png", True),
                ("b/preview.png", True),
            )
        ]
        for job in ("j1", "j2")
    }
    client.get_job_artifact_metadata = lambda job_id: client.manifests[job_id]
    return client


def timer_callback(tab, seconds):
    """Use the callback actually bound to the natural Timer tick, without sleeping."""
    timer = next(block for block in tab.blocks.values() if isinstance(block, gr.Timer) and block.value == seconds)
    dependency = next(item for item in tab.config["dependencies"] if (timer._id, "tick") in item["targets"])
    return tab.fns[dependency["id"]].fn


@pytest.mark.parametrize("seconds", [1.0, 30.0])
def test_timer_preserves_nonimage_download_and_last_image_preview(artifact_client, seconds):
    """An ordinary tick retains the exact selected download and image preview independently."""
    tab = create_jobs_tab(artifact_client)
    request = gr.Request(session_hash="artifact-session")
    callback(tab, "select_job")("j1", request)
    tick = timer_callback(tab, seconds)
    initial = tick("j1", "en", True, "", request)
    assert initial[11]["value"] == "/cache/j1/preview.png"  # Image-first default, despite manifest order.
    metadata = initial[10]
    image, download = callback(tab, "on_artifact_select")("b/preview.png", metadata, "en", request=request)
    assert image["value"] == download["value"] == "/cache/j1/b/preview.png"
    event = gr.SelectData(None, {"index": [0, 0], "value": "nested/large_payload.bin"})
    selected = callback(tab, "on_artifact_table_select")(event, metadata, "en", request=request)
    assert selected[2]["value"] == "/cache/j1/nested/large_payload.bin"  # Immediate download still works.
    for _ in range(2):
        synced = tick("j1", "en", True, "", request)
        assert synced[11]["value"] == "/cache/j1/nested/large_payload.bin"
        assert synced[6]["value"] == "b/preview.png"
        assert synced[8]["value"] == "/cache/j1/b/preview.png"
    selector_fn = next(fn for fn in tab.fns.values() if fn.fn and fn.fn.__name__ == "on_artifact_select")
    dependency = next(item for item in tab.config["dependencies"] if item["id"] == selector_fn._id)
    assert dependency["targets"][0][1] == "input"  # Programmatic refresh must not select the preview as download.


@pytest.mark.parametrize("reset", ["removed", "job", "endpoint", "session"])
def test_artifact_selection_is_valid_only_for_current_manifest_and_scope(artifact_client, reset):
    """Invalid artifacts, job/API switches and other browser sessions cannot inherit the bin selection."""
    tab = create_jobs_tab(artifact_client)
    request = gr.Request(session_hash="artifact-session")
    callback(tab, "select_job")("j1", request)
    tick = timer_callback(tab, 30.0)
    metadata = tick("j1", "en", True, "", request)[10]
    event = gr.SelectData(None, {"index": [0, 0], "value": "nested/large_payload.bin"})
    callback(tab, "on_artifact_table_select")(event, metadata, "en", request=request)
    job = "j1"
    if reset == "removed":
        artifact_client.manifests[job] = metadata[1:]
    elif reset == "job":
        job = "j2"
        cleared = callback(tab, "select_job")(job, request)
        assert cleared[4]["value"] is None
        stale = callback(tab, "on_artifact_table_select")(event, metadata, "en", job_id="j1", request=request)
        assert all(update == gr.skip() for update in stale)
    elif reset == "endpoint":
        artifact_client.base_url = "http://127.0.0.1:9000"
    else:
        request = gr.Request(session_hash="another-browser")
    result = tick(job, "en", True, "", request)
    assert result[11]["value"] == f"/cache/{job}/preview.png"
    if reset == "removed":
        artifact_client.manifests[job] = []
        empty = tick(job, "en", True, "", request)
        assert empty[11]["value"] is None and not empty[11]["interactive"]
        assert empty[8]["value"] is None and empty[6]["value"] is None


def test_sync_uses_selection_made_during_http_read(artifact_client):
    """A slow snapshot reads the latest browser selection when publishing its controls."""
    tab = create_jobs_tab(artifact_client)
    request = gr.Request(session_hash="artifact-session")
    callback(tab, "select_job")("j1", request)
    tick = timer_callback(tab, 30.0)
    metadata = tick("j1", "en", True, "", request)[10]
    entered, release = Event(), Event()
    original = artifact_client.get_job_status

    def blocked(job_id):
        entered.set()
        assert release.wait(5)
        return original(job_id)

    artifact_client.get_job_status = blocked
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(tick, "j1", "en", True, "", request)
        assert entered.wait(5)
        event = gr.SelectData(None, {"index": [0, 0], "value": "nested/large_payload.bin"})
        callback(tab, "on_artifact_table_select")(event, metadata, "en", request=request)
        release.set()
        result = future.result(timeout=5)
    assert result[11]["value"] == "/cache/j1/nested/large_payload.bin"


def test_image_paging_becomes_the_download_selection_and_survives_sync(artifact_client):
    """Next/previous image buttons keep their existing preview and immediate-download behavior."""
    tab = create_jobs_tab(artifact_client)
    request = gr.Request(session_hash="paging-session")
    callback(tab, "select_job")("j1", request)
    tick = timer_callback(tab, 30.0)
    metadata = tick("j1", "en", True, "", request)[10]
    next_button = next(
        block for block in tab.blocks.values() if isinstance(block, gr.Button) and block.value == "Next ▶"
    )
    step = next(fn.fn for fn in tab.fns.values() if (next_button._id, "click") in fn.targets)
    selected, image, download = step("preview.png", metadata, "en", "j1", request)
    assert selected["value"] == "b/preview.png"
    assert image["value"] == download["value"] == "/cache/j1/b/preview.png"
    result = tick("j1", "en", True, "", request)
    assert result[8]["value"] == result[11]["value"] == "/cache/j1/b/preview.png"


def test_manifest_without_images_defaults_to_first_download(artifact_client):
    """A file-only manifest remains downloadable and has no image preview."""
    artifact_client.manifests["j1"] = artifact_client.manifests["j1"][:1]
    tab = create_jobs_tab(artifact_client)
    result = timer_callback(tab, 30.0)("j1", "en")
    assert result[11]["value"] == "/cache/j1/nested/large_payload.bin"
    assert result[8]["value"] is None and not result[6]["choices"]
