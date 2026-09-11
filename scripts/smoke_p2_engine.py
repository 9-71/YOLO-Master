"""Automated end-to-end smoke verification for the P2 FastAPI engine.

P2 freeze gate (Step 4): this script drives a *running* ``main_engine``
instance over HTTP and asserts the full job lifecycle contract exposed by the
standalone FastAPI engine:

    1. ``GET  /health``                                — service availability
    2. ``POST /api/v1/jobs/`` (``task_type="diagnose"``, CPU) — dispatch
    3. ``GET  /api/v1/jobs/{job_id}``                  — terminal polling
    4. ``GET  /api/v1/jobs/{job_id}/logs?offset=...``  — incremental log streaming
    5. ``POST /api/v1/jobs/{job_id}/cancel``           — cooperative cancellation
    6. ``GET  /api/v1/jobs/{job_id}/artifacts`` + ``GET /static/artifacts/...``
                                                       — artifact delivery

Cooperative cancellation is exercised against a deliberately slow mock job
(four sequential CPU predictions of a local sample image, one engine call per
chunk): the job is observed RUNNING, cancelled over the API, and must land in
``CANCELLED`` with ``error_code == "USER_CANCELLED"`` — never in ``COMPLETED``.

The script is intentionally standalone and CPU-only: it talks to the engine
exactly like a browser or CLI client would, uses only the standard library
plus whichever HTTP client is already installed (``httpx`` preferred,
``requests`` fallback), and requires no network access on its own — the local
``yolov8n.pt`` / ``ultralytics/assets/bus.jpg`` shipped in the repository
satisfy the mock job.

Usage::

    python scripts/smoke_p2_engine.py                     # engine on 127.0.0.1:8000
    python scripts/smoke_p2_engine.py --base-url http://127.0.0.1:8765
    python scripts/smoke_p2_engine.py --deadline 120 --poll-interval 0.2

Exit code is 0 when every executed check passes and non-zero otherwise, so the
script can gate CI or the Phase 2 freeze checklist directly.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

#: Repository root (``scripts/..``): hosts the engine, the local model weights
#: and the sample media used by the mock long-running cancellation job.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Terminal states returned by ``GET /api/v1/jobs/{job_id}``.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

#: HTTP client preference: httpx is the FastAPI-stack client, requests ships
#: with Ultralytics itself; either one satisfies the "already-installed only"
#: dependency rule.
try:
    import httpx  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised on httpx-less environments
    httpx = None

try:
    import requests  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised on requests-less environments
    requests = None

if httpx is None and requests is None:
    raise SystemExit("smoke_p2_engine requires either 'httpx' or 'requests' installed")

#: Maximum window size accepted by ``GET /api/v1/jobs/{job_id}/logs``.
_LOG_LIMIT = 500


class SmokeError(RuntimeError):
    """Raised when one verification check fails; message becomes the row detail."""


class SmokeHttp:
    """Tiny adapter over ``httpx`` (preferred) or ``requests``.

    Both clients share the keyword surface used here (``params=``, ``json=``)
    and expose ``status_code``/``json()``/``text``/``content`` responses, so
    the adapter keeps every call site identical regardless of which backend is
    installed. Transport failures are normalized to :class:`SmokeError`.

    Args:
        base_url: Engine origin, e.g. ``http://127.0.0.1:8000`` (no trailing slash).
        timeout: Per-request timeout in seconds.
    """

    def __init__(self, base_url: str, timeout: float) -> None:
        """Initialize the client session bound to ``base_url``."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Loopback traffic must never travel through a system/registry proxy:
        # on Windows, httpx/requests pick up the OS proxy configuration via
        # ``trust_env`` and a corporate proxy answers loopback requests with
        # 502 Bad Gateway, breaking local-engine verification. Remote targets
        # keep proxy resolution enabled.
        host = urlparse(self.base_url).hostname or ""
        trust_env = not (host == "localhost" or host.startswith("127."))
        if httpx is not None:
            self._session = httpx.Client(base_url=self.base_url, timeout=timeout, trust_env=trust_env)
        else:
            self._session = requests.Session()
            self._session.trust_env = trust_env

    def _request(self, method: Callable[..., object], path: str, **kwargs: object):
        """Run one request, converting transport errors into :class:`SmokeError`.

        Args:
            method: Bound ``get``/``post`` callable of the backend session.
            path: API path with a leading slash (e.g. ``/health``).
            **kwargs: Request options (``params=``, ``json=``).

        Returns:
            The backend response object.

        Raises:
            SmokeError: When the engine is unreachable or the request times out.
        """
        try:
            return method(f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        except Exception as exc:
            raise SmokeError(f"request to {path} failed: {type(exc).__name__}: {exc}") from exc

    def get(self, path: str, params: dict[str, object] | None = None):
        """Perform a GET request against ``self.base_url + path``.

        Args:
            path: API path with a leading slash (e.g. ``/health``).
            params: Optional query-string parameters.

        Returns:
            The backend response object.
        """
        return self._request(self._session.get, path, params=params)

    def post(self, path: str, json: dict[str, object] | None = None):
        """Perform a POST request against ``self.base_url + path``.

        Args:
            path: API path with a leading slash (e.g. ``/api/v1/jobs/``).
            json: Optional JSON body.

        Returns:
            The backend response object.
        """
        return self._request(self._session.post, path, json=json)


class CheckRow:
    """One row of the terminal summary table.

    Args:
        name: Short step label (e.g. ``"Cancellation"``).
        status: ``"PASS"``, ``"FAIL"`` or ``"SKIP"``.
        detail: One-line human-readable outcome attached to the row.
    """

    def __init__(self, name: str, status: str, detail: str) -> None:
        """Initialize the row with its summary fields."""
        self.name = name
        self.status = status
        self.detail = detail


def now_tag() -> str:
    """Return a compact UTC timestamp used to name smoke runs and job ids.

    Returns:
        str: ``YYYYmmdd_HHMMSS`` UTC string.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def unique_job_id(task_type: str) -> str:
    """Return a collision-free job id for the persistent engine state file.

    The engine persists jobs across restarts (``runs/jobs_state.json``), so
    repeated smoke runs must never reuse a ``job_id``.

    Args:
        task_type: Task type prefix (e.g. ``"diagnose"``).

    Returns:
        str: ``p2_smoke_{task_type}_{timestamp}_{uuid8}``.
    """
    return f"p2_smoke_{task_type}_{now_tag()}_{uuid.uuid4().hex[:8]}"


def job_payload(
    job_id: str, task_type: str, params: dict[str, object], *, diagnostics: bool = False
) -> dict[str, object]:
    """Build a valid submission body for ``POST /api/v1/jobs/``.

    The engine normalizes the payload fail-closed server-side (fresh PENDING
    lifecycle, ``allow_shell=False``, ``path_whitelisted=True``, output_dir
    added to the dynamic whitelist); the body mirrors a real console
    submission so the response reflects the enforced server-side state.

    Args:
        job_id: Unique job identifier.
        task_type: Registered task type (``diagnose`` or ``predict``).
        params: Task-specific parameters.
        diagnostics: When True, route outputs under ``runs/diagnose`` (the
            diagnose handler's canonical location); otherwise ``runs/predict``.

    Returns:
        dict[str, object]: JSON-serializable submission payload.
    """
    output_dir = str(REPO_ROOT / "runs" / ("diagnose" if diagnostics else "predict"))
    return {
        "job_id": job_id,
        "task_type": task_type,
        "params": params,
        "output": {"output_dir": output_dir},
        "security_constraints": {
            "allow_shell": False,
            "path_whitelisted": True,
            "allowed_paths": [str(REPO_ROOT), str(REPO_ROOT / "runs")],
            "allowed_path_patterns": [],
        },
        "runtime_tracking": {
            "stream_logs": True,
            "timeout_seconds": 120,
            "cancellable": True,
            "cancel_requested": False,
        },
    }


def diagnose_payload(job_id: str) -> dict[str, object]:
    """Return the fast smoke-job payload: CPU diagnostics, no model load.

    Args:
        job_id: Unique job identifier.

    Returns:
        dict[str, object]: Submission body with ``device="cpu"`` params.
    """
    return job_payload(job_id, "diagnose", {"device": "cpu"}, diagnostics=True)


def mock_long_running_payload(job_id: str) -> dict[str, object]:
    """Return the mock long-running job payload used by the cancellation check.

    Four sequential CPU predictions of the bundled sample image (one engine
    call per chunk, ``batch_size=1``) keep the job RUNNING for a comfortably
    cancellable window — the predict handler runs a cooperative-cancellation
    checkpoint before every chunk — without any network access: both
    ``yolov8n.pt`` and ``ultralytics/assets/bus.jpg`` exist in the repository.

    Args:
        job_id: Unique job identifier.

    Returns:
        dict[str, object]: Submission body for a predict job on CPU.
    """
    bus_image = str(REPO_ROOT / "ultralytics" / "assets" / "bus.jpg")
    params: dict[str, object] = {
        "model_path": str(REPO_ROOT / "yolov8n.pt"),
        "data_source": [bus_image, bus_image, bus_image, bus_image],
        "batch_size": 1,
        "device": "cpu",
        "conf": 0.25,
    }
    return job_payload(job_id, "predict", params)


def verify_health(http: SmokeHttp) -> str:
    """Check 1: ping ``GET /health`` and assert the liveness contract.

    Args:
        http: Bound HTTP client.

    Returns:
        str: Human-readable success detail.

    Raises:
        SmokeError: When the engine is unreachable or reports a non-ok status.
    """
    response = http.get("/health")
    if response.status_code != 200:
        raise SmokeError(f"GET /health returned HTTP {response.status_code}")
    body = response.json()
    if body.get("status") != "ok":
        raise SmokeError(f"GET /health returned unexpected body: {body}")
    service = body.get("service", "unknown")
    return f"HTTP 200, status=ok (service={service})"


def verify_dispatch(http: SmokeHttp, payload: dict[str, object]) -> str:
    """Check 2: submit a job and assert the 201 contract.

    Args:
        http: Bound HTTP client.
        payload: Submission body (see :func:`diagnose_payload`).

    Returns:
        str: Human-readable success detail including the assigned job id.

    Raises:
        SmokeError: When the submission is rejected or not accepted as 201.
    """
    response = http.post("/api/v1/jobs/", json=payload)
    if response.status_code != 201:
        raise SmokeError(f"POST /api/v1/jobs/ returned HTTP {response.status_code}: {response.text[:300]}")
    body = response.json()
    if body.get("job_id") != payload["job_id"]:
        raise SmokeError(f"response job_id mismatch: {body.get('job_id')!r}")
    if body.get("status") != "pending":
        raise SmokeError(f"expected fresh lifecycle status 'pending', got {body.get('status')!r}")
    return f"HTTP 201, job {body['job_id']} registered as pending"


def fetch_log_window(http: SmokeHttp, job_id: str, offset: int) -> dict[str, object]:
    """Fetch one paginated log window starting at ``offset``.

    Args:
        http: Bound HTTP client.
        job_id: Job whose logs are streamed.
        offset: Cursor of the first line to fetch.

    Returns:
        dict[str, object]: The ``LogsResponse`` body.

    Raises:
        SmokeError: When the endpoint fails (job unknown or transport error).
    """
    response = http.get(f"/api/v1/jobs/{job_id}/logs", params={"offset": offset, "limit": _LOG_LIMIT})
    if response.status_code != 200:
        raise SmokeError(f"GET /api/v1/jobs/{job_id}/logs?offset={offset} returned HTTP {response.status_code}")
    return response.json()


def stream_new_logs(http: SmokeHttp, job_id: str, offset: int) -> int:
    """Print every undelivered log line since ``offset`` and return the new cursor.

    Args:
        http: Bound HTTP client.
        job_id: Job whose logs are streamed.
        offset: Cursor of the first undelivered line.

    Returns:
        int: New cursor (``total`` when the tail has been reached).
    """
    while True:
        window = fetch_log_window(http, job_id, offset)
        lines = window["logs"]
        for line in lines:
            print(line)
        next_offset = window["next_offset"]
        if next_offset is None:
            return window["total"]
        offset = next_offset


def wait_terminal_with_log_streaming(
    http: SmokeHttp, job_id: str, deadline: float, poll_interval: float
) -> dict[str, object]:
    """Poll ``GET /api/v1/jobs/{job_id}`` to a terminal state, streaming logs.

    Each poll iteration streams the incremental log window since the last
    delivered cursor (``GET .../logs?offset=...``) to standard output, then
    re-reads the job status. The loop is guarded by an absolute execution
    deadline so a hung job fails the check instead of blocking forever.

    Args:
        http: Bound HTTP client.
        job_id: Job to monitor.
        deadline: Maximum seconds to wait for a terminal status.
        poll_interval: Seconds between status polls.

    Returns:
        dict[str, object]: The final job-status body.

    Raises:
        SmokeError: When the job does not reach a terminal state in time.
    """
    start = time.monotonic()
    offset = 0
    while True:
        response = http.get(f"/api/v1/jobs/{job_id}")
        if response.status_code != 200:
            raise SmokeError(f"GET /api/v1/jobs/{job_id} returned HTTP {response.status_code}")
        body = response.json()
        status = body["status"]
        if status not in _TERMINAL_STATUSES and time.monotonic() - start > deadline:
            raise SmokeError(f"job {job_id} still {status!r} after {deadline:.0f}s deadline")
        offset = stream_new_logs(http, job_id, offset)
        if status in _TERMINAL_STATUSES:
            return body
        time.sleep(poll_interval)


def verify_polling(http: SmokeHttp, job_id: str, deadline: float, poll_interval: float) -> dict[str, object]:
    """Check 3: monitor the smoke job to completion within the deadline.

    Args:
        http: Bound HTTP client.
        job_id: Diagnose job identifier.
        deadline: Terminal-wait deadline in seconds.
        poll_interval: Poll cadence in seconds.

    Returns:
        dict[str, object]: The final job-status body.

    Raises:
        SmokeError: On deadline breach, non-200 status reads or a FAILED job.
    """
    body = wait_terminal_with_log_streaming(http, job_id, deadline, poll_interval)
    if body["status"] != "completed":
        raise SmokeError(
            f"job ended {body['status']!r} (error_code={body.get('error_code')!r}, "
            f"error_message={body.get('error_message')!r})"
        )
    return body


def verify_log_streaming(http: SmokeHttp, job_id: str, expected_marker: str) -> str:
    """Check 4: page the whole log buffer and assert cursor integrity + marker.

    The buffer is walked in ``?offset=...`` windows from the start (without
    re-printing — the incremental lines were streamed during polling). The
    pagination contract is asserted window by window: every window but the
    last advances the cursor by exactly its own line count and the final
    window reports ``next_offset=None`` at ``total``. The completion marker
    emitted by the dispatcher state machine must appear in the buffer.

    Args:
        http: Bound HTTP client.
        job_id: Diagnose job identifier.
        expected_marker: Log fragment that must appear (e.g. the state-machine
            completion line ``"transitioned to: COMPLETED"``).

    Returns:
        str: Human-readable success detail.

    Raises:
        SmokeError: When the cursor math is inconsistent or the marker missing.
    """
    offset = 0
    seen: list[str] = []
    while True:
        window = fetch_log_window(http, job_id, offset)
        lines = window["logs"]
        next_offset = window["next_offset"]
        if next_offset is not None and next_offset != offset + len(lines):
            raise SmokeError(
                f"log cursor jumped: offset {offset} returned {len(lines)} line(s) but next_offset={next_offset}"
            )
        seen.extend(lines)
        if next_offset is None:
            break
        offset = next_offset
    if not seen:
        raise SmokeError("job produced no log lines at all")
    if not any(expected_marker in line for line in seen):
        raise SmokeError(f"completion marker {expected_marker!r} missing from {len(seen)} logged lines")
    return f"{len(seen)} lines streamed incrementally (cursor clean to the tail), marker {expected_marker!r} found"


def verify_cancellation(
    http: SmokeHttp, job_id: str, start_deadline: float, terminal_deadline: float, poll_interval: float
) -> str:
    """Check 5: cancel the mock long-running job and assert CANCELLED.

    Flow: dispatch the predict mock job, wait until it is observed RUNNING
    (so the exercise targets the dispatcher's *in-flight* cooperative
    cancellation path rather than the pre-execution short-circuit), POST the
    cancel request (expect 202 + ``cancel_requested``) and poll to the
    terminal state, which must be ``CANCELLED`` with error code
    ``USER_CANCELLED`` — a COMPLETED transition would fail the check.

    Args:
        http: Bound HTTP client.
        job_id: Mock long-running job identifier.
        start_deadline: Seconds to wait for the RUNNING observation.
        terminal_deadline: Seconds to wait for the terminal state after cancel.
        poll_interval: Poll cadence in seconds.

    Returns:
        str: Human-readable success detail including the observed duration.

    Raises:
        SmokeError: On any deviation from the cancellation contract.
    """
    # Phase A: the job must be observed RUNNING (never silently completed).
    start = time.monotonic()
    while True:
        response = http.get(f"/api/v1/jobs/{job_id}")
        if response.status_code != 200:
            raise SmokeError(f"GET /api/v1/jobs/{job_id} returned HTTP {response.status_code}")
        body = response.json()
        status = body["status"]
        if status == "running":
            break
        if status in _TERMINAL_STATUSES:
            raise SmokeError(
                f"mock job reached {status!r} before cancellation "
                f"(error_code={body.get('error_code')!r}) — RUNNING window too short"
            )
        if time.monotonic() - start > start_deadline:
            raise SmokeError(f"mock job never entered RUNNING within {start_deadline:.0f}s (stuck at {status!r})")
        time.sleep(poll_interval)

    # Phase B: request cooperative cancellation (202 + cancel_requested).
    response = http.post(f"/api/v1/jobs/{job_id}/cancel")
    if response.status_code != 202:
        raise SmokeError(
            f"POST /api/v1/jobs/{job_id}/cancel returned HTTP {response.status_code}: {response.text[:300]}"
        )
    cancel_body = response.json()
    if cancel_body.get("status") != "cancel_requested":
        raise SmokeError(f"cancel response missing 'cancel_requested' status: {cancel_body}")

    # Phase C: the job must land in CANCELLED with the USER_CANCELLED error code.
    terminal = time.monotonic() + terminal_deadline
    while time.monotonic() < terminal:
        response = http.get(f"/api/v1/jobs/{job_id}")
        body = response.json()
        if body["status"] in _TERMINAL_STATUSES:
            if body["status"] != "cancelled":
                raise SmokeError(
                    f"cancelled job ended {body['status']!r} instead of cancelled (error_code={body.get('error_code')!r})"
                )
            if body.get("error_code") != "USER_CANCELLED":
                raise SmokeError(
                    f"cancelled job ended with error_code={body.get('error_code')!r} "
                    f"instead of 'USER_CANCELLED' (error_message={body.get('error_message')!r})"
                )
            return f"202 cancel_requested → CANCELLED (USER_CANCELLED) in {body.get('duration', 'N/A')}"
        time.sleep(poll_interval)
    raise SmokeError(f"cancelled job did not reach a terminal state within {terminal_deadline:.0f}s")


def verify_artifacts(http: SmokeHttp, job_id: str) -> str:
    """Check 6: verify the manifest and the static artifact download route.

    The completed job must report its files in the manifest with
    ``/static/artifacts/...`` download references, and every referenced URL
    must stream the actual file bytes. The route is fail-closed (it serves
    only manifest-listed files), so a 200 here proves manifest-to-disk
    delivery end to end.

    Args:
        http: Bound HTTP client.
        job_id: Completed diagnose job identifier.

    Returns:
        str: Human-readable success detail listing delivered files.

    Raises:
        SmokeError: When the manifest or any download URL misbehaves.
    """
    response = http.get(f"/api/v1/jobs/{job_id}/artifacts")
    if response.status_code != 200:
        raise SmokeError(f"GET /api/v1/jobs/{job_id}/artifacts returned HTTP {response.status_code}")
    body = response.json()
    artifacts = body.get("artifacts", [])
    if not artifacts:
        raise SmokeError("artifact manifest is empty for the completed job")
    delivered: list[str] = []
    for entry in artifacts:
        filename = entry.get("filename")
        download_url = entry.get("download_url", "")
        if not download_url.startswith(f"/static/artifacts/{job_id}/"):
            raise SmokeError(f"artifact {filename!r} has unexpected download_url {download_url!r}")
        fetched = http.get(download_url)
        if fetched.status_code != 200:
            raise SmokeError(f"GET {download_url} returned HTTP {fetched.status_code} (artifact {filename!r})")
        if not fetched.content:
            raise SmokeError(f"GET {download_url} returned an empty body (artifact {filename!r})")
        delivered.append(f"{filename} ({len(fetched.content)} bytes)")
    return f"{len(delivered)} artifact(s) delivered via /static/artifacts: {', '.join(delivered)}"


def force_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8 so engine emoji log lines never crash printing.

    Windows consoles frequently run legacy code pages (e.g. GBK) that cannot
    encode the emoji the engine embeds in log lines (✅/❌/🚫), so printing a
    streamed line would raise ``UnicodeEncodeError``. UTF-8 keeps both modern
    terminals and redirected log files safe; the reconfigure is best-effort and
    skipped on streams that do not support it (e.g. under embedded callers).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def print_table(rows: list[CheckRow], *, failures: int) -> None:
    """Render the structured step-by-step summary table on stdout.

    Args:
        rows: Rows in execution order (Health, Dispatch, Polling, Logs
            Streaming, Cancellation, Artifacts Delivery).
        failures: Number of FAIL rows (drives the exit code).
    """
    print()
    print("=" * 80)
    print("P2 ENGINE SMOKE — TERMINAL SUMMARY")
    print("=" * 80)
    for row in rows:
        print(f"  {row.name:<24} {row.status:<6} {row.detail}")
    print("-" * 80)
    total = len(rows)
    outcome = "ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED"
    print(f"  RESULT: {outcome} ({total - failures}/{total} passed)")
    print("=" * 80)


def run_checks(http: SmokeHttp, args: argparse.Namespace) -> int:
    """Execute the six verification steps in order and print the summary.

    Args:
        http: Bound HTTP client.
        args: Parsed CLI arguments (deadlines, poll cadence).

    Returns:
        int: Process exit code — 0 when every executed check passed.
    """
    print(f"[*] P2 FastAPI engine smoke against {http.base_url}")
    rows: list[CheckRow] = []
    failures = 0

    def run_step(name: str, fn: Callable[[], str]) -> None:
        """Run one check, record its row and keep going after a failure."""
        nonlocal failures
        try:
            detail = fn()
        except SmokeError as exc:
            failures += 1
            rows.append(CheckRow(name, "FAIL", str(exc)))
        except Exception as exc:  # noqa: BLE001 - unexpected faults surface as FAIL rows
            failures += 1
            rows.append(CheckRow(name, "FAIL", f"unexpected {type(exc).__name__}: {exc}"))
        else:
            rows.append(CheckRow(name, "PASS", detail))
        print(f"  [{len(rows)}/6] {rows[-1].name:<24} ... {rows[-1].status}")

    # Check 1 — service availability.
    run_step("Health", lambda: verify_health(http))

    # Checks 2-4 — dispatch, terminal polling and log streaming on the diagnose job.
    diagnose_job_id = unique_job_id("diagnose")
    run_step("Dispatch", lambda: verify_dispatch(http, diagnose_payload(diagnose_job_id)))
    dispatch_ok = rows[-1].status == "PASS"

    if dispatch_ok:

        def _polling() -> str:
            body = verify_polling(http, diagnose_job_id, args.deadline, args.poll_interval)
            return (
                f"status={body['status']}, duration={body.get('duration', 'N/A')}, "
                f"artifact_count={body.get('artifact_count', 0)}"
            )

        run_step("Polling", _polling)
        polling_ok = rows[-1].status == "PASS"

        def _logs() -> str:
            if not polling_ok:
                raise SmokeError("skipped: Polling check failed")
            return verify_log_streaming(http, diagnose_job_id, "Completed. Artifacts:")

        run_step("Logs Streaming", _logs)
    else:
        rows.append(CheckRow("Polling", "SKIP", "diagnose dispatch failed"))
        rows.append(CheckRow("Logs Streaming", "SKIP", "diagnose dispatch failed"))
        print("  [3/6] Polling                 ... SKIP")
        print("  [4/6] Logs Streaming          ... SKIP")

    # Check 5 — cooperative cancellation of a mock long-running job.
    cancel_job_id = unique_job_id("predict")

    def _cancellation() -> str:
        dispatched = verify_dispatch(http, mock_long_running_payload(cancel_job_id))
        outcome = verify_cancellation(
            http, cancel_job_id, args.start_deadline, args.cancel_deadline, args.poll_interval
        )
        return f"{dispatched}; {outcome}"

    run_step("Cancellation", _cancellation)

    # Check 6 — artifact manifest and static delivery for the completed job.
    def _artifacts() -> str:
        if not dispatch_ok:
            raise SmokeError("skipped: no completed diagnose job")
        return verify_artifacts(http, diagnose_job_id)

    run_step("Artifacts Delivery", _artifacts)

    print_table(rows, failures=failures)
    return 0 if failures == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the smoke script command line.

    Args:
        argv: Argument list; ``None`` reads ``sys.argv``.

    Returns:
        argparse.Namespace: ``--base-url``, deadline and cadence options.
    """
    parser = argparse.ArgumentParser(
        description="End-to-end smoke verification for the P2 standalone FastAPI engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running FastAPI engine",
    )
    parser.add_argument(
        "--deadline",
        type=float,
        default=90.0,
        help="Max seconds to wait for the diagnose job's terminal state",
    )
    parser.add_argument(
        "--start-deadline",
        type=float,
        default=30.0,
        help="Max seconds to wait for the mock job to enter RUNNING",
    )
    parser.add_argument(
        "--cancel-deadline",
        type=float,
        default=30.0,
        help="Max seconds to wait for the cancelled job's terminal state",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds between job-status polls",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=20.0,
        help="Per-request HTTP timeout in seconds",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: run the smoke verification and return the exit code.

    Args:
        argv: Argument list forwarded to :func:`parse_args`.

    Returns:
        int: 0 when every executed check passed, 1 otherwise.
    """
    args = parse_args(argv)
    force_utf8_stdio()
    http = SmokeHttp(args.base_url, args.http_timeout)
    try:
        return run_checks(http, args)
    except Exception as exc:  # noqa: BLE001 - top-level guard keeps the exit code honest
        print(f"[!] smoke runner aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
