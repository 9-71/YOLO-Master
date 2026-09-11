"""Gradio Jobs Tab for YOLO-Master Studio.

This module provides a dedicated Jobs Tab UI component for submitting tasks, monitoring
real-time lifecycle states, viewing live logs, and downloading generated artifacts.

Architecture:
    UI Components → Job Submission → JobDispatcherStateMachine.execute()
                 ↓
            Adaptive Polling → 1s lifecycle timer while PENDING/RUNNING
                 ↓            → 30s background sync (recent jobs, artifacts)
            Log Console → stdout/stderr streaming
                 ↓
            Artifacts → File explorer with download buttons

Polling:
    The fast lifecycle timer (1s) ticks while the selected job is active. The tick that
    observes a terminal state performs the final refresh (including artifacts) and
    deactivates itself via ``gr.update(active=False)``. A slow always-on timer (30s)
    keeps the recent-jobs table and final artifacts fresh while the fast timer is idle.

Security:
    - Fail-closed path whitelisting (auto-fill allowed_paths from inputs/outputs)
    - Shell execution permanently disabled (allow_shell=False)
    - Path traversal protection via security constraints validation
    - SEC_ERR_001 / PARAM_VALIDATION_FAILED map to one-shot gr.Warning toasts and a
      persistent localized status banner

P2 decoupling:
    The backend core (``JobsManager`` and its thread-safe job/log/artifact helpers)
    lives in :mod:`f1.jobs_manager` so it can be imported and executed headlessly
    by the FastAPI engine (``api.v1.jobs``) without any Gradio UI state. This module
    re-exports those public names for backward compatibility with ``app.py``,
    ``demo_jobs_tab.py`` and the F1 test suites.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

from f1.jobs_manager import (
    ACTIVE_STATUSES,
    IMAGE_EXTENSIONS,
    JobsManager,
    _compute_duration,
    _resolve_completion_time,
    get_job_image_artifacts,
    is_terminal_status,
)
from f1.ui.i18n import DEFAULT_LANGUAGE, get_columns, get_text
from f1.ui.studio_jobs_client import ArtifactMetadata, StudioJobsApiError

__all__ = [
    "ACTIVE_STATUSES",
    "IMAGE_EXTENSIONS",
    "POLL_CONCURRENCY_ID",
    "JobsManager",
    "alert_banner",
    "compute_poll_state",
    "create_jobs_tab",
    "format_created_at",
    "get_job_image_artifacts",
    "is_terminal_status",
    "jobs_tab_language_updates",
    "platform_error_diagnostics",
    "recent_jobs_rows",
    "security_alert_toast",
]

#: Backend error codes that map to visual security/validation user alerts.
SECURITY_ALERT_CODES = frozenset({"SEC_ERR_001", "PARAM_VALIDATION_FAILED"})
#: Fast lifecycle polling interval (seconds) while a job is active.
POLL_FAST_SECONDS = 1.0
#: Slow background sync interval (seconds) for recent jobs and final artifacts.
POLL_SLOW_SECONDS = 30.0
#: Shared queue for every callback that writes the Jobs monitoring panels.
POLL_CONCURRENCY_ID = "gradio-jobs-status-sync"


def platform_error_diagnostics(lang: str, error: StudioJobsApiError, api_url: str = "") -> str:
    """Render a platform-scoped API error through the existing i18n resources."""
    key = f"{error.i18n_key}.http" if error.status_code else error.i18n_key
    message = get_text(lang, key).format(api_url=api_url, status_code=error.status_code)
    return f"[{error.code}] {message}"


#: Default form values per task type. Selecting a task_type repopulates model_path,
#: data_source and output_dir together (mirroring Inference Studio's dynamic weight
#: switching) so downstream handlers always receive engine-compatible inputs:
#: predict takes an image source while train/val require a dataset YAML. The per-field
#: 🔄 reset buttons restore the same preset values on demand without changing the task.
TASK_FORM_PRESETS: dict[str, dict[str, str]] = {
    "predict": {
        "model_path": "./ckpts/yolov8n.pt",
        "data_source": "ultralytics/assets/bus.jpg",
        "output_dir": "runs/predict",
    },
    "train": {
        "model_path": "./ckpts/yolov8n.pt",
        "data_source": "coco8.yaml",
        "output_dir": "runs/train",
    },
    "val": {
        "model_path": "./ckpts/yolov8n.pt",
        "data_source": "coco8.yaml",
        "output_dir": "runs/val",
    },
    "export": {
        "model_path": "./ckpts/yolov8n.pt",
        "data_source": "",
        "output_dir": "runs/export",
    },
    "diagnose": {
        "model_path": "",
        "data_source": "",
        "output_dir": "runs/diagnose",
    },
}

#: Localized artifact-preview label resolved by the language broadcast. Kept
#: local (rather than in ``i18n.py``) so the shared i18n module stays untouched.
_ARTIFACT_PREVIEW_BASE: dict[str, str] = {"en": "Artifact Preview", "zh": "产物预览"}
#: Localized label for the multi-image selector dropdown.
_ARTIFACT_SELECTOR_LABEL: dict[str, str] = {"en": "Select Image Artifact", "zh": "选择预览图片"}
#: Localized labels for the prev/next paging buttons.
_ARTIFACT_PREV_LABEL: dict[str, str] = {"en": "◀ Prev", "zh": "◀ 上一张"}
_ARTIFACT_NEXT_LABEL: dict[str, str] = {"en": "Next ▶", "zh": "下一张 ▶"}
#: Localized label for the open-output-folder button.
_OPEN_FOLDER_LABEL: dict[str, str] = {"en": "📂 Open Folder", "zh": "📂 打开输出目录"}
#: Local CSS injected into the Jobs zone to hide the Gradio Dataframe
#: column-header options button (the three-dot "Open cell menu" trigger). The
#: tab's Dataframes are read-only monitors, so the built-in column sort/filter
#: menu — whose labels are not localized by this app's i18n layer — is hidden
#: entirely. ``.cell-menu-button`` is the stable semantic class Gradio applies
#: to that trigger (``aria-label="Open cell menu"``).
_DATAFRAME_HEADER_MENU_CSS: str = "<style>.cell-menu-button { display: none !important; }</style>"


def _artifact_preview_label(lang: str | None, filename: str | None) -> str:
    """Return the localized preview label, appending the current filename.

    Args:
        lang: ISO language code ("en" or "zh").
        filename: Base name of the currently displayed image (may be ``None``).

    Returns:
        str: ``"Artifact Preview"`` (or ``"产物预览"``), with ``": {filename}"``
        appended when a file is being shown.
    """
    base = _ARTIFACT_PREVIEW_BASE.get(lang or DEFAULT_LANGUAGE, _ARTIFACT_PREVIEW_BASE[DEFAULT_LANGUAGE])
    return f"{base}: {filename}" if filename else base


def _image_preview_update(image_artifacts: list[str], lang: str | None) -> Any:
    """Build the ``gr.Image`` update for the first image artifact.

    Args:
        image_artifacts: Absolute paths of image files for the job.
        lang: ISO language code used to localize the label.

    Returns:
        Any: A ``gr.update`` carrying the selected path and localized label.
    """
    path = image_artifacts[0] if image_artifacts else None
    filename = Path(path).name if path else None
    return gr.update(value=path, label=_artifact_preview_label(lang, filename))


def _artifact_selector_update(image_artifacts: list[str], lang: str | None) -> Any:
    """Build the ``gr.Dropdown`` update listing switchable image filenames.

    Args:
        image_artifacts: Absolute paths of image files for the job.
        lang: ISO language code used to localize the label.

    Returns:
        Any: A ``gr.update`` with the filename choices, selected value and
        visibility (shown only when more than one image exists).
    """
    names = [Path(p).name for p in image_artifacts]
    value = names[0] if names else None
    label = _ARTIFACT_SELECTOR_LABEL.get(lang or DEFAULT_LANGUAGE, _ARTIFACT_SELECTOR_LABEL[DEFAULT_LANGUAGE])
    return gr.update(choices=names, value=value, label=label, visible=len(names) > 1)


def _cycle_artifact(paths: list[str], selected: str | None, step: int) -> tuple[str | None, str | None]:
    """Return ``(filename, path)`` of the item ``step`` positions away (cyclic).

    Args:
        paths: Absolute image paths for the job.
        selected: Currently selected filename (may be ``None``).
        step: Offset to advance (+1 next, -1 prev); wraps around the list.

    Returns:
        tuple[str | None, str | None]: The selected filename and full path, or
        ``(None, None)`` when ``paths`` is empty.
    """
    if not paths:
        return None, None
    names = [Path(p).name for p in paths]
    idx = names.index(selected) if selected and selected in names else 0
    new_idx = (idx + step) % len(names)
    return names[new_idx], paths[new_idx]


def _artifact_download_update(metadata: list[ArtifactMetadata], filename: str | None = None) -> Any:
    """Point the download control at the selected artifact's original source path."""
    selected = next((item for item in metadata if filename and item["filename"] == filename), None)
    if selected is None:
        selected = next((item for item in metadata if item["is_image"]), metadata[0] if metadata else None)
    source_path = selected["source_path"] if selected else None
    return gr.update(value=source_path, interactive=bool(source_path))


def format_created_at(created_at: str | None) -> str:
    """Format an ISO 8601 UTC ``created_at`` as local time for the Recent Jobs table.

    Backend timestamps remain raw ISO UTC strings under the hood; this helper only
    formats them for dataframe display.

    Args:
        created_at: Raw ISO 8601 UTC timestamp from ``JobRequest.metadata.created_at``.

    Returns:
        str: Local time as ``YYYY-MM-DD HH:MM:SS``; ``"-"`` when the value is empty
        or missing, and the raw string when it cannot be parsed.

    Example:
        >>> format_created_at("")
        '-'
        >>> format_created_at(None)
        '-'
        >>> format_created_at("not-a-timestamp")
        'not-a-timestamp'
        >>> from datetime import datetime
        >>> now_local = datetime.now().astimezone()
        >>> format_created_at(now_local.isoformat()) == now_local.strftime("%Y-%m-%d %H:%M:%S")
        True
    """
    if not created_at:
        return "-"
    try:
        return datetime.fromisoformat(created_at).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(created_at)


def recent_jobs_rows(jobs_manager: JobsManager, limit: int = 20) -> list[list[str]]:
    """Format the recent-jobs listing into Recent Jobs dataframe rows.

    Each row is ``[job_id, task_type, status, local_time]``, newest first, with
    raw ISO UTC timestamps localized via :func:`format_created_at`. Extracted
    from :func:`compute_poll_state` so the Recent Jobs table can be pre-populated
    at construction time (its initial ``value``) and stay byte-for-byte
    consistent with every polling refresh.

    Args:
        jobs_manager: JobsManager instance (or duck-typed equivalent).
        limit: Maximum number of recent jobs to include.

    Returns:
        list[list[str]]: Formatted rows for the Recent Jobs ``gr.Dataframe``.

    Example:
        >>> manager = JobsManager()
        >>> recent_jobs_rows(manager)
        []
    """
    return [
        [j["job_id"], j["task_type"], j["status"], format_created_at(j["created_at"])]
        for j in jobs_manager.list_recent_jobs(limit=limit)
    ]


def alert_banner(lang: str, code: str, message: str | None) -> str:
    """Build a localized Markdown alert banner for a failed job.

    Security/validation error codes get dedicated localized titles and bodies;
    other failure codes fall back to the generic failure title with the raw message.

    Args:
        lang: ISO language code.
        code: Backend error code (e.g. "SEC_ERR_001").
        message: Raw backend error message.

    Returns:
        str: Markdown blockquote banner.

    Example:
        >>> banner = alert_banner("en", "SEC_ERR_001", "path not in whitelist")
        >>> "Security Policy Violation" in banner
        True
        >>> banner = alert_banner("zh", "EXEC_ERR_500", "boom")
        >>> "任务失败" in banner
        True
    """
    if code in SECURITY_ALERT_CODES:
        title = get_text(lang, f"alert.{code}.title")
        body = get_text(lang, f"alert.{code}.body")
        detail = f"{code}: {message}" if message else code
        return f"> **{title}**\n> {body}\n> `{detail}`"
    title = get_text(lang, "alert.generic.title")
    detail = f"{code}: {message}" if message else code
    return f"> **{title}**\n> `{detail}`"


def security_alert_toast(lang: str, code: str) -> str:
    """Compose the one-shot gr.Warning toast text for a security/validation error.

    Args:
        lang: ISO language code.
        code: Backend error code from SECURITY_ALERT_CODES.

    Returns:
        str: Localized toast text.

    Example:
        >>> toast = security_alert_toast("zh", "SEC_ERR_001")
        >>> "安全策略违规" in toast
        True
    """
    return f"{get_text(lang, f'alert.{code}.title')} — {get_text(lang, f'alert.{code}.body')}"


@dataclass(frozen=True)
class PollState:
    """Snapshot of all job-monitoring panels produced by one polling cycle.

    Attributes:
        status: Canonical status dict (backend keys/values only) rendered by the Status Monitor JSON panel.
        error_text: Diagnostics text for the error box (empty when healthy).
        banner: Markdown alert banner (empty when the job has no failure).
        logs: Formatted execution logs.
        artifacts: Rows for the artifacts dataframe (filename, path).
        image_artifacts: Local paths of image files used only by the Gradio preview.
        artifact_metadata: Dual-path metadata retained for artifact UI callbacks.
        recent: Rows for the recent-jobs dataframe (timestamps formatted as local time).
        keep_polling: True while the job is active; False once terminal (deactivates the fast timer).
    """

    status: dict[str, Any]
    error_text: str = ""
    banner: str = ""
    logs: str = ""
    artifacts: list[list[str]] = field(default_factory=list)
    image_artifacts: list[str] = field(default_factory=list)
    artifact_metadata: list[ArtifactMetadata] = field(default_factory=list)
    recent: list[list[str]] = field(default_factory=list)
    keep_polling: bool = False


def compute_poll_state(
    jobs_manager: JobsManager,
    job_id: str,
    lang: str = DEFAULT_LANGUAGE,
    raw_status: dict[str, Any] | None = None,
) -> PollState:
    """Compute the complete polling snapshot for one job in the given language.

    Pure presentation logic over the JobsManager backend API: the backend responses
    are never modified, so backend-level test assertions remain valid. The Status
    Monitor JSON payload carries canonical backend keys/values only — localized
    display text is confined to banners, toasts and column headers. Recent-jobs
    timestamps are converted to local time for display while the backend keeps
    raw ISO UTC strings. In every state — including the idle "no selection"
    branch — the returned ``recent`` field is populated via
    :func:`recent_jobs_rows`, so polling never clears the Recent Jobs table.

    Args:
        jobs_manager: JobsManager instance (or duck-typed equivalent for tests).
        job_id: Selected job identifier (may be empty).
        lang: ISO language code passed to i18n lookups.
        raw_status: Optional status already fetched by the polling callback. This
            avoids a second API read inside the same UI snapshot.

    Returns:
        PollState: Snapshot with localized status, banner, logs, artifacts, recent
        jobs, and the keep_polling flag used to deactivate high-frequency polling.

    Example:
        >>> manager = JobsManager()
        >>> state = compute_poll_state(manager, "", "en")
        >>> state.keep_polling
        False
        >>> state.status
        {'status': 'NO_SELECTION'}
    """
    if not job_id:
        # Idle state: there is no selected job to monitor, but the Recent Jobs
        # table must still reflect the persisted history. Populate ``recent``
        # explicitly so the always-on slow sync timer (and any poll tick) never
        # overwrites the table with an empty list and blanks out the rows that
        # were pre-populated on first render.
        return PollState(
            status={"status": "NO_SELECTION"},
            recent=recent_jobs_rows(jobs_manager, limit=20),
        )

    raw = raw_status if raw_status is not None else jobs_manager.get_job_status(job_id)
    status_str = raw.get("status", "NOT_FOUND")

    status = {
        "job_id": job_id,
        "status": status_str,
        "duration": raw.get("duration"),
        "error_code": raw.get("error_code"),
        "error_message": raw.get("error_message"),
        "artifact_count": raw.get("artifact_count"),
    }

    # Live read-seconds apply strictly to RUNNING jobs. Terminal statuses must
    # freeze to a previously computed value when no completion timestamp is
    # available. NOT_FOUND has no job record and is left untouched so its empty
    # duration payload remains ``None``.
    if status_str != "NOT_FOUND" and not status["duration"]:
        created_at: str | None = None
        completed_at: str | None = None
        jobs = getattr(jobs_manager, "jobs", None)
        job = jobs.get(job_id) if isinstance(jobs, dict) else None
        if job is not None:
            created_at = getattr(job.metadata, "created_at", None)
            completed_at = _resolve_completion_time(job)
        resolved = _compute_duration(status_str, created_at, completed_at)
        if resolved:
            status["duration"] = resolved

    error_text = ""
    banner = ""
    if status_str == "FAILED":
        code = raw.get("error_code") or "UNKNOWN"
        error_text = f"[{code}] {raw.get('error_message') or ''}"
        banner = alert_banner(lang, code, raw.get("error_message"))

    logs = jobs_manager.get_job_logs(job_id)
    metadata_getter = getattr(jobs_manager, "get_job_artifact_metadata", None)
    if callable(metadata_getter):
        artifact_metadata = metadata_getter(job_id)
        artifacts = [[item["artifact_id"], item["source_path"]] for item in artifact_metadata]
        image_artifacts = [item["preview_path"] for item in artifact_metadata if item["preview_path"]]
    else:
        raw_artifacts = jobs_manager.get_job_artifacts(job_id)
        artifacts = [[name, path] for name, path in raw_artifacts]
        # Safe reflection: test Mocks may not implement get_job_image_artifacts.
        getter = getattr(jobs_manager, "get_job_image_artifacts", None)
        image_artifacts = getter(job_id) if callable(getter) else []
        preview_by_name = {Path(path).name: path for path in image_artifacts}
        artifact_metadata = [
            {
                "filename": Path(name).name,
                "artifact_id": name,
                "preview_path": preview_by_name.get(Path(name).name, ""),
                "source_path": path,
                "is_image": Path(path).suffix.lower() in IMAGE_EXTENSIONS,
                "download_url": path,
            }
            for name, path in raw_artifacts
        ]
    recent = recent_jobs_rows(jobs_manager, limit=20)

    return PollState(
        status=status,
        error_text=error_text,
        banner=banner,
        logs=logs,
        artifacts=artifacts,
        image_artifacts=image_artifacts,
        artifact_metadata=artifact_metadata,
        recent=recent,
        keep_polling=not is_terminal_status(status_str),
    )


def jobs_tab_language_updates(lang_value: str) -> tuple[Any, ...]:
    """Build the localized relabel payload covering every Jobs Tab component.

    Pure presentation helper used by the top-level language broadcast wired in
    app.py (the Jobs zone no longer has its own language listener). The payload
    mirrors the Jobs zone language outputs 1-to-1: element at position ``i``
    updates the component at position ``i`` of the zone output tuple (also
    exposed as ``create_jobs_tab(...)._language_outputs``).

    Args:
        lang_value: ISO language code ("en" or "zh"); unknown codes fall back
            to English via the i18n layer.

    Returns:
        tuple[Any, ...]: 33-element payload: the raw language value first (for
            the shared language State), then one ``gr.update`` per localizable
            component.
    """
    return (
        lang_value,  # lang_state
        gr.update(value=f"# {get_text(lang_value, 'tab.title')}"),  # title_md
        gr.update(value=f"### {get_text(lang_value, 'panel.submit')}"),  # submit_md
        gr.update(label=get_text(lang_value, "field.task_type")),  # task_type_radio
        gr.update(
            label=get_text(lang_value, "field.model_path"),
            placeholder=get_text(lang_value, "field.model_path.placeholder"),
        ),  # model_path_txt
        gr.update(value=get_text(lang_value, "button.reset_model")),  # model_path_reset_btn
        gr.update(
            label=get_text(lang_value, "field.data_source"),
            placeholder=get_text(lang_value, "field.data_source.placeholder"),
        ),  # data_source_txt
        gr.update(value=get_text(lang_value, "button.reset_data")),  # data_source_reset_btn
        gr.update(
            label=get_text(lang_value, "field.output_dir"),
            placeholder=get_text(lang_value, "field.output_dir.placeholder"),
        ),  # output_dir_txt
        gr.update(label=get_text(lang_value, "accordion.hyperparams")),  # hyperparams_accordion
        gr.update(label=get_text(lang_value, "field.conf")),  # conf_slider
        gr.update(label=get_text(lang_value, "field.device")),  # device_txt
        gr.update(label=get_text(lang_value, "accordion.security")),  # security_accordion
        gr.update(
            label=get_text(lang_value, "field.allowed_paths"),
            info=get_text(lang_value, "field.allowed_paths.info"),
        ),  # allowed_paths_txt
        gr.update(value=get_text(lang_value, "security.policy")),  # security_md
        gr.update(value=get_text(lang_value, "button.submit")),  # submit_btn
        gr.update(value=get_text(lang_value, "button.cancel")),  # cancel_job_btn
        gr.update(label=get_text(lang_value, "subtab.status")),  # status_tab
        gr.update(label=get_text(lang_value, "field.job_id")),  # job_id_display
        gr.update(label=get_text(lang_value, "field.status")),  # status_display
        gr.update(label=get_text(lang_value, "field.error")),  # error_box
        gr.update(label=get_text(lang_value, "subtab.logs")),  # logs_tab
        gr.update(label=get_text(lang_value, "field.logs")),  # logs_console
        gr.update(label=get_text(lang_value, "subtab.artifacts")),  # artifacts_tab
        gr.update(value=_OPEN_FOLDER_LABEL.get(lang_value, _OPEN_FOLDER_LABEL[DEFAULT_LANGUAGE])),  # open_folder_btn
        gr.update(
            headers=get_columns(lang_value, "artifacts"),
            label=get_text(lang_value, "df.artifacts"),
        ),  # artifacts_list
        gr.update(value=_ARTIFACT_PREV_LABEL.get(lang_value, _ARTIFACT_PREV_LABEL[DEFAULT_LANGUAGE])),  # prev_btn
        gr.update(
            label=_ARTIFACT_SELECTOR_LABEL.get(lang_value, _ARTIFACT_SELECTOR_LABEL[DEFAULT_LANGUAGE])
        ),  # artifact_selector
        gr.update(value=_ARTIFACT_NEXT_LABEL.get(lang_value, _ARTIFACT_NEXT_LABEL[DEFAULT_LANGUAGE])),  # next_btn
        gr.update(
            label=_ARTIFACT_PREVIEW_BASE.get(lang_value, _ARTIFACT_PREVIEW_BASE[DEFAULT_LANGUAGE])
        ),  # artifacts_image
        gr.update(label=get_text(lang_value, "subtab.recent")),  # recent_tab
        gr.update(
            headers=get_columns(lang_value, "recent"),
            label=get_text(lang_value, "df.recent"),
        ),  # recent_jobs_table
        gr.update(value=get_text(lang_value, "poll.note")),  # poll_note_md
    )


def create_jobs_tab(jobs_manager: JobsManager, lang: str = DEFAULT_LANGUAGE) -> gr.Blocks:
    """Create the Jobs Tab UI with adaptive polling, security alerts and i18n.

    Args:
        jobs_manager: Application-level JobsManager singleton shared across tabs.
        lang: Initial UI language ("en" or "zh"); the host app (app.py) owns the
            language selector and relabels this zone via the language broadcast.

    Returns:
        gr.Blocks: The Jobs Tab as a Gradio Blocks.

    Mounting contract:
        Mount the returned Blocks by calling it bare inside the parent tab context::

            with gr.TabItem("📋 Jobs"):
                create_jobs_tab(jobs_manager)

        Gradio auto-embeds a child Blocks on context exit, so do NOT additionally
        call ``.render()`` — that would mount every Jobs component a second time
        (duplicated tabs in the DOM).

    Polling design:
        - Fast lifecycle timer (1s): ticks while the selected job is PENDING/RUNNING and
          refreshes status, logs, artifacts and recent jobs on every tick. The tick that
          observes a terminal state performs the final refresh (including artifacts) and
          deactivates the timer via ``gr.update(active=False)``.
        - Slow sync timer (30s): always-on low-frequency refresh of the same panels so
          recent jobs and final artifacts stay fresh while the fast timer is idle.

    Security alerts:
        - The first tick observing SEC_ERR_001 / PARAM_VALIDATION_FAILED emits a
          one-shot gr.Warning toast; a persistent localized banner stays visible in the
          Status Monitor tab.

    Language-state lifting (unidirectional):
        Language selection is owned exclusively by the host app (app.py), whose
        top-level selector is the single source of truth. The tab itself never
        listens for language changes. Exposed broadcast handles:

        - ``jobs_tab._language_state``: the tab-local language State consumed by
          the polling handlers to localize status panels.
        - ``jobs_tab._language_outputs``: 33-element tuple of every localizable
          component, position-aligned with :func:`jobs_tab_language_updates`.

        The host relabels this zone with one explicit output list built from
        ``jobs_tab_language_updates(lang)`` — never through chained events.
    """
    # Job IDs whose security alert has already been toasted (one-shot warning guard).
    _security_warned: set[str] = set()
    try:
        initial_recent_jobs = recent_jobs_rows(jobs_manager, limit=20)
        initial_platform_error = ""
    except StudioJobsApiError as exc:
        initial_recent_jobs = []
        initial_platform_error = platform_error_diagnostics(lang, exc, getattr(jobs_manager, "base_url", ""))

    with gr.Blocks() as jobs_tab:
        lang_state = gr.State(lang)
        artifact_metadata_state = gr.State([])
        title_md = gr.Markdown(f"# {get_text(lang, 'tab.title')}")
        # Hide the Dataframe column-header options (three-dot) menu button across
        # the read-only monitoring tables in this zone (see module constant).
        gr.HTML(_DATAFRAME_HEADER_MENU_CSS)

        with gr.Row(equal_height=False):
            # ==================== Left Panel: Job Submission ====================
            with gr.Column(scale=1, variant="panel"):
                submit_md = gr.Markdown(f"### {get_text(lang, 'panel.submit')}")

                # Task type selector
                task_type_radio = gr.Radio(
                    choices=["predict", "train", "val", "export", "diagnose"],
                    value="predict",
                    label=get_text(lang, "field.task_type"),
                )

                # Dynamic input parameters form
                with gr.Group():
                    model_path_txt = gr.Textbox(
                        value="./ckpts/yolov8n.pt",
                        label=get_text(lang, "field.model_path"),
                        placeholder=get_text(lang, "field.model_path.placeholder"),
                    )
                    model_path_reset_btn = gr.Button(
                        get_text(lang, "button.reset_model"), size="sm", variant="secondary"
                    )
                    data_source_txt = gr.Textbox(
                        value="ultralytics/assets/bus.jpg",
                        label=get_text(lang, "field.data_source"),
                        placeholder=get_text(lang, "field.data_source.placeholder"),
                    )
                    data_source_reset_btn = gr.Button(
                        get_text(lang, "button.reset_data"), size="sm", variant="secondary"
                    )
                    output_dir_txt = gr.Textbox(
                        value="runs/predict",
                        label=get_text(lang, "field.output_dir"),
                        placeholder=get_text(lang, "field.output_dir.placeholder"),
                    )

                # Hyperparameters
                with gr.Accordion(get_text(lang, "accordion.hyperparams"), open=True) as hyperparams_accordion:
                    conf_slider = gr.Slider(0.0, 1.0, 0.25, step=0.01, label=get_text(lang, "field.conf"))
                    device_txt = gr.Textbox("0", label=get_text(lang, "field.device"))

                # Security constraints
                with gr.Accordion(get_text(lang, "accordion.security"), open=False) as security_accordion:
                    allowed_paths_txt = gr.Textbox(
                        value="., ultralytics/assets, runs, ckpts",
                        label=get_text(lang, "field.allowed_paths"),
                        info=get_text(lang, "field.allowed_paths.info"),
                    )
                    security_md = gr.Markdown(get_text(lang, "security.policy"))

                submit_btn = gr.Button(get_text(lang, "button.submit"), variant="primary", size="lg")
                submit_msg = gr.Markdown()

            # ==================== Right Panel: Monitoring ====================
            with gr.Column(scale=2), gr.Tabs():
                # Tab 1: State & Progress Monitor
                with gr.TabItem(get_text(lang, "subtab.status")) as status_tab:
                    job_id_display = gr.Textbox(label=get_text(lang, "field.job_id"), interactive=False)
                    status_display = gr.JSON(label=get_text(lang, "field.status"))
                    cancel_job_btn = gr.Button(get_text(lang, "button.cancel"), size="sm", variant="stop")
                    banner_md = gr.Markdown()
                    error_box = gr.Textbox(
                        value=initial_platform_error,
                        label=get_text(lang, "field.error"),
                        interactive=False,
                        lines=3,
                    )

                # Tab 2: Live Logs & Output Console
                with gr.TabItem(get_text(lang, "subtab.logs")) as logs_tab:
                    logs_console = gr.Textbox(
                        label=get_text(lang, "field.logs"),
                        lines=20,
                        interactive=False,
                        max_lines=100,
                    )

                # Tab 3: Artifacts Section
                with gr.TabItem(get_text(lang, "subtab.artifacts")) as artifacts_tab:
                    open_folder_btn = gr.Button(_OPEN_FOLDER_LABEL["en"], size="sm")
                    download_artifact_btn = gr.DownloadButton("⬇ Download", value=None, size="sm", interactive=False)
                    artifacts_list = gr.Dataframe(
                        headers=get_columns(lang, "artifacts"),
                        label=get_text(lang, "df.artifacts"),
                        interactive=False,
                    )
                    artifacts_image = gr.Image(
                        label=_ARTIFACT_PREVIEW_BASE["en"],
                        interactive=False,
                        type="filepath",
                        height=420,
                        visible=True,
                    )
                    with gr.Row(equal_height=True):
                        prev_btn = gr.Button(
                            _ARTIFACT_PREV_LABEL["en"],
                            size="sm",
                            scale=1,
                            min_width=80,
                            visible=False,
                        )
                        artifact_selector = gr.Dropdown(
                            label=_ARTIFACT_SELECTOR_LABEL["en"],
                            choices=[],
                            value=None,
                            interactive=True,
                            show_label=False,
                            container=False,
                            scale=6,
                            visible=False,
                        )
                        next_btn = gr.Button(
                            _ARTIFACT_NEXT_LABEL["en"],
                            size="sm",
                            scale=1,
                            min_width=80,
                            visible=False,
                        )

                # Tab 4: Recent Jobs
                with gr.TabItem(get_text(lang, "subtab.recent")) as recent_tab:
                    recent_jobs_table = gr.Dataframe(
                        headers=get_columns(lang, "recent"),
                        label=get_text(lang, "df.recent"),
                        value=initial_recent_jobs,
                        interactive=False,
                    )
                    poll_note_md = gr.Markdown(get_text(lang, "poll.note"))

        # Adaptive timers: fast lifecycle poll (activated on submit, self-deactivates
        # on terminal state) and slow always-on background sync.
        poll_timer = gr.Timer(POLL_FAST_SECONDS, active=False)
        sync_timer = gr.Timer(POLL_SLOW_SECONDS)

        # Localizable outputs relabeled by the host language broadcast: exactly 33
        # distinct components, position-aligned with the jobs_tab_language_updates()
        # payload. Exposed on the returned Blocks as ``_language_outputs`` so the host
        # app can broadcast a language change into this zone with one explicit output
        # list.
        language_outputs: tuple[gr.Component, ...] = (
            lang_state,
            title_md,
            submit_md,
            task_type_radio,
            model_path_txt,
            model_path_reset_btn,
            data_source_txt,
            data_source_reset_btn,
            output_dir_txt,
            hyperparams_accordion,
            conf_slider,
            device_txt,
            security_accordion,
            allowed_paths_txt,
            security_md,
            submit_btn,
            cancel_job_btn,
            status_tab,
            job_id_display,
            status_display,
            error_box,
            logs_tab,
            logs_console,
            artifacts_tab,
            open_folder_btn,
            artifacts_list,
            prev_btn,
            artifact_selector,
            next_btn,
            artifacts_image,
            recent_tab,
            recent_jobs_table,
            poll_note_md,
        )

        # ==================== Event Handlers ====================

        def poll_snapshot(job_id: str, lang_value: str) -> PollState:
            """Compute one snapshot, emitting a one-shot warning toast for new security alerts.

            The warning is queued as a toast rather than raised: raising terminates the
            event, whereas the snapshot below must still reach the monitoring panels.
            """
            try:
                raw = None
                if job_id:
                    raw = jobs_manager.get_job_status(job_id)
                    code = raw.get("error_code")
                    if code in SECURITY_ALERT_CODES and job_id not in _security_warned:
                        _security_warned.add(job_id)
                        gr.Warning(security_alert_toast(lang_value, code))
                return compute_poll_state(jobs_manager, job_id, lang_value, raw_status=raw)
            except StudioJobsApiError as exc:
                return PollState(
                    status={"scope": "platform", "error_code": exc.code},
                    error_text=platform_error_diagnostics(lang_value, exc, getattr(jobs_manager, "base_url", "")),
                    keep_polling=False,
                )

        def poll_handler(job_id: str, lang_value: str) -> tuple:
            """Fast lifecycle poll: refresh every panel and deactivate on terminal state."""
            state = poll_snapshot(job_id, lang_value)
            nav_visible = len(state.image_artifacts) > 1
            return (
                state.status,
                state.error_text,
                state.banner,
                state.logs,
                state.artifacts,
                gr.update(visible=nav_visible),
                _artifact_selector_update(state.image_artifacts, lang_value),
                gr.update(visible=nav_visible),
                _image_preview_update(state.image_artifacts, lang_value),
                state.recent,
                state.artifact_metadata,
                _artifact_download_update(state.artifact_metadata),
                gr.Timer(value=POLL_FAST_SECONDS, active=state.keep_polling),
            )

        def sync_handler(job_id: str, lang_value: str) -> tuple:
            """Slow background sync: refresh panels without touching the fast timer."""
            state = poll_snapshot(job_id, lang_value)
            nav_visible = len(state.image_artifacts) > 1
            return (
                state.status,
                state.error_text,
                state.banner,
                state.logs,
                state.artifacts,
                gr.update(visible=nav_visible),
                _artifact_selector_update(state.image_artifacts, lang_value),
                gr.update(visible=nav_visible),
                _image_preview_update(state.image_artifacts, lang_value),
                state.recent,
                state.artifact_metadata,
                _artifact_download_update(state.artifact_metadata),
            )

        def submit_job_handler(
            task_type: str,
            model_path: str,
            data_source: str,
            output_dir: str,
            conf: float,
            device: str,
            allowed_paths_str: str,
            lang_value: str,
        ) -> tuple[str, str, Any, str]:
            """Submit a job and (re)activate the fast lifecycle timer."""
            # Parse allowed_paths from comma-separated string
            allowed_paths = [p.strip() for p in allowed_paths_str.split(",") if p.strip()]

            try:
                job_id, _message = jobs_manager.submit_job(
                    task_type=task_type,
                    model_path=model_path,
                    data_source=data_source,
                    output_dir=output_dir,
                    conf=conf,
                    device=device,
                    allowed_paths=allowed_paths,
                )
            except StudioJobsApiError as exc:
                diagnostics = platform_error_diagnostics(lang_value, exc, getattr(jobs_manager, "base_url", ""))
                gr.Warning(diagnostics)
                return "", diagnostics, gr.Timer(value=POLL_FAST_SECONDS, active=False), diagnostics

            # A fresh submission may reuse the security-warning guard.
            _security_warned.discard(job_id)

            return (
                job_id,
                get_text(lang_value, "msg.job_submitted").format(job_id=job_id),
                gr.Timer(value=POLL_FAST_SECONDS, active=True),
                "",
            )

        def cancel_job_handler(job_id: str, lang_value: str) -> tuple[str, Any, str]:
            """Request cancellation, surfacing localized warnings for invalid states.

            Invalid requests (no selection, unknown job, terminal state) queue a
            one-shot gr.Warning toast and return a fallback update: the localized
            warning in the message panel and a deactivated fast poll timer.
            """
            if not job_id:
                message = get_text(lang_value, "msg.no_job_selected")
                gr.Warning(message)
                return message, gr.Timer(value=POLL_FAST_SECONDS, active=False), ""
            try:
                raw = jobs_manager.get_job_status(job_id)
            except StudioJobsApiError as exc:
                diagnostics = platform_error_diagnostics(lang_value, exc, getattr(jobs_manager, "base_url", ""))
                gr.Warning(diagnostics)
                return diagnostics, gr.Timer(value=POLL_FAST_SECONDS, active=False), diagnostics
            if raw.get("status") == "NOT_FOUND":
                message = get_text(lang_value, "msg.job_not_found")
                gr.Warning(message)
                return message, gr.Timer(value=POLL_FAST_SECONDS, active=False), ""
            if is_terminal_status(raw.get("status", "")):
                message = get_text(lang_value, "msg.terminal_state").format(status=raw.get("status", ""))
                gr.Warning(message)
                return message, gr.Timer(value=POLL_FAST_SECONDS, active=False), ""

            try:
                jobs_manager.cancel_job(job_id)
            except StudioJobsApiError as exc:
                diagnostics = platform_error_diagnostics(lang_value, exc, getattr(jobs_manager, "base_url", ""))
                gr.Warning(diagnostics)
                return diagnostics, gr.Timer(value=POLL_FAST_SECONDS, active=False), diagnostics
            return (
                get_text(lang_value, "msg.cancel_requested").format(job_id=job_id),
                gr.Timer(value=POLL_FAST_SECONDS, active=True),
                "",
            )

        # ==================== Event Bindings ====================

        # NOTE: this zone deliberately has no language listener. Language changes
        # are broadcast by the host app (app.py) through
        # jobs_tab_language_updates() -> _language_outputs; binding a change
        # listener here would let the host's programmatic writes re-trigger this
        # zone and form a bidirectional event loop.

        def _on_task_type_change(task_type: str) -> tuple[Any, Any, Any]:
            """Repopulate output_dir, data_source and model_path for the selected task type.

            train/val need a dataset YAML (coco8.yaml) while predict needs an image
            source; switching tasks updates the whole form at once so stale values
            from the previous task (e.g. data_source=bus.jpg) never reach the engine.
            """
            preset = TASK_FORM_PRESETS.get(task_type, TASK_FORM_PRESETS["predict"])
            return (
                gr.update(value=preset["output_dir"]),
                gr.update(value=preset["data_source"]),
                gr.update(value=preset["model_path"]),
            )

        task_type_radio.change(
            fn=_on_task_type_change,
            inputs=task_type_radio,
            outputs=[output_dir_txt, data_source_txt, model_path_txt],
        )

        def _make_form_reset(field_key: str):
            """Build a click handler restoring one form field from the active task preset.

            The handler reads the current task_type and writes the preset value for
            ``field_key`` back into its own textbox. It only touches the form fields —
            never language state or language outputs — so it cannot disturb the
            unidirectional language broadcast owned by the host app (app.py).

            Args:
                field_key: Key into TASK_FORM_PRESETS ("model_path" or "data_source").

            Returns:
                Callable (task_type: str) -> gr.update restoring the preset value.
            """

            def _reset_to_preset(task_type: str) -> Any:
                preset = TASK_FORM_PRESETS.get(task_type, TASK_FORM_PRESETS["predict"])
                return gr.update(value=preset[field_key])

            return _reset_to_preset

        model_path_reset_btn.click(
            fn=_make_form_reset("model_path"),
            inputs=task_type_radio,
            outputs=model_path_txt,
        )
        data_source_reset_btn.click(
            fn=_make_form_reset("data_source"),
            inputs=task_type_radio,
            outputs=data_source_txt,
        )

        def on_artifact_select(selected: str, metadata: list[ArtifactMetadata], lang_value: str) -> tuple[Any, Any]:
            """Switch the preview image when the user picks a different artifact.

            Args:
                selected: Selected image filename (or ``None`` when cleared).
                metadata: Dual-path artifact metadata from the latest poll.
                lang_value: ISO language code used to localize the label.

            Returns:
                Any: A ``gr.update`` with the matched path and localized label.
            """
            paths = [item["preview_path"] for item in metadata if item.get("preview_path")]
            match = next((p for p in paths if selected and Path(p).name == selected), None)
            filename = Path(match).name if match else None
            return (
                gr.update(value=match, label=_artifact_preview_label(lang_value, filename)),
                _artifact_download_update(metadata, filename),
            )

        artifact_selector.change(
            fn=on_artifact_select,
            inputs=[artifact_selector, artifact_metadata_state, lang_state],
            outputs=[artifacts_image, download_artifact_btn],
        )

        def _make_artifact_step(step: int):
            """Return a handler advancing the preview by ``step`` (cyclic).

            Args:
                step: Offset to advance (+1 next, -1 prev).
            """

            def _step(selected: str | None, metadata: list[ArtifactMetadata], lang_value: str) -> tuple[Any, Any, Any]:
                paths = [item["preview_path"] for item in metadata if item.get("preview_path")]
                filename, path = _cycle_artifact(paths, selected, step)
                return (
                    gr.update(value=filename),
                    gr.update(value=path, label=_artifact_preview_label(lang_value, filename)),
                    _artifact_download_update(metadata, filename),
                )

            return _step

        prev_btn.click(
            fn=_make_artifact_step(-1),
            inputs=[artifact_selector, artifact_metadata_state, lang_state],
            outputs=[artifact_selector, artifacts_image, download_artifact_btn],
        )
        next_btn.click(
            fn=_make_artifact_step(1),
            inputs=[artifact_selector, artifact_metadata_state, lang_state],
            outputs=[artifact_selector, artifacts_image, download_artifact_btn],
        )

        def on_artifact_table_select(
            evt: gr.SelectData, metadata: list[ArtifactMetadata], lang_value: str
        ) -> tuple[Any, Any, Any]:
            """Preview an image row selected in the artifacts table.

            Non-image rows (e.g. ``.pt``/``.csv``) are ignored so the current
            preview stays untouched. The selected image also syncs the selector.

            Args:
                evt: Gradio selection event carrying the clicked row index.
                metadata: Dual-path artifact metadata from the latest poll.
                lang_value: ISO language code used to localize the label.

            Returns:
                tuple[Any, Any]: Updates for the selector value and the preview
                image (path + localized label); empty updates when ignored.
            """
            row = evt.index[0] if evt.index else -1
            if row < 0 or row >= len(metadata):
                return gr.update(), gr.update(), gr.update()
            artifact = metadata[row]
            filename = artifact["filename"]
            if not artifact["is_image"]:
                if lang_value == "zh":
                    gr.Info(f"'{filename}' 不是图片文件，无法预览。")
                else:
                    gr.Info(f"'{filename}' is not an image file and cannot be previewed.")
                return gr.update(), gr.update(), _artifact_download_update(metadata, filename)
            local_path = artifact["preview_path"]
            if not local_path:
                return gr.update(), gr.update(), _artifact_download_update(metadata, filename)
            return (
                gr.update(value=Path(filename).name),
                gr.update(value=local_path, label=_artifact_preview_label(lang_value, Path(filename).name)),
                _artifact_download_update(metadata, filename),
            )

        artifacts_list.select(
            fn=on_artifact_table_select,
            inputs=[artifact_metadata_state, lang_state],
            outputs=[artifact_selector, artifacts_image, download_artifact_btn],
        )

        def open_output_folder(job_id: str, metadata: list[ArtifactMetadata], lang_value: str) -> None:
            """Open the current job's specific output folder in the OS file manager.

            Uses the source path retained in the artifact callback state. Windows
            uses ``os.startfile``, macOS uses ``open`` and Linux uses ``xdg-open``;
            a missing folder surfaces a localized warning.
            """

            def _warn() -> None:
                if lang_value == "zh":
                    gr.Warning("输出目录尚不存在。")
                else:
                    gr.Warning("Output directory does not exist.")

            # Walk every artifact's parent chain to find the directory whose
            # name contains this job's id (the job-specific root folder).
            target: Path | None = None
            for artifact in metadata:
                source_path = artifact.get("source_path", "")
                if not source_path or source_path.startswith(("http://", "https://")):
                    continue
                first = Path(source_path)
                for parent in first.parents:
                    if job_id in parent.name:
                        target = parent
                        break
                if target is not None:
                    break

            if target is None or not target.is_dir():
                _warn()
                return

            try:
                if os.name == "nt":
                    os.startfile(str(target))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(target)])
                else:
                    subprocess.Popen(["xdg-open", str(target)])
            except OSError:
                _warn()

        open_folder_btn.click(
            fn=open_output_folder,
            inputs=[job_id_display, artifact_metadata_state, lang_state],
            outputs=[],
        )

        submit_btn.click(
            fn=submit_job_handler,
            inputs=[
                task_type_radio,
                model_path_txt,
                data_source_txt,
                output_dir_txt,
                conf_slider,
                device_txt,
                allowed_paths_txt,
                lang_state,
            ],
            outputs=[job_id_display, submit_msg, poll_timer, error_box],
        ).then(
            fn=poll_handler,
            inputs=[job_id_display, lang_state],
            outputs=[
                status_display,
                error_box,
                banner_md,
                logs_console,
                artifacts_list,
                prev_btn,
                artifact_selector,
                next_btn,
                artifacts_image,
                recent_jobs_table,
                artifact_metadata_state,
                download_artifact_btn,
                poll_timer,
            ],
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id=POLL_CONCURRENCY_ID,
        )

        poll_timer.tick(
            fn=poll_handler,
            inputs=[job_id_display, lang_state],
            outputs=[
                status_display,
                error_box,
                banner_md,
                logs_console,
                artifacts_list,
                prev_btn,
                artifact_selector,
                next_btn,
                artifacts_image,
                recent_jobs_table,
                artifact_metadata_state,
                download_artifact_btn,
                poll_timer,
            ],
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id=POLL_CONCURRENCY_ID,
        )

        sync_timer.tick(
            fn=sync_handler,
            inputs=[job_id_display, lang_state],
            outputs=[
                status_display,
                error_box,
                banner_md,
                logs_console,
                artifacts_list,
                prev_btn,
                artifact_selector,
                next_btn,
                artifacts_image,
                recent_jobs_table,
                artifact_metadata_state,
                download_artifact_btn,
            ],
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id=POLL_CONCURRENCY_ID,
        )

        cancel_job_btn.click(
            fn=cancel_job_handler,
            inputs=[job_id_display, lang_state],
            outputs=[submit_msg, poll_timer, error_box],
        ).then(
            fn=poll_handler,
            inputs=[job_id_display, lang_state],
            outputs=[
                status_display,
                error_box,
                banner_md,
                logs_console,
                artifacts_list,
                prev_btn,
                artifact_selector,
                next_btn,
                artifacts_image,
                recent_jobs_table,
                artifact_metadata_state,
                download_artifact_btn,
                poll_timer,
            ],
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id=POLL_CONCURRENCY_ID,
        )

    # Language-state lifting handles for the host app (app.py): the top level owns
    # the language choice and broadcasts relabels into this zone. The language state
    # is consumed by polling handlers to localize status panels.
    jobs_tab._language_state = lang_state
    jobs_tab._language_outputs = language_outputs

    return jobs_tab
