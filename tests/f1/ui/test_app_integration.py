"""Integration tests for Jobs Tab i18n, adaptive polling state and app wiring.

Covers:
    - i18n lookups (English, Simplified Chinese, fallbacks)
    - compute_poll_state adaptive deactivation and security-alert banners
    - Jobs Tab construction (two timers, no manual refresh buttons)
    - app.py build_app() headless wiring (top-level tabs, timers, Studio API client)
    - unidirectional language broadcast: the top-level radio is the single source
      of truth and the only language selector, the Jobs zone has no language
      listener, and every relabel payload is position-aligned with its output
      component (regression for the language switch crash)

Run:
    pytest f1/ui/test_app_integration.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pytest

from f1.ui.i18n import DEFAULT_LANGUAGE, get_columns, get_text
from f1.ui.jobs_tab import (
    POLL_CONCURRENCY_ID,
    POLL_FAST_SECONDS,
    POLL_SLOW_SECONDS,
    SECURITY_ALERT_CODES,
    JobsManager,
    alert_banner,
    compute_poll_state,
    create_jobs_tab,
    is_terminal_status,
    jobs_tab_language_updates,
    security_alert_toast,
)


class FakeJobsManager:
    """Duck-typed JobsManager stub returning canned backend responses."""

    def __init__(self, status, logs="log line", artifacts=(), recent=(), image_artifacts=None):
        self._status = dict(status)
        self._logs = logs
        self._artifacts = list(artifacts)
        self._recent = [dict(r) for r in recent]
        self._image_artifacts = image_artifacts or []

    def get_job_status(self, _job_id):
        return dict(self._status)

    def get_job_logs(self, _job_id):
        return self._logs

    def get_job_artifacts(self, _job_id):
        return list(self._artifacts)

    def get_job_image_artifacts(self, _job_id):
        return list(self._image_artifacts)

    def list_recent_jobs(self, limit=10):
        return self._recent[:limit]


def _walk_layout_ids(cfg):
    """Return component ids in layout traversal order (catches duplicate mounts)."""
    ids = []

    def walk(node):
        ids.append(node["id"])
        for child in node.get("children", []):
            walk(child)

    for root in cfg["layout"]["children"]:
        walk(root)
    return ids


def _count_marker(cfg, needle):
    """Count string occurrences of needle anywhere inside the config tree."""
    total = 0

    def scan(obj):
        nonlocal total
        if isinstance(obj, str):
            total += obj.count(needle)
        elif isinstance(obj, dict):
            for v in obj.values():
                scan(v)
        elif isinstance(obj, list):
            for v in obj:
                scan(v)

    scan(cfg)
    return total


def _find_ids(cfg, comp_type, *, label=None, value=None):
    """Find component ids by type and optional label/value."""
    ids = []
    for c in cfg["components"]:
        if c.get("type") != comp_type:
            continue
        props = c.get("props") or {}
        if label is not None and props.get("label") != label:
            continue
        if value is not None and props.get("value") != value:
            continue
        ids.append(c["id"])
    return ids


#: gr.update() prop keys valid per component class (the keys actually used by the
#: studio_relabels / jobs_tab_language_updates payloads). A payload key outside
#: this map would be silently ignored by Gradio, i.e. an untranslated component.
_ALLOWED_UPDATE_PROPS: dict[type, set[str]] = {
    gr.Radio: {"value", "label"},
    gr.Markdown: {"value", "label"},
    gr.Textbox: {"value", "label", "placeholder", "info"},
    gr.Dataframe: {"headers", "label", "value"},
    gr.Tab: {"label"},  # covers gr.TabItem
    gr.Accordion: {"label"},
    gr.Slider: {"label", "value"},
    gr.Button: {"value", "label"},
    gr.JSON: {"label", "value"},
    gr.Dropdown: {"value", "label", "choices"},
    gr.Image: {"label", "value"},
    gr.Number: {"label", "value"},
    gr.Checkbox: {"label", "value"},
    gr.CheckboxGroup: {"label", "value"},
    gr.Gallery: {"label", "visible"},
}


def _assert_update_fits_component(update, component):
    """Assert a gr.update payload only sets props valid for the component's type."""
    assert isinstance(update, dict) and update.get("__type__") == "update", (
        f"{type(component).__name__} at must receive a gr.update payload, got {update!r}"
    )
    comp_type = next(t for t in _ALLOWED_UPDATE_PROPS if isinstance(component, t))
    unexpected = set(update) - {"__type__"} - _ALLOWED_UPDATE_PROPS[comp_type]
    assert not unexpected, f"{type(component).__name__} received unsupported update keys: {unexpected}"


RUNNING_STATUS = {
    "status": "RUNNING",
    "duration": 3.0,
    "error_code": None,
    "error_message": None,
    "artifact_count": 2,
}

SECURITY_STATUS = {
    "status": "FAILED",
    "duration": 0.1,
    "error_code": "SEC_ERR_001",
    "error_message": "Security policy violation: Data_source path '../../etc/passwd' not in whitelist",
    "artifact_count": 0,
}


class TestI18N:
    """Localization lookup behavior."""

    def test_get_text_english(self):
        assert get_text("en", "button.submit") == "🔥 Submit Job"

    def test_get_text_chinese(self):
        assert get_text("zh", "button.submit") == "🔥 提交任务"
        assert get_text("zh", "button.cancel") == "🚫 取消任务"
        assert get_text("zh", "tab.title") == "📋 任务管理"
        assert get_text("zh", "subtab.artifacts") == "📁 产物列表"
        assert get_text("zh", "field.device") == "计算设备（0 为 GPU，cpu 为 CPU）"
        assert get_text("zh", "status.FAILED") == "失败"

    def test_get_text_unknown_language_falls_back_to_english(self):
        assert get_text("de", "button.submit") == "🔥 Submit Job"
        assert get_text(None, "button.submit") == "🔥 Submit Job"

    def test_get_text_missing_key_returns_key(self):
        assert get_text("en", "missing.key") == "missing.key"

    def test_form_field_placeholders_show_example_paths(self):
        """model_path and data_source placeholders carry concrete example inputs."""
        assert "e.g.," in get_text("en", "field.model_path.placeholder")
        assert "runs/train/weights/best.pt" in get_text("en", "field.model_path.placeholder")
        assert "coco8.yaml" in get_text("en", "field.data_source.placeholder")
        assert "path/to/image.jpg" in get_text("en", "field.data_source.placeholder")
        assert "例如：" in get_text("zh", "field.model_path.placeholder")
        assert "coco8.yaml" in get_text("zh", "field.data_source.placeholder")

    def test_get_columns_localized_and_fallback(self):
        assert get_columns("zh", "artifacts") == ["文件名", "路径"]
        assert get_columns("zh", "recent") == ["任务 ID", "任务类型", "状态", "创建时间"]
        assert get_columns("en", "recent") == ["Job ID", "Task Type", "Status", "Created At"]
        assert get_columns("de", "artifacts") == ["Filename", "Path"]

    def test_default_language_is_english(self):
        assert DEFAULT_LANGUAGE == "en"


class TestPollingState:
    """Adaptive polling: keep_polling flag, localized panels and security banners."""

    def test_active_job_keeps_polling(self):
        state = compute_poll_state(FakeJobsManager(RUNNING_STATUS), "j1", "zh")

        assert state.keep_polling is True
        assert state.status == {
            "job_id": "j1",
            "status": "RUNNING",
            "duration": 3.0,
            "error_code": None,
            "error_message": None,
            "artifact_count": 2,
        }
        assert state.banner == ""
        assert state.error_text == ""

    def test_running_job_does_not_synthesize_duration(self):
        status = {
            **RUNNING_STATUS,
            "duration": None,
            "started_at": "2026-09-11T00:00:00+00:00",
            "completed_at": "2026-09-11T00:00:03+00:00",
        }

        state = compute_poll_state(FakeJobsManager(status), "j1", "en")

        assert state.status["duration"] is None

    def test_terminal_legacy_fallback_uses_started_and_completed_at(self):
        status = {
            **RUNNING_STATUS,
            "status": "COMPLETED",
            "duration": None,
            "started_at": "2026-09-11T00:00:01+00:00",
            "completed_at": "2026-09-11T00:00:03.500000+00:00",
        }

        state = compute_poll_state(FakeJobsManager(status), "j1", "en")

        assert state.status["duration"] == 2.5

    def test_api_duration_takes_precedence_over_timestamp_fallback(self):
        status = {
            **RUNNING_STATUS,
            "status": "COMPLETED",
            "duration": 0.0,
            "started_at": "2026-09-11T00:00:01+00:00",
            "completed_at": "2026-09-11T00:00:03+00:00",
        }

        state = compute_poll_state(FakeJobsManager(status), "j1", "en")

        assert state.status["duration"] == 0.0

    def test_terminal_legacy_fallback_without_started_at_stays_null(self):
        status = {
            **RUNNING_STATUS,
            "status": "FAILED",
            "duration": None,
            "started_at": None,
            "completed_at": "2026-09-11T00:00:03+00:00",
        }

        state = compute_poll_state(FakeJobsManager(status), "j1", "en")

        assert state.status["duration"] is None

    def test_terminal_job_stops_polling_and_includes_final_artifacts(self):
        manager = FakeJobsManager(
            {**RUNNING_STATUS, "status": "COMPLETED", "artifact_count": 1},
            artifacts=[("bus.jpg", "C:/runs/bus.jpg")],
        )
        state = compute_poll_state(manager, "j1", "en")

        assert state.keep_polling is False
        assert state.artifacts == [["bus.jpg", "C:/runs/bus.jpg"]]

    def test_security_error_maps_to_localized_banner(self):
        state = compute_poll_state(FakeJobsManager(SECURITY_STATUS), "j1", "zh")

        assert state.keep_polling is False
        assert "安全策略违规" in state.banner
        assert "SEC_ERR_001" in state.banner
        assert "SEC_ERR_001" in state.error_text

    def test_param_validation_error_maps_to_banner(self):
        manager = FakeJobsManager({**SECURITY_STATUS, "error_code": "PARAM_VALIDATION_FAILED"})
        state = compute_poll_state(manager, "j1", "en")

        assert "Parameter Validation Failed" in state.banner
        assert state.keep_polling is False

    def test_no_selection_stops_polling(self):
        state = compute_poll_state(FakeJobsManager(RUNNING_STATUS), "", "en")

        assert state.keep_polling is False
        assert state.status == {"status": "NO_SELECTION"}

    def test_not_found_stops_polling(self):
        manager = FakeJobsManager({"status": "NOT_FOUND", "message": "Job not found"})
        state = compute_poll_state(manager, "j1", "zh")

        assert state.keep_polling is False
        assert state.status == {
            "job_id": "j1",
            "status": "NOT_FOUND",
            "duration": None,
            "error_code": None,
            "error_message": None,
            "artifact_count": None,
        }

    def test_is_terminal_status_matrix(self):
        assert not is_terminal_status("PENDING")
        assert not is_terminal_status("RUNNING")
        for terminal in ("COMPLETED", "FAILED", "CANCELLED", "NOT_FOUND"):
            assert is_terminal_status(terminal)

    def test_security_alert_toast_and_banner_localized(self):
        assert "安全策略违规" in security_alert_toast("zh", "SEC_ERR_001")
        banner = alert_banner("en", "SEC_ERR_001", "boom")
        assert "Security Policy Violation" in banner and "boom" in banner

    def test_security_alert_codes_registered(self):
        assert SECURITY_ALERT_CODES == {"SEC_ERR_001", "PARAM_VALIDATION_FAILED"}

    def test_recent_jobs_rows_format_created_at_as_local_time(self):
        raw = datetime(2026, 9, 2, 10, 14, 44, 136175, tzinfo=timezone.utc).isoformat()
        manager = FakeJobsManager(
            RUNNING_STATUS,
            recent=[{"job_id": "j1", "task_type": "predict", "status": "RUNNING", "created_at": raw}],
        )
        state = compute_poll_state(manager, "j1", "en")

        expected = datetime.fromisoformat(raw).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        assert state.recent == [["j1", "predict", "RUNNING", expected]]
        assert "T" not in expected

    def test_recent_jobs_rows_fallback_for_missing_or_malformed_timestamp(self):
        manager = FakeJobsManager(
            RUNNING_STATUS,
            recent=[
                {"job_id": "j1", "task_type": "predict", "status": "RUNNING", "created_at": ""},
                {"job_id": "j2", "task_type": "export", "status": "FAILED", "created_at": "garbage"},
            ],
        )
        state = compute_poll_state(manager, "j1", "en")

        assert state.recent[0][3] == "-"
        assert state.recent[1][3] == "garbage"


class TestJobsTabWiring:
    """Structural wiring of the Jobs Tab Blocks."""

    def test_jobs_tab_has_two_timers_and_no_refresh_buttons(self):
        tab = create_jobs_tab(JobsManager())

        timers = [b for b in tab.blocks.values() if isinstance(b, gr.Timer)]
        buttons = [b.value for b in tab.blocks.values() if isinstance(b, gr.Button)]

        assert len(timers) == 2
        assert "🔥 Submit Job" in buttons
        assert "🚫 Cancel Job" in buttons
        assert not any("refresh" in (v or "").lower() for v in buttons)

    def test_jobs_tab_has_no_language_radio(self):
        """The Jobs tab no longer has its own language radio; it was removed to
        eliminate the inner language selector from the left panel.
        """
        tab = create_jobs_tab(JobsManager(), lang="zh")
        radios = [b for b in tab.blocks.values() if isinstance(b, gr.Radio)]
        # Only the task_type radio remains (no language radio)
        assert not any("语言" in (b.label or "") or "Language" in (b.label or "") for b in radios)

    def test_task_type_change_rewires_form_fields(self):
        """Switching task type repopulates output_dir, data_source and model_path together.

        Regression for the E2E train/val failures: the form stayed stuck on the
        previous task's values (e.g. data_source=bus.jpg), which the train/val
        engines reject. The radio's change event must feed all three fields at once.
        """
        tab = create_jobs_tab(JobsManager())
        radio = next(b for b in tab.blocks.values() if isinstance(b, gr.Radio))

        def textbox(value):
            return next(b for b in tab.blocks.values() if isinstance(b, gr.Textbox) and b.value == value)

        output_dir = textbox("runs/predict")
        data_source = textbox("ultralytics/assets/bus.jpg")
        model_path = textbox("./ckpts/yolov8n.pt")

        # The radio owns exactly one change event, feeding the three form fields
        bf = next(bf for bf in tab.fns.values() if bf.fn and (radio._id, "change") in bf.targets)
        assert [c._id for c in bf.outputs] == [output_dir._id, data_source._id, model_path._id]

        expected = {
            "predict": ("runs/predict", "ultralytics/assets/bus.jpg", "./ckpts/yolov8n.pt"),
            "train": ("runs/train", "coco8.yaml", "./ckpts/yolov8n.pt"),
            "val": ("runs/val", "coco8.yaml", "./ckpts/yolov8n.pt"),
            "export": ("runs/export", "", "./ckpts/yolov8n.pt"),
            "diagnose": ("runs/diagnose", "", ""),
        }
        for task_type, (out_dir, source, model) in expected.items():
            assert bf.fn(task_type) == (
                {"__type__": "update", "value": out_dir},
                {"__type__": "update", "value": source},
                {"__type__": "update", "value": model},
            )

    def test_form_field_placeholders_show_example_paths(self):
        """The two textboxes mount with the example-path placeholder text."""
        tab = create_jobs_tab(JobsManager())
        placeholders = {tb.label: tb.placeholder for tb in tab.blocks.values() if isinstance(tb, gr.Textbox)}

        assert "e.g.," in placeholders["Model Path"]
        assert "runs/train/weights/best.pt" in placeholders["Model Path"]
        assert "e.g.," in placeholders["Data Source"]
        assert "coco8.yaml" in placeholders["Data Source"]
        assert "path/to/image.jpg" in placeholders["Data Source"]

    def test_form_reset_buttons_restore_task_presets(self):
        """Each 🔄 reset button restores its own field from the active task's preset.

        Every button owns exactly one click event (task_type in, its own textbox
        out) and returns the TASK_FORM_PRESETS value for the active task. The
        buttons sit below their textboxes and are localized via the language
        broadcast (their text switches with the app language).
        """
        tab = create_jobs_tab(JobsManager())

        def textbox(value):
            return next(b for b in tab.blocks.values() if isinstance(b, gr.Textbox) and b.value == value)

        model_path = textbox("./ckpts/yolov8n.pt")
        data_source = textbox("ultralytics/assets/bus.jpg")
        radio = next(b for b in tab.blocks.values() if isinstance(b, gr.Radio))

        reset_btns = [
            b for b in tab.blocks.values() if isinstance(b, gr.Button) and (b.value or "").startswith("🔄 Reset")
        ]
        assert len(reset_btns) == 2

        def click_binding(btn):
            matches = [bf for bf in tab.fns.values() if bf.fn and (btn._id, "click") in bf.targets]
            assert len(matches) == 1, "each reset button must own exactly one click event"
            return matches[0]

        bindings = {bf.outputs[0]._id: bf for bf in (click_binding(btn) for btn in reset_btns)}
        assert set(bindings) == {model_path._id, data_source._id}
        for bf in bindings.values():
            assert [c._id for c in bf.inputs] == [radio._id]
            assert len(bf.outputs) == 1

        model_bf = bindings[model_path._id]
        data_bf = bindings[data_source._id]
        assert model_bf.fn("train") == {"__type__": "update", "value": "./ckpts/yolov8n.pt"}
        assert data_bf.fn("train") == {"__type__": "update", "value": "coco8.yaml"}
        assert data_bf.fn("export") == {"__type__": "update", "value": ""}
        assert model_bf.fn("diagnose") == {"__type__": "update", "value": ""}

        # Reset buttons are localized via the language broadcast
        lang_output_ids = {c._id for c in tab._language_outputs}
        assert all(btn._id in lang_output_ids for btn in reset_btns)


class TestI18NBinding:
    """Unidirectional language contract: the Jobs zone never listens for language changes."""

    def test_jobs_tab_has_no_language_radio_in_config(self):
        """The inner language radio was removed; no radio with label 'Language' exists."""
        tab = create_jobs_tab(JobsManager())
        cfg = tab.get_config_file()

        lang_radio_ids = _find_ids(cfg, "radio", label="Language")
        assert len(lang_radio_ids) == 0, "Inner language radio should be removed"

    def test_jobs_tab_payload_matches_component_types(self):
        """jobs_tab_language_updates stays position-aligned with _language_outputs.

        Element i of the payload updates the component at position i of the output
        tuple; a misalignment would silently skip or double-relabel components.
        """
        tab = create_jobs_tab(JobsManager())
        payload = jobs_tab_language_updates("zh")
        components = tab._language_outputs

        assert len(payload) == len(components) == 33
        # Position 0 is the raw language value for the shared language State
        assert payload[0] == "zh"
        assert isinstance(components[0], gr.State)
        for update, component in zip(payload[1:], components[1:]):
            _assert_update_fits_component(update, component)


class TestAppLanguageBroadcast:
    """Unidirectional language broadcast wired in app.py (single source of truth)."""

    def _build(self, tmp_path):
        from app import YOLO_Master_WebUI

        ui = YOLO_Master_WebUI(str(tmp_path))
        return ui, ui.build_app()

    def _top_radio_change_dep(self, app):
        """Return (config, top radio, its change dependency) for the built app."""
        cfg = app.get_config_file()
        top = next(
            b
            for b in app.blocks.values()
            if isinstance(b, gr.Radio) and b.label == "Language" and b.interactive is not False
        )
        deps = [d for d in cfg["dependencies"] if (top._id, "change") in d["targets"]]
        assert len(deps) == 1, "the top-level language radio must own the only language change event"
        return cfg, top, deps[0]

    def test_top_radio_is_the_only_language_listener(self, tmp_path):
        _, app = self._build(tmp_path)
        _cfg, top, dep = self._top_radio_change_dep(app)

        # The inner language radio was removed; the top-level radio is the only
        # language radio in the entire app.
        all_radios = [b for b in app.blocks.values() if isinstance(b, gr.Radio)]
        lang_radios = [r for r in all_radios if "Language" in (r.label or "") or "语言" in (r.label or "")]
        assert len(lang_radios) == 1
        assert lang_radios[0] is top

        # Regression guard for the bidirectional ping-pong crash: the top radio must
        # not write back to itself from its own handler.
        assert top._id not in dep["outputs"]

    def test_broadcast_targets_every_component_exactly_once(self, tmp_path):
        _, app = self._build(tmp_path)
        cfg, _top, dep = self._top_radio_change_dep(app)

        outputs = dep["outputs"]
        assert len(outputs) == 1 + 2 + 22 + 33  # state + outer tabs + studio zone + jobs zone
        assert len(set(outputs)) == len(outputs), "no localizable component bound twice"

        bound = set(outputs)
        # Lifted chrome: both outer tab labels
        assert _find_ids(cfg, "tabitem", label="🖼️ Inference Studio")[0] in bound
        assert _find_ids(cfg, "tabitem", label="📋 Jobs")[0] in bound
        # Studio zone — incl. the Settings header and Task selector that stayed
        # English in the crash report
        assert _find_ids(cfg, "markdown", value="### 🛠 Settings")[0] in bound
        assert _find_ids(cfg, "radio", label="Task")[0] in bound
        # Jobs zone markers
        assert _find_ids(cfg, "markdown", value="# 📋 Jobs Management")[0] in bound
        assert _find_ids(cfg, "markdown", value="### 🚀 Submit Job")[0] in bound
        for label in ("📊 Status Monitor", "📜 Live Logs", "📁 Artifacts", "🕒 Recent Jobs"):
            assert _find_ids(cfg, "tabitem", label=label)[0] in bound
        # Timers are runtime components, never relabeled
        timer_ids = {c["id"] for c in cfg["components"] if c.get("type") == "timer"}
        assert timer_ids.isdisjoint(bound)

    def test_studio_and_jobs_payloads_align_with_broadcast_outputs(self, tmp_path):
        from app import studio_relabels

        _, app = self._build(tmp_path)
        _cfg, _top, dep = self._top_radio_change_dep(app)

        studio_updates = studio_relabels("zh")
        jobs_updates = jobs_tab_language_updates("zh")
        output_ids = dep["outputs"]

        # chrome (state + 2 outer tabs), then the studio zone, then the jobs zone
        assert len(output_ids) == 1 + 2 + len(studio_updates) + len(jobs_updates)
        studio_ids = output_ids[3 : 3 + len(studio_updates)]
        jobs_ids = output_ids[3 + len(studio_updates) :]

        blocks = app.blocks
        assert isinstance(blocks[output_ids[0]], gr.State)
        assert isinstance(blocks[output_ids[1]], gr.Tab)
        assert isinstance(blocks[output_ids[2]], gr.Tab)
        for update, comp_id in zip(studio_updates, studio_ids):
            _assert_update_fits_component(update, blocks[comp_id])
        assert isinstance(blocks[jobs_ids[0]], gr.State)
        assert jobs_updates[0] == "zh"
        for update, comp_id in zip(jobs_updates[1:], jobs_ids[1:]):
            _assert_update_fits_component(update, blocks[comp_id])

    def test_language_handler_switches_all_zones_without_crashing(self, tmp_path):
        """Invoke the wired handler exactly as Gradio does when zh-CN is selected."""
        _, app = self._build(tmp_path)
        _cfg, top, dep = self._top_radio_change_dep(app)

        # The serialized config flags backend_fn as a bool; the live callable is
        # registered on the BlockFunction keyed by the radio's change target.
        block_fn = next(bf for bf in app.fns.values() if bf.fn and (top._id, "change") in bf.targets)
        result = block_fn.fn("zh")
        assert isinstance(result, tuple)
        assert len(result) == len(dep["outputs"])

        # Lifted chrome
        assert result[0] == "zh"
        assert result[1]["label"] == "🖼️ 推理工作台"
        assert result[2]["label"] == "📋 任务管理"
        # Studio zone: Settings header and Task selector (crash-report regression)
        assert "设置" in result[3 + 6]["value"]
        assert result[3 + 7]["label"] == "任务"


class TestModelDropdownDisplay:
    """Inference Studio Model Weights dropdown shows clean filenames, not full paths."""

    def _build_ui(self, tmp_path):
        from app import YOLO_Master_WebUI

        ckpts = tmp_path / "ckpts"
        (ckpts / "seg").mkdir(parents=True)
        detect_pt = ckpts / "yolov8n.pt"
        detect_pt.write_bytes(b"dummy")
        seg_pt = ckpts / "seg" / "yolov8n-seg.pt"
        seg_pt.write_bytes(b"dummy")

        ui = YOLO_Master_WebUI(str(ckpts))
        return ui, detect_pt, seg_pt

    def test_dropdown_shows_clean_filenames(self, tmp_path):
        """Initial dropdown choices and value are bare filenames; full paths stay
        available on model_map for backend resolution."""
        ui, detect_pt, seg_pt = self._build_ui(tmp_path)
        app = ui.build_app()

        model_dd = next(b for b in app.blocks.values() if isinstance(b, gr.Dropdown) and b.label == "Model Weights")

        # Gradio 6 normalizes choices to (value, label) tuples on the component
        display_choices = [c[0] if isinstance(c, tuple) else c for c in model_dd.choices]
        assert display_choices == ["yolov8n.pt"]
        assert model_dd.value == "yolov8n.pt"
        for choice in display_choices:
            assert Path(choice).name == choice  # no directory components

        # Backend keeps the full absolute paths for execution
        assert ui.model_map["detect"] == [str(detect_pt.absolute())]
        assert ui.model_map["seg"] == [str(seg_pt.absolute())]

    def test_task_switch_repopulates_with_clean_names(self, tmp_path):
        """update_model_dropdown returns display names only, for every task."""
        ui, _detect_pt, _seg_pt = self._build_ui(tmp_path)

        update = ui.update_model_dropdown("seg")
        assert update["choices"] == ["yolov8n-seg.pt"]
        assert update["value"] == "yolov8n-seg.pt"
        for choice in update["choices"]:
            assert Path(choice).name == choice

    def test_refresh_models_rebuilds_display_map(self, tmp_path):
        """A checkpoint added after the initial scan appears as a clean name."""
        ui, _detect_pt, _seg_pt = self._build_ui(tmp_path)

        (Path(ui.ckpts_root) / "yolov8n-obb.pt").write_bytes(b"dummy")
        update = ui.refresh_models("obb")
        assert update["choices"] == ["yolov8n-obb.pt"]
        assert update["value"] == "yolov8n-obb.pt"

    def test_resolve_checkpoint_path_maps_display_name_to_full_path(self, tmp_path):
        """Dropdown names resolve back to the real checkpoint; unknown values pass through."""
        ui, detect_pt, seg_pt = self._build_ui(tmp_path)

        resolved = ui.resolve_checkpoint_path("yolov8n.pt", "detect")
        assert resolved == str(detect_pt.absolute())
        assert Path(resolved).exists()

        # Cross-task fallback: a stale selection after a task switch still resolves
        assert ui.resolve_checkpoint_path("yolov8n-seg.pt", "detect") == str(seg_pt.absolute())

        # Unknown names and custom paths pass through unchanged
        assert ui.resolve_checkpoint_path("", "detect") == ""
        assert ui.resolve_checkpoint_path("custom/model.pt", "detect") == "custom/model.pt"


class TestSynchronousInferenceBoundary:
    """The classic Gradio inference path remains a direct synchronous YOLO call."""

    def test_inference_still_calls_loaded_model_directly(self, tmp_path, monkeypatch):
        import numpy as np

        from app import YOLO_Master_WebUI

        class FakeResult:
            def __init__(self):
                self.boxes = []
                self.speed = {"inference": 3.5}

            def plot(self):
                return np.zeros((4, 4, 3), dtype=np.uint8)

        class FakeModel:
            def __init__(self):
                self.names = {}
                self.calls = []

            def __call__(self, image, **kwargs):
                self.calls.append((image, kwargs))
                return [FakeResult()]

        ui = YOLO_Master_WebUI(str(tmp_path))
        model = FakeModel()
        monkeypatch.setattr(ui.model_manager, "load_model", lambda model_path, task: model)
        monkeypatch.setattr(ui.model_manager, "get_current_model_info", lambda: "cpu")
        ui.model_manager.current_model_path = "yolov8n.pt"

        result_image, detections, summary = ui.inference(
            "detect",
            np.zeros((4, 4, 3), dtype=np.uint8),
            "yolov8n.pt",
            "",
            0.25,
            0.7,
            "cpu",
            100,
            2,
            True,
            [],
        )

        assert result_image.shape == (4, 4, 3)
        assert detections.empty
        assert "Inference Done" in summary
        assert len(model.calls) == 1


class TestAppIntegration:
    """app.py integration: Jobs tab mounted into the top-level tab container."""

    def test_build_app_wiring(self, tmp_path):
        from app import YOLO_Master_WebUI
        from f1.jobs_manager import JobsManager
        from f1.ui.studio_jobs_client import StudioJobsApiClient

        ui = YOLO_Master_WebUI(str(tmp_path))
        app = ui.build_app()

        assert isinstance(app, gr.Blocks)
        assert isinstance(ui.jobs_client, StudioJobsApiClient)
        assert not isinstance(ui.jobs_client, JobsManager)
        assert not hasattr(ui, "jobs_manager")

        # Top-level tabs include the inference studio and the Jobs tab
        tabs = [getattr(b, "label", "") for b in app.blocks.values() if type(b).__name__ == "Tab"]
        assert "🖼️ Inference Studio" in tabs
        assert "📋 Jobs" in tabs

        # Jobs tab contributes the adaptive polling timers (fast + slow sync)
        timers = [b for b in app.blocks.values() if isinstance(b, gr.Timer)]
        assert len(timers) >= 2

        # The Jobs zone replaces manual refresh with automatic polling; the only
        # refresh button is the Inference Studio's model-list refresh.
        buttons = [b.value for b in app.blocks.values() if isinstance(b, gr.Button)]
        refresh_buttons = [v for v in buttons if "refresh" in (v or "").lower()]
        assert len(refresh_buttons) == 1  # Only the Inference Studio model refresh

    def test_build_app_layout_has_single_mount(self, tmp_path):
        from app import YOLO_Master_WebUI

        app = YOLO_Master_WebUI(str(tmp_path)).build_app()
        cfg = app.get_config_file()

        # Every component must appear exactly once in the layout tree: a component
        # mounted twice renders two DOM copies (duplicated Jobs tab) and desyncs
        # the language-switch event bindings.
        ids = _walk_layout_ids(cfg)
        assert len(ids) == len(set(ids)), "Jobs components mounted more than once in the layout tree"

        for marker in (
            "Jobs Management",
            "🔥 Submit Job",
            "📊 Status Monitor",
            "📜 Live Logs",
            "📁 Artifacts",
            "🕒 Recent Jobs",
        ):
            assert _count_marker(cfg, marker) == 1, f"{marker!r} expected once, found {_count_marker(cfg, marker)}"

        # Exactly the two Jobs Tab timers (fast lifecycle + slow sync), no strays
        assert sum(1 for c in cfg["components"] if c.get("type") == "timer") == 2

    def test_build_app_launches_headless(self, tmp_path):
        from app import YOLO_Master_WebUI

        ui = YOLO_Master_WebUI(str(tmp_path))
        app = ui.build_app()

        app.launch(prevent_thread_lock=True, quiet=True)
        try:
            assert app.server_name is not None
        finally:
            app.close()


class TestStudioApiPlatformErrors:
    """Studio API outages are localized platform diagnostics, never Job failures."""

    @staticmethod
    def _build_unavailable_tab(lang: str):
        import requests

        from f1.ui.studio_jobs_client import StudioJobsApiClient

        class UnavailableSession:
            def request(self, method, url, **kwargs):
                raise requests.ConnectionError("connection refused")

        return create_jobs_tab(StudioJobsApiClient("http://studio.test", session=UnavailableSession()), lang)

    @pytest.mark.parametrize(
        ("lang", "expected"),
        [("en", "Studio Job API is unavailable"), ("zh", "无法连接位于")],
    )
    def test_initial_diagnostics_are_bilingual(self, lang, expected):
        tab = self._build_unavailable_tab(lang)
        error_box = next(block for block in tab.blocks.values() if isinstance(block, gr.Textbox) and block.lines == 3)

        assert expected in error_box.value
        assert "FAILED" not in error_box.value

    def test_failed_submission_creates_no_job_identity(self, monkeypatch):
        tab = self._build_unavailable_tab("zh")
        submit = next(
            block for block in tab.blocks.values() if isinstance(block, gr.Button) and block.value == "🔥 提交任务"
        )
        submit_fn = next(bf.fn for bf in tab.fns.values() if bf.fn and (submit._id, "click") in bf.targets)
        warnings = []
        monkeypatch.setattr("gradio.Warning", lambda message: warnings.append(message))

        job_id, message, timer_update, diagnostics = submit_fn(
            "predict",
            "yolov8n.pt",
            "bus.jpg",
            "runs/predict",
            0.25,
            "cpu",
            "., runs",
            "zh",
        )

        assert job_id == ""
        assert "无法连接位于" in message == diagnostics
        assert isinstance(timer_update, gr.Timer)
        assert timer_update.active is False
        assert warnings == [diagnostics]


class TestStudioApiPollingSynchronization:
    """API-backed polling applies terminal snapshots after active snapshots."""

    class TransitioningBackend:
        base_url = "http://studio.test"

        def __init__(self):
            self.statuses = ["RUNNING", "COMPLETED"]
            self.current_status = "RUNNING"
            self.status_calls = 0

        def get_job_status(self, job_id):
            self.status_calls += 1
            self.current_status = self.statuses.pop(0)
            return {
                "status": self.current_status,
                "duration": 1.0,
                "error_code": None,
                "error_message": None,
                "artifact_count": 2 if self.current_status == "COMPLETED" else 0,
            }

        def get_job_logs(self, job_id):
            return self.current_status

        def get_job_artifacts(self, job_id):
            return [("metrics.csv", "http://studio.test/metrics.csv")] if self.current_status == "COMPLETED" else []

        def get_job_image_artifacts(self, job_id):
            return []

        def list_recent_jobs(self, limit=20):
            return [
                {
                    "job_id": "job-1",
                    "task_type": "predict",
                    "status": self.current_status,
                    "created_at": "2026-09-11T00:00:00+00:00",
                }
            ]

    def test_running_to_completed_refreshes_all_panels_and_stops_fast_timer(self):
        backend = self.TransitioningBackend()
        tab = create_jobs_tab(backend, "en")
        poll_timer = next(
            block for block in tab.blocks.values() if isinstance(block, gr.Timer) and block.value == POLL_FAST_SECONDS
        )
        poll_fn = next(bf.fn for bf in tab.fns.values() if bf.fn and (poll_timer._id, "tick") in bf.targets)

        running = poll_fn("job-1", "en")
        completed = poll_fn("job-1", "en")

        assert running[0]["status"] == "RUNNING"
        assert running[-1].active is True
        assert completed[0]["status"] == "COMPLETED"
        assert completed[0]["artifact_count"] == 2
        assert completed[4] == [["metrics.csv", "http://studio.test/metrics.csv"]]
        assert completed[9][0][2] == "COMPLETED"
        assert completed[-1].active is False
        assert backend.status_calls == 2

    def test_fast_and_slow_timers_share_one_latest_only_queue(self):
        tab = create_jobs_tab(self.TransitioningBackend(), "en")
        timer_ids = {block._id for block in tab.blocks.values() if isinstance(block, gr.Timer)}
        poll_functions = [
            block_function
            for block_function in tab.fns.values()
            if any(component_id in timer_ids and event == "tick" for component_id, event in block_function.targets)
        ]

        assert len(poll_functions) == 2
        assert {block_function.trigger_mode for block_function in poll_functions} == {"always_last"}
        assert {block_function.concurrency_id for block_function in poll_functions} == {POLL_CONCURRENCY_ID}
        assert {block_function.concurrency_limit for block_function in poll_functions} == {1}
        assert {
            tab.blocks[target[0]].value for block_function in poll_functions for target in block_function.targets
        } == {
            POLL_FAST_SECONDS,
            POLL_SLOW_SECONDS,
        }


class TestJobsZoneBroadcastAlignment:
    """Strict 33-item broadcast contract after the artifact-hint removal.

    The bilingual download-hint Markdown was dropped from the Artifacts zone,
    shrinking the jobs-zone broadcast from 34 to 33 positions.  These tests
    pin the exact tail ordering so no future component insertion can drift
    silently.
    """

    def _build_tab(self) -> gr.Blocks:
        return create_jobs_tab(JobsManager(), "en")

    def test_outputs_exactly_33_unique_and_registered(self):
        """The output tuple has 33 distinct components, all live in the Blocks."""
        tab = self._build_tab()
        components = tab._language_outputs

        assert len(components) == 33
        assert len({c._id for c in components}) == 33  # no duplicate outputs
        assert all(c._id in tab.blocks for c in components)

    def test_payload_length_and_language_head_alignment(self):
        """jobs_tab_language_updates returns 1 language head + 32 relabel updates."""
        components = self._build_tab()._language_outputs
        payload = jobs_tab_language_updates("zh")

        assert payload[0] == "zh"
        assert len(payload) == len(components) == 33

    def test_tail_positions_typed_after_hint_removal(self):
        """Type-pin the jobs-zone tail: hint slot is gone, no Markdown in between."""
        components = self._build_tab()._language_outputs

        assert isinstance(components[24], gr.Button)  # open_folder_btn
        assert isinstance(components[25], gr.Dataframe)  # artifacts_list
        assert isinstance(components[26], gr.Button)  # prev_btn
        assert isinstance(components[27], gr.Dropdown)  # artifact_selector
        assert isinstance(components[28], gr.Button)  # next_btn
        assert isinstance(components[29], gr.Image)  # artifacts_image
        assert isinstance(components[30], gr.TabItem)  # recent_tab
        assert isinstance(components[31], gr.Dataframe)  # recent_jobs_table
        assert isinstance(components[32], gr.Markdown)  # poll_note_md (tail anchor)
        # The removed hint was a Markdown between artifacts_image and recent_tab:
        # that gap must stay empty.
        assert not any(isinstance(c, gr.Markdown) for c in components[24:32])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
