"""Integration tests for Jobs Tab i18n, adaptive polling state and app wiring.

Covers:
    - i18n lookups (English, Simplified Chinese, fallbacks)
    - compute_poll_state adaptive deactivation and security-alert banners
    - Jobs Tab construction (two timers, no manual refresh buttons)
    - app.py build_app() headless wiring (top-level tabs, timers, singleton manager)

Run:
    pytest smoke/f1/ui/test_app_integration.py -v
"""

from __future__ import annotations

import gradio as gr
import pytest

from smoke.f1.ui.i18n import DEFAULT_LANGUAGE, get_columns, get_text
from smoke.f1.ui.jobs_tab import (
    SECURITY_ALERT_CODES,
    JobsManager,
    alert_banner,
    compute_poll_state,
    create_jobs_tab,
    is_terminal_status,
    security_alert_toast,
)


class FakeJobsManager:
    """Duck-typed JobsManager stub returning canned backend responses."""

    def __init__(self, status, logs="log line", artifacts=(), recent=()):
        self._status = dict(status)
        self._logs = logs
        self._artifacts = list(artifacts)
        self._recent = [dict(r) for r in recent]

    def get_job_status(self, _job_id):
        return dict(self._status)

    def get_job_logs(self, _job_id):
        return self._logs

    def get_job_artifacts(self, _job_id):
        return list(self._artifacts)

    def list_recent_jobs(self, limit=10):
        return self._recent[:limit]


RUNNING_STATUS = {
    "status": "RUNNING",
    "duration": "3.0s",
    "error_code": None,
    "error_message": None,
    "artifact_count": 2,
}

SECURITY_STATUS = {
    "status": "FAILED",
    "duration": "0.1s",
    "error_code": "SEC_ERR_001",
    "error_message": "Security policy violation: Data_source path '../../etc/passwd' not in whitelist",
    "artifact_count": 0,
}


class TestI18N:
    """Localization lookup behavior."""

    def test_get_text_english(self):
        assert get_text("en", "button.submit") == "🔥 Submit Job"

    def test_get_text_chinese(self):
        assert get_text("zh", "button.submit") == "🔥 提交作业"
        assert get_text("zh", "status.FAILED") == "失败"

    def test_get_text_unknown_language_falls_back_to_english(self):
        assert get_text("de", "button.submit") == "🔥 Submit Job"
        assert get_text(None, "button.submit") == "🔥 Submit Job"

    def test_get_text_missing_key_returns_key(self):
        assert get_text("en", "missing.key") == "missing.key"

    def test_get_columns_localized_and_fallback(self):
        assert get_columns("zh", "artifacts") == ["文件名", "路径"]
        assert get_columns("en", "recent") == ["Job ID", "Task Type", "Status", "Created At"]
        assert get_columns("de", "artifacts") == ["Filename", "Path"]

    def test_default_language_is_english(self):
        assert DEFAULT_LANGUAGE == "en"


class TestPollingState:
    """Adaptive polling: keep_polling flag, localized panels and security banners."""

    def test_active_job_keeps_polling(self):
        state = compute_poll_state(FakeJobsManager(RUNNING_STATUS), "j1", "zh")

        assert state.keep_polling is True
        assert state.status["status"] == "RUNNING"
        assert state.status["status_label"] == "运行中"
        assert state.banner == ""
        assert state.error_text == ""

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
        assert state.status["status"] == "NO_SELECTION"

    def test_not_found_stops_polling(self):
        manager = FakeJobsManager({"status": "NOT_FOUND", "message": "Job not found"})
        state = compute_poll_state(manager, "j1", "zh")

        assert state.keep_polling is False
        assert state.status["status_label"] == "未找到"

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

    def test_jobs_tab_has_language_selector(self):
        tab = create_jobs_tab(JobsManager(), lang="zh")

        radios = [b for b in tab.blocks.values() if isinstance(b, gr.Radio)]
        assert any(b.label == "语言" for b in radios)


class TestAppIntegration:
    """app.py integration: Jobs tab mounted into the top-level tab container."""

    def test_build_app_wiring(self, tmp_path):
        from app import YOLO_Master_WebUI

        ui = YOLO_Master_WebUI(str(tmp_path))
        app = ui.build_app()

        assert isinstance(app, gr.Blocks)
        assert ui.jobs_manager is not None

        # Top-level tabs include the inference studio and the Jobs tab
        tabs = [getattr(b, "label", "") for b in app.blocks.values() if type(b).__name__ == "Tab"]
        assert "🖼️ Inference Studio" in tabs
        assert "📋 Jobs" in tabs

        # Jobs tab contributes the adaptive polling timers (fast + slow sync)
        timers = [b for b in app.blocks.values() if isinstance(b, gr.Timer)]
        assert len(timers) >= 2

        # Manual refresh buttons are replaced by automatic polling
        buttons = [b.value for b in app.blocks.values() if isinstance(b, gr.Button)]
        assert not any("refresh" in (v or "").lower() for v in buttons)

    def test_build_app_launches_headless(self, tmp_path):
        from app import YOLO_Master_WebUI

        ui = YOLO_Master_WebUI(str(tmp_path))
        app = ui.build_app()

        app.launch(prevent_thread_lock=True, quiet=True)
        try:
            assert app.server_name is not None
        finally:
            app.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
