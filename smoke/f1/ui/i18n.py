"""Localization (i18n) dictionaries for the YOLO-Master Jobs Tab UI.

This module decouples UI-facing strings from the JobsManager backend so that
backend API messages (asserted by the test suites) remain stable while the UI
can be presented in English or Simplified Chinese.

The mapping covers component labels, button texts, placeholders, tooltips,
status labels, user messages and security-alert copy. Unknown languages and
missing keys always fall back to English.

Example:
    >>> from smoke.f1.ui.i18n import get_text
    >>> get_text("zh", "button.submit")
    '🔥 提交任务'
"""

from __future__ import annotations

DEFAULT_LANGUAGE = "en"

#: Column headers for the artifacts and recent-jobs dataframes, per language.
COLUMNS: dict[str, dict[str, list[str]]] = {
    "artifacts": {
        "en": ["Filename", "Path"],
        "zh": ["文件名", "路径"],
    },
    "recent": {
        "en": ["Job ID", "Task Type", "Status", "Created At"],
        "zh": ["任务 ID", "任务类型", "状态", "创建时间"],
    },
}

I18N: dict[str, dict[str, str]] = {
    "en": {
        # Language selector
        "lang.label": "Language",
        # Panel headers
        "tab.title": "📋 Jobs Management",
        "panel.submit": "🚀 Submit Job",
        # Submission form
        "field.task_type": "Task Type",
        "field.model_path": "Model Path",
        "field.model_path.placeholder": "yolov8n.pt or ./ckpts/model.pt",
        "field.data_source": "Data Source",
        "field.data_source.placeholder": "Image/video/directory path",
        "field.output_dir": "Output Directory",
        "field.output_dir.placeholder": "runs/predict",
        "accordion.hyperparams": "⚙️ Hyperparameters",
        "field.conf": "Confidence Threshold",
        "field.device": "Device (0 for GPU, cpu for CPU)",
        "accordion.security": "🔒 Security Constraints",
        "field.allowed_paths": "Allowed Paths (comma-separated)",
        "field.allowed_paths.info": "Whitelist of allowed directory roots",
        "security.policy": (
            "**Security Policy**: Shell execution is **permanently disabled**. "
            "All paths are validated against the whitelist."
        ),
        # Buttons
        "button.submit": "🔥 Submit Job",
        "button.cancel": "🚫 Cancel Job",
        # Monitoring sub-tabs
        "subtab.status": "📊 Status Monitor",
        "field.job_id": "Current Job ID",
        "field.status": "Job Status",
        "field.error": "Error Diagnostics",
        "subtab.logs": "📜 Live Logs",
        "field.logs": "Execution Logs",
        "subtab.artifacts": "📁 Artifacts",
        "df.artifacts": "Generated Artifacts",
        "hint.artifacts": "**Download**: Click on an artifact path to copy it, then retrieve the file from your file explorer",
        "subtab.recent": "🕒 Recent Jobs",
        "df.recent": "Recent Jobs",
        "poll.note": (
            "🔄 Status, logs and artifacts refresh automatically every second while a job is active; "
            "high-frequency polling stops once the job reaches a terminal state."
        ),
        # Status labels (raw backend status -> localized display text)
        "status.PENDING": "Pending",
        "status.RUNNING": "Running",
        "status.COMPLETED": "Completed",
        "status.FAILED": "Failed",
        "status.NOT_FOUND": "Not Found",
        # User messages (placeholders: {job_id}, {status})
        "msg.no_job_selected": "⚠️ No job selected",
        "msg.job_submitted": "✅ Job {job_id} submitted",
        "msg.cancel_requested": "✅ Cancellation requested for {job_id}",
        "msg.job_not_found": "❌ Job not found",
        "msg.terminal_state": "⚠️ Job already in terminal state: {status}",
        # Security / validation alerts
        "alert.SEC_ERR_001.title": "🔒 Security Policy Violation",
        "alert.SEC_ERR_001.body": (
            "The job was blocked by the fail-closed security policy. "
            "Verify that every input/output path resides inside the allowed-paths whitelist."
        ),
        "alert.PARAM_VALIDATION_FAILED.title": "⚠️ Parameter Validation Failed",
        "alert.PARAM_VALIDATION_FAILED.body": (
            "The job parameters failed validation. Review the submitted model path, data source and output directory."
        ),
        "alert.generic.title": "❌ Job Failed",
    },
    "zh": {
        # Language selector
        "lang.label": "语言",
        # Panel headers
        "tab.title": "📋 任务管理",
        "panel.submit": "🚀 提交任务",
        # Submission form
        "field.task_type": "任务类型",
        "field.model_path": "模型路径",
        "field.model_path.placeholder": "yolov8n.pt 或 ./ckpts/model.pt",
        "field.data_source": "数据源",
        "field.data_source.placeholder": "图片/视频/目录路径",
        "field.output_dir": "输出目录",
        "field.output_dir.placeholder": "runs/predict",
        "accordion.hyperparams": "⚙️ 超参数",
        "field.conf": "置信度阈值",
        "field.device": "计算设备（0 为 GPU，cpu 为 CPU）",
        "accordion.security": "🔒 安全约束",
        "field.allowed_paths": "允许路径（逗号分隔）",
        "field.allowed_paths.info": "允许的目录根路径白名单",
        "security.policy": "**安全策略**：Shell 执行被**永久禁用**。所有路径均须通过白名单校验。",
        # Buttons
        "button.submit": "🔥 提交任务",
        "button.cancel": "🚫 取消任务",
        # Monitoring sub-tabs
        "subtab.status": "📊 状态监控",
        "field.job_id": "当前任务 ID",
        "field.status": "任务状态",
        "field.error": "错误诊断",
        "subtab.logs": "📜 实时日志",
        "field.logs": "执行日志",
        "subtab.artifacts": "📁 产物列表",
        "df.artifacts": "生成的产物",
        "hint.artifacts": "**下载**：点击产物路径即可复制，然后通过文件浏览器获取文件",
        "subtab.recent": "🕒 最近任务",
        "df.recent": "最近任务",
        "poll.note": "🔄 任务运行期间状态、日志与产物每秒自动刷新；任务进入终态后自动停止高频轮询。",
        # Status labels
        "status.PENDING": "等待中",
        "status.RUNNING": "运行中",
        "status.COMPLETED": "已完成",
        "status.FAILED": "失败",
        "status.NOT_FOUND": "未找到",
        # User messages
        "msg.no_job_selected": "⚠️ 未选择任务",
        "msg.job_submitted": "✅ 任务 {job_id} 已提交",
        "msg.cancel_requested": "✅ 已请求取消任务 {job_id}",
        "msg.job_not_found": "❌ 未找到任务",
        "msg.terminal_state": "⚠️ 任务已处于终态：{status}",
        # Security / validation alerts
        "alert.SEC_ERR_001.title": "🔒 安全策略违规",
        "alert.SEC_ERR_001.body": "该任务被 fail-closed 安全策略拦截。请确保所有输入/输出路径均位于允许路径白名单内。",
        "alert.PARAM_VALIDATION_FAILED.title": "⚠️ 参数校验失败",
        "alert.PARAM_VALIDATION_FAILED.body": "任务参数未通过校验。请检查提交的模型路径、数据源与输出目录。",
        "alert.generic.title": "❌ 任务失败",
    },
}


def get_text(lang: str | None, key: str) -> str:
    """Return the localized string for ``key`` in ``lang``, falling back to English.

    Args:
        lang: ISO language code ("en" or "zh"); unknown codes fall back to English.
        key: Dictionary key defined in I18N.

    Returns:
        str: The localized string, or the key itself when undefined.

    Example:
        >>> get_text("en", "button.submit")
        '🔥 Submit Job'
        >>> get_text("zh", "button.submit")
        '🔥 提交任务'
        >>> get_text(None, "button.submit")
        '🔥 Submit Job'
        >>> get_text("de", "button.submit")
        '🔥 Submit Job'
        >>> get_text("en", "missing.key")
        'missing.key'
    """
    if lang not in I18N:
        lang = DEFAULT_LANGUAGE
    return I18N[lang].get(key, I18N[DEFAULT_LANGUAGE].get(key, key))


def get_columns(lang: str | None, table: str) -> list[str]:
    """Return the localized dataframe column headers for a given table.

    Args:
        lang: ISO language code ("en" or "zh"); unknown codes fall back to English.
        table: Table identifier ("artifacts" or "recent").

    Returns:
        list[str]: Localized column headers, falling back to English.

    Example:
        >>> get_columns("zh", "artifacts")
        ['文件名', '路径']
        >>> get_columns("en", "recent")
        ['Job ID', 'Task Type', 'Status', 'Created At']
        >>> get_columns("de", "artifacts")
        ['Filename', 'Path']
    """
    if lang not in COLUMNS.get(table, {}):
        lang = DEFAULT_LANGUAGE
    return COLUMNS.get(table, {}).get(lang, COLUMNS.get(table, {}).get(DEFAULT_LANGUAGE, []))
