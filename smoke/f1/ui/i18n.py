"""Localization (i18n) dictionaries for the YOLO-Master Studio UI.

This module decouples UI-facing strings from the backend (JobsManager,
dispatcher, handlers) so that backend API messages (asserted by the test
suites) remain stable while the UI can be presented in English or Simplified
Chinese. It backs the whole Studio app:

- Jobs Tab zone keys (``tab.*``, ``field.*``, ``button.*``, ...)
- Inference Studio zone keys (``studio.*``, ``app.tab.*``) consumed by the
  top-level language broadcast wired in ``app.py``
- Shared selectors (``LANGUAGE_CHOICES``) and dataframe column headers
  (``COLUMNS``) for both zones

Unknown languages and missing keys always fall back to English.

Example:
    >>> from smoke.f1.ui.i18n import get_text
    >>> get_text("zh", "button.submit")
    '🔥 提交任务'
"""

from __future__ import annotations

DEFAULT_LANGUAGE = "en"

#: Selector choices shared by every language radio in the app (each language is
#: labeled in its own language, so the choices are language-invariant).
LANGUAGE_CHOICES: tuple[tuple[str, str], ...] = (("English", "en"), ("中文", "zh"))

#: Column headers for the artifacts, recent-jobs and detections dataframes, per language.
COLUMNS: dict[str, dict[str, list[str]]] = {
    "artifacts": {
        "en": ["Filename", "Path"],
        "zh": ["文件名", "路径"],
    },
    "recent": {
        "en": ["Job ID", "Task Type", "Status", "Created At"],
        "zh": ["任务 ID", "任务类型", "状态", "创建时间"],
    },
    "detections": {
        "en": ["Class ID", "Class Name", "Confidence", "x1", "y1", "x2", "y2"],
        "zh": ["类别 ID", "类别名称", "置信度", "x1", "y1", "x2", "y2"],
    },
}

I18N: dict[str, dict[str, str]] = {
    "en": {
        # Language selector
        "lang.label": "Language",
        # Top-level tab labels shared by the whole Studio app
        "app.tab.studio": "🖼️ Inference Studio",
        "app.tab.jobs": "📋 Jobs",
        # Inference Studio zone
        "studio.heading": "# 🚀 YOLO-Master Dashboard",
        "studio.settings": "### 🛠 Settings",
        "studio.field.task": "Task",
        "studio.field.model_weights": "Model Weights",
        "studio.field.custom_model_path": "Custom Model Path (file or directory)",
        "studio.field.custom_model_path.placeholder": "./ckpts/yolo_master_n.pt",
        "studio.button.validate": "✅ Validate Path",
        "studio.validate.empty": "⚠️ Please enter a model path.",
        "studio.validate.valid": "✅ Model path is valid.",
        "studio.validate.invalid": "❌ Model path does not exist or is invalid.",
        "studio.accordion.advanced": "⚙️ Advanced Parameters",
        "studio.field.conf": "Confidence (Conf)",
        "studio.field.iou": "IoU Threshold",
        "studio.field.max_objects": "Max Objects",
        "studio.field.line_width": "Line Width",
        "studio.field.device": "Device ID (e.g. 0, cpu)",
        "studio.field.force_cpu": "Force CPU",
        "studio.field.output_options": "Output Options",
        "studio.button.run": "🔥 Start Inference",
        "studio.subtab.visualization": "🖼️ Visualization",
        "studio.subtab.data_analysis": "📊 Data Analysis",
        "studio.field.input_image": "Input Image",
        "studio.field.result_image": "Inference Result",
        "studio.message.waiting": "Waiting for input...",
        "studio.heading.detections": "### Detections Data",
        "studio.df.detections": "Raw Detections",
        # Panel headers
        "tab.title": "📋 Jobs Management",
        "panel.submit": "🚀 Submit Job",
        # Submission form
        "field.task_type": "Task Type",
        "field.model_path": "Model Path",
        "field.model_path.placeholder": "e.g., ./ckpts/yolov8n.pt or runs/train/weights/best.pt",
        "field.data_source": "Data Source",
        "field.data_source.placeholder": "e.g., coco8.yaml, data/custom.yaml, or path/to/image.jpg",
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
        "button.reset_model": "🔄 Reset to default model",
        "button.reset_data": "🔄 Reset to default data source",
        "studio.button.refresh": "🔄 Refresh model list",
        # Monitoring sub-tabs
        "subtab.status": "📊 Status Monitor",
        "field.job_id": "Current Job ID",
        "field.status": "Job Status",
        "field.error": "Error Diagnostics",
        "subtab.logs": "📜 Live Logs",
        "field.logs": "Execution Logs",
        "subtab.artifacts": "📁 Artifacts",
        "df.artifacts": "Generated Artifacts",
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
        # Top-level tab labels shared by the whole Studio app
        "app.tab.studio": "🖼️ 推理工作台",
        "app.tab.jobs": "📋 任务管理",
        # Inference Studio zone
        "studio.heading": "# 🚀 YOLO-Master 仪表盘",
        "studio.settings": "### 🛠 设置",
        "studio.field.task": "任务",
        "studio.field.model_weights": "模型权重",
        "studio.field.custom_model_path": "自定义模型路径（文件或目录）",
        "studio.field.custom_model_path.placeholder": "./ckpts/yolo_master_n.pt",
        "studio.button.validate": "✅ 校验路径",
        "studio.validate.empty": "⚠️ 请输入模型路径。",
        "studio.validate.valid": "✅ 模型路径有效。",
        "studio.validate.invalid": "❌ 模型路径不存在或无效。",
        "studio.accordion.advanced": "⚙️ 高级参数",
        "studio.field.conf": "置信度 (Conf)",
        "studio.field.iou": "IoU 阈值",
        "studio.field.max_objects": "最大目标数",
        "studio.field.line_width": "线宽",
        "studio.field.device": "设备 ID（如 0、cpu）",
        "studio.field.force_cpu": "强制 CPU",
        "studio.field.output_options": "输出选项",
        "studio.button.run": "🔥 开始推理",
        "studio.subtab.visualization": "🖼️ 可视化",
        "studio.subtab.data_analysis": "📊 数据分析",
        "studio.field.input_image": "输入图像",
        "studio.field.result_image": "推理结果",
        "studio.message.waiting": "等待输入...",
        "studio.heading.detections": "### 检测数据",
        "studio.df.detections": "原始检测结果",
        # Panel headers
        "tab.title": "📋 任务管理",
        "panel.submit": "🚀 提交任务",
        # Submission form
        "field.task_type": "任务类型",
        "field.model_path": "模型路径",
        "field.model_path.placeholder": "例如：./ckpts/yolov8n.pt 或 runs/train/weights/best.pt",
        "field.data_source": "数据源",
        "field.data_source.placeholder": "例如：coco8.yaml、data/custom.yaml 或 path/to/image.jpg",
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
        "button.reset_model": "🔄 重置为默认模型",
        "button.reset_data": "🔄 重置为默认数据源",
        "studio.button.refresh": "🔄 刷新模型列表",
        # Monitoring sub-tabs
        "subtab.status": "📊 状态监控",
        "field.job_id": "当前任务 ID",
        "field.status": "任务状态",
        "field.error": "错误诊断",
        "subtab.logs": "📜 实时日志",
        "field.logs": "执行日志",
        "subtab.artifacts": "📁 产物列表",
        "df.artifacts": "生成的产物",
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
