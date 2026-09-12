"""YOLO-Master Gradio WebUI entrypoint.

Launches the interactive Studio in which the classic inference playground runs
synchronously and the F1 Jobs tab routes background work through the Studio Job API.

Localization:
    A top-level language selector lifts the language state above the tab container;
    one shared State drives the Inference Studio zone and the Jobs zone together
    (see :meth:`YOLO_Master_WebUI.build_app` for the unidirectional broadcast
    design: the top selector is the single source of truth).

Usage:
    python start_studio.py  # Recommended: Studio Job API + Gradio
    python app.py           # Gradio only; requires an external Studio Job API
"""

from __future__ import annotations

import gc
import os
import warnings
from pathlib import Path
from typing import Any, ClassVar

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch

from f1.ui.i18n import DEFAULT_LANGUAGE, LANGUAGE_CHOICES, get_columns, get_text
from f1.ui.jobs_tab import create_jobs_tab, jobs_tab_language_updates
from f1.ui.studio_jobs_client import StudioJobsApiClient
from ultralytics import YOLO

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")


class GlobalConfig:
    """Global configuration parameters for easy modification."""

    # Default model files mapping
    DEFAULT_MODELS: ClassVar[dict[str, str]] = {
        "detect": "yolov8n.pt",
        "seg": "yolov8n-seg.pt",
        "cls": "yolov8n-cls.pt",
        "pose": "yolov8n-pose.pt",
        "obb": "yolov8n-obb.pt",
    }
    # Allowed image formats
    IMAGE_EXTENSIONS: ClassVar[set[str]] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    # UI Theme
    THEME: ClassVar = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")


class ModelManager:
    """Handles model scanning, loading, and memory management."""

    def __init__(self, ckpts_root: Path):
        self.ckpts_root = ckpts_root
        self.current_model: YOLO | None = None
        self.current_model_path: str = ""
        self.current_task: str = "detect"

    def scan_checkpoints(self) -> dict[str, list[str]]:
        """Scan the checkpoint directory and categorize models by task."""
        model_map = {k: [] for k in GlobalConfig.DEFAULT_MODELS}

        if not self.ckpts_root.exists():
            return model_map

        # Recursively find all .pt files
        for p in self.ckpts_root.rglob("*.pt"):
            if p.is_dir():
                continue

            path_str = str(p.absolute())
            filename = p.name.lower()
            parent = p.parent.name.lower()

            # Intelligent classification logic
            if "seg" in filename or "seg" in parent:
                model_map["seg"].append(path_str)
            elif "cls" in filename or "class" in filename or "cls" in parent:
                model_map["cls"].append(path_str)
            elif "pose" in filename or "pose" in parent:
                model_map["pose"].append(path_str)
            elif "obb" in filename or "obb" in parent:
                model_map["obb"].append(path_str)
            else:
                model_map["detect"].append(path_str)  # Default to detect

        # Deduplicate and sort
        for k, paths in model_map.items():
            model_map[k] = sorted(set(paths))

        return model_map

    def unload_model(self) -> None:
        """Force clear GPU memory."""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("INFO: Memory cleared.")

    def load_model(self, model_path: str, task: str) -> YOLO:
        """Load model with caching and memory management."""
        target_path = model_path
        if not target_path or not os.path.exists(target_path):
            target_path = GlobalConfig.DEFAULT_MODELS.get(task, "yolov8n.pt")
        else:
            # Support directory path, auto-resolve to weights file
            if os.path.isdir(target_path):
                candidates = [
                    os.path.join(target_path, "weights", "best.pt"),
                    os.path.join(target_path, "weights", "last.pt"),
                    os.path.join(target_path, "best.pt"),
                    os.path.join(target_path, "last.pt"),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        target_path = c
                        break

        if self.current_model is not None and self.current_model_path == target_path:
            return self.current_model

        self.unload_model()

        print(f"INFO: Loading model from {target_path}...")
        try:
            model = YOLO(target_path)
            self.current_model = model
            self.current_model_path = target_path
            self.current_task = task
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def get_current_model_info(self) -> str:
        """Return the device of the currently loaded model."""
        try:
            if self.current_model:
                return str(next(self.current_model.model.parameters()).device)
        except Exception:  # noqa: BLE001, S110 - best-effort device probe, fall back to "unknown"
            pass
        return "unknown"


def studio_relabels(lang_value: str) -> tuple[Any, ...]:
    """Build the localized relabel payload covering every Inference Studio component.

    Pure presentation helper for the app-level language broadcast: element at
    position ``i`` updates the component at position ``i`` of the studio zone
    component tuple assembled in :meth:`YOLO_Master_WebUI.build_app`. Transient
    content (inference result image, summary markdown, detections dataframe
    body) is deliberately excluded so a language switch never discards runtime
    output; only static labels, placeholders and headers are relabeled.

    Args:
        lang_value: ISO language code ("en" or "zh"); unknown codes fall back
            to English via the i18n layer.

    Returns:
        tuple[Any, ...]: One ``gr.update`` per studio zone localizable component.
    """
    return (
        gr.update(label=get_text(lang_value, "studio.subtab.visualization")),  # viz_tabitem
        gr.update(label=get_text(lang_value, "studio.field.input_image")),  # inp_img
        gr.update(label=get_text(lang_value, "studio.field.result_image")),  # out_img
        gr.update(label=get_text(lang_value, "studio.subtab.data_analysis")),  # analysis_tabitem
        gr.update(value=get_text(lang_value, "studio.heading.detections")),  # detections_md
        gr.update(
            label=get_text(lang_value, "studio.df.detections"),
            headers=get_columns(lang_value, "detections"),
        ),  # out_df
        gr.update(value=get_text(lang_value, "studio.settings")),  # settings_md
        gr.update(label=get_text(lang_value, "studio.field.task")),  # task_radio
        gr.update(label=get_text(lang_value, "studio.field.model_weights")),  # model_dd
        gr.update(value=get_text(lang_value, "studio.button.refresh")),  # refresh_btn
        gr.update(
            label=get_text(lang_value, "studio.field.custom_model_path"),
            placeholder=get_text(lang_value, "studio.field.custom_model_path.placeholder"),
        ),  # custom_model_txt
        gr.update(value=get_text(lang_value, "studio.button.validate")),  # validate_btn
        gr.update(label=get_text(lang_value, "studio.accordion.advanced")),  # advanced_accordion
        gr.update(label=get_text(lang_value, "studio.field.conf")),  # conf_slider
        gr.update(label=get_text(lang_value, "studio.field.iou")),  # iou_slider
        gr.update(label=get_text(lang_value, "studio.field.max_objects")),  # max_det_num
        gr.update(label=get_text(lang_value, "studio.field.line_width")),  # line_width_num
        gr.update(label=get_text(lang_value, "studio.field.device")),  # device_txt
        gr.update(label=get_text(lang_value, "studio.field.force_cpu")),  # cpu_chk
        gr.update(label=get_text(lang_value, "studio.field.output_options")),  # options_chk
        gr.update(value=get_text(lang_value, "studio.button.run")),  # run_btn
        gr.update(value=get_text(lang_value, "studio.heading")),  # heading_md
    )


class YOLO_Master_WebUI:
    """Top-level WebUI application hosting the inference studio and the Jobs tab."""

    def __init__(self, ckpts_root: str):
        self.ckpts_root = Path(ckpts_root)
        self.model_manager = ModelManager(self.ckpts_root)
        # model_map keeps FULL checkpoint paths for backend resolution; the dropdown
        # only ever shows clean filenames via the derived display map.
        self.model_map = self.model_manager.scan_checkpoints()
        self.model_display_map = self._display_names(self.model_map)
        # Stateless adapter only: the FastAPI service is the sole owner of
        # JobsManager, lifecycle state, workers and persistence.
        self.jobs_client = StudioJobsApiClient()

    @staticmethod
    def _display_names(model_map: dict[str, list[str]]) -> dict[str, list[str]]:
        """Derive dropdown display names (Path(p).name) from scanned full paths."""
        return {task: [Path(p).name for p in paths] for task, paths in model_map.items()}

    def resolve_checkpoint_path(self, display_name: str, task: str) -> str:
        """Resolve a dropdown display name back to the full scanned checkpoint path.

        The dropdown shows clean filenames (``Path(p).name``); the YOLO engine still
        needs the real file location. The active task's checkpoint list is searched
        first, then every other task's list, so a stale dropdown selection after a
        task switch still resolves. Values that are not scanned display names
        (empty strings, custom paths) pass through unchanged.
        """
        if not display_name:
            return display_name
        task_lists = [self.model_map.get(task, [])]
        task_lists.extend(paths for key, paths in self.model_map.items() if key != task)
        for paths in task_lists:
            for full_path in paths:
                if Path(full_path).name == display_name:
                    return full_path
        return display_name

    def inference(
        self,
        task: str,
        image: np.ndarray,
        model_dropdown: str,
        custom_model_path: str,
        conf: float,
        iou: float,
        device: str,
        max_det: float,
        line_width: float,
        cpu: bool,
        checkboxes: list[str],
    ) -> tuple[np.ndarray | None, pd.DataFrame | None, str]:
        """Run core inference.

        Returns:
            (Annotated Image, Results DataFrame, Summary Text)
        """
        if image is None:
            return None, None, "⚠️ Please upload an image first."

        # 1. Parameter Sanitization
        device_opt = "cpu" if cpu else (device if device else "")
        line_width_opt = int(line_width) if line_width > 0 else None
        max_det_opt = int(max_det)
        options = {k: True for k in checkboxes}

        # Optimization for segmentation task
        if task == "seg" and "retina_masks" not in options:
            options["retina_masks"] = True

        # 2. Model Loading
        # Prioritize custom path, then dropdown. The dropdown value is a display
        # name (clean filename); resolve it back to the full scanned checkpoint
        # path before handing it to the model manager / YOLO engine.
        model_path = (custom_model_path or "").strip()
        if not model_path:
            model_path = self.resolve_checkpoint_path((model_dropdown or "").strip(), task)
        try:
            model = self.model_manager.load_model(model_path, task)
        except Exception as e:  # noqa: BLE001 - report any load failure to the UI
            return image, None, f"❌ Error loading model: {e}"

        # 3. Execution
        try:
            # Gradio input is RGB, but Ultralytics expects BGR for numpy arrays
            # We convert to BGR to ensure correct inference and plotting colors
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            results = model(
                image_bgr,
                conf=conf,
                iou=iou,
                device=device_opt,
                max_det=max_det_opt,
                line_width=line_width_opt,
                **options,
            )
        except Exception as e:  # noqa: BLE001 - report any inference failure to the UI
            return image, None, f"❌ Inference Error: {e}"

        # 4. Result Parsing
        res = results[0]

        # 4.1 Image Processing
        res_img = res.plot()
        res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)  # Convert back to RGB

        # 4.2 Data Extraction (Build DataFrame)
        data_list = []
        if res.boxes:
            for box in res.boxes:
                try:
                    # Compatibility handling: box.cls might be tensor or float
                    cls_id = int(box.cls[0]) if box.cls.numel() > 0 else 0
                    cls_name = model.names[cls_id]
                    conf_val = float(box.conf[0]) if box.conf.numel() > 0 else 0.0
                    coords = box.xyxy[0].tolist()

                    row = {
                        "Class ID": cls_id,
                        "Class Name": cls_name,
                        "Confidence": round(conf_val, 3),
                        "x1": round(coords[0], 1),
                        "y1": round(coords[1], 1),
                        "x2": round(coords[2], 1),
                        "y2": round(coords[3], 1),
                    }
                    data_list.append(row)
                except Exception:  # noqa: BLE001, S110 - skip malformed box entries
                    pass

        df = pd.DataFrame(data_list)

        # 4.3 Summary Info
        speed = res.speed
        infer_time = speed.get("inference", 0.0)
        model_device = self.model_manager.get_current_model_info()

        summary = (
            f"### ✅ Inference Done\n"
            f"- **Model:** `{Path(self.model_manager.current_model_path).name}`\n"
            f"- **Time:** `{infer_time:.1f}ms`\n"
            f"- **Objects:** {len(data_list)}\n"
            f"- **Device:** `{model_device}`"
        )

        return res_img, df, summary

    def describe_model(self, task: str, model_path: str, lang: str = DEFAULT_LANGUAGE) -> str:
        """Validate and describe the model, returning localized messages.

        Args:
            task: Current task type (detect, seg, cls, pose, obb).
            model_path: User-entered custom model path.
            lang: ISO language code for localized output messages.

        Returns:
            str: Localized validation result message.
        """
        if not model_path or not model_path.strip():
            return get_text(lang, "studio.validate.empty")

        path = Path(model_path.strip())
        if not path.exists():
            return get_text(lang, "studio.validate.invalid")

        try:
            # Check if it's a directory, try to find pt file
            if path.is_dir():
                candidates = [
                    path / "weights" / "best.pt",
                    path / "weights" / "last.pt",
                    path / "best.pt",
                    path / "last.pt",
                ]
                found = False
                for c in candidates:
                    if c.exists():
                        path = c
                        found = True
                        break
                if not found:
                    return get_text(lang, "studio.validate.invalid")

            # Load model to verify validity (temporary load, no caching here)
            YOLO(str(path))
            return get_text(lang, "studio.validate.valid")
        except Exception:  # noqa: BLE001 - report any validation failure to the UI
            return get_text(lang, "studio.validate.invalid")

    def update_model_dropdown(self, task: str):
        """UI Event: Update model list when task changes (clean display names only)."""
        choices = self.model_display_map.get(task, [])
        if not choices:
            choices = [GlobalConfig.DEFAULT_MODELS.get(task, "yolov8n.pt")]
        return gr.update(choices=choices, value=choices[0])

    def refresh_models(self, task: str):
        """UI Event: Manually refresh model list."""
        self.model_map = self.model_manager.scan_checkpoints()
        self.model_display_map = self._display_names(self.model_map)
        return self.update_model_dropdown(task)

    def build_app(self) -> gr.Blocks:
        """Assemble the complete Gradio application (not launched yet).

        The Jobs tab shares the top-level tab container with the inference studio and
        uses a stateless client for the Studio Job API (self.jobs_client).

        Language-state lifting (unidirectional broadcast):
            The top-level selector and :class:`gr.State` live above the tab
            container and are the SINGLE SOURCE OF TRUTH for the app-wide language
            choice. Exactly one change event exists in the whole app:

            - Top selector change -> app state, outer tab labels, every studio
              component (``studio_relabels``), and the whole Jobs zone payload
              (``jobs_tab_language_updates``, 27 explicit outputs).

            There is no reverse path: the Jobs zone has no language listener of
            its own, and the top selector never writes to itself, so no write in
            this handler can re-trigger any language event (Gradio 6 re-fires
            ``change`` on programmatic component writes, which made the previous
            bidirectional wiring loop until the server crashed).

        Returns:
            gr.Blocks: The fully wired application; call .launch() to serve it.
        """
        # NOTE: Gradio 6 moved `theme` from the Blocks constructor to launch().
        with gr.Blocks(title="YOLO-Master WebUI") as app:
            # ============ Top-level language broadcast (whole-app scope) ============
            # The lifted state and selector live above the tab container and are the
            # single source of truth: one change event relabels every zone below.
            # Neither zone below has a language listener of its own, so this
            # broadcast is strictly unidirectional.
            app_lang_state = gr.State(DEFAULT_LANGUAGE)
            app_lang_radio = gr.Radio(
                choices=LANGUAGE_CHOICES,
                value=DEFAULT_LANGUAGE,
                label=get_text(DEFAULT_LANGUAGE, "lang.label"),
            )
            with gr.Tabs():
                # ================= Tab 1: Inference Studio =================
                with gr.TabItem(get_text(DEFAULT_LANGUAGE, "app.tab.studio")) as studio_tabitem:
                    heading_md = gr.Markdown("# 🚀 YOLO-Master Dashboard")

                    with gr.Row(equal_height=False):
                        # ================= Sidebar: Control Panel =================
                        with gr.Column(scale=1, variant="panel"):
                            settings_md = gr.Markdown("### 🛠 Settings")

                            # Task and Model Selection
                            with gr.Group():
                                task_radio = gr.Radio(
                                    choices=["detect", "seg", "cls", "pose", "obb"],
                                    value="detect",
                                    label="Task",
                                )
                                model_dd = gr.Dropdown(
                                    choices=self.model_display_map["detect"],
                                    value=self.model_display_map["detect"][0]
                                    if self.model_display_map["detect"]
                                    else None,
                                    label="Model Weights",
                                    interactive=True,
                                )
                                refresh_btn = gr.Button(
                                    get_text(DEFAULT_LANGUAGE, "studio.button.refresh"),
                                    size="sm",
                                    variant="secondary",
                                )
                                custom_model_txt = gr.Textbox(
                                    value="",
                                    label="Custom Model Path (file or directory)",
                                    placeholder="./ckpts/yolo_master_n.pt",
                                    interactive=True,
                                )
                                validate_btn = gr.Button("✅ Validate Path", size="sm")

                            # Advanced Parameters
                            with gr.Accordion("⚙️ Advanced Parameters", open=True) as advanced_accordion:
                                conf_slider = gr.Slider(0, 1, 0.25, step=0.01, label="Confidence (Conf)")
                                iou_slider = gr.Slider(0, 1, 0.7, step=0.01, label="IoU Threshold")

                                with gr.Row():
                                    max_det_num = gr.Number(300, label="Max Objects", precision=0)
                                    line_width_num = gr.Number(0, label="Line Width", precision=0)

                                with gr.Row():
                                    device_txt = gr.Textbox(
                                        "0", label="Device ID (e.g. 0, cpu)", placeholder="0 or cpu"
                                    )
                                    cpu_chk = gr.Checkbox(False, label="Force CPU")

                            # Output Options
                            options_chk = gr.CheckboxGroup(
                                [
                                    "half",
                                    "show",
                                    "save",
                                    "save_txt",
                                    "save_crop",
                                    "hide_labels",
                                    "hide_conf",
                                    "agnostic_nms",
                                    "retina_masks",
                                ],
                                label="Output Options",
                                value=[],
                            )

                            # Run Button
                            run_btn = gr.Button("🔥 Start Inference", variant="primary", size="lg")

                        # ================= Main Area: Display Panel =================
                        with gr.Column(scale=3), gr.Tabs():
                            with gr.TabItem("🖼️ Visualization") as viz_tabitem:
                                with gr.Row():
                                    inp_img = gr.Image(type="numpy", label="Input Image", height=500)
                                    out_img = gr.Image(
                                        type="numpy", label="Inference Result", height=500, interactive=False
                                    )
                                info_md = gr.Markdown(value="Waiting for input...")

                            with gr.TabItem("📊 Data Analysis") as analysis_tabitem:
                                detections_md = gr.Markdown("### Detections Data")
                                out_df = gr.Dataframe(
                                    headers=["Class ID", "Class Name", "Confidence", "x1", "y1", "x2", "y2"],
                                    label="Raw Detections",
                                )

                # ================= Tab 2: Jobs =================
                # Single mount: entering the child Blocks inside this active context
                # auto-embeds it on context exit; an explicit .render() would mount
                # every Jobs component a second time (duplicate tabs in the DOM).
                with gr.TabItem(get_text(DEFAULT_LANGUAGE, "app.tab.jobs")) as jobs_tabitem:
                    jobs_zone = create_jobs_tab(self.jobs_client)

            # ================= Event Binding =================

            # 1. Auto-refresh model list
            task_radio.change(fn=self.update_model_dropdown, inputs=task_radio, outputs=model_dd)
            refresh_btn.click(fn=self.refresh_models, inputs=task_radio, outputs=model_dd)
            validate_btn.click(
                fn=self.describe_model,
                inputs=[task_radio, custom_model_txt, app_lang_state],
                outputs=info_md,
            )

            # 2. Inference Logic
            run_btn.click(
                fn=self.inference,
                inputs=[
                    task_radio,
                    inp_img,
                    model_dd,
                    custom_model_txt,
                    conf_slider,
                    iou_slider,
                    device_txt,
                    max_det_num,
                    line_width_num,
                    cpu_chk,
                    options_chk,
                ],
                outputs=[out_img, out_df, info_md],
            )

            # 3. Unified language broadcast -------------------------------------------------
            # Studio zone localizable components, position-aligned 1-to-1 with the
            # studio_relabels() payload (transient outputs such as info_md are excluded).
            studio_components = [
                viz_tabitem,
                inp_img,
                out_img,
                analysis_tabitem,
                detections_md,
                out_df,
                settings_md,
                task_radio,
                model_dd,
                refresh_btn,
                custom_model_txt,
                validate_btn,
                advanced_accordion,
                conf_slider,
                iou_slider,
                max_det_num,
                line_width_num,
                device_txt,
                cpu_chk,
                options_chk,
                run_btn,
                heading_md,
            ]

            def apply_app_language(lang_value: str) -> tuple[Any, ...]:
                """Broadcast one language choice from the top-level selector to every zone.

                Unidirectional by design: the top-level selector is the single source
                of truth and the only language change listener in the app. The Jobs
                zone has no language listener of its own, and this handler never writes
                back to the top selector, so no output in this tuple can re-trigger any
                language event.
                """
                return (
                    lang_value,  # app_lang_state
                    gr.update(label=get_text(lang_value, "app.tab.studio")),  # studio_tabitem
                    gr.update(label=get_text(lang_value, "app.tab.jobs")),  # jobs_tabitem
                    *studio_relabels(lang_value),
                    *jobs_tab_language_updates(lang_value),
                )

            # Top-level selector -> whole app (studio zone + Jobs zone). The
            # selector is deliberately not among the outputs: a programmatic
            # self-write could re-trigger its own change event in Gradio 6 and
            # loop back into this handler.
            app_lang_radio.change(
                fn=apply_app_language,
                inputs=app_lang_radio,
                outputs=[
                    app_lang_state,
                    studio_tabitem,
                    jobs_tabitem,
                    *studio_components,
                    *jobs_zone._language_outputs,
                ],
            )

        return app

    def launch(self) -> None:
        """Build and launch the WebUI server (blocking call)."""
        app = self.build_app()
        app.launch(share=False, inbrowser=True, theme=GlobalConfig.THEME)


if __name__ == "__main__":
    # Configure your checkpoints path
    CKPTS_DIR = Path(__file__).parent / "ckpts"

    # Create default dir if not exists
    if not CKPTS_DIR.exists():
        CKPTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created default checkpoints dir: {CKPTS_DIR}")

    print("Starting YOLO-Master WebUI...")
    print(f"Scanning models in: {CKPTS_DIR}")

    ui = YOLO_Master_WebUI(str(CKPTS_DIR))
    ui.launch()
