"""Demo application for Jobs Tab integration with YOLO-Master Studio.

This script demonstrates how to integrate the Jobs Tab into the existing Gradio WebUI.
It provides a minimal example that can be extended with the full Studio UI from app.py.

Usage:
    python smoke/f1/ui/demo_jobs_tab.py

Requirements:
    - gradio >= 4.0
    - ultralytics >= 8.4
    - pydantic >= 2.0
"""

from __future__ import annotations

import gradio as gr

from smoke.f1.ui.jobs_tab import JobsManager, create_jobs_tab


def create_demo_app() -> gr.Blocks:
    """Create demo application with Jobs Tab.

    Returns:
        gr.Blocks: Gradio app with Jobs Tab
    """
    # Initialize shared jobs manager
    jobs_manager = JobsManager()

    with gr.Blocks(title="YOLO-Master Studio - Jobs Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🚀 YOLO-Master Studio - Jobs Management Demo")
        gr.Markdown(
            """
            This demo showcases the **Jobs Tab** for YOLO-Master Studio Platform.

            **Features**:
            - Submit tasks (Predict, Train, Export, Diagnose)
            - Real-time status monitoring with lifecycle state tracking
            - Live log streaming console
            - Artifact explorer with download capability
            - Graceful cancellation handling

            **Security**:
            - Fail-closed path whitelisting (all paths validated against allowed_paths)
            - Shell execution permanently disabled
            - Path traversal protection
            """
        )

        with gr.Tabs():
            # Jobs Tab
            with gr.TabItem("📋 Jobs"):
                create_jobs_tab(jobs_manager)

            # Quick Start Guide
            with gr.TabItem("📖 Quick Start"):
                gr.Markdown(
                    """
                    ## Quick Start Guide

                    ### 1. Submit a Job

                    **Predict Task (Default)**:
                    1. Navigate to **Jobs** tab
                    2. Keep default settings:
                       - Task Type: `predict`
                       - Model Path: `yolov8n.pt`
                       - Data Source: `ultralytics/assets/bus.jpg`
                       - Output Directory: `runs/predict`
                    3. Click **🔥 Submit Job**
                    4. Monitor status in **Status Monitor** tab

                    ### 2. Monitor Execution

                    - **📊 Status Monitor**: View real-time job status, duration, and error diagnostics
                    - **📜 Live Logs**: Stream execution logs (click 🔄 Refresh Logs)
                    - **📁 Artifacts**: Download generated outputs (annotated images, labels, metadata)

                    ### 3. Cancel a Job

                    1. Copy Job ID from status display
                    2. Click **🚫 Cancel Job** button
                    3. Job will transition to FAILED with USER_CANCELLED error code

                    ### 4. Security Constraints

                    **Path Whitelisting**:
                    - All input/output paths MUST be within `allowed_paths`
                    - Default: `. (current dir), ultralytics/assets, runs, ckpts`
                    - Path traversal attempts (e.g., `../../etc/passwd`) are rejected

                    **Shell Execution**:
                    - Permanently disabled (`allow_shell=False`)
                    - Jobs requesting shell access immediately fail with `SEC_ERR_001`

                    ### 5. Task Types

                    | Task Type | Parameters | Artifacts |
                    |-----------|------------|-----------|
                    | **predict** | model_path, data_source, conf, device | Annotated images, labels |
                    | **train** | model_path, data_source, epochs, device | Checkpoints, metrics |
                    | **export** | model_path, format | Exported model (.onnx, .tflite) |
                    | **diagnose** | output_dir | System diagnostics (JSON, TXT) |

                    ### 6. Troubleshooting

                    **Job Stuck in PENDING**:
                    - Check if model file exists at specified path
                    - Verify data_source path is valid and within allowed_paths

                    **Job Failed with SEC_ERR_001**:
                    - Security policy violation detected
                    - Check path whitelisting constraints
                    - Ensure all paths are within allowed_paths

                    **No Artifacts Generated**:
                    - Job may have failed during execution
                    - Check error diagnostics in Status Monitor tab
                    - Review logs for exception details
                    """
                )

            # API Reference
            with gr.TabItem("📚 API Reference"):
                gr.Markdown(
                    """
                    ## API Reference

                    ### JobsManager Class

                    ```python
                    from smoke.f1.ui.jobs_tab import JobsManager

                    manager = JobsManager()

                    # Submit a job
                    job_id, message = manager.submit_job(
                        task_type="predict",
                        model_path="yolov8n.pt",
                        data_source="bus.jpg",
                        output_dir="runs/predict",
                        conf=0.25,
                        device="0",
                        allowed_paths=[".", "runs"],
                    )

                    # Get job status
                    status = manager.get_job_status(job_id)
                    print(status["status"], status["duration"])

                    # Get logs
                    logs = manager.get_job_logs(job_id)
                    print(logs)

                    # Get artifacts
                    artifacts = manager.get_job_artifacts(job_id)
                    for name, path in artifacts:
                        print(f"{name}: {path}")

                    # Cancel job
                    result = manager.cancel_job(job_id)
                    print(result)
                    ```

                    ### create_jobs_tab Function

                    ```python
                    from smoke.f1.ui.jobs_tab import create_jobs_tab

                    jobs_manager = JobsManager()
                    jobs_tab = create_jobs_tab(jobs_manager)

                    # Integrate into existing Gradio app
                    with gr.Blocks() as app:
                        with gr.Tabs():
                            with gr.TabItem("Jobs"):
                                jobs_tab.render()
                    ```

                    ### JobRequest Contract

                    ```python
                    from smoke.f1.test_f1_smoke import JobRequest, TaskType, SecurityConstraints

                    job = JobRequest(
                        job_id="test-001",
                        task_type=TaskType.PREDICT,
                        params={
                            "model_path": "yolov8n.pt",
                            "data_source": "bus.jpg",
                            "conf": 0.25,
                            "device": "0",
                        },
                        output={"output_dir": "runs/predict"},
                        security_constraints=SecurityConstraints(
                            path_whitelisted=True,
                            allow_shell=False,
                            allowed_paths=[".", "runs"],
                        ),
                    )
                    ```
                    """
                )

    return app


def main() -> None:
    """Launch demo application."""
    print("Starting YOLO-Master Studio Jobs Tab Demo...")
    print("Navigate to http://localhost:7860 to access the UI")

    app = create_demo_app()
    app.launch(
        share=False,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7860,
    )


if __name__ == "__main__":
    main()
