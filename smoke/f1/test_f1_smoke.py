from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# 1. Interface Contract Definitions
class TaskType(str, Enum):
    PREDICT = "predict"
    TRAIN = "train"
    EXPORT = "export"
    DIAGNOSE = "diagnose"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Metadata(BaseModel):
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: str = "anonymous"
    description: str | None = None
    priority: str = "normal"
    tags: list[str] = Field(default_factory=list)


class OutputConfig(BaseModel):
    output_dir: str = "runs/predict/job_20260824_f1_001"
    save_images: bool = True
    save_labels: bool = False
    save_logs: bool = True
    artifacts: list[str] = Field(default_factory=list)


class SecurityConstraints(BaseModel):
    path_whitelisted: bool = True
    allow_shell: bool = False
    allowed_paths: list[str] = Field(default_factory=list)


class RuntimeTracking(BaseModel):
    stream_logs: bool = True
    timeout_seconds: int = 300
    cancellable: bool = True
    cancel_requested: bool = False


class ErrorInfo(BaseModel):
    code: str
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class JobRequest(BaseModel):
    job_id: str
    task_type: TaskType
    status: JobStatus = JobStatus.PENDING
    metadata: Metadata = Metadata()
    params: dict[str, Any] = Field(default_factory=dict)
    output: OutputConfig = OutputConfig()
    security_constraints: SecurityConstraints = SecurityConstraints()
    runtime_tracking: RuntimeTracking = RuntimeTracking()
    error: ErrorInfo | None = None


# 2. Security Policy Enforcement
def _is_path_safe(target_path: str, allowed_roots: list[str]) -> bool:
    """Validate that target_path is within one of the allowed_roots directories."""
    if not allowed_roots:
        return False
    try:
        resolved = Path(target_path).resolve()
        for root in allowed_roots:
            root_resolved = Path(root).resolve()
            if resolved == root_resolved or root_resolved in resolved.parents:
                return True
        return False
    except (ValueError, OSError):
        return False


def _validate_job_security(job: JobRequest) -> tuple[bool, str | None]:
    """Validate job security constraints. Returns (is_valid, error_message)."""
    # Rule 1: Shell execution prohibited
    if job.security_constraints.allow_shell:
        return False, "Shell execution not permitted"

    # Rule 2: Path whitelisting required
    if not job.security_constraints.path_whitelisted:
        return False, "Path whitelisting disabled"

    # Rule 3: Validate actual paths against whitelist
    allowed = job.security_constraints.allowed_paths
    if job.task_type == TaskType.PREDICT:
        model_path = job.params.get("model_path", "")
        source_path = job.params.get("data_source", "")
        output_dir = job.output.output_dir

        for desc, path in [("model", model_path), ("data_source", source_path), ("output", output_dir)]:
            if path and not _is_path_safe(path, allowed):
                return False, f"{desc.capitalize()} path '{path}' not in whitelist"

    return True, None


# 3. Task State Machine & Dispatcher Core
class JobDispatcherStateMachine:
    def __init__(self):
        self.valid_transitions = {
            JobStatus.PENDING: [JobStatus.RUNNING, JobStatus.FAILED],
            JobStatus.RUNNING: [JobStatus.COMPLETED, JobStatus.FAILED],
            JobStatus.COMPLETED: [],
            JobStatus.FAILED: [],
        }

    def transition(
        self, job: JobRequest, target_status: JobStatus, err_code: str | None = None, err_msg: str | None = None
    ):
        """Enforce strict task state transitions."""
        if target_status not in self.valid_transitions[job.status]:
            raise ValueError(f"Illegal state transition: {job.status} -> {target_status}")
        job.status = target_status
        if err_code:
            job.error = ErrorInfo(code=err_code, message=err_msg or "")
        print(f"  [StateMachine] Job {job.job_id} transitioned to: {job.status.value.upper()}")

    def execute(self, job: JobRequest) -> JobRequest:
        """Execute job under security constraints and state machine control."""
        # Security Policy Enforcement
        is_valid, err_msg = _validate_job_security(job)
        if not is_valid:
            self.transition(
                job, JobStatus.FAILED, err_code="SEC_ERR_001", err_msg=f"Security policy violation: {err_msg}"
            )
            return job

        # State Transition: PENDING -> RUNNING
        self.transition(job, JobStatus.RUNNING)
        try:
            if job.task_type == TaskType.PREDICT:
                from ultralytics import YOLO

                model_path = job.params.get("model_path", "yolov8n.pt")
                source = job.params.get("data_source", "ultralytics/assets/bus.jpg")
                device = job.params.get("device", "0")

                # Real YOLO execution on target device
                model = YOLO(model_path)
                res = model.predict(
                    source=source,
                    device=device,
                    save=job.output.save_images,
                    project="runs/predict",
                    name=job.job_id,
                    exist_ok=True,
                    verbose=False,
                )
                det_count = len(res[0].boxes)
                print(f"  [Executor] Real inference executed on device={device}. Detected count: {det_count}")

                # Accurately capture output artifacts from engine result
                actual_save_dir = Path(res[0].save_dir)
                job.output.output_dir = str(actual_save_dir)
                artifacts = [str(p) for p in actual_save_dir.glob("*.jpg")]
                job.output.artifacts = artifacts
                print(f"  [Artifacts] Verified artifacts ({len(artifacts)} files) at: {artifacts}")

                # State Transition: RUNNING -> COMPLETED
                self.transition(job, JobStatus.COMPLETED)
            else:
                raise NotImplementedError(f"Task type {job.task_type} unsupported in smoke test.")
        except Exception as e:  # noqa: BLE001 - dispatcher must convert any engine error into FAILED state
            self.transition(job, JobStatus.FAILED, err_code="EXEC_ERR_500", err_msg=str(e))

        return job


# 4. Multi-Scenario Smoke Verification Suite
def run_smoke_test_suite():
    print("\n=======================================================")
    print("  [INIT] F1 Smoke Test Suite (Multi-Scenario Verification)")
    print("=======================================================")

    dispatcher = JobDispatcherStateMachine()
    suite_results = []

    # Scenario 1: E2E Happy Path Execution
    print("\n[Case 1] Contract Parsing & E2E Real Inference Execution...")
    t0 = time.time()
    with open("smoke/f1/job_request_draft.json", encoding="utf-8-sig") as f:
        data = json.load(f)
    job1 = JobRequest(**data)
    final_job1 = dispatcher.execute(job1)
    e2e_duration = time.time() - t0
    c1_pass = final_job1.status == JobStatus.COMPLETED and len(final_job1.output.artifacts) > 0
    suite_results.append(
        (
            "E2E Happy Path (Real Predict)",
            c1_pass,
            f"Time: {e2e_duration:.2f}s, Artifacts: {len(final_job1.output.artifacts)}",
        )
    )
    print(f"  {'[PASS]' if c1_pass else '[FAIL]'} in {e2e_duration:.2f}s")

    # Scenario 2: Security Violation Defense (Shell Execution)
    print("\n[Case 2] Security Constraint Defense (Shell Attempt)...")
    job2 = JobRequest(
        job_id="smoke-sec-002",
        task_type=TaskType.PREDICT,
        params={"model_path": "yolov8n.pt"},
        security_constraints=SecurityConstraints(
            path_whitelisted=True, allow_shell=True, allowed_paths=["ultralytics/assets"]
        ),
    )
    final_job2 = dispatcher.execute(job2)
    c2_pass = final_job2.status == JobStatus.FAILED and final_job2.error.code == "SEC_ERR_001"
    suite_results.append(("Security Defense (Shell)", c2_pass, f"ErrorCode: {final_job2.error.code}"))
    print(f"  {'[PASS]' if c2_pass else '[FAIL]'} (Shell execution blocked)")

    # Scenario 3: Security Violation Defense (Path Traversal)
    print("\n[Case 3] Security Constraint Defense (Path Traversal)...")
    job3 = JobRequest(
        job_id="smoke-sec-003",
        task_type=TaskType.PREDICT,
        params={"model_path": "yolov8n.pt", "data_source": "../../etc/passwd"},
        security_constraints=SecurityConstraints(
            path_whitelisted=True, allow_shell=False, allowed_paths=["ultralytics/assets"]
        ),
    )
    final_job3 = dispatcher.execute(job3)
    c3_pass = final_job3.status == JobStatus.FAILED and final_job3.error.code == "SEC_ERR_001"
    suite_results.append(
        (
            "Security Defense (Path Traversal)",
            c3_pass,
            f"ErrorCode: {final_job3.error.code if final_job3.error else 'N/A'}",
        )
    )
    print(f"  {'[PASS]' if c3_pass else '[FAIL]'} (Path traversal blocked)")

    # Scenario 4: State Machine Illegal Transition (COMPLETED -> RUNNING)
    print("\n[Case 4] State Machine Guard (COMPLETED -> RUNNING)...")
    job4 = JobRequest(job_id="smoke-state-004", task_type=TaskType.PREDICT, params={})
    job4.status = JobStatus.COMPLETED
    c4_pass = False
    try:
        dispatcher.transition(job4, JobStatus.RUNNING)
    except ValueError:
        c4_pass = True
    suite_results.append(("State Guard (COMPLETED->RUNNING)", c4_pass, "Prevented illegal transition"))
    print(f"  {'[PASS]' if c4_pass else '[FAIL]'} (Illegal transition prevented)")

    # Scenario 5: State Machine Illegal Transition (PENDING -> COMPLETED)
    print("\n[Case 5] State Machine Guard (PENDING -> COMPLETED)...")
    job5 = JobRequest(job_id="smoke-state-005", task_type=TaskType.PREDICT, params={})
    c5_pass = False
    try:
        dispatcher.transition(job5, JobStatus.COMPLETED)
    except ValueError:
        c5_pass = True
    suite_results.append(("State Guard (PENDING->COMPLETED)", c5_pass, "Prevented skip to terminal"))
    print(f"  {'[PASS]' if c5_pass else '[FAIL]'} (Skip prevented)")

    # Summary Report
    print("\n" + "=" * 55)
    print("Smoke Test Execution Summary:")
    all_passed = True
    for name, passed, detail in suite_results:
        flag = "[PASS]" if passed else "[FAIL]"
        print(f"  - {name:<35} {flag} ({detail})")
        if not passed:
            all_passed = False
    print("=" * 55)

    if all_passed:
        print(">>> ALL F1 SMOKE SCENARIOS PASSED SUCCESSFULLY! <<<\n")
        sys.exit(0)
    else:
        print(">>> SOME SMOKE SCENARIOS FAILED! <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test_suite()
