"""Subprocess-only Service verification harness, never a production launcher."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from main_engine import ServiceSettings, create_app, run_service
from studio.job_logs import sanitize_log_text
from studio.jobs_manager import JobsManager
from studio.worker_runtime import ManagedWorker
from tests.studio.test_service_api import service_executor
from tests.studio.test_training_shutdown import cooperative_executor, real_training_executor


def main():
    """Run an actual server and record owner shutdown before service process exit."""
    root, port, mode = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    events, managers = [], []
    if mode == "cleanup_failure":
        original_stop = ManagedWorker.stop

        def fail_cleanup(worker, grace):
            if not (root / "release-cleanup").exists():
                raise OSError("injected cleanup password=should-not-leak")
            return original_stop(worker, grace)

        ManagedWorker.stop = fail_cleanup

    class ObservedManager(JobsManager):
        def shutdown(self):
            events.append({"event": "manager.shutdown.enter", "monotonic": time.monotonic()})
            try:
                super().shutdown()
            finally:
                events.append(
                    {"event": "manager.shutdown.return", "monotonic": time.monotonic(), "owners": len(self._workers)}
                )

    def factory():
        grace = 120 if mode == "real" else 1
        manager = ObservedManager(
            storage_path=root / "state.json",
            output_root=root,
            model_roots=[Path(os.environ["STUDIO_TEST_MODEL"]).parent] if mode == "real" else [root],
            data_roots=[root],
            cpu_concurrency=1,
            gpu_concurrency=1,
            shutdown_grace_seconds=grace,
            stop_grace_seconds=0.1,
        )
        manager._worker_executor = (
            real_training_executor if mode == "real" else service_executor if mode == "rest" else cooperative_executor
        )
        managers.append(manager)
        return manager

    application = create_app(settings=ServiceSettings(port=port, http_drain_seconds=0.2), manager_factory=factory)

    @application.get("/_test/identities")
    def identities():
        import psutil

        facts = []
        manager = managers[0]
        with manager.lock:
            for worker in manager._workers.values():
                process = psutil.Process(worker.process.pid)
                for child in [process, *process.children(recursive=True)]:
                    facts.append({"pid": child.pid, "created": child.create_time(), "status": child.status()})
        return facts

    exit_code = 0
    try:
        run_service(application)
    except RuntimeError as exc:
        events.append(
            {"event": "service.failure", "message": sanitize_log_text(str(exc)), "monotonic": time.monotonic()}
        )
        exit_code = 1
    finally:
        manager = managers[0] if managers else None
        facts = {
            "mode": mode,
            "service_failure": application.state.service_failure,
            "owner_retained": application.state.jobs_manager is not None,
            "runtime_budget": manager.shutdown_budget_seconds if manager else None,
            "http_drain_budget": application.state.settings.http_drain_seconds,
            "events": events,
            "monotonic": time.monotonic(),
            "jobs": manager.list_jobs_snapshot()["jobs"] if manager else [],
            "logs": manager.get_job_log_lines("shutdown-train") if manager else [],
        }
        (root / "service-facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
        if mode == "cleanup_failure" and manager is not None:
            # Restore OS cleanup after measuring real timeout/retention; leave no test orphan.
            (root / "release-cleanup").touch()
            manager.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
