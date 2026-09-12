"""Descriptive, immutable catalog of aspirational Studio/Agent task links.

Handler Convergence Phase 1B.

This module is a pure DESCRIPTION artifact. It records candidate convergence
relationships between the Studio task surface (the production
``core.schema.TaskType`` enum values) and the Agent skill surface (the keys of
the production ``agent.runtime.cli.dispatcher.HANDLERS`` mapping). It has NO
runtime routing responsibility by design:

    - no mutable registry and no ``register()``/``dispatch()``/lookup API;
    - no handler callables, deps, signatures or GPU/family metadata;
    - no lifecycle, process, artifact, security or logging logic;
    - no imports from ``core.schema``, ``f1``, ``agent`` or any third-party
      package (Python stdlib only).

Every pair below is an ASPIRATIONAL convergence candidate, not production
dispatch wiring. In particular the ``diagnose -> yolo.system`` pair has no
production wiring today: nothing in ``f1`` invokes ``yolo.system`` when a
diagnose job runs (the active f1 tool name remains ``system_doctor``). The
candidate only states the intended semantic alignment target.

In Phase 1B this module is consumed solely by conformance tests
(``tests/f1/test_task_catalog.py``); no production module imports it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ASPIRATIONAL_TASK_LINKS", "TaskLinkCandidate"]


@dataclass(frozen=True)
class TaskLinkCandidate:
    """One aspirational convergence candidate between the two task surfaces.

    Attributes:
        studio_task_id: Task id on the Studio side; conformance tests pin it
            against the production ``core.schema.TaskType`` enum values.
        agent_skill_id: Skill id on the Agent side; conformance tests pin it
            against the production ``agent.runtime.cli.dispatcher.HANDLERS``
            keys.
    """

    studio_task_id: str
    agent_skill_id: str


# Frozen table of convergence candidates. Declaration order is documentary
# only and carries no dispatch semantics. Conformance tests verify that every
# id named here currently exists on its respective production surface; the
# table itself never participates in runtime dispatch.
ASPIRATIONAL_TASK_LINKS: tuple[TaskLinkCandidate, ...] = (
    TaskLinkCandidate(studio_task_id="predict", agent_skill_id="yolo.predict"),
    TaskLinkCandidate(studio_task_id="train", agent_skill_id="yolo.train"),
    TaskLinkCandidate(studio_task_id="val", agent_skill_id="yolo.val"),
    TaskLinkCandidate(studio_task_id="export", agent_skill_id="yolo.export"),
    TaskLinkCandidate(studio_task_id="diagnose", agent_skill_id="yolo.system"),
)
