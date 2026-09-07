"""Export the F1 platform kernel contract models as a JSON schema registry.

This script serializes the canonical domain models defined in :mod:`core.schema`
into the official cross-topic shared-infrastructure deliverable
``docs/api/job_schema.json``. Pydantic models are exported through the
standard Pydantic v2 ``model_json_schema`` method, while the ``JobStatus``
enumeration is recorded directly as a lightweight list of its member values
and names, as required by the registry specification.

The generated document follows the official top-level envelope:

    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "_title": "YOLO-Master F1 Platform Kernel Schema Registry",
        "_version": <core.schema.__version__>,
        "_owner": "F1",
        "schemas": { ... }
    }

Example:
    >>> from core.export_schema import _enum_schema_record
    >>> from core.schema import JobStatus
    >>> _enum_schema_record(JobStatus)["values"]
    ['pending', 'running', 'completed', 'failed']
"""

from __future__ import annotations

import json
import pathlib
from enum import Enum
from typing import Any

from core.schema import (
    ArtifactManifest,
    JobRequest,
    JobStatus,
    SecurityConstraints,
    __version__,
)

DEFAULT_OUTPUT_PATH = "docs/api/job_schema.json"

__all__ = ["DEFAULT_OUTPUT_PATH", "export_schemas"]


def _enum_schema_record(enum_cls: type[Enum]) -> dict[str, Any]:
    """Build a lightweight registry record for an enumeration.

    Args:
        enum_cls: The enumeration class to describe.

    Returns:
        dict[str, Any]: A record carrying the enumeration name, the ordered
            list of member values and the corresponding member names.
    """
    return {
        "enum": enum_cls.__name__,
        "values": [member.value for member in enum_cls],
        "names": [member.name for member in enum_cls],
    }


def export_schemas(output_path: pathlib.Path | str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """Export the core contract models to a JSON schema registry document.

    Pydantic models (``JobRequest``, ``ArtifactManifest`` and
    ``SecurityConstraints``) are exported with the standard Pydantic v2
    ``model_json_schema`` method; the ``JobStatus`` enumeration is recorded
    directly as its member value and name lists. The resulting registry is
    written to ``output_path`` with its parent directories created as needed.

    Args:
        output_path: Destination file for the registry document. Parent
            directories are created automatically when missing.

    Returns:
        dict[str, Any]: The registry dictionary that was written to disk.
    """
    registry: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "_title": "YOLO-Master F1 Platform Kernel Schema Registry",
        "_version": __version__,
        "_owner": "F1",
        "schemas": {
            "JobRequest": JobRequest.model_json_schema(),
            "JobStatus": _enum_schema_record(JobStatus),
            "ArtifactManifest": ArtifactManifest.model_json_schema(),
            "SecurityConstraints": SecurityConstraints.model_json_schema(),
        },
    }
    target = pathlib.Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return registry


if __name__ == "__main__":
    export_schemas()
    print(f"Schema registry written to {DEFAULT_OUTPUT_PATH}")
