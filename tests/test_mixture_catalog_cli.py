"""Tests for the `yolo mixtures` catalog command."""

from __future__ import annotations

import json

import pytest

from ultralytics.cfg import entrypoint
from ultralytics.cfg.mixture_catalog import MixtureProfile


def _profile() -> MixtureProfile:
    """Return a representative immutable CLI record."""
    return MixtureProfile(
        profile_id="master/v0_10/det/yolo-master-mot-n",
        path="master/v0_10/det/yolo-master-mot-n.yaml",
        task="detect",
        family="master/v0_10",
        scales=("n",),
        mixture_kinds=("moe", "mot"),
        mixture_modules=("VisualEnhancedAdaptiveGateMoE", "C2fMoT"),
    )


def test_mixtures_cli_forwards_filters_and_emits_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose exact catalog filters and a stable machine-readable representation."""
    from ultralytics.cfg import LOGGER, mixture_catalog

    calls = []
    messages = []

    def fake_list(**kwargs):
        calls.append(kwargs)
        return (_profile(),)

    monkeypatch.setattr(mixture_catalog, "list_mixture_profiles", fake_list)
    monkeypatch.setattr(LOGGER, "info", messages.append)

    entrypoint("yolo mixtures kind=mot task=detect family=master/v0_10 format=json")

    assert calls == [{"kind": "mot", "task": "detect", "family": "master/v0_10"}]
    assert json.loads(messages[-1]) == [_profile().as_dict()]


def test_mixtures_cli_emits_table_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep human output deterministic and show how many profiles matched."""
    from ultralytics.cfg import LOGGER, mixture_catalog

    messages = []
    monkeypatch.setattr(mixture_catalog, "list_mixture_profiles", lambda **_: (_profile(),))
    monkeypatch.setattr(LOGGER, "info", messages.append)

    entrypoint("yolo mixtures")

    assert "PROFILE" in messages[-1]
    assert "master/v0_10/det/yolo-master-mot-n" in messages[-1]
    assert "1 mixture profile" in messages[-1]


def test_mixtures_cli_handles_an_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render a valid table and count when no packaged profile matches."""
    from ultralytics.cfg import LOGGER, mixture_catalog

    messages = []
    monkeypatch.setattr(mixture_catalog, "list_mixture_profiles", lambda **_: ())
    monkeypatch.setattr(LOGGER, "info", messages.append)

    entrypoint("yolo mixtures kind=mot")

    assert "PROFILE" in messages[-1]
    assert messages[-1].endswith("0 mixture profiles")


@pytest.mark.parametrize(
    "command, match",
    [
        ("yolo mixtures output=csv", "unknown mixtures argument 'output'"),
        ("yolo mixtures format=csv", "unknown mixtures format 'csv'"),
        ("yolo mixtures mot", "expected key=value"),
    ],
)
def test_mixtures_cli_rejects_invalid_arguments(command: str, match: str) -> None:
    """Fail before catalog discovery when CLI arguments violate the command contract."""
    with pytest.raises(ValueError, match=match):
        entrypoint(command)
