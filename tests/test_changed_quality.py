from pathlib import Path
import subprocess

import pytest

from scripts import check_changed_quality


def _touch(root: Path, relative: str) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test\n", encoding="utf-8")
    return relative


def test_select_quality_files_separates_python_and_spelling_scopes(tmp_path):
    source = _touch(tmp_path, "ultralytics/core.py")
    spaced = _touch(tmp_path, "tests/path with spaces/test_contract.py")
    docs = _touch(tmp_path, "docs/mixture.md")
    ignored = _touch(tmp_path, "output/generated.py")
    binary = _touch(tmp_path, "docs/model.bin")

    selected = check_changed_quality.select_quality_files([source, spaced, docs, ignored, binary], repo_root=tmp_path)

    assert selected.python == (spaced, source)
    assert selected.spelling == (docs, spaced, source)


def test_build_quality_commands_preserves_each_path_as_one_argument():
    selected = check_changed_quality.QualityFiles(
        python=("tests/path with spaces/test_contract.py",),
        spelling=("docs/mixture report.md",),
    )

    commands = check_changed_quality.build_quality_commands(selected, python_executable="python")

    assert commands == (
        ("python", "-m", "ruff", "check", "tests/path with spaces/test_contract.py"),
        ("python", "-m", "ruff", "format", "--check", "tests/path with spaces/test_contract.py"),
        ("python", "-m", "codespell_lib", "docs/mixture report.md"),
    )


def test_build_quality_commands_limits_format_scope_without_narrowing_lint():
    selected = check_changed_quality.QualityFiles(
        python=("tests/legacy.py", "tests/new.py"),
        spelling=(),
    )

    commands = check_changed_quality.build_quality_commands(
        selected,
        format_python=("tests/new.py",),
        python_executable="python",
    )

    assert commands == (
        ("python", "-m", "ruff", "check", "tests/legacy.py", "tests/new.py"),
        ("python", "-m", "ruff", "format", "--check", "tests/new.py"),
    )


def test_select_incremental_format_files_checks_new_and_preformatted_baselines(monkeypatch, tmp_path):
    baseline_sources = {
        "ultralytics/clean.py": "clean baseline",
        "ultralytics/legacy.py": "legacy baseline",
    }

    monkeypatch.setattr(
        check_changed_quality,
        "git_file_at_revision",
        lambda path, **_: baseline_sources.get(path),
    )
    monkeypatch.setattr(
        check_changed_quality,
        "source_is_ruff_formatted",
        lambda source, **_: source == "clean baseline",
    )

    selection = check_changed_quality.select_incremental_format_files(
        ("tests/new.py", "ultralytics/clean.py", "ultralytics/legacy.py"),
        repo_root=tmp_path,
        baseline="HEAD",
        python_executable="python",
    )

    assert selection.check == ("tests/new.py", "ultralytics/clean.py")
    assert selection.baseline_debt == ("ultralytics/legacy.py",)


def test_source_is_ruff_formatted_rejects_formatter_infrastructure_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        check_changed_quality.subprocess,
        "run",
        lambda *_, **__: subprocess.CompletedProcess([], 2, stdout="", stderr="formatter failed"),
    )

    with pytest.raises(subprocess.CalledProcessError, match="returned non-zero exit status 2"):
        check_changed_quality.source_is_ruff_formatted(
            "x = 1\n",
            path="tests/example.py",
            repo_root=tmp_path,
            python_executable="python",
        )


def test_parse_args_accepts_strict_format():
    args = check_changed_quality.parse_args(["--strict-format"])

    assert args.strict_format is True


def test_resolve_changed_paths_prefers_explicit_files(monkeypatch, tmp_path):
    def fail_if_called(**_):
        raise AssertionError("git discovery must not run for explicit files")

    monkeypatch.setattr(check_changed_quality, "git_changed_files", fail_if_called)

    assert check_changed_quality.resolve_changed_paths(("ultralytics/a.py", "tests/b.py"), repo_root=tmp_path) == (
        "ultralytics/a.py",
        "tests/b.py",
    )
