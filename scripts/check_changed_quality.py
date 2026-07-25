"""Run repository quality tools only on supported changed files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable, Sequence


PYTHON_ROOTS = frozenset({"agent", "scripts", "tests", "ultralytics"})
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})
SPELLING_SUFFIXES = frozenset({".json", ".md", ".py", ".pyi", ".sh", ".toml", ".txt", ".yaml", ".yml"})
IGNORED_ROOTS = frozenset({".git", ".qoder", ".workbuddy", ".venv", "output"})
IGNORED_PREFIXES = (("agent", "logs"),)


@dataclass(frozen=True)
class QualityFiles:
    """Selected paths for each quality tool family."""

    python: tuple[str, ...]
    spelling: tuple[str, ...]


@dataclass(frozen=True)
class IncrementalFormatFiles:
    """Python files to format-check and files exempted by baseline debt."""

    check: tuple[str, ...]
    baseline_debt: tuple[str, ...]


def _relative_existing_file(path: str, repo_root: Path) -> Path | None:
    root = repo_root.resolve()
    candidate = Path(path)
    candidate = candidate if candidate.is_absolute() else root / candidate
    try:
        relative = candidate.resolve().relative_to(root)
    except (FileNotFoundError, ValueError):
        return None
    return relative if candidate.is_file() else None


def _is_ignored(path: Path) -> bool:
    parts = path.parts
    return bool(
        not parts or parts[0] in IGNORED_ROOTS or any(parts[: len(prefix)] == prefix for prefix in IGNORED_PREFIXES)
    )


def select_quality_files(paths: Iterable[str], *, repo_root: Path) -> QualityFiles:
    """Filter paths into deterministic Ruff and codespell inputs."""
    python_files: set[str] = set()
    spelling_files: set[str] = set()
    for raw_path in paths:
        relative = _relative_existing_file(raw_path, repo_root)
        if relative is None or _is_ignored(relative):
            continue
        normalized = relative.as_posix()
        suffix = relative.suffix.lower()
        if suffix in PYTHON_SUFFIXES and relative.parts[0] in PYTHON_ROOTS:
            python_files.add(normalized)
        if suffix in SPELLING_SUFFIXES:
            spelling_files.add(normalized)
    return QualityFiles(tuple(sorted(python_files)), tuple(sorted(spelling_files)))


def git_changed_files(
    *,
    repo_root: Path,
    base: str | None = None,
    staged: bool = False,
    include_untracked: bool = True,
) -> tuple[str, ...]:
    """Return added/copied/modified/renamed tracked paths plus optional untracked paths."""
    command = ["git", "diff", "--name-only", "--diff-filter=ACMRT"]
    if staged:
        command.append("--cached")
    elif base:
        command.append(base)
    command.append("--")
    tracked = subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True).stdout.splitlines()
    untracked: list[str] = []
    if include_untracked and not staged:
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    return tuple(sorted(set(tracked + untracked)))


def resolve_changed_paths(
    explicit_files: Sequence[str],
    *,
    repo_root: Path,
    base: str | None = None,
    staged: bool = False,
    include_untracked: bool = True,
) -> tuple[str, ...]:
    """Prefer explicitly supplied paths, otherwise discover them from Git."""
    if explicit_files:
        return tuple(dict.fromkeys(explicit_files))
    return git_changed_files(
        repo_root=repo_root,
        base=base,
        staged=staged,
        include_untracked=include_untracked,
    )


def git_file_at_revision(path: str, *, repo_root: Path, revision: str) -> str | None:
    """Read a UTF-8 repository file at a revision, or return None when the path is new."""
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def source_is_ruff_formatted(
    source: str,
    *,
    path: str,
    repo_root: Path,
    python_executable: str = sys.executable,
) -> bool:
    """Return whether source passes Ruff format, raising on formatter infrastructure errors."""
    result = subprocess.run(
        (python_executable, "-m", "ruff", "format", "--check", "--stdin-filename", path, "-"),
        cwd=repo_root,
        check=False,
        capture_output=True,
        input=source,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.returncode == 0


def select_incremental_format_files(
    paths: Iterable[str],
    *,
    repo_root: Path,
    baseline: str,
    python_executable: str = sys.executable,
) -> IncrementalFormatFiles:
    """Require formatting for new files and files whose baseline was already formatted."""
    check: list[str] = []
    baseline_debt: list[str] = []
    for path in sorted(set(paths)):
        baseline_source = git_file_at_revision(path, repo_root=repo_root, revision=baseline)
        if baseline_source is None or source_is_ruff_formatted(
            baseline_source,
            path=path,
            repo_root=repo_root,
            python_executable=python_executable,
        ):
            check.append(path)
        else:
            baseline_debt.append(path)
    return IncrementalFormatFiles(tuple(check), tuple(baseline_debt))


def build_quality_commands(
    selected: QualityFiles,
    *,
    format_python: Sequence[str] | None = None,
    python_executable: str = sys.executable,
) -> tuple[tuple[str, ...], ...]:
    """Build subprocess-safe commands without shell interpolation."""
    commands: list[tuple[str, ...]] = []
    format_python = selected.python if format_python is None else tuple(format_python)
    if selected.python:
        commands.append((python_executable, "-m", "ruff", "check", *selected.python))
    if format_python:
        commands.append((python_executable, "-m", "ruff", "format", "--check", *format_python))
    if selected.spelling:
        commands.append((python_executable, "-m", "codespell_lib", *selected.spelling))
    return tuple(commands)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Explicit files to check instead of discovering Git changes.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--base", help="Git revision to diff against, for example origin/main.")
    source.add_argument("--staged", action="store_true", help="Check only staged tracked files.")
    parser.add_argument("--no-untracked", action="store_true", help="Exclude untracked files from discovery.")
    parser.add_argument(
        "--strict-format",
        action="store_true",
        help="Format-check every selected Python file, including files with baseline formatting debt.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        changed = resolve_changed_paths(
            args.files,
            repo_root=args.repo_root,
            base=args.base,
            staged=args.staged,
            include_untracked=not args.no_untracked,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Git change discovery failed with exit code {exc.returncode}.", file=sys.stderr)
        return 2

    selected = select_quality_files(changed, repo_root=args.repo_root)
    format_python = selected.python
    baseline_debt: tuple[str, ...] = ()
    if selected.python and not args.strict_format and not args.files:
        baseline = args.base or "HEAD"
        try:
            format_selection = select_incremental_format_files(
                selected.python,
                repo_root=args.repo_root,
                baseline=baseline,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Ruff baseline format check failed with exit code {exc.returncode}.", file=sys.stderr)
            return 2
        format_python = format_selection.check
        baseline_debt = format_selection.baseline_debt

    if baseline_debt:
        print(
            f"Skipping Ruff format for {len(baseline_debt)} file(s) with baseline formatting debt; "
            "use --strict-format to enforce all selected files."
        )
        for path in baseline_debt:
            print(f"  - {path}")

    commands = build_quality_commands(selected, format_python=format_python)
    if not commands:
        print("No supported changed files to check.")
        return 0

    failed = False
    for command in commands:
        print("+ " + " ".join(shlex.quote(argument) for argument in command))
        failed = subprocess.run(command, cwd=args.repo_root, check=False).returncode != 0 or failed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
