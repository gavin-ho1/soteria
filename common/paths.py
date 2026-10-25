"""Path helpers for project-root-relative commands and run metadata"""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str | Path, *, field: str) -> tuple[Path, str]:
    """Return (path, project-relative POSIX string) for writing into artifacts

    Paths outside the repo are rejected, they have no portable relative form
    The resolved location is used when it is inside the repo
    A repo path that leaves the repo through a symlink (e.g. data/ on an external drive) is also allowed, but may not contain '..': after a symlink, '..' means the target's parent, so the written and real locations would disagree
    """
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved, resolved.relative_to(PROJECT_ROOT).as_posix()
    if ".." not in path.parts:
        written = Path(os.path.normpath(path))  # only drops "." and "//" here
        if written.is_relative_to(PROJECT_ROOT):
            return path, written.relative_to(PROJECT_ROOT).as_posix()
    raise ValueError(
        f"{field} must be inside the project root ({PROJECT_ROOT}) without "
        f"'..' through a symlinked directory, got {value!s}"
    )
