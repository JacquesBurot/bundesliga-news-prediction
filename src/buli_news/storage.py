"""Storage helpers for raw and derived data files."""

from __future__ import annotations

from pathlib import Path


def write_text(content: str, path: Path) -> None:
    """Write text to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
