"""Storage helpers for raw and derived data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_text(content: str, path: Path) -> None:
    """Write text to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read JSON data from a file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records to a JSON Lines file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
