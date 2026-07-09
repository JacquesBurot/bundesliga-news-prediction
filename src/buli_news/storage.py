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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read records from a JSON Lines file."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                msg = f"{path} must contain one JSON object per line."
                raise ValueError(msg)
            records.append(record)
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records to a JSON Lines file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    """Append one record to a JSON Lines file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
