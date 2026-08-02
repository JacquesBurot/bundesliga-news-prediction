"""Storage helpers for raw and derived data files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_text(content: str, path: Path) -> None:
    """Write text to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_bytes(content: bytes, path: Path) -> None:
    """Write bytes to a file unchanged, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def read_bytes(path: Path) -> bytes:
    """Read bytes from a file."""
    return path.read_bytes()


def read_json(path: Path) -> Any:
    """Read JSON data from a file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: Any, path: Path) -> None:
    """Write formatted JSON data, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(content + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read records from a JSON Lines file, tolerating an optional UTF-8 BOM."""
    records = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
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


def write_csv(
    records: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
    path: Path,
) -> None:
    """Write flat records to CSV with a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    """Append one record to a JSON Lines file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
