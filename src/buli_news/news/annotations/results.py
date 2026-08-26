"""Build, persist, and validate annotation result and failure records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._validation import require_positive_int, require_text
from .config import AnnotationConfig, INDICATORS
from .ollama import InvalidOllamaAnnotationError


NEWS_ANNOTATION_OUTPUT_SCHEMA_VERSION = 3
NEWS_ANNOTATION_FAILURE_SCHEMA_VERSION = 1
NEWS_ANNOTATION_OUTPUT_COLUMNS = (
    "annotation_id",
    "output_schema_version",
    "task_id",
    "annotation_schema_id",
    "annotation_schema_version",
    "prompt_version",
    "annotation_config_sha256",
    "model",
    "model_digest",
    "request_id",
    "content_id",
    "match_id",
    "side",
    "target_team_id",
    "target_team",
    "selected_article_id",
    *INDICATORS,
    "ollama_created_at",
    "total_duration_ns",
    "load_duration_ns",
    "prompt_eval_count",
    "prompt_eval_duration_ns",
    "eval_count",
    "eval_duration_ns",
)
NEWS_ANNOTATION_FAILURE_COLUMNS = (
    "failure_event_id",
    "failure_schema_version",
    "failure_sequence",
    "annotation_id",
    "task_id",
    "annotation_schema_id",
    "annotation_schema_version",
    "prompt_version",
    "annotation_config_sha256",
    "model",
    "model_digest",
    "request_id",
    "content_id",
    "match_id",
    "side",
    "target_team_id",
    "target_team",
    "selected_article_id",
    "attempt_count",
    "error",
    "last_response_content",
    "ollama_created_at",
    "total_duration_ns",
    "prompt_eval_count",
    "eval_count",
    "recorded_at",
)


def build_annotation_row(
    task: dict[str, Any],
    response: dict[str, Any],
    metadata: dict[str, Any],
    config: AnnotationConfig,
    model: str,
    model_digest: str,
    annotation_id: str,
) -> dict[str, Any]:
    """Combine the validated response with stable provenance and timings."""
    row = {
        "annotation_id": annotation_id,
        "output_schema_version": NEWS_ANNOTATION_OUTPUT_SCHEMA_VERSION,
        "task_id": task["task_id"],
        "annotation_schema_id": config.schema_id,
        "annotation_schema_version": config.schema_version,
        "prompt_version": config.prompt_version,
        "annotation_config_sha256": config.sha256,
        "model": model,
        "model_digest": model_digest,
        "request_id": task["request_id"],
        "content_id": task["content_id"],
        "match_id": task["match_id"],
        "side": task["side"],
        "target_team_id": task["target_team_id"],
        "target_team": task["target_team"],
        "selected_article_id": task["selected_article_id"],
        **{indicator: response[indicator] for indicator in INDICATORS},
        "ollama_created_at": metadata.get("created_at"),
        "total_duration_ns": metadata.get("total_duration"),
        "load_duration_ns": metadata.get("load_duration"),
        "prompt_eval_count": metadata.get("prompt_eval_count"),
        "prompt_eval_duration_ns": metadata.get("prompt_eval_duration"),
        "eval_count": metadata.get("eval_count"),
        "eval_duration_ns": metadata.get("eval_duration"),
    }
    if tuple(row) != NEWS_ANNOTATION_OUTPUT_COLUMNS:
        msg = "Internal annotation output schema does not match."
        raise ValueError(msg)
    return row


def build_annotation_failure_row(
    task: dict[str, Any],
    error: InvalidOllamaAnnotationError,
    config: AnnotationConfig,
    model: str,
    model_digest: str,
    annotation_id: str,
    failure_sequence: int,
) -> dict[str, Any]:
    """Persist one exhausted response-validation failure without marking it done."""
    metadata = error.last_metadata
    row = {
        "failure_event_id": build_failure_event_id(
            annotation_id=annotation_id,
            failure_sequence=failure_sequence,
        ),
        "failure_schema_version": NEWS_ANNOTATION_FAILURE_SCHEMA_VERSION,
        "failure_sequence": failure_sequence,
        "annotation_id": annotation_id,
        "task_id": task["task_id"],
        "annotation_schema_id": config.schema_id,
        "annotation_schema_version": config.schema_version,
        "prompt_version": config.prompt_version,
        "annotation_config_sha256": config.sha256,
        "model": model,
        "model_digest": model_digest,
        "request_id": task["request_id"],
        "content_id": task["content_id"],
        "match_id": task["match_id"],
        "side": task["side"],
        "target_team_id": task["target_team_id"],
        "target_team": task["target_team"],
        "selected_article_id": task["selected_article_id"],
        "attempt_count": error.attempt_count,
        "error": str(error),
        "last_response_content": error.last_response_content,
        "ollama_created_at": metadata.get("created_at"),
        "total_duration_ns": metadata.get("total_duration"),
        "prompt_eval_count": metadata.get("prompt_eval_count"),
        "eval_count": metadata.get("eval_count"),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    if tuple(row) != NEWS_ANNOTATION_FAILURE_COLUMNS:
        msg = "Internal annotation failure schema does not match."
        raise ValueError(msg)
    return row


def read_existing_annotations(path: Path) -> list[dict[str, Any]]:
    """Read and validate an existing append-only annotation output."""
    if not path.exists():
        return []
    rows = []
    annotation_ids = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {path} at line {line_number}: {exc}."
            raise ValueError(msg) from exc
        if (
            not isinstance(row, dict)
            or row.get("output_schema_version")
            != NEWS_ANNOTATION_OUTPUT_SCHEMA_VERSION
            or tuple(row) != NEWS_ANNOTATION_OUTPUT_COLUMNS
        ):
            msg = f"Annotation schema does not match in {path} at line {line_number}."
            raise ValueError(msg)
        annotation_id = require_text(row, "annotation_id", "Stored annotation")
        if annotation_id in annotation_ids:
            msg = f"Duplicate annotation ID {annotation_id!r} in {path}."
            raise ValueError(msg)
        annotation_ids.add(annotation_id)
        rows.append(row)
    return rows


def read_existing_annotation_failures(path: Path) -> list[dict[str, Any]]:
    """Read and validate the append-only deferred annotation failures."""
    if not path.exists():
        return []
    rows = []
    event_ids = set()
    event_keys = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {path} at line {line_number}: {exc}."
            raise ValueError(msg) from exc
        if not isinstance(row, dict) or tuple(row) != NEWS_ANNOTATION_FAILURE_COLUMNS:
            msg = (
                f"Annotation failure schema does not match in {path} "
                f"at line {line_number}."
            )
            raise ValueError(msg)
        event_id = require_text(row, "failure_event_id", "Stored failure")
        annotation_id = require_text(row, "annotation_id", "Stored failure")
        sequence = require_positive_int(
            row["failure_sequence"], "Stored failure sequence"
        )
        expected_event_id = build_failure_event_id(annotation_id, sequence)
        if event_id != expected_event_id:
            msg = f"Stored failure event ID {event_id!r} is not deterministic."
            raise ValueError(msg)
        event_key = (annotation_id, sequence)
        if event_id in event_ids or event_key in event_keys:
            msg = f"Duplicate annotation failure event {event_id!r} in {path}."
            raise ValueError(msg)
        event_ids.add(event_id)
        event_keys.add(event_key)
        rows.append(row)
    return rows


def build_annotation_id(
    task_id: str, config_sha256: str, model_digest: str
) -> str:
    """Build a stable ID so reruns skip only the exact same inference setup."""
    value = json.dumps(
        [
            NEWS_ANNOTATION_OUTPUT_SCHEMA_VERSION,
            task_id,
            config_sha256,
            model_digest,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"news_annotation_sha256:{digest}"


def build_failure_event_id(annotation_id: str, failure_sequence: int) -> str:
    """Build a stable ID for one deferred failure attempt group."""
    value = json.dumps(
        [
            NEWS_ANNOTATION_FAILURE_SCHEMA_VERSION,
            annotation_id,
            failure_sequence,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"news_annotation_failure_sha256:{digest}"


def append_jsonl_record(record: dict[str, Any], path: Path) -> None:
    """Append one UTF-8 JSON object immediately for resume safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
