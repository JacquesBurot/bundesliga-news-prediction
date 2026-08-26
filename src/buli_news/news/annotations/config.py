"""Load and validate the versioned news-annotation configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._validation import (
    require_non_empty_text,
    require_positive_int,
    require_text,
)


DEFAULT_ANNOTATION_CONFIG_PATH = Path("config/news_annotation_schema_v3.json")
INDICATORS = (
    "sporting_form",
    "personnel_situation",
    "physical_readiness",
    "confidence_and_motivation",
)


@dataclass(frozen=True)
class AnnotationConfig:
    """Validated, versioned annotation prompt and response schema."""

    schema_id: str
    schema_version: int
    prompt_version: int
    system_prompt: str
    rating_mapping: dict[str, int | None]
    response_json_schema: dict[str, Any]
    target_team_aliases: dict[str, tuple[str, ...]]
    sha256: str


def load_annotation_config(path: Path) -> AnnotationConfig:
    """Load and validate the versioned prompt and structured-output schema."""
    raw_bytes = path.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        msg = f"Annotation config {path} is not valid JSON: {exc}."
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"Annotation config {path} must contain a JSON object."
        raise ValueError(msg)
    expected_keys = (
        "schema_id",
        "schema_version",
        "prompt_version",
        "system_prompt",
        "rating_mapping",
        "response_json_schema",
        "target_team_aliases",
    )
    if tuple(raw) != expected_keys:
        msg = f"Annotation config schema or order does not match: {list(raw)}."
        raise ValueError(msg)

    schema_id = require_text(raw, "schema_id", "Annotation config")
    schema_version = require_positive_int(
        raw.get("schema_version"), "Annotation config schema_version"
    )
    prompt_version = require_positive_int(
        raw.get("prompt_version"), "Annotation config prompt_version"
    )
    system_prompt = require_text(raw, "system_prompt", "Annotation config")
    rating_mapping = raw.get("rating_mapping")
    response_schema = raw.get("response_json_schema")
    if not isinstance(rating_mapping, dict):
        msg = "Annotation config rating_mapping must be an object."
        raise ValueError(msg)
    if not isinstance(response_schema, dict):
        msg = "Annotation config response_json_schema must be an object."
        raise ValueError(msg)
    validate_response_json_schema(response_schema, rating_mapping)
    target_team_aliases = validate_target_team_aliases(
        raw.get("target_team_aliases")
    )

    canonical = json.dumps(
        raw,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return AnnotationConfig(
        schema_id=schema_id,
        schema_version=schema_version,
        prompt_version=prompt_version,
        system_prompt=system_prompt,
        rating_mapping=rating_mapping,
        response_json_schema=response_schema,
        target_team_aliases=target_team_aliases,
        sha256=hashlib.sha256(canonical).hexdigest(),
    )


def validate_response_json_schema(
    schema: dict[str, Any], rating_mapping: dict[str, int | None]
) -> None:
    """Validate the exact response-schema shape relied upon by the pipeline."""
    expected_rating_mapping = {
        "strong_negative": -2,
        "negative": -1,
        "neutral": 0,
        "positive": 1,
        "strong_positive": 2,
        "not_mentioned": None,
    }
    if rating_mapping != expected_rating_mapping:
        msg = "Annotation rating mapping does not match the expected scale."
        raise ValueError(msg)
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
    ):
        msg = "Response JSON schema must be a closed object."
        raise ValueError(msg)
    properties = schema.get("properties")
    if not isinstance(properties, dict) or tuple(properties) != INDICATORS:
        msg = "Response JSON schema properties do not match the four indicators."
        raise ValueError(msg)
    if schema.get("required") != list(properties):
        msg = "Every response JSON schema property must be required in order."
        raise ValueError(msg)
    for indicator in INDICATORS:
        indicator_schema = properties[indicator]
        if (
            not isinstance(indicator_schema, dict)
            or indicator_schema.get("type") != "object"
            or indicator_schema.get("additionalProperties") is not False
        ):
            msg = f"Schema for indicator {indicator!r} must be a closed object."
            raise ValueError(msg)
        indicator_properties = indicator_schema.get("properties")
        if (
            not isinstance(indicator_properties, dict)
            or tuple(indicator_properties) != ("rating", "evidence")
            or indicator_schema.get("required") != ["rating", "evidence"]
        ):
            msg = f"Schema for indicator {indicator!r} needs rating and evidence."
            raise ValueError(msg)
        enum = indicator_properties["rating"].get("enum")
        if not isinstance(enum, list) or "not_mentioned" not in enum:
            msg = f"Schema enum for indicator {indicator!r} is invalid."
            raise ValueError(msg)
        for rating in enum:
            if rating not in rating_mapping:
                msg = f"Rating {rating!r} has no numeric mapping."
                raise ValueError(msg)
        evidence_schema = indicator_properties["evidence"]
        if (
            evidence_schema.get("type") != "array"
            or evidence_schema.get("maxItems") != 2
            or evidence_schema.get("items", {}).get("type") != "string"
            or evidence_schema.get("items", {}).get("minLength") != 1
        ):
            msg = f"Evidence schema for indicator {indicator!r} is invalid."
            raise ValueError(msg)


def validate_target_team_aliases(value: object) -> dict[str, tuple[str, ...]]:
    """Validate the team aliases supplied to the model as article context."""
    if not isinstance(value, dict) or not value:
        msg = "Annotation target_team_aliases must be a non-empty object."
        raise ValueError(msg)
    aliases_by_team: dict[str, tuple[str, ...]] = {}
    for team, aliases in value.items():
        team_name = require_non_empty_text(team, "Target-team alias key")
        if not isinstance(aliases, list) or not aliases:
            msg = f"Target-team aliases for {team_name!r} must be a non-empty array."
            raise ValueError(msg)
        parsed = tuple(
            require_non_empty_text(alias, f"Alias for {team_name}")
            for alias in aliases
        )
        if len(set(alias.casefold() for alias in parsed)) != len(parsed):
            msg = f"Target-team aliases for {team_name!r} contain duplicates."
            raise ValueError(msg)
        aliases_by_team[team_name] = parsed
    return aliases_by_team
